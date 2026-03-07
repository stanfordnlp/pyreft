import pyvene as pv
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollator,
    DataCollatorForSeq2Seq,
    AutoTokenizer
)
from transformers.trainer_utils import (
    EvalPrediction,
    has_length,
    denumpify_detensorize
)
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Union, Iterable, Tuple

from tqdm import tqdm
import os
import torch
import re

import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging
import torch.distributed as dist

logger = logging.get_logger(__name__)

@dataclass
class ReftDataCollator(object):
    """Collate examples for ReFT."""

    data_collator: DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][..., :max_seq_length]
        return batch_inputs


def make_data_collator(tokenizer, model) -> ReftDataCollator:
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )
    return ReftDataCollator(data_collator=data_collator_fn)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: DataCollatorForSeq2Seq,
    shuffle: bool,
    sampler: Union[Sampler, Iterable, None]=None
) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)


class ReftTrainer(Trainer):
    def save_model(self, output_dir, _internal_call=False, **kwargs):
        # Handle CPU training and non-distributed cases
        try:
            is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        except (RuntimeError, AttributeError) as e:  # Catches case when torch.distributed is not available or other dist errors
            logger.error(f"Error checking distributed training status: {str(e)}")
            is_main_process = True
        
        if is_main_process:
            target_dir = f"{output_dir}/intervenable_model"
            # Log warning if target directory exists and has content
            if os.path.exists(target_dir) and os.listdir(target_dir):
                logger.warning(
                    f"Directory {target_dir} already exists and contains files. "
                    "Skipping save to prevent overwriting existing model."
                )
                return
                
            try:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                self.model.save_intervention(
                    save_directory=target_dir, 
                    include_model=True
                )
            except Exception as e:
                logger.error(f"Error saving model to {target_dir}: {str(e)}")
                raise  # Re-raise the exception after logging

    def _load_best_model(self, **kwargs):
        logger.warning(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self.model.load_intervention(
            f"{self.state.best_model_checkpoint}/intervenable_model", 
            include_model=True
        )
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None, **kwargs):
        if model is None:
            model = self.model

        logger.warning(f"Loading checkpoint from {resume_from_checkpoint}.")
        model.load_intervention(
            f"{resume_from_checkpoint}/intervenable_model", 
            include_model=True
        )

    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False,
        **kwargs
    ):
        # run intervened forward pass
        unit_locations = None
        if "intervention_locations" in inputs:
            if inputs["intervention_locations"].dim() == 3:
                unit_locations={"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}
            else:
                # this is dummy for lora only baseline
                unit_locations={"sources->base": (None, 0)}
        base_outputs, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )
        # return
        output = cf_outputs
        if cf_outputs is None:
            output = base_outputs # in case of lora only training

        return (output, output) if return_outputs else output.loss

class ReftTrainerForCausalLM(ReftTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)

class ReftTrainerForCausalLMDistributed(ReftTrainer):
    def save_model(self, output_dir, _internal_call=False):
        if dist.get_rank() == 0:
            super().save_model(output_dir, _internal_call)

    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(
            self.train_dataset,
            self._train_batch_size,
            self.data_collator,
            shuffle=False,
            sampler=DistributedSampler(self.train_dataset, shuffle=True),
        )
    
class ReftTrainerForSequenceClassification(ReftTrainer):
    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # run intervened forward pass
        unit_locations = None
        if "intervention_locations" in inputs:
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
            
        _, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )
        # classification loss on counterfactual labels
        logits = cf_outputs.logits
        labels = inputs["labels"]

        if self.model.model.config.problem_type is None:
            if self.model.model.num_labels == 1:
                problem_type = "regression"
            elif self.model.model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
        else:
            problem_type = self.model.model.config.problem_type
            
        if problem_type == "regression":
            loss_fct = MSELoss()
            if self.model.model.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze().to(torch.bfloat16))
            else:
                loss = loss_fct(logits, labels.to(torch.bfloat16))
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.model.num_labels), labels.view(-1))
        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        # return
        return (loss, cf_outputs) if return_outputs else loss
    
    def evaluate(
        self, ignore_keys,
    ):

        # ensure everything is in eval mode
        self.model.model.eval()
        for k,v in  self.model.interventions.items():
            _ = v[0].eval()
        
        batch_size = self.args.eval_batch_size
        data_collator = self.data_collator
        eval_dataset = self.eval_dataset
        intervenable = self.model
        
        dataloader = make_dataloader(
            eval_dataset, batch_size, data_collator, shuffle=False)

        logger.info(f"***** Running In-Training Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        eval_iterator = tqdm(dataloader, position=0, leave=True)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for step, inputs in enumerate(eval_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.model.get_device())
                
                # [layers, batch_size, positions]
                intervention_locations = inputs["intervention_locations"].permute(1, 0, 2).tolist()
                _, cf_outputs = intervenable(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    unit_locations={"sources->base": (None, intervention_locations)})
            
                all_preds += [cf_outputs.logits]
                all_labels += [inputs["labels"]]
        all_preds = torch.cat(all_preds, dim=0).cpu().to(torch.float32)
        all_labels = torch.cat(all_labels, dim=0).cpu().to(torch.float32)
        metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        metrics = denumpify_detensorize(metrics)
        
        metric_key_prefix = "eval"
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        
        return metrics


def _create_reft_dpo_trainer():
    try:
        from trl import DPOTrainer
    except ImportError:
        return None

    class _ReftDPOTrainer(DPOTrainer):
        def concatenated_forward(
            self,
            model: nn.Module,
            batch: Dict[str, Union[List, torch.LongTensor]],
            reference: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            concatenated_batch = self.concatenated_inputs(
                batch,
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
                padding_value=self.padding_value,
                device=self.accelerator.device,
            )
            len_chosen = batch["chosen_labels"].shape[0]
            il = batch["intervention_locations"]
            intervention_locations = torch.cat([il, il], dim=0).permute(1, 0, 2).tolist()

            model_kwargs = (
                {
                    "labels": concatenated_batch["concatenated_labels"],
                    "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
                }
                if self.is_encoder_decoder
                else {}
            )
            if reference:
                all_outputs = model.model(
                    input_ids=concatenated_batch["concatenated_input_ids"].to(model.get_device()),
                    attention_mask=concatenated_batch["concatenated_attention_mask"].to(model.get_device()),
                    use_cache=False,
                    **model_kwargs,
                )
            else:
                _, all_outputs = model(
                    {
                        "input_ids": concatenated_batch["concatenated_input_ids"].to(model.get_device()),
                        "attention_mask": concatenated_batch["concatenated_attention_mask"].to(model.get_device()),
                    },
                    unit_locations={"sources->base": (None, intervention_locations)},
                    use_cache=False,
                    **model_kwargs,
                )

            all_logits = all_outputs.logits
            all_logps = self.get_batch_logps(
                all_logits,
                concatenated_batch["concatenated_labels"],
                average_log_prob=self.loss_type == "ipo",
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
            )
            chosen_logps = all_logps[:len_chosen]
            rejected_logps = all_logps[len_chosen:]
            chosen_logits = all_logits[:len_chosen]
            rejected_logits = all_logits[len_chosen:]
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

        def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
        ):
            metrics = {}
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
            ) = self.concatenated_forward(model, batch, reference=False)

            if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
                reference_chosen_logps = batch["reference_chosen_logps"]
                reference_rejected_logps = batch["reference_rejected_logps"]
            else:
                with torch.no_grad():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch, reference=True)

            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
            metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
            metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
            metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
            metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
            metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
            metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
            return losses.mean(), metrics

        def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
            if output_dir is None:
                output_dir = self.args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.model.save_intervention(
                save_directory=os.path.join(output_dir, "intervenable_model"),
                include_model=True,
            )

    return _ReftDPOTrainer


ReftDPOTrainer = _create_reft_dpo_trainer()
