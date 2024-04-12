import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, Any, List

from pyvene.models.intervenable_base import IntervenableModel
import torch
import transformers
from datasets import load_dataset
import numpy as np

from pyreft import (
    get_reft_model,
    ReftConfig,
    LoreftIntervention,
    ReftDataCollator,
    ReftRewardDataset,
    ReftTrainer,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ReftRewardCollator:
    tokenizer: transformers.PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []

        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["chosen_output"],
                    "attention_mask": feature["chosen_output_mask"],
                    "reward": feature["chosen_reward"],
                    "intervention_locations": feature["intervention_locations"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["rejected_output"],
                    "attention_mask": feature["rejected_output_mask"],
                    "reward": feature["rejected_reward"],
                    "intervention_locations": feature["intervention_locations"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "reward": batch["reward"],
            "intervention_locations": batch["intervention_locations"],
        }
        max_seq_length = batch["input_ids"].shape[-1]
        batch["intervention_locations"] = batch["intervention_locations"][..., :max_seq_length]
        return batch


class ReftTrainerForRewardModelling(ReftTrainer):
    def compute_loss(
        self,
        intervenable: IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # reward
        rewards = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )},
            subspaces=None
        )

        # masks
        chosen_mask = torch.arange(inputs["input_ids"].shape[0]) % 2 == 0
        rejected_mask = ~chosen_mask

        # compute reward diff, maximise gap
        rewards_chosen = rewards[-1].logits[chosen_mask]
        rewards_rejected = rewards[-1].logits[rejected_mask]
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss

    def prediction_step(
        self,
        model: IntervenableModel,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        loss, reward = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach().cpu()
        logits = (reward["rewards_chosen"] - reward["rewards_rejected"]).detach().cpu()
        labels = torch.ones_like(logits)
        return (loss, logits, labels)


def compute_metrics(eval_pred):
    result = {}
    diffs = eval_pred.predictions.reshape(-1)
    result['accuracy'] = np.sum(diffs > 0.0) / len(diffs)
    print(result)
    return result


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/gemma-2b-it")


@dataclass
class DataArguments:
    data_path: str = field(default="llm-blender/Unified-Feedback", metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    layers: str = field(
        default="all",
        metadata={"help": "Intervening layers."},
    )
    position: str = field(
        default="f1+l1",
        metadata={"help": "Intervening position string."},
    )
    share_weights: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    rank: int = field(default=1)
    max_n_train_example: int = field(default=None)
    max_n_eval_example: int = field(default=None)
    wandb_project: str = field(default="reft-reward")
    wandb_entity: str = field(default="none")
    logging_steps: int = field(default=10)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, model, layers, training_args, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    # field setup
    fields = {
        "conv_A_field": "conv_A", "conv_B_field": "conv_B",
        "conv_A_reward_field": "conv_A_rating", "conv_B_reward_field": "conv_B_rating"
    }

    # load data and rename columns
    train_dataset = ReftRewardDataset(
        "reward", None, tokenizer,
        dataset=load_dataset(data_args.data_path, "all", split="train"),
        data_split="train",
        seed=training_args.seed, max_n_example=training_args.max_n_train_example,
        **{"num_interventions": len(layers), "position": training_args.position, 
           "share_weights": training_args.share_weights},
        **fields,
    )
    eval_dataset = ReftRewardDataset(
        "reward", None, tokenizer,
        dataset=load_dataset(data_args.data_path, "all", split="val"),
        data_split="val",
        seed=training_args.seed, max_n_example=training_args.max_n_eval_example,
        **{"num_interventions": len(layers), "position": training_args.position, 
           "share_weights": training_args.share_weights},
        **fields,
    )
    data_collator = ReftRewardCollator(
        tokenizer=tokenizer,
        padding=True,
        max_length=tokenizer.model_max_length
    )
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # wandb setup
    os.environ['WANDB_ENTITY'] = training_args.wandb_entity
    os.environ['WANDB_PROJECT'] = training_args.wandb_project

    # asserts
    assert training_args.per_device_train_batch_size % 2 == 0, "Batch size must be even."

    # parsing layers arg
    if training_args.layers != "all":
        layers = [int(l) for l in training_args.layers.split(";")]
    else:
        temp_config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
        layers = [l for l in range(temp_config.num_hidden_layers)]
    if "+" in training_args.position and not training_args.share_weights:
        layers += layers

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # get reft model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    representations = [{
        "layer": l, "component": f"model.layers[{l}].output",
        "intervention": LoreftIntervention(
            embed_dim=model.config.hidden_size, 
            low_rank_dimension=training_args.rank,
        )
    } for l in layers]

    reft_config = ReftConfig(representations=representations)
    reft_model = get_reft_model(model, reft_config)
    for param in reft_model.model.score.parameters():
        # reft_model with HF trainer will automatically pick up these params to optimize
        param.requires_grad = True
    reft_model.print_trainable_parameters()

    # get training data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, model=None, layers=layers,
        training_args=training_args, data_args=data_args)

    # train
    trainer = ReftTrainerForRewardModelling(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        **data_module
    )
    trainer.train()

    # ensure everything is in eval mode
    trainer.model.model.eval()
    for k,v in  trainer.model.interventions.items():
        _ = v[0].eval()

    # eval
    trainer.evaluate()

    # save
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()