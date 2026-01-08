"""
Custom trainers for ReFT experiments with evaluation support.
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import has_length
from transformers.utils import logging

from pyreft import ReftTrainerForCausalLM
from pyreft.reft_trainer import make_dataloader

logger = logging.get_logger(__name__)


class FullFinetuneTrainer(Trainer):
    """
    Standard full fine-tuning trainer with NLL evaluation.
    Used as baseline comparison for ReFT experiments.
    """

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Evaluate the model and log NLL (negative log-likelihood) loss."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            logger.warning("No eval dataset provided, skipping evaluation.")
            return {}

        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        logger.info(f"***** Running Evaluation *****")
        if has_length(eval_dataloader):
            logger.info(f"  Num examples = {self.num_examples(eval_dataloader)}")
        logger.info(f"  Batch size = {self.args.per_device_eval_batch_size}")

        total_loss = 0.0
        total_tokens = 0
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                # Move inputs to device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                )
                loss = outputs.loss
                
                # Count non-padding tokens for proper averaging
                labels = inputs["labels"]
                num_tokens = (labels != -100).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        # Compute average NLL per token
        avg_nll = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = np.exp(avg_nll) if avg_nll < 100 else float('inf')
        
        metrics = {
            f"{metric_key_prefix}_loss": avg_nll,
            f"{metric_key_prefix}_nll": avg_nll,
            f"{metric_key_prefix}_perplexity": perplexity,
        }
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        self.model.train()
        return metrics


class ReftTrainerForCausalLMWithEval(ReftTrainerForCausalLM):
    """
    Extends ReftTrainerForCausalLM with evaluation support.
    Logs NLL (negative log-likelihood) and perplexity on held-out eval set.
    Also logs intervention metrics (delta norms, etc.) to wandb.
    """

    def _get_intervention_metrics(self, clear_after=True):
        """Collect metrics from all interventions and aggregate them."""
        if not hasattr(self.model, 'interventions'):
            return {}
        
        all_delta_norms = []
        all_diff_norms = []
        all_delta_base_ratios = []
        
        for key, v in self.model.interventions.items():
            intervention = v[0] if isinstance(v, (list, tuple)) else v
            if hasattr(intervention, 'metrics') and intervention.metrics:
                metrics = intervention.metrics
                if 'delta_norm' in metrics:
                    all_delta_norms.append(metrics['delta_norm'])
                if 'diff_norm' in metrics:
                    all_diff_norms.append(metrics['diff_norm'])
                if 'delta_base_ratio' in metrics:
                    all_delta_base_ratios.append(metrics['delta_base_ratio'])
                # Clear metrics after collecting
                if clear_after:
                    intervention.metrics = {}
        
        result = {}
        if all_delta_norms:
            result['intervention/delta_norm_mean'] = np.mean(all_delta_norms)
            result['intervention/delta_norm_max'] = np.max(all_delta_norms)
        if all_diff_norms:
            result['intervention/diff_norm_mean'] = np.mean(all_diff_norms)
            result['intervention/diff_norm_max'] = np.max(all_diff_norms)
        if all_delta_base_ratios:
            result['intervention/delta_base_ratio_mean'] = np.mean(all_delta_base_ratios)
            result['intervention/delta_base_ratio_max'] = np.max(all_delta_base_ratios)
        
        return result

    def log(self, logs):
        """Override log to include intervention metrics."""
        # Add intervention metrics if this is a training log (has 'loss' key)
        if 'loss' in logs:
            intervention_metrics = self._get_intervention_metrics()
            logs.update(intervention_metrics)
        super().log(logs)

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return make_dataloader(
            eval_dataset, 
            self.args.per_device_eval_batch_size, 
            self.data_collator, 
            shuffle=False
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Evaluate the model and log NLL (negative log-likelihood) loss."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            logger.warning("No eval dataset provided, skipping evaluation.")
            return {}

        # Ensure model is in eval mode
        self.model.model.eval()
        for k, v in self.model.interventions.items():
            # Handle both old (tuple) and new (direct) intervention formats
            intervention = v[0] if isinstance(v, (list, tuple)) else v
            intervention.eval()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        logger.info(f"***** Running Evaluation *****")
        if has_length(eval_dataloader):
            logger.info(f"  Num examples = {self.num_examples(eval_dataloader)}")
        logger.info(f"  Batch size = {self.args.per_device_eval_batch_size}")

        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                # Move inputs to device
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.model.get_device())
                
                # Compute loss (NLL)
                loss = self.compute_loss(self.model, inputs, return_outputs=False)
                
                # Count non-padding tokens for proper averaging
                labels = inputs["labels"]
                num_tokens = (labels != -100).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        # Compute average NLL per token
        avg_nll = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = np.exp(avg_nll) if avg_nll < 100 else float('inf')  # Avoid overflow
        
        metrics = {
            f"{metric_key_prefix}_loss": avg_nll,
            f"{metric_key_prefix}_nll": avg_nll,
            f"{metric_key_prefix}_perplexity": perplexity,
        }
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        # Set model back to train mode
        self.model.model.train()
        for k, v in self.model.interventions.items():
            intervention = v[0] if isinstance(v, (list, tuple)) else v
            intervention.train()
        
        return metrics

