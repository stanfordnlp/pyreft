import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import os

from pyvene.models.intervenable_base import IntervenableModel
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from datasets import load_dataset
from functools import partial

import sys
sys.path.append("../..")

from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    LoreftIntervention,
    ReftDataCollator,
    ReftSupervisedDataset,
    ReftPreferenceDataset,
    ReftTrainer,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

class ReftTrainerForRewardModelling(ReftTrainer):
    def compute_loss(
        self,
        intervenable: IntervenableModel,
        inputs,
        return_outputs=False
    ):
        _, rewards = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )},
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )

        print(rewards)
        # loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        # if return_outputs:
        #     return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return 0.0


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Maykeye/TinyLLama-v0")


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


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, model, layers, training_args, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    # load data and rename columns
    dataset = load_dataset(data_args.data_path, "synthetic-instruct-gptj-pairwise", split="val")

    def format_data(example):
        result = {}
        chosen_output = tokenizer.apply_chat_template(example["conv_A"], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        rejected_output = tokenizer.apply_chat_template(example["conv_B"], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        instruction = os.path.commonprefix([chosen_output, rejected_output])
        chosen_output = chosen_output[len(instruction):]
        rejected_output = rejected_output[len(instruction):]

        result["instruction"] = instruction
        result["chosen_output"] = chosen_output
        result["rejected_output"] = rejected_output
        result["chosen_reward"] = example["conv_A_rating"]
        result["rejected_reward"] = example["conv_B_rating"]
        return result
    
    dataset = dataset.map(format_data)

    train_dataset = ReftPreferenceDataset(
        "reward", None, tokenizer, dataset=dataset, data_split="train",
        seed=training_args.seed, max_n_example=training_args.max_n_train_example,
        **{"num_interventions": len(layers), "position": training_args.position, 
           "share_weights": training_args.share_weights}
    )
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    print(model)
    representations = [{
        "layer": l, "component": f"model.layers[{l}].output",
        "intervention": LoreftIntervention(
            embed_dim=model.config.hidden_size, 
            low_rank_dimension=training_args.rank,
        )
    } for l in layers]

    reft_config = ReftConfig(representations=representations)
    reft_model = get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()

    # get training data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, model=None, layers=layers,
        training_args=training_args, data_args=data_args)

    # train
    trainer = ReftTrainerForRewardModelling(
        model=reft_model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()