import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    LoreftIntervention,
    ReftDataCollator,
    ReftSupervisedDataset,
    ReftModel
)
import pyvene as pv
import os
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def count_parameters(model):
    """Count parameters of a model that require gradients"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="yahma/llama-7b-hf")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


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
    train_dataset = ReftSupervisedDataset(
        "alpaca", data_args.data_path, tokenizer, data_split="train", seed=training_args.seed,
        max_n_example=training_args.max_n_train_example,
        input_field="input", instruction_field="instruction", output_field="output",
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


def train(rank, world_size):
    device_id = rank
    device = torch.device(f'cuda:{device_id}')
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
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    representations = [{
        "layer": l, "component": "block_output",
        # this is needed for loading although dummy.
        "low_rank_dimension": training_args.rank, 
        "intervention": LoreftIntervention(
            embed_dim=model.config.hidden_size, 
            low_rank_dimension=training_args.rank,
        )
    } for l in layers]

    reft_config = ReftConfig(representations=representations)
    reft_model = get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()
    reft_model_ddp = DDP(reft_model) # , device_ids=[device_id], find_unused_parameters=False)

    # check params and devices
    original_params = {name for name, _ in reft_model.named_parameters()}
    ddp_params = {name for name, _ in reft_model_ddp.named_parameters()}
    
    missing_in_ddp = original_params - ddp_params
    missing_in_ddp = sorted(missing_in_ddp)
    print("Missing in DDP is", missing_in_ddp)
    print("Printing original params")
    for x in reft_model.named_parameters():
        print(f"{x[0]} -> {x[1].device}")
    print("Printing DDP params")
    for x in reft_model_ddp.named_parameters():
        print(f"{x[0]} -> {x[1].device}")

    # get training data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, model=model, layers=layers,
        training_args=training_args, data_args=data_args)

    # train
    trainer = ReftTrainerForCausalLM(
        model=reft_model_ddp, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    print("Rank is", rank)
    if rank == 0:
        trainer.save_state()
        reft_model_ddp.module.save(save_directory=training_args.output_dir)
        # uncomment this line to only saving the interventons, 
        # you need to reinit the reft model with random init 
        # interventions mounted then load these weights
        # trainer.save_model(output_dir=training_args.output_dir)

    # test if we can load.
    ReftModel.load(training_args.output_dir, model)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank = rank, world_size = world_size)

def cleanup():
    dist.destroy_process_group()

def process_fn(rank, world_size):
    setup(rank, world_size)
    print("Rank", rank, "world size", world_size)
    train(rank, world_size)
    cleanup()

if __name__ == "__main__":
    assert torch.cuda.is_available(), "MultiGPU script needs CUDA to run"
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(process_fn, args=(world_size,), nprocs=world_size, join=True)
