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
    ReftTrainerForCausalLMDistributed, 
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
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    reft_model.set_device(device)
    reft_model.train()
    reft_model.model.train()
    reft_model.training = True
    reft_model = reft_model.to(rank)
    reft_model_ddp = DDP(reft_model, device_ids=[device_id])

    # check params and devices
    original_params = {name for name, _ in reft_model.named_parameters()}
    ddp_params = {name for name, _ in reft_model_ddp.named_parameters()}
    
    missing_in_ddp = original_params - ddp_params
    missing_in_ddp = sorted(missing_in_ddp)
    for param_name in missing_in_ddp:
        param = dict(reft_model.named_parameters())[param_name]
        new_param_name = param_name.replace(".", "_")
        reft_model_ddp.register_parameter(new_param_name, param)
        dist.broadcast(param.data, src=0)  # Broadcast from rank 0 to other processes

    reft_model_ddp.train()
    reft_model_ddp.module.train()
    reft_model_ddp.training = True

    # log to wandb from main process only
    if rank == 0:
        training_args.report_to = ['wandb']
        training_args.run_name = 'multigpu_reft_alpaca_example'
        training_args.logging_steps = 1
    else:
        training_args.report_to = []

    # get training data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, model=model, layers=layers,
        training_args=training_args, data_args=data_args)

    trainer = ReftTrainerForCausalLMDistributed(
        model=reft_model_ddp, tokenizer=tokenizer, args=training_args, **data_module)
    # assert all parameters on same device
    for (n, p) in trainer.model.named_parameters():
        assert(p.get_device() == rank)

    # train
    trainer.train()
    if rank == 0:
        print("Saving")
        trainer.save_state()
        reft_model_ddp.module.save(save_directory=training_args.output_dir)
        # uncomment this line to only save the interventons, 
        # you need to reinit the reft model with random init 
        # interventions mounted then load these weights
        # trainer.save_model(output_dir=training_args.output_dir)
        print("Loading")
        # test if we can load.
        ReftModel.load(training_args.output_dir, model)
        print("Complete")

if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print("Starting on rank", rank)
    train(rank, -1)
    dist.destroy_process_group()
    print("Finished on rank", rank)
