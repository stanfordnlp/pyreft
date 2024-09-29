#!/usr/bin/env python
# coding: utf-8

# ## Domain Transfer Task
# ### Setup

# In[1]:


import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from tqdm import tqdm

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
import math

from promptsource.templates import DatasetTemplates

import transformers
from filelock import FileLock
from transformers import (
    # AdapterConfig,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    # MultiLingAdapterArguments,
    # Seq2SeqAdapterTrainer,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version


# In[2]:


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


# In[3]:


model_name_or_path="t5-base"
seed = 42
dropout = 0
max_length = 512
low_rank = 8
set_seed(seed)
max_train_examples = 10000
train_batch_size = 64
fp16 = True
testing = False
intervention_type = "nodireft"


# In[4]:


config = AutoConfig.from_pretrained(
    model_name_or_path,
    dropout_rate=dropout,
    max_length=max_length,
)


# In[5]:


import torch
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_or_path,
    config=config,
    torch_dtype=torch.bfloat16 if fp16 else torch.float32,
)


# In[6]:


model.resize_token_embeddings(len(tokenizer))


# ### Reft Model

# In[48]:


import torch 
from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainer,
    ReftTrainerForCausalLM, 
    ReftDataCollator,
    ReftRawDataset,
    LoreftIntervention,
    NodireftIntervention,
    DireftIntervention,
)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pyvene.models.modeling_utils import *


def setup_distributed():
    if dist.is_available() and dist.is_initialized():
        return int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = setup_distributed()
print("CUDA:", torch.cuda.is_available())
device = torch.device(f'cuda:{local_rank}')

# device = "cpu"

# Let's create a subspace with 8 dims
FULL_SUBSPACE = list(range(low_rank))



class SubloreftIntervention(LoreftIntervention):
    """
    This is a LoReFT that supports subspace interventions with coefficients!
    """
    def __init__(self, **kwargs):
        subspace_coeff = None
        # Subspace coefficients are the coefficients applied to each subspace.
        # When `subspace_coeff` is a ones tensor, this intervention is the same as a loreft intervention with subspaces
        # When `subspace_coeff` is a negative-ones tensor, this intervention is the negation of the loreft intervention
        # There is no intervention when `subspace_coeff` is zero.
        if "subspace_coeff" in kwargs:
            subspace_coeff = kwargs["subspace_coeff"].copy()
            del kwargs["subspace_coeff"]
        subspace_coeff = torch.tensor(subspace_coeff) if subspace_coeff is not None else torch.ones(kwargs["low_rank_dimension"])
        self.subspace_coeff = subspace_coeff.to(device)
        super().__init__(**kwargs)
        print("loreft", kwargs)
        if not fp16:
            self.learned_source = self.learned_source.to(torch.float32) 
            
    def forward(
        self, base, source=None, subspaces=None, **kwargs,
    ):
        assert subspaces is not None
        original_output = kwargs["_pyvene_model_input_args"][0]
        # print("mag:", self.subspace_coeff)
        output = []

        rotated_base = self.rotate_layer(original_output)
        diff = self.act_fn(self.learned_source(original_output)) - rotated_base
        
        batched_subspace = []
        batched_weights = []
        
        if len(diff) > 1:
            subspaces = [subspaces[0]] * len(diff)
        elif len(diff) != len(subspaces):
            print(f"Warning! lengths do not match {len(diff)} {len(subspaces)}")

        # Expand subspaces to match dimensions
        subspaces = torch.tensor(subspaces).to(base.device)
        subspaces_expanded = subspaces.unsqueeze(1).expand(diff.size(0), diff.size(1), -1)
        
        LHS = torch.gather(diff, 2, subspaces_expanded) * self.subspace_coeff[subspaces_expanded]
        
        # Transpose and gather the corresponding weights for each subspace
        RHS = self.rotate_layer.weight[..., subspaces].permute(1, 2, 0)
        output = base + torch.bmm(LHS, RHS)
        
        return self.dropout(output.to(base.dtype))

class CoeffloreftIntervention(LoreftIntervention):
    """
    This is a LoReFT that supports subspace interventions with coefficients!
    """
    def __init__(self, **kwargs):
        subspace_coeff = None
        # Subspace coefficients are the coefficients applied to each subspace.
        # When `subspace_coeff` is a ones tensor, this intervention is the same as a loreft intervention with subspaces
        # When `subspace_coeff` is a negative-ones tensor, this intervention is the negation of the loreft intervention
        # There is no intervention when `subspace_coeff` is zero.
        if "subspace_coeff" in kwargs:
            subspace_coeff = kwargs["subspace_coeff"].copy()
            del kwargs["subspace_coeff"]
        self.subspace_coeff = torch.tensor(subspace_coeff) if subspace_coeff is not None else torch.ones(1)
        self.subspace_coeff = self.subspace_coeff.to(device)
        super().__init__(**kwargs)
        print("loreft", kwargs)
        if not fp16:
            self.learned_source = self.learned_source.to(torch.float32)        
            
    def forward(
        self, base, source=None, subspaces=None, **kwargs,
    ):
        original_output = kwargs["_pyvene_model_input_args"][0]
        # print(base.shape, original_output.shape, torch.equal(base, original_output))
        # print(len(kwargs["_pyvene_model_input_args"]), len(kwargs["_pyvene_model_output"]))
        # print("mag:", self.subspace_coeff)
        # print(kwargs.keys())
        rotated_base = self.rotate_layer(original_output)
        val = torch.matmul(
            (self.act_fn(self.learned_source(original_output)) - rotated_base), self.rotate_layer.weight.T
        )
        # print(f"mag: {self.subspace_coeff}, val: {val.norm()}")
        
        output = base + self.subspace_coeff * val
        return self.dropout(output.to(base.dtype))

class SubNodireftIntervention(NodireftIntervention):
    """
    This is a NodiReft that supports subspace interventions with coefficients!
    """
    def __init__(self, **kwargs):
        subspace_coeff = None
        # Subspace coefficients are the coefficients applied to each subspace.
        # When `subspace_coeff` is a ones tensor, this intervention is the same as a loreft intervention with subspaces
        # When `subspace_coeff` is a negative-ones tensor, this intervention is the negation of the loreft intervention
        # There is no intervention when `subspace_coeff` is zero.
        if "subspace_coeff" in kwargs:
            subspace_coeff = kwargs["subspace_coeff"].copy()
            del kwargs["subspace_coeff"]
        self.subspace_coeff = torch.tensor(subspace_coeff) if subspace_coeff is not None else torch.ones(1)
        self.subspace_coeff = self.subspace_coeff.to(device)
        super().__init__(**kwargs)
        print("nodireft", kwargs)
        if not fp16:
            self.learned_source = self.learned_source.to(torch.float32)
            self.subspace_coeff = self.subspace_coeff.to(torch.float32)
        else:
            self.subspace_coeff = self.subspace_coeff.to(torch.bfloat16)
            
    def forward(
        self, base, source=None, subspaces=None, **kwargs
    ):
        original_output = kwargs["_pyvene_model_input_args"][0]
        output = base + self.subspace_coeff * torch.matmul(
             self.act_fn(self.learned_source(original_output)), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))


class SubDireftIntervention(DireftIntervention):
    """
    This is a DiReft that supports subspace interventions with coefficients!
    """
    def __init__(self, **kwargs):
        subspace_coeff = None
        if "subspace_coeff" in kwargs:
            subspace_coeff = kwargs["subspace_coeff"].copy()
            del kwargs["subspace_coeff"]
        self.subspace_coeff = torch.tensor(subspace_coeff) if subspace_coeff is not None else torch.ones(1)
        self.subspace_coeff = self.subspace_coeff.to(device)
        super().__init__(**kwargs)
        print("direft", kwargs)
        if not fp16:
            self.learned_source = self.learned_source.to(torch.float32)
            self.subspace_coeff = self.subspace_coeff.to(torch.float32)
        else:
            self.subspace_coeff = self.subspace_coeff.to(torch.bfloat16)
            
    def forward(
        self, base, source=None, subspaces=None, **kwargs
    ):
        original_output = kwargs["_pyvene_model_input_args"][0]
        cast_base = original_output.to(self.learned_source.weight.dtype)
        output = base + self.subspace_coeff * torch.matmul(
            (self.act_fn(self.learned_source(cast_base))).to(self.rotate_layer.weight.dtype), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


layers = list(range(12))
num_interventions = 2 * len(layers)

# get reft model

if intervention_type == "nodireft":
    reft_config = ReftConfig(representations=
        [{
                "layer": l, "component": "encoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubNodireftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16 if fp16 else torch.float32, 
                    add_bias=False,
                )
            } for l in layers]
        + [{
                "layer": l, "component": "decoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubNodireftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16 if fp16 else torch.float32, 
                    add_bias=False,
                )
            } for l in layers]
    )
elif intervention_type == "loreft":
    reft_config = ReftConfig(representations=
        [{
                "layer": l, "component": "encoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubloreftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16 if fp16 else torch.float32, 
                    init_orth=True,
                )
            } for l in layers]
        + [{
                "layer": l, "component": "decoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubloreftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16 if fp16 else torch.float32, 
                    init_orth=True,
                )
            } for l in layers]
    )
elif intervention_type == "direft":
    reft_config = ReftConfig(representations=
        [{
                "layer": l, "component": "encoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubDireftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16 if fp16 else torch.float32, 
                    init_orth=True,
                )
            } for l in layers]
        + [{
                "layer": l, "component": "decoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubDireftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16 if fp16 else torch.float32, 
                    init_orth=True,
                )
            } for l in layers]
    )
elif intervention_type == "coeffloreft":
    reft_config = ReftConfig(representations=
        [{
                "layer": l, "component": "encoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": CoeffloreftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16 if fp16 else torch.float32, 
                    init_orth=True,
                )
            } for l in layers]
        + [{
                "layer": l, "component": "decoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": CoeffloreftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16 if fp16 else torch.float32, 
                    init_orth=True,
                )
            } for l in layers]
    )
else:
    raise ValueError(f'No support for intervention {intervention_type}')


# In[56]:


from dataclasses import dataclass, field
from datasets import Dataset
from typing import Dict, Optional, Sequence, Union, List, Any


@dataclass
class AdaptorReftDataCollator(object):
    """Collate examples for ReFT."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        if "decoder_input_ids" in batch_inputs.keys():
            del batch_inputs["decoder_input_ids"]
        return batch_inputs


# In[118]:


def data_generator(tokenizer, inputs, num_interventions):
    """Generator function to yield data lazily."""
    for i in range(len(inputs)):
        _input = inputs[i]
        
        output_ids = [(l if l != tokenizer.pad_token_id else -100) for l in _input["labels"]]
        
        yield {
            "input_ids": _input["input_ids"],
            "labels": _input["labels"],
            "subspaces": [FULL_SUBSPACE] * num_interventions,
            # "intervention_locations": [[0]] * num_interventions
        }

def make_all_positions_unsupervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, model, inputs, 
    num_interventions=1, nonstop=False, fp16=False
):
    """Make dataset and collator for unsupervised (or really, semi-supervised) fine-tuning with streaming."""
    
    # Using a generator to lazily load the dataset
    train_dataset = Dataset.from_generator(
        lambda: data_generator(tokenizer, inputs, num_interventions),
    )
    
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if fp16 else None,
    )
    data_collator = AdaptorReftDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


# def make_all_positions_unsupervised_data_module(
#     tokenizer: transformers.PreTrainedTokenizer, model, inputs, 
#     num_interventions=1, nonstop=False,
# ):
#     """Make dataset and collator for un-supervised (or really, semi-supervised) fine-tuning."""
    
#     all_base_input_ids, all_intervention_locations, all_output_ids, all_subspaces, all_attention_masks = [], [], [], [], []
#     for i in range(len(inputs)):
#         _input = inputs[i]
#         # print(_input.keys())

#         output_ids = [(l if l != tokenizer.pad_token_id else -100) for l in _input["labels"]] 

#         all_base_input_ids.append(_input["input_ids"])
#         all_output_ids.append(_input["labels"])
#         all_attention_masks.append(_input["attention_mask"])
#         all_subspaces.append([FULL_SUBSPACE] * num_interventions)
#         all_intervention_locations.append([[0]] * num_interventions)
        
#     train_dataset = Dataset.from_dict({
#         "input_ids": all_base_input_ids,
#         "labels": all_output_ids,
#         # "intervention_locations": all_intervention_locations,
#         "subspaces": all_subspaces,
#     })
        
#     data_collator_fn = transformers.DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         model=model,
#         label_pad_token_id=-100,
#         pad_to_multiple_of=8 if fp16 else None,
#     )
#     data_collator = AdaptorReftDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
#     return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)



# In[61]:
from torch.utils.data import DataLoader, DistributedSampler
import pyvene as pv
class ReftTrainerDistributed(ReftTrainerForCausalLM):
    def save_model(self, output_dir, _internal_call=False):
        # Only save the model if this is the main process (rank 0)
        if dist.get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)
            # Access the underlying model in DDP with self.model.module
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            # print(model_to_save)
            model_to_save.save_intervention(
                save_directory=f"{output_dir}/intervenable_model",
                include_model=True
            )

    def get_train_dataloader(self) -> DataLoader:
        return Trainer.get_train_dataloader(self)
        
        # # if dist.is_initialized():
        # #     train_sampler = DistributedSampler(self.train_dataset)
        # # else:
        # train_sampler = None
        # # print("Train batch size:", self.args.train_batch_size)

        # return DataLoader(
        #     self.train_dataset,
        #     batch_size=self.args.train_batch_size,
        #     sampler=train_sampler,
        #     collate_fn=self.data_collator,
        #     num_workers=2,
        #     pin_memory=True,
        #     shuffle=(train_sampler is None),
        # )
    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # Print weight ..
        # local_rank = int(os.environ["LOCAL_RANK"])
        # for name, param in self.model.module.named_parameters():
        #     if "comp" in name:
        #         print(f"{local_rank}, {name}, {param.norm().device}, {param.norm().item()}")
        
        loss = super().compute_loss(intervenable, inputs, return_outputs)
        
        # cnt = False
        # # print("Training:", self.model.training, self.model.module.training)
        
        return loss


def get_module(model, representation, backend="native") -> nn.Module:
    """Render the intervening module with a hook."""
    if (
        get_internal_model_type(model) in type_to_module_mapping and
        representation.component
        in type_to_module_mapping[get_internal_model_type(model)]
    ):
        type_info = type_to_module_mapping[get_internal_model_type(model)][
            representation.component
        ]
        parameter_name = type_info[0]
        if "%s" in parameter_name and representation.moe_key is None:
            # we assume it is for the layer.
            parameter_name = parameter_name % (representation.layer)
        elif "%s" in parameter_name and representation.moe_key is not None:
            parameter_name = parameter_name % (
                int(representation.layer),
                int(representation.moe_key),
            )
    else:
        parameter_name = ".".join(representation.component.split(".")[:-1])

    module = getattr_for_torch_module(model, parameter_name)
    return module, parameter_name


import copy
def handle_training(dataset_name, batch_size_multiplier=1, profiling=False):    
    reft_model = get_reft_model(model, copy.deepcopy(reft_config), set_device=device)
    reft_model.set_device(device)
    print(reft_model.get_device(), device)
    reft_model.print_trainable_parameters()
    train_dataset = datasets.load_from_disk(dataset_name)
    if testing: train_dataset = train_dataset.select(range(max_train_examples))
    train_dataset = make_all_positions_unsupervised_data_module(tokenizer, model, train_dataset, num_interventions=num_interventions, nonstop=False)
    train_dataset, data_collator = train_dataset["train_dataset"], train_dataset["data_collator"]
    print(len(train_dataset))
    
    # Double checked, we can use ReftTrainerForCausalLM for training Seq2Seq models
    reft_model.train()
    reft_model.model.train()
    reft_model.training = True
    
    # Move model to the appropriate device and wrap it in DDP
    reft_model = reft_model.to(local_rank)  # Move model to GPU
    reft_model_ddp = DDP(reft_model, device_ids=[local_rank])  # Wrap in DD

    # Compare the parameters between model and ddp_model
    # if local_rank == 0:
    original_params = {name for name, _ in reft_model.named_parameters()}
    ddp_params = {name for name, _ in reft_model_ddp.named_parameters()}
    
    missing_in_ddp = original_params - ddp_params
    missing_in_ddp = sorted(missing_in_ddp)
    for param_name in missing_in_ddp:
        param = dict(reft_model.named_parameters())[param_name]
        new_param_name = param_name.replace(".", "_")
        reft_model_ddp.register_parameter(new_param_name, param)
        print(f"Broadcasting {param_name}")
        dist.broadcast(param.data, src=0)  # Broadcast from rank 0 to other processes
    
    # Double checked, we can use ReftTrainerForCausalLM for training Seq2Seq models
    reft_model_ddp.train()
    reft_model_ddp.module.train()
    reft_model_ddp.training = True

    training_args = transformers.TrainingArguments(
        num_train_epochs=1.0, output_dir="./results_domain", learning_rate=5e-4, report_to=[],
        per_device_train_batch_size=batch_size_multiplier * train_batch_size, logging_steps=50, bf16=fp16,
        local_rank=local_rank,
        dataloader_num_workers = 2,
        adam_beta1 = 0.9,
        adam_beta2 = 0.98,
        adam_epsilon = 1e-6,
        max_grad_norm = 1,
        dataloader_pin_memory = True,
        remove_unused_columns = False,
        # warmup_ratio=0.06,
        # save_steps=50,
        gradient_accumulation_steps=2,
    )
    # print("Global batch size:", training_args.train_batch_size)
    # print("Per device batch size:", training_args.per_device_train_batch_size)
    trainer = ReftTrainerDistributed(
        model=reft_model_ddp, tokenizer=tokenizer, args=training_args, 
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    class ProfCallback(TrainerCallback):
        def __init__(self, prof):
            self.prof = prof
    
        def on_step_end(self, args, state, control, **kwargs):
            self.prof.step()

    # def trace_handler(prof):
    #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #     print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    #     print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    #     print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
    #     prof.export_chrome_trace("/tmp/trace.json")

    if profiling:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA], 
                                    schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=4, repeat=1),
                                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./hf-training-trainer/grad/'),
                                    profile_memory=True,
                                    with_stack=True,
                                    record_shapes=True) as prof:
            
            trainer.add_callback(ProfCallback(prof=prof))
            trainer.train()
    else:
        trainer.train()
    
    # prof.export_chrome_trace("my_trainer.json")
    
    return reft_model


# ### Train Amazon LM

# In[63]:


reft_amazon_lm = handle_training("train/amazon_lm_data_new")


# ### Train Yelp LM

# In[64]:


reft_yelp_lm = handle_training("train/yelp_lm_data_new")


# ### Train Yelp Classifier

# In[65]:


reft_yelp_classifier = handle_training("train/yelp_classify_data")


# ### Train Amazon Classifier

# In[66]:


reft_amazon_classifier = handle_training("train/amazon_classify_data")



######################################


# ### Man - Woman = King - Queen relationship



if intervention_type == "nodireft":
    representations = []
    sub_representation = [{
                "layer": l, "component": "encoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubNodireftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16, 
                    add_bias=False,
                )
            } for l in layers]
    for _ in range(3):
        representations += copy.deepcopy(sub_representation)
    sub_representation = [{
                "layer": l, "component": "decoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubNodireftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16, 
                    add_bias=False,
                )
            } for l in layers]
    for _ in range(3):
        representations += copy.deepcopy(sub_representation)
    composed_reft_config = ReftConfig(representations=representations)
elif intervention_type == "loreft":
    representations = []
    sub_representation = [{
                "layer": l, "component": "encoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubloreftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16, 
                    init_orth=True,
                )
            } for l in layers]
    for _ in range(3):
        representations += copy.deepcopy(sub_representation)
    sub_representation = [{
                "layer": l, "component": "decoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubloreftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16, 
                    init_orth=True,
                )
            } for l in layers]
    for _ in range(3):
        representations += copy.deepcopy(sub_representation)
    composed_reft_config = ReftConfig(representations=representations)
elif intervention_type == "direft":
    representations = []
    sub_representation = [{
                "layer": l, "component": "encoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubDireftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16, 
                )
            } for l in layers]
    for _ in range(3):
        representations += copy.deepcopy(sub_representation)
    sub_representation = [{
                "layer": l, "component": "decoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": SubDireftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16, 
                )
            } for l in layers]
    for _ in range(3):
        representations += copy.deepcopy(sub_representation)
    composed_reft_config = ReftConfig(representations=representations)
elif intervention_type == "coeffloreft":
    representations = []
    sub_representation = [{
                "layer": l, "component": "encoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": CoeffloreftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16,
                    init_orth=True,
                )
            } for l in layers]
    for _ in range(3):
        representations += copy.deepcopy(sub_representation)
    sub_representation = [{
                "layer": l, "component": "decoder.block." + str(l) + ".output",
                "low_rank_dimension": low_rank,
                "intervention": CoeffloreftIntervention(
                    embed_dim=model.config.hidden_size, low_rank_dimension=low_rank,
                    dtype=torch.bfloat16, 
                    init_orth=True,
                )
            } for l in layers]
    for _ in range(3):
        representations += copy.deepcopy(sub_representation)
    composed_reft_config = ReftConfig(representations=representations)


reft_composed = get_reft_model(model, composed_reft_config, set_device=False)
reft_composed.set_device(device)
print(reft_composed.get_device())
reft_composed.print_trainable_parameters()


# In[103]:


train_dataset = datasets.load_from_disk("validation/yelp_classify_data") # amazon classifier
if testing: train_dataset = train_dataset.select(range(max_train_examples))
len(train_dataset)


# In[140]:


import copy
def set_lm(reft_model, reft_composed, layer, l, intervention_id):
    
    composed_key = "comp.encoder.block." + str(layer) + ".output.unit.pos.nunit.1#" + str(intervention_id)
    original_key = "comp.encoder.block." + str(layer) + ".output.unit.pos.nunit.1#0"
    
    if intervention_type == "loreft":
        reft_composed.interventions[composed_key][0].rotate_layer = copy.deepcopy(reft_model.interventions[original_key][0].rotate_layer)
        subspace_coeff = l * torch.ones(low_rank).to(device)
    elif intervention_type == "nodireft":
        reft_composed.interventions[composed_key][0].proj_layer = copy.deepcopy(reft_model.interventions[original_key][0].proj_layer)
        subspace_coeff = l * torch.ones(1).to(device)
    elif intervention_type == "direft" or intervention_type == "coeffloreft":
        reft_composed.interventions[composed_key][0].rotate_layer = copy.deepcopy(reft_model.interventions[original_key][0].rotate_layer)
        subspace_coeff = l * torch.ones(1).to(device)
    
    
    reft_composed.interventions[composed_key][0].learned_source = copy.deepcopy(reft_model.interventions[original_key][0].learned_source)

    # subspace_coeff = subspace_coeff.to(torch.bfloat16) if fp16 else subspace_coeff.to(torch.float32)
    reft_composed.interventions[composed_key][0].subspace_coeff = subspace_coeff
    # print(f"In set_lm: {composed_key}, {reft_model.interventions[original_key][0].learned_source.weight[0][0]}, {reft_model.interventions[original_key][0].learned_source.bias[0]},{reft_model.interventions[original_key][0].rotate_layer.parametrizations.weight.original[0][0]}")
    # print(f"In set_lm: {composed_key}", reft_composed.interventions[composed_key][0].subspace_coeff)
    
    composed_key = "comp.decoder.block." + str(layer) + ".output.unit.pos.nunit.1#" + str(intervention_id)
    original_key = "comp.decoder.block." + str(layer) + ".output.unit.pos.nunit.1#0"
    
    if intervention_type == "loreft":
        reft_composed.interventions[composed_key][0].rotate_layer = copy.deepcopy(reft_model.interventions[original_key][0].rotate_layer)
        subspace_coeff = l * torch.ones(low_rank).to(device)
    elif intervention_type == "nodireft":
        reft_composed.interventions[composed_key][0].proj_layer = copy.deepcopy(reft_model.interventions[original_key][0].proj_layer)
        subspace_coeff = l * torch.ones(1).to(device)
    elif intervention_type == "direft" or intervention_type == "coeffloreft":
        reft_composed.interventions[composed_key][0].rotate_layer = copy.deepcopy(reft_model.interventions[original_key][0].rotate_layer)
        subspace_coeff = l * torch.ones(1).to(device)
    
    reft_composed.interventions[composed_key][0].learned_source = copy.deepcopy(reft_model.interventions[original_key][0].learned_source)

    # subspace_coeff = subspace_coeff.to(torch.bfloat16) if fp16 else subspace_coeff.to(torch.float32)
    reft_composed.interventions[composed_key][0].subspace_coeff = subspace_coeff
    
    # print(f"In set_lm: {composed_key}", reft_composed.interventions[composed_key][0].subspace_coeff)
    # print(f"In set_lm: {reft_composed.interventions['comp.encoder.block.10.output.unit.pos.nunit.1#0'][0].subspace_coeff}")
    return reft_composed

# In[141]:


def set_eval(reft_model):
    reft_model.eval()
    reft_model.model.eval()
    reft_model.training = False

def set_eval_ddp(reft_model):
    reft_model.eval()
    set_eval(reft_model.module)

set_eval(reft_yelp_lm)
set_eval(reft_amazon_lm)
set_eval(reft_yelp_classifier)
set_eval(reft_amazon_classifier)



gen_batch_size = train_batch_size * 2
force_flexible = ["negative","positive"]
force_words_ids = [tokenizer(force_flexible, add_special_tokens=True).input_ids]


# In[144]:

# In[145]:

gen_max_length = 2
def generate_texts(reft_model, prompt, allowed_token_ids, num_interventions=1, intervene_on_all=True):
    # instruction = " "
    
    # print(prompt)

    for k, v in prompt.items():
        if isinstance(v, list):
            prompt[k] = torch.tensor(v,dtype=torch.long).to(device)

    # prompt = prompt.to(device)
    gen_batch_size = prompt["input_ids"].shape[0]
    # print(gen_batch_size)
    # print(prompt)
    
    generated_texts = []
    subspaces = [[FULL_SUBSPACE] * gen_batch_size] * num_interventions
    # print(subspaces)
    # print(allowed_token_ids)
    _, reft_response = reft_model.generate(
        prompt, 
        unit_locations= None if intervene_on_all else {"sources->base": (None, [[[0] ] ] * len(layers)) },
        subspaces=subspaces,
        intervene_on_prompt=True, 
        max_new_tokens=2,
        min_new_tokens=1,
        # do_sample=True, 
        # no_repeat_ngram_size=2, 
        # repetition_penalty=1.1, 
        force_words_ids=allowed_token_ids,
        num_beams = 5,
        # top_k = 50,
        eos_token_id=tokenizer.eos_token_id, early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        remove_invalid_values=True,
    )

    generated_text = tokenizer.batch_decode(reft_response, skip_special_tokens=True)
    # generated_text = [t[len(instruction):] for t in generated_text]
    generated_texts += generated_text

    # print(generated_texts[0])
    return generated_texts



def eval(reft_model, num_interventions, train_dataset, data_collator):
    acc = torch.tensor(0.0, device=device)
    tot = torch.tensor(0.0, device=device)

    eval_sampler = DistributedSampler(train_dataset,  rank=local_rank)
    dataloader = DataLoader(
        train_dataset,
        batch_size=gen_batch_size,
        collate_fn=data_collator,
        sampler=eval_sampler,
    )
    
    # Iterate over batches
    with torch.no_grad():
        pbar = tqdm(dataloader, disable=(local_rank != 0))
        for batch in pbar:
            if 'subspaces' in batch:
                del batch['subspaces']
            batch = {k: v.to(device) for k, v in batch.items()}
            output_labels = generate_texts(reft_model, batch, force_words_ids, num_interventions)
            filtered_labels = [
                [token_id for token_id in sequence if token_id != -100]
                for sequence in batch["labels"]
            ]
            true_labels = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)
            # Compute accuracy locally on each GPU
            correct = torch.tensor(0.0, device=device)
            for i in range(len(output_labels)):
                if output_labels[i] == true_labels[i]:
                    correct += 1
            
            acc += correct
            tot += len(output_labels)
            if local_rank == 0:
                pbar.set_postfix({"Correct": acc.item(), "Accuracy": (acc / tot).item()})

    dist.reduce(acc, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(tot, dst=0, op=dist.ReduceOp.SUM)
    if local_rank == 0:
        final_acc = (acc/tot).item()
        print(f"Final Accuracy: {final_acc:.4f}")

# In[ ]:

for i in [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]:

    for l in layers:
        # set_lm(reft_yelp_classifier, l, 1.0, 0)
        reft_composed = set_lm(reft_yelp_lm, reft_composed, l, i, 0) # 1.0
        reft_composed = set_lm(reft_amazon_lm, reft_composed, l, -i, 1) # -1.0
        # reft_composed = set_lm(reft_yelp_lm, reft_composed, l, -1.0, 1) # -1.0
        reft_composed = set_lm(reft_amazon_classifier, reft_composed, l, 1.0, 2) # 1.0
    set_eval(reft_composed)
    
    
    reft_train_dataset = make_all_positions_unsupervised_data_module(tokenizer, model, train_dataset, num_interventions= 3 * num_interventions, nonstop=False)
    reft_train_dataset, data_collator = reft_train_dataset["train_dataset"], reft_train_dataset["data_collator"]
    eval(reft_composed, 3 * num_interventions, reft_train_dataset, data_collator)



reft_train_dataset = make_all_positions_unsupervised_data_module(tokenizer, model, train_dataset, num_interventions= num_interventions, nonstop=False)
reft_train_dataset, data_collator = reft_train_dataset["train_dataset"], reft_train_dataset["data_collator"]
eval(reft_amazon_classifier, num_interventions, reft_train_dataset, data_collator)




reft_train_dataset = make_all_positions_unsupervised_data_module(tokenizer, model, train_dataset, num_interventions=num_interventions, nonstop=False)
reft_train_dataset, data_collator = reft_train_dataset["train_dataset"], reft_train_dataset["data_collator"]
eval(reft_yelp_classifier, num_interventions, reft_train_dataset, data_collator)


# In[ ]:
reft_composed = get_reft_model(model, composed_reft_config, set_device=False)
reft_composed.set_device(device)
print(reft_composed.get_device())
reft_composed.print_trainable_parameters()


train_dataset = datasets.load_from_disk("validation/amazon_classify_data") # amazon classifier
if testing: train_dataset = train_dataset.select(range(max_train_examples))
len(train_dataset)

for i in [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]:

    for l in layers:
        reft_composed = set_lm(reft_amazon_lm, reft_composed, l, i, 0) # 1.0
        reft_composed = set_lm(reft_yelp_lm, reft_composed, l, -i, 1) # -1.0
        reft_composed = set_lm(reft_yelp_classifier, reft_composed, l, 1.0, 2) # 1.0
    set_eval(reft_composed)
    
    
    reft_train_dataset = make_all_positions_unsupervised_data_module(tokenizer, model, train_dataset, num_interventions=3 * num_interventions, nonstop=False)
    reft_train_dataset, data_collator = reft_train_dataset["train_dataset"], reft_train_dataset["data_collator"]
    eval(reft_composed, 3 * num_interventions, reft_train_dataset, data_collator)



reft_train_dataset = make_all_positions_unsupervised_data_module(tokenizer, model, train_dataset, num_interventions=num_interventions, nonstop=False)
reft_train_dataset, data_collator = reft_train_dataset["train_dataset"], reft_train_dataset["data_collator"]
eval(reft_amazon_classifier, num_interventions, reft_train_dataset, data_collator)




reft_train_dataset = make_all_positions_unsupervised_data_module(tokenizer, model, train_dataset, num_interventions=num_interventions, nonstop=False)
reft_train_dataset, data_collator = reft_train_dataset["train_dataset"], reft_train_dataset["data_collator"]
eval(reft_yelp_classifier, num_interventions, reft_train_dataset, data_collator)




# # In[ ]:
