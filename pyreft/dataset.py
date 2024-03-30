IGNORE_INDEX = -100

no_header_prompt_template = """\
### Instruction:
%s

### Response:
"""

prompt_input = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

prompt_no_input = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""

import copy
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import random
import transformers
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
from collections import defaultdict

from transformers import DataCollator


def parse_positions(positions: str):
    # parse position
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


def get_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
    last_position = kwargs["last_position"]
    if "positions" in kwargs:
        _first_n, _last_n = parse_positions(kwargs["positions"])
    else:
        _first_n, _last_n = kwargs["first_n"], kwargs["last_n"]
    num_interventions = kwargs["num_interventions"]
    pad_mode = kwargs["pad_mode"] if "pad_mode" in kwargs else "first"

    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1 if pad_mode == "first" else last_position
    if share_weights or (first_n == 0 or last_n == 0):
        position_list = [i for i in range(first_n)] + \
            [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(pad_amount)]
        intervention_locations = [position_list]*num_interventions
    else:
        left_pad_amount = (_first_n - first_n)
        right_pad_amount = (_last_n - last_n)
        left_intervention_locations = [i for i in range(first_n)] + [pad_position for _ in range(left_pad_amount)]
        right_intervention_locations = [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(right_pad_amount)]
        # after padding, there could be still length diff, we need to do another check
        left_len = len(left_intervention_locations)
        right_len = len(right_intervention_locations)
        if left_len > right_len:
            right_intervention_locations += [pad_position for _ in range(left_len-right_len)]
        else:
            left_intervention_locations += [pad_position for _ in range(right_len-left_len)]
        intervention_locations = [left_intervention_locations]*(num_interventions//2) + \
            [right_intervention_locations]*(num_interventions//2)
    
    return intervention_locations


@dataclass
class ReftDataCollator(object):
    """Collate examples for ReFT."""

    data_collator: DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][..., :max_seq_length]
        return batch_inputs


class ReftDataset(Dataset):
        
    def get_intervention_locations(self, **kwargs):
        return get_intervention_locations(**kwargs)


class ReftSupervisedDataset(ReftDataset):

    def __init__(
        self, task: str, data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_split="train", dataset=None, seed=42, max_n_example=None, 
        **kwargs,
    ):
        super(ReftSupervisedDataset, self).__init__()
        result = defaultdict(list)

        if dataset is None:
            print("loading data for dataset: ", data_path)
            if data_path.endswith(".json"):
                task_dataset = load_dataset("json", data_files=data_path)[data_split]
            else:
                task_dataset = load_dataset(data_path)[data_split]
        else:
            task_dataset = dataset
        if max_n_example is not None:
            task_dataset = task_dataset.shuffle(seed=seed)
            task_dataset = task_dataset.select(range(max_n_example))

        # save raw_dataset pointer for access raw strings
        self.raw_dataset = task_dataset if data_split != "train" else None
        first_n, last_n = parse_positions(kwargs["position"])
        
        # tokenize and intervene
        for i, data_item in enumerate(tqdm(task_dataset)):
            if 'input' not in data_item or data_item['input'] == "":
                base_prompt = prompt_no_input % (data_item['instruction'])
            else:
                base_prompt = prompt_input % (data_item['instruction'], data_item['input'])
            base_input = base_prompt + data_item["output"] + tokenizer.eos_token

            # tokenize
            base_prompt_ids = tokenizer(
                base_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
            base_prompt_length = len(base_prompt_ids)
            if data_split == "train":
                base_input_ids = tokenizer(
                    base_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
                output_ids = copy.deepcopy(base_input_ids)
                output_ids[:base_prompt_length] = IGNORE_INDEX
                    
                result["input_ids"].append(base_input_ids)
                result["labels"].append(output_ids)
            else:
                # print("Assuming test split for now")
                result["input_ids"].append(base_prompt_ids)
            last_position = base_prompt_length
                
            # get intervention locations
            intervention_locations = self.get_intervention_locations(
                last_position=last_position, 
                first_n=first_n, 
                last_n=last_n,
                pad_mode="first",
                **kwargs
            )
            result["intervention_locations"].append(intervention_locations)
            result["id"].append(i)
            
            # add a single padding token BEFORE input_ids and fix everything
            result["input_ids"][-1] = torch.cat((torch.tensor([tokenizer.pad_token_id,]), result["input_ids"][-1]))
            if data_split == "train":
                result["labels"][-1] = torch.cat((torch.tensor([IGNORE_INDEX]), result["labels"][-1]))
            result["intervention_locations"][-1] = (torch.IntTensor(result["intervention_locations"][-1]) + 1).tolist()
            result["attention_mask"].append((result["input_ids"][-1] != tokenizer.pad_token_id).int())
            if "subspaces" in data_item:
                num_interventions = kwargs["num_interventions"]
                share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
                if share_weights:
                    num_interventions = num_interventions // 2
                # we now assume each task has a constant subspaces
                _subspaces = [data_item["subspaces"]] * num_interventions
                result["subspaces"].append(_subspaces)
        
        self.input_ids = result["input_ids"]
        self.attention_mask = result["attention_mask"]
        self.intervention_locations = result["intervention_locations"]
        self.labels = result["labels"] if "labels" in result else None
        self.subspaces = result["subspaces"] if "subspaces" in result else None
        self.id = result["id"]
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return_dict = dict(
            input_ids=self.input_ids[i],
            attention_mask=self.attention_mask[i],
            intervention_locations=self.intervention_locations[i],
            id=self.id[i],
        )
        if self.labels is not None:
            return_dict["labels"] = self.labels[i]
        if self.subspaces is not None:
            return_dict["subspaces"] = self.subspaces[i]
        return return_dict


def make_last_position_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, model, inputs, outputs) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []
    for i in range(len(inputs)):
        _input = inputs[i]
        _output = outputs[i]
    
        base_prompt = _input
        base_input = base_prompt + _output + tokenizer.eos_token
    
        # tokenize
        base_prompt_ids = tokenizer(
            base_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_input_ids = tokenizer(
            base_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX
        
        all_base_input_ids.append(base_input_ids)
        all_intervention_locations.append([[base_prompt_length - 1]])
        all_output_ids.append(output_ids)
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
    })
        
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

