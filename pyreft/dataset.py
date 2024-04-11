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

import os
import abc
import copy
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any

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
    __metaclass__ = abc.ABCMeta

    def __init__(
        self, task: str, data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_split="train", dataset=None, seed=42, max_n_example=None,
        **kwargs,
    ):
        super(ReftDataset, self).__init__()
        result = defaultdict(list)

        # setup
        self.tokenizer = tokenizer
        self.first_n, self.last_n = parse_positions(kwargs["position"])

        # load the dataset
        self.data_path = data_path
        self.data_split = data_split
        self.dataset = dataset
        self.seed = seed
        self.max_n_example = max_n_example
        task_dataset = self.load_dataset()

        # tokenize and intervene
        self.result = []
        for i, data_item in enumerate(tqdm(task_dataset)):
            tokenized, last_position = self.tokenize(data_item, **kwargs)
            tokenized = self.compute_intervention_and_subspaces(i, data_item, tokenized, last_position, **kwargs)
            self.result.append(tokenized)

    @abc.abstractmethod
    def tokenize(self, data_item, **kwargs):
        """How to tokenize a single data item. Override this function!"""
        return
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.result[i]

    def load_dataset(self):
        """Load the dataset (or a portion of it) from HF or a local file."""
        if self.dataset is None:
            print("loading data for dataset: ", self.data_path)
            if self.data_path.endswith(".json"):
                task_dataset = load_dataset("json", data_files=self.data_path)[self.data_split]
            else:
                task_dataset = load_dataset(self.data_path)[self.data_split]
        else:
            task_dataset = self.dataset
        if self.max_n_example is not None:
            task_dataset = task_dataset.shuffle(seed=self.seed)
            task_dataset = task_dataset.select(range(self.max_n_example))

        # save raw_dataset pointer for access raw strings
        self.raw_dataset = task_dataset if self.data_split != "train" else None
        return task_dataset
        
    def get_intervention_locations(self, **kwargs):
        return get_intervention_locations(**kwargs)
    
    def compute_intervention_and_subspaces(self, id: int, data_item, result: dict, last_position: int, **kwargs):
        # compute intervention locs
        intervention_locations = self.get_intervention_locations(last_position=last_position, first_n=self.first_n, 
            last_n=self.last_n, pad_mode="first", **kwargs)
        result["intervention_locations"] = intervention_locations
        result["id"] = id
            
        # add a single padding token BEFORE input_ids and fix everything
        result["input_ids"] = torch.cat((torch.tensor([self.tokenizer.pad_token_id,]), result["input_ids"]))
        if "labels" in result:
            result["labels"] = torch.cat((torch.tensor([IGNORE_INDEX]), result["labels"]))
        result["intervention_locations"] = (torch.IntTensor(result["intervention_locations"]) + 1).tolist()
        if "subspaces" in data_item:
            num_interventions = kwargs["num_interventions"]
            share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
            if share_weights:
                num_interventions = num_interventions // 2
            # we now assume each task has a constant subspaces
            _subspaces = [data_item["subspaces"]] * num_interventions
            result["subspaces"].append(_subspaces)

        return result


class ReftClassificationDataset(ReftDataset):
    """
    A ReftClassificationDataset only contains a single text field
    that we tokenize, intervene on a prefix + suffix of, and
    compute subspace settings for. This is intended for classification
    tasks.

    Remember to pass in the input_field and label_field as kwargs.
    """

    def tokenize(self, data_item, **kwargs):
        result = {}
        
        # input
        input_ids = self.tokenizer(data_item[kwargs["input_field"]], max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(input_ids)
        last_position = base_prompt_length - 1
        result["input_ids"] = input_ids
        result["attention_mask"] = ((input_ids != self.tokenizer.pad_token_id).int())

        # labels
        if kwargs["label_field"] == kwargs["input_field"]:
            result["labels"] = input_ids.clone()
        elif kwargs["label_field"] is not None:
            labels = self.tokenizer(data_item[kwargs["label_field"]], max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt")["input_ids"][0]
            result["labels"] = labels
            
        return result, last_position


class ReftGenerationDataset(ReftDataset):
    """
    A ReftGenerationDataset contains an instruction and a 
    completion for each data item. We intervene on a prefix + suffix
    of *only the instruction*. This is suitable for generation tasks
    where you don't want inference overhead during decoding.

    Remember to pass in the prompt_field and completion_field as kwargs.
    """

    def tokenize(self, data_item, **kwargs):
        result = {}
        
        # prompt
        prompt_ids = self.tokenizer(data_item[kwargs["prompt_field"]], max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(prompt_ids)
        last_position = base_prompt_length - 1
        
        # input
        full_input = data_item[kwargs["prompt_field"]] + data_item[kwargs["completion_field"]] + self.tokenizer.eos_token
        input_ids = self.tokenizer(full_input, max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")["input_ids"][0]
        result["input_ids"] = input_ids
        result["attention_mask"] = ((input_ids != self.tokenizer.pad_token_id).int())

        # labels
        output_ids = copy.deepcopy(input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX
        result["labels"] = output_ids
            
        return result, last_position


class ReftSupervisedDataset(ReftDataset):
    """
    Alpaca-style supervised dataset. We intervene on a prefix + suffix
    of the input. This is suitable for supervised fine-tuning tasks.

    Remember to pass in the input_field, output_field, and instruction_field as kwargs.
    """

    def tokenize(self, data_item, **kwargs):
        result = {}

        # prompt
        if kwargs['input_field'] not in data_item or data_item[kwargs['input_field']] == "":
            base_prompt = prompt_no_input % (data_item[kwargs['instruction_field']])
        else:
            base_prompt = prompt_input % (data_item[kwargs['instruction_field']], data_item[kwargs['input_field']])
        prompt_ids = self.tokenizer(base_prompt, max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(prompt_ids)
        last_position = base_prompt_length - 1
        
        # input
        base_input = base_prompt + data_item[kwargs['output_field']] + self.tokenizer.eos_token
        input_ids = self.tokenizer(base_input, max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")["input_ids"][0]
        result["input_ids"] = input_ids
        result["attention_mask"] = (input_ids != self.tokenizer.pad_token_id).int()

        # labels
        output_ids = copy.deepcopy(input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX
        result["labels"] = output_ids
            
        return result, last_position


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


# class ReftPreferenceDataset(ReftDataset):
#     """
#     Different from ReftSupervisedDataset where we have
#     (x, y)
#     ReftPreferenceDataset contains (x, y1, y2) where y1 and y2
#     are constrastive pairs.
#     ReFT training objective is to generate y2, given (x, y1) and
#     the intervention.
#     """
#     def __init__(
#         self, task: str, data_path: str,
#         tokenizer: transformers.PreTrainedTokenizer,
#         data_split="train", dataset=None, seed=42, max_n_example=None, 
#         **kwargs,
#     ):
#         super(ReftPreferenceDataset, self).__init__()
#         result = defaultdict(list)

#         if dataset is None:
#             print("loading data for dataset: ", data_path)
#             if data_path.endswith(".json"):
#                 task_dataset = load_dataset("json", data_files=data_path)[data_split]
#             else:
#                 task_dataset = load_dataset(data_path)[data_split]
#         else:
#             task_dataset = dataset
#         if max_n_example is not None:
#             task_dataset = task_dataset.shuffle(seed=seed)
#             task_dataset = task_dataset.select(range(max_n_example))

#         # save raw_dataset pointer for access raw strings
#         self.raw_dataset = task_dataset if data_split != "train" else None
#         first_n, last_n = parse_positions(kwargs["position"])

#         # tokenize and intervene
#         for i, data_item in enumerate(tqdm(task_dataset)):
#             if 'input' not in data_item or data_item['input'] == "":
#                 base_prompt = prompt_no_input % (data_item['instruction'])
#             else:
#                 base_prompt = prompt_input % (data_item['instruction'], data_item['input'])
#             # base input takes rejected output to steer away from.
#             base_input = base_prompt + data_item["rejected_output"] + tokenizer.eos_token

#             # tokenize
#             base_prompt_ids = tokenizer(
#                 base_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
#             base_prompt_length = len(base_prompt_ids)

#             if data_split == "train":
#                 base_input_ids = tokenizer(
#                     base_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
#                 # base output takes chosen output to steer towards to.
#                 base_output = base_prompt + data_item["chosen_output"] + tokenizer.eos_token

#                 base_output_ids = tokenizer(
#                     base_output, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
#                 output_ids = base_output_ids
#                 output_ids[:base_prompt_length] = IGNORE_INDEX

#                 # padding! needs to be cautious here. let's unpack:
#                 # pad inputs with pad_token_id so that attention masks can ignore these tokens.
#                 # pad outputs with IGNORE_INDEX so that loss calculation can ignore these tokens.
#                 # and the goal is to have input and output have the same length.
#                 max_length = max(base_input_ids.size(0), output_ids.size(0))
#                 input_pad_length = max_length - base_input_ids.size(0)
#                 output_pad_length = max_length - output_ids.size(0)

#                 input_pad_tensor = torch.full((input_pad_length,), tokenizer.pad_token_id, dtype=torch.long)
#                 output_pad_tensor = torch.full((output_pad_length,), IGNORE_INDEX, dtype=torch.long)

#                 base_input_ids_padded = torch.cat((base_input_ids, input_pad_tensor), dim=0)
#                 output_ids_padded = torch.cat((output_ids, output_pad_tensor), dim=0)

#                 result["input_ids"].append(base_input_ids_padded)
#                 result["labels"].append(output_ids_padded)
#             else:
#                 # print("Assuming test split for now")
#                 result["input_ids"].append(base_prompt_ids)
#             last_position = base_prompt_length

#             # get intervention locations
#             intervention_locations = self.get_intervention_locations(
#                 last_position=last_position, 
#                 first_n=first_n, 
#                 last_n=last_n,
#                 pad_mode="first",
#                 **kwargs
#             )
#             result["intervention_locations"].append(intervention_locations)
#             result["id"].append(i)

#             # add a single padding token BEFORE input_ids and fix everything
#             result["input_ids"][-1] = torch.cat((torch.tensor([tokenizer.pad_token_id,]), result["input_ids"][-1]))
#             if data_split == "train":
#                 result["labels"][-1] = torch.cat((torch.tensor([IGNORE_INDEX]), result["labels"][-1]))
#             result["intervention_locations"][-1] = (torch.IntTensor(result["intervention_locations"][-1]) + 1).tolist()
#             result["attention_mask"].append((result["input_ids"][-1] != tokenizer.pad_token_id).int())
#             if "subspaces" in data_item:
#                 num_interventions = kwargs["num_interventions"]
#                 share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
#                 if share_weights:
#                     num_interventions = num_interventions // 2
#                 # we now assume each task has a constant subspaces
#                 _subspaces = [data_item["subspaces"]] * num_interventions
#                 result["subspaces"].append(_subspaces)

#         self.input_ids = result["input_ids"]
#         self.attention_mask = result["attention_mask"]
#         self.intervention_locations = result["intervention_locations"]
#         self.labels = result["labels"] if "labels" in result else None
#         self.subspaces = result["subspaces"] if "subspaces" in result else None
#         self.id = result["id"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return_dict = dict(
#             input_ids=self.input_ids[i],
#             attention_mask=self.attention_mask[i],
#             intervention_locations=self.intervention_locations[i],
#             id=self.id[i],
#         )
#         if self.labels is not None:
#             return_dict["labels"] = self.labels[i]
#         if self.subspaces is not None:
#             return_dict["subspaces"] = self.subspaces[i]
#         return return_dict


# class ReftRewardDataset(ReftDataset):
#     """
#     ReftRewardDataset contains tuples (x, y1, y2, r1, r2) where x is
#     a common instruction prefix, y1 and y2 are two possible continuations
#     of the instruction, and r1 and r2 are the rewards associated with
#     y1 and y2, respectively.
#     """
#     def __init__(
#         self, task: str, data_path: str,
#         tokenizer: transformers.PreTrainedTokenizer,
#         data_split="train", dataset=None, seed=42, max_n_example=None, 
#         **kwargs,
#     ):
#         super(ReftRewardDataset, self).__init__()
#         if dataset is None:
#             print("loading data for dataset: ", data_path)
#             if data_path.endswith(".json"):
#                 task_dataset = load_dataset("json", data_files=data_path)[data_split]
#             else:
#                 task_dataset = load_dataset(data_path)[data_split]
#         else:
#             task_dataset = dataset
#         if max_n_example is not None:
#             task_dataset = task_dataset.select(range(max_n_example))

#         # parse pos
#         first_n, last_n = parse_positions(kwargs["position"])
        
#         # tokenize and intervene
#         def tokenize(data_item):
#             # generate prompt format
#             result = {}
#             chosen_output = tokenizer.apply_chat_template(data_item["conv_A"], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
#             rejected_output = tokenizer.apply_chat_template(data_item["conv_B"], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
#             result["chosen_reward"] = data_item["conv_A_rating"]
#             result["rejected_reward"] = data_item["conv_B_rating"]

#             # swap so that chosen is better
#             if result["chosen_reward"] < result["rejected_reward"]:
#                 chosen_output, rejected_output = rejected_output, chosen_output
#                 result["chosen_reward"], result["rejected_reward"] = result["rejected_reward"], result["chosen_reward"]

#             # get common prefix, which is what we intervene on

#             # tokenize
#             chosen_ids = tokenizer(
#                 chosen_output, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
#             rejected_ids = tokenizer(
#                 rejected_output, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
#             base_prompt_length = 0
#             for i in range(min(len(chosen_ids), len(rejected_ids))):
#                 base_prompt_length += 1
#                 if chosen_ids[i] != rejected_ids[i]:
#                     break

#             result["chosen_output"] = chosen_ids
#             result["rejected_output"] = rejected_ids

#             # get intervention locations
#             intervention_locations = self.get_intervention_locations(
#                 last_position=base_prompt_length - 1, 
#                 first_n=first_n, 
#                 last_n=last_n,
#                 pad_mode="first",
#                 **kwargs
#             )
#             result["intervention_locations"] = intervention_locations

#             # add a single padding token BEFORE input_ids and fix everything
#             result["chosen_output"] = torch.cat((torch.tensor([tokenizer.pad_token_id,]), result["chosen_output"]))
#             result["rejected_output"] = torch.cat((torch.tensor([tokenizer.pad_token_id,]), result["rejected_output"]))
#             result["intervention_locations"] = (torch.IntTensor(result["intervention_locations"]) + 1).tolist()
#             result["chosen_mask"] = (result["chosen_output"] != tokenizer.pad_token_id).int()
#             result["rejected_mask"] = (result["rejected_output"] != tokenizer.pad_token_id).int()
#             return result

#         final_results = task_dataset.map(tokenize, num_proc=4)

#         self.chosen_output = final_results["chosen_output"]
#         self.chosen_mask = final_results["chosen_mask"]
#         self.chosen_reward = final_results["chosen_reward"]
#         self.rejected_output = final_results["rejected_output"]
#         self.rejected_mask = final_results["rejected_mask"]
#         self.rejected_reward = final_results["rejected_reward"]
#         self.intervention_locations = final_results["intervention_locations"]
    
#     def __len__(self):
#         return len(self.chosen_output)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return_dict = dict(
#             chosen_output=self.chosen_output[i],
#             chosen_mask=self.chosen_mask[i],
#             chosen_reward=self.chosen_reward[i],
#             rejected_output=self.rejected_output[i],
#             rejected_mask=self.rejected_mask[i],
#             rejected_reward=self.rejected_reward[i],
#             intervention_locations=self.intervention_locations[i],
#         )
#         return return_dict