import os
from copy import deepcopy

import torch
import random
import transformers
from datasets import load_dataset
from collections import defaultdict

from task_config import task_config
from templates import *

from pyreft import ReftDataset

glue_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


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


class LoReftGLUEDataset(ReftDataset):

    def preprocess(self, kwargs):
        # basic setup
        self.raw_dataset, self.trigger_tokens, self.num_labels = None, None, None
        self.pad_mode = "last" # pad token placed at end for intervention sink
        self.fields_to_pad = ["input_ids"] # labels are classification so no need to pad

        # keys for prompt
        self.sentence1_key, self.sentence2_key = glue_task_to_keys[self.data_path]
    
    def postprocess(self, kwargs):
        # get the number of classification labels
        is_regression = self.data_path == "stsb"
        if not is_regression:
            label_list = self.task_dataset.features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
        self.num_labels = num_labels

    def tokenize(self, data_item):
        result = {}

        # tokenize
        args = ((data_item[self.sentence1_key],)
                if self.sentence2_key is None
                else (data_item[self.sentence1_key], data_item[self.sentence2_key]))
        base_input_ids = self.tokenizer(
            *args, max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors="pt"
        )["input_ids"][0]
        output_ids = data_item["label"]
        last_position = len(base_input_ids)

        # store
        result["input_ids"] = base_input_ids
        result["labels"] = output_ids

        return result, last_position
    

class LoReftSupervisedDataset(ReftDataset):

    def preprocess(self, kwargs):
        print(kwargs)
        # basic setup
        self.raw_dataset, self.trigger_tokens, self.num_labels = None, None, None
        dataset_config = task_config[self.task]
        self.task_prompt_template = dataset_config["task_prompt_template"]
        self.trigger_tokens = dataset_config["trigger_tokens"]
        self.original_data_split = self.data_split
        self.test_split = kwargs["test_split"] if "test_split" in kwargs else None
        
        # where to pull dataset from
        # instruction-tuning tasks should all eval on alpaca_eval
        if self.task in ["alpaca", "instruct", "ultrafeedback", "ultrafeedback_pair"] and self.data_split != "train":
            self.task = "tatsu-lab/alpaca_eval"
            self.data_path = "alpaca_eval"
            self.data_split = "eval"
        if self.task in ["gsm8k"]:
            self.data_path = "main" # huggingface dir.
            if self.data_split != "test":
                self.data_split = "train" # we split l300 examples from train for validation.
        elif self.task in ["math", "commonsense", "ultrafeedback"]:
            self.data_path = os.path.join(self.data_path, self.data_split + ".json")

    def postprocess(self, kwargs):
        original_dataset_size = len(self.task_dataset)
        if self.task in ["gsm8k"] and \
            self.original_data_split == "train" and self.test_split == "validation":
            self.task_dataset = self.task_dataset.select(
                range(original_dataset_size - 300))
        if self.task in ["gsm8k"] and self.original_data_split == "validation":
            self.task_dataset = self.task_dataset.select(
                range(original_dataset_size - 300, original_dataset_size))
        self.raw_dataset = self.task_dataset # also update the raw dataset pointer.
        return
    
    def tokenize(self, data_item):
        result = {}

        if "Meta-Llama-3" in self.tokenizer.name_or_path:
            base_prompt = self.tokenizer.bos_token
        else:
            base_prompt = ""

        # set up prompt
        if self.task == "commonsense":
            base_prompt += self.task_prompt_template % (data_item['instruction'])
            base_input = base_prompt + self.trigger_tokens + data_item["answer"] + self.tokenizer.eos_token
        elif self.task == "math": # we strip since these are model generated examples.
            base_prompt += self.task_prompt_template % (data_item['instruction'])
            base_input = base_prompt + data_item["output"] + self.tokenizer.eos_token
        elif self.task in ["alpaca", "instruct", "ultrafeedback", "ultrafeedback_pair", "tatsu-lab/alpaca_eval"]:
            if 'input' not in data_item or data_item['input'] == "":
                base_prompt += alpaca_prompt_no_input_template % (data_item['instruction'])
            else:
                base_prompt += self.task_prompt_template % (data_item['instruction'], data_item['input'])
            if self.task == "ultrafeedback_pair" and self.data_split == "train":
                # base input takes rejected output to steer away from.
                base_input = base_prompt + data_item["rejected_output"] + self.tokenizer.eos_token
            else:
                base_input = base_prompt + data_item["output"] + self.tokenizer.eos_token
        elif self.task == "gsm8k": # setup is from https://github.com/yxli2123/LoftQ/
            base_prompt += self.task_prompt_template % (
                "Answer the above question. First think step by step and then answer the final number.",
                data_item['question']
            )
            base_input = base_prompt + data_item["answer"].replace("####", "The final answer is: ") + \
                self.tokenizer.eos_token
        else:
            raise ValueError(f"Unrecognized task: {self.task}")
            
        # tokenize
        base_prompt_ids = self.tokenizer(
            base_prompt, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        if self.original_data_split == "train":
            base_input_ids = self.tokenizer(
                base_input, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]

            if self.task == "ultrafeedback_pair" and self.data_split == "train":
                # base output takes chosen output to steer towards to.
                base_output = base_prompt + data_item["chosen_output"] + self.tokenizer.eos_token
                
                base_output_ids = self.tokenizer(
                    base_output, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
                output_ids = base_output_ids
                output_ids[:base_prompt_length] = IGNORE_INDEX

                # padding! needs to be cautious here. let's unpack:
                # pad inputs with pad_token_id so that attention masks can ignore these tokens.
                # pad outputs with IGNORE_INDEX so that loss calculation can ignore these tokens.
                # and the goal is to have input and output have the same length.
                max_length = max(base_input_ids.size(0), output_ids.size(0))
                input_pad_length = max_length - base_input_ids.size(0)
                output_pad_length = max_length - output_ids.size(0)

                input_pad_tensor = torch.full((input_pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                output_pad_tensor = torch.full((output_pad_length,), IGNORE_INDEX, dtype=torch.long)

                base_input_ids = torch.cat((base_input_ids, input_pad_tensor), dim=0)
                output_ids = torch.cat((output_ids, output_pad_tensor), dim=0)
            else:
                output_ids = deepcopy(base_input_ids)
                output_ids[:base_prompt_length] = IGNORE_INDEX
                
            result["input_ids"] = base_input_ids
            result["labels"] = output_ids
        else:
            # print("Assuming test split for now")
            result["input_ids"] = base_prompt_ids
        last_position = base_prompt_length

        return result, last_position