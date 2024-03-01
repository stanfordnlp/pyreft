"""
Data loading and preprocessing.
"""

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
from templates import *
from tqdm import tqdm
import torch
import json
import os
import io
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def create_directory(path):
    """Create directory if not exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")
        

def chunk(iterable, chunksize):
    # if iterable is a list, we chunk with simple list indexing
    if isinstance(iterable, list):
        return [iterable[i:i+chunksize] for i in range(0, len(iterable), chunksize)]
    # otherwise if iterable is a Hf Dataset, we leverage the select() function to create mini datasets
    elif isinstance(iterable, Dataset):
        chunks = []
        for i in range(0, len(iterable), chunksize):
            if i+chunksize < len(iterable):
                chunks.append(iterable.select(list(range(i, i+chunksize))))
            else:
                chunks.append(iterable.select(list(range(i, len(iterable)))))
        return chunks
    else:
        raise Exception(f"Unrecognizable type of iterable for batchification: {type(iterable)}")


def reformat_by_task(
    task: str,
    dataset: str,
    task_prompt_template: str,
    trigger_tokens: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    position: str="last",
    layers: int=[1],
    split: str='train'
) -> tuple:
    """Reformat the dataset based on task template and generate tokenized inputs."""

    print("loading data for dataset: ", dataset)
    task_dataset = jload(f"./datasets/{dataset}/{split}.json")
    result = defaultdict(list)
    for i, data_item in enumerate(tqdm(task_dataset)):

        # format task-specific prompt
        if task == "commonsense":
            base_prompt = task_prompt_template % (data_item['instruction'])
            base_input = base_prompt + trigger_tokens + data_item["answer"] + tokenizer.eos_token
        elif task == "math":
            # we strip since these are model generated examples.
            base_prompt = task_prompt_template % (data_item['instruction'])
            base_input = base_prompt + data_item["output"] + tokenizer.eos_token
        elif task == "alpaca" or task == "instruct":
            if data_item['input'] == "":
                base_prompt = alpaca_prompt_no_input_template % (data_item['instruction'])
                base_input = base_prompt + data_item["output"] + tokenizer.eos_token
            else:
                base_prompt = task_prompt_template % (data_item['instruction'], data_item['input'])
                base_input = base_prompt + data_item["output"] + tokenizer.eos_token
        
        # tokenize
        base_prompt_length = len(tokenizer(
            base_prompt, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"][0])
        base_input_ids = tokenizer(
            base_input, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        output_ids = tokenizer(
            base_input, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"][0]

        # mask prompt in labels
        output_ids[:base_prompt_length] = -100
        base_input_ids[-1] = tokenizer.eos_token_id # enforce the last token to be eos
        output_ids[-1] = tokenizer.eos_token_id # enforce the last token to be eos
        last_position = torch.tensor([base_prompt_length-1,])

        # compute intervention positions
        base_last_location = last_position.tolist()
        if position in {"first+last"}:
            base_first_location = torch.zeros_like(last_position).tolist()
            intervention_locations = [base_first_location]*(len(layers)//2)+[base_last_location]*(len(layers)//2)
        else:
            intervention_locations = [base_last_location]*len(layers)

        result["input_ids"].append(base_input_ids)
        result["intervention_locations"].append(intervention_locations)
        result["labels"].append(output_ids)
        result["id"].append(i)
    return result, task_dataset

def load_task(
    task: str,
    tokenizer: AutoTokenizer,
    max_n_train_example: int=None,
    max_n_eval_example: int=None,
    train_dataset: list=None,
    eval_dataset: list=None,
    seed: int=42,
    eval_batch_size: int=1,
    position: str="last",
    layers: int=[1],
):
    # config
    if task == "commonsense":
        max_length = 512
        train_datasets = [
            "boolq", "piqa", "social_i_qa", "hellaswag", 
            "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"
        ] if train_dataset is None else [train_dataset]
        eval_datasets = [
            "boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"
        ] if eval_dataset is None else [eval_dataset]
        task_prompt_template = "%s\n"
        trigger_tokens = "the correct answer is "
    elif task == "math":
        max_length = 512
        train_datasets = [
            "math_10k"
        ] if train_dataset is None else [train_dataset]
        eval_datasets = [
            "MultiArith", "gsm8k", "SVAMP", "mawps", "AddSub", "AQuA", "SingleEq", 
        ] if eval_dataset is None else [eval_dataset]
        task_prompt_template = alpaca_prompt_no_input_template
        trigger_tokens = "### Response:\n"
    elif task == "alpaca":
        max_length = 512
        train_datasets = [
            "alpaca_data_cleaned"
        ]
        task_prompt_template = alpaca_prompt_template
        trigger_tokens = "### Response:\n"
    elif task == "instruct":
        max_length = 2048
        train_datasets = [
            "instruct"
        ]
        task_prompt_template = alpaca_prompt_template
        trigger_tokens = "### Response:\n"
    
    # load data
    raw_train = defaultdict(list)
    for dataset in train_datasets:
        result, _ = reformat_by_task(
            task, dataset, task_prompt_template, trigger_tokens, tokenizer,
            max_length, position, layers, split='train')
        for key in result:
            raw_train[key].extend(result[key])
        del _ # remove the task_dataset variable from memory
    
    # make dataset obj
    train_dataset = Dataset.from_dict(raw_train).shuffle(seed=seed)
    if max_n_train_example is not None:
        train_dataset = train_dataset.select(range(max_n_train_example))
    
    # eval
    all_eval_datasets = {}
    for dataset in eval_datasets:
        result, task_dataset = reformat_by_task(
            task, dataset, task_prompt_template, trigger_tokens, tokenizer,
            max_length, position, layers, split='test')
        eval_dataset = Dataset.from_dict(result)
        if max_n_eval_example is not None:
            eval_dataset = eval_dataset.select(range(max_n_eval_example))
        all_eval_datasets[dataset] = (eval_dataset, task_dataset)

    return train_dataset, all_eval_datasets, trigger_tokens