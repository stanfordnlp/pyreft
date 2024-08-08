"""
Data loading and preprocessing.
"""

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
from task_config import task_config
from copy import deepcopy
from templates import *
from tqdm import tqdm
import torch
import json
import os
import io
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
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


def normalize_text(text):
    """
    Normalize text by removing spaces, linebreaks.
    """
    cleaned_text = text.strip()
    cleaned_text = re.sub(r'\n\n+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+-\s+', '-', cleaned_text)
    cleaned_text = re.sub(r'\s+([$%.,!?;:])', r'\1', cleaned_text)
    return cleaned_text
    

def create_directory(path):
    """Create directory if not exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")


def tokenize(tokenizer, prompt, add_eos_token=True, cutoff_len=255):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(tokenizer, data_point, train_on_inputs=True, cutoff_len=256):
    full_prompt = generate_prompt_for_train(data_point)
    tokenized_full_prompt = tokenize(tokenizer, full_prompt, cutoff_len=cutoff_len)
    user_prompt = generate_prompt_for_train({**data_point, "output": ""})
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=False, cutoff_len=cutoff_len)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    if not train_on_inputs:
        tokenized_full_prompt["labels"] = [
                                              -100
                                          ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                user_prompt_len:
                                                                ]  # could be sped up, probably
    return tokenized_full_prompt, user_prompt_len

    
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


def get_intervention_locations(share_weights, position, last_position, _first_n, _last_n, layers, pad_mode="last"):

    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1 if pad_mode == "first" else last_position
    if share_weights or "+" not in position:
        position_list = [i for i in range(first_n)] + \
            [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(pad_amount)]
        intervention_locations = [position_list]*len(layers)
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
        intervention_locations = [left_intervention_locations]*(len(layers)//2) + \
            [right_intervention_locations]*(len(layers)//2)
    
    return intervention_locations


def reformat_by_task(
    task: str,
    dataset: str,
    task_prompt_template: str,
    trigger_tokens: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    position: str="f1+l1",
    layers: int=[1],
    train_on_inputs: bool=False,
    use_normalized_template: bool=False,
    share_weights: bool=False,
    split: str='train',
    max_n_example: int=None,
    seed: int=42,
    test_split: str="validation",
) -> tuple:
    """Reformat the dataset based on task template and generate tokenized inputs."""

    print("loading data for dataset: ", dataset)
    result = defaultdict(list)
    num_labels = None

    # parse position
    first_n, last_n = 0, 0
    if "+" in position:
        first_n = int(position.split("+")[0].strip("f"))
        last_n = int(position.split("+")[1].strip("l"))
    else:
        if "f" in position:
            first_n = int(position.strip("f"))
        elif "l" in position:
            last_n = int(position.strip("l"))
    
    if task == "glue":
        task_dataset = load_dataset(task, dataset)
        task_dataset = task_dataset[split]

        if split == "train":
            task_dataset = task_dataset.shuffle(seed=seed)
        if max_n_example is not None:
            task_dataset = task_dataset.select(range(max_n_example))
        
        sentence1_key, sentence2_key = glue_task_to_keys[dataset]

        # get the number of classification labels
        is_regression = dataset == "stsb"
        if not is_regression:
            label_list = task_dataset.features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
        
        for i, data_item in enumerate(tqdm(task_dataset)):

            # tokenize
            args = ((data_item[sentence1_key],)
                    if sentence2_key is None
                    else (data_item[sentence1_key], data_item[sentence2_key]))
            base_input_ids = tokenizer(*args, max_length=max_length, truncation=True,
                                       return_tensors="pt")["input_ids"][0]
            output_ids = data_item["label"]

            # get intervention locations
            last_position = len(base_input_ids)
            intervention_locations = get_intervention_locations(
                share_weights, position, last_position, first_n, last_n, layers, pad_mode="last")

            # append to result
            result["input_ids"].append(base_input_ids)
            result["intervention_locations"].append(intervention_locations)
            result["labels"].append(output_ids)
            result["id"].append(i)
            
            # add a single padding token AFTER input_ids and fix everything
            result["input_ids"][-1] = torch.cat((result["input_ids"][-1], torch.tensor([tokenizer.pad_token_id,])))
            result["attention_mask"].append((result["input_ids"][-1] != tokenizer.pad_token_id).int())

    else:
        # set up datasets
        if task in ["alpaca", "instruct", "ultrafeedback"] and split != "train":
            if dataset == "alpaca_eval":
                # alpaca eval test script for now
                task_dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
            else:
                raise NotImplementedError() # not implemented yet
        elif task == "gsm8k":
            dataset = load_dataset("gsm8k", "main")
            train_size = len(dataset["train"])
            if split == "train" and test_split != "test":
                # fixed split, last 300 as our dev
                task_dataset = dataset["train"].select(range(train_size - 300))
            elif split == "train" and test_split == "test":
                # fixed split, last 300 as our dev
                task_dataset = dataset["train"]
            elif split == "validation":
                # fixed split, last 300 as our dev
                task_dataset = dataset["train"].select(range(train_size - 300, train_size))
            elif split == "test":
                task_dataset = dataset["test"]
        else:
            data_path = f"./datasets/{dataset}/{split}.json"
            task_dataset = load_dataset("json", data_files=data_path)["train"]

        # format splits
        if split == "train":
            task_dataset = task_dataset.shuffle(seed=seed)
        if max_n_example is not None:
            task_dataset = task_dataset.select(range(max_n_example))
        
        # tokenize and intervene
        for i, data_item in enumerate(tqdm(task_dataset)):
            
            if use_normalized_template: # format task-specific prompt

                # set up prompt
                if task == "commonsense":
                    base_prompt = task_prompt_template % (data_item['instruction'])
                    base_input = base_prompt + trigger_tokens + data_item["answer"] + tokenizer.eos_token
                elif task == "math": # we strip since these are model generated examples.
                    base_prompt = task_prompt_template % (data_item['instruction'])
                    base_input = base_prompt + data_item["output"] + tokenizer.eos_token
                elif task == "alpaca" or task == "instruct" or task == "ultrafeedback":
                    if 'input' not in data_item or data_item['input'] == "":
                        base_prompt = alpaca_prompt_no_input_template % (data_item['instruction'])
                    else:
                        base_prompt = task_prompt_template % (data_item['instruction'], data_item['input'])
                    base_input = base_prompt + data_item["output"] + tokenizer.eos_token
                elif task == "gsm8k": # setup is from https://github.com/yxli2123/LoftQ/
                    base_prompt = task_prompt_template % (
                        "Answer the above question. First think step by step and then answer the final number.",
                        data_item['question']
                    )
                    base_input = base_prompt + data_item["answer"].replace("####", "The final answer is: ") + \
                        tokenizer.eos_token
                else:
                    raise ValueError(f"Unrecognized task: {task}")
                
                # tokenize
                base_prompt_ids = tokenizer(
                    base_prompt, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"][0]
                base_prompt_length = len(base_prompt_ids)
                if split == "train":
                    base_input_ids = tokenizer(
                        base_input, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"][0]
                    output_ids = deepcopy(base_input_ids)
                    # mask prompt in labels
                    if not train_on_inputs:
                        output_ids[:base_prompt_length] = -100
                        
                    result["input_ids"].append(base_input_ids)
                    result["labels"].append(output_ids)
                else:
                    # print("Assuming test split for now")
                    result["input_ids"].append(base_prompt_ids)
                last_position = base_prompt_length

            else:

                # for this subroutine, we follow the setup of
                # https://github.com/AGI-Edgerunners/LLM-Adapters
                # this is for the commonsense and math reasoning tasks only
                assert (task == "commonsense" or task == "math")
                if split == "train":
                    tokenized_full_prompt, user_prompt_len = \
                        generate_and_tokenize_prompt(
                            tokenizer, data_item,
                            train_on_inputs=train_on_inputs,
                            cutoff_len=max_length
                        )
                    result["input_ids"].append(torch.tensor(tokenized_full_prompt["input_ids"]))
                    result["labels"].append(torch.tensor(tokenized_full_prompt["labels"]))
                else:
                    prompt = generate_prompt_for_generate(data_item.get('instruction'), input=None)
                    tokenized_user_prompt = tokenizer(prompt)
                    user_prompt_len = len(tokenized_user_prompt["input_ids"])
                    result["input_ids"].append(torch.tensor(tokenized_user_prompt["input_ids"]))
                last_position = user_prompt_len

            # get intervention locations
            intervention_locations = get_intervention_locations(
                share_weights, position, last_position, first_n, last_n, layers, pad_mode="first")
            result["intervention_locations"].append(intervention_locations)
            result["id"].append(i)
            
            # add a single padding token BEFORE input_ids and fix everything
            result["input_ids"][-1] = torch.cat((torch.tensor([tokenizer.pad_token_id,]), result["input_ids"][-1]))
            if split == "train":
                result["labels"][-1] = torch.cat((torch.tensor([-100]), result["labels"][-1]))
            result["intervention_locations"][-1] = (torch.IntTensor(result["intervention_locations"][-1]) + 1).tolist()
            result["attention_mask"].append((result["input_ids"][-1] != tokenizer.pad_token_id).int())

    return result, task_dataset, num_labels

def load_task(
    task: str,
    tokenizer: AutoTokenizer,
    max_n_train_example: int=None,
    max_n_eval_example: int=None,
    train_dataset: list=None,
    eval_dataset: list=None,
    test_split: str="validation",
    seed: int=42,
    eval_batch_size: int=1,
    position: str="last",
    layers: int=[1],
    train_on_inputs: bool=False,
    max_length: int=512,
    use_normalized_template: bool=False,
    share_weights: bool=False,
):
    # config
    max_length -= 1
    assert task in task_config, f"Unrecognized task: {task}"
    train_datasets = task_config[task]["train_datasets"] if train_dataset is None else [train_dataset]
    if task == "glue":
        eval_datasets = [train_dataset]
    else:
        eval_datasets = task_config[task]["eval_datasets"] if eval_dataset is None else [eval_dataset]
    task_prompt_template = task_config[task]["task_prompt_template"]
    trigger_tokens = task_config[task]["trigger_tokens"]
    
    # load data
    raw_train = defaultdict(list)
    for dataset in train_datasets:
        result, _, num_labels = reformat_by_task(
            task, dataset, task_prompt_template, trigger_tokens, tokenizer,
            max_length, position, layers, train_on_inputs, use_normalized_template, share_weights,
            split='train', max_n_example=max_n_train_example, seed=seed, test_split=test_split
        )
        for key in result:
            raw_train[key].extend(result[key])
        del _ # remove the task_dataset variable from memory
    train_dataset = Dataset.from_dict(raw_train)

    # eval
    all_eval_datasets = {}
    for dataset in eval_datasets:
        test_splits = test_split.split(";")
        all_eval_datasets[dataset] = {}
        for split in test_splits:
            result, task_dataset, num_labels = reformat_by_task(
                task, dataset, task_prompt_template, trigger_tokens, tokenizer,
                max_length, position, layers, False, use_normalized_template, share_weights,
                split=split, max_n_example=max_n_eval_example
            )
            eval_dataset = Dataset.from_dict(result)
            all_eval_datasets[dataset][split] = [eval_dataset, task_dataset]

    return train_dataset, all_eval_datasets, trigger_tokens, num_labels