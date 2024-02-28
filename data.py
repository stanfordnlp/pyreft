"""
Data loading and preprocessing.
"""

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import os
import json
import io
from tqdm import tqdm

device = "cuda"
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

no_header_prompt_template = """\
### Instruction:
%s

### Response:
"""

alpaca_prompt_template = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

alpaca_prompt_no_input_template = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""


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
    split: str='train',
    batch_size: int=None,
    max_n_eval_example: int=None,
) -> tuple:
    """Reformat the dataset based on task template and generate tokenized inputs."""

    print("loading data for dataset: ", dataset)
    task_dataset = jload(f"./datasets/{dataset}/{split}.json")
    if batch_size is None:
        all_base_input_ids = []
        all_base_positions = []
        all_output_ids = []
        for data_item in task_dataset:
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
            base_prompt_length = len(tokenizer(
                base_prompt, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"][0])
            base_input_ids = tokenizer(
                base_input, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"][0]
            output_ids = tokenizer(
                base_input, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"][0]
            output_ids[:base_prompt_length] = -100
            base_input_ids[-1] = tokenizer.eos_token_id # enforce the last token to be eos
            output_ids[-1] = tokenizer.eos_token_id # enforce the last token to be eos
        
            all_base_input_ids.append(base_input_ids)
            all_base_positions.append([base_prompt_length-1]) # intervene on the last prompt token
            all_output_ids.append(output_ids)
        return all_base_input_ids, all_base_positions, all_output_ids
    
    # batching
    if batch_size is not None:
        if max_n_eval_example is not None:
            task_dataset = task_dataset[:max_n_eval_example]
        eval_iterator = tqdm(
            chunk(task_dataset, batch_size), position=0, leave=True
        )
        all_base_input_ids = []
        all_attention_masks = []
        all_output_ids = []
        all_base_last_unit_location = []
        all_base_first_unit_location = []
        all_batch_example = []
        for batch_example in eval_iterator:
            actual_batch = []
            for _, example in enumerate(batch_example):
                prompt = task_prompt_template % (example['instruction'])
                actual_batch.append(prompt)
            all_batch_example.append(batch_example)
                    
            # tokenize in batch
            tokenized = tokenizer.batch_encode_plus(
                actual_batch, return_tensors='pt', padding=True).to(device)
            batch_length = tokenized["attention_mask"].sum(dim=-1).tolist()
            base_last_unit_location = tokenized["input_ids"].shape[-1] - 1 
            base_last_unit_location = [[base_last_unit_location]]*len(batch_example)
            base_first_unit_location = [[
                tokenized["input_ids"].shape[-1] - batch_length[i]] 
                for i in range(len(batch_example))]

            # append
            all_base_input_ids.append(tokenized["input_ids"])
            all_attention_masks.append(tokenized["attention_mask"])
            all_output_ids.append(tokenized["input_ids"])
            all_base_last_unit_location.append(base_last_unit_location)
            all_base_first_unit_location.append(base_first_unit_location)
        return all_base_input_ids, all_attention_masks, all_output_ids, all_base_first_unit_location, all_base_last_unit_location, all_batch_example


def load_task(
    task: str,
    tokenizer: AutoTokenizer,
    max_n_train_example: int=None,
    max_n_eval_example: int=None,
    train_dataset: list=None,
    eval_dataset: list=None,
    seed: int=42,
    eval_batch_size: int=1,
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
    all_base_input_ids, all_base_positions, all_output_ids = [], [], []
    for dataset in train_datasets:
        base_input_ids, base_positions, output_ids = reformat_by_task(
            task, dataset, task_prompt_template, trigger_tokens, tokenizer, max_length)
        all_base_input_ids.extend(base_input_ids)
        all_base_positions.extend(base_positions)
        all_output_ids.extend(output_ids)
    
    # make dataset obj
    raw_train = (
        all_base_input_ids,
        all_base_positions,
        all_output_ids,
    )
    train_dataset = Dataset.from_dict({
        "input_ids": raw_train[0],
        "intervention_position": raw_train[1],
        "labels": raw_train[2],
    }).shuffle(seed=seed)
    if max_n_train_example is not None:
        train_dataset = train_dataset.select(range(max_n_train_example))
    
    # eval
    tokenizer.padding_side = "left" # switch padding side for generation
    all_eval_datasets = {}
    for dataset in eval_datasets:
        base_input_ids, attention_masks, output_ids, base_first_unit_location, base_last_unit_location, batch_example = reformat_by_task(
            task, dataset, task_prompt_template, trigger_tokens, tokenizer, max_length,
            split='test', batch_size=eval_batch_size, max_n_eval_example=max_n_eval_example)
        all_eval_datasets[dataset] = Dataset.from_dict({
            "input_ids": base_input_ids,
            "attention_mask": attention_masks,
            "labels": output_ids,
            "base_first_unit_location": base_first_unit_location,
            "base_last_unit_location": base_last_unit_location,
            "example": batch_example
        })

    return train_dataset, all_eval_datasets, trigger_tokens