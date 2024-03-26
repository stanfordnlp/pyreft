import pyvene as pv
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollator,
    DataCollatorForSeq2Seq,
    AutoTokenizer
)
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, Optional, Sequence
from task_config import task_config
from tqdm import tqdm
import os
import torch
import re
import evaluate
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging
from transformers.trainer_utils import (
    EvalPrediction,
    has_length,
    denumpify_detensorize
)
from pyreft import ReftDataCollator

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.get_logger(__name__)

def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False
        
def extract_answer_number(sentence: str) -> float:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
    """
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py

    Note that it becomes ambiguous whether to extract the
    first letter or the last letter. Either way may lead
    to inaccurately assess the model performance. 

    We choose to follow the LLM-Adaptor repo, but leave this note
    for future research to explore the impact of this.
    """
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''
        
        
def extract_output(pred, trigger=''):
    if not trigger:
        return pred
    # for causallm only, use special trigger to detect new tokens.
    # if cannot find trigger --> generation is too long; default to empty generation
    start = pred.find(trigger)
    if start < 0:
        return ''
    output = pred[start+len(trigger):].lstrip() # left strip any whitespaces
    return output


def make_data_collator(tokenizer, model) -> ReftDataCollator:
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )
    return ReftDataCollator(data_collator=data_collator_fn)


def make_dataloader(dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)


def compute_metrics(
    task: str,
    dataset_name: str,
    intervenable: pv.IntervenableModel,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    data_items: list,
    trigger_tokens: str,
    run_name: str,
    batch_size: int=4,
    data_collator=None,
    split=None,
    greedy_decoding=False,
    temperature=None, 
    top_p=None, 
    top_k=None
):
    # switch the tokenizer mode first for generation tasks
    if task != "glue":
        tokenizer.padding_side = "left" # switch padding side for collator
        num_beams = 4 if task in ["commonsense", "math"] and not greedy_decoding else 1

    data_collator = data_collator if data_collator is not None else \
        make_data_collator(tokenizer, intervenable.model)
    eval_dataloader = make_dataloader(eval_dataset, batch_size, data_collator, shuffle=False)
    correct_count = 0
    total_count = 0
    generations = []
    eval_iterator = tqdm(eval_dataloader, position=0, leave=True)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for step, inputs in enumerate(eval_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            
            # [layers, batch_size, positions]
            intervention_locations = inputs["intervention_locations"].permute(1, 0, 2)
    
            if task == "glue":
    
                _, cf_outputs = intervenable(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    unit_locations={"sources->base": (None, intervention_locations.tolist())})
            
                # lm loss on counterfactual labels
                if dataset_name != "stsb":
                    preds = cf_outputs.logits.argmax(dim=-1)
                else:
                    preds = cf_outputs.logits.squeeze(dim=1)
                
                labels = inputs["labels"]
                all_preds += preds.tolist()
                all_labels += labels.tolist()
            
            else:
                # get left padding count, [batch_size], and add to locations
                left_padding = (inputs["input_ids"] == tokenizer.bos_token_id).nonzero(as_tuple=True)[1]
                left_padding = left_padding.reshape(1, -1, 1).to(device) # [1, batch_size, 1]
                intervention_locations += left_padding
                intervention_locations -= 1 # offset for the sink padding

                # repeat each batch by num_beams times in intervention locations
                # -> [layers, batch_size * num_beams, positions]
                intervention_locations = intervention_locations.repeat_interleave(num_beams, dim=1).tolist()
                
                # set generation args depending on task
                generation_args = {
                    "base": {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    "unit_locations": {"sources->base": (None, intervention_locations)},
                    "intervene_on_prompt": True,
                    "eos_token_id": tokenizer.eos_token_id,
                    "early_stopping": True,
                }
                if "generation_args" in task_config[task]:
                    generation_args.update(task_config[task]["generation_args"][greedy_decoding])
                
                # override generation args if necessary
                if temperature is not None:
                    generation_args["temperature"] = temperature
                if top_p is not None:
                    generation_args["top_p"] = top_p
                if top_k is not None:
                    generation_args["top_k"] = top_k

                # generate with intervention on prompt
                _, steered_response = intervenable.generate(**generation_args)
        
                # detokenize in batch
                actual_preds = tokenizer.batch_decode(steered_response, skip_special_tokens=True)
                
                for id, pred in zip(inputs["id"].tolist(), actual_preds):
                    example = data_items[id]
                    try:
                        raw_generation = extract_output(pred, trigger_tokens)
                    except:
                        print("get not split based on trigger tokens: ", raw_generation)
                        raw_generation = "WRONG"
        
                    # check if generation is correct
                    if task == "commonsense":
                        answer = example["answer"]
                        generation = raw_generation[:]
                        if generation.strip() == answer.strip():
                            correct_count += 1
                    elif task == "math":
                        answer = example["answer"]
                        answer = answer.strip()
                        if not is_float(answer): # assuming this is from AQuA:
                            generation = extract_answer_letter(raw_generation)
                            if generation.strip() == answer.strip():
                                correct_count += 1
                        else:
                            generation = extract_answer_number(raw_generation)
                            if abs(float(answer) - generation) <= 0.001:
                                correct_count += 1
                    elif task == "gsm8k":
                        answer = example["answer"].split("####")[-1].strip()
                        generation = extract_answer_number(raw_generation)
                        if abs(float(extract_answer_number(answer)) - generation) <= 0.001:
                            correct_count += 1
                            
                    # log
                    total_count += 1
                    if task not in ["alpaca", "instruct", "ultrafeedback"]:
                        metric_str = round(correct_count / total_count, 3)
                        eval_iterator.set_postfix({"em": metric_str})
                        instruction = example["question"] if task == "gsm8k" else example["instruction"]
                        generations += [{
                            "instruction": instruction,
                            "raw_generation": raw_generation,
                            "generation": generation,
                            "answer": answer
                        }]
                    else:
                        generations += [{
                            "instruction": example["instruction"],
                            "output": raw_generation,
                            "dataset": dataset_name,
                            "generator": run_name
                        }]
    # compute metrics
    if task == "glue":
        metric = evaluate.load("glue", dataset_name)
        def compute_metrics_glue(preds, labels):
            result = metric.compute(predictions=preds, references=labels)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        
        report = compute_metrics_glue(all_labels, all_preds)
        print_str = "task metrics "
        if split:
            report = {split + "_" + k: v for k, v in report.items()}
            print_str += "[" + split + "]"
        print_str += ":"
        print(report)
        return [], report
    if task in ["alpaca", "instruct", "ultrafeedback"]:
        return generations, {}
    else:
        return generations, {f"eval/{dataset_name}": correct_count / total_count}