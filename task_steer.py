import sys
sys.path.append("../pyvene/")

import torch
import random, copy, argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F
import seaborn as sns
from tqdm import tqdm, trange
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import wandb

from data import _make_r_io_base, _make_w_io_base, jload, jdump, create_directory, load_task, chunk

from pyvene import (
    IntervenableModel,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
    ConstantSourceIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from pyvene import create_llama
from pyvene import set_seed, count_parameters
from pyvene.models.layers import LowRankRotateLayer

import io
import json
import os
import re

def extract_answer_number(sentence: str) -> float:
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
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[-1]
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

class LearnedSourceLowRankRotatedSpaceIntervention(
    ConstantSourceIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.dropout = torch.nn.Dropout(0.05)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

class ConditionedSourceLowRankRotatedSpaceIntervention(
    ConstantSourceIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(torch.bfloat16)
        self.act_fn = ACT2FN["silu"]
        self.dropout = torch.nn.Dropout(0.05)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))
    
class ConditionedSourceLowRankIntervention(
    ConstantSourceIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=False).to(torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(torch.bfloat16)
        self.act_fn = ACT2FN["silu"]
        self.dropout = torch.nn.Dropout(0.05)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        proj_base = self.proj_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - proj_base), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))
    
def main():
    """
    Generic Representation Finetuning.
    """

    parser = argparse.ArgumentParser(description="A simple script that takes different arguments.")
    
    parser.add_argument('-task', '--task', type=str, default=None)
    parser.add_argument('-train_dataset', '--train_dataset', type=str, default=None)
    parser.add_argument('-eval_dataset', '--eval_dataset', type=str, default=None)
    parser.add_argument('-model', '--model', type=str, help='yahma/llama-7b-hf', default='yahma/llama-7b-hf')
    parser.add_argument('-seed', '--seed', type=int, help='42', default=42)
    parser.add_argument('-l', '--layers', type=str, help='2;10;18;26', default='2;10;18;26')
    parser.add_argument('-r', '--rank', type=int, help=8, default=8)
    parser.add_argument('-p', '--position', type=str, help='last', default='last')
    parser.add_argument('-e', '--epochs', type=int, help='1', default=1)
    parser.add_argument('-is_wandb', '--is_wandb', type=bool, default=False)
    parser.add_argument('-save_model', '--save_model', type=bool, default=False)
    parser.add_argument('-max_n_train_example', '--max_n_train_example', type=int, default=None)
    parser.add_argument('-max_n_eval_example', '--max_n_eval_example', type=int, default=None)
    parser.add_argument(
        '-type', '--intervention_type', type=str, 
        help='LearnedSourceLowRankRotatedSpaceIntervention', default="LearnedSourceLowRankRotatedSpaceIntervention")
    parser.add_argument('-gradient_accumulation_steps', '--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=4)
    parser.add_argument('-eval_batch_size', '--eval_batch_size', type=int, default=4)
    parser.add_argument('-output_dir', '--output_dir', type=str, default="./official_results")
    parser.add_argument('-lr', '--lr', type=float, default=5e-3)
    
    args = parser.parse_args()

    model = args.model
    layers = args.layers
    rank = args.rank
    position = args.position
    epochs = args.epochs
    seed = args.seed
    intervention_type = args.intervention_type
    set_seed(seed)
    max_n_train_example = args.max_n_train_example
    max_n_eval_example = args.max_n_eval_example
    is_wandb = args.is_wandb
    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size = args.batch_size
    output_dir = args.output_dir
    task = args.task
    lr = args.lr
    train_dataset = args.train_dataset
    eval_dataset = args.eval_dataset
    save_model = args.save_model
    eval_batch_size = args.eval_batch_size
    
    assert task in {"commonsense", "math", "alpaca", "instruct"}
    
    print(
        f"task: {task}, model: {model}, intervention_type: {intervention_type}"
        f"layers: {layers}, rank: {rank}, "
        f"position: {position}, epoch: {epochs}"
    )
    model_str = model.split("/")[-1]
    run_name = f"{model_str}.{task}.intervention_type={intervention_type}.layers={layers}."\
               f"rank={rank}.position={position}.epoch={epochs}.lr={lr}"
    
    config, _, llama = create_llama(model)
    _ = llama.to(device)
    _ = llama.eval()
    
    # post-processing the inputs
    if intervention_type == "LearnedSourceLowRankRotatedSpaceIntervention":
        intervention_type = LearnedSourceLowRankRotatedSpaceIntervention
    elif intervention_type == "ConditionedSourceLowRankRotatedSpaceIntervention":
        intervention_type = ConditionedSourceLowRankRotatedSpaceIntervention
    elif intervention_type == "ConditionedSourceLowRankIntervention":
        intervention_type = ConditionedSourceLowRankIntervention
        
    user_give_all_layers = False
    if layers != "all":
        if "+" in layers:
            parsed_layers = []
            for l in layers.split("+"):
                for ll in l.split(";"):
                    parsed_layers += [int(ll)]
            user_give_all_layers = True
            layers = parsed_layers
        else:
            layers = [int(l) for l in layers.split(";")]
    else:
        layers = [l for l in range(config.num_hidden_layers)]
    assert position in {"last", "first+last"}
    if position in {"first+last"}:
        if user_give_all_layers:
            pass
        else:
            layers += layers
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.padding_side = "right" # we will use right padding for training with teacher-forcing
    tokenizer.pad_token = tokenizer.unk_token

    # load dataset splits
    train_dataset, eval_datasets, trigger_tokens = load_task(
        task, tokenizer, max_n_train_example, max_n_eval_example, train_dataset, eval_dataset, seed, eval_batch_size)
    print(eval_datasets)
    print("loaded", len(train_dataset), len(eval_datasets))
    
    # prep train
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=llama,
        label_pad_token_id=-100,
        padding="longest"
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    
    initial_lr = lr
    total_step = 0
    t_total = int(len(train_dataloader) * epochs) // gradient_accumulation_steps

    # intervention config
    config = IntervenableConfig([{
        "layer": l,
        "component": "block_output",
        "low_rank_dimension": rank} for l in layers],
        intervention_type
    )
    intervenable = IntervenableModel(config, llama)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    n_params = intervenable.count_parameters()
    
    # optimiser + scheduler
    if intervention_type == "ConditionedSourceLowRankIntervention":
        optimizer = torch.optim.AdamW(
            intervenable.get_trainable_parameters(), lr=initial_lr
        )
    else:
        optimizer = torch.optim.Adam(
            intervenable.get_trainable_parameters(), lr=initial_lr
        )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=20 if task == "math" else 100, 
        num_training_steps=t_total
    )

    intervenable.model.train()  # train enables drop-off but no grads
    print("llama trainable parameters: ", count_parameters(intervenable.model))
    print("intervention trainable parameters: ", n_params)
    
    if is_wandb:
        run = wandb.init(
            project=f"Steer_LM_{task}", 
            entity="wuzhengx",
            name=run_name,
        )
        wandb.log({"train/n_params": n_params})
    
    # MAIN TRAIN LOOP
    train_iterator = trange(0, int(epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
        )

        # each step for the epoch
        for step, inputs in enumerate(epoch_iterator):

            # move to device
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            
            # intervention position
            b_s = inputs["input_ids"].shape[0]
            base_unit_location = inputs["intervention_position"].tolist()
            if position in {"first+last"}:
                base_first_token = torch.zeros_like(inputs["intervention_position"]).tolist()
                intervention_locations = [base_first_token]*(len(layers)//2)+[base_unit_location]*(len(layers)//2)
            else:
                intervention_locations = [base_unit_location]*len(layers)

            # perform intervention
            _, cf_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                unit_locations={"sources->base": (None, intervention_locations)})

            # lm loss on counterfactual labels
            lm_logits = cf_outputs.logits
            labels = inputs["labels"]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_str = round(loss.item(), 2)
            epoch_iterator.set_postfix({"loss": loss_str})

            # backprop
            if is_wandb:
                wandb.log({"train/loss": loss_str})
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if total_step % gradient_accumulation_steps == 0:
                if not (gradient_accumulation_steps > 1 and total_step == 0):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            total_step += 1
    
    # save model
    if save_model:
        intervenable.save(save_directory=f"{output_dir}/{run_name}")
    create_directory(f"{output_dir}/{run_name}")
    
    # dump config
    args_dict = vars(args)
    args_dict["n_params"] = n_params
    json_file_name = f"{output_dir}/{run_name}/args.json"
    with open(json_file_name, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
    
    # ensure everything is in eval mode
    intervenable.model.eval()
    for k,v in intervenable.interventions.items():
        _ = v[0].eval()
    tokenizer.padding_side = "left" # switch padding side for generation

    # do eval
    eval_results = {}
    for dataset_name in eval_datasets:
        correct_count = 0
        total_count = 0
        generations = []

        # split evalset into chunks
        eval_dataset = eval_datasets[dataset_name]
        eval_iterator = tqdm(
            eval_dataset, position=0, leave=True
        )

        with torch.no_grad():
            for batch in eval_iterator:
                
                # intervention locations
                last_shape = torch.tensor(batch["input_ids"]).shape[-1]
                if position in {"first+last"}:
                    base_unit_location = last_shape - 1 
                    base_unit_location = {"sources->base": (
                        None, [batch["base_first_unit_location"]]*(len(layers)//2) + 
                        [batch["base_last_unit_location"]]*(len(layers)//2))}
                else:
                    base_unit_location = last_shape - 1 
                    base_unit_location = {"sources->base": (None, [batch["base_last_unit_location"]]*(len(layers)))}

                tokenized = {
                    "input_ids": torch.tensor(batch["input_ids"]).to(device),
                    "attention_mask": torch.tensor(batch["attention_mask"]).to(device),
                    "labels": torch.tensor(batch["labels"]).to(device),
                }
                
                # collect generations
                if task == "commonsense":
                    _, steered_response = intervenable.generate(
                        tokenized,
                        unit_locations=base_unit_location,
                        intervene_on_prompt=True,
                        max_new_tokens=10, do_sample=False, 
                        eos_token_id=tokenizer.eos_token_id, early_stopping=True
                    )
                elif task == "math":
                    _, steered_response = intervenable.generate(
                        tokenized, 
                        unit_locations=base_unit_location,
                        intervene_on_prompt=True,
                        # since we are generating the CoT, we follow previous works setting.
                        max_new_tokens=256,
                        temperature=0.1,
                        top_p=0.75,
                        top_k=40,
                        num_beams=1,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id, 
                        early_stopping=True,
                    )
                elif task in ["alpaca", "instruct"]:
                    _, steered_response = intervenable.generate(
                        tokenized, 
                        unit_locations=base_unit_location,
                        intervene_on_prompt=True,
                        # much longer generation
                        max_new_tokens=2048,
                        temperature=0.7,
                        top_p=1.0,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id, 
                        early_stopping=True
                    )
                batch_example = batch["example"]
                    
                # detokenize in batch
                actual_preds = tokenizer.batch_decode(steered_response, skip_special_tokens=True)
                for pred, example in zip(actual_preds, batch_example):

                    try:
                        raw_generation = extract_output(pred, trigger_tokens)
                    except:
                        print("get not split based on trigger tokens: ", raw_generation)
                        raw_generation = "WRONG"

                    # check if generation is correct
                    answer = example["answer"]
                    if task == "commonsense":
                        generation = raw_generation[:]
                        if generation.strip() == answer.strip():
                            correct_count += 1
                    elif task == "math":
                        answer = answer.strip()
                        if dataset_name == "AQuA":
                            generation = extract_answer_letter(raw_generation)
                            if generation.strip() == answer.strip():
                                correct_count += 1
                        else:
                            generation = extract_answer_number(raw_generation)
                            if generation == float(answer):
                                correct_count += 1
                    
                    # log
                    total_count += 1
                    if task not in ["alpaca", "instruct"]:
                        metric_str = round(correct_count / total_count, 3)
                        eval_iterator.set_postfix({"em": metric_str})
                        generations += [{
                            "instruction": example["instruction"],
                            "raw_generation": pred,
                            "generation": generation,
                            "answer": answer,
                            "generator": run_name,
                        }]
        
        # log stats
        if task not in ["alpaca", "instruct"]:
            metric_str = round(correct_count / total_count, 3)
            if is_wandb:
                wandb.log({f"eval/{dataset_name}": metric_str})
            eval_results[dataset_name] = metric_str
        result_json_file_name = f"{output_dir}/{run_name}/{dataset_name}_outputs.json"
        with open(result_json_file_name, 'w') as json_file:
            json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{output_dir}/{run_name}/eval_results.json"
    eval_results["n_params"] = n_params
    with open(result_json_file_name, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)
        
if __name__ == "__main__":
    main()