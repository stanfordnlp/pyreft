import sys
sys.path.append("../../pyvene/")

import torch
import argparse
from tqdm import tqdm, trange
from dataclasses import dataclass
from typing import Dict, Optional, Sequence
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    DataCollator,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed
)
from transformers.trainer_utils import EvalPrediction
import wandb
import evaluate
import datetime
import json
import math
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import pyvene as pv
from data import load_task
from trainer import (
    ReftDataCollator,
    ReftTrainer,
    ReftTrainerForSequenceClassification,
    TrainingArguments,
    compute_metrics,
)
from interventions import *

device = "cuda" if torch.cuda.is_available() else "cpu"
classification_tasks = {"glue"}
residual_stream_component_mapping = {
    "robertaformaskedlm": "roberta.encoder.layer[%s].output"
}
dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8": "float8",
}


def reftune(
    act_fn: str,
    add_bias: bool,
    model: str,
    layers: str,
    rank: int,
    position: str,
    epochs: int,
    seed: int,
    intervention_type: str,
    max_n_train_example: int,
    max_n_eval_example: int,
    is_wandb: bool,
    wandb_name: str,
    gradient_accumulation_steps: int,
    batch_size: int,
    output_dir: str,
    task: str,
    lr: float,
    schedule: str,
    train_dataset: str,
    eval_dataset: str,
    save_model: bool,
    eval_batch_size: int,
    warmup_ratio: float,
    weight_decay: float,
    dropout: float,
    test_split: str,
    train_on_inputs: bool,
    max_length: int,
    use_normalized_template: bool,
    allow_cls_grad: bool,
    metric_for_best_model: str,
    dtype: str,
    logging_steps: int,
    wandb_dir: str,
    wandb_proj: str,
    share_weights: bool,
    args,
):
    """
    Generic Representation Finetuning.
    """

    assert task in {
        "commonsense", "math", "alpaca", "instruct", "ultrafeedback", "glue", "gsm8k"
    }
    dtype = dtype_mapping[dtype]
    
    # store/log run details
    print(
        f"task: {task}, model: {model}, intervention_type: {intervention_type}, "
        f"layers: {layers}, rank: {rank}, "
        f"position: {position}, epoch: {epochs}, train_on_inputs: {train_on_inputs}, "
        f"max_length: {max_length}, allow_cls_grad: {allow_cls_grad}"
    )

    # everything is guarded by a single seed
    set_seed(seed)

    model_name = model
    model_str = model.split("/")[-1]
    train_dataset_str = train_dataset
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    if train_dataset is not None:
        run_name = f"{model_str}.{task}.{train_dataset_str}.{test_split}.{now}"
    else:
        run_name = f"{model_str}.{task}.{now}"

    # which layers to intervene on
    if layers != "all":
        if "+" in layers:
            raise ValueError("We disallow + now. Layers will be shared cross positions.")
        else:
            layers = [int(l) for l in layers.split(";")]
    else:
        temp_config = AutoConfig.from_pretrained(model)
        layers = [l for l in range(temp_config.num_hidden_layers)]

    # position str takes the following formats:
    # f1 -> first token; f2 -> first two tokens.
    # f1+l1 -> first and last tokens; f2+l2 -> first and last two tokens.
    # fn or ln shares the same intervention.
    if "+" in position and not share_weights:
        layers += layers

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right" # we will use right padding for training with teacher-forcing
    tokenizer.pad_token = tokenizer.unk_token
    
    # load dataset splits
    train_dataset, eval_datasets, trigger_tokens, num_labels = load_task(
        task, tokenizer, max_n_train_example, max_n_eval_example, train_dataset,
        eval_dataset, test_split, seed, eval_batch_size, position, layers, train_on_inputs,
        max_length, use_normalized_template, share_weights
    )
    print("loaded", train_dataset, eval_datasets, num_labels)
    if task == "glue":
        # we repartition the eval_datatsets into [1] 50% validation + [2] 50% test
        # we select the best model on [1] during training
        # we test the selected model on [2] to ensure fairness
        to_split_eval_datasets = eval_datasets[train_dataset_str][test_split][0]
        if len(to_split_eval_datasets) > 5000:
            in_train_n_eval_sample = 1000
        else:
            in_train_n_eval_sample = len(to_split_eval_datasets) // 2

        new_splits = to_split_eval_datasets.train_test_split(test_size=in_train_n_eval_sample)
        in_test_eval_datasets, in_train_eval_datasets = new_splits["train"], new_splits["test"]
        eval_datasets[train_dataset_str][test_split][0] = in_test_eval_datasets
        print("GLUE validation split (in training): ", in_train_eval_datasets)
        print("GLUE validation split (testing): ", eval_datasets[train_dataset_str][test_split][0])

        is_regression = train_dataset_str == "stsb"
        metric = evaluate.load("glue", train_dataset_str)
        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def in_training_compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

    # load model based on task type.
    if task in classification_tasks:
        config = AutoConfig.from_pretrained(
            model, num_labels=num_labels,
            finetuning_task=train_dataset_str,
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device
        )
        # full precision loading since usually for small models
        model = AutoModelForSequenceClassification.from_pretrained(
            model,
            config=config, # just providing the label
            torch_dtype=dtype if dtype != "float8" else None,
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype if dtype != "float8" else None,  # save memory
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device
        )
        config = model.config
    dtype = torch.bfloat16 if dtype == "float8" else dtype

    if intervention_type == "ConditionedSourceLowRankRotatedSpaceIntervention":
        intervention_type = ConditionedSourceLowRankRotatedSpaceIntervention
    elif intervention_type == "ConditionedSourceLowRankIntervention":
        intervention_type = ConditionedSourceLowRankIntervention
        
    # select collator based on the type
    if task in classification_tasks:
        data_collator_fn = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest"
        )
    else:
        data_collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest"
        )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    
    # intervention config based on model type
    model_arch = model.config.architectures[0].lower()
    if model_arch in residual_stream_component_mapping:
        intervention_list = [{
            "component": residual_stream_component_mapping[model_arch] % l,
            "intervention": intervention_type(
                embed_dim=config.hidden_size, low_rank_dimension=rank,
                dropout=dropout, dtype=dtype, act_fn=act_fn, device=device,
                add_bias=add_bias, keep_last_dim=True
            )
        } for l in layers]
        config = pv.IntervenableConfig(intervention_list)
    else:
        config = pv.IntervenableConfig([{
            "layer": l, "component": "block_output",
            "low_rank_dimension": rank,
            "intervention": intervention_type(
                embed_dim=config.hidden_size, low_rank_dimension=rank,
                dropout=dropout, dtype=dtype, act_fn=act_fn, device=device,
                add_bias=add_bias, keep_last_dim=True
            )
        } for l in layers])

    reft_model = pv.IntervenableModel(config, model)
    reft_model.set_device(device)
    reft_model.disable_model_gradients()

    # for GLUE tasks, we enable gradients on the classifier head.
    # the parameter will be counted as well.
    if task == "glue" and allow_cls_grad:
        for param in reft_model.model.classifier.parameters():
            # reft_model with HF trainer will automatically pick up these params to optimize
            param.requires_grad = True

    # train enables dropout but no grads.
    # this line might not be necessary since HF trainer enables this by default.
    reft_model.model.train()
    n_params = reft_model.count_parameters(include_model=False)

    # start wandb logging
    if is_wandb:
        run = wandb.init(
            project=f"{wandb_proj}_{task}", 
            entity=wandb_name,
            name=run_name,
            dir=wandb_dir,
        )
        run.summary.update(vars(args))
        wandb.log(
            {"train/n_params": n_params})

    # # training args
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{run_name}",
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch" if task == "glue" else "no",
        save_strategy="epoch" if task == "glue" else "no",
        metric_for_best_model=metric_for_best_model if task == "glue" else None,
        load_best_model_at_end=True if task == "glue" else False,
        logging_strategy="steps",
        save_total_limit=1, # for GLUE, it will save 2 at max.
        logging_steps=logging_steps,
        lr_scheduler_type=schedule,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        weight_decay=weight_decay,
        report_to="wandb" if is_wandb else "none",
        use_cpu=False if device == "cuda" else True,
        seed=seed
    )

    # make trainer
    trainer_class = ReftTrainerForSequenceClassification if task in classification_tasks else ReftTrainer
    trainer = trainer_class(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=in_train_eval_datasets if task == "glue" else None,
        compute_metrics=in_training_compute_metrics if task == "glue" else None,
    )
    trainer.train()

    # dump config
    args_dict = vars(args)
    args_dict["n_params"] = n_params
    json_file_name = f"{output_dir}/{run_name}/args.json"
    with open(json_file_name, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    # ensure everything is in eval mode
    reft_model.model.eval()
    for k,v in reft_model.interventions.items():
        _ = v[0].eval()

    print({"n_params": n_params})
    # do eval
    eval_results = {}
    for dataset_name in eval_datasets:
        # split evalset into chunks
        for split, (eval_dataset, data_items) in eval_datasets[dataset_name].items():
            
            generations, stats = compute_metrics(
                task, dataset_name, reft_model, tokenizer, eval_dataset, data_items,
                trigger_tokens, run_name, eval_batch_size, 
                data_collator if task in classification_tasks else None,
                split,
            )

            # log
            eval_results.update(stats)
            if is_wandb:
                wandb.log(stats)
            generations = stats if generations is None else generations
            result_json_file_name = f"{output_dir}/{run_name}/{dataset_name}_{split}_outputs.json"
            with open(result_json_file_name, 'w') as json_file:
                json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{output_dir}/{run_name}/eval_results.json"
    eval_results["n_params"] = n_params
    with open(result_json_file_name, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)
    
    # save model
    if save_model:
        reft_model.save(f"{output_dir}/{run_name}")
        

def main():
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
    parser.add_argument('-is_wandb', '--is_wandb', action='store_true')
    parser.add_argument('-wandb_name', '--wandb_name', type=str, default="reft")
    parser.add_argument('-save_model', '--save_model', action='store_true')
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
    parser.add_argument('-schedule', '--schedule', type=str, default='linear')
    parser.add_argument('-wu', '--warmup_ratio', type=float, default=0.00)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.00)
    parser.add_argument('-act_fn', '--act_fn', type=str, default=None)
    parser.add_argument('-add_bias', '--add_bias', action='store_true')
    parser.add_argument('-test_split', '--test_split', type=str, default="validation")
    parser.add_argument('-train_on_inputs', '--train_on_inputs', action='store_true')
    parser.add_argument('-max_length', '--max_length', type=int, help=512, default=512)
    parser.add_argument('-nt', '--use_normalized_template', action='store_true')
    parser.add_argument('-allow_cls_grad', '--allow_cls_grad', action='store_true')
    parser.add_argument('-metric_for_best_model', '--metric_for_best_model', type=str, default="accuracy")
    parser.add_argument('-dtype', '--dtype', type=str, default="bfloat16" if device == "cuda" else "float32")
    parser.add_argument('-logging_steps', '--logging_steps', type=int, help=1, default=1)
    parser.add_argument('-wandb_dir', '--wandb_dir', type=str, default='wandb')
    parser.add_argument('-wandb_proj', '--wandb_proj', type=str, default='MyReFT')
    parser.add_argument('-sw', '--share_weights', action='store_true')
    
    args = parser.parse_args()

    reftune(**vars(args), args=args)


if __name__ == "__main__":
    main()
