import sys
sys.path.append("../pyvene/")

import torch
import argparse
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
import wandb
import datetime
import json

import pyvene as pv
from data import load_task
from trainer import ReftTrainer, TrainingArguments, compute_metrics
from interventions import *

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    pv.set_seed(seed)
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
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    assert task in {"commonsense", "math", "alpaca", "instruct"}
    
    # store/log run details
    print(
        f"task: {task}, model: {model}, intervention_type: {intervention_type}, "
        f"layers: {layers}, rank: {rank}, "
        f"position: {position}, epoch: {epochs}"
    )
    model_str = model.split("/")[-1]
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    run_name = f"{model_str}.{task}.{now}"
    
    # load model
    config, _, llama = pv.create_llama(model, dtype=dtype)
    _ = llama.to(device)
    _ = llama.eval()
    
    # post-processing the inputs
    if intervention_type == "LearnedSourceLowRankRotatedSpaceIntervention":
        intervention_type = LearnedSourceLowRankRotatedSpaceIntervention
    elif intervention_type == "ConditionedSourceLowRankRotatedSpaceIntervention":
        intervention_type = ConditionedSourceLowRankRotatedSpaceIntervention
    elif intervention_type == "ConditionedSourceLowRankIntervention":
        intervention_type = ConditionedSourceLowRankIntervention
    
    # which layers to intervene on
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
        task, tokenizer, max_n_train_example, max_n_eval_example, train_dataset,
        eval_dataset, seed, eval_batch_size, position, layers)
    print("loaded", len(train_dataset), len(eval_datasets))
    
    # prep train
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=llama,
        label_pad_token_id=-100,
        padding="longest"
    )

    # intervention config
    config = pv.IntervenableConfig([{
        "layer": l,
        "component": "block_output",
        "low_rank_dimension": rank} for l in layers],
        intervention_type
    )
    intervenable = pv.IntervenableModel(config, llama)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    n_params = intervenable.count_parameters()
    
    # start wandb logging
    run = wandb.init(
        project=f"Steer_LM_{task}", 
        entity="reft",
        name=run_name,
    )
    run.summary.update(vars(args))
    wandb.log({"train/n_params": n_params})
    
    # training args
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{run_name}",
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        save_total_limit=1,
        logging_steps=1,
        learning_rate=lr,
        warmup_steps=20 if task == "math" else 100,
        weight_decay=0.01,
        report_to="wandb" if is_wandb else None,
        use_cpu=False if device == "cuda" else True,
    )

    # make trainer
    trainer = ReftTrainer(
        model=intervenable,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        compute_metrics=None,
        tokenizer=tokenizer,
    )
    trainer.train()
    
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

    # do eval
    eval_results = {}
    for dataset_name in eval_datasets:
        # split evalset into chunks
        eval_dataset, data_items = eval_datasets[dataset_name]
        generations, stats = compute_metrics(
            task, dataset_name, intervenable, tokenizer, eval_dataset, data_items,
            trigger_tokens, eval_batch_size
        )

        # log
        eval_results.update(stats)
        wandb.log(stats)
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