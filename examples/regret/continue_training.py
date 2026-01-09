#!/usr/bin/env python3
"""
Continue training from best checkpoints for 10x longer.

For each (rank, position) setting, this script:
1. Finds the best-performing LR from wandb
2. Loads the saved checkpoint
3. Continues training for 10x as long
4. Logs to wandb as a new run with proper step offset

Usage:
    # Analyze what would be run (dry run)
    python continue_training.py --dry-run
    
    # Run continuation for all best configs
    python continue_training.py
    
    # Run for specific rank/position
    python continue_training.py --rank 4 --position f1+l1
"""

import os
import argparse
import datetime
import json
import glob
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset

from pyreft import (
    ReftModel,
    ReftConfig,
    LoreftIntervention,
    ReftDataCollator,
    ReftGenerationDataset,
)
from trainer import ReftTrainerForCausalLMWithEval, FullFinetuneTrainer

# Check for peft availability
try:
    import peft
    is_peft_available = True
except ModuleNotFoundError:
    is_peft_available = False

device = "cuda" if torch.cuda.is_available() else "cpu"

dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

IGNORE_INDEX = -100


def fetch_best_runs(project: str, entity: str = None):
    """Fetch best LR for each (rank, position) from wandb."""
    import wandb
    import pandas as pd
    
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    runs = api.runs(project_path)
    
    records = []
    for run in runs:
        if run.state != "finished":
            continue
        
        config = run.config
        summary = run.summary._json_dict
        
        # Skip LoRA and full finetune for now
        use_lora = config.get("use_lora", False)
        disable_reft = config.get("disable_reft", False)
        full_finetune = config.get("full_finetune", False)
        
        if full_finetune or (use_lora and disable_reft):
            continue
        
        # Get eval NLL
        eval_nll = summary.get("eval/nll") or summary.get("eval_nll")
        if eval_nll is None:
            continue
        
        record = {
            "run_name": run.name,
            "run_id": run.id,
            "rank": config.get("rank"),
            "lr": config.get("lr"),
            "position": config.get("position"),
            "share_weights": config.get("share_weights", False),
            "eval_nll": eval_nll,
            "output_dir": config.get("output_dir", "./outputs"),
            # Store full config for continuation
            "config": config,
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    if df.empty:
        return {}
    
    # Find best LR for each (rank, position)
    best_runs = {}
    for (rank, position), group in df.groupby(["rank", "position"]):
        best_idx = group["eval_nll"].idxmin()
        best_row = group.loc[best_idx]
        best_runs[(rank, position)] = best_row.to_dict()
    
    return best_runs


def load_original_args(checkpoint_dir: str):
    """Load training args from checkpoint directory."""
    args_file = os.path.join(checkpoint_dir, "training_args.json")
    if os.path.exists(args_file):
        with open(args_file, "r") as f:
            return json.load(f)
    return None


def find_checkpoint_dirs(run_dir: str):
    """Find checkpoint directories for a run.
    
    Returns:
        (reft_dir, trainer_checkpoint_dir)
        - reft_dir: where ReFT intervention weights are saved (usually run_dir itself)
        - trainer_checkpoint_dir: where HF Trainer state (optimizer, etc) is saved
    """
    if not os.path.exists(run_dir):
        return None, None
    
    # ReFT interventions are saved directly in run_dir
    has_reft_files = (
        os.path.exists(os.path.join(run_dir, "config.json")) or
        len(glob.glob(os.path.join(run_dir, "intkey_*.bin"))) > 0
    )
    reft_dir = run_dir if has_reft_files else None
    
    # Look for HF Trainer checkpoints (checkpoint-XXXX)
    checkpoints = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")))
    trainer_dir = checkpoints[-1] if checkpoints else None
    
    return reft_dir, trainer_dir


def preprocess_tulu3_to_prompt_completion(dataset, tokenizer):
    """Preprocess Tulu-3 dataset to add 'prompt' and 'completion' fields."""
    def convert_example(example):
        messages = example.get("messages", [])
        if not messages:
            return {"prompt": "", "completion": ""}
        
        prompt_messages = []
        completion = ""
        
        for msg in messages:
            if msg["role"] == "assistant":
                completion = msg["content"]
                break
            prompt_messages.append(msg)
        
        prompt = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return {"prompt": prompt, "completion": completion}
    
    return dataset.map(convert_example, desc="Preprocessing")


def continue_training(
    original_run: dict,
    output_base_dir: str = "./outputs_10x",
    wandb_project: str = "loreft-regret-10x",
    epochs_multiplier: int = 10,
    dry_run: bool = False,
):
    """Continue training from a checkpoint for 10x longer."""
    config = original_run["config"]
    run_name = original_run["run_name"]
    
    # Find checkpoint directories
    # output_dir in wandb config already includes the run name
    run_dir = config.get("output_dir", "./outputs")
    reft_dir, trainer_checkpoint_dir = find_checkpoint_dirs(run_dir)
    
    if reft_dir is None:
        print(f"  ERROR: No ReFT checkpoint found in {run_dir}")
        return None
    
    # Load original training args
    original_args = load_original_args(run_dir)
    if original_args is None:
        print(f"  WARNING: No training_args.json found, using wandb config")
        original_args = config
    
    # Build new run name
    new_run_name = f"{run_name}___10x"
    new_output_dir = os.path.join(output_base_dir, new_run_name)
    
    print(f"\n{'='*60}")
    print(f"Continuing: {run_name}")
    print(f"  ReFT checkpoint: {reft_dir}")
    print(f"  Trainer checkpoint: {trainer_checkpoint_dir}")
    print(f"  Original epochs: {original_args.get('epochs', 1)}")
    print(f"  New epochs: {original_args.get('epochs', 1) * epochs_multiplier}")
    print(f"  Output: {new_output_dir}")
    print(f"{'='*60}")
    
    if dry_run:
        print("  [DRY RUN] Would continue training")
        return None
    
    # Set seed
    seed = original_args.get("seed", 42)
    set_seed(seed)
    
    # Parse layers
    layers_str = original_args.get("layers", "all")
    position = original_args.get("position", "f1+l1")
    share_weights = original_args.get("share_weights", False)
    rank = original_args.get("rank", 4)
    model_name = original_args.get("model_name_or_path", "meta-llama/Llama-3.2-1B-Instruct")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=original_args.get("max_length", 2048),
        padding_side="right",
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            need_resize = False
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            need_resize = True
    else:
        need_resize = False
    
    # Load model and ReFT checkpoint
    dtype = dtype_mapping.get(original_args.get("dtype", "bfloat16"), torch.bfloat16)
    print(f"Loading ReFT model from {reft_dir}...")
    
    # First load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    
    # Resize embeddings if we added a new pad token (to match original training)
    if need_resize:
        base_model.resize_token_embeddings(len(tokenizer))
    
    # Then load ReFT interventions on top
    reft_model = ReftModel.load(reft_dir, model=base_model)
    reft_model.set_device(device)
    
    # Count interventions
    num_interventions = len(reft_model.interventions)
    print(f"Loaded {num_interventions} interventions")
    
    # Prepare dataset (same as original)
    print("Loading dataset...")
    raw_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    
    max_n_train = original_args.get("max_n_train_example")
    eval_split = original_args.get("eval_split", 0.05)
    
    if max_n_train is not None:
        raw_dataset = raw_dataset.shuffle(seed=seed)
        total_needed = int(max_n_train / (1 - eval_split))
        raw_dataset = raw_dataset.select(range(min(total_needed, len(raw_dataset))))
    
    processed_dataset = preprocess_tulu3_to_prompt_completion(raw_dataset, tokenizer)
    
    # Split into train/eval
    if eval_split > 0:
        split_dataset = processed_dataset.train_test_split(test_size=eval_split, seed=seed)
        train_hf_dataset = split_dataset["train"]
        eval_hf_dataset = split_dataset["test"]
        
        max_eval = original_args.get("max_eval_samples")
        if max_eval is not None and len(eval_hf_dataset) > max_eval:
            eval_hf_dataset = eval_hf_dataset.shuffle(seed=seed).select(range(max_eval))
    else:
        train_hf_dataset = processed_dataset
        eval_hf_dataset = None
    
    print(f"Dataset: {len(train_hf_dataset)} train, {len(eval_hf_dataset) if eval_hf_dataset else 0} eval")
    
    # Create datasets
    train_dataset = ReftGenerationDataset(
        task="tulu3",
        data_path=None,
        tokenizer=tokenizer,
        data_split="train",
        dataset=train_hf_dataset,
        seed=seed,
        max_n_example=None,
        prompt_field="prompt",
        completion_field="completion",
        num_interventions=num_interventions,
        position=position,
        share_weights=share_weights,
    )
    
    eval_dataset = None
    if eval_hf_dataset is not None:
        eval_dataset = ReftGenerationDataset(
            task="tulu3",
            data_path=None,
            tokenizer=tokenizer,
            data_split="train",
            dataset=eval_hf_dataset,
            seed=seed,
            max_n_example=None,
            prompt_field="prompt",
            completion_field="completion",
            num_interventions=num_interventions,
            position=position,
            share_weights=share_weights,
        )
    
    # Create data collator
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=reft_model.model,
        label_pad_token_id=-100,
        padding="longest",
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    
    # Create output directory
    os.makedirs(new_output_dir, exist_ok=True)
    
    # Calculate new epochs
    original_epochs = original_args.get("epochs", 1)
    new_epochs = original_epochs * epochs_multiplier
    
    # Training arguments - continue from checkpoint
    training_args = TrainingArguments(
        output_dir=new_output_dir,
        run_name=new_run_name,
        num_train_epochs=new_epochs,
        per_device_train_batch_size=original_args.get("batch_size", 2),
        per_device_eval_batch_size=original_args.get("eval_batch_size", 8),
        gradient_accumulation_steps=original_args.get("gradient_accumulation_steps", 16),
        learning_rate=original_args.get("lr", 5e-4),
        lr_scheduler_type=original_args.get("schedule", "constant"),
        warmup_ratio=original_args.get("warmup_ratio", 0.0),
        weight_decay=original_args.get("weight_decay", 0.0),
        logging_steps=original_args.get("logging_steps", 10),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=original_args.get("eval_steps", 50) if eval_dataset is not None else None,
        eval_delay=0,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=(original_args.get("dtype") == "bfloat16" and device == "cuda"),
        fp16=(original_args.get("dtype") == "float16" and device == "cuda"),
        optim="adamw_torch",
        report_to="wandb",
        seed=seed,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=original_args.get("gradient_checkpointing", False),
        # Resume from checkpoint - this handles dataloader state
        ignore_data_skip=False,  # Important: continue from where we left off
    )
    
    # Initialize wandb
    import wandb
    wandb.init(
        project=wandb_project,
        name=new_run_name,
        config={
            **original_args,
            "continued_from": run_name,
            "original_epochs": original_epochs,
            "new_epochs": new_epochs,
            "epochs_multiplier": epochs_multiplier,
        },
    )
    
    # Create trainer
    trainer = ReftTrainerForCausalLMWithEval(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Resume training from checkpoint
    if trainer_checkpoint_dir:
        print(f"Resuming training from {trainer_checkpoint_dir}...")
        trainer.train(resume_from_checkpoint=trainer_checkpoint_dir)
    else:
        print("No trainer checkpoint found, starting fresh training...")
        trainer.train()
    
    # Save final model
    print(f"Saving model to {new_output_dir}")
    reft_model.save(new_output_dir)
    
    # Save training args
    args_dict = {
        **original_args,
        "continued_from": run_name,
        "original_epochs": original_epochs,
        "new_epochs": new_epochs,
        "epochs_multiplier": epochs_multiplier,
        "reft_checkpoint_dir": reft_dir,
        "trainer_checkpoint_dir": trainer_checkpoint_dir,
    }
    with open(os.path.join(new_output_dir, "training_args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
    
    print(f"Training complete! Model saved to {new_output_dir}")
    wandb.finish()
    
    return reft_model


def main():
    parser = argparse.ArgumentParser(description="Continue training from best checkpoints")
    parser.add_argument("--wandb_project", type=str, default="loreft-regret",
                        help="Source wandb project to find best runs")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity (username or team)")
    parser.add_argument("--output_dir", type=str, default="./outputs_10x",
                        help="Output directory for continued training")
    parser.add_argument("--output_wandb_project", type=str, default="loreft-regret-10x",
                        help="Wandb project for continued runs")
    parser.add_argument("--epochs_multiplier", type=int, default=10,
                        help="Multiply original epochs by this factor")
    parser.add_argument("--rank", type=int, default=None,
                        help="Only continue training for this rank")
    parser.add_argument("--position", type=str, default=None,
                        help="Only continue training for this position")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without actually running")
    args = parser.parse_args()
    
    print("Fetching best runs from wandb...")
    best_runs = fetch_best_runs(args.wandb_project, args.wandb_entity)
    
    if not best_runs:
        print("No completed runs found!")
        return
    
    print(f"\nFound {len(best_runs)} unique (rank, position) configurations:")
    for (rank, position), run in sorted(best_runs.items()):
        print(f"  rank={rank}, position={position}: LR={run['lr']}, NLL={run['eval_nll']:.4f}")
    
    # Filter if specific rank/position requested
    if args.rank is not None or args.position is not None:
        filtered = {}
        for (rank, position), run in best_runs.items():
            if args.rank is not None and rank != args.rank:
                continue
            if args.position is not None and position != args.position:
                continue
            filtered[(rank, position)] = run
        best_runs = filtered
        print(f"\nFiltered to {len(best_runs)} configurations")
    
    # Continue training for each
    for (rank, position), run in sorted(best_runs.items()):
        print(f"\n{'#'*60}")
        print(f"# Processing: rank={rank}, position={position}")
        print(f"# Best LR: {run['lr']}, Eval NLL: {run['eval_nll']:.4f}")
        print(f"{'#'*60}")
        
        try:
            continue_training(
                original_run=run,
                output_base_dir=args.output_dir,
                wandb_project=args.output_wandb_project,
                epochs_multiplier=args.epochs_multiplier,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"ERROR: Failed to continue training for rank={rank}, position={position}")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

