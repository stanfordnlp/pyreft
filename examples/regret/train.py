"""
LoReFT training script for Llama 3.2 1B on allenai/tulu-3-sft-mixture.

Usage:
    python train.py \
        --model_name_or_path meta-llama/Llama-3.2-1B \
        --rank 4 \
        --layers "all" \
        --position "f1+l1" \
        --lr 5e-4 \
        --epochs 1 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --max_length 2048 \
        --max_n_train_example 10000 \
        --output_dir ./outputs
"""

import os
import argparse
import datetime
import json

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
from tqdm import tqdm

from pyreft import (
    get_reft_model,
    ReftConfig,
    LoreftIntervention,
    ReftDataCollator,
    ReftGenerationDataset,
)
from trainer import ReftTrainerForCausalLMWithEval

device = "cuda" if torch.cuda.is_available() else "cpu"

dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def preprocess_tulu3_to_prompt_completion(dataset, tokenizer):
    """
    Preprocess Tulu-3 dataset to add 'prompt' and 'completion' fields
    that can be used with ReftGenerationDataset.
    
    Converts chat messages to prompt/completion format using the tokenizer's
    chat template when available.
    """
    def convert_example(example):
        messages = example.get("messages", [])
        if not messages:
            return {"prompt": "", "completion": ""}
        
        # Split messages into prompt (everything before assistant) and completion
        prompt_messages = []
        completion = ""
        
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                # First assistant message is the completion
                completion = msg["content"]
                break
            prompt_messages.append(msg)
        
        # Format prompt using chat template
        if not hasattr(tokenizer, 'apply_chat_template') or tokenizer.chat_template is None:
            raise ValueError(
                "Tokenizer does not have a chat template. "
                "Please use a tokenizer with apply_chat_template support (e.g., Llama 3.2)."
            )
        
        prompt = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return {"prompt": prompt, "completion": completion}
    
    print("Converting Tulu-3 messages to prompt/completion format...")
    return dataset.map(convert_example, desc="Preprocessing")


def train(args):
    """Main training function."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup run name
    model_str = args.model_name_or_path.split("/")[-1]
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{model_str}.tulu3.r{args.rank}.{now}"
    
    print(f"Starting training run: {run_name}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Rank: {args.rank}, Layers: {args.layers}, Position: {args.position}")
    print(f"LR: {args.lr}, Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    # Parse layers
    if args.layers == "all":
        temp_config = AutoConfig.from_pretrained(args.model_name_or_path)
        layers = list(range(temp_config.num_hidden_layers))
    elif args.layers.strip() == "":
        layers = []
    else:
        layers = [int(l) for l in args.layers.split(";")]
    
    # Duplicate layers if using multiple positions without weight sharing
    if "+" in args.position and not args.share_weights:
        layers = layers + layers
    
    print(f"Intervening on {len(layers)} layer(s): {layers[:5]}..." if len(layers) > 5 else f"Intervening on layers: {layers}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=True,
    )
    
    # Handle special tokens for Llama 3.2
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # Add a new pad token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            need_resize = True
        need_resize = tokenizer.pad_token_id is None or tokenizer.pad_token == '[PAD]'
    else:
        need_resize = False
    
    # Load model
    dtype = dtype_mapping.get(args.dtype, torch.bfloat16)
    print(f"Loading model with dtype: {args.dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    
    # Resize embeddings if needed
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))
    
    # Create LoReFT interventions
    print(f"Creating LoReFT interventions with rank={args.rank}")
    representations = [{
        "layer": l,
        "component": "block_output",
        "low_rank_dimension": args.rank,
        "intervention": LoreftIntervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=args.rank,
            dropout=args.dropout,
            dtype=dtype,
            act_fn=args.act_fn,
        )
    } for l in layers]
    
    # Create ReFT config and model
    reft_config = ReftConfig(representations=representations)
    reft_model = get_reft_model(model, reft_config, set_device=(device == "cuda"))
    reft_model.print_trainable_parameters()
    
    # Load and preprocess Tulu-3 dataset
    # Note: We preprocess externally because ReftGenerationDataset.tokenize() expects
    # prompt/completion fields, but Tulu-3 has 'messages'. There's no hook in ReftDataset
    # to transform data between load_dataset() and tokenize().
    # We subset BEFORE preprocessing to avoid processing millions of unused examples.
    print("Loading allenai/tulu-3-sft-mixture dataset...")
    raw_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    
    if args.max_n_train_example is not None:
        raw_dataset = raw_dataset.shuffle(seed=args.seed)
        # Request extra examples for eval split
        total_needed = int(args.max_n_train_example / (1 - args.eval_split))
        raw_dataset = raw_dataset.select(range(min(total_needed, len(raw_dataset))))
    
    processed_dataset = preprocess_tulu3_to_prompt_completion(raw_dataset, tokenizer)
    
    # Split into train/eval
    if args.eval_split > 0:
        split_dataset = processed_dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_hf_dataset = split_dataset["train"]
        eval_hf_dataset = split_dataset["test"]
        print(f"Split dataset: {len(train_hf_dataset)} train, {len(eval_hf_dataset)} eval")
    else:
        train_hf_dataset = processed_dataset
        eval_hf_dataset = None
    
    # Use ReftGenerationDataset from pyreft
    train_dataset = ReftGenerationDataset(
        task="tulu3",
        data_path=None,
        tokenizer=tokenizer,
        data_split="train",
        dataset=train_hf_dataset,
        seed=args.seed,
        max_n_example=None,  # Already handled above
        prompt_field="prompt",
        completion_field="completion",
        num_interventions=len(layers),
        position=args.position,
        share_weights=args.share_weights,
    )
    
    # Create eval dataset if we have eval data
    eval_dataset = None
    if eval_hf_dataset is not None:
        eval_dataset = ReftGenerationDataset(
            task="tulu3",
            data_path=None,
            tokenizer=tokenizer,
            data_split="train",  # Use "train" to get labels
            dataset=eval_hf_dataset,
            seed=args.seed,
            max_n_example=None,
            prompt_field="prompt",
            completion_field="completion",
            num_interventions=len(layers),
            position=args.position,
            share_weights=args.share_weights,
        )
    
    # Create data collator
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.schedule,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=(args.dtype == "bfloat16" and device == "cuda"),
        fp16=(args.dtype == "float16" and device == "cuda"),
        optim="adamw_torch",
        report_to="wandb" if args.use_wandb else "none",
        seed=args.seed,
        remove_unused_columns=False,  # Required for ReFT
        dataloader_pin_memory=True,
        gradient_checkpointing=args.gradient_checkpointing,
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
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}")
    reft_model.save(output_dir)
    
    # Save training args
    args_dict = vars(args)
    args_dict["layers_used"] = layers
    args_dict["n_params"] = reft_model.count_parameters(include_model=False)
    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
    
    print(f"Training complete! Model saved to {output_dir}")
    return reft_model


def main():
    parser = argparse.ArgumentParser(
        description="Train Llama 3.2 1B on Tulu-3 SFT Mixture using LoReFT"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Path to the base model (default: meta-llama/Llama-3.2-1B)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights (default: bfloat16)"
    )
    
    # LoReFT arguments
    parser.add_argument(
        "--rank", "-r",
        type=int,
        default=4,
        help="Low-rank dimension for LoReFT intervention (default: 4)"
    )
    parser.add_argument(
        "--layers", "-l",
        type=str,
        default="all",
        help="Layers to intervene on, semicolon-separated (e.g., '0;4;8;12') or 'all' (default: all)"
    )
    parser.add_argument(
        "--position", "-p",
        type=str,
        default="f1+l1",
        help="Position string for intervention (e.g., 'f1+l1', 'l1') (default: f1+l1)"
    )
    parser.add_argument(
        "--share_weights",
        action="store_true",
        help="Share intervention weights across positions"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate for LoReFT (default: 0.0)"
    )
    parser.add_argument(
        "--act_fn",
        type=str,
        default=None,
        help="Activation function for LoReFT (default: None/linear)"
    )
    
    # Training arguments
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=4,
        help="Per-device training batch size (default: 4)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for learning rate scheduler (default: 0.03)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (default: 0.0)"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="linear",
        help="Learning rate schedule (default: linear)"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory"
    )
    
    # Data arguments
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--max_n_train_example",
        type=int,
        default=None,
        help="Maximum number of training examples (default: None = use all)"
    )
    parser.add_argument(
        "--eval_split",
        type=float,
        default=0.05,
        help="Fraction of data to hold out for evaluation (default: 0.05)"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500)"
    )
    
    # Logging and output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for model checkpoints (default: ./outputs)"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging frequency in steps (default: 10)"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="loreft-tulu3",
        help="Weights & Biases project name (default: loreft-tulu3)"
    )
    
    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set wandb project if using wandb
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project)
    
    train(args)


if __name__ == "__main__":
    main()

