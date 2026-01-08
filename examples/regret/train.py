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
    LoreftIntervention_Scale,
    LoreftIntervention_SigmoidScale,
    LoreftIntervention_DataDepScale,
    LoreftIntervention_TokenScale,
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


class SimpleGenerationDataset(torch.utils.data.Dataset):
    """
    Simple dataset for full fine-tuning (no intervention locations).
    """
    def __init__(self, hf_dataset, tokenizer, max_length=2048):
        self.data = []
        print("Tokenizing dataset for full fine-tuning...")
        
        for item in hf_dataset:
            prompt = item["prompt"]
            completion = item["completion"]
            
            # Tokenize prompt to get its length
            prompt_ids = tokenizer(
                prompt, max_length=max_length, truncation=True, return_tensors="pt"
            )["input_ids"][0]
            prompt_length = len(prompt_ids)
            
            # Tokenize full sequence
            full_text = prompt + completion + tokenizer.eos_token
            full_ids = tokenizer(
                full_text, max_length=max_length, truncation=True, return_tensors="pt"
            )["input_ids"][0]
            
            # Create labels (mask prompt)
            labels = full_ids.clone()
            labels[:prompt_length] = IGNORE_INDEX
            
            self.data.append({
                "input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids),
                "labels": labels,
            })
        
        print(f"Prepared {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


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
    
    # Validate arguments (do this first before building run_name)
    if args.disable_reft and not args.use_lora:
        raise ValueError("--disable_reft requires --use_lora (otherwise nothing would be trained)")
    if args.full_finetune and (args.use_lora or args.disable_reft):
        raise ValueError("--full_finetune cannot be combined with --use_lora or --disable_reft")
    
    # Setup run name
    if args.run_name:
        run_name = args.run_name
    else:
        model_str = args.model_name_or_path.split("/")[-1]
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if args.full_finetune:
            run_name = f"{model_str}.tulu3.fullft.{now}"
        elif args.use_lora and args.disable_reft:
            run_name = f"{model_str}.tulu3.lora_r{args.lora_rank}.{now}"
        elif args.use_lora:
            run_name = f"{model_str}.tulu3.lora_r{args.lora_rank}_reft_r{args.rank}.{now}"
        else:
            run_name = f"{model_str}.tulu3.reft_r{args.rank}.{now}"
    
    # Determine training mode
    use_reft = not args.full_finetune and not args.disable_reft
    
    print(f"Starting training run: {run_name}")
    print(f"Model: {args.model_name_or_path}")
    if args.full_finetune:
        print("Mode: Full fine-tuning (baseline)")
    elif args.use_lora and args.disable_reft:
        print(f"Mode: LoRA-only (rank={args.lora_rank}, alpha={args.lora_alpha}, modules={args.lora_modules})")
    elif args.use_lora:
        scale_info = f", scale={args.scale_type}" if args.scale_type else ""
        print(f"Mode: LoRA + LoReFT (LoRA rank={args.lora_rank}, ReFT rank={args.rank}{scale_info})")
    else:
        scale_info = f", scale={args.scale_type}" if args.scale_type else ""
        print(f"Mode: LoReFT (rank={args.rank}, layers={args.layers}, position={args.position}{scale_info})")
    print(f"LR: {args.lr}, Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    # Parse ReFT layers (only needed when using ReFT)
    layers = []
    if use_reft:
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
        
        print(f"ReFT: Intervening on {len(layers)} layer(s): {layers[:5]}..." if len(layers) > 5 else f"ReFT: Intervening on layers: {layers}")
    
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
    
    # Use flash attention if available (requires flash-attn package)
    attn_implementation = "flash_attention_2" if args.use_flash_attn else None
    if attn_implementation:
        print("Using Flash Attention 2")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    
    # Resize embeddings if needed
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA if requested (before ReFT wrapping)
    if args.use_lora:
        if not is_peft_available:
            raise ModuleNotFoundError(
                "peft package is required for LoRA. Install with: pip install peft"
            )
        from peft import LoraConfig, get_peft_model
        
        print(f"Enabling LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")
        lora_modules = [m.strip() for m in args.lora_modules.split(";")]
        
        # Parse LoRA layers
        if args.lora_layers == "all":
            lora_layers_to_transform = None  # peft default: all layers
        else:
            lora_layers_to_transform = [int(l) for l in args.lora_layers.split(";")]
        
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=lora_modules,
            layers_to_transform=lora_layers_to_transform,
            use_rslora=False,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        print(f"LoRA target modules: {lora_modules}")
        model.print_trainable_parameters()
    
    # Setup model based on training mode
    if args.full_finetune:
        # Full fine-tuning: enable all gradients
        print("Enabling gradients on all model parameters...")
        for param in model.parameters():
            param.requires_grad = True
        reft_model = None
        train_model = model
    elif use_reft:
        # Select intervention class based on scale_type
        scale_type = args.scale_type or "none"
        intervention_classes = {
            "none": LoreftIntervention,
            "scalar": LoreftIntervention_Scale,
            "sigmoid": LoreftIntervention_SigmoidScale,
            "datadep": LoreftIntervention_DataDepScale,
            "token": LoreftIntervention_TokenScale,
        }
        intervention_cls = intervention_classes[scale_type]
        
        # Create LoReFT interventions
        scale_str = f", scale={scale_type}" if scale_type != "none" else ""
        print(f"Creating LoReFT interventions with rank={args.rank}{scale_str}")
        
        # Component path depends on whether we're wrapping a PEFT model
        if args.use_lora:
            # PEFT model has a different module structure
            component = "base_model.model.model.layers[{layer}].output"
        else:
            component = "block_output"
        
        representations = [{
            "layer": l,
            "component": component.format(layer=l) if args.use_lora else component,
            "low_rank_dimension": args.rank,
            "intervention": intervention_cls(
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
        
        # Re-enable LoRA adapter layers after ReFT wrapping
        if args.use_lora:
            reft_model.model.enable_adapter_layers()
        
        reft_model.print_trainable_parameters()
        train_model = reft_model
    else:
        # LoRA-only mode (args.use_lora and args.disable_reft)
        reft_model = None
        train_model = model
    
    # Count params with requires_grad=True (catches bugs where grads aren't set correctly)
    trainable_params = sum(p.numel() for p in train_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in train_model.parameters())
    print(f"trainable params: {trainable_params:,d} || total params: {total_params:,d} || trainable%: {100 * trainable_params / total_params:.4f}")
    
    # Log params to wandb
    if args.use_wandb:
        import wandb
        wandb.log({
            "trainable_params": trainable_params,
            "total_params": total_params,
        })
    
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
        # Subsample eval set if requested (for faster evaluation)
        if args.max_eval_samples is not None and len(eval_hf_dataset) > args.max_eval_samples:
            eval_hf_dataset = eval_hf_dataset.shuffle(seed=args.seed).select(range(args.max_eval_samples))
            print(f"Split dataset: {len(train_hf_dataset)} train, {len(eval_hf_dataset)} eval (subsampled from {len(split_dataset['test'])})")
        else:
            print(f"Split dataset: {len(train_hf_dataset)} train, {len(eval_hf_dataset)} eval")
    else:
        train_hf_dataset = processed_dataset
        eval_hf_dataset = None
    
    # Create datasets based on training mode
    if args.full_finetune or (args.use_lora and args.disable_reft):
        # Simple dataset for full fine-tuning or LoRA-only (no intervention locations)
        train_dataset = SimpleGenerationDataset(train_hf_dataset, tokenizer, args.max_length)
        eval_dataset = None
        if eval_hf_dataset is not None:
            eval_dataset = SimpleGenerationDataset(eval_hf_dataset, tokenizer, args.max_length)
        
        # Standard data collator
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest",
        )
    else:
        # Use ReftGenerationDataset from pyreft (for LoReFT or LoRA+LoReFT)
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
        eval_delay=0,  # Evaluate at step 0
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
    
    # Create trainer based on mode
    if args.full_finetune or (args.use_lora and args.disable_reft):
        # Use standard trainer for full finetune or LoRA-only
        trainer = FullFinetuneTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
    else:
        # Use ReFT trainer for LoReFT or LoRA+LoReFT
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
    if args.full_finetune:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    elif args.use_lora and args.disable_reft:
        # LoRA-only: save using peft
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    elif args.use_lora:
        # LoRA + LoReFT: save both
        reft_model.save(output_dir)
        # LoRA adapters are saved within the reft_model
    else:
        # LoReFT-only
        reft_model.save(output_dir)
    
    # Save training args
    args_dict = vars(args)
    args_dict["layers_used"] = layers
    args_dict["trainable_params"] = trainable_params
    args_dict["total_params"] = total_params
    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
    
    print(f"Training complete! Model saved to {output_dir}")
    if args.full_finetune or (args.use_lora and args.disable_reft):
        return model
    else:
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
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use Flash Attention 2 (requires flash-attn package)"
    )
    
    # Training mode
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Full fine-tuning baseline (no ReFT, trains all parameters)"
    )
    
    # LoRA arguments (can be combined with LoReFT or used alone with --disable_reft)
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA fine-tuning (can be combined with LoReFT or used alone with --disable_reft)"
    )
    parser.add_argument(
        "--disable_reft",
        action="store_true",
        help="Disable ReFT interventions (use with --use_lora for LoRA-only baseline)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32)"
    )
    parser.add_argument(
        "--lora_modules",
        type=str,
        default="q_proj;v_proj",
        help="Target modules for LoRA, semicolon-separated (default: q_proj;v_proj)"
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default="all",
        help="Layers to apply LoRA, semicolon-separated or 'all' (default: all)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default: 0.05)"
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
    parser.add_argument(
        "--scale_type",
        type=str,
        default=None,
        choices=[None, "none", "scalar", "sigmoid", "datadep", "token"],
        help="Gating type for LoReFT: none (default), scalar (learned), sigmoid (bounded [0,2]), datadep (per-sequence), token (per-token)"
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
        default=0.0,
        help="Warmup ratio for learning rate scheduler (default: 0.0)"
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
        default="constant",
        help="Learning rate schedule (default: constant)"
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
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum eval samples to use during evaluation (default: None = use all)"
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
        "--run_name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)"
    )
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
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )
    
    train(args)


if __name__ == "__main__":
    main()

