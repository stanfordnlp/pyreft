# LoReFT Training on Tulu-3 SFT Mixture

This directory contains a script for fine-tuning Llama 3.2 1B (base) on the [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset using LoReFT (Low-Rank Representation Fine-Tuning).

## Requirements

Using [uv](https://github.com/astral-sh/uv):
```bash
uv sync  # Install dependencies from pyproject.toml
```

Or with pip:
```bash
pip install torch transformers datasets pyvene tqdm wandb
```

You'll also need access to the Llama 3.2 model on Hugging Face. Make sure you're logged in:
```bash
huggingface-cli login
```

## Quick Start

Basic training with default settings:
```bash
uv run train.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --max_n_train_example 10000 \
    --output_dir ./outputs
```

## Full Usage

```bash
uv run train.py \
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
    --output_dir ./outputs \
    --use_wandb
```

## Key Arguments

### LoReFT Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--rank` | 4 | Low-rank dimension for LoReFT intervention |
| `--layers` | "all" | Layers to intervene on (semicolon-separated, e.g., "0;4;8;12") |
| `--position` | "f1+l1" | Position string for intervention |
| `--share_weights` | False | Share intervention weights across positions |
| `--dropout` | 0.0 | Dropout rate for LoReFT |
| `--act_fn` | None | Activation function (linear by default) |

### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 5e-4 | Learning rate |
| `--epochs` | 1 | Number of training epochs |
| `--batch_size` | 4 | Per-device batch size |
| `--gradient_accumulation_steps` | 8 | Gradient accumulation steps |
| `--warmup_ratio` | 0.03 | Warmup ratio |
| `--weight_decay` | 0.0 | Weight decay |
| `--schedule` | "linear" | LR scheduler type |
| `--gradient_checkpointing` | False | Enable gradient checkpointing |

### Data Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_length` | 2048 | Maximum sequence length |
| `--max_n_train_example` | None | Limit training examples (useful for debugging) |

### Evaluation Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval_split` | 0.05 | Fraction of data to hold out for evaluation |
| `--eval_batch_size` | 8 | Batch size for evaluation |
| `--eval_steps` | 500 | Evaluate every N steps |

The trainer logs **NLL (negative log-likelihood)** and **perplexity** on the held-out eval set to wandb (if enabled) or console.

### Other Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--dtype` | "bfloat16" | Model dtype (float32/float16/bfloat16) |
| `--output_dir` | "./outputs" | Output directory |
| `--seed` | 42 | Random seed |
| `--use_wandb` | False | Enable W&B logging |

## Position String Format

The position string controls where interventions are applied:

- `f1` - First token only
- `l1` - Last token only  
- `f1+l1` - First and last tokens (recommended)
- `f2+l2` - First 2 and last 2 tokens

## Layer Selection

- `"all"` - Intervene on all layers
- `"0;4;8;12"` - Intervene on specific layers (semicolon-separated)
- `""` - No intervention (for ablation)

## Example Configurations

### Small-scale experiment (for debugging)
```bash
uv run train.py --rank 2 --max_n_train_example 1000 --epochs 1
```

### Medium rank, subset of layers
```bash
uv run train.py --rank 8 --layers "4;8;12" --max_n_train_example 50000
```

### Full training with higher rank
```bash
uv run train.py --rank 16 --layers "all" --epochs 3 --gradient_checkpointing
```

## Running Sweeps (SLURM)

For hyperparameter sweeps on a cluster:

```bash
# Preview the jobs that will be submitted
./launch_sweep.sh --dry-run

# Submit full grid (7 ranks x 6 LRs = 42 jobs)
./launch_sweep.sh
```

The sweep covers:
- **Ranks**: 1, 2, 4, 8, 16, 32, 64
- **Learning rates**: 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3

Modify `launch_sweep.sh` to adjust the grid, or run individual jobs:
```bash
sbatch --export=RANK=8,LR=5e-4 sweep.sbatch
```

Edit `sweep.sbatch` to set your cluster-specific options (partition, account, modules, conda env).

## Output

The script saves:
- Model checkpoints (via HuggingFace Trainer)
- ReFT intervention weights
- `training_args.json` with all hyperparameters used

## Loading a Trained Model

```python
from pyreft import ReftModel

# Load the trained ReFT model
reft_model = ReftModel.load("./outputs/<run_name>")
```

