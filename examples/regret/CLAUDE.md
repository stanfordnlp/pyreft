# Claude Context for regret experiments

## Project Owner
Aryaman Arora - co-author of the ReFT paper.

## Goal
Replicating the ["LoRA Without Regret" blog post](https://thinkingmachines.ai/blog/lora/) but for **LoReFT** instead of LoRA. Analyzing how LoReFT performance scales with rank and learning rate.

## Key Finding: ReFT vs LoRA Sample Efficiency
When plotting NLL vs log(steps), **ReFT and LoRA have the same slope but different intercepts**. This means:
- Both follow the same scaling law trajectory
- ReFT needs a **constant multiplier more data** to achieve the same loss as LoRA
- The multiplier k = exp((c_reft - c_lora) / |slope|)
- This is fundamentally about **sample efficiency**, not initialization

A potential fix: reparameterize as `W = R + ΔW` where `ΔW` is learned and initialized to 0.

## Key Files
- `train.py` - Main training script for LoReFT, LoRA, LoRA+LoReFT, and full fine-tuning
- `trainer.py` - Custom trainers with NLL/perplexity evaluation and intervention metric logging
- `continue_training.py` - Retrain with best LR for 10x epochs (fresh start, not checkpoint continuation)
- `scripts/` - SLURM job templates and launch scripts
  - `sweep.sbatch`, `launch_sweep.sh` - Main sweep
  - `continue_sweep.sbatch`, `launch_continue_sweep.sh` - 10x retraining
- `analysis/plot_sweep.py` - Visualization: scaling curves, coefficients, method comparison

## Position Parsing (Strict Mode)
There was an off-by-one bug in the original position parsing. For backwards compatibility:
- `f1+l1`, `all` - Legacy mode (last token is one before actual last)
- `f1+s1`, `alls` - **Strict mode** (correctly includes the actual last token)

Use strict mode (`f1+s1`, `alls`) for new experiments.

## Component Selection (New Feature)
ReFT interventions can now be applied to different transformer components via `--component`:
- **`block_output`** (default) - Residual stream (current experiments use this)
- **`mlp_output`** - MLP/FFN output
- **`attention_output`** - Attention module output
- **`mlp_activation`** - MLP intermediate activations
- And more (see pyvene docs for full list)

This allows comparing intervention effectiveness at different points in the model.

## Sweep Configuration
- Model: Llama 3.2 1B Instruct
- Dataset: allenai/tulu-3-sft-mixture (50K examples)
- Ranks: 1, 2, 4, 8, 16, 32, 64
- LRs: 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3
- Positions: `f1+l1`, `all`, `f1+s1`, `alls`
- Component: `block_output` (default, residual stream)
- wandb projects: `loreft-regret` (1 epoch), `loreft-regret-10x-restart` (10 epochs)

## Launch Script Flags
```bash
./scripts/launch_sweep.sh --dry-run          # Preview without submitting
./scripts/launch_sweep.sh --skip-done        # Skip completed runs
./scripts/launch_sweep.sh --with-lora        # Include LoRA baseline
./scripts/launch_sweep.sh --all-positions    # All positions (f1+l1, all, f1+s1, alls)
./scripts/launch_sweep.sh --with-mlp         # Add mlp_activation experiments
./scripts/launch_sweep.sh --all-components   # All components (block_output, mlp_activation)
./scripts/launch_sweep.sh --rank1-only       # Only rank=1 experiments
```

## Useful Commands
```bash
# Install with uv
uv sync
uv sync --extra flash  # optional, for Flash Attention 2
uv sync --extra peft   # optional, for LoRA support

# Run sweep (default: f1+s1 position, block_output component)
./scripts/launch_sweep.sh --dry-run
./scripts/launch_sweep.sh --skip-done

# Run sweep with mlp_activation experiments (compares residual stream vs MLP output)
./scripts/launch_sweep.sh --with-mlp --dry-run
./scripts/launch_sweep.sh --with-mlp --skip-done

# Single test run
uv run train.py --max_n_train_example 100 --position f1+s1 --rank 4

# Test with different intervention components
uv run train.py --max_n_train_example 100 --position f1+s1 --rank 4 --component mlp_output
uv run train.py --max_n_train_example 100 --position f1+s1 --rank 4 --component attention_output

# LoRA-only baseline
uv run train.py --max_n_train_example 100 --use_lora --disable_reft --lora_rank 8

# 10x retraining with best LR
python continue_training.py --dry-run
python continue_training.py --rank 4 --position f1+s1

# Generate plots
cd analysis
python plot_sweep.py --project loreft-regret --project-10x loreft-regret-10x-restart --curves
```

## Known Issues
1. **position vs positions**: Inconsistent naming in `pyreft/dataset.py`. `get_intervention_locations` checks both keys as a workaround.
2. **Trainer.log() signature**: HuggingFace changed it to include `start_time` - our custom trainer override must match.

## Branch
`aryaman/regret`
