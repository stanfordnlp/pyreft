# Claude Context for regret experiments

## Project Owner
Aryaman Arora - co-author of the ReFT paper.

## Goal
Replicating the ["LoRA Without Regret" blog post](https://thinkingmachines.ai/blog/lora/) but for **LoReFT** instead of LoRA. Analyzing how LoReFT performance scales with rank and learning rate.

## Key Files
- `train.py` - Main training script for LoReFT and full fine-tuning baseline
- `trainer.py` - Custom trainers with NLL/perplexity evaluation
- `sweep.sbatch` - SLURM job template
- `launch_sweep.sh` - Launches grid search over rank × LR × position

## Sweep Configuration
- Model: Llama 3.2 1B Instruct
- Dataset: allenai/tulu-3-sft-mixture (50K examples)
- Ranks: 1, 2, 4, 8, 16, 32, 64
- LRs: 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3
- Positions: `f1+l1` (first + last token) and `all` (all prompt tokens)
- 84 total jobs

## Recent Additions
- `position="all"` - intervenes on all prompt tokens (requires `share_weights=True`)
- `--max_eval_samples` - subsample eval set for faster training (default: 500)
- `--use_flash_attn` - Flash Attention 2 support
- `--skip-done` flag in launch_sweep.sh to skip completed jobs

## Known Issues / Tech Debt
1. **position vs positions**: Inconsistent naming in `pyreft/dataset.py`. `get_intervention_locations` checks both keys as a workaround.
2. **flash-attn install**: Requires separate sync (see below)
3. **Variable-length intervention_locations**: Fixed in `ReftDataCollator` by padding to max length with -1

## Useful Commands
```bash
# Install with uv
uv sync
uv sync --extra flash  # optional, for Flash Attention 2

# Run sweep (dry run first)
./launch_sweep.sh --dry-run
./launch_sweep.sh --skip-done

# Single test run
uv run train.py --max_n_train_example 100 --position f1+l1 --rank 4
```

## Branch
`aryaman/regret`

