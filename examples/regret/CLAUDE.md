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

### Why Identity Init Didn't Help
We tried initializing LoReFT to act as identity at step 0 (W=R^T, b=0). This gave low loss at step 0, but after one gradient step the loss spiked. The problem is **gradient dynamics**, not initialization:
- LoRA: B=0 init means BA stays small early (self-regularizing)
- ReFT: W and R are full-sized, so W-R can grow large immediately after one gradient step

A potential fix: reparameterize as `W = R + ΔW` where `ΔW` is learned and initialized to 0.

## Key Files
- `train.py` - Main training script for LoReFT, LoRA, LoRA+LoReFT, and full fine-tuning
- `trainer.py` - Custom trainers with NLL/perplexity evaluation and intervention metric logging
- `continue_training.py` - Resume training from checkpoints for 10x continuation experiments
- `scripts/` - SLURM job templates and launch scripts
  - `sweep.sbatch`, `launch_sweep.sh` - Main sweep
  - `continue_sweep.sbatch`, `launch_continue_sweep.sh` - 10x continuation
  - `rank1_sweep.sbatch`, `launch_rank1_sweep.sh` - Quick rank-1 experiments
- `analysis/plot_sweep.py` - Visualization: scaling curves, coefficients, method comparison

## Position Parsing (Strict Mode)
There was an off-by-one bug in the original position parsing. For backwards compatibility:
- `f1+l1`, `all` - Legacy mode (last token is one before actual last)
- `f1+s1`, `alls` - **Strict mode** (correctly includes the actual last token)

Use strict mode (`f1+s1`, `alls`) for new experiments.

## Sweep Configuration
- Model: Llama 3.2 1B Instruct
- Dataset: allenai/tulu-3-sft-mixture (50K examples)
- Ranks: 1, 2, 4, 8, 16, 32, 64
- LRs: 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3
- Positions: `f1+l1`, `all`, `f1+s1`, `alls`
- wandb project: `loreft-regret` (original), `loreft-regret-10x` (continuations)

## Launch Script Flags
```bash
./scripts/launch_sweep.sh --dry-run        # Preview without submitting
./scripts/launch_sweep.sh --skip-done      # Skip completed runs
./scripts/launch_sweep.sh --skip-lora      # ReFT only, no LoRA baseline
./scripts/launch_sweep.sh --rank1-only     # Only rank=1 experiments
./scripts/launch_sweep.sh --strict-only    # Only f1+s1 and alls positions
```

## Checkpoint Loading Quirks
The `continue_training.py` script has to handle a complex checkpoint structure:
- `outputs/{run_name}/config.json` - ReFT config (in parent dir)
- `outputs/{run_name}/checkpoint-{step}/` - HuggingFace Trainer state
- `outputs/{run_name}/checkpoint-{step}/intervenable_model/` - ReFT intervention weights

The `pyvene.IntervenableModel.load()` can be finicky. If you get `TypeError: 'type' object is not iterable`, the config and weights are likely in different locations than expected.

## Known Issues / Tech Debt
1. **position vs positions**: Inconsistent naming in `pyreft/dataset.py`. `get_intervention_locations` checks both keys as a workaround.
2. **flash-attn install**: Requires separate sync (see below)
3. **Variable-length intervention_locations**: Fixed in `ReftDataCollator` by padding to max length with -1
4. **Trainer.log() signature**: HuggingFace changed it to include `start_time` - our custom trainer override must match
5. **resize_token_embeddings**: When loading models for continuation, must call `model.resize_token_embeddings(len(tokenizer))` after adding pad token

## Useful Commands
```bash
# Install with uv
uv sync
uv sync --extra flash  # optional, for Flash Attention 2
uv sync --extra peft   # optional, for LoRA support

# Run sweep (dry run first)
./scripts/launch_sweep.sh --dry-run
./scripts/launch_sweep.sh --skip-done

# Single test run
uv run train.py --max_n_train_example 100 --position f1+s1 --rank 4

# LoReFT-only (default)
uv run train.py --max_n_train_example 100 --rank 4

# LoRA-only
uv run train.py --max_n_train_example 100 --use_lora --disable_reft --lora_rank 8

# LoRA + LoReFT combined
uv run train.py --max_n_train_example 100 --use_lora --lora_rank 8 --rank 4

# Full fine-tuning baseline
uv run train.py --max_n_train_example 100 --full_finetune --gradient_checkpointing

# Generate plots (with 10x continuation data)
cd analysis
python plot_sweep.py --project loreft-regret --project-10x loreft-regret-10x --curves
```

## Intervention Metric Logging
The trainer logs intervention metrics to wandb:
- `intervention/delta_norm` - Norm of the intervention delta (how much the hidden state changes)
- `intervention/diff_norm` - Norm of difference from identity
- `intervention/delta_base_ratio` - Ratio of delta to base hidden state norm

## Branch
`aryaman/regret`

## Theoretical Connection
LoReFT has structural similarities to the **Widrow-Hoff / Delta Rule** (used in DeltaNet):
- Both involve projecting onto a learned subspace, applying a transformation, and projecting back
- The key difference: in LoReFT, R (the rotation matrix) is also learned, not fixed
- This might explain some of the optimization challenges