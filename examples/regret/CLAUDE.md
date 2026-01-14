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

**Tried and didn't work:**
- Zero-init (W = R^T, b = 0 so diff ≈ 0 at init) — no improvement in sample efficiency

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

## Component Selection
ReFT can target different transformer components via `--component`:
- **`block_output`** (default) - Residual stream
- **`mlp_activation`** - MLP intermediate layer (use `--with-mlp` flag in sweep)
- **`mlp_output`** - MLP/FFN output
- **`attention_output`** - Attention module output

## Intervention Types
Four ReFT variants are available via `--intervention_type`:
- **`loreft`** (default): LoReFT(h) = h + R^T(Wh + b − Rh) — replaces R-subspace component (R orthogonal)
- **`direft`**: DiReFT(h) = h + R^T(Wh + b) — adds to R-subspace (R orthogonal, no subtraction)
- **`nodireft`**: NoDiReFT(h) = h + W2^T(W1h + b) — no orthogonality constraint on W2
- **`moeloreft`**: MoE-LoReFT(h) = h + R^T(∑ᵢsᵢWᵢh + b − Rh) — mixture-of-experts on W with top-k routing

Key differences:
- **LoReFT vs DiReFT**: In DiReFT, when W=0 and b=0, the intervention is identity. In LoReFT, it subtracts the R-subspace. This affects weight decay regularization.
- **DiReFT vs NoDiReFT**: NoDiReFT removes the orthogonality constraint on the projection matrix, making it a standard low-rank adapter.
- **MoE-LoReFT**: Uses `--num_experts` (default 4) and `--top_k` (default 2) to route each token to a subset of expert W matrices. Logs per-expert activation percentages.

## Sweep Configuration
- **Model (1B)**: Llama 3.2 1B Instruct
- **Model (8B)**: Llama 3.1 8B Instruct (use `--model-8b` flag)
- Dataset: allenai/tulu-3-sft-mixture (50K examples)
- Ranks: 1, 2, 4, 8, 16, 32, 64
- LRs: 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3
- Positions: `f1+l1`, `all`, `f1+s1`, `alls`
- Component: `block_output` (default, residual stream)
- wandb projects:
  - `loreft-regret` (1B, 1 epoch)
  - `loreft-regret-10x-restart` (1B, 10 epochs)
  - `loreft-regret-8b` (8B, 1 epoch)

## Launch Script Flags
- `--dry-run` - Preview jobs without submitting
- `--skip-done` - Skip completed runs
- `--rank1-only` - Only rank=1 (for testing)
- `--all-positions` - All positions: f1+l1, all, f1+s1, alls
- `--with-mlp` - Add mlp_activation component experiments
- `--with-lora` - Include LoRA baseline
- `--with-direft` - Include DiReFT experiments
- `--with-nodireft` - Include NoDiReFT experiments (no orthogonality)
- `--with-moeloreft` - Include MoE-LoReFT experiments (mixture-of-experts)
- `--model-8b` - Use Llama 3.1 8B instead of 3.2 1B (auto-enables gradient checkpointing, 48G memory)

## Quick Start
```bash
# Setup
uv sync --extra flash --extra peft

# Run 1B sweep (7 ranks × 6 LRs × 1 position = 42 jobs)
./scripts/launch_sweep.sh --dry-run --skip-done

# Run 8B sweep (same grid, larger model)
./scripts/launch_sweep.sh --model-8b --dry-run --skip-done

# Add mlp_activation experiments (doubles jobs: 84 total)
./scripts/launch_sweep.sh --with-mlp --skip-done

# Add DiReFT experiments (doubles ReFT jobs: 84 total)
./scripts/launch_sweep.sh --with-direft --skip-done

# Single test run (1B)
uv run train.py --max_n_train_example 100 --position f1+s1 --rank 4 --component mlp_activation

# DiReFT test run
uv run train.py --max_n_train_example 100 --position f1+s1 --rank 4 --intervention_type direft

# NoDiReFT test run (no orthogonality)
uv run train.py --max_n_train_example 100 --position f1+s1 --rank 4 --intervention_type nodireft

# MoE-LoReFT test run (mixture-of-experts)
uv run train.py --max_n_train_example 100 --position f1+s1 --rank 4 --intervention_type moeloreft --num_experts 4 --top_k 2

# Single test run (8B)
uv run train.py --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --max_n_train_example 100 --position f1+s1 --rank 4 \
    --gradient_checkpointing

# LoRA baseline
uv run train.py --max_n_train_example 100 --use_lora --disable_reft --lora_rank 8

# 10x retraining
python continue_training.py --rank 4 --position f1+s1

# Generate plots
cd analysis && python plot_sweep.py --project loreft-regret --project-10x loreft-regret-10x-restart --curves
```

## Intervention Debug Logging
Pass `--debug_interventions` to enable metrics logging to wandb:
- `intervention/base_norm_mean` - norm of input activations (h)
- `intervention/diff_norm_mean` / `_max` - norm of (Wh+b - Rh)
- `intervention/b_norm_mean` - bias norm
- `intervention/delta_base_ratio_mean` / `_max` - relative intervention magnitude

This flag is enabled by default in all sweep scripts.

## Orthogonality Save/Load Fix
The orthogonal parameterization in LoReFT now correctly preserves internal state during checkpoint save/load. Previously, only the computed orthogonal weight was saved, which broke orthogonality during training continuation (causing loss spikes).

**New state_dict format** (backwards compatible):
- `rotate_layer` - computed orthogonal weight (for inference)
- `rotate_layer_original` - internal optimization variable
- `rotate_layer_base` - trivialization base matrix

Legacy checkpoints (without `_original` and `_base`) still work for inference but may break during continued training.

See `tests/test_orthogonality_save_load.py` for verification.

## Known Issues
1. **position vs positions**: Inconsistent naming in `pyreft/dataset.py`. `get_intervention_locations` checks both keys as a workaround.
2. **Trainer.log() signature**: HuggingFace changed it to include `start_time` - our custom trainer override must match.
3. **Gradient checkpointing + pyvene**: Incompatible — pyvene hooks cause tensor count mismatch during recomputation. Use smaller batch size + gradient accumulation instead. See [pyvene issue #231](https://github.com/stanfordnlp/pyvene/issues/231).

## Weight Decay & Intervention Design (Research Notes)
**Key insight**: Weight decay behaves differently for LoReFT vs DiReFT/NoDiReFT:
- **LoRA**: W = W₀ + BA, B zero-init → weight decay pushes toward pretrained (do nothing)
- **LoReFT**: W random, weight decay pushes W→0 → does NOT regularize toward identity
- **DiReFT/NoDiReFT**: When W=0, b=0, intervention is identity → weight decay regularizes toward "do nothing"

This may explain sample efficiency differences. DiReFT with proper weight decay could behave more like LoRA.

**Norm growth observation**: During training, `diff_norm` grows (log-linear), but `delta_base_ratio` saturates below 1. Each layer's intervention adds to the norm — with 32 layers this compounds. LayerNorm prevents explosion but geometry still shifts. Potential fix: norm-preserving interventions.

## Custom Slash Commands
Located in `.claude/commands/` at repo root:
- `/project:commit-push [message]` - Stage, commit, and push changes
- `/project:add-intervention [name]` - Add new intervention type (imports, argparse, sweep scripts, docs)
- `/project:update-docs [context]` - Update CLAUDE.md based on recent changes

## Branch
`aryaman/regret`
