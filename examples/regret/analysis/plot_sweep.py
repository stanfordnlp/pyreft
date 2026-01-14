#!/usr/bin/env python3
"""
Plot results from the LoReFT sweep experiments.

Usage:
    python plot_sweep.py --project loreft-regret
    python plot_sweep.py --project loreft-regret --project-10x loreft-regret-10x-restart --curves
    python plot_sweep.py --project loreft-regret --output plots/
    python plot_sweep.py --csv results.csv  # Use cached CSV instead of wandb
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.cm.viridis(np.linspace(0, 0.9, 7))

# Model config for Llama 3.2 1B (default)
DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 2048,
    "intermediate_dim": 8192,
    "num_layers": 16,
    "num_heads": 32,
    "head_dim": 64,
}


def compute_lora_flops(
    lora_rank: int,
    num_tokens: int,
    modules: str = "q_proj;k_proj;v_proj;o_proj;gate_proj;up_proj;down_proj",
    model_config: dict = None,
) -> int:
    """
    Compute additional FLOPs for LoRA per forward pass.
    
    LoRA adds low-rank adapters: W' = W + BA where B is (d, r) and A is (r, d).
    FLOPs per adapter per token: 2 * d * r (for x @ B @ A).
    
    Args:
        lora_rank: LoRA rank
        num_tokens: Number of tokens in sequence
        modules: Semicolon-separated list of modules with LoRA
        model_config: Model architecture config
    
    Returns:
        Total additional FLOPs for one forward pass
    """
    if model_config is None:
        model_config = DEFAULT_MODEL_CONFIG
    
    d = model_config["hidden_dim"]
    intermediate = model_config["intermediate_dim"]
    num_layers = model_config["num_layers"]
    
    module_list = modules.split(";") if isinstance(modules, str) else modules
    
    flops_per_layer = 0
    for module in module_list:
        module = module.strip()
        if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            # Attention projections: d -> d
            flops_per_layer += 2 * d * lora_rank * num_tokens
        elif module in ["gate_proj", "up_proj"]:
            # MLP up projections: d -> intermediate
            flops_per_layer += 2 * d * lora_rank * num_tokens
        elif module == "down_proj":
            # MLP down projection: intermediate -> d
            flops_per_layer += 2 * intermediate * lora_rank * num_tokens
    
    return flops_per_layer * num_layers


def compute_reft_flops(
    reft_rank: int,
    prompt_length: int,
    position: str = "f1+s1",
    component: str = "block_output",
    num_layers: int = None,
    model_config: dict = None,
) -> int:
    """
    Compute additional FLOPs for ReFT per forward pass.

    ReFT intervention: h' = h + R(Wh_proj + b - h_proj) where h_proj = R^T h
    FLOPs per intervention per position:
        - R^T @ h: d * r
        - W @ h_proj: r * r
        - R @ delta: r * d
        - Total: ~2*d*r + r*r per intervened position

    Note: ReFT only intervenes on PROMPT tokens, not response tokens.

    Args:
        reft_rank: ReFT rank (low_rank_dimension)
        prompt_length: Number of prompt tokens (not full sequence)
        position: Position string (f1+s1, all, etc.)
        component: Transformer component (block_output, mlp_activation, etc.)
        num_layers: Override number of layers with interventions
        model_config: Model architecture config

    Returns:
        Total additional FLOPs for one forward pass
    """
    if model_config is None:
        model_config = DEFAULT_MODEL_CONFIG

    # Embedding dimension depends on component
    if component == "mlp_activation":
        d = model_config["intermediate_dim"]
    else:
        d = model_config["hidden_dim"]

    if num_layers is None:
        num_layers = model_config["num_layers"]

    r = reft_rank

    # Determine number of positions intervened (always on prompt only)
    if position in ["all", "alls"]:
        n_positions = prompt_length  # All prompt tokens
    elif position in ["f1+l1", "f1+s1"]:
        n_positions = 2  # First and last prompt token
    else:
        n_positions = 2  # Default assumption

    # FLOPs per intervention: project, transform, unproject
    # R^T @ h (d*r) + W @ Rh (r*r) + R @ result (r*d) ≈ 2*d*r + r*r
    flops_per_position = 2 * d * r + r * r

    # Total: per position * positions * layers
    # Note: ReFT typically has 2 interventions per layer (for f1+l1 style)
    # but with share_weights, same params are used
    interventions_per_layer = 1 if position in ["all", "alls"] else 2

    return flops_per_position * n_positions * num_layers * interventions_per_layer


def fetch_wandb_runs(project: str, entity: str = None, include_incomplete: bool = False) -> pd.DataFrame:
    """Fetch runs from wandb and return as DataFrame."""
    import wandb

    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    runs = api.runs(project_path)

    records = []
    for run in runs:
        if not include_incomplete and run.state != "finished":
            continue
        
        config = run.config
        summary = run.summary._json_dict
        
        # Determine method type
        use_lora = config.get("use_lora", False)
        disable_reft = config.get("disable_reft", False)
        full_finetune = config.get("full_finetune", False)
        intervention_type = config.get("intervention_type", "loreft")  # default is loreft

        if full_finetune:
            method = "full_finetune"
        elif use_lora and disable_reft:
            method = "lora"
        elif use_lora:
            method = "lora+reft"
        else:
            method = "reft"
        
        record = {
            "run_name": run.name,
            "run_id": run.id,
            # ReFT config
            "rank": config.get("rank"),
            "lr": config.get("lr"),
            "position": config.get("position"),
            "component": config.get("component", "block_output"),
            "share_weights": config.get("share_weights", False),
            # LoRA config
            "use_lora": use_lora,
            "disable_reft": disable_reft,
            "lora_rank": config.get("lora_rank"),
            "lora_modules": config.get("lora_modules"),
            # Method type
            "method": method,
            "intervention_type": intervention_type,
            "full_finetune": full_finetune,
            # Params
            "trainable_params": config.get("trainable_params") or summary.get("trainable_params"),
            # Handle both key formats: eval/nll and eval_nll
            "eval_loss": summary.get("eval/loss") or summary.get("eval_loss"),
            "eval_nll": summary.get("eval/nll") or summary.get("eval_nll"),
            "eval_perplexity": summary.get("eval/perplexity") or summary.get("eval_perplexity"),
            "train_loss": summary.get("train/loss") or summary.get("train_loss"),
        }
        records.append(record)
    
    return pd.DataFrame(records)


def fetch_10x_runs(project_10x: str, entity: str = None, include_incomplete: bool = False) -> dict:
    """
    Fetch 10x continuation runs and return a mapping from original run config to 10x run.

    Returns dict: {(method_type, rank, position, component): {"run_id": ..., "original_run_name": ...}}
    """
    import wandb

    api = wandb.Api()
    project_path = f"{entity}/{project_10x}" if entity else project_10x

    try:
        runs = api.runs(project_path)
    except Exception as e:
        print(f"Warning: Could not fetch 10x runs from {project_10x}: {e}")
        return {}

    runs_10x = {}
    for run in runs:
        if not include_incomplete and run.state != "finished":
            continue

        config = run.config

        # Parse the original run info from config or name
        # The 10x runs should have stored the original config
        rank = config.get("rank")
        position = config.get("position")
        component = config.get("component", "block_output")
        use_lora = config.get("use_lora", False)
        disable_reft = config.get("disable_reft", False)

        if use_lora and disable_reft:
            method_type = "lora"
        elif use_lora:
            method_type = "lora+reft"
        else:
            method_type = "reft"

        # For LoRA, use lora_rank
        if method_type == "lora":
            rank = config.get("lora_rank", rank)

        key = (method_type, rank, position, component)
        runs_10x[key] = {
            "run_id": run.id,
            "run_name": run.name,
            "config": config,
        }

    return runs_10x


def fetch_run_history(run_id: str, project: str, entity: str = None):
    """Fetch history for a single run."""
    import wandb
    
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    
    try:
        wandb_run = api.run(f"{project_path}/{run_id}")
        history = list(wandb_run.scan_history(keys=["_step", "train/global_step", "eval/nll"]))
        
        if not history:
            return None, None
        
        eval_points = []
        for h in history:
            nll = h.get("eval/nll")
            if nll is not None:
                step = h.get("train/global_step") or h.get("_step", 0)
                eval_points.append((step, nll))
        
        if not eval_points:
            return None, None
        
        steps = np.array([p[0] for p in eval_points])
        nll = np.array([p[1] for p in eval_points])
        
        mask = steps > 0
        return steps[mask], nll[mask]
        
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return None, None


def plot_position_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare different positions and methods."""
    reft_df = df[df["method"] == "reft"].copy()
    lora_df = df[df["method"] == "lora"].copy()
    
    positions = reft_df["position"].dropna().unique()
    if len(positions) < 1:
        print("No ReFT positions to compare")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: NLL vs Rank for each position (ReFT only)
    ax = axes[0]
    markers = ['o', 's', '^', 'D', 'v', '<']
    colors = plt.cm.tab10(np.linspace(0, 1, len(positions) + 1))
    
    for i, pos in enumerate(sorted(positions)):
        # Get best LR for each rank
        pos_data = reft_df[reft_df["position"] == pos]
        if pos_data.empty:
            continue
        best_per_rank = pos_data.groupby("rank")["eval_nll"].min().reset_index()
        best_per_rank = best_per_rank.sort_values("rank")
        ax.plot(best_per_rank["rank"], best_per_rank["eval_nll"],
                marker=markers[i % len(markers)], label=f"ReFT ({pos})", 
                color=colors[i], linewidth=2, markersize=8)
    
    # Add LoRA if available
    if not lora_df.empty:
        best_per_rank = lora_df.groupby("lora_rank")["eval_nll"].min().reset_index()
        best_per_rank = best_per_rank.sort_values("lora_rank")
        ax.plot(best_per_rank["lora_rank"], best_per_rank["eval_nll"],
                marker='p', label="LoRA", color='orange', linewidth=2, markersize=8)
    
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Best Eval NLL", fontsize=12)
    ax.set_title("Best NLL vs Rank (optimized over LR)", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Legacy vs Strict comparison
    ax = axes[1]
    legacy_strict_pairs = [("f1+l1", "f1+s1"), ("all", "alls")]
    
    has_comparison = False
    for legacy, strict in legacy_strict_pairs:
        if legacy in positions and strict in positions:
            legacy_data = reft_df[reft_df["position"] == legacy]
            strict_data = reft_df[reft_df["position"] == strict]
            
            # Merge on rank and lr
            merged = pd.merge(
                legacy_data[["rank", "lr", "eval_nll"]],
                strict_data[["rank", "lr", "eval_nll"]],
                on=["rank", "lr"],
                suffixes=("_legacy", "_strict")
            )
            
            if not merged.empty:
                has_comparison = True
                ax.scatter(merged["eval_nll_legacy"], merged["eval_nll_strict"], 
                          alpha=0.7, s=60, label=f"{legacy} vs {strict}")
    
    if has_comparison:
        # Add diagonal line
        all_vals = ax.get_xlim() + ax.get_ylim()
        lims = [min(all_vals), max(all_vals)]
        ax.plot(lims, lims, 'k--', alpha=0.5, label="y=x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_xlabel("NLL (legacy)", fontsize=12)
        ax.set_ylabel("NLL (strict)", fontsize=12)
        ax.set_title("Legacy vs Strict Position Comparison", fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No legacy/strict pairs to compare", 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / "position_comparison.png", dpi=150)
    plt.savefig(output_dir / "position_comparison.pdf")
    plt.close()
    print("Saved: position_comparison.png")


def plot_lora_results(df: pd.DataFrame, output_dir: Path):
    """Plot LoRA-specific results."""
    lora_df = df[df["method"] == "lora"].copy()
    
    if lora_df.empty:
        print("No LoRA data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: NLL vs LoRA rank
    ax = axes[0]
    lrs = sorted(lora_df["lr"].dropna().unique())
    for i, lr in enumerate(lrs):
        lr_data = lora_df[lora_df["lr"] == lr].sort_values("lora_rank")
        ax.plot(lr_data["lora_rank"], lr_data["eval_nll"], 
                marker='o', label=f"LR={lr:.0e}", color=COLORS[i % len(COLORS)],
                linewidth=2, markersize=8)
    
    ax.set_xlabel("LoRA Rank", fontsize=12)
    ax.set_ylabel("Eval NLL", fontsize=12)
    ax.set_title("LoRA: NLL vs Rank", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend(title="Learning Rate", loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: NLL vs LR for different ranks
    ax = axes[1]
    ranks = sorted(lora_df["lora_rank"].dropna().unique())
    for i, rank in enumerate(ranks):
        rank_data = lora_df[lora_df["lora_rank"] == rank].sort_values("lr")
        ax.plot(rank_data["lr"], rank_data["eval_nll"],
                marker='s', label=f"Rank={int(rank)}", color=COLORS[i % len(COLORS)],
                linewidth=2, markersize=8)
    
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Eval NLL", fontsize=12)
    ax.set_title("LoRA: NLL vs Learning Rate", fontsize=14)
    ax.set_xscale("log")
    ax.legend(title="Rank", loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "lora_results.png", dpi=150)
    plt.savefig(output_dir / "lora_results.pdf")
    plt.close()
    print("Saved: lora_results.png")


def plot_method_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare ReFT (by position and component) vs LoRA head-to-head."""
    reft_df = df[df["method"] == "reft"].copy()
    lora_df = df[df["method"] == "lora"].copy()

    if reft_df.empty:
        print("No ReFT data for comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for positions, different line styles for components
    position_colors = {
        'f1+l1': 'blue',
        'all': 'green',
        'f1+s1': 'red',
        'alls': 'purple',
    }
    component_styles = {
        'block_output': {'linestyle': '-', 'marker': 'o'},
        'mlp_activation': {'linestyle': '--', 'marker': 's'},
        'mlp_output': {'linestyle': ':', 'marker': '^'},
        'attention_output': {'linestyle': '-.', 'marker': 'D'},
    }

    # Plot ReFT by position and component
    for position in sorted(reft_df["position"].dropna().unique()):
        pos_df = reft_df[reft_df["position"] == position]

        for component in sorted(pos_df["component"].dropna().unique()):
            comp_df = pos_df[pos_df["component"] == component]

            # Get best NLL for each rank, keeping trainable_params
            best = comp_df.groupby("rank").agg({
                "eval_nll": "min",
                "trainable_params": "first",
            }).reset_index().sort_values("trainable_params")

            color = position_colors.get(position, 'gray')
            style = component_styles.get(component, {'linestyle': '-', 'marker': 'x'})

            # Label: include component only if not default
            if component == "block_output":
                label = f"ReFT ({position})"
            else:
                label = f"ReFT ({position}, {component})"

            ax.plot(best["trainable_params"], best["eval_nll"],
                    marker=style['marker'], linestyle=style['linestyle'],
                    label=label, color=color, linewidth=2, markersize=8)

    # Plot LoRA if available
    if not lora_df.empty:
        lora_best = lora_df.groupby("lora_rank").agg({
            "eval_nll": "min",
            "trainable_params": "first",
        }).reset_index().sort_values("trainable_params")

        ax.plot(lora_best["trainable_params"], lora_best["eval_nll"],
                marker='p', label="LoRA", color='orange', linewidth=2, markersize=8)

    ax.set_xlabel("Trainable Parameters", fontsize=12)
    ax.set_ylabel("Best Eval NLL", fontsize=12)
    ax.set_title("ReFT vs LoRA: Efficiency Comparison", fontsize=14)
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.png", dpi=150)
    plt.savefig(output_dir / "method_comparison.pdf")
    plt.close()
    print("Saved: method_comparison.png")


def plot_flops_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    prompt_length: int = 128,
    total_seq_len: int = 512
):
    """
    Compare ReFT (by position and component) vs LoRA with FLOPs on x-axis.

    Note: ReFT only intervenes on PROMPT tokens, while LoRA is applied to
    all tokens (prompt + response) during the forward pass.
    """
    reft_df = df[df["method"] == "reft"].copy()
    lora_df = df[df["method"] == "lora"].copy()

    if reft_df.empty:
        print("No ReFT data for FLOPs comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for positions, different line styles for components
    position_colors = {
        'f1+l1': 'blue',
        'all': 'green',
        'f1+s1': 'red',
        'alls': 'purple',
    }
    component_styles = {
        'block_output': {'linestyle': '-', 'marker': 'o'},
        'mlp_activation': {'linestyle': '--', 'marker': 's'},
        'mlp_output': {'linestyle': ':', 'marker': '^'},
        'attention_output': {'linestyle': '-.', 'marker': 'D'},
    }

    # Plot ReFT by position and component
    for position in sorted(reft_df["position"].dropna().unique()):
        pos_df = reft_df[reft_df["position"] == position]

        for component in sorted(pos_df["component"].dropna().unique()):
            comp_df = pos_df[pos_df["component"] == component]

            # Get best NLL for each rank
            best = comp_df.groupby("rank").agg({
                "eval_nll": "min",
            }).reset_index()

            # Compute FLOPs for each rank (ReFT uses prompt_length)
            best["flops"] = best["rank"].apply(
                lambda r: compute_reft_flops(int(r), prompt_length, position, component)
            )
            best = best.sort_values("flops")

            color = position_colors.get(position, 'gray')
            style = component_styles.get(component, {'linestyle': '-', 'marker': 'x'})

            # Label: include component only if not default
            if component == "block_output":
                label = f"ReFT ({position})"
            else:
                label = f"ReFT ({position}, {component})"

            ax.plot(best["flops"], best["eval_nll"],
                    marker=style['marker'], linestyle=style['linestyle'],
                    label=label, color=color, linewidth=2, markersize=8)

    # Plot LoRA if available (uses total_seq_len since LoRA applies to all tokens)
    if not lora_df.empty:
        lora_best = lora_df.groupby("lora_rank").agg({
            "eval_nll": "min",
        }).reset_index()

        # Compute FLOPs for each LoRA rank (LoRA uses full sequence)
        lora_best["flops"] = lora_best["lora_rank"].apply(
            lambda r: compute_lora_flops(int(r), total_seq_len)
        )
        lora_best = lora_best.sort_values("flops")

        ax.plot(lora_best["flops"], lora_best["eval_nll"],
                marker='p', label="LoRA", color='orange', linewidth=2, markersize=8)

    ax.set_xlabel("Additional FLOPs per Forward Pass", fontsize=12)
    ax.set_ylabel("Best Eval NLL", fontsize=12)
    ax.set_title(f"ReFT vs LoRA: FLOPs Efficiency\n(ReFT: prompt_len={prompt_length}, LoRA: seq_len={total_seq_len})", fontsize=14)
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "flops_comparison.png", dpi=150)
    plt.savefig(output_dir / "flops_comparison.pdf")
    plt.close()
    print("Saved: flops_comparison.png")


def get_best_runs(df: pd.DataFrame):
    """Get best LR run for each (method, rank, position, component, intervention_type)."""
    best_runs = []

    # ReFT runs
    reft_df = df[df["method"] == "reft"].copy()

    for intervention_type in reft_df["intervention_type"].dropna().unique():
        int_data = reft_df[reft_df["intervention_type"] == intervention_type]
        for position in int_data["position"].dropna().unique():
            pos_data = int_data[int_data["position"] == position]
            for component in pos_data["component"].dropna().unique():
                comp_data = pos_data[pos_data["component"] == component]
                for rank in comp_data["rank"].dropna().unique():
                    rank_data = comp_data[comp_data["rank"] == rank]
                    if not rank_data.empty:
                        best_idx = rank_data["eval_nll"].idxmin()
                        best_row = rank_data.loc[best_idx].to_dict()
                        # Build facet name: intervention_type (position, component)
                        int_name = intervention_type.upper() if intervention_type != "loreft" else "LoReFT"
                        if component == "block_output":
                            best_row["facet"] = f"{int_name} ({position})"
                        else:
                            best_row["facet"] = f"{int_name} ({position}, {component})"
                        best_row["rank_val"] = rank
                        best_row["method_type"] = "reft"
                        best_row["intervention_type_val"] = intervention_type
                        best_row["position_val"] = position
                        best_row["component_val"] = component
                        best_runs.append(best_row)

    # LoRA runs
    lora_df = df[df["method"] == "lora"].copy()
    for lora_rank in lora_df["lora_rank"].dropna().unique():
        rank_data = lora_df[lora_df["lora_rank"] == lora_rank]
        if not rank_data.empty:
            best_idx = rank_data["eval_nll"].idxmin()
            best_row = rank_data.loc[best_idx].to_dict()
            best_row["facet"] = "LoRA"
            best_row["rank_val"] = lora_rank
            best_row["method_type"] = "lora"
            best_row["position_val"] = None
            best_row["component_val"] = None
            best_runs.append(best_row)

    return best_runs


def fetch_scaling_coefficients(best_runs: list, project: str, entity: str = None,
                               project_10x: str = None, runs_10x: dict = None):
    """
    Fetch history and compute linear fit coefficients for each run.

    If project_10x and runs_10x are provided, uses 10x runs instead of original
    (10x runs are now independent full retraining, not continuations).
    """
    import wandb

    coefficients = []

    for run in best_runs:
        run_id = run["run_id"]
        method_type = run["method_type"]
        rank = run["rank_val"]
        position = run.get("position_val")
        component = run.get("component_val", "block_output")

        # Check if there's a 10x run (independent retrain with 10x epochs)
        use_10x = False
        if runs_10x:
            key = (method_type, rank, position, component)
            if key in runs_10x:
                run_id = runs_10x[key]["run_id"]
                use_10x = True

        # Fetch from appropriate project
        fetch_project = project_10x if (use_10x and project_10x) else project

        try:
            steps, nll = fetch_run_history(run_id, fetch_project, entity)

            if steps is None or len(steps) < 2:
                continue

            # Linear fit in log-space: NLL = slope * log10(step) + intercept
            log_steps = np.log10(steps)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_steps, nll)

            coefficients.append({
                "run_id": run_id,
                "facet": run["facet"],
                "method_type": method_type,
                "intervention_type_val": run.get("intervention_type_val", "loreft"),
                "position_val": position,
                "component_val": component,
                "rank": rank,
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "steps": steps,
                "nll": nll,
                "is_10x": use_10x,
                "max_step": steps.max(),
                "trainable_params": run.get("trainable_params"),
            })

        except Exception as e:
            print(f"Error fetching history for {run_id}: {e}")

    return coefficients


def plot_scaling_curves(coefficients: list, output_dir: Path, include_10x: bool = False):
    """
    Plot NLL curves for best LR per (method, rank, position), faceted by method+position.
    Each curve gets a linear fit in log-space with equation inscribed.
    
    If include_10x, indicates which runs include 10x continuation data.
    """
    if not coefficients:
        print("No coefficients for scaling curves")
        return
    
    # Group by facet
    facets = sorted(set(c["facet"] for c in coefficients))
    n_facets = len(facets)
    
    if n_facets == 0:
        print("No facets found")
        return
    
    # Create figure with subplots (shared y-axis for comparison)
    n_cols = min(3, n_facets)
    n_rows = (n_facets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False, sharey=True)
    axes = axes.flatten()
    
    # Hide unused axes
    for i in range(n_facets, len(axes)):
        axes[i].set_visible(False)
    
    # Plot each facet
    for facet_idx, facet in enumerate(facets):
        ax = axes[facet_idx]
        facet_data = [c for c in coefficients if c["facet"] == facet]
        
        # Sort by rank for consistent coloring
        facet_data = sorted(facet_data, key=lambda x: x["rank"])
        ranks = [c["rank"] for c in facet_data]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ranks)))
        
        for i, coef in enumerate(facet_data):
            steps = coef["steps"]
            nll = coef["nll"]
            slope = coef["slope"]
            intercept = coef["intercept"]
            rank = coef["rank"]
            is_10x = coef.get("is_10x", False)
            max_step = coef.get("max_step", steps.max())
            
            # Plot the curve (low opacity - background)
            ax.plot(steps, nll, color=colors[i], linewidth=1.5, alpha=0.2)
            
            # Plot fit line (high opacity - foreground)
            fit_steps = np.logspace(np.log10(steps.min()), np.log10(steps.max()), 100)
            fit_nll = slope * np.log10(fit_steps) + intercept
            
            # Label includes 10x indicator and max step if relevant
            if include_10x and is_10x:
                label = f"r={int(rank)} (10x, {int(max_step):,} steps)"
            else:
                label = f"r={int(rank)}"
            
            ax.plot(fit_steps, fit_nll, color=colors[i], linewidth=2.5, 
                    linestyle='-', alpha=0.9, label=label)
            
            # Add equation text along the line
            text_idx = int(len(fit_steps) * 0.6)
            text_x = fit_steps[text_idx]
            text_y = fit_nll[text_idx]
            
            # Calculate angle for text rotation
            angle = np.degrees(np.arctan(slope / (text_x * np.log(10))))
            
            eq_text = f"{slope:.3f}·log(x)+{intercept:.2f}"
            ax.annotate(eq_text, (text_x, text_y), fontsize=7, 
                       color=colors[i], alpha=0.9,
                       rotation=angle, rotation_mode='anchor',
                       ha='center', va='bottom')
        
        ax.set_xscale("log")
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Eval NLL", fontsize=11)
        ax.set_title(facet, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with different names based on 10x inclusion
    suffix = "_with_10x" if include_10x else ""
    plt.savefig(output_dir / f"scaling_curves{suffix}.png", dpi=150)
    plt.savefig(output_dir / f"scaling_curves{suffix}.pdf")
    plt.close()
    print(f"Saved: scaling_curves{suffix}.png")


def plot_scaling_by_rank(coefficients: list, output_dir: Path, include_10x: bool = False):
    """
    Plot NLL curves facetted by RANK, with each position/method as a line.
    This shows how different positions compare at each rank level.
    """
    if not coefficients:
        print("No coefficients for scaling by rank plot")
        return
    
    # Get unique ranks
    ranks = sorted(set(c["rank"] for c in coefficients))
    n_ranks = len(ranks)
    
    if n_ranks == 0:
        print("No ranks found")
        return
    
    # Create figure with subplots
    n_cols = min(4, n_ranks)
    n_rows = (n_ranks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3*n_rows), squeeze=False, sharey=True)
    axes = axes.flatten()
    
    # Hide unused axes
    for i in range(n_ranks, len(axes)):
        axes[i].set_visible(False)
    
    # Color map for different positions/methods
    facets = sorted(set(c["facet"] for c in coefficients))
    colors = plt.cm.tab10(np.linspace(0, 1, len(facets)))
    facet_colors = {f: colors[i] for i, f in enumerate(facets)}
    # Build markers dict dynamically based on facets found
    base_markers = {'f1+l1': 'o', 'all': 's', 'f1+s1': '^', 'alls': 'D'}
    markers = {'LoRA': 'p'}
    for facet in facets:
        if facet.startswith('ReFT'):
            # Extract position from facet name
            for pos, marker in base_markers.items():
                if pos in facet:
                    markers[facet] = marker
                    break
            if facet not in markers:
                markers[facet] = 'x'
    
    # Plot each rank
    for rank_idx, rank in enumerate(ranks):
        ax = axes[rank_idx]
        rank_data = [c for c in coefficients if c["rank"] == rank]
        
        for coef in rank_data:
            facet = coef["facet"]
            steps = coef["steps"]
            nll = coef["nll"]
            slope = coef["slope"]
            intercept = coef["intercept"]
            
            color = facet_colors.get(facet, 'gray')
            marker = markers.get(facet, 'o')
            
            # Plot actual data (low opacity)
            ax.plot(steps, nll, color=color, linewidth=1, alpha=0.15)
            
            # Plot fit line (high opacity)
            fit_steps = np.logspace(np.log10(steps.min()), np.log10(steps.max()), 100)
            fit_nll = slope * np.log10(fit_steps) + intercept
            ax.plot(fit_steps, fit_nll, color=color, linewidth=2.5, 
                    linestyle='-', alpha=0.9, label=facet, marker=marker,
                    markevery=[0, -1], markersize=6)
        
        ax.set_xscale("log")
        ax.set_xlabel("Step", fontsize=10)
        if rank_idx % n_cols == 0:
            ax.set_ylabel("Eval NLL", fontsize=10)
        ax.set_title(f"Rank {int(rank)}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Only show legend on first subplot
        if rank_idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    plt.tight_layout()
    
    suffix = "_with_10x" if include_10x else ""
    plt.savefig(output_dir / f"scaling_by_rank{suffix}.png", dpi=150)
    plt.savefig(output_dir / f"scaling_by_rank{suffix}.pdf")
    plt.close()
    print(f"Saved: scaling_by_rank{suffix}.png")


def plot_scaling_coefficients(coefficients: list, output_dir: Path):
    """
    Plot slope and intercept vs rank for each method/position.
    """
    if not coefficients:
        print("No coefficients for scaling coefficient plots")
        return
    
    # Convert to DataFrame for easier plotting
    coef_df = pd.DataFrame([{
        "facet": c["facet"],
        "method_type": c["method_type"],
        "position": c["position_val"],
        "rank": c["rank"],
        "slope": c["slope"],
        "intercept": c["intercept"],
        "r_squared": c["r_squared"],
    } for c in coefficients])
    
    facets = sorted(coef_df["facet"].unique())
    markers = ['o', 's', '^', 'D', 'v', '<', 'p', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, len(facets)))
    
    # Plot 1: Slope vs Rank
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    for i, facet in enumerate(facets):
        facet_data = coef_df[coef_df["facet"] == facet].sort_values("rank")
        ax.plot(facet_data["rank"], facet_data["slope"],
                marker=markers[i % len(markers)], color=colors[i],
                linewidth=2, markersize=10, label=facet)
    
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Slope (rate of NLL decrease)", fontsize=12)
    ax.set_title("Scaling Slope vs Rank", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Intercept vs Rank
    ax = axes[1]
    for i, facet in enumerate(facets):
        facet_data = coef_df[coef_df["facet"] == facet].sort_values("rank")
        ax.plot(facet_data["rank"], facet_data["intercept"],
                marker=markers[i % len(markers)], color=colors[i],
                linewidth=2, markersize=10, label=facet)
    
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Intercept (initial NLL at step=1)", fontsize=12)
    ax.set_title("Scaling Intercept vs Rank", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_coefficients.png", dpi=150)
    plt.savefig(output_dir / "scaling_coefficients.pdf")
    plt.close()
    print("Saved: scaling_coefficients.png")


def plot_scaling_coefficients_with_10x(coefficients: list, output_dir: Path):
    """
    Plot slope and intercept vs rank for runs with 10x continuation data.
    Also generates a table with max_step info.
    """
    if not coefficients:
        print("No coefficients for 10x scaling coefficient plots")
        return
    
    # Convert to DataFrame for easier plotting
    coef_df = pd.DataFrame([{
        "facet": c["facet"],
        "method_type": c["method_type"],
        "position": c["position_val"],
        "rank": c["rank"],
        "slope": c["slope"],
        "intercept": c["intercept"],
        "r_squared": c["r_squared"],
        "is_10x": c.get("is_10x", False),
        "max_step": c.get("max_step", 0),
    } for c in coefficients])
    
    facets = sorted(coef_df["facet"].unique())
    markers = ['o', 's', '^', 'D', 'v', '<', 'p', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, len(facets)))
    
    # Plot: Slope vs Rank and Intercept vs Rank with 10x data
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    for i, facet in enumerate(facets):
        facet_data = coef_df[coef_df["facet"] == facet].sort_values("rank")
        ax.plot(facet_data["rank"], facet_data["slope"],
                marker=markers[i % len(markers)], color=colors[i],
                linewidth=2, markersize=10, label=facet)
    
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Slope (rate of NLL decrease)", fontsize=12)
    ax.set_title("Scaling Slope vs Rank (with 10x data)", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax = axes[1]
    for i, facet in enumerate(facets):
        facet_data = coef_df[coef_df["facet"] == facet].sort_values("rank")
        ax.plot(facet_data["rank"], facet_data["intercept"],
                marker=markers[i % len(markers)], color=colors[i],
                linewidth=2, markersize=10, label=facet)
    
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Intercept (initial NLL at step=1)", fontsize=12)
    ax.set_title("Scaling Intercept vs Rank (with 10x data)", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_coefficients_with_10x.png", dpi=150)
    plt.savefig(output_dir / "scaling_coefficients_with_10x.pdf")
    plt.close()
    print("Saved: scaling_coefficients_with_10x.png")


def bootstrap_linear_fit(steps, nll, n_bootstrap=100, target_nll=None):
    """
    Bootstrap the linear fit to get confidence intervals on predicted steps.

    Returns: (mean_k, std_k) where k = steps_to_target / baseline_steps
    """
    log_steps = np.log10(steps)
    n_points = len(steps)

    slopes = []
    intercepts = []

    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = rng.choice(n_points, size=n_points, replace=True)
        boot_log_steps = log_steps[idx]
        boot_nll = nll[idx]

        # Fit
        slope, intercept, _, _, _ = stats.linregress(boot_log_steps, boot_nll)
        slopes.append(slope)
        intercepts.append(intercept)

    return np.array(slopes), np.array(intercepts)


def compute_sample_efficiency(coefficients: list, output_dir: Path, n_bootstrap=100):
    """
    Compare sample efficiency between methods by estimating steps to reach target NLL.

    Uses LoRA rank=1's end NLL as the single target for all methods across all ranks.
    For each method, computes steps needed using its fitted curve:
        steps = 10^((target_NLL - intercept) / slope)

    Uses bootstrap resampling to estimate uncertainty in k.
    """
    if not coefficients:
        print("No coefficients for sample efficiency comparison")
        return

    # Convert to DataFrame, keeping raw data for bootstrap
    coef_df = pd.DataFrame([{
        "facet": c["facet"],
        "method_type": c["method_type"],
        "intervention_type": c.get("intervention_type_val", "loreft"),
        "position": c.get("position_val"),
        "component": c.get("component_val", "block_output"),
        "rank": c["rank"],
        "slope": c["slope"],
        "intercept": c["intercept"],
        "r_squared": c["r_squared"],
        "max_step": c.get("max_step", 0),
        "steps_raw": c.get("steps"),
        "nll_raw": c.get("nll"),
        "trainable_params": c.get("trainable_params"),
    } for c in coefficients])

    # Get LoRA rank=1 as baseline
    lora_df = coef_df[coef_df["method_type"] == "lora"].copy()

    if lora_df.empty:
        print("No LoRA runs found, cannot compute sample efficiency")
        return

    lora_r1 = lora_df[lora_df["rank"] == 1]
    if lora_r1.empty:
        min_rank = lora_df["rank"].min()
        lora_r1 = lora_df[lora_df["rank"] == min_rank]
        print(f"No LoRA rank=1 found, using rank={int(min_rank)} as baseline")

    baseline = lora_r1.iloc[0]
    baseline_max_step = baseline["max_step"]

    if baseline_max_step <= 0:
        print("Baseline LoRA has no valid max_step")
        return

    # Bootstrap the baseline to get target_nll distribution
    baseline_steps_raw = baseline.get("steps_raw")
    baseline_nll_raw = baseline.get("nll_raw")

    if baseline_steps_raw is None or baseline_nll_raw is None:
        print("No raw data for bootstrap, using point estimates only")
        use_bootstrap = False
        target_nll = baseline["slope"] * np.log10(baseline_max_step) + baseline["intercept"]
        lora_r1_steps = baseline_max_step
    else:
        use_bootstrap = True
        baseline_slopes, baseline_intercepts = bootstrap_linear_fit(
            baseline_steps_raw, baseline_nll_raw, n_bootstrap
        )
        # Target NLL distribution (at baseline max_step)
        target_nlls = baseline_slopes * np.log10(baseline_max_step) + baseline_intercepts
        target_nll = np.mean(target_nlls)
        lora_r1_steps = baseline_max_step

    # Compute steps needed for each method at each rank to reach target NLL
    comparisons = []

    for _, row in coef_df.iterrows():
        slope = row["slope"]
        intercept = row["intercept"]
        steps_raw = row.get("steps_raw")
        nll_raw = row.get("nll_raw")

        # Point estimate: steps = 10^((target_nll - intercept) / slope)
        if slope != 0:
            log_steps = (target_nll - intercept) / slope
            steps_needed = 10 ** log_steps
            k = steps_needed / lora_r1_steps
        else:
            steps_needed = np.nan
            k = np.nan

        # Bootstrap for error bars
        k_std = np.nan
        if use_bootstrap and steps_raw is not None and nll_raw is not None:
            try:
                method_slopes, method_intercepts = bootstrap_linear_fit(
                    steps_raw, nll_raw, n_bootstrap
                )
                # Compute k for each bootstrap sample
                # Use both target_nll uncertainty and method fit uncertainty
                k_samples = []
                for i in range(n_bootstrap):
                    t_nll = target_nlls[i] if use_bootstrap else target_nll
                    m_slope = method_slopes[i]
                    m_intercept = method_intercepts[i]
                    if m_slope != 0:
                        log_s = (t_nll - m_intercept) / m_slope
                        k_samples.append((10 ** log_s) / lora_r1_steps)
                if k_samples:
                    k_std = np.std(k_samples)
            except Exception:
                pass

        # Check if we're extrapolating (target below min observed NLL)
        min_observed_nll = np.min(nll_raw) if nll_raw is not None else np.nan
        extrapolating = target_nll < min_observed_nll if not np.isnan(min_observed_nll) else False

        comparisons.append({
            "rank": row["rank"],
            "method": row["facet"],
            "steps_needed": steps_needed,
            "data_multiplier": k,
            "k_std": k_std,
            "slope": slope,
            "min_nll": min_observed_nll,
            "extrapolating": extrapolating,
            "trainable_params": row.get("trainable_params"),
        })

    comp_df = pd.DataFrame(comparisons)

    # Plot: Data multiplier vs rank for each method with error bars
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = sorted(comp_df["method"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        method_data = comp_df[comp_df["method"] == method].sort_values("rank")
        ranks = method_data["rank"].values
        k_vals = method_data["data_multiplier"].values
        k_errs = method_data["k_std"].values

        # Plot with error bars if available
        if not np.all(np.isnan(k_errs)):
            ax.errorbar(ranks, k_vals, yerr=k_errs,
                        marker='o', color=colors[i], linewidth=2, markersize=8,
                        capsize=3, capthick=1.5, label=method)
        else:
            ax.plot(ranks, k_vals,
                    marker='o', color=colors[i], linewidth=2, markersize=8,
                    label=method)

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Data Multiplier (k)", fontsize=12)
    ax.set_title(f"Sample Efficiency: Steps to Reach LoRA r=1 End NLL ({target_nll:.4f})\n"
                 f"(k = steps / {int(lora_r1_steps):,})", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "sample_efficiency.png", dpi=150)
    plt.savefig(output_dir / "sample_efficiency.pdf")
    plt.close()
    print("Saved: sample_efficiency.png")

    # Plot 2: Data multiplier vs trainable params
    # Group by method type (not position) for cleaner visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define method groups and their styles
    method_styles = {
        "LoRA": {"color": "tab:orange", "marker": "s"},
        "LoReFT": {"color": "tab:blue", "marker": "o"},
        "DiReFT": {"color": "tab:green", "marker": "^"},
        "NoDiReFT": {"color": "tab:red", "marker": "D"},
    }

    for method in methods:
        method_data = comp_df[comp_df["method"] == method].copy()
        method_data = method_data.dropna(subset=["trainable_params"])

        if method_data.empty:
            continue

        method_data = method_data.sort_values("trainable_params")
        params = method_data["trainable_params"].values
        k_vals = method_data["data_multiplier"].values
        k_errs = method_data["k_std"].values

        # Determine style based on method name
        style = {"color": "gray", "marker": "x"}
        for key, s in method_styles.items():
            if key in method:
                style = s
                break

        if not np.all(np.isnan(k_errs)):
            ax.errorbar(params, k_vals, yerr=k_errs,
                        marker=style["marker"], color=style["color"],
                        linewidth=2, markersize=8, capsize=3, capthick=1.5,
                        label=method, alpha=0.8)
        else:
            ax.plot(params, k_vals,
                    marker=style["marker"], color=style["color"],
                    linewidth=2, markersize=8, label=method, alpha=0.8)

    ax.set_xlabel("Trainable Parameters", fontsize=12)
    ax.set_ylabel("Data Multiplier (k)", fontsize=12)
    ax.set_title(f"Sample Efficiency vs Parameters\n"
                 f"(target NLL = {target_nll:.4f}, baseline = LoRA r=1)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "sample_efficiency_vs_params.png", dpi=150)
    plt.savefig(output_dir / "sample_efficiency_vs_params.pdf")
    plt.close()
    print("Saved: sample_efficiency_vs_params.png")

    # Plot 3: Residuals of linear fits to check for systematic deviation
    # Collect all residuals
    residual_data = []
    for _, row in coef_df.iterrows():
        steps_raw = row.get("steps_raw")
        nll_raw = row.get("nll_raw")
        slope = row["slope"]
        intercept = row["intercept"]

        if steps_raw is None or nll_raw is None:
            continue

        log_steps = np.log10(steps_raw)
        predicted = slope * log_steps + intercept
        residuals = nll_raw - predicted

        for i in range(len(steps_raw)):
            residual_data.append({
                "method": row["facet"],
                "method_type": row["method_type"],
                "rank": row["rank"],
                "log_step": log_steps[i],
                "step": steps_raw[i],
                "residual": residuals[i],
                "nll": nll_raw[i],
                "predicted": predicted[i],
            })

    if residual_data:
        resid_df = pd.DataFrame(residual_data)

        # Plot residuals vs log(step), colored by method type
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Residuals vs log(step) by method type
        ax = axes[0]
        for method_type, color in [("lora", "tab:orange"), ("reft", "tab:blue")]:
            type_data = resid_df[resid_df["method_type"] == method_type]
            if not type_data.empty:
                ax.scatter(type_data["log_step"], type_data["residual"],
                          alpha=0.3, s=10, color=color, label=method_type.upper())

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("log₁₀(step)", fontsize=12)
        ax.set_ylabel("Residual (actual - predicted NLL)", fontsize=12)
        ax.set_title("Residuals vs Training Step", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: Residuals vs predicted NLL (to check heteroscedasticity)
        ax = axes[1]
        for method_type, color in [("lora", "tab:orange"), ("reft", "tab:blue")]:
            type_data = resid_df[resid_df["method_type"] == method_type]
            if not type_data.empty:
                ax.scatter(type_data["predicted"], type_data["residual"],
                          alpha=0.3, s=10, color=color, label=method_type.upper())

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("Predicted NLL", fontsize=12)
        ax.set_ylabel("Residual (actual - predicted NLL)", fontsize=12)
        ax.set_title("Residuals vs Predicted Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "fit_residuals.png", dpi=150)
        plt.savefig(output_dir / "fit_residuals.pdf")
        plt.close()
        print("Saved: fit_residuals.png")

        # Also plot residuals faceted by rank to see if pattern varies
        ranks = sorted(resid_df["rank"].unique())
        n_ranks = len(ranks)
        if n_ranks > 0:
            n_cols = min(4, n_ranks)
            n_rows = (n_ranks + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows),
                                     squeeze=False, sharey=True)
            axes = axes.flatten()

            for i in range(n_ranks, len(axes)):
                axes[i].set_visible(False)

            for idx, rank in enumerate(ranks):
                ax = axes[idx]
                rank_data = resid_df[resid_df["rank"] == rank]

                for method_type, color in [("lora", "tab:orange"), ("reft", "tab:blue")]:
                    type_data = rank_data[rank_data["method_type"] == method_type]
                    if not type_data.empty:
                        ax.scatter(type_data["log_step"], type_data["residual"],
                                  alpha=0.4, s=15, color=color, label=method_type.upper())

                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax.set_xlabel("log₁₀(step)", fontsize=10)
                if idx % n_cols == 0:
                    ax.set_ylabel("Residual", fontsize=10)
                ax.set_title(f"Rank {int(rank)}", fontsize=11)
                ax.grid(True, alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=8)

            plt.suptitle("Residuals by Rank (positive = model underpredicts NLL)", fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / "fit_residuals_by_rank.png", dpi=150)
            plt.savefig(output_dir / "fit_residuals_by_rank.pdf")
            plt.close()
            print("Saved: fit_residuals_by_rank.png")

    # Print nicely formatted console table, separated by rank
    print(f"\n{'='*90}")
    print(f"SAMPLE EFFICIENCY: Steps to reach NLL = {target_nll:.4f}")
    print(f"{'='*90}")
    print(f"Baseline: LoRA r=1 reaches target in {int(lora_r1_steps):,} steps")
    print(f"k = (steps needed) / {int(lora_r1_steps):,}  [± 1 std from bootstrap]")
    print(f"Ex = extrapolating (target NLL below min observed, high uncertainty)")
    print()

    for rank in sorted(comp_df["rank"].unique()):
        rank_data = comp_df[comp_df["rank"] == rank].copy()
        rank_data = rank_data.sort_values("data_multiplier")

        print(f"{'─'*90}")
        print(f"RANK {int(rank)}")
        print(f"{'─'*90}")
        print(f"{'Method':<32} {'Slope':>8} {'Min NLL':>8} {'Steps':>10} {'k':>15} {'Ex':>3}")
        print(f"{'-'*32} {'-'*8} {'-'*8} {'-'*10} {'-'*15} {'-'*3}")

        for _, row in rank_data.iterrows():
            method = row["method"][:32]
            steps = row["steps_needed"]
            k = row["data_multiplier"]
            k_std = row["k_std"]
            slope = row["slope"]
            min_nll = row["min_nll"]
            extrap = "!" if row["extrapolating"] else ""

            slope_str = f"{slope:.4f}" if not np.isnan(slope) else "N/A"
            min_nll_str = f"{min_nll:.4f}" if not np.isnan(min_nll) else "N/A"

            if not np.isnan(steps) and steps > 0:
                steps_str = f"{int(steps):,}"
                if not np.isnan(k_std):
                    k_str = f"{k:.2f}x ± {k_std:.2f}"
                else:
                    k_str = f"{k:.2f}x"
            else:
                steps_str = "N/A"
                k_str = "N/A"

            print(f"{method:<32} {slope_str:>8} {min_nll_str:>8} {steps_str:>10} {k_str:>15} {extrap:>3}")

        print()

    # Summary: average k across ranks for each method
    print(f"{'='*90}")
    print("SUMMARY (average k across all ranks)")
    print(f"{'='*90}")
    summary = comp_df.groupby("method").agg({
        "data_multiplier": "mean",
        "k_std": "mean",
    }).sort_values("data_multiplier")

    print(f"{'Method':<40} {'Avg k':>15}")
    print(f"{'-'*40} {'-'*15}")
    for method, row in summary.iterrows():
        avg_k = row["data_multiplier"]
        avg_std = row["k_std"]
        if not np.isnan(avg_k):
            if not np.isnan(avg_std):
                k_str = f"{avg_k:.2f}x ± {avg_std:.2f}"
            else:
                k_str = f"{avg_k:.2f}x"
        else:
            k_str = "N/A"
        print(f"{method:<40} {k_str:>15}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Plot LoReFT sweep results")
    parser.add_argument("--project", type=str, default="loreft-regret",
                        help="Wandb project name")
    parser.add_argument("--project-10x", type=str, default=None,
                        help="Wandb project name for 10x continuation runs")
    parser.add_argument("--entity", type=str, default=None,
                        help="Wandb entity (username or team)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Load from CSV instead of wandb")
    parser.add_argument("--output", type=str, default="./plots",
                        help="Output directory for plots")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save fetched data to CSV")
    parser.add_argument("--curves", action="store_true",
                        help="Plot scaling curves with linear fits (slow, fetches history)")
    parser.add_argument("--include-incomplete", action="store_true",
                        help="Include runs that haven't finished (running, crashed, etc.)")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if args.csv:
        print(f"Loading from {args.csv}")
        df = pd.read_csv(args.csv)
    else:
        print(f"Fetching runs from wandb project: {args.project}")
        if args.include_incomplete:
            print("Including incomplete runs")
        df = fetch_wandb_runs(args.project, args.entity, args.include_incomplete)
        
        if args.save_csv:
            csv_path = output_dir / "sweep_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved data to {csv_path}")
    
    print(f"Loaded {len(df)} runs")
    print(f"Methods: {df['method'].unique()}")
    print(f"Intervention types: {df['intervention_type'].dropna().unique()}")
    print(f"Positions: {df['position'].dropna().unique()}")
    print(f"Components: {df['component'].dropna().unique()}")
    print(f"ReFT Ranks: {sorted(df['rank'].dropna().unique())}")
    print(f"LoRA Ranks: {sorted(df['lora_rank'].dropna().unique())}")
    print(f"LRs: {sorted(df['lr'].dropna().unique())}")
    
    # Fetch 10x continuation runs if specified
    runs_10x = {}
    if args.project_10x and not args.csv:
        print(f"\nFetching 10x continuation runs from: {args.project_10x}")
        runs_10x = fetch_10x_runs(args.project_10x, args.entity, args.include_incomplete)
        print(f"Found {len(runs_10x)} continuation runs")
    
    # Generate plots (only the useful ones)
    plot_lora_results(df, output_dir)
    plot_method_comparison(df, output_dir)
    plot_flops_comparison(df, output_dir)
    plot_position_comparison(df, output_dir)
    
    # Scaling curves with linear fits (requires fetching history)
    if args.curves and not args.csv:
        print("\nFetching run histories for scaling analysis...")
        best_runs = get_best_runs(df)
        
        # First plot without 10x
        coefficients = fetch_scaling_coefficients(best_runs, args.project, args.entity)
        plot_scaling_curves(coefficients, output_dir, include_10x=False)
        plot_scaling_by_rank(coefficients, output_dir, include_10x=False)
        plot_scaling_coefficients(coefficients, output_dir)
        compute_sample_efficiency(coefficients, output_dir)
        
        # If 10x data available, also plot combined curves
        if runs_10x:
            print("\nFetching combined histories with 10x continuation...")
            coefficients_10x = fetch_scaling_coefficients(
                best_runs, args.project, args.entity,
                project_10x=args.project_10x, runs_10x=runs_10x
            )
            plot_scaling_curves(coefficients_10x, output_dir, include_10x=True)
            plot_scaling_by_rank(coefficients_10x, output_dir, include_10x=True)
            
            # Also save updated coefficients table with 10x data
            plot_scaling_coefficients_with_10x(coefficients_10x, output_dir)
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
