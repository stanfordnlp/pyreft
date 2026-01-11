#!/usr/bin/env python3
"""
Plot results from the LoReFT sweep experiments.

Usage:
    python plot_sweep.py --project loreft-regret
    python plot_sweep.py --project loreft-regret --project-10x loreft-regret-10x --curves
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


def fetch_wandb_runs(project: str, entity: str = None) -> pd.DataFrame:
    """Fetch runs from wandb and return as DataFrame."""
    import wandb
    
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    runs = api.runs(project_path)
    
    records = []
    for run in runs:
        if run.state != "finished":
            continue
        
        config = run.config
        summary = run.summary._json_dict
        
        # Determine method type
        use_lora = config.get("use_lora", False)
        disable_reft = config.get("disable_reft", False)
        full_finetune = config.get("full_finetune", False)
        
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
            "share_weights": config.get("share_weights", False),
            # LoRA config
            "use_lora": use_lora,
            "disable_reft": disable_reft,
            "lora_rank": config.get("lora_rank"),
            "lora_modules": config.get("lora_modules"),
            # Method type
            "method": method,
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


def fetch_10x_runs(project_10x: str, entity: str = None) -> dict:
    """
    Fetch 10x continuation runs and return a mapping from original run config to 10x run.
    
    Returns dict: {(method_type, rank, position): {"run_id": ..., "original_run_name": ...}}
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
        if run.state != "finished":
            continue
        
        config = run.config
        
        # Parse the original run info from config or name
        # The 10x runs should have stored the original config
        rank = config.get("rank")
        position = config.get("position")
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
        
        key = (method_type, rank, position)
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
    """Compare ReFT vs LoRA head-to-head."""
    reft_df = df[df["method"] == "reft"].copy()
    lora_df = df[df["method"] == "lora"].copy()
    
    if reft_df.empty or lora_df.empty:
        print("Need both ReFT and LoRA data for comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get best NLL for each trainable params level
    reft_best = reft_df.groupby("trainable_params").agg({
        "eval_nll": "min",
        "rank": "first",
        "position": "first",
    }).reset_index().sort_values("trainable_params")
    
    lora_best = lora_df.groupby("trainable_params").agg({
        "eval_nll": "min",
        "lora_rank": "first",
    }).reset_index().sort_values("trainable_params")
    
    ax.plot(reft_best["trainable_params"], reft_best["eval_nll"],
            marker='o', label="ReFT (best)", color='blue', linewidth=2, markersize=8)
    ax.plot(lora_best["trainable_params"], lora_best["eval_nll"],
            marker='s', label="LoRA", color='orange', linewidth=2, markersize=8)
    
    ax.set_xlabel("Trainable Parameters", fontsize=12)
    ax.set_ylabel("Best Eval NLL", fontsize=12)
    ax.set_title("ReFT vs LoRA: Efficiency Comparison", fontsize=14)
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.png", dpi=150)
    plt.savefig(output_dir / "method_comparison.pdf")
    plt.close()
    print("Saved: method_comparison.png")


def get_best_runs(df: pd.DataFrame):
    """Get best LR run for each (method, rank, position)."""
    best_runs = []
    
    # ReFT runs
    reft_df = df[df["method"] == "reft"].copy()
    
    for position in reft_df["position"].dropna().unique():
        pos_data = reft_df[reft_df["position"] == position]
        for rank in pos_data["rank"].dropna().unique():
            rank_data = pos_data[pos_data["rank"] == rank]
            if not rank_data.empty:
                best_idx = rank_data["eval_nll"].idxmin()
                best_row = rank_data.loc[best_idx].to_dict()
                best_row["facet"] = f"ReFT ({position})"
                best_row["rank_val"] = rank
                best_row["method_type"] = "reft"
                best_row["position_val"] = position
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
        
        # Check if there's a 10x run (independent retrain with 10x epochs)
        use_10x = False
        if runs_10x:
            key = (method_type, rank, position)
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
                "position_val": position,
                "rank": rank,
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "steps": steps,
                "nll": nll,
                "is_10x": use_10x,
                "max_step": steps.max(),
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
    
    # Plot 3: Coefficient table/summary
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create table data
    table_data = []
    for _, row in coef_df.sort_values(["facet", "rank"]).iterrows():
        table_data.append([
            row["facet"],
            f"{int(row['rank'])}",
            f"{row['slope']:.4f}",
            f"{row['intercept']:.3f}",
            f"{row['r_squared']:.4f}",
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Method/Position", "Rank", "Slope", "Intercept", "R²"],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
    
    plt.title("Scaling Law Coefficients: NLL = slope·log₁₀(step) + intercept", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_coefficients_table.png", dpi=150)
    plt.savefig(output_dir / "scaling_coefficients_table.pdf")
    plt.close()
    print("Saved: scaling_coefficients_table.png")


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
    
    # Table with 10x info
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    table_data = []
    for _, row in coef_df.sort_values(["facet", "rank"]).iterrows():
        steps_str = f"{int(row['max_step']):,}"
        table_data.append([
            row["facet"],
            f"{int(row['rank'])}",
            f"{row['slope']:.4f}",
            f"{row['intercept']:.3f}",
            f"{row['r_squared']:.4f}",
            steps_str,
            "✓" if row["is_10x"] else "",
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Method/Position", "Rank", "Slope", "Intercept", "R²", "Max Steps", "10x"],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors, highlight 10x rows
    for i in range(1, len(table_data) + 1):
        for j in range(7):
            if table_data[i-1][6] == "✓":  # is 10x run
                table[(i, j)].set_facecolor('#E2EFDA')  # light green
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
    
    plt.title("Scaling Law Coefficients (with 10x continuation data)\nNLL = slope·log₁₀(step) + intercept", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_coefficients_table_with_10x.png", dpi=150)
    plt.savefig(output_dir / "scaling_coefficients_table_with_10x.pdf")
    plt.close()
    print("Saved: scaling_coefficients_table_with_10x.png")


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
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if args.csv:
        print(f"Loading from {args.csv}")
        df = pd.read_csv(args.csv)
    else:
        print(f"Fetching runs from wandb project: {args.project}")
        df = fetch_wandb_runs(args.project, args.entity)
        
        if args.save_csv:
            csv_path = output_dir / "sweep_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved data to {csv_path}")
    
    print(f"Loaded {len(df)} runs")
    print(f"Methods: {df['method'].unique()}")
    print(f"Positions: {df['position'].dropna().unique()}")
    print(f"ReFT Ranks: {sorted(df['rank'].dropna().unique())}")
    print(f"LoRA Ranks: {sorted(df['lora_rank'].dropna().unique())}")
    print(f"LRs: {sorted(df['lr'].dropna().unique())}")
    
    # Fetch 10x continuation runs if specified
    runs_10x = {}
    if args.project_10x and not args.csv:
        print(f"\nFetching 10x continuation runs from: {args.project_10x}")
        runs_10x = fetch_10x_runs(args.project_10x, args.entity)
        print(f"Found {len(runs_10x)} continuation runs")
    
    # Generate plots (only the useful ones)
    plot_lora_results(df, output_dir)
    plot_method_comparison(df, output_dir)
    plot_position_comparison(df, output_dir)
    
    # Scaling curves with linear fits (requires fetching history)
    if args.curves and not args.csv:
        print("\nFetching run histories for scaling analysis...")
        best_runs = get_best_runs(df)
        
        # First plot without 10x
        coefficients = fetch_scaling_coefficients(best_runs, args.project, args.entity)
        plot_scaling_curves(coefficients, output_dir, include_10x=False)
        plot_scaling_coefficients(coefficients, output_dir)
        
        # If 10x data available, also plot combined curves
        if runs_10x:
            print("\nFetching combined histories with 10x continuation...")
            coefficients_10x = fetch_scaling_coefficients(
                best_runs, args.project, args.entity,
                project_10x=args.project_10x, runs_10x=runs_10x
            )
            plot_scaling_curves(coefficients_10x, output_dir, include_10x=True)
            
            # Also save updated coefficients table with 10x data
            plot_scaling_coefficients_with_10x(coefficients_10x, output_dir)
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
