#!/usr/bin/env python3
"""
Plot results from the LoReFT sweep experiments.

Usage:
    python plot_sweep.py --project loreft-regret
    python plot_sweep.py --project loreft-regret --output plots/
    python plot_sweep.py --csv results.csv  # Use cached CSV instead of wandb
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.cm.viridis(np.linspace(0, 0.9, 7))


def has_complete_hparams(row, method="reft"):
    """Check if a row has all required hyperparameters."""
    if method == "reft":
        return pd.notna(row.get("rank")) and pd.notna(row.get("lr")) and pd.notna(row.get("position"))
    elif method == "lora":
        return pd.notna(row.get("lora_rank")) and pd.notna(row.get("lr"))
    elif method == "full_finetune":
        return pd.notna(row.get("lr"))
    return True


def plot_with_completeness(ax, x, y, complete_mask, marker='o', color=None, label=None, 
                           linewidth=2, markersize=8, **kwargs):
    """Plot with filled markers for complete data, hollow for incomplete."""
    x = np.array(x)
    y = np.array(y)
    complete_mask = np.array(complete_mask)
    
    # Plot the line through all points
    sort_idx = np.argsort(x)
    ax.plot(x[sort_idx], y[sort_idx], color=color, linewidth=linewidth, label=label, **kwargs)
    
    # Plot filled markers for complete data
    if complete_mask.any():
        ax.scatter(x[complete_mask], y[complete_mask], marker=marker, c=color, 
                   s=markersize**2, zorder=5)
    
    # Plot hollow markers for incomplete data
    if (~complete_mask).any():
        ax.scatter(x[~complete_mask], y[~complete_mask], marker=marker, 
                   facecolors='none', edgecolors=color, s=markersize**2, 
                   linewidths=1.5, zorder=5)


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


def fetch_run_history(project: str, run_id: str, entity: str = None) -> pd.DataFrame:
    """Fetch training history for a single run."""
    import wandb
    
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    run = api.run(f"{project_path}/{run_id}")
    
    history = run.history(keys=["_step", "eval/nll", "eval/loss", "train/loss"])
    return history


def plot_nll_curves(df: pd.DataFrame, output_dir: Path, project: str, entity: str = None,
                    position: str = "f1+l1", group_by: str = "rank"):
    """Plot NLL over training steps for different configurations."""
    import wandb
    
    subset = df[(df["position"] == position) & (df["method"] == "reft")].copy()
    if subset.empty:
        print(f"No ReFT data for position={position}")
        return
    
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if group_by == "rank":
        # Plot curves grouped by rank (best LR for each rank)
        best_runs = subset.loc[subset.groupby("rank")["eval_nll"].idxmin()]
        groups = sorted(best_runs["rank"].unique())
        label_fmt = "rank={:.0f}"
    else:
        # Plot curves grouped by LR (best rank for each LR)
        best_runs = subset.loc[subset.groupby("lr")["eval_nll"].idxmin()]
        groups = sorted(best_runs["lr"].unique())
        label_fmt = "LR={:.0e}"
    
    for i, group_val in enumerate(groups):
        if group_by == "rank":
            run_row = best_runs[best_runs["rank"] == group_val].iloc[0]
        else:
            run_row = best_runs[best_runs["lr"] == group_val].iloc[0]
        
        run_id = run_row["run_id"]
        
        try:
            run = api.run(f"{project_path}/{run_id}")
            history = run.history(keys=["_step", "eval/nll"])
            history = history.dropna(subset=["eval/nll"])
            
            if not history.empty:
                ax.plot(history["_step"], history["eval/nll"],
                        label=label_fmt.format(group_val),
                        color=COLORS[i % len(COLORS)],
                        linewidth=2, alpha=0.8)
        except Exception as e:
            print(f"Error fetching history for run {run_id}: {e}")
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Eval NLL", fontsize=12)
    ax.set_title(f"NLL over Training (position={position}, best {group_by}s)", fontsize=14)
    ax.set_xscale("log")
    ax.legend(title=group_by.capitalize(), loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"nll_curves_by_{group_by}_{position.replace('+', '_')}.png", dpi=150)
    plt.savefig(output_dir / f"nll_curves_by_{group_by}_{position.replace('+', '_')}.pdf")
    plt.close()
    print(f"Saved: nll_curves_by_{group_by}_{position.replace('+', '_')}.png")


def plot_all_nll_curves(df: pd.DataFrame, output_dir: Path, project: str, entity: str = None,
                        position: str = "f1+l1"):
    """Plot all NLL curves on one plot, colored by rank."""
    import wandb
    
    subset = df[(df["position"] == position) & (df["method"] == "reft")].copy()
    if subset.empty:
        print(f"No ReFT data for position={position}")
        return
    
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ranks = sorted(subset["rank"].unique())
    rank_colors = {r: COLORS[i % len(COLORS)] for i, r in enumerate(ranks)}
    
    for _, run_row in subset.iterrows():
        run_id = run_row["run_id"]
        rank = run_row["rank"]
        lr = run_row["lr"]
        
        try:
            run = api.run(f"{project_path}/{run_id}")
            history = run.history(keys=["_step", "eval/nll"])
            history = history.dropna(subset=["eval/nll"])
            
            if not history.empty:
                ax.plot(history["_step"], history["eval/nll"],
                        color=rank_colors[rank],
                        linewidth=1.5, alpha=0.6)
        except Exception as e:
            print(f"Error fetching history for run {run_id}: {e}")
    
    # Add legend for ranks
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=rank_colors[r], linewidth=2, label=f"rank={int(r)}") 
                       for r in ranks]
    ax.legend(handles=legend_elements, title="Rank", loc="best")
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Eval NLL", fontsize=12)
    ax.set_title(f"All NLL Curves (position={position})", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"all_nll_curves_{position.replace('+', '_')}.png", dpi=150)
    plt.savefig(output_dir / f"all_nll_curves_{position.replace('+', '_')}.pdf")
    plt.close()
    print(f"Saved: all_nll_curves_{position.replace('+', '_')}.png")


def plot_nll_vs_rank(df: pd.DataFrame, output_dir: Path, position: str = "f1+l1"):
    """Plot NLL vs rank for different learning rates."""
    subset = df[(df["position"] == position) & (df["method"] == "reft")].copy()
    if subset.empty:
        print(f"No ReFT data for position={position}")
        return
    
    # Add completeness check
    subset["complete"] = subset.apply(lambda r: has_complete_hparams(r, "reft"), axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lrs = sorted(subset["lr"].dropna().unique())
    for i, lr in enumerate(lrs):
        lr_data = subset[subset["lr"] == lr].sort_values("rank")
        plot_with_completeness(
            ax, lr_data["rank"], lr_data["eval_nll"], lr_data["complete"],
            marker='o', color=COLORS[i % len(COLORS)], label=f"LR={lr:.0e}",
            linewidth=2, markersize=8
        )
    
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Eval NLL", fontsize=12)
    ax.set_title(f"LoReFT NLL vs Rank (position={position})", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend(title="Learning Rate", loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"nll_vs_rank_{position.replace('+', '_')}.png", dpi=150)
    plt.savefig(output_dir / f"nll_vs_rank_{position.replace('+', '_')}.pdf")
    plt.close()
    print(f"Saved: nll_vs_rank_{position.replace('+', '_')}.png")


def plot_nll_vs_lr(df: pd.DataFrame, output_dir: Path, position: str = "f1+l1"):
    """Plot NLL vs learning rate for different ranks."""
    subset = df[(df["position"] == position) & (df["method"] == "reft")].copy()
    if subset.empty:
        print(f"No ReFT data for position={position}")
        return
    
    # Add completeness check
    subset["complete"] = subset.apply(lambda r: has_complete_hparams(r, "reft"), axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ranks = sorted(subset["rank"].dropna().unique())
    for i, rank in enumerate(ranks):
        rank_data = subset[subset["rank"] == rank].sort_values("lr")
        plot_with_completeness(
            ax, rank_data["lr"], rank_data["eval_nll"], rank_data["complete"],
            marker='s', color=COLORS[i % len(COLORS)], label=f"Rank={int(rank)}",
            linewidth=2, markersize=8
        )
    
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Eval NLL", fontsize=12)
    ax.set_title(f"LoReFT NLL vs Learning Rate (position={position})", fontsize=14)
    ax.set_xscale("log")
    ax.legend(title="Rank", loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"nll_vs_lr_{position.replace('+', '_')}.png", dpi=150)
    plt.savefig(output_dir / f"nll_vs_lr_{position.replace('+', '_')}.pdf")
    plt.close()
    print(f"Saved: nll_vs_lr_{position.replace('+', '_')}.png")


def plot_best_nll_vs_params(df: pd.DataFrame, output_dir: Path):
    """Plot best NLL vs trainable params (like the LoRA Without Regret plot)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot ReFT runs (grouped by position)
    reft_df = df[df["method"] == "reft"].copy()
    if not reft_df.empty:
        reft_df["complete"] = reft_df.apply(lambda r: has_complete_hparams(r, "reft"), axis=1)
        best_per_config = reft_df.groupby(["rank", "position"]).agg({
            "eval_nll": "min",
            "trainable_params": "first",
            "complete": "all",
        }).reset_index()
        
        positions = best_per_config["position"].unique()
        reft_markers = ['o', 's', '^', 'D', 'v', '<']
        reft_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(positions)))
        
        for i, pos in enumerate(positions):
            pos_data = best_per_config[best_per_config["position"] == pos].sort_values("trainable_params")
            plot_with_completeness(
                ax, pos_data["trainable_params"], pos_data["eval_nll"], pos_data["complete"],
                marker=reft_markers[i % len(reft_markers)], color=reft_colors[i],
                label=f"ReFT ({pos})", linewidth=2, markersize=10
            )
    
    # Plot LoRA runs
    lora_df = df[df["method"] == "lora"].copy()
    if not lora_df.empty:
        lora_df["complete"] = lora_df.apply(lambda r: has_complete_hparams(r, "lora"), axis=1)
        best_per_rank = lora_df.groupby("lora_rank").agg({
            "eval_nll": "min",
            "trainable_params": "first",
            "complete": "all",
        }).reset_index()
        best_per_rank = best_per_rank.sort_values("trainable_params")
        
        plot_with_completeness(
            ax, best_per_rank["trainable_params"], best_per_rank["eval_nll"], best_per_rank["complete"],
            marker='p', color='orange', label="LoRA", linewidth=2, markersize=10
        )
    
    # Plot LoRA+ReFT runs if any
    lora_reft_df = df[df["method"] == "lora+reft"].copy()
    if not lora_reft_df.empty:
        lora_reft_df["complete"] = lora_reft_df.apply(
            lambda r: has_complete_hparams(r, "reft") and has_complete_hparams(r, "lora"), axis=1
        )
        best_per_config = lora_reft_df.groupby(["lora_rank", "rank"]).agg({
            "eval_nll": "min",
            "trainable_params": "first",
            "complete": "all",
        }).reset_index()
        best_per_config = best_per_config.sort_values("trainable_params")
        
        plot_with_completeness(
            ax, best_per_config["trainable_params"], best_per_config["eval_nll"], best_per_config["complete"],
            marker='h', color='purple', label="LoRA+ReFT", linewidth=2, markersize=10
        )
    
    # Add full finetune baseline if available
    full_ft = df[df["full_finetune"]]
    if not full_ft.empty:
        best_ft = full_ft.loc[full_ft["eval_nll"].idxmin()]
        ax.axhline(y=best_ft["eval_nll"], color='red', linestyle='--', 
                   label=f"Full FT (best)", linewidth=2)
        ax.scatter([best_ft["trainable_params"]], [best_ft["eval_nll"]], 
                   color='red', s=150, zorder=5, marker='*')
    
    ax.set_xlabel("Trainable Parameters", fontsize=12)
    ax.set_ylabel("Best Eval NLL", fontsize=12)
    ax.set_title("Best NLL vs Trainable Parameters", fontsize=14)
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "best_nll_vs_params.png", dpi=150)
    plt.savefig(output_dir / "best_nll_vs_params.pdf")
    plt.close()
    print("Saved: best_nll_vs_params.png")


def plot_position_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare different positions and methods."""
    reft_df = df[df["method"] == "reft"].copy()
    lora_df = df[df["method"] == "lora"].copy()
    
    positions = reft_df["position"].dropna().unique()
    if len(positions) < 1:
        print("No ReFT positions to compare")
        return
    
    # Add completeness checks
    reft_df["complete"] = reft_df.apply(lambda r: has_complete_hparams(r, "reft"), axis=1)
    if not lora_df.empty:
        lora_df["complete"] = lora_df.apply(lambda r: has_complete_hparams(r, "lora"), axis=1)
    
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
        
        # Group by rank and get best NLL, track if all runs for that rank are complete
        best_per_rank = pos_data.groupby("rank").agg({
            "eval_nll": "min",
            "complete": "all"  # True only if all runs at this rank are complete
        }).reset_index()
        best_per_rank = best_per_rank.sort_values("rank")
        
        plot_with_completeness(
            ax, best_per_rank["rank"], best_per_rank["eval_nll"], best_per_rank["complete"],
            marker=markers[i % len(markers)], color=colors[i], label=f"ReFT ({pos})",
            linewidth=2, markersize=8
        )
    
    # Add LoRA if available
    if not lora_df.empty:
        best_per_rank = lora_df.groupby("lora_rank").agg({
            "eval_nll": "min",
            "complete": "all"
        }).reset_index()
        best_per_rank = best_per_rank.sort_values("lora_rank")
        
        plot_with_completeness(
            ax, best_per_rank["lora_rank"], best_per_rank["eval_nll"], best_per_rank["complete"],
            marker='p', color='orange', label="LoRA",
            linewidth=2, markersize=8
        )
    
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
                legacy_data[["rank", "lr", "eval_nll", "complete"]],
                strict_data[["rank", "lr", "eval_nll", "complete"]],
                on=["rank", "lr"],
                suffixes=("_legacy", "_strict")
            )
            
            if not merged.empty:
                has_comparison = True
                # Both must be complete for filled marker
                complete_mask = merged["complete_legacy"] & merged["complete_strict"]
                
                # Plot filled for complete
                if complete_mask.any():
                    ax.scatter(merged.loc[complete_mask, "eval_nll_legacy"], 
                              merged.loc[complete_mask, "eval_nll_strict"], 
                              alpha=0.7, s=60, label=f"{legacy} vs {strict}")
                # Plot hollow for incomplete
                if (~complete_mask).any():
                    ax.scatter(merged.loc[~complete_mask, "eval_nll_legacy"], 
                              merged.loc[~complete_mask, "eval_nll_strict"], 
                              facecolors='none', edgecolors=plt.cm.tab10(0), 
                              alpha=0.7, s=60, linewidths=1.5,
                              label=f"{legacy} vs {strict} (incomplete)" if complete_mask.any() else f"{legacy} vs {strict}")
    
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
    
    # Add completeness check
    lora_df["complete"] = lora_df.apply(lambda r: has_complete_hparams(r, "lora"), axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: NLL vs LoRA rank
    ax = axes[0]
    lrs = sorted(lora_df["lr"].dropna().unique())
    for i, lr in enumerate(lrs):
        lr_data = lora_df[lora_df["lr"] == lr].sort_values("lora_rank")
        plot_with_completeness(
            ax, lr_data["lora_rank"], lr_data["eval_nll"], lr_data["complete"],
            marker='o', color=COLORS[i % len(COLORS)], label=f"LR={lr:.0e}",
            linewidth=2, markersize=8
        )
    
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
        plot_with_completeness(
            ax, rank_data["lr"], rank_data["eval_nll"], rank_data["complete"],
            marker='s', color=COLORS[i % len(COLORS)], label=f"Rank={int(rank)}",
            linewidth=2, markersize=8
        )
    
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
    
    # Add completeness checks
    reft_df["complete"] = reft_df.apply(lambda r: has_complete_hparams(r, "reft"), axis=1)
    lora_df["complete"] = lora_df.apply(lambda r: has_complete_hparams(r, "lora"), axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get best NLL for each trainable params level
    reft_best = reft_df.groupby("trainable_params").agg({
        "eval_nll": "min",
        "rank": "first",
        "position": "first",
        "complete": "all",
    }).reset_index().sort_values("trainable_params")
    
    lora_best = lora_df.groupby("trainable_params").agg({
        "eval_nll": "min",
        "lora_rank": "first",
        "complete": "all",
    }).reset_index().sort_values("trainable_params")
    
    plot_with_completeness(
        ax, reft_best["trainable_params"], reft_best["eval_nll"], reft_best["complete"],
        marker='o', color='blue', label="ReFT (best)", linewidth=2, markersize=8
    )
    plot_with_completeness(
        ax, lora_best["trainable_params"], lora_best["eval_nll"], lora_best["complete"],
        marker='s', color='orange', label="LoRA", linewidth=2, markersize=8
    )
    
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


def plot_heatmap(df: pd.DataFrame, output_dir: Path, position: str = "f1+l1"):
    """Create heatmap of NLL for rank x LR grid."""
    subset = df[(df["position"] == position) & (df["method"] == "reft")]
    if subset.empty:
        print(f"No ReFT data for position={position}")
        return
    
    pivot = subset.pivot_table(index="rank", columns="lr", values="eval_nll")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    
    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([int(r) for r in pivot.index])
    
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Rank", fontsize=12)
    ax.set_title(f"Eval NLL Heatmap (position={position})", fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Eval NLL", fontsize=12)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > pivot.values.mean() else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", 
                        color=text_color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_{position.replace('+', '_')}.png", dpi=150)
    plt.savefig(output_dir / f"heatmap_{position.replace('+', '_')}.pdf")
    plt.close()
    print(f"Saved: heatmap_{position.replace('+', '_')}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot LoReFT sweep results")
    parser.add_argument("--project", type=str, default="loreft-regret",
                        help="Wandb project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="Wandb entity (username or team)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Load from CSV instead of wandb")
    parser.add_argument("--output", type=str, default="./plots",
                        help="Output directory for plots")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save fetched data to CSV")
    parser.add_argument("--curves", action="store_true",
                        help="Plot NLL curves over training steps (slow, fetches history)")
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
    
    # Generate ReFT plots for each position
    reft_df = df[df["method"] == "reft"]
    for position in reft_df["position"].dropna().unique():
        if position:
            plot_nll_vs_rank(df, output_dir, position)
            plot_nll_vs_lr(df, output_dir, position)
            plot_heatmap(df, output_dir, position)
            
            # NLL curves over steps (slow - requires fetching history)
            if args.curves and not args.csv:
                plot_nll_curves(df, output_dir, args.project, args.entity, position, group_by="rank")
                plot_nll_curves(df, output_dir, args.project, args.entity, position, group_by="lr")
                plot_all_nll_curves(df, output_dir, args.project, args.entity, position)
    
    # Generate LoRA plots
    plot_lora_results(df, output_dir)
    
    # Generate comparison plots
    plot_best_nll_vs_params(df, output_dir)
    plot_position_comparison(df, output_dir)
    plot_method_comparison(df, output_dir)
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()

