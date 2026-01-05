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
        
        record = {
            "run_name": run.name,
            "run_id": run.id,
            "rank": config.get("rank"),
            "lr": config.get("lr"),
            "position": config.get("position"),
            "share_weights": config.get("share_weights", False),
            "full_finetune": config.get("full_finetune", False),
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
    
    subset = df[(df["position"] == position) & (~df["full_finetune"])].copy()
    if subset.empty:
        print(f"No data for position={position}")
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
    
    subset = df[(df["position"] == position) & (~df["full_finetune"])].copy()
    if subset.empty:
        print(f"No data for position={position}")
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
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"all_nll_curves_{position.replace('+', '_')}.png", dpi=150)
    plt.savefig(output_dir / f"all_nll_curves_{position.replace('+', '_')}.pdf")
    plt.close()
    print(f"Saved: all_nll_curves_{position.replace('+', '_')}.png")


def plot_nll_vs_rank(df: pd.DataFrame, output_dir: Path, position: str = "f1+l1"):
    """Plot NLL vs rank for different learning rates."""
    subset = df[(df["position"] == position) & (~df["full_finetune"])]
    if subset.empty:
        print(f"No data for position={position}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lrs = sorted(subset["lr"].unique())
    for i, lr in enumerate(lrs):
        lr_data = subset[subset["lr"] == lr].sort_values("rank")
        ax.plot(lr_data["rank"], lr_data["eval_nll"], 
                marker='o', label=f"LR={lr:.0e}", color=COLORS[i % len(COLORS)],
                linewidth=2, markersize=8)
    
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
    subset = df[(df["position"] == position) & (~df["full_finetune"])]
    if subset.empty:
        print(f"No data for position={position}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ranks = sorted(subset["rank"].unique())
    for i, rank in enumerate(ranks):
        rank_data = subset[subset["rank"] == rank].sort_values("lr")
        ax.plot(rank_data["lr"], rank_data["eval_nll"],
                marker='s', label=f"Rank={int(rank)}", color=COLORS[i % len(COLORS)],
                linewidth=2, markersize=8)
    
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
    # Get best NLL for each (rank, position) combo across all LRs
    loreft_df = df[~df["full_finetune"]].copy()
    
    if loreft_df.empty:
        print("No LoReFT data")
        return
    
    best_per_config = loreft_df.groupby(["rank", "position"]).agg({
        "eval_nll": "min",
        "trainable_params": "first",
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = best_per_config["position"].unique()
    markers = ['o', 's', '^', 'D']
    
    for i, pos in enumerate(positions):
        pos_data = best_per_config[best_per_config["position"] == pos].sort_values("trainable_params")
        ax.plot(pos_data["trainable_params"], pos_data["eval_nll"],
                marker=markers[i % len(markers)], label=f"position={pos}",
                linewidth=2, markersize=10)
    
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
    ax.set_title("LoReFT: Best NLL vs Trainable Parameters", fontsize=14)
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "best_nll_vs_params.png", dpi=150)
    plt.savefig(output_dir / "best_nll_vs_params.pdf")
    plt.close()
    print("Saved: best_nll_vs_params.png")


def plot_position_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare f1+l1 vs all positions."""
    loreft_df = df[~df["full_finetune"]].copy()
    
    positions = loreft_df["position"].unique()
    if len(positions) < 2:
        print("Need multiple positions to compare")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: NLL vs Rank for each position
    ax = axes[0]
    for pos in positions:
        # Get best LR for each rank
        pos_data = loreft_df[loreft_df["position"] == pos]
        best_per_rank = pos_data.groupby("rank")["eval_nll"].min().reset_index()
        best_per_rank = best_per_rank.sort_values("rank")
        ax.plot(best_per_rank["rank"], best_per_rank["eval_nll"],
                marker='o', label=f"position={pos}", linewidth=2, markersize=8)
    
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Best Eval NLL", fontsize=12)
    ax.set_title("Best NLL vs Rank (optimized over LR)", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Scatter of f1+l1 vs all (if both exist)
    ax = axes[1]
    if "f1+l1" in positions and "all" in positions:
        merged = loreft_df.pivot_table(
            index=["rank", "lr"], 
            columns="position", 
            values="eval_nll"
        ).reset_index()
        
        if "f1+l1" in merged.columns and "all" in merged.columns:
            ax.scatter(merged["f1+l1"], merged["all"], alpha=0.7, s=60)
            
            # Add diagonal line
            lims = [
                min(merged["f1+l1"].min(), merged["all"].min()),
                max(merged["f1+l1"].max(), merged["all"].max()),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, label="y=x")
            
            ax.set_xlabel("NLL (f1+l1)", fontsize=12)
            ax.set_ylabel("NLL (all)", fontsize=12)
            ax.set_title("Position Comparison: f1+l1 vs all", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "position_comparison.png", dpi=150)
    plt.savefig(output_dir / "position_comparison.pdf")
    plt.close()
    print("Saved: position_comparison.png")


def plot_heatmap(df: pd.DataFrame, output_dir: Path, position: str = "f1+l1"):
    """Create heatmap of NLL for rank x LR grid."""
    subset = df[(df["position"] == position) & (~df["full_finetune"])]
    if subset.empty:
        print(f"No data for position={position}")
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
    print(f"Positions: {df['position'].unique()}")
    print(f"Ranks: {sorted(df['rank'].dropna().unique())}")
    print(f"LRs: {sorted(df['lr'].dropna().unique())}")
    
    # Generate plots
    for position in df["position"].dropna().unique():
        if position and not df[df["position"] == position]["full_finetune"].all():
            plot_nll_vs_rank(df, output_dir, position)
            plot_nll_vs_lr(df, output_dir, position)
            plot_heatmap(df, output_dir, position)
            
            # NLL curves over steps (slow - requires fetching history)
            if args.curves and not args.csv:
                plot_nll_curves(df, output_dir, args.project, args.entity, position, group_by="rank")
                plot_nll_curves(df, output_dir, args.project, args.entity, position, group_by="lr")
                plot_all_nll_curves(df, output_dir, args.project, args.entity, position)
    
    plot_best_nll_vs_params(df, output_dir)
    plot_position_comparison(df, output_dir)
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()

