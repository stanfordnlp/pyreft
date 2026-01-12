#!/usr/bin/env python3
"""
Retrain with best LR from sweep for 10x longer.

For each (rank, position) configuration, this script:
1. Finds the best-performing LR from wandb
2. Retrains from scratch with that LR for 10x epochs

Usage:
    # Analyze what would be run (dry run)
    python continue_training.py --dry-run
    
    # Run retraining for all best configs
    python continue_training.py
    
    # Run for specific rank/position
    python continue_training.py --rank 4 --position f1+l1
"""

import argparse
import subprocess
import sys


def fetch_best_runs(project: str, entity: str = None):
    """Fetch best LR for each (rank, position) from wandb."""
    import wandb
    import pandas as pd
    
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    runs = api.runs(project_path)
    
    records = []
    for run in runs:
        if run.state != "finished":
            continue
        
        config = run.config
        summary = run.summary._json_dict
        
        # Skip LoRA and full finetune
        use_lora = config.get("use_lora", False)
        disable_reft = config.get("disable_reft", False)
        full_finetune = config.get("full_finetune", False)
        
        if full_finetune or (use_lora and disable_reft):
            continue
        
        # Get eval NLL
        eval_nll = summary.get("eval/nll") or summary.get("eval_nll")
        if eval_nll is None:
            continue
        
        record = {
            "run_name": run.name,
            "run_id": run.id,
            "rank": config.get("rank"),
            "lr": config.get("lr"),
            "position": config.get("position"),
            "share_weights": config.get("share_weights", False),
            "eval_nll": eval_nll,
            "config": config,
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    if df.empty:
        return {}
    
    # Find best LR for each (rank, position)
    best_runs = {}
    for (rank, position), group in df.groupby(["rank", "position"]):
        best_idx = group["eval_nll"].idxmin()
        best_row = group.loc[best_idx]
        best_runs[(rank, position)] = best_row.to_dict()
    
    return best_runs


def format_lr(lr: float) -> str:
    """Format LR for run name (e.g., 0.001 -> '1e-3')."""
    if lr >= 1:
        return str(int(lr))
    exp = 0
    while lr < 1:
        lr *= 10
        exp += 1
    return f"{int(lr)}e-{exp}"


def run_training(
    rank: int,
    position: str,
    lr: float,
    epochs: int = 10,
    output_dir: str = "./outputs_10x",
    wandb_project: str = "loreft-regret-10x",
    dry_run: bool = False,
    extra_args: list = None,
):
    """Run train.py with the given config."""
    
    # Build run name
    run_name = f"r{int(rank)}___10x_{position}___lr{format_lr(lr)}"
    full_output_dir = f"{output_dir}/{run_name}"
    
    # Determine share_weights based on position
    share_weights = position in ["all", "alls"]
    
    cmd = [
        sys.executable, "train.py",
        "--rank", str(int(rank)),
        "--position", position,
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--output_dir", full_output_dir,
        "--wandb_project", wandb_project,
        "--run_name", run_name,
        "--debug_interventions",
    ]

    if share_weights:
        cmd.append("--share_weights")
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Training: rank={rank}, position={position}, lr={lr}")
    print(f"Run name: {run_name}")
    print(f"Epochs: {epochs}")
    print(f"Output: {full_output_dir}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("[DRY RUN] Would execute above command")
        return 0
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Retrain with best LR for 10x longer")
    parser.add_argument("--wandb_project", type=str, default="loreft-regret",
                        help="Source wandb project to find best runs")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity (username or team)")
    parser.add_argument("--output_dir", type=str, default="./outputs_10x",
                        help="Output directory for retrained models")
    parser.add_argument("--output_wandb_project", type=str, default="loreft-regret-10x-restart",
                        help="Wandb project for retrained runs")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument("--rank", type=int, default=None,
                        help="Only train for this rank")
    parser.add_argument("--position", type=str, default=None,
                        help="Only train for this position")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without actually running")
    # Pass through additional args to train.py
    parser.add_argument("extra_args", nargs="*",
                        help="Additional arguments to pass to train.py")
    args = parser.parse_args()
    
    print("Fetching best runs from wandb...")
    best_runs = fetch_best_runs(args.wandb_project, args.wandb_entity)
    
    if not best_runs:
        print("No completed runs found!")
        return
    
    print(f"\nFound {len(best_runs)} unique (rank, position) configurations:")
    for (rank, position), run in sorted(best_runs.items()):
        print(f"  rank={rank}, position={position}: LR={run['lr']}, NLL={run['eval_nll']:.4f}")
    
    # Filter if specific rank/position requested
    if args.rank is not None or args.position is not None:
        filtered = {}
        for (rank, position), run in best_runs.items():
            if args.rank is not None and rank != args.rank:
                continue
            if args.position is not None and position != args.position:
                continue
            filtered[(rank, position)] = run
        best_runs = filtered
        print(f"\nFiltered to {len(best_runs)} configurations")
    
    # Run training for each
    failed = []
    for (rank, position), run in sorted(best_runs.items()):
        print(f"\n{'#'*60}")
        print(f"# Processing: rank={rank}, position={position}")
        print(f"# Best LR: {run['lr']}, Original Eval NLL: {run['eval_nll']:.4f}")
        print(f"{'#'*60}")
        
        returncode = run_training(
            rank=rank,
            position=position,
            lr=run["lr"],
            epochs=args.epochs,
            output_dir=args.output_dir,
            wandb_project=args.output_wandb_project,
            dry_run=args.dry_run,
            extra_args=args.extra_args,
        )
        
        if returncode != 0:
            failed.append((rank, position))
            print(f"ERROR: Training failed for rank={rank}, position={position}")
    
    print(f"\n{'='*60}")
    print(f"Complete!")
    if failed:
        print(f"Failed: {len(failed)} configurations")
        for rank, position in failed:
            print(f"  - rank={rank}, position={position}")
    else:
        print(f"All {len(best_runs)} configurations trained successfully")


if __name__ == "__main__":
    main()
