"""Automated curriculum training pipeline.

Runs a multi-stage training progression, each stage resuming from the
previous best checkpoint. Sleep-and-forget: just launch it and check
results in the morning.

Usage:
    .\venv\Scripts\python.exe run_curriculum.py --device cuda
    .\venv\Scripts\python.exe run_curriculum.py --device cuda --prefix v5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PYTHON = os.path.join("venv", "Scripts", "python.exe")

STAGES = [
    {
        "name": "stage1_random",
        "timesteps": 200_000,
        "opponents": "random,random,random,random",
        "n_envs": 4,
        "eval_opponent": "random",
        "eval_battles": 100,
        "eval_freq": 20_000,
        "lr": 3e-4,
    },
    {
        "name": "stage2_maxpower",
        "timesteps": 400_000,
        "opponents": "max_base_power,max_base_power,max_base_power,max_base_power",
        "n_envs": 4,
        "eval_opponent": "max_base_power",
        "eval_battles": 100,
        "eval_freq": 20_000,
        "lr": 1e-4,
    },
    {
        "name": "stage3_heuristic",
        "timesteps": 600_000,
        "opponents": "simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,max_base_power,max_base_power",
        "n_envs": 8,
        "eval_opponent": "simple_heuristic",
        "eval_battles": 100,
        "eval_freq": 20_000,
        "lr": 5e-5,
    },
    {
        "name": "stage4_heuristic_long",
        "timesteps": 800_000,
        "opponents": "simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic",
        "n_envs": 8,
        "eval_opponent": "simple_heuristic",
        "eval_battles": 100,
        "eval_freq": 20_000,
        "lr": 3e-5,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run curriculum training pipeline.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix", default="v5", help="Run name prefix")
    parser.add_argument("--start-stage", type=int, default=1,
                        help="Stage to start from (1-indexed, default=1)")
    parser.add_argument("--resume-from", help="Checkpoint to resume stage 1 from")
    parser.add_argument("--output-dir", default="results/ppo")
    return parser.parse_args()


def best_model_path(output_dir: str, run_name: str) -> str:
    return os.path.join(output_dir, run_name, "best_model", "best_model.zip")


def fallback_model_path(output_dir: str, run_name: str) -> str:
    return os.path.join(output_dir, run_name, "model.zip")


def run_stage(
    stage: dict,
    *,
    prefix: str,
    device: str,
    output_dir: str,
    resume_from: str | None,
    stage_num: int,
) -> str | None:
    """Run one training stage. Returns path to best checkpoint, or None on failure."""
    run_name = f"{prefix}_{stage['name']}"
    print(f"\n{'='*60}")
    print(f"STAGE {stage_num}: {run_name}")
    print(f"  timesteps: {stage['timesteps']:,}")
    print(f"  opponents: {stage['opponents']}")
    print(f"  n_envs: {stage['n_envs']}")
    print(f"  eval: {stage['eval_battles']} battles vs {stage['eval_opponent']}")
    print(f"  lr: {stage['lr']}")
    if resume_from:
        print(f"  resume_from: {resume_from}")
    print(f"{'='*60}\n")

    cmd = [
        PYTHON, "train_ppo.py",
        "--train-timesteps", str(stage["timesteps"]),
        "--train-opponents", stage["opponents"],
        "--n-envs", str(stage["n_envs"]),
        "--eval-opponent", stage["eval_opponent"],
        "--eval-freq", str(stage["eval_freq"]),
        "--eval-battles", str(stage["eval_battles"]),
        "--device", device,
        "--learning-rate", str(stage["lr"]),
        "--run-name", run_name,
        "--output-dir", output_dir,
    ]
    if resume_from:
        cmd.extend(["--resume-from", resume_from])

    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.perf_counter() - start

    print(f"\nStage {stage_num} finished in {elapsed/60:.1f} minutes (exit code {result.returncode})")

    if result.returncode != 0:
        print(f"  WARNING: Stage {stage_num} failed!")
        return None

    # Find the best checkpoint
    best = best_model_path(output_dir, run_name)
    if os.path.exists(best):
        print(f"  Best model: {best}")
        return best

    fallback = fallback_model_path(output_dir, run_name)
    if os.path.exists(fallback):
        print(f"  Using fallback model: {fallback}")
        return fallback

    print(f"  WARNING: No checkpoint found for {run_name}")
    return None


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    print(f"Curriculum pipeline: {len(STAGES)} stages, prefix={args.prefix}")
    print(f"Total planned timesteps: {sum(s['timesteps'] for s in STAGES):,}")

    resume_from = args.resume_from
    completed = 0

    for i, stage in enumerate(STAGES, 1):
        if i < args.start_stage:
            # Skip stages before start_stage, but track the expected checkpoint
            expected = best_model_path(args.output_dir, f"{args.prefix}_{stage['name']}")
            if os.path.exists(expected):
                resume_from = expected
                print(f"Skipping stage {i}, found checkpoint: {expected}")
            continue

        checkpoint = run_stage(
            stage,
            prefix=args.prefix,
            device=args.device,
            output_dir=args.output_dir,
            resume_from=resume_from,
            stage_num=i,
        )

        if checkpoint is None:
            print(f"\nPipeline stopped at stage {i} due to failure.")
            break

        resume_from = checkpoint
        completed += 1

    total_elapsed = time.perf_counter() - start_time
    print(f"\n{'='*60}")
    print(f"CURRICULUM COMPLETE")
    print(f"  Stages completed: {completed}/{len(STAGES)}")
    print(f"  Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    if resume_from:
        print(f"  Final best model: {resume_from}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
