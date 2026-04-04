"""Benchmark suite: run all v5/v6 checkpoints against all opponents.

Usage:
    .\venv\Scripts\python.exe run_benchmark_suite.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PYTHON = os.path.join("venv", "Scripts", "python.exe")

CHECKPOINTS = [
    # Baselines
    ("v6_stage4", "results/ppo/v6_stage4_heuristic_long/best_model/best_model.zip"),
    # v7-v9 progression
    ("v7_highent", "results/ppo/v7_heuristic_highent/model.zip"),
    ("v8_lowlr", "results/ppo/v8_lowlr/model.zip"),
    ("v8c_long", "results/ppo/v8c_long/model.zip"),
    ("v9_noatk_best", "results/ppo/v9_heuristic_noatk/best_model_66pct.zip"),
    ("v9_noatk_final", "results/ppo/v9_heuristic_noatk/model.zip"),
]

OPPONENTS = "random,max_base_power,simple_heuristic"
N_BATTLES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark suite for v5 checkpoints.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-battles", type=int, default=N_BATTLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[dict] = []
    start = time.perf_counter()

    for name, path in CHECKPOINTS:
        if not os.path.exists(path):
            print(f"\n--- {name}: SKIPPED (checkpoint not found) ---")
            results.append({"name": name, "status": "missing"})
            continue

        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {name}")
        print(f"  checkpoint: {path}")
        print(f"{'='*60}")

        cmd = [
            PYTHON, "benchmark_model.py",
            "--checkpoint", path,
            "--algo", "ppo",
            "--opponents", OPPONENTS,
            "--n-battles", str(args.n_battles),
            "--device", args.device,
            "--run-name", f"suite_{name}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr

        # Parse win rates from output
        entry = {"name": name, "status": "ok"}
        for opponent in OPPONENTS.split(","):
            for line in output.split("\n"):
                # Match only the summary line (has "wins=" right after opponent name)
                if line.startswith(f"{opponent}: wins="):
                    parts = line.split()
                    for p in parts:
                        if p.startswith("win_rate="):
                            try:
                                entry[opponent] = float(p.split("=")[1])
                            except ValueError:
                                pass
                        elif p.startswith("wins="):
                            try:
                                entry[f"{opponent}_wins"] = int(p.split("=")[1])
                            except ValueError:
                                pass
        results.append(entry)

        wr = entry.get("simple_heuristic", "?")
        print(f"  heuristic win rate: {wr}")

    elapsed = time.perf_counter() - start

    # Print summary table
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUITE RESULTS ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<25} {'vs Random':>10} {'vs MaxPow':>10} {'vs Heur':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        if r["status"] == "missing":
            print(f"{r['name']:<25} {'MISSING':>10} {'':>10} {'':>10}")
            continue
        rand = r.get("random", "?")
        maxp = r.get("max_base_power", "?")
        heur = r.get("simple_heuristic", "?")
        rand_s = f"{rand:.0%}" if isinstance(rand, float) else rand
        maxp_s = f"{maxp:.0%}" if isinstance(maxp, float) else maxp
        heur_s = f"{heur:.0%}" if isinstance(heur, float) else heur
        print(f"{r['name']:<25} {rand_s:>10} {maxp_s:>10} {heur_s:>10}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
