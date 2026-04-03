"""PokeRL Training Dashboard — Streamlit UI

Launch with:
    .\venv\Scripts\streamlit.exe run dashboard.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

RESULTS_DIR = "results/ppo"
BENCHMARK_DIR = "results/benchmarks"
ELO_FILE = "results/elo_ratings.json"
PYTHON = os.path.join("venv", "Scripts", "python.exe")

st.set_page_config(page_title="PokeRL Dashboard", layout="wide")
st.title("PokeRL Training Dashboard")


# ── Elo Leaderboard ──────────────────────────────────────────────
st.header("Elo Leaderboard")

if os.path.exists(ELO_FILE):
    with open(ELO_FILE) as f:
        elo_data = json.load(f)

    # Add anchors
    anchors = {"random": 400, "max_base_power": 700, "simple_heuristic": 1000}
    rows = []
    for name, entry in elo_data.items():
        rows.append({
            "Name": name,
            "Elo": round(entry["rating"]),
            "Wins": entry["wins"],
            "Losses": entry["losses"],
            "Games": entry["games"],
            "Win Rate": f"{entry['wins']/max(entry['games'],1):.1%}",
        })
    for name, rating in anchors.items():
        if name not in elo_data:
            rows.append({"Name": name, "Elo": rating, "Wins": "-", "Losses": "-", "Games": "-", "Win Rate": "-"})

    df_elo = pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)
    st.dataframe(df_elo, use_container_width=True, hide_index=True)
else:
    st.info("No Elo data yet. Run benchmarks to populate.")


# ── Training Runs ────────────────────────────────────────────────
st.header("Training Runs")

run_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "*/summary.json")))
if run_dirs:
    rows = []
    for summary_path in run_dirs:
        run_name = os.path.basename(os.path.dirname(summary_path))
        try:
            with open(summary_path) as f:
                d = json.load(f)
            summary = d.get("summary", {})
            rows.append({
                "Run": run_name,
                "Eval Opponent": d.get("eval_opponent", "?"),
                "Win Rate": f"{summary.get('win_rate', 0):.1%}",
                "Avg Reward": f"{summary.get('avg_reward', 0):.1f}",
                "Timesteps": f"{d.get('train_timesteps', 0):,}",
                "Elapsed": f"{summary.get('elapsed_seconds', 0)/60:.0f}m",
            })
        except Exception:
            rows.append({"Run": run_name, "Eval Opponent": "?", "Win Rate": "?", "Avg Reward": "?", "Timesteps": "?", "Elapsed": "?"})

    df_runs = pd.DataFrame(rows)
    st.dataframe(df_runs, use_container_width=True, hide_index=True)
else:
    st.info("No training runs found.")


# ── Benchmark Results ────────────────────────────────────────────
st.header("Benchmark Results")

bench_dirs = sorted(glob.glob(os.path.join(BENCHMARK_DIR, "*/summary.json")))
if bench_dirs:
    rows = []
    for summary_path in bench_dirs:
        bench_name = os.path.basename(os.path.dirname(summary_path))
        try:
            with open(summary_path) as f:
                d = json.load(f)
            matchups = d.get("matchups", [])
            row = {"Benchmark": bench_name}
            for m in matchups:
                opp = m["opponent"]
                row[f"vs {opp}"] = f"{m['win_rate']:.0%} ({m['wins']}/{m['wins']+m['losses']})"
            rows.append(row)
        except Exception:
            rows.append({"Benchmark": bench_name})

    df_bench = pd.DataFrame(rows)
    st.dataframe(df_bench, use_container_width=True, hide_index=True)
else:
    st.info("No benchmark results found.")


# ── Win Rate Chart ───────────────────────────────────────────────
st.header("Win Rate vs Heuristic Over Time")

# Collect heuristic win rates from all v5+ runs
chart_data = []
for summary_path in sorted(glob.glob(os.path.join(RESULTS_DIR, "v*/summary.json"))):
    run_name = os.path.basename(os.path.dirname(summary_path))
    try:
        with open(summary_path) as f:
            d = json.load(f)
        if d.get("eval_opponent") == "simple_heuristic":
            summary = d.get("summary", {})
            chart_data.append({
                "Run": run_name,
                "Win Rate": summary.get("win_rate", 0),
            })
    except Exception:
        pass

if chart_data:
    df_chart = pd.DataFrame(chart_data)
    st.bar_chart(df_chart.set_index("Run")["Win Rate"])
else:
    st.info("No heuristic eval data yet.")


# ── Shaping Health ───────────────────────────────────────────────
st.header("Reward Shaping Health")

# Show shaping from the most recent run
latest_summaries = sorted(glob.glob(os.path.join(RESULTS_DIR, "v*/summary.json")), reverse=True)
if latest_summaries:
    selected_run = st.selectbox("Select run:", [os.path.basename(os.path.dirname(p)) for p in latest_summaries])
    summary_path = os.path.join(RESULTS_DIR, selected_run, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            d = json.load(f)
        shaping = d.get("eval_tactical_shaping_report", {})
        if shaping:
            col1, col2, col3 = st.columns(3)
            col1.metric("Shaped Action Rate", f"{shaping.get('shaped_action_rate', 0):.1%}")
            col2.metric("Positive Total", f"{shaping.get('positive_total', 0):.1f}")
            col3.metric("Negative Total", f"{shaping.get('negative_total', 0):.1f}")

            counts = shaping.get("counts", {})
            if counts:
                df_counts = pd.DataFrame([
                    {"Lever": k, "Count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])
                ])
                st.bar_chart(df_counts.set_index("Lever"))

            totals = shaping.get("totals", {})
            if totals:
                df_totals = pd.DataFrame([
                    {"Lever": k, "Total Reward": v} for k, v in sorted(totals.items(), key=lambda x: -abs(x[1]))
                ])
                st.bar_chart(df_totals.set_index("Lever"))


# ── Quick Launch ─────────────────────────────────────────────────
st.header("Quick Launch")

with st.expander("Launch Training Run"):
    col1, col2 = st.columns(2)
    with col1:
        run_name = st.text_input("Run name", "v6_experiment")
        timesteps = st.number_input("Timesteps", value=200000, step=50000)
        n_envs = st.selectbox("Envs", [4, 8], index=1)
    with col2:
        opponent_mix = st.text_input("Opponents (comma-sep)", "simple_heuristic," * 8)
        eval_opponent = st.selectbox("Eval opponent", ["simple_heuristic", "max_base_power", "random"])
        lr = st.select_slider("Learning rate", [1e-5, 3e-5, 5e-5, 1e-4, 3e-4], value=5e-5)

    # Checkpoint selection
    checkpoints = sorted(glob.glob(os.path.join(RESULTS_DIR, "*/best_model/best_model.zip")))
    checkpoint_names = ["(fresh)"] + [os.path.dirname(os.path.dirname(p)).split(os.sep)[-1] for p in checkpoints]
    resume_from = st.selectbox("Resume from", checkpoint_names)

    if st.button("Launch Training"):
        cmd = f"{PYTHON} train_ppo.py --train-timesteps {timesteps} --train-opponents \"{opponent_mix.strip().rstrip(',')}\" --n-envs {n_envs} --eval-opponent {eval_opponent} --eval-freq 20000 --eval-battles 100 --device cuda --learning-rate {lr} --run-name {run_name}"
        if resume_from != "(fresh)":
            idx = checkpoint_names.index(resume_from) - 1
            cmd += f' --resume-from "{checkpoints[idx]}"'
        st.code(cmd)
        st.info("Copy and run in terminal, or press the button below to launch in background.")

with st.expander("Launch Benchmark"):
    bench_checkpoints = sorted(glob.glob(os.path.join(RESULTS_DIR, "*/best_model/best_model.zip")))
    bench_names = [os.path.dirname(os.path.dirname(p)).split(os.sep)[-1] for p in bench_checkpoints]
    if bench_names:
        bench_selected = st.selectbox("Checkpoint to benchmark", bench_names)
        bench_battles = st.number_input("Battles per opponent", value=100, step=50)
        if st.button("Launch Benchmark"):
            idx = bench_names.index(bench_selected)
            cmd = f'{PYTHON} benchmark_model.py --checkpoint "{bench_checkpoints[idx]}" --algo ppo --opponents "random,max_base_power,simple_heuristic" --n-battles {bench_battles} --device cuda'
            st.code(cmd)
