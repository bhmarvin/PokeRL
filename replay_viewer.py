"""Battle replay viewer with per-step reward breakdown.

Usage:
    # Generate replays:
    .\venv\Scripts\python.exe replay_viewer.py generate --checkpoint results/ppo/v8c_long/best_model/best_model.zip --n-battles 5

    # View in Streamlit:
    .\venv\Scripts\streamlit.exe run replay_viewer.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from stable_baselines3.common.monitor import Monitor

from brent_agent import BrentsRLAgent, VECTOR_LENGTH
from opponents import create_opponent

REPLAY_DIR = "results/replays"


def generate_replays(
    checkpoint: str,
    opponent: str = "simple_heuristic",
    n_battles: int = 5,
    device: str = "cpu",
) -> list[dict]:
    """Run battles with replay logging and save per-step data."""
    env = BrentsRLAgent(
        battle_format="gen9randombattle",
        log_level=40,
        open_timeout=None,
        strict=True,
    )
    opp = create_opponent(
        opponent,
        battle_format="gen9randombattle",
        log_level=40,
        start_listening=False,
    )
    wrapped = Monitor(SingleAgentWrapper(env, opp))
    model = PPO.load(checkpoint, env=wrapped, device=device)

    replays = []
    for i in range(n_battles):
        # Enable logging for this battle
        env.enable_replay_logging()

        obs, _ = wrapped.reset(seed=42 + i)
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = wrapped.step(action)
            done = bool(terminated or truncated)
            steps += 1

        battle = env.battle1
        replay = {
            "battle_index": i + 1,
            "won": bool(getattr(battle, "won", False)),
            "lost": bool(getattr(battle, "lost", False)),
            "opponent": opponent,
            "steps": steps,
            "log": env.get_replay_log(),
        }
        replays.append(replay)

        result = "WIN" if replay["won"] else "LOSS" if replay["lost"] else "DRAW"
        total_reward = sum(s["total"] for s in replay["log"])
        print(f"  Battle {i+1}/{n_battles}: {result} in {steps} steps, total_reward={total_reward:.2f}")

        env.disable_replay_logging()

    wrapped.close()
    return replays


def save_replays(replays: list[dict], run_name: str) -> str:
    out_dir = os.path.join(REPLAY_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "replays.json")
    with open(out_path, "w") as f:
        json.dump(replays, f, indent=2)
    print(f"Saved {len(replays)} replays to {out_path}")
    return out_path


def main_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", default="simple_heuristic")
    parser.add_argument("--n-battles", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or Path(args.checkpoint).parent.parent.name
    print(f"Generating {args.n_battles} replays from {args.checkpoint} vs {args.opponent}")
    replays = generate_replays(args.checkpoint, args.opponent, args.n_battles, args.device)
    save_replays(replays, run_name)


# ── Streamlit UI ──────────────────────────────────────────────────
def run_streamlit():
    import streamlit as st
    import pandas as pd

    st.set_page_config(page_title="PokeRL Replay Viewer", layout="wide")
    st.title("Battle Replay Viewer")

    # Find replay files
    replay_files = list(Path(REPLAY_DIR).glob("*/replays.json"))
    if not replay_files:
        st.warning(f"No replays found in {REPLAY_DIR}/. Generate some first:\n\n"
                   "`python replay_viewer.py generate --checkpoint <path>`")
        return

    # Select replay set
    run_names = [f.parent.name for f in replay_files]
    selected_run = st.sidebar.selectbox("Run", run_names)
    replay_path = Path(REPLAY_DIR) / selected_run / "replays.json"

    with open(replay_path) as f:
        replays = json.load(f)

    # Select battle
    battle_labels = [
        f"Battle {r['battle_index']}: {'WIN' if r['won'] else 'LOSS' if r['lost'] else 'DRAW'} ({r['steps']} steps)"
        for r in replays
    ]
    selected_idx = st.sidebar.selectbox("Battle", range(len(replays)), format_func=lambda i: battle_labels[i])
    battle = replays[selected_idx]
    log = battle["log"]

    if not log:
        st.warning("No step data for this battle.")
        return

    # Collapse duplicate turns into one row per turn
    turns: list[dict] = []
    for step in log:
        if turns and turns[-1]["turn"] == step["turn"]:
            # Same turn — merge: sum rewards, keep final HP, collect all shaping
            prev = turns[-1]
            prev["active_hp"] = step["active_hp"]
            prev["opponent_hp"] = step["opponent_hp"]
            prev["base_reward"] += step["base_reward"]
            prev["shaping"] += step["shaping"]
            prev["head_hunter"] += step["head_hunter"]
            prev["predicted_switch"] += step["predicted_switch"]
            prev["total"] += step["total"]
            prev["shaping_details"].extend(step.get("shaping_details", []))
            # Update active/opponent if they changed (faint/switch mid-turn)
            if step["active"] != prev["active"]:
                prev["active"] = step["active"]
            if step["opponent"] != prev["opponent"]:
                prev["opponent"] = step["opponent"]
        else:
            turns.append({**step, "shaping_details": list(step.get("shaping_details", []))})

    # Recalculate cumulative
    cumul = 0.0
    for t in turns:
        cumul += t["total"]
        t["cumulative"] = round(cumul, 3)

    # Summary
    total_reward = sum(t["total"] for t in turns)
    total_base = sum(t["base_reward"] for t in turns)
    total_shaping = sum(t["shaping"] for t in turns)
    total_hh = sum(t["head_hunter"] for t in turns)
    total_ps = sum(t["predicted_switch"] for t in turns)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Reward", f"{total_reward:.2f}")
    col2.metric("Base (HP/KO/Win)", f"{total_base:.2f}")
    col3.metric("Shaping", f"{total_shaping:.2f}")
    col4.metric("Head Hunter", f"{total_hh:.2f}")
    col5.metric("Predicted Switch", f"{total_ps:.2f}")

    st.divider()

    # Reward chart
    df = pd.DataFrame(turns)
    st.subheader("Reward Over Time")
    st.line_chart(df.set_index("turn")[["cumulative"]], height=200)

    st.subheader("Reward Components Per Turn")
    st.bar_chart(df.set_index("turn")[["base_reward", "shaping", "head_hunter", "predicted_switch"]], height=200)

    st.divider()

    # Step-by-step table
    st.subheader("Turn-by-Turn Log")

    def _color_reward(val):
        if val > 0.5:
            return "background-color: #1a472a; color: #4ade80"
        elif val > 0:
            return "background-color: #14332a; color: #86efac"
        elif val < -0.5:
            return "background-color: #4a1a1a; color: #f87171"
        elif val < 0:
            return "background-color: #331414; color: #fca5a5"
        return ""

    rows = []
    for t in turns:
        shaping_tags = ", ".join(
            f"{d['reason']} ({d['reward']:+.2f})" for d in t.get("shaping_details", [])
        )
        rows.append({
            "Turn": t["turn"],
            "Action": t["action"],
            "Active": f"{t['active']} ({t['active_hp']:.0%})",
            "Opponent": f"{t['opponent']} ({t['opponent_hp']:.0%})",
            "Base": round(t["base_reward"], 2),
            "Shaping": round(t["shaping"], 2),
            "HH": round(t["head_hunter"], 2),
            "PSwitch": round(t["predicted_switch"], 2),
            "Total": round(t["total"], 2),
            "Cumul.": t["cumulative"],
            "Shaping Details": shaping_tags or "-",
        })

    table_df = pd.DataFrame(rows)
    styled = table_df.style.applymap(
        _color_reward, subset=["Base", "Total", "Cumul."]
    ).format({
        "Base": "{:+.2f}",
        "Shaping": "{:+.2f}",
        "HH": "{:+.2f}",
        "PSwitch": "{:+.2f}",
        "Total": "{:+.2f}",
        "Cumul.": "{:+.2f}",
    })
    st.dataframe(styled, use_container_width=True, height=600)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        main_generate()
    else:
        run_streamlit()
