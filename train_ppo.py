from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "pokrl-mpl-cache"),
)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from brent_agent import BrentsRLAgent, REWARD_CONFIG, VECTOR_LENGTH
from experiment_io import make_run_name, prepare_run_artifacts, write_summary
from opponents import OPPONENT_CHOICES, create_opponent
from policies import MaskedActorCriticPolicy


@dataclass
class EvalBattleRecord:
    battle_index: int
    won: bool
    lost: bool
    steps: int
    reward: float


@dataclass
class EvalSummary:
    opponent: str
    wins: int
    losses: int
    draws: int
    avg_reward: float
    avg_steps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on BrentsRLAgent.")
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--train-timesteps", type=int, default=2048)
    parser.add_argument("--eval-battles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", type=int, default=40)
    parser.add_argument("--verify-embedding", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-opponent", choices=OPPONENT_CHOICES, default="random")
    parser.add_argument("--eval-opponent", choices=OPPONENT_CHOICES, default="random")
    parser.add_argument("--output-dir", default="results/ppo")
    parser.add_argument("--run-name")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--resume-from")
    return parser.parse_args()


def create_wrapped_env(args: argparse.Namespace, opponent_name: str) -> Monitor:
    env = BrentsRLAgent(
        battle_format=args.battle_format,
        log_level=args.log_level,
        open_timeout=None,
        strict=True,
    )
    opponent = create_opponent(
        opponent_name,
        battle_format=args.battle_format,
        log_level=args.log_level,
        start_listening=False,
    )
    return Monitor(SingleAgentWrapper(env, opponent))


def verify_obs(env: Monitor, obs: dict[str, Any], verify_embedding: bool) -> None:
    raw_env = env.unwrapped.env
    vector = np.asarray(obs["observation"], dtype=np.float32)
    mask = np.asarray(obs["action_mask"], dtype=np.int64)

    if vector.shape != (VECTOR_LENGTH,):
        raise RuntimeError(f"Unexpected observation shape {vector.shape}")
    if mask.shape != (env.action_space.n,):
        raise RuntimeError(f"Unexpected action mask shape {mask.shape}")
    if not np.any(mask == 1):
        raise RuntimeError("No legal actions available.")

    if verify_embedding:
        issues = raw_env.vector_builder.verify_battle_embedding(raw_env.battle1, vector)
        if issues:
            raise RuntimeError("Embedding verification failed: " + "; ".join(issues[:5]))


def predict_masked_action(model: PPO, obs: dict[str, Any]) -> np.int64:
    batched_obs = {
        key: np.expand_dims(np.asarray(value), axis=0)
        for key, value in obs.items()
    }
    action, _ = model.predict(batched_obs, deterministic=True)
    return np.int64(np.asarray(action).reshape(-1)[0])


def evaluate_policy(
    model: PPO,
    args: argparse.Namespace,
) -> tuple[EvalSummary, list[EvalBattleRecord]]:
    env = create_wrapped_env(args, args.eval_opponent)
    records: list[EvalBattleRecord] = []

    try:
        for episode_idx in range(args.eval_battles):
            print(f"[eval {episode_idx + 1}/{args.eval_battles}] resetting battle")
            obs, _ = env.reset(seed=args.seed + 10_000 + episode_idx)
            print(f"[eval {episode_idx + 1}/{args.eval_battles}] reset complete")
            verify_obs(env, obs, args.verify_embedding)
            print(f"[eval {episode_idx + 1}/{args.eval_battles}] first observation ready")
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = predict_masked_action(model, obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = bool(terminated or truncated)
                steps += 1
                if not done:
                    verify_obs(env, obs, args.verify_embedding)

            battle = env.unwrapped.env.battle1
            record = EvalBattleRecord(
                battle_index=episode_idx + 1,
                won=bool(getattr(battle, "won", False)),
                lost=bool(getattr(battle, "lost", False)),
                steps=steps,
                reward=total_reward,
            )
            records.append(record)
            print(
                f"[eval {episode_idx + 1}/{args.eval_battles}] "
                f"won={record.won} steps={record.steps} reward={record.reward:.3f}"
            )
    finally:
        env.close()

    wins = sum(int(record.won) for record in records)
    losses = sum(int(record.lost) for record in records)
    draws = len(records) - wins - losses
    summary = EvalSummary(
        opponent=args.eval_opponent,
        wins=wins,
        losses=losses,
        draws=draws,
        avg_reward=float(np.mean([record.reward for record in records]) if records else 0.0),
        avg_steps=float(np.mean([record.steps for record in records]) if records else 0.0),
    )
    return summary, records


def build_model(args: argparse.Namespace, env: Monitor) -> PPO:
    if args.resume_from:
        print(f"Loading PPO checkpoint from {args.resume_from}")
        return PPO.load(args.resume_from, env=env, device=args.device)

    return PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        device=args.device,
        seed=args.seed,
        verbose=1,
    )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    run_name = args.run_name or make_run_name("ppo", args.seed)
    artifacts = prepare_run_artifacts(
        output_dir=args.output_dir,
        run_name=run_name,
        checkpoint_path=args.checkpoint_path,
    )

    env = create_wrapped_env(args, args.train_opponent)
    env.reset(seed=args.seed)
    model = build_model(args, env)

    try:
        if args.train_timesteps > 0:
            print(f"Starting PPO training for {args.train_timesteps} timesteps...")
            model.learn(total_timesteps=args.train_timesteps, progress_bar=False)
        else:
            print("Skipping PPO training because train_timesteps=0.")
    finally:
        env.close()

    model.save(artifacts["checkpoint_path"])
    print(f"Saved checkpoint to {artifacts['checkpoint_path']}")

    print(f"Training complete. Starting evaluation over {args.eval_battles} battle(s)...")
    summary, records = evaluate_policy(model, args)
    elapsed = time.perf_counter() - start

    payload = {
        "run_name": artifacts["run_name"],
        "battle_format": args.battle_format,
        "train_timesteps": args.train_timesteps,
        "eval_battles": args.eval_battles,
        "seed": args.seed,
        "train_opponent": args.train_opponent,
        "eval_opponent": args.eval_opponent,
        "verify_embedding": args.verify_embedding,
        "device": args.device,
        "resume_from": args.resume_from,
        "reward_config": REWARD_CONFIG,
        "checkpoint_path": artifacts["checkpoint_path"],
        "summary": {
            **asdict(summary),
            "win_rate": (summary.wins / args.eval_battles) if args.eval_battles else 0.0,
            "elapsed_seconds": elapsed,
        },
        "eval_records": [asdict(record) for record in records],
    }
    write_summary(artifacts["summary_path"], payload)

    print("\n=== PPO Summary ===")
    print(f"run_name: {artifacts['run_name']}")
    print(f"battle_format: {args.battle_format}")
    print(f"train_opponent: {args.train_opponent}")
    print(f"eval_opponent: {args.eval_opponent}")
    print(f"train_timesteps: {args.train_timesteps}")
    print(f"eval_battles: {args.eval_battles}")
    print(f"wins: {summary.wins}")
    print(f"losses: {summary.losses}")
    print(f"draws: {summary.draws}")
    print(f"win_rate: {summary.wins / args.eval_battles:.3f}")
    print(f"avg_reward: {summary.avg_reward:.3f}")
    print(f"avg_steps: {summary.avg_steps:.2f}")
    print(f"verify_embedding: {args.verify_embedding}")
    print(f"checkpoint_path: {artifacts['checkpoint_path']}")
    print(f"summary_path: {artifacts['summary_path']}")
    print(f"elapsed_seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
