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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from brent_agent import BrentsRLAgent, REWARD_CONFIG, VECTOR_LENGTH
from experiment_io import make_run_name, prepare_run_artifacts, write_summary
from opponents import OPPONENT_CHOICES, create_opponent
from policies import MaskedActorCriticPolicy


def format_penalty_report(report: dict[str, Any]) -> str:
    counts = report.get("counts", {})
    counts_text = ", ".join(f"{key}={value}" for key, value in counts.items()) if counts else "none"
    return (
        "decision_count={decision_count} switch_actions={switch_actions} switch_rate={switch_rate:.3f} "
        "move_checks={move_checks} penalized_actions={penalized_actions} "
        "penalized_action_rate={penalized_action_rate:.3f} total_penalty={total_penalty:.3f} "
        "counts=[{counts}]"
    ).format(
        decision_count=report.get("decision_count", 0),
        move_checks=report.get("move_checks", 0),
        penalized_actions=report.get("penalized_actions", 0),
        switch_actions=report.get("switch_actions", 0),
        switch_rate=report.get("switch_rate", 0.0),
        penalized_action_rate=report.get("penalized_action_rate", 0.0),
        total_penalty=report.get("total_penalty", 0.0),
        counts=counts_text,
    )


def format_tactical_shaping_report(report: dict[str, Any]) -> str:
    counts = report.get("counts", {})
    totals = report.get("totals", {})
    counts_text = ", ".join(f"{key}={value}" for key, value in counts.items()) if counts else "none"
    totals_text = ", ".join(f"{key}={value:.3f}" for key, value in totals.items()) if totals else "none"
    return (
        "decision_count={decision_count} switch_actions={switch_actions} switch_rate={switch_rate:.3f} "
        "move_checks={move_checks} shaped_actions={shaped_actions} rewarded_actions={rewarded_actions} "
        "penalized_actions={penalized_actions} shaped_action_rate={shaped_action_rate:.3f} "
        "total_shaping={total_shaping:.3f} positive_total={positive_total:.3f} "
        "negative_total={negative_total:.3f} counts=[{counts}] totals=[{totals}]"
    ).format(
        decision_count=report.get("decision_count", 0),
        move_checks=report.get("move_checks", 0),
        shaped_actions=report.get("shaped_actions", 0),
        rewarded_actions=report.get("rewarded_actions", 0),
        penalized_actions=report.get("penalized_actions", 0),
        switch_actions=report.get("switch_actions", 0),
        switch_rate=report.get("switch_rate", 0.0),
        shaped_action_rate=report.get("shaped_action_rate", 0.0),
        total_shaping=report.get("total_shaping", 0.0),
        positive_total=report.get("positive_total", 0.0),
        negative_total=report.get("negative_total", 0.0),
        counts=counts_text,
        totals=totals_text,
    )


def format_decision_audit_report(report: dict[str, Any]) -> str:
    counts = report.get("counts", {})
    counts_text = ", ".join(f"{key}={value}" for key, value in counts.items()) if counts else "none"
    return (
        "decision_count={decision_count} switch_actions={switch_actions} switch_rate={switch_rate:.3f} "
        "move_checks={move_checks} flagged_actions={flagged_actions} "
        "flagged_action_rate={flagged_action_rate:.3f} counts=[{counts}]"
    ).format(
        decision_count=report.get("decision_count", 0),
        move_checks=report.get("move_checks", 0),
        flagged_actions=report.get("flagged_actions", 0),
        switch_actions=report.get("switch_actions", 0),
        switch_rate=report.get("switch_rate", 0.0),
        flagged_action_rate=report.get("flagged_action_rate", 0.0),
        counts=counts_text,
    )


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
    parser.add_argument("--train-opponent", choices=OPPONENT_CHOICES, default="random",
                        help="Single training opponent (used when --n-envs=1)")
    parser.add_argument("--train-opponents",
                        help="Comma-separated opponent mix for SubprocVecEnv (e.g. 'random,random,max_base_power,simple_heuristic')")
    parser.add_argument("--eval-opponent", choices=OPPONENT_CHOICES, default="random")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments (1=single env, >1=SubprocVecEnv)")
    parser.add_argument("--output-dir", default="results/ppo")
    parser.add_argument("--run-name")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--resume-from")
    parser.add_argument("--save-freq", type=int, default=50000, help="Save checkpoint every X steps")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluate every X steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="PPO learning rate (default 3e-4, lower for curriculum transitions)")
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


def _make_env_fn(opponent_name: str, battle_format: str, log_level: int):
    """Lazy factory — all poke_env objects are created inside the subprocess."""
    def _init():
        env = BrentsRLAgent(
            battle_format=battle_format,
            log_level=log_level,
            open_timeout=None,
            strict=True,
        )
        opponent = create_opponent(
            opponent_name,
            battle_format=battle_format,
            log_level=log_level,
            start_listening=False,
        )
        return Monitor(SingleAgentWrapper(env, opponent))
    return _init


def _parse_opponent_mix(args: argparse.Namespace) -> list[str]:
    """Build the list of opponents for SubprocVecEnv workers."""
    if args.train_opponents:
        opponents = [o.strip() for o in args.train_opponents.split(",")]
        for o in opponents:
            if o not in OPPONENT_CHOICES:
                raise ValueError(f"Unknown opponent '{o}'. Choices: {', '.join(OPPONENT_CHOICES)}")
        return opponents
    return [args.train_opponent] * args.n_envs


def create_vec_env(args: argparse.Namespace) -> SubprocVecEnv:
    """Create a SubprocVecEnv with opponent blending. Each worker lazily
    instantiates its own poke_env objects to avoid asyncio event loop issues."""
    opponents = _parse_opponent_mix(args)
    if len(opponents) != args.n_envs:
        raise ValueError(
            f"--train-opponents has {len(opponents)} entries but --n-envs is {args.n_envs}; they must match"
        )
    return SubprocVecEnv([
        _make_env_fn(opp, args.battle_format, args.log_level)
        for opp in opponents
    ])


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
) -> tuple[EvalSummary, list[EvalBattleRecord], dict[str, Any], dict[str, Any]]:
    env = create_wrapped_env(args, args.eval_opponent)
    records: list[EvalBattleRecord] = []
    eval_decision_audit_report: dict[str, Any] = {}
    eval_tactical_shaping_report: dict[str, Any] = {}

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
        eval_decision_audit_report = env.unwrapped.env.get_decision_audit_report()
        eval_tactical_shaping_report = env.unwrapped.env.get_tactical_shaping_report()
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
    return summary, records, eval_decision_audit_report, eval_tactical_shaping_report


class PokeEnvEvalCallback(BaseCallback):
    """EvalCallback that creates a fresh poke-env each cycle to avoid stale battle state."""

    def __init__(
        self,
        args: argparse.Namespace,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: str,
        log_path: str,
        tracked_env: BrentsRLAgent | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.args = args
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.tracked_env = tracked_env
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        if self.verbose and self.tracked_env is not None:
            print(
                f"[train penalties @ {self.num_timesteps} steps] "
                f"{format_penalty_report(self.tracked_env.get_strategic_penalty_report())}"
            )
            print(
                f"[train shaping @ {self.num_timesteps} steps] "
                f"{format_tactical_shaping_report(self.tracked_env.get_tactical_shaping_report())}"
            )
            print(
                f"[train audit @ {self.num_timesteps} steps] "
                f"{format_decision_audit_report(self.tracked_env.get_decision_audit_report())}"
            )

        env = create_wrapped_env(self.args, self.args.eval_opponent)
        rewards: list[float] = []
        steps: list[int] = []
        wins = 0
        losses = 0
        eval_decision_audit_report: dict[str, Any] = {}
        eval_tactical_shaping_report: dict[str, Any] = {}
        try:
            for ep in range(self.n_eval_episodes):
                obs, _ = env.reset()
                done, total_reward = False, 0.0
                episode_steps = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    done = bool(terminated or truncated)
                    episode_steps += 1
                rewards.append(total_reward)
                steps.append(episode_steps)
                battle = env.unwrapped.env.battle1
                wins += int(bool(getattr(battle, "won", False)))
                losses += int(bool(getattr(battle, "lost", False)))
            eval_decision_audit_report = env.unwrapped.env.get_decision_audit_report()
            eval_tactical_shaping_report = env.unwrapped.env.get_tactical_shaping_report()
        finally:
            try:
                env.close()
            except OSError:
                pass

        mean_reward = float(np.mean(rewards))
        mean_steps = float(np.mean(steps) if steps else 0.0)
        draws = self.n_eval_episodes - wins - losses
        win_rate = wins / self.n_eval_episodes if self.n_eval_episodes else 0.0
        if self.verbose:
            print(
                f"[eval @ {self.num_timesteps} steps] wins={wins} losses={losses} "
                f"draws={draws} win_rate={win_rate:.3f} mean_reward={mean_reward:.3f} "
                f"avg_steps={mean_steps:.2f}"
            )
            print(
                f"[eval audit @ {self.num_timesteps} steps] "
                f"{format_decision_audit_report(eval_decision_audit_report)}"
            )
            print(
                f"[eval shaping @ {self.num_timesteps} steps] "
                f"{format_tactical_shaping_report(eval_tactical_shaping_report)}"
            )

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            os.makedirs(self.best_model_save_path, exist_ok=True)
            self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            if self.verbose:
                print(f"  New best model saved (reward={mean_reward:.3f})")

        return True


def build_model(args: argparse.Namespace, env: Monitor) -> PPO:
    if args.resume_from:
        print(f"Loading PPO checkpoint from {args.resume_from}")
        model = PPO.load(args.resume_from, env=env, device=args.device, tensorboard_log=os.path.join(args.output_dir, "logs"))
        model.learning_rate = args.learning_rate
        print(f"  learning_rate overridden to {args.learning_rate}")
        return model

    return PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        ent_coef=0.02,
        device=args.device,
        seed=args.seed,
        verbose=1,
        tensorboard_log=os.path.join(args.output_dir, "logs"),
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

    print(f"REWARD_CONFIG: {REWARD_CONFIG}")

    use_vec_env = args.n_envs > 1
    train_env: BrentsRLAgent | None = None

    if use_vec_env:
        print(f"Creating SubprocVecEnv with {args.n_envs} workers...")
        env = create_vec_env(args)
        # SubprocVecEnv doesn't expose inner envs; train reports unavailable
        train_env = None
    else:
        env = create_wrapped_env(args, args.train_opponent)
        env.reset(seed=args.seed)
        train_env = env.unwrapped.env

    model = build_model(args, env)

    try:
        if args.train_timesteps > 0:
            print(f"Starting PPO training for {args.train_timesteps} timesteps...")

            # Create the callbacks
            checkpoint_callback = CheckpointCallback(
                save_freq=args.save_freq,
                save_path=os.path.join(artifacts["run_dir"], "checkpoints"),
                name_prefix="rl_model",
            )

            eval_callback = PokeEnvEvalCallback(
                args=args,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_battles,
                best_model_save_path=os.path.join(artifacts["run_dir"], "best_model"),
                log_path=artifacts["run_dir"],
                tracked_env=train_env,
            )

            model.learn(
                total_timesteps=args.train_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=False,
            )
        else:
            print("Skipping PPO training because train_timesteps=0.")
    finally:
        env.close()

    model.save(artifacts["checkpoint_path"])
    print(f"Saved checkpoint to {artifacts['checkpoint_path']}")

    if train_env is not None:
        train_penalty_report = train_env.get_strategic_penalty_report()
        train_tactical_shaping_report = train_env.get_tactical_shaping_report()
        train_decision_audit_report = train_env.get_decision_audit_report()
    else:
        train_penalty_report = {}
        train_tactical_shaping_report = {}
        train_decision_audit_report = {}
        print("(SubprocVecEnv: per-worker training reports not available)")
    print(f"Training strategic penalties: {format_penalty_report(train_penalty_report)}")
    print(f"Training tactical shaping: {format_tactical_shaping_report(train_tactical_shaping_report)}")
    print(f"Training decision audit: {format_decision_audit_report(train_decision_audit_report)}")

    print(f"Training complete. Starting evaluation over {args.eval_battles} battle(s)...")
    summary, records, eval_decision_audit_report, eval_tactical_shaping_report = evaluate_policy(model, args)
    elapsed = time.perf_counter() - start

    payload = {
        "run_name": artifacts["run_name"],
        "battle_format": args.battle_format,
        "train_timesteps": args.train_timesteps,
        "eval_battles": args.eval_battles,
        "seed": args.seed,
        "train_opponent": args.train_opponent,
        "train_opponents": args.train_opponents,
        "n_envs": args.n_envs,
        "eval_opponent": args.eval_opponent,
        "verify_embedding": args.verify_embedding,
        "device": args.device,
        "resume_from": args.resume_from,
        "reward_config": REWARD_CONFIG,
        "checkpoint_path": artifacts["checkpoint_path"],
        "train_penalty_report": train_penalty_report,
        "train_tactical_shaping_report": train_tactical_shaping_report,
        "train_decision_audit_report": train_decision_audit_report,
        "eval_decision_audit_report": eval_decision_audit_report,
        "eval_tactical_shaping_report": eval_tactical_shaping_report,
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
    print(f"train_penalties: {format_penalty_report(train_penalty_report)}")
    print(f"train_tactical_shaping: {format_tactical_shaping_report(train_tactical_shaping_report)}")
    print(f"train_decision_audit: {format_decision_audit_report(train_decision_audit_report)}")
    print(f"eval_tactical_shaping: {format_tactical_shaping_report(eval_tactical_shaping_report)}")
    print(f"eval_decision_audit: {format_decision_audit_report(eval_decision_audit_report)}")
    print(f"verify_embedding: {args.verify_embedding}")
    print(f"checkpoint_path: {artifacts['checkpoint_path']}")
    print(f"summary_path: {artifacts['summary_path']}")
    print(f"elapsed_seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
