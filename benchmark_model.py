from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import policies as _policies
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

from brent_agent import BrentsRLAgent, VECTOR_LENGTH
from experiment_io import make_run_name, write_summary
from opponents import OPPONENT_CHOICES, create_opponent

_ = _policies

DEFAULT_OPPONENTS = "random,max_base_power,simple_heuristic"
ALGO_CHOICES = ("ppo", "a2c", "dqn")
OPPONENT_ALIASES = {
    "random": "random",
    "max_base_power": "max_base_power",
    "maxpower": "max_base_power",
    "max_damage": "max_base_power",
    "max-damage": "max_base_power",
    "simple_heuristic": "simple_heuristic",
    "simple": "simple_heuristic",
    "heuristic": "simple_heuristic",
    "self_play": "self_play",
    "selfplay": "self_play",
    "adaptive": "adaptive",
}


class PredictiveModel(Protocol):
    def predict(
        self,
        observation: Any,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[Any, Any]:
        ...


@dataclass
class BenchmarkBattleRecord:
    opponent: str
    battle_index: int
    won: bool
    lost: bool
    steps: int
    reward: float


@dataclass
class MatchupSummary:
    opponent: str
    requested_battles: int
    completed_battles: int
    failures: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_reward: float
    avg_steps: float


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a trained SB3 model against built-in baseline opponents."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the saved SB3 checkpoint (.zip).")
    parser.add_argument("--algo", choices=ALGO_CHOICES, default="ppo")
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--opponents", default=DEFAULT_OPPONENTS)
    parser.add_argument("--n-battles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--log-level", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify-embedding", action="store_true")
    parser.add_argument("--output-dir", default="results/benchmarks")
    parser.add_argument("--run-name")
    parser.add_argument("--opponent-checkpoint",
                        help="Path to checkpoint for self_play opponent")
    return parser.parse_args()


def normalize_opponent_name(name: str) -> str:
    key = name.strip().lower()
    try:
        return OPPONENT_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported opponent '{name}'. Choices: {', '.join(OPPONENT_CHOICES)}"
        ) from exc


def parse_opponents(raw_value: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for chunk in raw_value.split(","):
        if not chunk.strip():
            continue
        normalized = normalize_opponent_name(chunk)
        if normalized not in seen:
            seen.add(normalized)
            names.append(normalized)
    if not names:
        raise ValueError("At least one opponent must be provided.")
    return names


def create_env(args: argparse.Namespace, opponent_name: str) -> Monitor:
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
        checkpoint_path=getattr(args, "opponent_checkpoint", None),
    )
    return Monitor(SingleAgentWrapper(env, opponent))


def validate_observation(env: Monitor, obs: dict[str, Any], verify_embedding: bool) -> None:
    vector = np.asarray(obs["observation"], dtype=np.float32)
    mask = np.asarray(obs["action_mask"], dtype=np.int64)

    if vector.shape != (VECTOR_LENGTH,):
        raise RuntimeError(f"Unexpected observation shape {vector.shape}")
    if mask.shape != (env.action_space.n,):
        raise RuntimeError(f"Unexpected action mask shape {mask.shape}")
    if np.any((mask != 0) & (mask != 1)):
        raise RuntimeError("Action mask must be binary.")
    if not np.any(mask == 1):
        raise RuntimeError("No legal actions available.")

    if verify_embedding:
        raw_env = env.unwrapped.env
        issues = raw_env.vector_builder.verify_battle_embedding(raw_env.battle1, vector)
        if issues:
            raise RuntimeError("Embedding verification failed: " + "; ".join(issues[:5]))


def load_model(args: argparse.Namespace, env: Monitor) -> PredictiveModel:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    algo_loaders = {
        "ppo": PPO,
        "a2c": A2C,
        "dqn": DQN,
    }
    model_cls = algo_loaders[args.algo]
    print(f"Loading {args.algo.upper()} checkpoint from {checkpoint_path}")
    return model_cls.load(str(checkpoint_path), env=env, device=args.device)


def predict_action(model: PredictiveModel, obs: dict[str, Any]) -> np.int64:
    action, _ = model.predict(obs, deterministic=True)
    return np.int64(np.asarray(action).reshape(-1)[0])


def run_episode(
    env: Monitor,
    model: PredictiveModel,
    seed: int,
    max_steps: int,
    verify_embedding: bool,
) -> BenchmarkBattleRecord:
    obs, _ = env.reset(seed=seed)
    validate_observation(env, obs, verify_embedding)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if steps >= max_steps:
            raise RuntimeError(f"Episode exceeded max_steps={max_steps} without termination.")

        action = predict_action(model, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1

        if not (terminated or truncated):
            validate_observation(env, obs, verify_embedding)

    battle = env.unwrapped.env.battle1
    return BenchmarkBattleRecord(
        opponent="",
        battle_index=0,
        won=bool(getattr(battle, "won", False)),
        lost=bool(getattr(battle, "lost", False)),
        steps=steps,
        reward=total_reward,
    )


def benchmark_opponent(
    args: argparse.Namespace,
    opponent_name: str,
) -> tuple[MatchupSummary, list[BenchmarkBattleRecord], dict[str, Any], dict[str, Any]]:
    env = create_env(args, opponent_name)
    model = load_model(args, env)
    records: list[BenchmarkBattleRecord] = []
    failures = 0
    decision_audit_report: dict[str, Any] = {}
    tactical_shaping_report: dict[str, Any] = {}

    try:
        for battle_idx in range(args.n_battles):
            try:
                record = run_episode(
                    env=env,
                    model=model,
                    seed=args.seed + battle_idx,
                    max_steps=args.max_steps,
                    verify_embedding=args.verify_embedding,
                )
            except Exception as exc:
                failures += 1
                print(f"[{opponent_name} {battle_idx + 1}/{args.n_battles}] failure: {exc}")
                try:
                    env.close()
                except OSError:
                    pass
                env = create_env(args, opponent_name)
                model = load_model(args, env)
                continue

            record.opponent = opponent_name
            record.battle_index = battle_idx + 1
            records.append(record)
            print(
                f"[{opponent_name} {battle_idx + 1}/{args.n_battles}] "
                f"won={record.won} steps={record.steps} reward={record.reward:.3f}"
            )
        decision_audit_report = env.unwrapped.env.get_decision_audit_report()
        tactical_shaping_report = env.unwrapped.env.get_tactical_shaping_report()
    finally:
        try:
            env.close()
        except OSError:
            pass

    wins = sum(int(record.won) for record in records)
    losses = sum(int(record.lost) for record in records)
    draws = len(records) - wins - losses
    summary = MatchupSummary(
        opponent=opponent_name,
        requested_battles=args.n_battles,
        completed_battles=len(records),
        failures=failures,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=(wins / len(records)) if records else 0.0,
        avg_reward=float(np.mean([record.reward for record in records]) if records else 0.0),
        avg_steps=float(np.mean([record.steps for record in records]) if records else 0.0),
    )
    return summary, records, decision_audit_report, tactical_shaping_report


def print_matchup_summary(
    summary: MatchupSummary,
    audit_report: dict[str, Any],
    tactical_shaping_report: dict[str, Any],
) -> None:
    print(
        f"{summary.opponent}: wins={summary.wins} losses={summary.losses} draws={summary.draws} "
        f"failures={summary.failures} win_rate={summary.win_rate:.3f} "
        f"avg_reward={summary.avg_reward:.3f} avg_steps={summary.avg_steps:.2f}"
    )
    print(f"{summary.opponent} audit: {format_decision_audit_report(audit_report)}")
    print(f"{summary.opponent} shaping: {format_tactical_shaping_report(tactical_shaping_report)}")


def main() -> None:
    args = parse_args()
    opponents = parse_opponents(args.opponents)
    run_name = args.run_name or make_run_name("benchmark", args.seed)
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    matchup_summaries: list[MatchupSummary] = []
    battle_records: list[BenchmarkBattleRecord] = []
    matchup_audit_reports: dict[str, dict[str, Any]] = {}
    matchup_tactical_shaping_reports: dict[str, dict[str, Any]] = {}
    start_time = time.perf_counter()

    for opponent_name in opponents:
        print(f"\n=== Benchmarking vs {opponent_name} ===")
        summary, records, audit_report, tactical_shaping_report = benchmark_opponent(args, opponent_name)
        matchup_summaries.append(summary)
        battle_records.extend(records)
        matchup_audit_reports[opponent_name] = audit_report
        matchup_tactical_shaping_reports[opponent_name] = tactical_shaping_report
        print_matchup_summary(summary, audit_report, tactical_shaping_report)

    completed_battles = sum(summary.completed_battles for summary in matchup_summaries)
    total_wins = sum(summary.wins for summary in matchup_summaries)
    total_losses = sum(summary.losses for summary in matchup_summaries)
    total_draws = sum(summary.draws for summary in matchup_summaries)
    total_failures = sum(summary.failures for summary in matchup_summaries)
    elapsed = time.perf_counter() - start_time
    total_decision_count = sum(
        report.get("decision_count", 0) for report in matchup_audit_reports.values()
    )
    total_move_checks = sum(
        report.get("move_checks", 0) for report in matchup_audit_reports.values()
    )
    total_switch_actions = sum(
        report.get("switch_actions", 0) for report in matchup_audit_reports.values()
    )
    total_flagged_actions = sum(
        report.get("flagged_actions", 0) for report in matchup_audit_reports.values()
    )
    overall_audit_counts: dict[str, int] = {}
    overall_audit_samples: dict[str, list[dict[str, Any]]] = {}
    for report in matchup_audit_reports.values():
        for category, count in report.get("counts", {}).items():
            overall_audit_counts[category] = overall_audit_counts.get(category, 0) + int(count)
        for category, samples in report.get("samples", {}).items():
            bucket = overall_audit_samples.setdefault(category, [])
            remaining = 20 - len(bucket)
            if remaining > 0:
                bucket.extend(list(samples)[:remaining])
    overall_decision_audit_report = {
        "decision_count": total_decision_count,
        "move_checks": total_move_checks,
        "switch_actions": total_switch_actions,
        "switch_rate": (total_switch_actions / total_decision_count) if total_decision_count else 0.0,
        "flagged_actions": total_flagged_actions,
        "flagged_action_rate": (
            total_flagged_actions / total_move_checks if total_move_checks else 0.0
        ),
        "counts": dict(sorted(overall_audit_counts.items())),
        "samples": overall_audit_samples,
    }
    total_shaped_actions = sum(
        report.get("shaped_actions", 0) for report in matchup_tactical_shaping_reports.values()
    )
    total_rewarded_actions = sum(
        report.get("rewarded_actions", 0) for report in matchup_tactical_shaping_reports.values()
    )
    total_penalized_actions = sum(
        report.get("penalized_actions", 0) for report in matchup_tactical_shaping_reports.values()
    )
    total_shaping = sum(
        float(report.get("total_shaping", 0.0)) for report in matchup_tactical_shaping_reports.values()
    )
    total_positive_shaping = sum(
        float(report.get("positive_total", 0.0)) for report in matchup_tactical_shaping_reports.values()
    )
    total_negative_shaping = sum(
        float(report.get("negative_total", 0.0)) for report in matchup_tactical_shaping_reports.values()
    )
    overall_shaping_counts: dict[str, int] = {}
    overall_shaping_totals: dict[str, float] = {}
    for report in matchup_tactical_shaping_reports.values():
        for category, count in report.get("counts", {}).items():
            overall_shaping_counts[category] = overall_shaping_counts.get(category, 0) + int(count)
        for category, total in report.get("totals", {}).items():
            overall_shaping_totals[category] = overall_shaping_totals.get(category, 0.0) + float(total)
    overall_tactical_shaping_report = {
        "decision_count": total_decision_count,
        "move_checks": sum(report.get("move_checks", 0) for report in matchup_tactical_shaping_reports.values()),
        "switch_actions": total_switch_actions,
        "switch_rate": (total_switch_actions / total_decision_count) if total_decision_count else 0.0,
        "shaped_actions": total_shaped_actions,
        "rewarded_actions": total_rewarded_actions,
        "penalized_actions": total_penalized_actions,
        "shaped_action_rate": (total_shaped_actions / total_decision_count) if total_decision_count else 0.0,
        "total_shaping": total_shaping,
        "positive_total": total_positive_shaping,
        "negative_total": total_negative_shaping,
        "counts": dict(sorted(overall_shaping_counts.items())),
        "totals": {
            category: round(total, 3)
            for category, total in sorted(overall_shaping_totals.items())
        },
    }

    payload = {
        "run_name": run_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "algo": args.algo,
        "battle_format": args.battle_format,
        "opponents": opponents,
        "n_battles_per_opponent": args.n_battles,
        "seed": args.seed,
        "device": args.device,
        "verify_embedding": args.verify_embedding,
        "summary": {
            "completed_battles": completed_battles,
            "wins": total_wins,
            "losses": total_losses,
            "draws": total_draws,
            "failures": total_failures,
            "overall_win_rate": (total_wins / completed_battles) if completed_battles else 0.0,
            "avg_reward": float(np.mean([record.reward for record in battle_records]) if battle_records else 0.0),
            "avg_steps": float(np.mean([record.steps for record in battle_records]) if battle_records else 0.0),
            "elapsed_seconds": elapsed,
        },
        "overall_decision_audit_report": overall_decision_audit_report,
        "overall_tactical_shaping_report": overall_tactical_shaping_report,
        "matchup_decision_audit_reports": matchup_audit_reports,
        "matchup_tactical_shaping_reports": matchup_tactical_shaping_reports,
        "matchups": [asdict(summary) for summary in matchup_summaries],
        "battle_records": [asdict(record) for record in battle_records],
    }
    summary_path = run_dir / "summary.json"
    write_summary(str(summary_path), payload)

    print("\n=== Overall Summary ===")
    print(f"checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"algo: {args.algo}")
    print(f"opponents: {', '.join(opponents)}")
    print(f"completed_battles: {completed_battles}")
    print(f"wins: {total_wins}")
    print(f"losses: {total_losses}")
    print(f"draws: {total_draws}")
    print(f"failures: {total_failures}")
    print(f"overall_win_rate: {(total_wins / completed_battles) if completed_battles else 0.0:.3f}")
    print(
        f"avg_reward: "
        f"{float(np.mean([record.reward for record in battle_records]) if battle_records else 0.0):.3f}"
    )
    print(
        f"avg_steps: "
        f"{float(np.mean([record.steps for record in battle_records]) if battle_records else 0.0):.2f}"
    )
    print(f"overall_decision_audit: {format_decision_audit_report(overall_decision_audit_report)}")
    print(f"overall_tactical_shaping: {format_tactical_shaping_report(overall_tactical_shaping_report)}")
    print(f"summary_path: {summary_path}")
    print(f"elapsed_seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
