from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from brent_agent import BrentsRLAgent, VECTOR_LENGTH
from opponents import OPPONENT_CHOICES, create_opponent

PolicyMode = Literal["random", "linear"]


@dataclass
class LinearPolicy:
    weights: np.ndarray
    bias: np.ndarray

    @classmethod
    def random_init(cls, action_dim: int, seed: int) -> "LinearPolicy":
        rng = np.random.default_rng(seed)
        weights = rng.standard_normal((action_dim, VECTOR_LENGTH), dtype=np.float32)
        bias = rng.standard_normal(action_dim, dtype=np.float32)
        return cls(weights=weights, bias=bias)

    def choose_action(self, obs: dict[str, Any]) -> np.int64:
        vector = np.asarray(obs["observation"], dtype=np.float32)
        mask = np.asarray(obs["action_mask"], dtype=np.int64)
        logits = self.weights @ vector + self.bias
        logits = np.where(mask == 1, logits, -np.inf)
        legal_actions = np.flatnonzero(mask == 1)
        if legal_actions.size == 0:
            raise RuntimeError("No legal actions available for linear policy.")
        return np.int64(np.argmax(logits))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate BrentsRLAgent in live poke-env battles."
    )
    parser.add_argument("--mode", choices=("random", "linear"), default="random")
    parser.add_argument("--n-battles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--log-level", type=int, default=40)
    parser.add_argument("--verify-embedding", action="store_true")
    parser.add_argument("--opponent", choices=OPPONENT_CHOICES, default="random")
    return parser.parse_args()


def create_env(args: argparse.Namespace) -> SingleAgentWrapper:
    env = BrentsRLAgent(
        battle_format=args.battle_format,
        log_level=args.log_level,
        open_timeout=None,
        strict=True,
    )
    opponent = create_opponent(
        args.opponent,
        battle_format=args.battle_format,
        log_level=args.log_level,
        start_listening=False,
    )
    return SingleAgentWrapper(env, opponent)


def validate_observation(obs: dict[str, Any], action_dim: int) -> None:
    if "observation" not in obs or "action_mask" not in obs:
        raise RuntimeError("Observation dict is missing observation or action_mask.")

    vector = np.asarray(obs["observation"], dtype=np.float32)
    mask = np.asarray(obs["action_mask"], dtype=np.int64)

    if vector.shape != (VECTOR_LENGTH,):
        raise RuntimeError(f"Unexpected observation shape: {vector.shape}")
    if mask.shape != (action_dim,):
        raise RuntimeError(f"Unexpected action mask shape: {mask.shape}")
    if np.any((mask != 0) & (mask != 1)):
        raise RuntimeError("Action mask must be binary.")
    if not np.any(mask == 1):
        raise RuntimeError("Action mask does not contain any legal actions.")


def choose_random_legal_action(obs: dict[str, Any], rng: np.random.Generator) -> np.int64:
    mask = np.asarray(obs["action_mask"], dtype=np.int64)
    legal_actions = np.flatnonzero(mask == 1)
    if legal_actions.size == 0:
        raise RuntimeError("No legal actions available for random policy.")
    return np.int64(rng.choice(legal_actions))


def run_episode(
    env: SingleAgentWrapper,
    mode: PolicyMode,
    rng: np.random.Generator,
    linear_policy: LinearPolicy | None,
    seed: int,
    max_steps: int,
    verify_embedding: bool,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    validate_observation(obs, env.action_space.n)
    verify_steps = 0
    if verify_embedding:
        issues = env.env.vector_builder.verify_battle_embedding(
            env.env.battle1,
            np.asarray(obs["observation"], dtype=np.float32),
        )
        if issues:
            raise RuntimeError("embedding verification failed on reset: " + "; ".join(issues[:5]))
        verify_steps += 1

    steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if steps >= max_steps:
            raise RuntimeError(
                f"Episode exceeded max_steps={max_steps} without termination."
            )

        if mode == "random":
            action = choose_random_legal_action(obs, rng)
        else:
            assert linear_policy is not None
            action = linear_policy.choose_action(obs)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1

        if not (terminated or truncated):
            validate_observation(obs, env.action_space.n)
            if verify_embedding:
                issues = env.env.vector_builder.verify_battle_embedding(
                    env.env.battle1,
                    np.asarray(obs["observation"], dtype=np.float32),
                )
                if issues:
                    raise RuntimeError(
                        "embedding verification failed during step: " + "; ".join(issues[:5])
                    )
                verify_steps += 1

    battle = env.env.battle1
    return {
        "won": bool(getattr(battle, "won", False)),
        "lost": bool(getattr(battle, "lost", False)),
        "steps": steps,
        "reward": total_reward,
        "verify_steps": verify_steps,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    linear_policy: LinearPolicy | None = None

    env: SingleAgentWrapper | None = None
    completed = 0
    wins = 0
    losses = 0
    failures = 0
    total_steps = 0
    total_reward = 0.0
    total_verify_steps = 0
    start_time = time.perf_counter()

    try:
        env = create_env(args)
        action_dim = env.action_space.n
        if args.mode == "linear":
            linear_policy = LinearPolicy.random_init(
                action_dim=action_dim,
                seed=args.seed + 1,
            )

        for battle_idx in range(args.n_battles):
            try:
                result = run_episode(
                    env=env,
                    mode=args.mode,
                    rng=rng,
                    linear_policy=linear_policy,
                    seed=args.seed + battle_idx,
                    max_steps=args.max_steps,
                    verify_embedding=args.verify_embedding,
                )
            except Exception as exc:
                failures += 1
                print(f"[battle {battle_idx + 1}] failure: {exc}")
                env.close()
                env = create_env(args)
                continue

            completed += 1
            wins += int(result["won"])
            losses += int(result["lost"])
            total_steps += int(result["steps"])
            total_reward += float(result["reward"])
            total_verify_steps += int(result["verify_steps"])

            print(
                f"[battle {battle_idx + 1}/{args.n_battles}] "
                f"won={result['won']} steps={result['steps']} reward={result['reward']:.3f}"
            )
    finally:
        if env is not None:
            env.close()

    elapsed = time.perf_counter() - start_time
    avg_steps = total_steps / completed if completed else 0.0
    avg_reward = total_reward / completed if completed else 0.0
    win_rate = wins / completed if completed else 0.0
    fallback_report = (
        env.env.vector_builder.get_fallback_report() if env is not None else {"counts": [], "samples": []}
    )

    print("\n=== Summary ===")
    print(f"mode: {args.mode}")
    print(f"battle_format: {args.battle_format}")
    print(f"opponent: {args.opponent}")
    print(f"requested_battles: {args.n_battles}")
    print(f"completed_battles: {completed}")
    print(f"wins: {wins}")
    print(f"losses: {losses}")
    print(f"failures: {failures}")
    print(f"win_rate: {win_rate:.3f}")
    print(f"avg_steps: {avg_steps:.2f}")
    print(f"avg_reward: {avg_reward:.3f}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"verify_embedding: {args.verify_embedding}")
    print(f"verified_states: {total_verify_steps}")
    print(f"fallback_count_kinds: {len(fallback_report['counts'])}")

    if fallback_report["counts"]:
        print("\n=== Fallback Counts ===")
        for item in fallback_report["counts"][:10]:
            print(
                f"{item['accessor']} move={item['move_id']} raw_type={item['raw_type']} count={item['count']}"
            )

    if fallback_report["samples"]:
        print("\n=== Fallback Samples ===")
        for sample in fallback_report["samples"][:10]:
            print(
                f"{sample['accessor']} move={sample['move_id']} raw={sample['raw_value_repr']} "
                f"entry_keys={sample['entry_keys']}"
            )


if __name__ == "__main__":
    main()
