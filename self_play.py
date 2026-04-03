"""Self-play opponent: wraps a frozen PPO checkpoint as a poke_env Player."""
from __future__ import annotations

import os
import random
from typing import Optional, Union

import numpy as np
from stable_baselines3 import PPO

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.ps_client import AccountConfiguration

from brent_agent import VECTOR_LENGTH, BrentObservationVectorBuilder


class SelfPlayPlayer(Player):
    """A poke_env Player that chooses moves by running inference on a frozen
    PPO checkpoint. Used as an opponent for self-play training.

    Maintains its own observation builder so it can construct the obs vector
    from the opponent's perspective (battle2 in SingleAgentWrapper).
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        battle_format: str = "gen9randombattle",
        log_level: int = 40,
        start_listening: bool = False,
    ):
        uid = f"{os.getpid()}{random.randint(1000, 9999)}"
        super().__init__(
            account_configuration=AccountConfiguration.generate(
                f"SelfPlay{uid}", rand=False,
            ),
            battle_format=battle_format,
            start_listening=start_listening,
            log_level=log_level,
        )
        self.model = PPO.load(checkpoint_path, device=device)
        self.vector_builder = BrentObservationVectorBuilder()

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Build obs from this battle's perspective, run policy, return order."""
        try:
            obs = self.vector_builder.embed_battle(battle)
            mask = self._build_action_mask(battle)

            batched_obs = {
                "observation": np.expand_dims(obs, axis=0),
                "action_mask": np.expand_dims(mask, axis=0),
            }
            action, _ = self.model.predict(batched_obs, deterministic=True)
            action_idx = np.int64(np.asarray(action).reshape(-1)[0])

            order = SinglesEnv.action_to_order(action_idx, battle, fake=False, strict=False)
            return order
        except Exception:
            return self.choose_default_move(battle)

    def _build_action_mask(self, battle: AbstractBattle) -> np.ndarray:
        """Build the action mask matching SinglesEnv's action space."""
        n_actions = 26  # 6 switch + 4 move + 4 mega + 4 z + 4 dmax + 4 tera
        mask = np.zeros(n_actions, dtype=np.int64)

        # Switches: actions 0-5
        available_switches = battle.available_switches or []
        team_list = list(battle.team.values())
        non_active = [mon for mon in team_list if not mon.active]
        for i, mon in enumerate(non_active[:6]):
            if mon in available_switches and not mon.fainted:
                mask[i] = 1

        # Moves: actions 6-9
        available_moves = battle.available_moves or []
        for i, move in enumerate(available_moves[:4]):
            mask[6 + i] = 1

        # Tera moves: actions 22-25
        if getattr(battle, "can_tera", False) and available_moves:
            for i in range(min(len(available_moves), 4)):
                mask[22 + i] = 1

        # Force switch: only switches are legal
        if getattr(battle, "force_switch", False):
            for i in range(6, n_actions):
                mask[i] = 0

        # Ensure at least one action is legal
        if not np.any(mask == 1):
            mask[0] = 1  # fallback

        return mask


class CheckpointPool:
    """Manages a pool of historical checkpoints for league-style self-play.
    Samples a random checkpoint from the pool as the opponent each game."""

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.checkpoints: list[str] = []

    def add(self, checkpoint_path: str) -> None:
        """Add a checkpoint to the pool, evicting oldest if full."""
        if checkpoint_path not in self.checkpoints:
            self.checkpoints.append(checkpoint_path)
            if len(self.checkpoints) > self.max_size:
                self.checkpoints.pop(0)

    def sample(self) -> Optional[str]:
        """Sample a random checkpoint from the pool."""
        if not self.checkpoints:
            return None
        return random.choice(self.checkpoints)

    def latest(self) -> Optional[str]:
        """Return the most recently added checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None

    def __len__(self) -> int:
        return len(self.checkpoints)
