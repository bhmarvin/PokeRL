"""Self-play opponent: wraps a PPO checkpoint as a poke_env Player.

Supports periodic weight refresh from a checkpoint file on disk,
enabling true self-play where the opponent evolves during training.
"""
from __future__ import annotations

import os
import random
import time
from typing import Optional, Union

import numpy as np
from stable_baselines3 import PPO

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.ps_client import AccountConfiguration

from brent_agent import VECTOR_LENGTH, BrentObservationVectorBuilder
from brent_agent.agent import BrentsRLAgent


class SelfPlayPlayer(Player):
    """A poke_env Player that chooses moves via a PPO checkpoint.

    Supports automatic weight refresh: periodically checks if the
    checkpoint file has been updated on disk and reloads weights.
    This enables true self-play where the opponent evolves.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        battle_format: str = "gen9randombattle",
        log_level: int = 40,
        start_listening: bool = False,
        refresh_interval: int = 200,
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
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._refresh_interval = refresh_interval
        self._move_count = 0
        self._last_mtime: float = 0.0

        self.model = PPO.load(checkpoint_path, device=device)
        self.vector_builder = BrentObservationVectorBuilder()
        self._record_mtime()

    def _record_mtime(self) -> None:
        try:
            self._last_mtime = os.path.getmtime(self._checkpoint_path)
        except OSError:
            pass

    def _maybe_refresh(self) -> None:
        """Reload weights if checkpoint file has been updated on disk."""
        self._move_count += 1
        if self._move_count % self._refresh_interval != 0:
            return
        try:
            current_mtime = os.path.getmtime(self._checkpoint_path)
        except OSError:
            return
        if current_mtime > self._last_mtime:
            try:
                self.model = PPO.load(self._checkpoint_path, device=self._device)
                self._last_mtime = current_mtime
            except Exception:
                pass  # keep old model if reload fails

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Build obs from this battle's perspective, run policy, return order."""
        self._maybe_refresh()
        try:
            obs = self.vector_builder.embed_battle(battle)
            mask = self._build_action_mask(battle)

            batched_obs = {
                "observation": np.expand_dims(obs, axis=0),
                "action_mask": np.expand_dims(mask, axis=0),
            }
            action, _ = self.model.predict(batched_obs, deterministic=False)
            action_idx = np.int64(np.asarray(action).reshape(-1)[0])

            order = SinglesEnv.action_to_order(action_idx, battle, fake=False, strict=False)
            return order
        except Exception:
            return self.choose_default_move(battle)

    def _build_action_mask(self, battle: AbstractBattle) -> np.ndarray:
        """Build action mask using the same logic as BrentsRLAgent to avoid
        index mismatches (especially the 5-move overflow bug fix)."""
        return np.array(BrentsRLAgent.get_action_mask(battle), dtype=np.int64)
