"""Adaptive opponent: dynamically adjusts difficulty based on agent win rate.

Wraps a ladder of opponent Players (easy -> hard) and promotes/demotes
based on a sliding window of recent game outcomes.
"""
from __future__ import annotations

import os
import random as _rng
import string as _str
from collections import deque

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder
from poke_env.player.baselines import (
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.ps_client import AccountConfiguration

TIER_NAMES = ("RandomPlayer", "MaxBasePowerPlayer", "SimpleHeuristicsPlayer", "SelfPlayPlayer")

PROMOTE_THRESHOLD = 0.70
DEMOTE_THRESHOLD = 0.30
DEFAULT_WINDOW = 20


class AdaptivePlayer(Player):
    """Dynamically selects opponent difficulty based on recent agent win rate.

    Wraps a list of opponent Players ordered easy -> hard.  Tracks recent
    game results in a sliding window and promotes to harder opponents when
    the agent wins too often, demotes when it loses too often.
    """

    def __init__(
        self,
        *,
        battle_format: str = "gen9randombattle",
        log_level: int = 40,
        start_listening: bool = False,
        checkpoint_path: str | None = None,
        window: int = DEFAULT_WINDOW,
    ):
        uid = f"{os.getpid()}{''.join(_rng.choices(_str.ascii_lowercase, k=4))}"
        super().__init__(
            account_configuration=AccountConfiguration.generate(
                f"Adaptive{uid}", rand=False,
            ),
            battle_format=battle_format,
            start_listening=start_listening,
            log_level=log_level,
        )

        self._window = window
        self._recent_results: deque[bool] = deque(maxlen=window)
        self._current_idx = 0
        self._prev_battle_tag: str | None = None
        self._prev_battle: AbstractBattle | None = None

        # Build the opponent ladder (easy -> hard)
        self._opponents: list[Player] = self._build_ladder(
            battle_format=battle_format,
            log_level=log_level,
            start_listening=start_listening,
            checkpoint_path=checkpoint_path,
            uid_base=uid,
        )

    @staticmethod
    def _build_ladder(
        *,
        battle_format: str,
        log_level: int,
        start_listening: bool,
        checkpoint_path: str | None,
        uid_base: str,
    ) -> list[Player]:
        """Create the opponent ladder from easy to hard."""
        def _acct(prefix: str) -> AccountConfiguration:
            suffix = ''.join(_rng.choices(_str.ascii_lowercase, k=3))
            return AccountConfiguration.generate(
                f"{prefix}{uid_base}{suffix}", rand=False,
            )

        common = dict(
            battle_format=battle_format,
            start_listening=start_listening,
            log_level=log_level,
        )
        ladder: list[Player] = [
            RandomPlayer(account_configuration=_acct("AdRnd"), **common),
            MaxBasePowerPlayer(account_configuration=_acct("AdMax"), **common),
            SimpleHeuristicsPlayer(account_configuration=_acct("AdHeu"), **common),
        ]

        if checkpoint_path is not None:
            from self_play import SelfPlayPlayer
            ladder.append(SelfPlayPlayer(
                checkpoint_path,
                battle_format=battle_format,
                log_level=log_level,
                start_listening=start_listening,
            ))

        return ladder

    @property
    def current_tier(self) -> int:
        return self._current_idx

    @property
    def current_tier_name(self) -> str:
        return TIER_NAMES[self._current_idx]

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Delegate to the current tier's opponent, tracking results."""
        self._check_prev_battle_result(battle)
        return self._opponents[self._current_idx].choose_move(battle)

    def _check_prev_battle_result(self, battle: AbstractBattle) -> None:
        """Detect when a new battle starts and record the previous outcome."""
        tag = battle.battle_tag
        if self._prev_battle_tag is None:
            self._prev_battle_tag = tag
            self._prev_battle = battle
            return

        if tag != self._prev_battle_tag:
            # A new battle started — check the previous battle's result.
            if self._prev_battle is not None and self._prev_battle.finished:
                # prev_battle is from the opponent's perspective (this player).
                # Agent won = opponent lost.
                agent_won = self._prev_battle.won is False
                self._record_result(agent_won)
            self._prev_battle_tag = tag
            self._prev_battle = battle
        else:
            # Same battle, update reference
            self._prev_battle = battle

    def _record_result(self, agent_won: bool) -> None:
        """Record a game outcome and check for tier promotion/demotion."""
        self._recent_results.append(agent_won)
        if len(self._recent_results) < self._window:
            return

        win_rate = sum(self._recent_results) / len(self._recent_results)

        if win_rate > PROMOTE_THRESHOLD and self._current_idx < len(self._opponents) - 1:
            self._current_idx += 1
            self._recent_results.clear()
            print(
                f"[AdaptiveOpponent] PROMOTED to tier {self._current_idx} "
                f"({self.current_tier_name}) — agent WR was {win_rate:.0%}"
            )
        elif win_rate < DEMOTE_THRESHOLD and self._current_idx > 0:
            self._current_idx -= 1
            self._recent_results.clear()
            print(
                f"[AdaptiveOpponent] DEMOTED to tier {self._current_idx} "
                f"({self.current_tier_name}) — agent WR was {win_rate:.0%}"
            )
