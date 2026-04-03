from __future__ import annotations

from typing import TypeAlias

from poke_env.player import Player
from poke_env.player.baselines import (
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.ps_client import AccountConfiguration

OpponentName: TypeAlias = str
OPPONENT_CHOICES: tuple[str, ...] = (
    "random",
    "max_base_power",
    "simple_heuristic",
    "self_play",
    "adaptive",
)

_OPPONENT_CLASSES: dict[OpponentName, type[Player]] = {
    "random": RandomPlayer,
    "max_base_power": MaxBasePowerPlayer,
    "simple_heuristic": SimpleHeuristicsPlayer,
}

_ACCOUNT_PREFIXES: dict[OpponentName, str] = {
    "random": "RandOpp",
    "max_base_power": "MaxPower",
    "simple_heuristic": "Heuristic",
}


def create_opponent(
    opponent_name: OpponentName,
    *,
    battle_format: str,
    log_level: int,
    start_listening: bool = False,
    checkpoint_path: str | None = None,
) -> Player:
    if opponent_name == "self_play":
        if checkpoint_path is None:
            raise ValueError(
                "--self-play-checkpoint is required when using 'self_play' opponent"
            )
        from self_play import SelfPlayPlayer
        return SelfPlayPlayer(
            checkpoint_path,
            battle_format=battle_format,
            log_level=log_level,
            start_listening=start_listening,
        )

    if opponent_name == "adaptive":
        from adaptive_opponent import AdaptivePlayer
        return AdaptivePlayer(
            battle_format=battle_format,
            log_level=log_level,
            start_listening=start_listening,
            checkpoint_path=checkpoint_path,
        )

    try:
        opponent_cls = _OPPONENT_CLASSES[opponent_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported opponent '{opponent_name}'. Choices: {', '.join(OPPONENT_CHOICES)}"
        ) from exc

    import os, random as _rng, string as _str
    _uid = f"{os.getpid()}{''.join(_rng.choices(_str.ascii_lowercase, k=4))}"
    return opponent_cls(
        account_configuration=AccountConfiguration.generate(
            f"{_ACCOUNT_PREFIXES[opponent_name]}{_uid}",
            rand=False,
        ),
        battle_format=battle_format,
        start_listening=start_listening,
        log_level=log_level,
    )
