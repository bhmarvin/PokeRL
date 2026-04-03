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
) -> Player:
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
