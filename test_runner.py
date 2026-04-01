from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status

from brent_agent import (
    MY_ACTIVE_START,
    ON_RECHARGE_INDEX,
    OPP_BENCH_START,
    OPP_MOVES_VS_ME_START,
    SPEED_ADVANTAGE_INDEX,
    TARGETING_START,
    VECTOR_LENGTH,
    BrentObservationVectorBuilder,
)


@dataclass
class FakeMove:
    id: str
    accuracy: float = 1.0
    category: MoveCategory = MoveCategory.PHYSICAL
    flags: set[str] = field(default_factory=set)
    priority: int = 0
    self_switch: bool = False
    heal: float = 0.0
    drain: float = 0.0
    type: PokemonType = PokemonType.NORMAL


@dataclass
class FakePokemon:
    name: str
    species: str
    types: tuple[PokemonType, ...]
    current_hp_fraction: float
    max_hp: int = 100
    active: bool = False
    revealed: bool = False
    fainted: bool = False
    status: Optional[Status] = None
    item: str = ""
    ability: Optional[str] = None
    is_terastallized: bool = False
    tera_type: Optional[PokemonType] = None
    boosts: dict[str, int] = field(
        default_factory=lambda: {
            "atk": 0,
            "def": 0,
            "spa": 0,
            "spd": 0,
            "spe": 0,
            "evasion": 0,
            "accuracy": 0,
        }
    )
    effects: dict[Effect, int] = field(default_factory=dict)
    moves: OrderedDict[str, FakeMove] = field(default_factory=OrderedDict)
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "hp": 100,
            "atk": 100,
            "def": 100,
            "spa": 100,
            "spd": 100,
            "spe": 100,
        }
    )
    last_move: Optional[FakeMove] = None

    @property
    def type_1(self) -> PokemonType:
        return self.types[0]

    def identifier(self, role: str) -> str:
        return f"{role}: {self.name}"


@dataclass
class FakeBattle:
    battle_tag: str
    player_role: str
    opponent_role: str
    active_pokemon: FakePokemon
    opponent_active_pokemon: FakePokemon
    available_moves: list[FakeMove]
    team: OrderedDict[str, FakePokemon]
    opponent_team: OrderedDict[str, FakePokemon]
    turn: int = 7
    weather: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    side_conditions: dict = field(default_factory=dict)
    opponent_side_conditions: dict = field(default_factory=dict)
    finished: bool = False


class DeterministicBuilder(BrentObservationVectorBuilder):
    def _estimate_damage_range(self, battle, attacker, defender, move, attacker_role, defender_role):
        damage_map = {
            "tackle": (20.0, 30.0),
            "recover": (0.0, 0.0),
            "flamethrower": (35.0, 45.0),
        }
        return damage_map.get(move.id, (0.0, 0.0))


def build_fake_battle() -> FakeBattle:
    tackle = FakeMove(id="tackle", accuracy=1.0, category=MoveCategory.PHYSICAL, flags={"contact"})
    recover = FakeMove(id="recover", accuracy=1.0, category=MoveCategory.STATUS, heal=0.5)
    flamethrower = FakeMove(
        id="flamethrower",
        accuracy=1.0,
        category=MoveCategory.SPECIAL,
        type=PokemonType.FIRE,
    )
    hidden_move = FakeMove(id="hiddenpower", accuracy=1.0, category=MoveCategory.SPECIAL)

    my_active = FakePokemon(
        name="Lead",
        species="dragonite",
        types=(PokemonType.DRAGON, PokemonType.FLYING),
        current_hp_fraction=0.8,
        active=True,
        revealed=True,
        status=Status.TOX,
        moves=OrderedDict([("tackle", tackle), ("recover", recover)]),
    )
    my_bench = FakePokemon(
        name="Bench1",
        species="rotomwash",
        types=(PokemonType.ELECTRIC, PokemonType.WATER),
        current_hp_fraction=1.0,
        revealed=False,
        moves=OrderedDict([("flamethrower", flamethrower)]),
    )
    opp_active = FakePokemon(
        name="OppLead",
        species="garchomp",
        types=(PokemonType.DRAGON, PokemonType.GROUND),
        current_hp_fraction=0.6,
        active=True,
        revealed=True,
        moves=OrderedDict([("flamethrower", flamethrower)]),
    )
    opp_revealed_bench = FakePokemon(
        name="OppBench1",
        species="corviknight",
        types=(PokemonType.FLYING, PokemonType.STEEL),
        current_hp_fraction=0.5,
        revealed=True,
        moves=OrderedDict([("hiddenpower", hidden_move)]),
    )

    return FakeBattle(
        battle_tag="smoke-test",
        player_role="p1",
        opponent_role="p2",
        active_pokemon=my_active,
        opponent_active_pokemon=opp_active,
        available_moves=[tackle, recover],
        team=OrderedDict([("p1a", my_active), ("p1b", my_bench)]),
        opponent_team=OrderedDict([("p2a", opp_active), ("p2b", opp_revealed_bench)]),
        weather={},
        fields={Field.TRICK_ROOM: 1},
        side_conditions={SideCondition.STEALTH_ROCK: 1},
        opponent_side_conditions={},
    )


def main() -> None:
    battle = build_fake_battle()
    builder = DeterministicBuilder()
    vector = builder.embed_battle(battle)
    assert builder.verify_battle_embedding(battle, vector) == []

    assert vector.shape == (VECTOR_LENGTH,)
    assert vector.dtype == np.float32
    assert vector[SPEED_ADVANTAGE_INDEX] == np.float32(0.5)
    assert vector[MY_ACTIVE_START + 30] == np.float32(1.0)
    assert vector[ON_RECHARGE_INDEX] == np.float32(0.0)
    assert np.allclose(vector[OPP_BENCH_START + 20 : TARGETING_START], 0.0)
    assert np.allclose(vector[OPP_MOVES_VS_ME_START + 6 : VECTOR_LENGTH], 0.0)

    battle.active_pokemon.status = Status.PSN
    poison_vector = builder.embed_battle(battle)
    assert poison_vector[MY_ACTIVE_START + 30] == np.float32(0.5)
    assert builder.verify_battle_embedding(battle, poison_vector) == []

    battle.finished = True
    builder.embed_battle(battle)
    assert "smoke-test" not in builder._my_team_revealed_memory

    print("Observation vector smoke test passed.")


if __name__ == "__main__":
    main()