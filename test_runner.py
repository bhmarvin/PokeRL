from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from poke_env.battle.move import Move
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status

from brent_agent import (
    MOVE_BLOCK_SIZE,
    REWARD_CONFIG,
    BrentsRLAgent,
    MY_ACTIVE_START,
    MY_MOVES_START,
    ON_RECHARGE_INDEX,
    OPP_BENCH_START,
    OPP_MOVES_VS_ME_START,
    OPP_THREAT_CONFIDENCE_START,
    OPP_THREAT_OHKO_START,
    OPP_THREAT_ROW_SIZE,
    OPP_THREAT_START,
    SPEED_ADVANTAGE_INDEX,
    TARGETING_START,
    TacticalRewardContext,
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
    self_boost: Optional[dict[str, int]] = None
    heal: float = 0.0
    drain: float = 0.0
    recoil: float = 0.0
    recharge: bool = False
    status: Optional[Status] = None
    volatile_status: Optional[Effect] = None
    secondary: list[dict[str, object]] = field(default_factory=list)
    type: PokemonType = PokemonType.NORMAL


@dataclass
class FakePokemon:
    name: str
    species: str
    types: tuple[PokemonType, ...]
    current_hp_fraction: float
    level: int = 80
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
    base_stats: dict[str, int] = field(
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

    @property
    def type_2(self) -> Optional[PokemonType]:
        if len(self.types) < 2:
            return None
        return self.types[1]

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
    available_switches: list[FakePokemon] = field(default_factory=list)
    turn: int = 7
    weather: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    side_conditions: dict = field(default_factory=dict)
    opponent_side_conditions: dict = field(default_factory=dict)
    finished: bool = False


class FakeMeta:
    def __init__(self) -> None:
        self.role_weights = {
            "wallbreaker": 0.7,
            "support": 0.3,
        }
        self.role_items = {
            "wallbreaker": {"choicescarf": 1.0},
            "support": {"leftovers": 1.0},
        }
        self.role_moves = {
            "wallbreaker": {
                "flamethrower": 1.0,
                "fireblast": 0.8,
                "earthquake": 0.6,
            },
            "support": {
                "flamethrower": 1.0,
                "willowisp": 0.9,
                "roost": 0.7,
                "uturn": 0.5,
            },
        }
        self.role_stats = {
            "wallbreaker": {"hp": 100, "atk": 120, "def": 95, "spa": 130, "spd": 85, "spe": 80},
            "support": {"hp": 100, "atk": 90, "def": 110, "spa": 90, "spd": 105, "spe": 60},
        }

    def get_species_data(self, species: str):
        return None

    def filter_roles(self, species_name: str, revealed_moves: list[str], revealed_item: Optional[str] = None) -> dict[str, float]:
        valid_roles: dict[str, float] = {}
        total_weight = 0.0
        for role_name, weight in self.role_weights.items():
            role_moves = self.role_moves[role_name]
            if any(move not in role_moves for move in revealed_moves):
                continue
            if revealed_item and revealed_item not in self.role_items[role_name]:
                continue
            valid_roles[role_name] = weight
            total_weight += weight
        if total_weight <= 0.0:
            return {}
        return {
            role_name: weight / total_weight
            for role_name, weight in valid_roles.items()
        }

    def get_role_move_distribution(self, species_name: str, role_name: str) -> dict[str, float]:
        return dict(self.role_moves.get(role_name, {}))

    def get_role_item_distribution(self, species_name: str, role_name: str) -> dict[str, float]:
        return dict(self.role_items.get(role_name, {}))

    def get_move_marginals(self, species_name: str, role_weights: dict[str, float]) -> dict[str, float]:
        marginals: dict[str, float] = {}
        for role_name, role_weight in role_weights.items():
            for move_id, move_prob in self.get_role_move_distribution(species_name, role_name).items():
                marginals[move_id] = marginals.get(move_id, 0.0) + role_weight * move_prob
        return marginals

    def get_item_marginals(self, species_name: str, role_weights: dict[str, float]) -> dict[str, float]:
        marginals: dict[str, float] = {}
        for role_name, role_weight in role_weights.items():
            for item_id, item_prob in self.get_role_item_distribution(species_name, role_name).items():
                marginals[item_id] = marginals.get(item_id, 0.0) + role_weight * item_prob
        return marginals

    def get_role_stats(self, species_name: str, role_name: str, base_stats: dict[str, int]) -> dict[str, int]:
        return dict(self.role_stats.get(role_name, base_stats))


class DeterministicBuilder(BrentObservationVectorBuilder):
    def __init__(self) -> None:
        super().__init__()
        self._meta = FakeMeta()

    def _estimate_damage_range(self, battle, attacker, defender, move, attacker_role, defender_role):
        targeted_damage_map = {
            ("fireblast", "altaria"): (20.0, 25.0),
            ("flamethrower", "altaria"): (10.0, 15.0),
            ("earthquake", "altaria"): (0.0, 0.0),
            ("dragonpulse", "altaria"): (15.0, 20.0),
            ("icebeam", "garchomp"): (65.0, 70.0),
        }
        damage_map = {
            "doubleedge": (40.0, 40.0),
            "recover": (0.0, 0.0),
            "overheat": (55.0, 55.0),
            "hyperbeam": (75.0, 75.0),
            "flamethrower": (35.0, 45.0),
            "fireblast": (85.0, 95.0),
            "earthquake": (70.0, 85.0),
            "dragonpulse": (25.0, 35.0),
            "willowisp": (0.0, 0.0),
            "roost": (0.0, 0.0),
            "uturn": (15.0, 20.0),
        }
        defender_species = getattr(defender, "species", "")
        if (move.id, defender_species) in targeted_damage_map:
            return targeted_damage_map[(move.id, defender_species)]
        return damage_map.get(move.id, (0.0, 0.0))


@dataclass
class FakeOrder:
    order: object
    terastallize: bool = False

    def __str__(self) -> str:
        action = getattr(self.order, "id", None) or getattr(self.order, "species", None) or "unknown"
        return f"FakeOrder({action})"


def build_fake_battle() -> FakeBattle:
    doubleedge = FakeMove(
        id="doubleedge",
        accuracy=1.0,
        category=MoveCategory.PHYSICAL,
        flags={"contact"},
        recoil=0.25,
    )
    recover = FakeMove(id="recover", accuracy=1.0, category=MoveCategory.STATUS, heal=0.5)
    overheat = FakeMove(
        id="overheat",
        accuracy=1.0,
        category=MoveCategory.SPECIAL,
        type=PokemonType.FIRE,
        self_boost={"spa": -2},
    )
    hyperbeam = FakeMove(
        id="hyperbeam",
        accuracy=1.0,
        category=MoveCategory.SPECIAL,
        recharge=True,
    )
    discharge = FakeMove(
        id="discharge",
        accuracy=1.0,
        category=MoveCategory.SPECIAL,
        type=PokemonType.ELECTRIC,
        secondary=[{"chance": 30, "status": "par"}],
    )
    icebeam = FakeMove(
        id="icebeam",
        accuracy=1.0,
        category=MoveCategory.SPECIAL,
        type=PokemonType.ICE,
        secondary=[{"chance": 10, "status": "frz"}],
    )
    hurricane = FakeMove(
        id="hurricane",
        accuracy=1.0,
        category=MoveCategory.SPECIAL,
        type=PokemonType.FLYING,
        secondary=[{"chance": 30, "volatileStatus": "confusion"}],
    )
    flamethrower = FakeMove(
        id="flamethrower",
        accuracy=1.0,
        category=MoveCategory.SPECIAL,
        type=PokemonType.FIRE,
        secondary=[{"chance": 10, "status": "brn"}],
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
        moves=OrderedDict(
            [
                ("doubleedge", doubleedge),
                ("recover", recover),
                ("overheat", overheat),
                ("hyperbeam", hyperbeam),
            ]
        ),
    )
    my_bench = FakePokemon(
        name="Bench1",
        species="rotomwash",
        types=(PokemonType.ELECTRIC, PokemonType.WATER),
        current_hp_fraction=1.0,
        revealed=False,
        moves=OrderedDict(
            [
                ("discharge", discharge),
                ("icebeam", icebeam),
                ("hurricane", hurricane),
                ("recover", recover),
            ]
        ),
    )
    safe_bench = FakePokemon(
        name="Bench2",
        species="altaria",
        types=(PokemonType.DRAGON, PokemonType.FLYING),
        current_hp_fraction=0.9,
        stats={"hp": 100, "atk": 80, "def": 100, "spa": 110, "spd": 100, "spe": 130},
        base_stats={"hp": 100, "atk": 80, "def": 100, "spa": 110, "spd": 100, "spe": 130},
        moves=OrderedDict(
            [
                ("icebeam", icebeam),
                ("hurricane", hurricane),
                ("recover", recover),
            ]
        ),
    )
    opp_active = FakePokemon(
        name="OppLead",
        species="garchomp",
        types=(PokemonType.DRAGON, PokemonType.GROUND),
        current_hp_fraction=0.6,
        active=True,
        revealed=True,
        item="choicescarf",
        moves=OrderedDict([("flamethrower", flamethrower)]),
        stats={"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100},
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
        available_moves=[doubleedge, recover, overheat, hyperbeam],
        team=OrderedDict([("p1a", my_active), ("p1b", my_bench), ("p1c", safe_bench)]),
        opponent_team=OrderedDict([("p2a", opp_active), ("p2b", opp_revealed_bench)]),
        available_switches=[my_bench, safe_bench],
        weather={},
        fields={Field.TRICK_ROOM: 1},
        side_conditions={SideCondition.STEALTH_ROCK: 1},
        opponent_side_conditions={},
    )


def build_test_agent() -> BrentsRLAgent:
    agent = BrentsRLAgent.__new__(BrentsRLAgent)
    agent.vector_builder = DeterministicBuilder()
    agent._tactical_reward_context = None
    agent._strategic_penalty_counts = {}
    agent._strategic_penalty_total = 0.0
    agent._strategic_penalty_move_checks = 0
    agent._strategic_penalty_penalized_actions = 0
    agent._tactical_shaping_counts = {}
    agent._tactical_shaping_totals = {}
    agent._tactical_shaping_total = 0.0
    agent._tactical_positive_total = 0.0
    agent._tactical_negative_total = 0.0
    agent._tactical_shaping_move_checks = 0
    agent._tactical_shaping_shaped_actions = 0
    agent._tactical_shaping_rewarded_actions = 0
    agent._tactical_shaping_penalized_actions = 0
    agent._decision_audit_counts = {}
    agent._decision_audit_flagged_actions = 0
    agent._decision_audit_move_checks = 0
    agent._decision_audit_samples = {}
    agent._decision_count = 0
    agent._switch_action_count = 0
    agent.reward_computing_helper = lambda battle, **kwargs: 0.0
    return agent


def queue_reward_context(agent: BrentsRLAgent, battle: FakeBattle, action: object, matches) -> float:
    agent._tactical_reward_context = TacticalRewardContext(
        battle_tag=battle.battle_tag,
        action=action,
        matches=tuple(match for match in matches if match is not None),
    )
    return agent.calc_reward(battle)


def main() -> None:
    battle = build_fake_battle()
    builder = DeterministicBuilder()
    vector = builder.embed_battle(battle)
    assert builder.verify_battle_embedding(battle, vector) == []

    assert vector.shape == (VECTOR_LENGTH,)
    assert vector.dtype == np.float32
    assert vector[SPEED_ADVANTAGE_INDEX] == np.float32(0.0)
    assert vector[MY_MOVES_START + 12] == np.float32(0.0)
    assert np.isclose(vector[MY_MOVES_START + 17], np.float32(0.1))
    assert vector[MY_MOVES_START + MOVE_BLOCK_SIZE + 12] == np.float32(0.2)
    assert vector[MY_MOVES_START + 2 * MOVE_BLOCK_SIZE + 15] == np.float32(-1.0)
    assert vector[MY_MOVES_START + 3 * MOVE_BLOCK_SIZE + 18] == np.float32(1.0)
    assert vector[MY_ACTIVE_START + 30] == np.float32(1.0)
    assert vector[ON_RECHARGE_INDEX] == np.float32(0.0)
    assert np.allclose(vector[OPP_BENCH_START + 20 : TARGETING_START], 0.0)

    posterior = builder._opponent_role_posterior(battle.opponent_active_pokemon)
    assert posterior == {"wallbreaker": 1.0}
    assert builder._speed_stat_estimate(battle.opponent_active_pokemon, posterior) == 80.0
    assert builder._speed_item_multiplier(battle.opponent_active_pokemon, posterior) == 1.5

    inferred_entries = builder._select_opponent_threat_entries(battle.opponent_active_pokemon, posterior)
    assert [entry.move.id for entry in inferred_entries] == [
        "flamethrower",
        "fireblast",
        "earthquake",
    ]

    first_row = OPP_THREAT_START
    second_row = OPP_THREAT_START + OPP_THREAT_ROW_SIZE
    assert vector[first_row] == np.float32(1.0)
    assert vector[first_row + 1] == np.float32(1.0)
    assert vector[second_row] == np.float32(0.8)
    assert vector[second_row + 1] == np.float32(0.0)
    assert vector[OPP_MOVES_VS_ME_START] == np.float32(0.4)
    assert vector[OPP_THREAT_CONFIDENCE_START] == np.float32(1.0)
    assert vector[OPP_THREAT_CONFIDENCE_START + 1] == np.float32(0.0)
    assert vector[OPP_THREAT_OHKO_START] > np.float32(0.5)
    assert np.allclose(vector[OPP_THREAT_OHKO_START + 3 : OPP_THREAT_CONFIDENCE_START], 0.0)
    assert vector[MY_MOVES_START + 19] == np.float32(0.0)
    assert vector[MY_MOVES_START + 20] == np.float32(0.0)
    assert vector[MY_MOVES_START + 21] == np.float32(0.0)
    assert vector[MY_MOVES_START + 22] == np.float32(0.0)
    assert vector[MY_MOVES_START + 23] == np.float32(0.0)
    assert vector[MY_MOVES_START + 24] == np.float32(0.0)
    assert builder._move_effect_chance(
        battle.team["p1b"].moves["discharge"],
        statuses=(Status.PAR,),
    ) == np.float32(0.3)
    assert builder._move_effect_chance(
        battle.team["p1b"].moves["icebeam"],
        statuses=(Status.FRZ,),
    ) == np.float32(0.1)
    assert builder._move_effect_chance(
        battle.team["p1b"].moves["hurricane"],
        volatile_effect=Effect.CONFUSION,
    ) == np.float32(0.3)
    assert builder._move_effect_chance(
        battle.opponent_active_pokemon.moves["flamethrower"],
        statuses=(Status.BRN,),
    ) == np.float32(0.1)

    battle.active_pokemon.status = Status.PSN
    poison_vector = builder.embed_battle(battle)
    assert poison_vector[MY_MOVES_START + MOVE_BLOCK_SIZE + 12] == np.float32(0.2)
    assert poison_vector[MY_ACTIVE_START + 30] == np.float32(0.5)
    assert builder.verify_battle_embedding(battle, poison_vector) == []

    battle.finished = True
    builder.embed_battle(battle)
    assert "smoke-test" not in builder._my_team_revealed_memory

    wasteful_battle = build_fake_battle()
    wasteful_agent = build_test_agent()
    wasteful_order = FakeOrder(Move("recover", 9))
    wasteful_agent._record_action_choice(wasteful_order)
    wasteful_agent._remember_tactical_reward_context(wasteful_battle, wasteful_order)
    wasteful_reward = wasteful_agent.calc_reward(wasteful_battle)
    assert np.isclose(wasteful_reward, REWARD_CONFIG["penalty_wasteful_heal_overflow"])

    neutral_heal_battle = build_fake_battle()
    neutral_heal_battle.active_pokemon.current_hp_fraction = 0.7
    neutral_heal_agent = build_test_agent()
    neutral_heal_order = FakeOrder(Move("recover", 9))
    neutral_heal_agent._record_action_choice(neutral_heal_order)
    neutral_heal_agent._remember_tactical_reward_context(neutral_heal_battle, neutral_heal_order)
    neutral_heal_reward = neutral_heal_agent.calc_reward(neutral_heal_battle)
    assert np.isclose(neutral_heal_reward, 0.0)

    low_hp_heal_battle = build_fake_battle()
    low_hp_heal_battle.active_pokemon.current_hp_fraction = 0.25
    low_hp_heal_agent = build_test_agent()
    low_hp_heal_order = FakeOrder(Move("recover", 9))
    low_hp_heal_agent._record_action_choice(low_hp_heal_order)
    low_hp_heal_agent._remember_tactical_reward_context(low_hp_heal_battle, low_hp_heal_order)
    low_hp_heal_reward = low_hp_heal_agent.calc_reward(low_hp_heal_battle)
    assert np.isclose(low_hp_heal_reward, REWARD_CONFIG["bonus_good_heal_timing"])

    self_drop_battle = build_fake_battle()
    self_drop_battle.active_pokemon.boosts["spa"] = -3
    self_drop_agent = build_test_agent()
    self_drop_order = FakeOrder(Move("overheat", 9))
    self_drop_agent._record_action_choice(self_drop_order)
    self_drop_agent._remember_tactical_reward_context(self_drop_battle, self_drop_order)
    self_drop_reward = self_drop_agent.calc_reward(self_drop_battle)
    assert np.isclose(self_drop_reward, REWARD_CONFIG["penalty_redundant_self_drop_move"])

    fresh_overheat_battle = build_fake_battle()
    fresh_overheat_agent = build_test_agent()
    fresh_overheat_order = FakeOrder(Move("overheat", 9))
    fresh_overheat_agent._record_action_choice(fresh_overheat_order)
    fresh_overheat_agent._remember_tactical_reward_context(fresh_overheat_battle, fresh_overheat_order)
    fresh_overheat_reward = fresh_overheat_agent.calc_reward(fresh_overheat_battle)
    assert np.isclose(fresh_overheat_reward, 0.0)

    attack_battle = build_fake_battle()
    attack_agent = build_test_agent()
    attack_order = FakeOrder(Move("hyperbeam", 9))
    attack_agent._record_action_choice(attack_order)
    attack_agent._remember_tactical_reward_context(attack_battle, attack_order)
    attack_reward = attack_agent.calc_reward(attack_battle)
    assert np.isclose(attack_reward, REWARD_CONFIG["bonus_good_attack_selection"])

    low_quality_attack_battle = build_fake_battle()
    low_quality_attack_agent = build_test_agent()
    low_quality_attack_order = FakeOrder(Move("doubleedge", 9))
    low_quality_attack_agent._record_action_choice(low_quality_attack_order)
    low_quality_attack_agent._remember_tactical_reward_context(low_quality_attack_battle, low_quality_attack_order)
    low_quality_attack_reward = low_quality_attack_agent.calc_reward(low_quality_attack_battle)
    assert np.isclose(low_quality_attack_reward, 0.0)

    safe_switch_battle = build_fake_battle()
    safe_switch_agent = build_test_agent()
    safe_switch_match = safe_switch_agent._make_tactical_match(
        "good_safe_switch",
        "bonus_good_safe_switch",
        safe_switch_agent._evaluate_good_safe_switch(safe_switch_battle, safe_switch_battle.team["p1c"]),
    )
    safe_switch_reward = queue_reward_context(
        safe_switch_agent,
        safe_switch_battle,
        None,
        [safe_switch_match],
    )
    assert np.isclose(safe_switch_reward, REWARD_CONFIG["bonus_good_safe_switch"])

    unsafe_switch_battle = build_fake_battle()
    unsafe_switch_agent = build_test_agent()
    unsafe_switch_match = unsafe_switch_agent._make_tactical_match(
        "good_safe_switch",
        "bonus_good_safe_switch",
        unsafe_switch_agent._evaluate_good_safe_switch(unsafe_switch_battle, unsafe_switch_battle.team["p1b"]),
    )
    unsafe_switch_reward = queue_reward_context(
        unsafe_switch_agent,
        unsafe_switch_battle,
        None,
        [unsafe_switch_match],
    )
    assert np.isclose(unsafe_switch_reward, 0.0)

    unsafe_stay_battle = build_fake_battle()
    unsafe_stay_agent = build_test_agent()
    unsafe_stay_order = FakeOrder(Move("doubleedge", 9))
    unsafe_stay_agent._record_action_choice(unsafe_stay_order)
    unsafe_stay_agent._remember_tactical_reward_context(unsafe_stay_battle, unsafe_stay_order)
    unsafe_stay_reward = unsafe_stay_agent.calc_reward(unsafe_stay_battle)
    assert np.isclose(unsafe_stay_reward, 0.0)
    unsafe_stay_report = unsafe_stay_agent.get_decision_audit_report()
    assert unsafe_stay_report["counts"]["unsafe_stay_in_with_fast_ko_switch"] == 1

    seismic_toss = Move("seismictoss", 9)
    toss_min, toss_max = builder._manual_damage_calc(
        battle.active_pokemon,
        battle.opponent_active_pokemon,
        seismic_toss,
        battle,
        battle.active_pokemon.stats,
        battle.opponent_active_pokemon.stats,
    )
    assert toss_min == 80.0
    assert toss_max == 80.0

    ghost_target = FakePokemon(
        name="Ghost",
        species="gengar",
        types=(PokemonType.GHOST, PokemonType.POISON),
        current_hp_fraction=1.0,
    )
    immune_min, immune_max = builder._manual_damage_calc(
        battle.active_pokemon,
        ghost_target,
        seismic_toss,
        battle,
        battle.active_pokemon.stats,
        ghost_target.stats,
    )
    assert immune_min == 0.0
    assert immune_max == 0.0

    print("Observation vector smoke test passed.")


if __name__ == "__main__":
    main()