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
from poke_env.battle.weather import Weather

from brent_agent import (
    MOVE_BLOCK_SIZE,
    REWARD_CONFIG,
    BrentsRLAgent,
    MY_ACTIVE_START,
    MY_MOVES_START,
    ON_RECHARGE_INDEX,
    OPP_ACTIVE_START,
    OPP_BENCH_START,
    OPP_MOVES_VS_ME_START,
    OPP_THREAT_CONFIDENCE_START,
    OPP_THREAT_OHKO_START,
    OPP_THREAT_ROW_SIZE,
    OPP_THREAT_START,
    SPEED_ADVANTAGE_INDEX,
    TARGETING_START,
    TYPE_ORDER,
    TacticalRewardContext,
    VECTOR_LENGTH,
    BrentObservationVectorBuilder,
)


@dataclass
class FakeMove:
    id: str
    accuracy: float = 1.0
    base_power: int = 0
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
    entry: dict = field(default_factory=dict)
    ignore_defensive: bool = False
    damage: object = 0


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
    base_species: str = ""
    available_z_moves: list = field(default_factory=list)

    def __post_init__(self):
        if not self.base_species:
            self.base_species = self.species

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
    can_tera: bool = False
    trapped: bool = False
    _wait: bool = False
    can_mega_evolve: bool = False
    can_z_move: bool = False
    can_dynamax: bool = False
    gen: int = 9


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
    agent._consecutive_heal_count = {}
    agent._last_action_was_heal = {}
    agent._entered_after_faint = {}
    agent._last_active_species = None
    agent._last_active_fainted = False
    agent._prev_opp_alive = set()
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
    # Both wasteful_heal_overflow and unsafe_stay_in fire here
    expected_wasteful = REWARD_CONFIG["penalty_wasteful_heal_overflow"] + REWARD_CONFIG["penalty_unsafe_stay_in_with_fast_ko_switch"]
    assert np.isclose(wasteful_reward, expected_wasteful), f"wasteful heal got {wasteful_reward}, expected {expected_wasteful}"

    neutral_heal_battle = build_fake_battle()
    neutral_heal_battle.active_pokemon.current_hp_fraction = 0.7
    neutral_heal_agent = build_test_agent()
    neutral_heal_order = FakeOrder(Move("recover", 9))
    neutral_heal_agent._record_action_choice(neutral_heal_order)
    neutral_heal_agent._remember_tactical_reward_context(neutral_heal_battle, neutral_heal_order)
    neutral_heal_reward = neutral_heal_agent.calc_reward(neutral_heal_battle)
    # unsafe_stay_in fires here because safe switch exists
    assert np.isclose(neutral_heal_reward, REWARD_CONFIG["penalty_unsafe_stay_in_with_fast_ko_switch"])

    low_hp_heal_battle = build_fake_battle()
    low_hp_heal_battle.active_pokemon.current_hp_fraction = 0.25
    low_hp_heal_agent = build_test_agent()
    low_hp_heal_order = FakeOrder(Move("recover", 9))
    low_hp_heal_agent._record_action_choice(low_hp_heal_order)
    low_hp_heal_agent._remember_tactical_reward_context(low_hp_heal_battle, low_hp_heal_order)
    low_hp_heal_reward = low_hp_heal_agent.calc_reward(low_hp_heal_battle)
    expected_low_hp = REWARD_CONFIG["bonus_good_heal_timing"] + REWARD_CONFIG["penalty_unsafe_stay_in_with_fast_ko_switch"]
    assert np.isclose(low_hp_heal_reward, expected_low_hp), f"low hp heal got {low_hp_heal_reward}"

    self_drop_battle = build_fake_battle()
    self_drop_battle.active_pokemon.boosts["spa"] = -3
    self_drop_agent = build_test_agent()
    self_drop_order = FakeOrder(Move("overheat", 9))
    self_drop_agent._record_action_choice(self_drop_order)
    self_drop_agent._remember_tactical_reward_context(self_drop_battle, self_drop_order)
    self_drop_reward = self_drop_agent.calc_reward(self_drop_battle)
    expected_self_drop = REWARD_CONFIG["penalty_redundant_self_drop_move"] + REWARD_CONFIG["penalty_unsafe_stay_in_with_fast_ko_switch"]
    assert np.isclose(self_drop_reward, expected_self_drop), f"self drop got {self_drop_reward}"

    fresh_overheat_battle = build_fake_battle()
    fresh_overheat_agent = build_test_agent()
    fresh_overheat_order = FakeOrder(Move("overheat", 9))
    fresh_overheat_agent._record_action_choice(fresh_overheat_order)
    fresh_overheat_agent._remember_tactical_reward_context(fresh_overheat_battle, fresh_overheat_order)
    fresh_overheat_reward = fresh_overheat_agent.calc_reward(fresh_overheat_battle)
    assert np.isclose(fresh_overheat_reward, REWARD_CONFIG["penalty_unsafe_stay_in_with_fast_ko_switch"])

    attack_battle = build_fake_battle()
    attack_agent = build_test_agent()
    attack_order = FakeOrder(Move("hyperbeam", 9))
    attack_agent._record_action_choice(attack_order)
    attack_agent._remember_tactical_reward_context(attack_battle, attack_order)
    attack_reward = attack_agent.calc_reward(attack_battle)
    # good_attack_selection is enabled, hyperbeam KOs so unsafe_stay doesn't fire
    assert np.isclose(attack_reward, REWARD_CONFIG["bonus_good_attack_selection"]), f"expected {REWARD_CONFIG['bonus_good_attack_selection']} reward, got {attack_reward}"

    low_quality_attack_battle = build_fake_battle()
    low_quality_attack_agent = build_test_agent()
    low_quality_attack_order = FakeOrder(Move("doubleedge", 9))
    low_quality_attack_agent._record_action_choice(low_quality_attack_order)
    low_quality_attack_agent._remember_tactical_reward_context(low_quality_attack_battle, low_quality_attack_order)
    low_quality_attack_reward = low_quality_attack_agent.calc_reward(low_quality_attack_battle)
    assert np.isclose(low_quality_attack_reward, REWARD_CONFIG["penalty_unsafe_stay_in_with_fast_ko_switch"])

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
    assert np.isclose(unsafe_stay_reward, REWARD_CONFIG["penalty_unsafe_stay_in_with_fast_ko_switch"])
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

    # ── Tera observation tests ──────────────────────────────────────
    tera_battle = build_fake_battle()
    tera_battle.can_tera = True
    tera_battle.active_pokemon.tera_type = PokemonType.FAIRY
    tera_vec = builder.embed_battle(tera_battle)

    # My active: is_terastallized=0, can_tera=1, tera_type=FAIRY one-hot
    assert tera_vec[MY_ACTIVE_START + 40] == np.float32(0.0), "should not be terastallized yet"
    assert tera_vec[MY_ACTIVE_START + 41] == np.float32(1.0), "can_tera should be 1"
    fairy_idx = list(TYPE_ORDER).index(PokemonType.FAIRY)
    assert tera_vec[MY_ACTIVE_START + 42 + fairy_idx] == np.float32(1.0), "tera type fairy should be 1"
    # Other tera type slots should be 0
    for i, t in enumerate(TYPE_ORDER):
        if t != PokemonType.FAIRY:
            assert tera_vec[MY_ACTIVE_START + 42 + i] == np.float32(0.0), f"tera type {t.name} should be 0"

    # Now terastallize and verify type one-hot changes to mono Fairy
    tera_battle2 = build_fake_battle()
    tera_battle2.active_pokemon.is_terastallized = True
    tera_battle2.active_pokemon.tera_type = PokemonType.FAIRY
    tera_vec2 = builder.embed_battle(tera_battle2)

    assert tera_vec2[MY_ACTIVE_START + 40] == np.float32(1.0), "should be terastallized"
    # Type one-hot should now be FAIRY only (mono-type)
    for i, t in enumerate(TYPE_ORDER):
        expected = 1.0 if t == PokemonType.FAIRY else 0.0
        assert tera_vec2[MY_ACTIVE_START + 1 + i] == np.float32(expected), \
            f"post-tera type {t.name} should be {expected}"

    # Opponent tera: is_terastallized flag
    opp_tera_battle = build_fake_battle()
    assert tera_vec[OPP_ACTIVE_START + 40] == np.float32(0.0), "opp not terastallized"
    opp_tera_battle.opponent_active_pokemon.is_terastallized = True
    opp_tera_battle.opponent_active_pokemon.tera_type = PokemonType.STEEL
    opp_tera_vec = builder.embed_battle(opp_tera_battle)
    assert opp_tera_vec[OPP_ACTIVE_START + 40] == np.float32(1.0), "opp terastallized"
    # Opp type one-hot should be mono STEEL
    steel_idx = list(TYPE_ORDER).index(PokemonType.STEEL)
    for i, t in enumerate(TYPE_ORDER):
        expected = 1.0 if t == PokemonType.STEEL else 0.0
        assert opp_tera_vec[OPP_ACTIVE_START + 1 + i] == np.float32(expected), \
            f"opp post-tera type {t.name} should be {expected}"

    print("  Tera observation tests passed.")

    # ── Tera reward: immunity test ───────────────────────────────
    # Scenario: Dragonite (Dragon/Flying) with tera_type=Fairy
    # Opponent Garchomp just used Dragon Pulse (Dragon type)
    # Tera to Fairy = immune to Dragon. Should trigger good_tera.
    tera_reward_battle = build_fake_battle()
    tera_reward_battle.can_tera = True
    tera_reward_battle.active_pokemon.tera_type = PokemonType.FAIRY
    dragonpulse = FakeMove(
        id="dragonpulse", accuracy=1.0,
        category=MoveCategory.SPECIAL, type=PokemonType.DRAGON,
    )
    tera_reward_battle.opponent_active_pokemon.last_move = dragonpulse

    tera_agent = build_test_agent()
    # Simulate choosing doubleedge + terastallize
    tera_order = FakeOrder(order=Move("doubleedge", 9), terastallize=True)
    tera_agent._record_action_choice(tera_order)
    tera_agent._remember_tactical_reward_context(tera_reward_battle, tera_order)
    tera_ctx = tera_agent._tactical_reward_context
    tera_reasons = {m.reason: m.reward for m in tera_ctx.matches}
    assert "good_tera" in tera_reasons, f"good_tera not found in {tera_reasons}"
    assert np.isclose(tera_reasons["good_tera"], REWARD_CONFIG["bonus_good_tera"]), \
        f"immunity tera bonus wrong: {tera_reasons['good_tera']}"
    tera_agent.calc_reward(tera_reward_battle)  # consume context
    print("  Tera immunity reward test passed.")

    # ── Tera reward: NO immunity (bad tera) ──────────────────────
    # Same setup but tera to Steel — not immune to Dragon
    bad_tera_battle = build_fake_battle()
    bad_tera_battle.can_tera = True
    bad_tera_battle.active_pokemon.tera_type = PokemonType.STEEL
    bad_tera_battle.opponent_active_pokemon.last_move = dragonpulse

    bad_tera_agent = build_test_agent()
    bad_tera_order = FakeOrder(order=Move("doubleedge", 9), terastallize=True)
    bad_tera_agent._record_action_choice(bad_tera_order)
    bad_tera_agent._remember_tactical_reward_context(bad_tera_battle, bad_tera_order)
    bad_tera_ctx = bad_tera_agent._tactical_reward_context
    bad_tera_reasons = {m.reason: m.reward for m in bad_tera_ctx.matches}
    assert "good_tera" not in bad_tera_reasons, \
        f"steel tera vs dragon should NOT trigger good_tera, got {bad_tera_reasons}"
    bad_tera_agent.calc_reward(bad_tera_battle)  # consume context
    print("  Bad tera (no reward) test passed.")

    # ── Tera damage calc: STAB and type effectiveness ────────────
    from brent_agent import _stab_multiplier, _defender_type_mult, _effective_types
    from poke_env.data import GenData

    gen_data = GenData.from_gen(9)

    # Tera'd Dragonite (tera Fairy) using Fairy move = 2.0x? No, Fairy not in original types
    # Original types: Dragon/Flying. Tera type: Fairy.
    # Using Fairy move: tera_stab=True, original_stab=False → 1.5x
    fake_tera_mon = FakePokemon(
        name="TeraTest", species="dragonite",
        types=(PokemonType.DRAGON, PokemonType.FLYING),
        current_hp_fraction=1.0,
        is_terastallized=True, tera_type=PokemonType.FAIRY,
    )
    assert _stab_multiplier(fake_tera_mon, PokemonType.FAIRY) == 1.5, "new STAB from tera"
    assert _stab_multiplier(fake_tera_mon, PokemonType.DRAGON) == 1.5, "original STAB retained"
    assert _stab_multiplier(fake_tera_mon, PokemonType.FIRE) == 1.0, "no STAB"

    # Tera'd Charizard (Fire/Flying, tera Fire) using Fire move = 2.0x (adaptability)
    fake_adapt_mon = FakePokemon(
        name="AdaptTest", species="charizard",
        types=(PokemonType.FIRE, PokemonType.FLYING),
        current_hp_fraction=1.0,
        is_terastallized=True, tera_type=PokemonType.FIRE,
    )
    assert _stab_multiplier(fake_adapt_mon, PokemonType.FIRE) == 2.0, "adaptability STAB"
    assert _stab_multiplier(fake_adapt_mon, PokemonType.FLYING) == 1.5, "original type retained"

    # Effective types: tera'd = mono-type
    assert _effective_types(fake_tera_mon) == (PokemonType.FAIRY,)
    assert _effective_types(fake_adapt_mon) == (PokemonType.FIRE,)

    # Non-tera'd = original types
    normal_mon = FakePokemon(
        name="Normal", species="dragonite",
        types=(PokemonType.DRAGON, PokemonType.FLYING),
        current_hp_fraction=1.0,
    )
    assert _effective_types(normal_mon) == (PokemonType.DRAGON, PokemonType.FLYING)
    assert _stab_multiplier(normal_mon, PokemonType.DRAGON) == 1.5
    assert _stab_multiplier(normal_mon, PokemonType.FIRE) == 1.0

    # Defender type mult: tera'd Garchomp (tera Steel) hit by Fire = 2x (not 1x from Dragon/Ground)
    tera_defender = FakePokemon(
        name="TeraDefender", species="garchomp",
        types=(PokemonType.DRAGON, PokemonType.GROUND),
        current_hp_fraction=1.0,
        is_terastallized=True, tera_type=PokemonType.STEEL,
    )
    fire_mult = _defender_type_mult(PokemonType.FIRE, tera_defender, gen_data.type_chart)
    assert fire_mult == 2.0, f"Fire vs tera-Steel should be 2x, got {fire_mult}"

    # Dragon vs tera-Fairy = immune
    fairy_defender = FakePokemon(
        name="FairyDef", species="dragonite",
        types=(PokemonType.DRAGON, PokemonType.FLYING),
        current_hp_fraction=1.0,
        is_terastallized=True, tera_type=PokemonType.FAIRY,
    )
    dragon_mult = _defender_type_mult(PokemonType.DRAGON, fairy_defender, gen_data.type_chart)
    assert dragon_mult == 0.0, f"Dragon vs tera-Fairy should be immune, got {dragon_mult}"

    print("  Tera damage calc tests passed.")

    print("Observation vector smoke test passed.")

    # =========================================================
    # Damage calc modifier tests
    # =========================================================
    print("\n-- Damage calc modifier tests ---------------------")
    from brent_agent import _clamp01
    from poke_env.data import GenData

    gen_data = GenData.from_gen(9)
    dmg_builder = DeterministicBuilder()

    def _quick_dmg(
        attacker: FakePokemon,
        defender: FakePokemon,
        move: FakeMove,
        battle: FakeBattle,
        att_stats=None,
        def_stats=None,
    ):
        """Helper: run _manual_damage_calc and return (min, max) raw damage."""
        return dmg_builder._manual_damage_calc(
            attacker, defender, move, battle,
            att_stats or attacker.stats,
            def_stats or defender.stats,
        )

    # --- Test 1: Vanilla physical (no modifiers) ---
    vanilla_battle = build_fake_battle()
    vanilla_battle.weather = {}
    vanilla_battle.fields = {}
    vanilla_battle.side_conditions = {}
    vanilla_battle.opponent_side_conditions = {}
    attacker = vanilla_battle.active_pokemon
    attacker.status = None
    attacker.ability = None
    attacker.item = ""
    defender = vanilla_battle.opponent_active_pokemon
    defender.ability = None
    defender.item = ""

    tackle = FakeMove("tackle", base_power=40, category=MoveCategory.PHYSICAL, type=PokemonType.NORMAL)
    d_min, d_max = _quick_dmg(attacker, defender, tackle, vanilla_battle)
    # Level 80, 100 atk vs 100 def, 40 bp: ((2*80/5+2)*40*100/100)/50+2 = (34*40*1)/50+2 = 29.2
    # No STAB (attacker types are Bug/Poison), no modifiers => 29.2 * 0.85 = 24.82, max 29.2
    assert 24.0 < d_min < 26.0, f"vanilla min={d_min}"
    assert 28.0 < d_max < 31.0, f"vanilla max={d_max}"
    print("  Vanilla physical calc: PASS")

    # --- Test 2: STAB ---
    # Give attacker Normal type
    stab_attacker = FakePokemon(
        name="StabMon", species="stoutland",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability=None, item="",
    )
    d_min_stab, d_max_stab = _quick_dmg(stab_attacker, defender, tackle, vanilla_battle)
    # Should be 1.5x the vanilla values
    ratio = d_max_stab / d_max
    assert 1.45 < ratio < 1.55, f"STAB ratio={ratio}, expected ~1.5"
    print("  STAB multiplier: PASS")

    # --- Test 3: Choice Band ---
    cb_attacker = FakePokemon(
        name="CBMon", species="stoutland",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability=None, item="choiceband",
    )
    d_min_cb, d_max_cb = _quick_dmg(cb_attacker, defender, tackle, vanilla_battle)
    ratio_cb = d_max_cb / d_max_stab
    assert 1.45 < ratio_cb < 1.55, f"Choice Band ratio={ratio_cb}, expected ~1.5"
    print("  Choice Band: PASS")

    # --- Test 4: Life Orb ---
    lo_attacker = FakePokemon(
        name="LOMon", species="stoutland",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability=None, item="lifeorb",
    )
    d_min_lo, d_max_lo = _quick_dmg(lo_attacker, defender, tackle, vanilla_battle)
    ratio_lo = d_max_lo / d_max_stab
    assert 1.25 < ratio_lo < 1.35, f"Life Orb ratio={ratio_lo}, expected ~1.3"
    print("  Life Orb: PASS")

    # --- Test 5: Technician ---
    tech_attacker = FakePokemon(
        name="TechMon", species="stoutland",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability="technician", item="",
    )
    d_min_tech, d_max_tech = _quick_dmg(tech_attacker, defender, tackle, vanilla_battle)
    ratio_tech = d_max_tech / d_max_stab
    assert 1.45 < ratio_tech < 1.55, f"Technician ratio={ratio_tech}, expected ~1.5"
    print("  Technician (bp<=60): PASS")

    # --- Test 6: Huge Power ---
    hp_attacker = FakePokemon(
        name="HPMon", species="azumarill",
        types=(PokemonType.WATER, PokemonType.FAIRY), current_hp_fraction=1.0,
        active=True, ability="hugepower", item="",
    )
    aquajet = FakeMove("aquajet", base_power=40, category=MoveCategory.PHYSICAL, type=PokemonType.WATER, priority=1)
    d_min_hp, d_max_hp = _quick_dmg(hp_attacker, defender, aquajet, vanilla_battle)
    # vs vanilla attacker with same move (no STAB normal)
    neutral_att = FakePokemon(
        name="NeutralMon", species="test",
        types=(PokemonType.WATER, PokemonType.FAIRY), current_hp_fraction=1.0,
        active=True, ability=None, item="",
    )
    d_min_neut, d_max_neut = _quick_dmg(neutral_att, defender, aquajet, vanilla_battle)
    ratio_hp = d_max_hp / d_max_neut
    assert 1.9 < ratio_hp < 2.1, f"Huge Power ratio={ratio_hp}, expected ~2.0"
    print("  Huge Power: PASS")

    # --- Test 7: Weather (Rain + Water move) ---
    rain_battle = build_fake_battle()
    rain_battle.weather = {Weather.RAINDANCE: 1}
    rain_battle.fields = {}
    rain_battle.side_conditions = {}
    rain_battle.opponent_side_conditions = {}
    rain_battle.active_pokemon.ability = None
    rain_battle.active_pokemon.item = ""
    rain_battle.opponent_active_pokemon.ability = None
    rain_battle.opponent_active_pokemon.item = ""
    rain_battle.active_pokemon.status = None
    surf = FakeMove("surf", base_power=90, category=MoveCategory.SPECIAL, type=PokemonType.WATER)
    d_min_rain, d_max_rain = _quick_dmg(rain_battle.active_pokemon, rain_battle.opponent_active_pokemon, surf, rain_battle)
    no_rain_battle = build_fake_battle()
    no_rain_battle.weather = {}
    no_rain_battle.fields = {}
    no_rain_battle.side_conditions = {}
    no_rain_battle.opponent_side_conditions = {}
    no_rain_battle.active_pokemon.ability = None
    no_rain_battle.active_pokemon.item = ""
    no_rain_battle.opponent_active_pokemon.ability = None
    no_rain_battle.opponent_active_pokemon.item = ""
    no_rain_battle.active_pokemon.status = None
    d_min_dry, d_max_dry = _quick_dmg(no_rain_battle.active_pokemon, no_rain_battle.opponent_active_pokemon, surf, no_rain_battle)
    ratio_rain = d_max_rain / d_max_dry
    assert 1.45 < ratio_rain < 1.55, f"Rain ratio={ratio_rain}, expected ~1.5"
    print("  Rain boost: PASS")

    # --- Test 8: Reflect (physical screen) ---
    screen_battle = build_fake_battle()
    screen_battle.weather = {}
    screen_battle.fields = {}
    # Defender has Reflect up (opponent side conditions when we attack)
    screen_battle.opponent_side_conditions = {SideCondition.REFLECT: 1}
    screen_battle.side_conditions = {}
    screen_battle.active_pokemon.ability = None
    screen_battle.active_pokemon.item = ""
    screen_battle.active_pokemon.status = None
    screen_battle.opponent_active_pokemon.ability = None
    screen_battle.opponent_active_pokemon.item = ""
    d_min_scr, d_max_scr = _quick_dmg(screen_battle.active_pokemon, screen_battle.opponent_active_pokemon, tackle, screen_battle)
    ratio_scr = d_max_scr / d_max  # vs vanilla
    assert 0.45 < ratio_scr < 0.55, f"Reflect ratio={ratio_scr}, expected ~0.5"
    print("  Reflect screen: PASS")

    # --- Test 9: Body Press uses Defense stat ---
    bodypress_move = FakeMove("bodypress", base_power=80, category=MoveCategory.PHYSICAL, type=PokemonType.FIGHTING)
    # Mon with high def, low atk
    bp_attacker = FakePokemon(
        name="BPMon", species="corviknight",
        types=(PokemonType.FLYING, PokemonType.STEEL), current_hp_fraction=1.0,
        active=True, ability=None, item="",
        stats={"hp": 250, "atk": 80, "def": 200, "spa": 60, "spd": 120, "spe": 90},
    )
    d_min_bp, d_max_bp = _quick_dmg(bp_attacker, defender, bodypress_move, vanilla_battle)
    # Now test with a normal fighting move (same bp) - should use atk=80 instead of def=200
    closecombat = FakeMove("closecombat", base_power=80, category=MoveCategory.PHYSICAL, type=PokemonType.FIGHTING)
    d_min_cc, d_max_cc = _quick_dmg(bp_attacker, defender, closecombat, vanilla_battle)
    # Body Press should do 200/80 = 2.5x more than close combat
    ratio_bp = d_max_bp / d_max_cc
    assert 2.4 < ratio_bp < 2.6, f"Body Press ratio={ratio_bp}, expected ~2.5"
    print("  Body Press (uses Def): PASS")

    # --- Test 10: Burn penalty (with Guts exception) ---
    burn_attacker = FakePokemon(
        name="BurnMon", species="stoutland",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability=None, item="", status=Status.BRN,
    )
    d_min_burn, d_max_burn = _quick_dmg(burn_attacker, defender, tackle, vanilla_battle)
    ratio_burn = d_max_burn / d_max_stab  # vs STAB normal (same types)
    assert 0.45 < ratio_burn < 0.55, f"Burn ratio={ratio_burn}, expected ~0.5"

    guts_attacker = FakePokemon(
        name="GutsMon", species="stoutland",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability="guts", item="", status=Status.BRN,
    )
    d_min_guts, d_max_guts = _quick_dmg(guts_attacker, defender, tackle, vanilla_battle)
    # Guts: no burn penalty, +50% atk = should be 1.5x normal STAB
    ratio_guts = d_max_guts / d_max_stab
    assert 1.45 < ratio_guts < 1.55, f"Guts ratio={ratio_guts}, expected ~1.5"
    print("  Burn penalty + Guts: PASS")

    # --- Test 11: Thick Fat ---
    tf_defender = FakePokemon(
        name="TFMon", species="snorlax",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability="thickfat", item="",
    )
    no_tf_defender = FakePokemon(
        name="NoTFMon", species="snorlax",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability=None, item="",
    )
    fire_move = FakeMove("flamethrower", base_power=90, category=MoveCategory.SPECIAL, type=PokemonType.FIRE)
    d_min_tf, d_max_tf = _quick_dmg(attacker, tf_defender, fire_move, vanilla_battle)
    d_min_notf, d_max_notf = _quick_dmg(attacker, no_tf_defender, fire_move, vanilla_battle)
    ratio_tf = d_max_tf / d_max_notf
    assert 0.45 < ratio_tf < 0.55, f"Thick Fat ratio={ratio_tf}, expected ~0.5"
    print("  Thick Fat: PASS")

    # --- Test 12: Electric Terrain + Electric move ---
    terrain_battle = build_fake_battle()
    terrain_battle.weather = {}
    terrain_battle.fields = {Field.ELECTRIC_TERRAIN: 1}
    terrain_battle.side_conditions = {}
    terrain_battle.opponent_side_conditions = {}
    terrain_battle.active_pokemon.ability = None
    terrain_battle.active_pokemon.item = ""
    terrain_battle.active_pokemon.status = None
    terrain_battle.opponent_active_pokemon.ability = None
    terrain_battle.opponent_active_pokemon.item = ""
    no_terrain_battle = build_fake_battle()
    no_terrain_battle.weather = {}
    no_terrain_battle.fields = {}
    no_terrain_battle.side_conditions = {}
    no_terrain_battle.opponent_side_conditions = {}
    no_terrain_battle.active_pokemon.ability = None
    no_terrain_battle.active_pokemon.item = ""
    no_terrain_battle.active_pokemon.status = None
    no_terrain_battle.opponent_active_pokemon.ability = None
    no_terrain_battle.opponent_active_pokemon.item = ""
    tbolt = FakeMove("thunderbolt", base_power=90, category=MoveCategory.SPECIAL, type=PokemonType.ELECTRIC)
    # Use a non-Ground defender so Electric isn't immune
    terrain_target = FakePokemon(
        name="TerrTarget", species="test",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability=None, item="",
    )
    # Dragonite is Dragon/Flying — Flying type means not grounded, no terrain boost
    # Use a grounded attacker for terrain test
    grounded_att = FakePokemon(
        name="GroundedAtt", species="test",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability=None, item="",
    )
    terrain_battle.active_pokemon = grounded_att
    no_terrain_battle.active_pokemon = grounded_att
    d_min_et, d_max_et = _quick_dmg(grounded_att, terrain_target, tbolt, terrain_battle)
    d_min_noet, d_max_noet = _quick_dmg(grounded_att, terrain_target, tbolt, no_terrain_battle)
    ratio_et = d_max_et / d_max_noet
    assert 1.25 < ratio_et < 1.35, f"Electric Terrain ratio={ratio_et}, expected ~1.3"
    print("  Electric Terrain: PASS")

    # --- Test 13: Eviolite ---
    evo_defender = FakePokemon(
        name="EvoMon", species="chansey",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability=None, item="eviolite",
    )
    d_min_evo, d_max_evo = _quick_dmg(attacker, evo_defender, tackle, vanilla_battle)
    ratio_evo = d_max_evo / d_max
    assert 0.6 < ratio_evo < 0.7, f"Eviolite ratio={ratio_evo}, expected ~0.667"
    print("  Eviolite: PASS")

    # --- Test 14: Sandstorm SpD boost for Rock types ---
    sand_battle = build_fake_battle()
    sand_battle.weather = {Weather.SANDSTORM: 1}
    sand_battle.fields = {}
    sand_battle.side_conditions = {}
    sand_battle.opponent_side_conditions = {}
    sand_battle.active_pokemon.ability = None
    sand_battle.active_pokemon.item = ""
    sand_battle.active_pokemon.status = None
    rock_defender = FakePokemon(
        name="RockMon", species="tyranitar",
        types=(PokemonType.ROCK, PokemonType.DARK), current_hp_fraction=1.0,
        active=True, ability=None, item="",
    )
    ice_beam = FakeMove("icebeam", base_power=90, category=MoveCategory.SPECIAL, type=PokemonType.ICE)
    d_min_sand, d_max_sand = _quick_dmg(sand_battle.active_pokemon, rock_defender, ice_beam, sand_battle)
    d_min_nosand, d_max_nosand = _quick_dmg(no_terrain_battle.active_pokemon, rock_defender, ice_beam, no_terrain_battle)
    ratio_sand = d_max_sand / d_max_nosand
    assert 0.6 < ratio_sand < 0.7, f"Sandstorm SpD ratio={ratio_sand}, expected ~0.667"
    print("  Sandstorm SpD boost: PASS")

    # --- Test 15: Multiscale at full HP ---
    ms_defender = FakePokemon(
        name="MSMon", species="dragonite",
        types=(PokemonType.DRAGON, PokemonType.FLYING), current_hp_fraction=1.0,
        active=True, ability="multiscale", item="",
    )
    d_min_ms, d_max_ms = _quick_dmg(attacker, ms_defender, ice_beam, vanilla_battle)
    ms_defender_hurt = FakePokemon(
        name="MSMon", species="dragonite",
        types=(PokemonType.DRAGON, PokemonType.FLYING), current_hp_fraction=0.9,
        active=True, ability="multiscale", item="",
    )
    d_min_ms2, d_max_ms2 = _quick_dmg(attacker, ms_defender_hurt, ice_beam, vanilla_battle)
    ratio_ms = d_max_ms / d_max_ms2
    assert 0.45 < ratio_ms < 0.55, f"Multiscale ratio={ratio_ms}, expected ~0.5"
    print("  Multiscale: PASS")

    # --- Test 16: Adaptability (non-tera) ---
    adapt_attacker = FakePokemon(
        name="AdaptMon", species="porygonz",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability="adaptability", item="",
    )
    d_min_adapt, d_max_adapt = _quick_dmg(adapt_attacker, defender, tackle, vanilla_battle)
    ratio_adapt = d_max_adapt / d_max_stab  # vs normal STAB (1.5x)
    # Adaptability = 2.0x STAB vs 1.5x, so ratio should be ~1.333
    assert 1.3 < ratio_adapt < 1.4, f"Adaptability ratio={ratio_adapt}, expected ~1.333"
    print("  Adaptability: PASS")

    # --- Test 17: Foul Play uses defender's Attack ---
    foulplay = FakeMove("foulplay", base_power=95, category=MoveCategory.PHYSICAL, type=PokemonType.DARK)
    fp_attacker = FakePokemon(
        name="FPMon", species="sableye",
        types=(PokemonType.DARK, PokemonType.GHOST), current_hp_fraction=1.0,
        active=True, ability=None, item="",
        stats={"hp": 100, "atk": 50, "def": 100, "spa": 100, "spd": 100, "spe": 100},
    )
    high_atk_defender = FakePokemon(
        name="HighAtk", species="test",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability=None, item="",
        stats={"hp": 100, "atk": 200, "def": 100, "spa": 100, "spd": 100, "spe": 100},
    )
    low_atk_defender = FakePokemon(
        name="LowAtk", species="test",
        types=(PokemonType.NORMAL,), current_hp_fraction=1.0,
        active=True, ability=None, item="",
        stats={"hp": 100, "atk": 50, "def": 100, "spa": 100, "spd": 100, "spe": 100},
    )
    d_min_fp_high, d_max_fp_high = _quick_dmg(fp_attacker, high_atk_defender, foulplay, vanilla_battle)
    d_min_fp_low, d_max_fp_low = _quick_dmg(fp_attacker, low_atk_defender, foulplay, vanilla_battle)
    ratio_fp = d_max_fp_high / d_max_fp_low
    assert 3.5 < ratio_fp < 4.5, f"Foul Play ratio={ratio_fp}, expected ~4.0"
    print("  Foul Play (uses target's Atk): PASS")

    print("  All damage calc modifier tests passed!")

    # =========================================================
    # Move block feature tests (new status-move features)
    # =========================================================
    print("\n-- Move block feature tests ------------------------")

    # Test setup move detection
    assert dmg_builder._move_is_setup(FakeMove("swordsdance", self_boost={"atk": 2})) == 1.0
    assert dmg_builder._move_is_setup(FakeMove("calmmind", self_boost={"spa": 1, "spd": 1})) == 1.0
    assert dmg_builder._move_is_setup(FakeMove("tackle")) == 0.0
    assert dmg_builder._move_is_setup(FakeMove("closecombat", base_power=120, self_boost={"def": -1, "spd": -1})) == 0.0  # net negative
    print("  is_setup detection: PASS")

    # Test hazard move detection
    assert dmg_builder._move_is_hazard(FakeMove("stealthrock")) == 1.0
    assert dmg_builder._move_is_hazard(FakeMove("spikes")) == 1.0
    assert dmg_builder._move_is_hazard(FakeMove("tackle")) == 0.0
    print("  is_hazard detection: PASS")

    # Test recovery move detection
    assert dmg_builder._move_is_recovery(FakeMove("recover", heal=0.5)) == 1.0
    assert dmg_builder._move_is_recovery(FakeMove("roost", heal=0.5)) == 1.0
    assert dmg_builder._move_is_recovery(FakeMove("tackle")) == 0.0
    assert dmg_builder._move_is_recovery(FakeMove("moonlight", heal=0.5)) == 1.0
    print("  is_recovery detection: PASS")

    # Test self_def_delta and self_spd_delta
    assert dmg_builder._move_self_delta(FakeMove("calmmind", self_boost={"spa": 1, "spd": 1}), "def") == 0.0
    assert dmg_builder._move_self_delta(FakeMove("calmmind", self_boost={"spa": 1, "spd": 1}), "spd") == 0.5
    assert dmg_builder._move_self_delta(FakeMove("irondefense", self_boost={"def": 2}), "def") == 1.0
    assert dmg_builder._move_self_delta(FakeMove("closecombat", base_power=120, self_boost={"def": -1, "spd": -1}), "def") == -0.5
    print("  self_def_delta / self_spd_delta: PASS")

    # Verify vector placement: setup move in slot 0 should appear at correct index
    setup_battle = build_fake_battle()
    # Replace first available move with Swords Dance
    sd_move = Move("swordsdance", 9)
    setup_battle.available_moves = [sd_move] + list(setup_battle.available_moves[1:])
    setup_vector = builder.embed_battle(setup_battle)
    # Index 27 (is_setup) of move slot 0
    setup_idx = MY_MOVES_START + 27
    assert setup_vector[setup_idx] == 1.0, f"Swords Dance is_setup flag not set at index {setup_idx}, got {setup_vector[setup_idx]}"
    print("  Vector placement (is_setup): PASS")

    print("  All move block feature tests passed!")

    # =========================================================
    # Volatile effect / Serene Grace / Sheer Force tests
    # =========================================================
    print("\n-- Volatile effect + ability tests ------------------")

    # --- Flinch chance basic ---
    air_slash = FakeMove(
        "airslash", base_power=75, category=MoveCategory.SPECIAL,
        type=PokemonType.FLYING,
        secondary=[{"chance": 30, "volatileStatus": "flinch"}],
    )
    flinch_chance = dmg_builder._move_effect_chance(air_slash, volatile_effect=Effect.FLINCH)
    assert abs(flinch_chance - 0.3) < 0.01, f"Flinch basic: got {flinch_chance}, expected 0.3"
    print("  Flinch chance basic: PASS")

    # --- Serene Grace doubles flinch ---
    sg_attacker = FakePokemon(
        name="SGMon", species="togekiss",
        types=(PokemonType.FAIRY, PokemonType.FLYING), current_hp_fraction=1.0,
        active=True, ability="serenegrace", item="",
    )
    flinch_sg = dmg_builder._move_effect_chance(air_slash, volatile_effect=Effect.FLINCH, attacker=sg_attacker)
    assert abs(flinch_sg - 0.6) < 0.01, f"Serene Grace flinch: got {flinch_sg}, expected 0.6"
    print("  Serene Grace doubles flinch (30% -> 60%): PASS")

    # --- Serene Grace doubles status chances ---
    discharge = FakeMove(
        "discharge", base_power=80, category=MoveCategory.SPECIAL,
        type=PokemonType.ELECTRIC,
        secondary=[{"chance": 30, "status": "par"}],
    )
    par_sg = dmg_builder._move_effect_chance(discharge, statuses=(Status.PAR,), attacker=sg_attacker)
    assert abs(par_sg - 0.6) < 0.01, f"Serene Grace paralysis: got {par_sg}, expected 0.6"
    par_normal = dmg_builder._move_effect_chance(discharge, statuses=(Status.PAR,))
    assert abs(par_normal - 0.3) < 0.01, f"Normal paralysis: got {par_normal}, expected 0.3"
    print("  Serene Grace doubles paralysis (30% -> 60%): PASS")

    # --- Serene Grace caps at 100% ---
    body_slam = FakeMove(
        "bodyslam", base_power=85, category=MoveCategory.PHYSICAL,
        type=PokemonType.NORMAL,
        secondary=[{"chance": 60, "status": "par"}],  # 60% * 2 = 120% -> cap at 100%
    )
    par_cap = dmg_builder._move_effect_chance(body_slam, statuses=(Status.PAR,), attacker=sg_attacker)
    assert abs(par_cap - 1.0) < 0.01, f"Serene Grace cap: got {par_cap}, expected 1.0"
    print("  Serene Grace caps at 100%: PASS")

    # --- Sheer Force zeroes secondary flinch ---
    sf_attacker = FakePokemon(
        name="SFMon", species="darmanitan",
        types=(PokemonType.FIRE,), current_hp_fraction=1.0,
        active=True, ability="sheerforce", item="",
    )
    flinch_sf = dmg_builder._move_effect_chance(air_slash, volatile_effect=Effect.FLINCH, attacker=sf_attacker)
    assert flinch_sf == 0.0, f"Sheer Force flinch: got {flinch_sf}, expected 0.0"
    print("  Sheer Force zeroes secondary flinch: PASS")

    # --- Sheer Force does NOT affect direct status ---
    willowisp = FakeMove(
        "willowisp", base_power=0, category=MoveCategory.STATUS,
        type=PokemonType.FIRE, status=Status.BRN,
    )
    burn_sf = dmg_builder._move_effect_chance(willowisp, statuses=(Status.BRN,), attacker=sf_attacker)
    assert burn_sf == 1.0, f"Sheer Force direct status: got {burn_sf}, expected 1.0"
    print("  Sheer Force preserves direct status (Will-O-Wisp): PASS")

    # --- Target stat drop chances ---
    moonblast = FakeMove(
        "moonblast", base_power=95, category=MoveCategory.SPECIAL,
        type=PokemonType.FAIRY,
        secondary=[{"chance": 30, "boosts": {"spa": -1}}],
    )
    spa_drop = dmg_builder._move_target_stat_drop_chance(moonblast, "spa")
    assert abs(spa_drop - 0.3) < 0.01, f"Moonblast SpA drop: got {spa_drop}, expected 0.3"
    spd_drop = dmg_builder._move_target_stat_drop_chance(moonblast, "spd")
    assert spd_drop == 0.0, f"Moonblast SpD drop should be 0, got {spd_drop}"
    print("  Target stat drop (Moonblast SpA -30%): PASS")

    shadow_ball = FakeMove(
        "shadowball", base_power=80, category=MoveCategory.SPECIAL,
        type=PokemonType.GHOST,
        secondary=[{"chance": 20, "boosts": {"spd": -1}}],
    )
    spd_drop_sb = dmg_builder._move_target_stat_drop_chance(shadow_ball, "spd")
    assert abs(spd_drop_sb - 0.2) < 0.01, f"Shadow Ball SpD drop: got {spd_drop_sb}, expected 0.2"
    print("  Target stat drop (Shadow Ball SpD -20%): PASS")

    # --- Icy Wind: 100% speed drop ---
    icy_wind = FakeMove(
        "icywind", base_power=55, category=MoveCategory.SPECIAL,
        type=PokemonType.ICE,
        secondary=[{"chance": 100, "boosts": {"spe": -1}}],
    )
    spe_drop = dmg_builder._move_target_stat_drop_chance(icy_wind, "spe")
    assert abs(spe_drop - 1.0) < 0.01, f"Icy Wind Spe drop: got {spe_drop}, expected 1.0"
    print("  Target stat drop (Icy Wind Spe -100%): PASS")

    # --- Serene Grace + target stat drops ---
    spa_drop_sg = dmg_builder._move_target_stat_drop_chance(moonblast, "spa", attacker=sg_attacker)
    assert abs(spa_drop_sg - 0.6) < 0.01, f"Serene Grace Moonblast SpA drop: got {spa_drop_sg}, expected 0.6"
    print("  Serene Grace doubles target stat drop (30% -> 60%): PASS")

    # --- Sheer Force zeroes target stat drops ---
    spa_drop_sf = dmg_builder._move_target_stat_drop_chance(moonblast, "spa", attacker=sf_attacker)
    assert spa_drop_sf == 0.0, f"Sheer Force SpA drop: got {spa_drop_sf}, expected 0.0"
    print("  Sheer Force zeroes target stat drops: PASS")

    # --- No attacker = backward compatible (no ability modification) ---
    flinch_no_att = dmg_builder._move_effect_chance(air_slash, volatile_effect=Effect.FLINCH, attacker=None)
    assert abs(flinch_no_att - 0.3) < 0.01, f"No attacker flinch: got {flinch_no_att}, expected 0.3"
    print("  Backward compatible (attacker=None): PASS")

    print("  All volatile effect + ability tests passed!")

    # =========================================================
    # Action mask smoke test: Ditto Transform 5-move collision
    # =========================================================
    print("\n-- Action mask 5-move collision test ---------------")
    from poke_env.environment.singles_env import SinglesEnv

    # Simulate a Ditto that Transformed: its moves dict has the original
    # "transform" plus 4 copied moves.  Only the 4 copied moves are in
    # available_moves, but the moves dict has 5 entries total.
    ditto = FakePokemon(
        name="Ditto", species="ditto",
        types=(PokemonType.NORMAL,),
        current_hp_fraction=1.0,
        active=True,
    )
    # Original move (index 0 in moves dict) — NOT available after Transform
    transform_move = FakeMove(id="transform")
    # 4 copied moves (indices 1-4) — these ARE available
    heavyslam = FakeMove(id="heavyslam", base_power=120, category=MoveCategory.PHYSICAL, type=PokemonType.STEEL)
    stealthrock = FakeMove(id="stealthrock", base_power=0, category=MoveCategory.STATUS, type=PokemonType.ROCK)
    earthquake = FakeMove(id="earthquake", base_power=100, category=MoveCategory.PHYSICAL, type=PokemonType.GROUND)
    roar = FakeMove(id="roar", base_power=0, category=MoveCategory.STATUS, type=PokemonType.NORMAL)

    ditto.moves = OrderedDict([
        ("transform", transform_move),
        ("heavyslam", heavyslam),
        ("stealthrock", stealthrock),
        ("earthquake", earthquake),
        ("roar", roar),
    ])

    lugia = FakePokemon(
        name="Lugia", species="lugia",
        types=(PokemonType.PSYCHIC, PokemonType.FLYING),
        current_hp_fraction=1.0,
    )

    mask_battle = FakeBattle(
        battle_tag="mask-test",
        player_role="p1",
        opponent_role="p2",
        active_pokemon=ditto,
        opponent_active_pokemon=FakePokemon(
            name="Foe", species="garchomp",
            types=(PokemonType.DRAGON, PokemonType.GROUND),
            current_hp_fraction=1.0,
        ),
        available_moves=[heavyslam, stealthrock, earthquake, roar],
        team=OrderedDict([
            ("p1a", ditto),
            ("p1b", lugia),
        ]),
        opponent_team=OrderedDict([
            ("p2a", FakePokemon(
                name="Foe", species="garchomp",
                types=(PokemonType.DRAGON, PokemonType.GROUND),
                current_hp_fraction=1.0,
            )),
        ]),
        available_switches=[lugia],
    )

    # Verify upstream poke_env bug still exists
    upstream_mask = SinglesEnv.get_action_mask(mask_battle)
    action_space_size = SinglesEnv.get_action_space_size(9)  # 26
    assert len(upstream_mask) == action_space_size, f"mask length {len(upstream_mask)} != {action_space_size}"
    upstream_bug = any(upstream_mask[i] == 1 for i in range(10, 14))
    if upstream_bug:
        print("  Upstream poke_env bug still present (5-move overflow into mega zone)")
    else:
        print("  Upstream bug appears fixed — our override may no longer be needed")

    # Verify OUR override fixes it
    fixed_mask = BrentsRLAgent.get_action_mask(mask_battle)
    assert len(fixed_mask) == action_space_size
    move_actions = [i for i in range(6, 10) if fixed_mask[i] == 1]
    mega_zone = [i for i in range(10, 14) if fixed_mask[i] == 1]
    assert not mega_zone, f"Override still has mega zone collision: {mega_zone}"
    # The 4 available moves (heavyslam, stealthrock, earthquake, roar) are at
    # indices 1-4 in active_pokemon.moves (index 0 is transform, not available).
    # So legal actions should be 7,8,9 (indices 1,2,3) — index 4 (roar) is
    # capped by the i<4 guard. Action 6 (transform) is correctly excluded.
    assert 6 not in move_actions, "transform (action 6) should not be legal"
    assert len(move_actions) >= 3, f"Expected at least 3 legal moves, got {move_actions}"
    print(f"  Override fix verified: move actions={move_actions}, mega zone={mega_zone}")
    print("  Action mask 5-move collision: FIXED")

    # Count total legal actions for sanity
    legal_count = sum(fixed_mask)
    print(f"  Total legal actions in fixed mask: {legal_count}")
    print(f"  Fixed mask: {fixed_mask}")

    print("\n== ALL TESTS PASSED ==")


if __name__ == "__main__":
    main()