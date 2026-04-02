from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium.spaces import Box
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.calc import calculate_damage
from poke_env.calc.damage_calc_gen9 import get_item_boost_type
from poke_env.data import GenData
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.battle_order import BattleOrder
from poke_env.ps_client import AccountConfiguration

from randbats_data import RandbatsMeta

POKE_ENV_REWARD_KEYS = (
    "fainted_value",
    "hp_value",
    "status_value",
    "victory_value",
)

REWARD_CONFIG = {
    "fainted_value": 1.0,
    "hp_value": 0.5,
    "status_value": 0.25,
    "victory_value": 30.0,
    "penalty_redundant_stealthrock": -0.1,
    "penalty_redundant_stickyweb": -0.1,
    "penalty_redundant_spikes": -0.1,
    "penalty_redundant_status": -0.1,
    "penalty_bad_encore": -0.1,
    "penalty_ineffective_heal": -0.1,
    "penalty_wasteful_heal_overflow": -0.05,
    "penalty_redundant_self_drop_move": -0.1,
    "penalty_unsafe_stay_in_with_fast_ko_switch": -0.2,
    "bonus_good_heal_timing": 0.2,
    "bonus_good_attack_selection": 0.0,
    "bonus_good_safe_switch": 0.2,
    "bonus_good_tera": 0.5,
    "penalty_abandon_boosted_mon": -0.1,
    "penalty_heal_satiation": -0.1,
    "penalty_wasted_free_switch": -0.1,
}

DECISION_AUDIT_SAMPLE_LIMIT = 20

VECTOR_LENGTH = 658
MOVE_BLOCK_SIZE = 25
MY_BENCH_SLOT_SIZE = 53
OPP_BENCH_SLOT_SIZE = 20
BENCH_MOVE_FLAG_SIZE = 8
OPP_THREAT_ROWS = 4
OPP_THREAT_ROW_SIZE = 8
MY_ACTIVE_BLOCK_SIZE = 60
OPP_ACTIVE_BLOCK_SIZE = 41

TURN_INDEX = 0
WEATHER_START = 1
TERRAIN_START = 5
TRICK_ROOM_INDEX = 9
MY_SIDE_START = 10
OPP_SIDE_START = 17
MY_ACTIVE_START = 24
OPP_ACTIVE_START = MY_ACTIVE_START + MY_ACTIVE_BLOCK_SIZE  # 84
SPEED_ADVANTAGE_INDEX = OPP_ACTIVE_START + OPP_ACTIVE_BLOCK_SIZE  # 125
MY_MOVES_START = SPEED_ADVANTAGE_INDEX + 1  # 126
MY_BENCH_START = MY_MOVES_START + 4 * MOVE_BLOCK_SIZE  # 226
OPP_BENCH_START = MY_BENCH_START + 5 * MY_BENCH_SLOT_SIZE  # 486
TARGETING_START = OPP_BENCH_START + 5 * OPP_BENCH_SLOT_SIZE  # 586
MY_TEAM_REVEALED_START = TARGETING_START + 20  # 606
OPP_THREAT_START = MY_TEAM_REVEALED_START + 6  # 612
OPP_MOVES_VS_ME_START = OPP_THREAT_START + 2
OPP_THREAT_OHKO_START = OPP_THREAT_START + OPP_THREAT_ROWS * OPP_THREAT_ROW_SIZE
OPP_THREAT_CONFIDENCE_START = OPP_THREAT_OHKO_START + 6
ON_RECHARGE_INDEX = OPP_THREAT_CONFIDENCE_START + 2

TYPE_ORDER = (
    PokemonType.BUG,
    PokemonType.DARK,
    PokemonType.DRAGON,
    PokemonType.ELECTRIC,
    PokemonType.FAIRY,
    PokemonType.FIGHTING,
    PokemonType.FIRE,
    PokemonType.FLYING,
    PokemonType.GHOST,
    PokemonType.GRASS,
    PokemonType.GROUND,
    PokemonType.ICE,
    PokemonType.NORMAL,
    PokemonType.POISON,
    PokemonType.PSYCHIC,
    PokemonType.ROCK,
    PokemonType.STEEL,
    PokemonType.WATER,
)

WEATHER_ORDER = (
    Weather.RAINDANCE,
    Weather.SUNNYDAY,
    Weather.SANDSTORM,
    Weather.SNOW,
)

TERRAIN_ORDER = (
    Field.ELECTRIC_TERRAIN,
    Field.GRASSY_TERRAIN,
    Field.PSYCHIC_TERRAIN,
    Field.MISTY_TERRAIN,
)

BOOST_ORDER = ("atk", "def", "spa", "spd", "spe", "evasion", "accuracy")
STATUS_ORDER = (
    Status.BRN,
    Status.PAR,
    Status.SLP,
    Status.FRZ,
)
VOLATILE_ORDER = (
    Effect.SUBSTITUTE,
    Effect.TAUNT,
    Effect.ENCORE,
    Effect.CONFUSION,
)
SIDE_CONDITION_ORDER = (
    SideCondition.STEALTH_ROCK,
    SideCondition.SPIKES,
    SideCondition.TOXIC_SPIKES,
    SideCondition.STICKY_WEB,
    SideCondition.REFLECT,
    SideCondition.LIGHT_SCREEN,
    SideCondition.AURORA_VEIL,
)

PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "teleport"}
WEATHER_HEAL_MOVES = {"synthesis", "moonlight", "morningsun"}
RECOVERY_ITEMS = {"leftovers", "blacksludge", "shellbell", "sitrusberry", "oranberry"}
DAMAGE_BOOST_ITEMS = {
    "choiceband",
    "choicespecs",
    "lifeorb",
    "expertbelt",
    "muscleband",
    "wiseglasses",
    "lightball",
    "thickclub",
    "adamantorb",
    "lustrousorb",
    "griseousorb",
}
SPEED_BOOST_ITEMS = {"choicescarf"}
SPEED_DROP_ITEMS = {
    "ironball",
    "machobrace",
    "poweranklet",
    "powerband",
    "powerbelt",
    "powerbracer",
    "powerlens",
    "powerweight",
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _stat_stage_multiplier(stage: int) -> float:
    if stage >= 0:
        return (2.0 + stage) / 2.0
    return 2.0 / (2.0 - stage)


def _safe_hp_fraction(mon: Optional[Pokemon]) -> float:
    if mon is None:
        return 0.0
    return _clamp01(float(getattr(mon, "current_hp_fraction", 0.0) or 0.0))


def _effective_types(mon: Pokemon) -> tuple[PokemonType, ...]:
    """Return the pokemon's effective types, accounting for terastallization.
    Tera'd pokemon become mono-type (their tera type only)."""
    if getattr(mon, "is_terastallized", False):
        tera_type = getattr(mon, "tera_type", None) or getattr(mon, "type_1", None)
        if tera_type is not None:
            return (tera_type,)
    return mon.types


def _stab_multiplier(attacker: Pokemon, move_type: PokemonType) -> float:
    """Return STAB multiplier accounting for tera mechanics.
    Gen 9: tera'd pokemon get STAB from both original types AND tera type.
    If tera type matches an original type, STAB is 2.0x (adaptability-like)."""
    if getattr(attacker, "is_terastallized", False):
        tera_type = getattr(attacker, "tera_type", None)
        original_types = attacker.types  # .types returns base types
        if tera_type is not None:
            tera_stab = move_type == tera_type
            original_stab = move_type in original_types
            if tera_stab and original_stab:
                return 2.0  # Adaptability-like bonus
            if tera_stab or original_stab:
                return 1.5
            return 1.0
    return 1.5 if move_type in attacker.types else 1.0


# Abilities that grant full immunity to a move type
_ABILITY_TYPE_IMMUNITIES: dict[str, PokemonType] = {
    "levitate": PokemonType.GROUND,
    "voltabsorb": PokemonType.ELECTRIC,
    "lightningrod": PokemonType.ELECTRIC,
    "motordrive": PokemonType.ELECTRIC,
    "waterabsorb": PokemonType.WATER,
    "stormdrain": PokemonType.WATER,
    "dryskin": PokemonType.WATER,
    "flashfire": PokemonType.FIRE,
    "sapsipper": PokemonType.GRASS,
    "windrider": PokemonType.FLYING,
    "eartheater": PokemonType.GROUND,
}


def _ability_immune(defender: Pokemon, move_type: PokemonType) -> bool:
    """Check if defender's ability grants immunity to the move type."""
    ability = getattr(defender, "ability", None)
    if ability is None:
        # Check possible_abilities for unrevealed mons — if ALL abilities
        # grant immunity, treat as immune. Otherwise assume not immune.
        possible = getattr(defender, "possible_abilities", None)
        if possible and len(possible) > 0:
            return all(
                _ABILITY_TYPE_IMMUNITIES.get(a.lower().replace(" ", ""), None) == move_type
                for a in possible
            )
        return False
    return _ABILITY_TYPE_IMMUNITIES.get(ability.lower().replace(" ", ""), None) == move_type


def _defender_type_mult(
    move_type: PokemonType,
    defender: Pokemon,
    type_chart: Any,
) -> float:
    """Type effectiveness accounting for tera (mono-type when tera'd) and ability immunities."""
    if _ability_immune(defender, move_type):
        return 0.0
    if getattr(defender, "is_terastallized", False):
        tera_type = getattr(defender, "tera_type", None) or defender.type_1
        return move_type.damage_multiplier(tera_type, None, type_chart=type_chart)
    return move_type.damage_multiplier(defender.type_1, defender.type_2, type_chart=type_chart)


def _safe_identifier(mon: Optional[Pokemon], role: Optional[str]) -> Optional[str]:
    if mon is None or role is None:
        return None
    try:
        return mon.identifier(role)
    except Exception:
        return None


def _battle_tag(battle: AbstractBattle) -> str:
    return getattr(battle, "battle_tag", "default")


def _mon_key(mon: Pokemon) -> str:
    name = getattr(mon, "name", "") or ""
    species = getattr(mon, "species", "") or ""
    return f"{name}:{species}"


def _poison_severity(status: Optional[Status]) -> float:
    if status == Status.TOX:
        return 1.0
    if status == Status.PSN:
        return 0.5
    return 0.0


@dataclass(frozen=True)
class OpponentThreatEntry:
    move: Move
    move_prob: float
    revealed_flag: float


@dataclass(frozen=True)
class ThreatAssessment:
    posterior: Dict[str, float]
    threat_entries: Tuple[OpponentThreatEntry, ...]
    active_max_threat: float
    active_ohko_risk: float
    active_speed: float | None
    opponent_speed: float | None


@dataclass(frozen=True)
class TacticalLeverMatch:
    reason: str
    reward: float
    details: Dict[str, Any]
    record_audit: bool = False


@dataclass
class TacticalRewardContext:
    battle_tag: str
    action: Move | Pokemon | None
    matches: Tuple[TacticalLeverMatch, ...]


class BrentObservationVectorBuilder:
    def __init__(self) -> None:
        self._my_team_revealed_memory: Dict[str, set[str]] = {}
        self._damage_cache: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
        self._fallback_counts: Dict[Tuple[str, str, str], int] = {}
        self._fallback_samples: list[Dict[str, Any]] = []
        self._inferred_move_cache: Dict[str, Optional[Move]] = {}
        self._meta = RandbatsMeta()
        self._gen_data = GenData.from_gen(9)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        self._damage_cache = {}
        self._update_reveal_memory(battle)
        try:
            vector = np.zeros(VECTOR_LENGTH, dtype=np.float32)

            self._fill_global_features(vector, battle)
            self._fill_side_conditions(vector, MY_SIDE_START, battle.side_conditions)
            self._fill_side_conditions(vector, OPP_SIDE_START, battle.opponent_side_conditions)
            self._fill_active_block(vector, MY_ACTIVE_START, battle.active_pokemon, battle.available_moves, battle)
            self._fill_my_active_tera(vector, MY_ACTIVE_START, battle.active_pokemon, battle)
            self._fill_active_block(
                vector,
                OPP_ACTIVE_START,
                battle.opponent_active_pokemon,
                tuple(battle.opponent_active_pokemon.moves.values()) if battle.opponent_active_pokemon else (),
                battle,
            )
            self._fill_opp_active_tera(vector, OPP_ACTIVE_START, battle.opponent_active_pokemon)
            vector[SPEED_ADVANTAGE_INDEX] = self._speed_advantage(battle)
            self._fill_available_move_blocks(vector, battle)
            self._fill_my_bench(vector, battle)
            self._fill_opponent_bench(vector, battle)
            self._fill_targeting_matrix(vector, battle)
            self._fill_opponent_threat_features(vector, battle)
            vector[ON_RECHARGE_INDEX] = self._on_recharge(battle.active_pokemon)
            return vector
        finally:
            self._damage_cache.clear()
            if getattr(battle, "finished", False):
                self._my_team_revealed_memory.pop(_battle_tag(battle), None)

    def verify_battle_embedding(
        self,
        battle: AbstractBattle,
        vector: np.ndarray,
    ) -> list[str]:
        issues: list[str] = []
        if vector.shape != (VECTOR_LENGTH,):
            issues.append(f"shape mismatch: expected {(VECTOR_LENGTH,)}, got {vector.shape}")
            return issues

        self._verify_scalar(
            issues,
            "turn",
            vector[TURN_INDEX],
            float(getattr(battle, "turn", 0)) / 100.0,
        )
        for idx, weather in enumerate(WEATHER_ORDER, start=WEATHER_START):
            self._verify_scalar(
                issues,
                f"weather:{weather.name}",
                vector[idx],
                1.0 if weather in battle.weather else 0.0,
            )
        for idx, terrain in enumerate(TERRAIN_ORDER, start=TERRAIN_START):
            self._verify_scalar(
                issues,
                f"terrain:{terrain.name}",
                vector[idx],
                1.0 if terrain in battle.fields else 0.0,
            )
        self._verify_scalar(
            issues,
            "trick_room",
            vector[TRICK_ROOM_INDEX],
            1.0 if Field.TRICK_ROOM in battle.fields else 0.0,
        )

        self._verify_side_conditions(issues, vector, MY_SIDE_START, battle.side_conditions, "my_side")
        self._verify_side_conditions(
            issues,
            vector,
            OPP_SIDE_START,
            battle.opponent_side_conditions,
            "opp_side",
        )
        self._verify_active_block(issues, vector, MY_ACTIVE_START, MY_ACTIVE_BLOCK_SIZE, battle.active_pokemon, "my_active")
        self._verify_active_block(
            issues,
            vector,
            OPP_ACTIVE_START,
            OPP_ACTIVE_BLOCK_SIZE,
            battle.opponent_active_pokemon,
            "opp_active",
        )
        self._verify_scalar(
            issues,
            "speed_advantage",
            vector[SPEED_ADVANTAGE_INDEX],
            self._speed_advantage(battle),
        )
        self._verify_scalar(
            issues,
            "on_recharge",
            vector[ON_RECHARGE_INDEX],
            self._on_recharge(battle.active_pokemon),
        )
        self._verify_opponent_bench_leaks(issues, vector, battle)
        self._verify_opponent_threat_ranges(issues, vector, battle)
        return issues

    def _update_reveal_memory(self, battle: AbstractBattle) -> None:
        memory = self._my_team_revealed_memory.setdefault(_battle_tag(battle), set())
        for mon in battle.team.values():
            if mon.active or mon.revealed:
                memory.add(_mon_key(mon))

    def _fill_global_features(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        vector[TURN_INDEX] = float(getattr(battle, "turn", 0)) / 100.0
        for idx, weather in enumerate(WEATHER_ORDER, start=WEATHER_START):
            vector[idx] = 1.0 if weather in battle.weather else 0.0
        for idx, terrain in enumerate(TERRAIN_ORDER, start=TERRAIN_START):
            vector[idx] = 1.0 if terrain in battle.fields else 0.0
        vector[TRICK_ROOM_INDEX] = 1.0 if Field.TRICK_ROOM in battle.fields else 0.0

    def _fill_side_conditions(
        self,
        vector: np.ndarray,
        start: int,
        side_conditions: Dict[SideCondition, int],
    ) -> None:
        vector[start] = 1.0 if SideCondition.STEALTH_ROCK in side_conditions else 0.0
        vector[start + 1] = _clamp01(side_conditions.get(SideCondition.SPIKES, 0) / 3.0)
        vector[start + 2] = _clamp01(side_conditions.get(SideCondition.TOXIC_SPIKES, 0) / 2.0)
        vector[start + 3] = 1.0 if SideCondition.STICKY_WEB in side_conditions else 0.0
        vector[start + 4] = 1.0 if SideCondition.REFLECT in side_conditions else 0.0
        vector[start + 5] = 1.0 if SideCondition.LIGHT_SCREEN in side_conditions else 0.0
        vector[start + 6] = 1.0 if SideCondition.AURORA_VEIL in side_conditions else 0.0

    def _fill_active_block(
        self,
        vector: np.ndarray,
        start: int,
        mon: Optional[Pokemon],
        moves: Sequence[Move],
        battle: AbstractBattle,
    ) -> None:
        if mon is None:
            return

        vector[start] = _safe_hp_fraction(mon)
        self._fill_type_one_hot(vector, start + 1, _effective_types(mon))

        for offset, stat in enumerate(BOOST_ORDER):
            vector[start + 19 + offset] = _clamp(float(mon.boosts.get(stat, 0)) / 6.0, -1.0, 1.0)

        for offset, status in enumerate(STATUS_ORDER):
            vector[start + 26 + offset] = 1.0 if mon.status == status else 0.0
        vector[start + 30] = _poison_severity(mon.status)

        for offset, effect in enumerate(VOLATILE_ORDER):
            vector[start + 31 + offset] = 1.0 if effect in mon.effects else 0.0

        item_flags = self._item_capabilities(mon, moves, battle)
        for offset, value in enumerate(item_flags):
            vector[start + 35 + offset] = value

    def _fill_my_active_tera(
        self,
        vector: np.ndarray,
        start: int,
        mon: Optional[Pokemon],
        battle: AbstractBattle,
    ) -> None:
        """Fill tera features for my active pokemon (offsets 40-59 within my active block).
        Layout: is_terastallized(1) + can_tera(1) + tera_type_one_hot(18) = 20 features."""
        if mon is None:
            return
        vector[start + 40] = 1.0 if getattr(mon, "is_terastallized", False) else 0.0
        vector[start + 41] = 1.0 if getattr(battle, "can_tera", False) else 0.0
        tera_type = getattr(mon, "tera_type", None)
        if tera_type is not None:
            for offset, poke_type in enumerate(TYPE_ORDER):
                vector[start + 42 + offset] = 1.0 if poke_type == tera_type else 0.0

    def _fill_opp_active_tera(
        self,
        vector: np.ndarray,
        start: int,
        mon: Optional[Pokemon],
    ) -> None:
        """Fill tera features for opponent active pokemon (offset 40 within opp active block).
        Layout: is_terastallized(1) = 1 feature.
        Opponent tera type is already reflected in the type one-hot via _effective_types."""
        if mon is None:
            return
        vector[start + 40] = 1.0 if getattr(mon, "is_terastallized", False) else 0.0

    def _fill_type_one_hot(
        self,
        vector: np.ndarray,
        start: int,
        types: Iterable[PokemonType],
    ) -> None:
        type_set = set(types)
        for offset, poke_type in enumerate(TYPE_ORDER):
            vector[start + offset] = 1.0 if poke_type in type_set else 0.0

    def _verify_scalar(
        self,
        issues: list[str],
        label: str,
        observed: float,
        expected: float,
        atol: float = 1e-6,
    ) -> None:
        if abs(float(observed) - float(expected)) > atol:
            issues.append(
                f"{label} mismatch: observed={float(observed):.6f} expected={float(expected):.6f}"
            )

    def _verify_side_conditions(
        self,
        issues: list[str],
        vector: np.ndarray,
        start: int,
        side_conditions: Dict[SideCondition, int],
        prefix: str,
    ) -> None:
        expected = (
            1.0 if SideCondition.STEALTH_ROCK in side_conditions else 0.0,
            _clamp01(side_conditions.get(SideCondition.SPIKES, 0) / 3.0),
            _clamp01(side_conditions.get(SideCondition.TOXIC_SPIKES, 0) / 2.0),
            1.0 if SideCondition.STICKY_WEB in side_conditions else 0.0,
            1.0 if SideCondition.REFLECT in side_conditions else 0.0,
            1.0 if SideCondition.LIGHT_SCREEN in side_conditions else 0.0,
            1.0 if SideCondition.AURORA_VEIL in side_conditions else 0.0,
        )
        for offset, value in enumerate(expected):
            self._verify_scalar(issues, f"{prefix}[{offset}]", vector[start + offset], value)

    def _verify_active_block(
        self,
        issues: list[str],
        vector: np.ndarray,
        start: int,
        block_size: int,
        mon: Optional[Pokemon],
        prefix: str,
    ) -> None:
        if mon is None:
            expected = np.zeros(block_size, dtype=np.float32)
            observed = vector[start : start + block_size]
            if not np.allclose(observed, expected):
                issues.append(f"{prefix} expected zero block when pokemon is None")
            return

        self._verify_scalar(issues, f"{prefix}.hp", vector[start], _safe_hp_fraction(mon))
        type_set = set(_effective_types(mon))
        for offset, poke_type in enumerate(TYPE_ORDER):
            self._verify_scalar(
                issues,
                f"{prefix}.type.{poke_type.name}",
                vector[start + 1 + offset],
                1.0 if poke_type in type_set else 0.0,
            )
        for offset, stat in enumerate(BOOST_ORDER):
            self._verify_scalar(
                issues,
                f"{prefix}.boost.{stat}",
                vector[start + 19 + offset],
                _clamp(float(mon.boosts.get(stat, 0)) / 6.0, -1.0, 1.0),
            )
        for offset, status in enumerate(STATUS_ORDER):
            self._verify_scalar(
                issues,
                f"{prefix}.status.{status.name}",
                vector[start + 26 + offset],
                1.0 if mon.status == status else 0.0,
            )
        self._verify_scalar(
            issues,
            f"{prefix}.status.poison",
            vector[start + 30],
            _poison_severity(mon.status),
        )
        for offset, effect in enumerate(VOLATILE_ORDER):
            self._verify_scalar(
                issues,
                f"{prefix}.volatile.{effect.name}",
                vector[start + 31 + offset],
                1.0 if effect in mon.effects else 0.0,
            )

    def _verify_opponent_bench_leaks(
        self,
        issues: list[str],
        vector: np.ndarray,
        battle: AbstractBattle,
    ) -> None:
        bench = [mon for mon in battle.opponent_team.values() if not mon.active][:5]
        for slot in range(5):
            start = OPP_BENCH_START + slot * OPP_BENCH_SLOT_SIZE
            slot_vec = vector[start : start + OPP_BENCH_SLOT_SIZE]
            mon = bench[slot] if slot < len(bench) else None
            if mon is None or not mon.revealed:
                if not np.allclose(slot_vec, 0.0):
                    issues.append(f"opp_bench[{slot}] leaked hidden information")

    def _verify_opponent_threat_ranges(
        self,
        issues: list[str],
        vector: np.ndarray,
        battle: AbstractBattle,
    ) -> None:
        opponent = battle.opponent_active_pokemon
        if opponent is None:
            if not np.allclose(vector[OPP_THREAT_START:ON_RECHARGE_INDEX], 0.0):
                issues.append("opponent threat block should be zero with no opposing active pokemon")
            return
        for move_idx in range(OPP_THREAT_ROWS):
            start = OPP_THREAT_START + move_idx * OPP_THREAT_ROW_SIZE
            self._verify_unit_interval(issues, f"opp_threat[{move_idx}].move_prob", vector[start])
            self._verify_unit_interval(issues, f"opp_threat[{move_idx}].revealed", vector[start + 1])
            for target_idx in range(6):
                self._verify_unit_interval(
                    issues,
                    f"opp_threat[{move_idx}].ev_vs_target[{target_idx}]",
                    vector[start + 2 + target_idx],
                )
        for target_idx in range(6):
            self._verify_unit_interval(
                issues,
                f"opp_threat.ohko_risk[{target_idx}]",
                vector[OPP_THREAT_OHKO_START + target_idx],
            )
        self._verify_unit_interval(
            issues,
            "opp_threat.top_role_mass",
            vector[OPP_THREAT_CONFIDENCE_START],
        )
        self._verify_unit_interval(
            issues,
            "opp_threat.role_entropy_norm",
            vector[OPP_THREAT_CONFIDENCE_START + 1],
        )

    def _verify_unit_interval(self, issues: list[str], name: str, value: float) -> None:
        if not np.isfinite(value) or value < -1e-6 or value > 1.0 + 1e-6:
            issues.append(f"{name} should be in [0, 1], got {value}")

    def _item_capabilities(
        self,
        mon: Pokemon,
        moves: Sequence[Move],
        battle: AbstractBattle,
    ) -> Tuple[float, float, float, float, float]:
        item = mon.item or ""
        is_choice_locked = 1.0 if (
            Effect.LOCKED_MOVE in mon.effects
            or (item in {"choiceband", "choicespecs", "choicescarf"} and mon.last_move is not None)
        ) else 0.0
        has_recovery = 1.0 if (
            item in RECOVERY_ITEMS
            or any(self._move_heal_amount(move, battle) > 0.0 or move.drain > 0.0 for move in moves)
        ) else 0.0
        has_dmg_boost = 1.0 if (item in DAMAGE_BOOST_ITEMS or get_item_boost_type(item) is not None) else 0.0
        has_spe_boost = 1.0 if (
            item in SPEED_BOOST_ITEMS
            or Effect.QUARKDRIVESPE in mon.effects
            or Effect.PROTOSYNTHESISSPE in mon.effects
        ) else 0.0
        is_boots = 1.0 if item == "heavydutyboots" else 0.0
        return is_choice_locked, has_recovery, has_dmg_boost, has_spe_boost, is_boots

    def _fill_available_move_blocks(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        attacker = battle.active_pokemon
        defender = battle.opponent_active_pokemon
        for slot, move in enumerate(tuple(battle.available_moves)[:4]):
            start = MY_MOVES_START + slot * MOVE_BLOCK_SIZE
            self._fill_move_block(
                vector,
                start,
                move,
                attacker,
                defender,
                battle,
                attacker_role=battle.player_role,
                defender_role=battle.opponent_role,
            )

    def _fill_move_block(
        self,
        vector: np.ndarray,
        start: int,
        move: Move,
        attacker: Optional[Pokemon],
        defender: Optional[Pokemon],
        battle: AbstractBattle,
        attacker_role: Optional[str],
        defender_role: Optional[str],
    ) -> None:
        if attacker is None or defender is None:
            return

        min_pct, max_pct = self._damage_range_percent(
            battle,
            attacker,
            defender,
            move,
            attacker_role,
            defender_role,
        )
        accuracy = _clamp01(float(move.accuracy))
        expected_value = _clamp01(((min_pct + max_pct) / 2.0) * accuracy)
        remaining_hp = _clamp01(_safe_hp_fraction(defender))
        flags = self._move_flags(move)
        category = self._move_category(move)
        priority = self._move_priority(move)

        vector[start] = min_pct
        vector[start + 1] = max_pct
        vector[start + 2] = accuracy
        vector[start + 3] = expected_value
        vector[start + 4] = 1.0 if max_pct >= remaining_hp and remaining_hp > 0.0 else 0.0
        vector[start + 5] = self._is_stab(attacker, move, battle)
        vector[start + 6] = 1.0 if category == MoveCategory.PHYSICAL else 0.0
        vector[start + 7] = 1.0 if category == MoveCategory.SPECIAL else 0.0
        vector[start + 8] = 1.0 if "contact" in flags else 0.0
        vector[start + 9] = 1.0 if "sound" in flags else 0.0
        vector[start + 10] = 1.0 if priority > 0 else 0.0
        vector[start + 11] = 1.0 if self._is_pivot(move) else 0.0
        vector[start + 12] = _clamp01(self._effective_move_heal_amount(move, attacker, battle))
        vector[start + 13] = _clamp01(self._move_drain(move))
        vector[start + 14] = self._move_self_delta(move, "atk")
        vector[start + 15] = self._move_self_delta(move, "spa")
        vector[start + 16] = self._move_self_delta(move, "spe")
        vector[start + 17] = self._estimated_recoil_fraction(
            attacker,
            defender,
            move,
            expected_value,
        )
        vector[start + 18] = self._move_causes_recharge(move)
        vector[start + 19] = self._move_effect_chance(move, statuses=(Status.BRN,))
        vector[start + 20] = self._move_effect_chance(move, statuses=(Status.PAR,))
        vector[start + 21] = self._move_effect_chance(move, statuses=(Status.PSN, Status.TOX))
        vector[start + 22] = self._move_effect_chance(move, statuses=(Status.FRZ,))
        vector[start + 23] = self._move_effect_chance(move, statuses=(Status.SLP,))
        vector[start + 24] = self._move_effect_chance(move, volatile_effect=Effect.CONFUSION)

    def _fill_my_bench(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        bench = [mon for mon in battle.team.values() if not mon.active][:5]
        for slot, mon in enumerate(bench):
            start = MY_BENCH_START + slot * MY_BENCH_SLOT_SIZE
            vector[start] = 0.0 if mon.fainted else 1.0
            vector[start + 1] = _safe_hp_fraction(mon)
            self._fill_type_one_hot(vector, start + 2, _effective_types(mon))
            for move_idx, move in enumerate(tuple(mon.moves.values())[:4]):
                move_start = start + 20 + move_idx * BENCH_MOVE_FLAG_SIZE
                self._fill_bench_move_flags(vector, move_start, move, battle)
            # Ability flag: Intimidate
            ability = getattr(mon, "ability", None)
            vector[start + 52] = 1.0 if ability and ability.lower().replace(" ", "") == "intimidate" else 0.0

    def _fill_bench_move_flags(
        self,
        vector: np.ndarray,
        start: int,
        move: Move,
        battle: AbstractBattle,
    ) -> None:
        flags = self._move_flags(move)
        category = self._move_category(move)
        priority = self._move_priority(move)
        vector[start] = 1.0 if category == MoveCategory.PHYSICAL else 0.0
        vector[start + 1] = 1.0 if category == MoveCategory.SPECIAL else 0.0
        vector[start + 2] = 1.0 if "contact" in flags else 0.0
        vector[start + 3] = 1.0 if "sound" in flags else 0.0
        vector[start + 4] = 1.0 if priority > 0 else 0.0
        vector[start + 5] = 1.0 if self._is_pivot(move) else 0.0
        vector[start + 6] = _clamp01(self._effective_move_heal_amount(move, None, battle))
        vector[start + 7] = _clamp01(self._move_drain(move))

    def _fill_opponent_bench(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        bench = [mon for mon in battle.opponent_team.values() if not mon.active][:5]
        for slot, mon in enumerate(bench):
            if not mon.revealed:
                continue
            start = OPP_BENCH_START + slot * OPP_BENCH_SLOT_SIZE
            vector[start] = 1.0
            vector[start + 1] = _safe_hp_fraction(mon)
            self._fill_type_one_hot(vector, start + 2, _effective_types(mon))

    def _fill_targeting_matrix(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        attacker = battle.active_pokemon
        if attacker is None:
            return

        opponent_bench = [mon for mon in battle.opponent_team.values() if not mon.active][:5]
        my_moves = tuple(battle.available_moves)[:4]
        for move_idx, move in enumerate(my_moves):
            for target_idx, target in enumerate(opponent_bench):
                if not target.revealed or target.fainted:
                    continue
                min_pct, max_pct = self._damage_range_percent(
                    battle,
                    attacker,
                    target,
                    move,
                    battle.player_role,
                    battle.opponent_role,
                )
                ev = _clamp01(((min_pct + max_pct) / 2.0) * _clamp01(float(move.accuracy)))
                vector[TARGETING_START + move_idx * 5 + target_idx] = ev

    def _fill_opponent_threat_features(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        my_team_order = self._my_team_order(battle)
        memory = self._my_team_revealed_memory.get(_battle_tag(battle), set())
        for idx, mon in enumerate(my_team_order):
            if mon is None:
                continue
            vector[MY_TEAM_REVEALED_START + idx] = 1.0 if _mon_key(mon) in memory else 0.0

        opponent = battle.opponent_active_pokemon
        if opponent is None:
            return

        posterior = self._opponent_role_posterior(opponent)
        threat_entries = self._select_opponent_threat_entries(opponent, posterior)
        for move_idx, entry in enumerate(threat_entries):
            row_start = OPP_THREAT_START + move_idx * OPP_THREAT_ROW_SIZE
            vector[row_start] = entry.move_prob
            vector[row_start + 1] = entry.revealed_flag
            for target_idx, target in enumerate(my_team_order):
                if target is None or target.fainted:
                    continue
                vector[row_start + 2 + target_idx] = self._move_expected_value(
                    battle,
                    opponent,
                    target,
                    entry.move,
                    battle.opponent_role,
                    battle.player_role,
                )
        for target_idx, target in enumerate(my_team_order):
            if target is None or target.fainted:
                continue
            vector[OPP_THREAT_OHKO_START + target_idx] = self._estimate_ohko_risk(
                battle,
                opponent,
                target,
                posterior,
            )
        top_role_mass, role_entropy_norm = self._role_posterior_summary(posterior)
        vector[OPP_THREAT_CONFIDENCE_START] = top_role_mass
        vector[OPP_THREAT_CONFIDENCE_START + 1] = role_entropy_norm

    def _opponent_role_posterior(self, opponent: Pokemon) -> Dict[str, float]:
        revealed_moves = list(opponent.moves.keys())
        posterior = self._meta.filter_roles(opponent.species, revealed_moves, opponent.item)
        if posterior:
            return posterior
        posterior = self._meta.filter_roles(opponent.species, revealed_moves, None)
        if posterior:
            return posterior
        return self._meta.filter_roles(opponent.species, [], None)

    def _select_opponent_threat_entries(
        self,
        opponent: Pokemon,
        posterior: Dict[str, float],
    ) -> List[OpponentThreatEntry]:
        entries: Dict[str, OpponentThreatEntry] = {}
        for move in tuple(opponent.moves.values())[:OPP_THREAT_ROWS]:
            entries[move.id] = OpponentThreatEntry(
                move=move,
                move_prob=1.0,
                revealed_flag=1.0,
            )
        for move_id, move_prob in self._meta.get_move_marginals(opponent.species, posterior).items():
            if move_id in entries:
                continue
            inferred_move = self._get_inferred_move(move_id)
            if inferred_move is None:
                continue
            entries[move_id] = OpponentThreatEntry(
                move=inferred_move,
                move_prob=_clamp01(move_prob),
                revealed_flag=0.0,
            )
        return sorted(
            entries.values(),
            key=lambda entry: (-entry.move_prob, -entry.revealed_flag, entry.move.id),
        )[:OPP_THREAT_ROWS]

    def _get_inferred_move(self, move_id: str) -> Optional[Move]:
        cached_move = self._inferred_move_cache.get(move_id, None)
        if move_id in self._inferred_move_cache:
            return cached_move
        try:
            inferred_move = Move(move_id, gen=9)
        except Exception:
            inferred_move = None
        self._inferred_move_cache[move_id] = inferred_move
        return inferred_move

    def _move_expected_value(
        self,
        battle: AbstractBattle,
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        attacker_role: Optional[str],
        defender_role: Optional[str],
    ) -> float:
        min_pct, max_pct = self._damage_range_percent(
            battle,
            attacker,
            defender,
            move,
            attacker_role,
            defender_role,
        )
        return _clamp01(((min_pct + max_pct) / 2.0) * _clamp01(float(move.accuracy)))

    def _estimate_ohko_risk(
        self,
        battle: AbstractBattle,
        opponent: Pokemon,
        target: Pokemon,
        posterior: Dict[str, float],
    ) -> float:
        target_hp = _safe_hp_fraction(target)
        if target_hp <= 0.0:
            return 0.0
        revealed_moves = {move.id: move for move in tuple(opponent.moves.values())[:OPP_THREAT_ROWS]}
        if not posterior:
            return 1.0 if any(
                self._move_has_ohko_roll(
                    battle,
                    opponent,
                    target,
                    move,
                    battle.opponent_role,
                    battle.player_role,
                    target_hp,
                )
                for move in revealed_moves.values()
            ) else 0.0

        total_risk = 0.0
        for role_name, role_weight in posterior.items():
            role_risk = 0.0
            role_moves = self._meta.get_role_move_distribution(opponent.species, role_name)
            for move_id, move_prob in role_moves.items():
                move = revealed_moves.get(move_id) or self._get_inferred_move(move_id)
                if move is None:
                    continue
                if self._move_has_ohko_roll(
                    battle,
                    opponent,
                    target,
                    move,
                    battle.opponent_role,
                    battle.player_role,
                    target_hp,
                ):
                    adjusted_prob = 1.0 if move_id in revealed_moves else float(move_prob)
                    role_risk += adjusted_prob
            total_risk += float(role_weight) * _clamp01(role_risk)
        return _clamp01(total_risk)

    def _move_has_ohko_roll(
        self,
        battle: AbstractBattle,
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        attacker_role: Optional[str],
        defender_role: Optional[str],
        target_hp: Optional[float] = None,
    ) -> bool:
        _, max_pct = self._damage_range_percent(
            battle,
            attacker,
            defender,
            move,
            attacker_role,
            defender_role,
        )
        remaining_hp = _safe_hp_fraction(defender) if target_hp is None else target_hp
        return remaining_hp > 0.0 and max_pct >= remaining_hp

    def _role_posterior_summary(self, posterior: Dict[str, float]) -> Tuple[float, float]:
        if not posterior:
            return 0.0, 1.0
        probabilities = np.asarray(list(posterior.values()), dtype=np.float32)
        top_role_mass = float(np.max(probabilities))
        if probabilities.size <= 1:
            return top_role_mass, 0.0
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
        max_entropy = float(np.log(float(probabilities.size)))
        if max_entropy <= 0.0:
            return top_role_mass, 0.0
        return top_role_mass, _clamp01(entropy / max_entropy)

    def _my_team_order(self, battle: AbstractBattle) -> Tuple[Optional[Pokemon], ...]:
        bench = [mon for mon in battle.team.values() if not mon.active][:5]
        ordered: list[Optional[Pokemon]] = [battle.active_pokemon]
        ordered.extend(bench)
        while len(ordered) < 6:
            ordered.append(None)
        return tuple(ordered[:6])

    def _speed_advantage(self, battle: AbstractBattle) -> float:
        my_speed = self._effective_speed(battle.active_pokemon, battle.side_conditions)
        opp_posterior = (
            self._opponent_role_posterior(battle.opponent_active_pokemon)
            if battle.opponent_active_pokemon is not None
            else None
        )
        opp_speed = self._effective_speed(
            battle.opponent_active_pokemon,
            battle.opponent_side_conditions,
            role_posterior=opp_posterior,
        )
        if my_speed is None or opp_speed is None:
            return 0.0
        if my_speed > opp_speed:
            return 1.0
        if my_speed == opp_speed:
            return 0.5
        return 0.0

    def _effective_speed(
        self,
        mon: Optional[Pokemon],
        side_conditions: Dict[SideCondition, int],
        role_posterior: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        if mon is None:
            return None
        base_speed = self._speed_stat_estimate(mon, role_posterior)
        if base_speed is None:
            return None

        speed = float(base_speed) * _stat_stage_multiplier(mon.boosts.get("spe", 0))

        if mon.status == Status.PAR:
            speed *= 0.5
        if SideCondition.TAILWIND in side_conditions:
            speed *= 2.0
        speed *= self._speed_item_multiplier(mon, role_posterior)
        if Effect.QUARKDRIVESPE in mon.effects or Effect.PROTOSYNTHESISSPE in mon.effects:
            speed *= 1.5
        return speed

    def _speed_stat_estimate(
        self,
        mon: Pokemon,
        role_posterior: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        stats = getattr(mon, "stats", {}) or {}
        live_speed = stats.get("spe")
        if isinstance(live_speed, (int, float)):
            return float(live_speed)
        if not role_posterior:
            return None

        species = getattr(mon, "species", None)
        if not species:
            return None
        base_stats = getattr(mon, "base_stats", None) or stats
        if not isinstance(base_stats, dict):
            return None

        weighted_speed = 0.0
        total_weight = 0.0
        for role_name, role_weight in role_posterior.items():
            if role_weight <= 0.0:
                continue
            role_stats = self._meta.get_role_stats(species, role_name, base_stats)
            role_speed = role_stats.get("spe")
            if not isinstance(role_speed, (int, float)):
                continue
            weighted_speed += float(role_weight) * float(role_speed)
            total_weight += float(role_weight)
        if total_weight <= 0.0:
            return None
        return weighted_speed / total_weight

    def _speed_item_multiplier(
        self,
        mon: Pokemon,
        role_posterior: Optional[Dict[str, float]] = None,
    ) -> float:
        item = (getattr(mon, "item", "") or "").strip()
        if item in SPEED_BOOST_ITEMS:
            return 1.5
        if item in SPEED_DROP_ITEMS:
            return 0.5
        if item:
            return 1.0
        if not role_posterior:
            return 1.0

        species = getattr(mon, "species", None)
        if not species:
            return 1.0
        item_marginals = self._meta.get_item_marginals(species, role_posterior)
        if not item_marginals:
            return 1.0

        expected_multiplier = 0.0
        total_prob = 0.0
        for item_id, item_prob in item_marginals.items():
            prob = _clamp01(float(item_prob))
            if prob <= 0.0:
                continue
            if item_id in SPEED_BOOST_ITEMS:
                multiplier = 1.5
            elif item_id in SPEED_DROP_ITEMS:
                multiplier = 0.5
            else:
                multiplier = 1.0
            expected_multiplier += prob * multiplier
            total_prob += prob
        if total_prob <= 0.0:
            return 1.0
        return expected_multiplier / total_prob

    def _on_recharge(self, mon: Optional[Pokemon]) -> float:
        if mon is None:
            return 0.0
        if Effect.MUST_RECHARGE in mon.effects:
            return 1.0
        return 1.0 if bool(getattr(mon, "must_recharge", False)) else 0.0

    def _damage_range_percent(
        self,
        battle: AbstractBattle,
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        attacker_role: Optional[str],
        defender_role: Optional[str],
    ) -> Tuple[float, float]:
        min_damage, max_damage = self._estimate_damage_range(
            battle,
            attacker,
            defender,
            move,
            attacker_role,
            defender_role,
        )
        defender_hp = self._defender_hp_scale(defender)
        return _clamp01(min_damage / defender_hp), _clamp01(max_damage / defender_hp)

    def _estimate_damage_range(
        self,
        battle: AbstractBattle,
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        attacker_role: Optional[str],
        defender_role: Optional[str],
    ) -> Tuple[float, float]:
        # Bayesian filtering based on revealed moves/items
        # 1. Identify species
        att_species = attacker.species if attacker else "Pikachu"
        def_species = defender.species if defender else "Pikachu"
        
        # 2. Get possible roles and weights
        # For opponent, filter roles. For me, assume 100% current stats.
        att_is_opp = (attacker == battle.opponent_active_pokemon)
        def_is_opp = (defender == battle.opponent_active_pokemon)
        
        if att_is_opp:
            att_roles = self._meta.filter_roles(att_species, list(attacker.moves.keys()), attacker.item)
        else:
            att_roles = {"Current": 1.0}
            
        if def_is_opp:
            def_roles = self._meta.filter_roles(def_species, list(defender.moves.keys()), defender.item)
        else:
            def_roles = {"Current": 1.0}

        # 3. Sum weighted damage across all role pairings
        weighted_min = 0.0
        weighted_max = 0.0
        
        for a_role, a_weight in att_roles.items():
            for d_role, d_weight in def_roles.items():
                pair_weight = a_weight * d_weight
                if pair_weight < 0.01: continue
                
                # Get stats for this pairing
                if a_role == "Current":
                    att_stats = attacker.stats
                else:
                    att_stats = self._meta.get_role_stats(att_species, a_role, attacker.base_stats)
                
                if d_role == "Current":
                    def_stats = defender.stats
                else:
                    def_stats = self._meta.get_role_stats(def_species, d_role, defender.base_stats)
                
                # Calculate damage with these stats
                # Using a manual formula if mon.stats is missing to avoid poke-env identifier issues
                d_min, d_max = self._manual_damage_calc(
                    attacker, defender, move, battle, att_stats, def_stats
                )
                weighted_min += d_min * pair_weight
                weighted_max += d_max * pair_weight
        
        return weighted_min, weighted_max

    def _manual_damage_calc(
        self,
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        battle: AbstractBattle,
        att_stats: Dict[str, Optional[int]],
        def_stats: Dict[str, Optional[int]],
    ) -> Tuple[float, float]:
        # Robust Damage Formula (Generation 9)
        level = attacker.level or 80
        fixed_damage = self._fixed_damage_amount(attacker, defender, move, battle)
        if fixed_damage is not None:
            return fixed_damage, fixed_damage

        bp = float(getattr(move, "base_power", 0))
        if bp == 0: bp = 5.0 # Min signal for status moves to differentiate types
        
        category = move.category
        if category == MoveCategory.PHYSICAL:
            a = float(att_stats.get("atk") or 100)
            d = float(def_stats.get("def") or 100)
        else:
            a = float(att_stats.get("spa") or 100)
            d = float(def_stats.get("spd") or 100)
            
        # Boosts
        a_boost = _stat_stage_multiplier(attacker.boosts.get("atk" if category == MoveCategory.PHYSICAL else "spa", 0))
        d_boost = _stat_stage_multiplier(defender.boosts.get("def" if category == MoveCategory.PHYSICAL else "spd", 0))

        a *= a_boost
        d *= d_boost
        
        # Base damage
        base_damage = (((2 * level / 5 + 2) * bp * a / d) / 50) + 2
        
        # Modifiers
        move_type = self._resolve_move_type(attacker, move, battle)
        type_mult = _defender_type_mult(move_type, defender, self._gen_data.type_chart)
        stab = _stab_multiplier(attacker, move_type)
        
        # Weather/Burn
        weather_mult = 1.0
        if move_type == PokemonType.FIRE:
            if Weather.SUNNYDAY in battle.weather: weather_mult = 1.5
            elif Weather.RAINDANCE in battle.weather: weather_mult = 0.5
        elif move_type == PokemonType.WATER:
            if Weather.RAINDANCE in battle.weather: weather_mult = 1.5
            elif Weather.SUNNYDAY in battle.weather: weather_mult = 0.5
            
        burn_mult = 0.5 if (attacker.status == Status.BRN and category == MoveCategory.PHYSICAL) else 1.0
        
        total_mult = type_mult * stab * weather_mult * burn_mult
        
        final_min = base_damage * total_mult * 0.85
        final_max = base_damage * total_mult * 1.0
        
        return final_min, final_max

    def _fixed_damage_amount(
        self,
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        battle: AbstractBattle,
    ) -> Optional[float]:
        raw_damage = getattr(move, "damage", 0)
        if raw_damage in (None, 0):
            return None

        move_type = self._resolve_move_type(attacker, move, battle)
        type_mult = _defender_type_mult(move_type, defender, self._gen_data.type_chart)
        if type_mult == 0.0:
            return 0.0

        if isinstance(raw_damage, (int, float)):
            return float(raw_damage)
        if raw_damage == "level":
            return float(getattr(attacker, "level", 80) or 80)
        return None

    def _stats_defined(self, mon: Pokemon) -> bool:
        return all(isinstance(value, (int, float)) for value in mon.stats.values())

    def _defender_hp_scale(self, mon: Pokemon) -> float:
        # Try meta-stats first for opponents
        species = mon.species if mon else None
        spec = self._meta.get_species_data(species) if species else None
        if spec:
            # Use level and base HP to estimate max HP
            level = spec.get("level", 80)
            base_hp = mon.base_stats.get("hp", 100)
            # Standard Randbats HP (usually 84 EVs)
            return float(self._meta.calculate_stat(base_hp, level, ev=84, is_hp=True))
            
        hp_stat = mon.stats.get("hp")
        if isinstance(hp_stat, (int, float)) and hp_stat > 0:
            return float(hp_stat)
        max_hp = getattr(mon, "max_hp", 0)
        if isinstance(max_hp, (int, float)) and max_hp > 0:
            return float(max_hp)
        return 100.0

    def _is_stab(self, attacker: Pokemon, move: Move, battle: AbstractBattle) -> float:
        move_type = self._resolve_move_type(attacker, move, battle)
        return 1.0 if _stab_multiplier(attacker, move_type) > 1.0 else 0.0

    def _resolve_move_type(
        self,
        attacker: Pokemon,
        move: Move,
        battle: AbstractBattle,
    ) -> PokemonType:
        if move.id == "weatherball":
            if Weather.SUNNYDAY in battle.weather and attacker.item != "utilityumbrella":
                return PokemonType.FIRE
            if Weather.RAINDANCE in battle.weather and attacker.item != "utilityumbrella":
                return PokemonType.WATER
            if Weather.SANDSTORM in battle.weather:
                return PokemonType.ROCK
            if Weather.SNOW in battle.weather or Weather.HAIL in battle.weather:
                return PokemonType.ICE
        elif move.id == "judgment" and attacker.item and attacker.item.endswith("plate"):
            return get_item_boost_type(attacker.item) or move.type
        elif move.id in {"naturepower", "terrainpulse"}:
            if Field.ELECTRIC_TERRAIN in battle.fields:
                return PokemonType.ELECTRIC
            if Field.GRASSY_TERRAIN in battle.fields:
                return PokemonType.GRASS
            if Field.MISTY_TERRAIN in battle.fields:
                return PokemonType.FAIRY
            if Field.PSYCHIC_TERRAIN in battle.fields:
                return PokemonType.PSYCHIC
        elif move.id == "revelationdance":
            return attacker.type_1
        elif move.id == "terablast" and attacker.is_terastallized:
            return attacker.type_1
        return move.type

    def _is_pivot(self, move: Move) -> bool:
        try:
            self_switch = bool(move.self_switch)
        except Exception:
            entry = getattr(move, "entry", {}) or {}
            self_switch = bool(entry.get("selfSwitch", False))
            self._record_move_fallback(
                accessor="self_switch",
                move=move,
                battle=None,
                raw_value=entry.get("selfSwitch"),
            )
        return self_switch or move.id in PIVOT_MOVES

    def _move_flags(self, move: Move) -> set[str]:
        try:
            flags = move.flags
        except Exception:
            entry = getattr(move, "entry", {}) or {}
            raw_flags = entry.get("flags", [])
            self._record_move_fallback(
                accessor="flags",
                move=move,
                battle=None,
                raw_value=raw_flags,
            )
            if isinstance(raw_flags, dict):
                return set(raw_flags.keys())
            return set(raw_flags)

        return set(flags)

    def _move_category(self, move: Move) -> MoveCategory | None:
        try:
            return move.category
        except Exception:
            entry = getattr(move, "entry", {}) or {}
            raw_category = entry.get("category")
            self._record_move_fallback(
                accessor="category",
                move=move,
                battle=None,
                raw_value=raw_category,
            )
            if not raw_category:
                return None
            try:
                return MoveCategory[str(raw_category).upper()]
            except KeyError:
                return None

    def _move_priority(self, move: Move) -> int:
        try:
            return int(move.priority)
        except Exception:
            entry = getattr(move, "entry", {}) or {}
            raw_priority = entry.get("priority", 0)
            self._record_move_fallback(
                accessor="priority",
                move=move,
                battle=None,
                raw_value=raw_priority,
            )
            return int(raw_priority)

    def _move_drain(self, move: Move) -> float:
        try:
            return float(move.drain)
        except Exception:
            entry = getattr(move, "entry", {}) or {}
            raw_drain = entry.get("drain")
            self._record_move_fallback(
                accessor="drain",
                move=move,
                battle=None,
                raw_value=raw_drain,
            )
            if isinstance(raw_drain, (list, tuple)) and len(raw_drain) == 2 and raw_drain[1]:
                return float(raw_drain[0]) / float(raw_drain[1])
            return 0.0

    def _move_self_delta(self, move: Move, stat: str) -> float:
        try:
            boosts = getattr(move, "self_boost", None)
        except Exception:
            boosts = None
        if not isinstance(boosts, dict):
            entry = getattr(move, "entry", {}) or {}
            if isinstance(entry.get("selfBoost"), dict):
                boosts = entry["selfBoost"].get("boosts")
            elif isinstance(entry.get("self"), dict):
                boosts = entry["self"].get("boosts")
        if not isinstance(boosts, dict):
            return 0.0
        return _clamp(float(boosts.get(stat, 0)) / 2.0, -1.0, 1.0)

    def _move_recoil_ratio(self, move: Move) -> float:
        try:
            return _clamp01(float(move.recoil))
        except Exception:
            entry = getattr(move, "entry", {}) or {}
            raw_recoil = entry.get("recoil")
            if isinstance(raw_recoil, (list, tuple)) and len(raw_recoil) == 2 and raw_recoil[1]:
                return _clamp01(float(raw_recoil[0]) / float(raw_recoil[1]))
            if entry.get("struggleRecoil"):
                return 0.25
            return 0.0

    def _estimated_recoil_fraction(
        self,
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        expected_value: float,
    ) -> float:
        recoil_ratio = self._move_recoil_ratio(move)
        if recoil_ratio <= 0.0:
            return 0.0
        defender_hp = self._defender_hp_scale(defender)
        attacker_hp = self._defender_hp_scale(attacker)
        if attacker_hp <= 0.0:
            return 0.0
        expected_damage = expected_value * defender_hp
        return _clamp01((expected_damage * recoil_ratio) / attacker_hp)

    def _move_causes_recharge(self, move: Move) -> float:
        entry = getattr(move, "entry", {}) or {}
        if entry.get("recharge"):
            return 1.0
        return 1.0 if bool(getattr(move, "recharge", False)) else 0.0

    def _move_effect_chance(
        self,
        move: Move,
        *,
        statuses: Tuple[Status, ...] = (),
        volatile_effect: Optional[Effect] = None,
    ) -> float:
        best = 0.0

        direct_status = getattr(move, "status", None)
        if direct_status in statuses:
            best = 1.0

        entry = getattr(move, "entry", {}) or {}
        if volatile_effect is not None and self._entry_effect(entry.get("volatileStatus")) == volatile_effect:
            best = 1.0

        secondary_effects = getattr(move, "secondary", None)
        if not isinstance(secondary_effects, list):
            if isinstance(entry.get("secondary"), dict):
                secondary_effects = [entry["secondary"]]
            elif isinstance(entry.get("secondaries"), list):
                secondary_effects = entry["secondaries"]
            else:
                secondary_effects = []

        for secondary in secondary_effects:
            if not isinstance(secondary, dict):
                continue
            chance = _clamp01(float(secondary.get("chance", 100)) / 100.0)
            if self._entry_status(secondary.get("status")) in statuses:
                best = max(best, chance)
            if volatile_effect is not None and self._entry_effect(secondary.get("volatileStatus")) == volatile_effect:
                best = max(best, chance)
        return best

    def _entry_status(self, raw_status: Any) -> Optional[Status]:
        if isinstance(raw_status, Status):
            return raw_status
        if not raw_status:
            return None
        try:
            return Status[str(raw_status).upper()]
        except Exception:
            return None

    def _entry_effect(self, raw_effect: Any) -> Optional[Effect]:
        if isinstance(raw_effect, Effect):
            return raw_effect
        if not raw_effect:
            return None
        try:
            return Effect.from_data(str(raw_effect))
        except Exception:
            return None

    def _move_heal_amount(self, move: Move, battle: AbstractBattle) -> float:
        if move.id in WEATHER_HEAL_MOVES:
            if Weather.SUNNYDAY in battle.weather:
                return 2.0 / 3.0
            if any(weather in battle.weather for weather in (Weather.RAINDANCE, Weather.SANDSTORM, Weather.SNOW, Weather.HAIL)):
                return 0.25
            return 0.5
        try:
            return float(move.heal)
        except Exception:
            entry = getattr(move, "entry", {}) or {}
            raw_heal = entry.get("heal")
            self._record_move_fallback(
                accessor="heal",
                move=move,
                battle=battle,
                raw_value=raw_heal,
            )
            if isinstance(raw_heal, (list, tuple)) and len(raw_heal) == 2 and raw_heal[1]:
                return float(raw_heal[0]) / float(raw_heal[1])
            return 0.0

    def _effective_move_heal_amount(
        self,
        move: Move,
        user: Optional[Pokemon],
        battle: AbstractBattle,
    ) -> float:
        raw_heal = self._move_heal_amount(move, battle)
        if user is None:
            return raw_heal
        missing_hp = 1.0 - _clamp01(_safe_hp_fraction(user))
        return min(raw_heal, missing_hp)

    def _record_move_fallback(
        self,
        accessor: str,
        move: Move,
        battle: Optional[AbstractBattle],
        raw_value: Any,
    ) -> None:
        move_id = getattr(move, "id", "<unknown>")
        raw_type = type(raw_value).__name__
        key = (accessor, move_id, raw_type)
        self._fallback_counts[key] = self._fallback_counts.get(key, 0) + 1

        if len(self._fallback_samples) >= 50:
            return

        entry = getattr(move, "entry", {}) or {}
        sample = {
            "accessor": accessor,
            "move_id": move_id,
            "move_type": type(move).__name__,
            "battle_tag": _battle_tag(battle) if battle is not None else None,
            "turn": getattr(battle, "turn", None) if battle is not None else None,
            "raw_value_repr": repr(raw_value),
            "entry_keys": sorted(entry.keys()) if isinstance(entry, dict) else [],
        }
        self._fallback_samples.append(sample)

    def get_fallback_report(self) -> dict[str, Any]:
        return {
            "counts": [
                {
                    "accessor": accessor,
                    "move_id": move_id,
                    "raw_type": raw_type,
                    "count": count,
                }
                for (accessor, move_id, raw_type), count in sorted(
                    self._fallback_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "samples": list(self._fallback_samples),
        }


class BrentsRLAgent(SinglesEnv):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault(
            "account_configuration1",
            AccountConfiguration.generate("PokeRL", rand=True),
        )
        kwargs.setdefault(
            "account_configuration2",
            AccountConfiguration.generate("PokeRLOpp", rand=True),
        )
        super().__init__(*args, **kwargs)
        self.vector_builder = BrentObservationVectorBuilder()
        self._tactical_reward_context: TacticalRewardContext | None = None
        self._strategic_penalty_counts: Dict[str, int] = {}
        self._strategic_penalty_total = 0.0
        self._strategic_penalty_move_checks = 0
        self._strategic_penalty_penalized_actions = 0
        self._tactical_shaping_counts: Dict[str, int] = {}
        self._tactical_shaping_totals: Dict[str, float] = {}
        self._tactical_shaping_total = 0.0
        self._tactical_positive_total = 0.0
        self._tactical_negative_total = 0.0
        self._tactical_shaping_move_checks = 0
        self._tactical_shaping_shaped_actions = 0
        self._tactical_shaping_rewarded_actions = 0
        self._tactical_shaping_penalized_actions = 0
        self._decision_audit_counts: Dict[str, int] = {}
        self._decision_audit_flagged_actions = 0
        self._decision_audit_move_checks = 0
        self._decision_audit_samples: Dict[str, list[Dict[str, Any]]] = {}
        self._decision_count = 0
        self._switch_action_count = 0
        # Track consecutive heals per mon (species → count)
        self._consecutive_heal_count: Dict[str, int] = {}
        self._last_action_was_heal: Dict[str, bool] = {}
        # Track wasted free switches: mon species that entered via forced switch after faint
        self._entered_after_faint: set[str] = set()
        self._last_active_species: Optional[str] = None
        self._last_active_fainted: bool = False
        # Head-Hunter: track opponent alive mons for faint detection
        self._prev_opp_alive: set[str] = set()
        self.observation_spaces = {
            agent: Box(
                low=-1.0,
                high=100.0,
                shape=(VECTOR_LENGTH,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        base_config = {key: REWARD_CONFIG[key] for key in POKE_ENV_REWARD_KEYS}
        reward = self.reward_computing_helper(battle, **base_config)
        shaping = self._consume_tactical_shaping(battle)
        head_hunter = self._head_hunter_bonus(battle)
        return reward + shaping + head_hunter

    def _head_hunter_bonus(self, battle: AbstractBattle) -> float:
        """Extra reward for KOing high-threat opponent mons.
        Scales fainted_value by how many of our team the mon threatened to OHKO."""
        curr_alive = set()
        for mon in battle.opponent_team.values():
            if not mon.fainted:
                curr_alive.add(getattr(mon, "species", str(mon)))

        newly_fainted = self._prev_opp_alive - curr_alive
        self._prev_opp_alive = curr_alive

        if not newly_fainted:
            return 0.0

        # For each newly fainted opponent, check how many of our team it threatened
        bonus = 0.0
        my_team = [mon for mon in battle.team.values() if not mon.fainted]
        for fainted_species in newly_fainted:
            # Find the fainted mon object
            fainted_mon = None
            for mon in battle.opponent_team.values():
                if getattr(mon, "species", None) == fainted_species:
                    fainted_mon = mon
                    break
            if fainted_mon is None or not fainted_mon.moves:
                continue

            # Count how many of our team this mon could OHKO
            ohko_count = 0
            for my_mon in my_team:
                for opp_move in fainted_mon.moves.values():
                    try:
                        _, max_pct = self.vector_builder._damage_range_percent(
                            battle, fainted_mon, my_mon, opp_move, None, None,
                        )
                        if max_pct >= _safe_hp_fraction(my_mon) and max_pct > 0.5:
                            ohko_count += 1
                            break
                    except Exception:
                        continue

            # Scale bonus: 0 threats = no extra, 1 = +0.25, 2 = +0.5, 3+ = +0.75
            if ohko_count >= 1:
                bonus += min(ohko_count * 0.25, 0.75)

        return bonus

    def action_to_order(
        self,
        action: np.int64,
        battle: AbstractBattle,
        fake: bool = False,
        strict: bool = True,
    ) -> BattleOrder:
        order = super().action_to_order(action, battle, fake=fake, strict=strict)
        self._record_action_choice(order)
        self._remember_tactical_reward_context(battle, order)
        return order

    def _record_action_choice(self, order: BattleOrder) -> None:
        action = getattr(order, "order", None)
        if isinstance(action, (Move, Pokemon)):
            self._decision_count += 1
        if isinstance(action, Pokemon):
            self._switch_action_count += 1

    def _remember_tactical_reward_context(
        self,
        battle: AbstractBattle,
        order: BattleOrder,
    ) -> None:
        action = getattr(order, "order", None)
        if not isinstance(action, (Move, Pokemon)):
            self._tactical_reward_context = None
            return

        self._update_action_tracking(battle, action)
        matches = tuple(self._evaluate_tactical_levers(battle, order))
        self._tactical_reward_context = TacticalRewardContext(
            battle_tag=_battle_tag(battle),
            action=action,
            matches=matches,
        )
        self._audit_tactical_matches(battle, order, action, matches)

    def _audit_tactical_matches(
        self,
        battle: AbstractBattle,
        order: BattleOrder,
        action: Move | Pokemon,
        matches: Sequence[TacticalLeverMatch],
    ) -> None:
        if not isinstance(action, Move):
            return

        self._decision_audit_move_checks += 1
        audit_matches = [match for match in matches if match.record_audit]
        if not audit_matches:
            return

        self._decision_audit_flagged_actions += 1
        for match in audit_matches:
            self._record_decision_audit(match.reason, battle, order, action, match.details)

    def _consume_tactical_shaping(self, battle: AbstractBattle) -> float:
        context = self._tactical_reward_context
        if context is None or context.battle_tag != _battle_tag(battle):
            return 0.0
        self._tactical_reward_context = None
        return self._apply_tactical_shaping(context)

    def _apply_tactical_shaping(self, context: TacticalRewardContext) -> float:
        if isinstance(context.action, Move):
            self._strategic_penalty_move_checks += 1
            self._tactical_shaping_move_checks += 1

        shaping_matches = [match for match in context.matches if match.reward != 0.0]
        if not shaping_matches:
            return 0.0

        total_reward = 0.0
        negative_reward = 0.0
        for match in shaping_matches:
            self._tactical_shaping_counts[match.reason] = self._tactical_shaping_counts.get(match.reason, 0) + 1
            self._tactical_shaping_totals[match.reason] = (
                self._tactical_shaping_totals.get(match.reason, 0.0) + match.reward
            )
            total_reward += match.reward
            if match.reward < 0.0:
                negative_reward += match.reward
                self._strategic_penalty_counts[match.reason] = self._strategic_penalty_counts.get(match.reason, 0) + 1

        self._tactical_shaping_total += total_reward
        if total_reward > 0.0:
            self._tactical_positive_total += total_reward
            self._tactical_shaping_rewarded_actions += 1
        if total_reward < 0.0:
            self._tactical_negative_total += total_reward
            self._tactical_shaping_penalized_actions += 1
        self._tactical_shaping_shaped_actions += 1

        if negative_reward < 0.0:
            self._strategic_penalty_penalized_actions += 1
            self._strategic_penalty_total += negative_reward

        return total_reward

    def _evaluate_tactical_levers(
        self,
        battle: AbstractBattle,
        order: BattleOrder,
    ) -> list[TacticalLeverMatch]:
        action = getattr(order, "order", None)
        if isinstance(action, Move):
            matches = self._evaluate_move_tactical_levers(battle, action)
            if getattr(order, "terastallize", False):
                tera_match = self._make_tactical_match(
                    "good_tera",
                    "bonus_good_tera",
                    self._evaluate_good_tera(battle, action),
                )
                if tera_match is not None:
                    matches.append(tera_match)
            return matches
        if isinstance(action, Pokemon):
            return self._evaluate_switch_tactical_levers(battle, action)
        return []

    def _make_tactical_match(
        self,
        reason: str,
        reward_key: str,
        details: Optional[Dict[str, Any]],
        *,
        record_audit: bool = False,
    ) -> Optional[TacticalLeverMatch]:
        if details is None:
            return None

        reward = float(REWARD_CONFIG.get(reward_key, 0.0))
        if reward == 0.0 and not record_audit:
            return None
        return TacticalLeverMatch(reason=reason, reward=reward, details=details, record_audit=record_audit)

    def _evaluate_move_tactical_levers(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> list[TacticalLeverMatch]:
        matches: list[TacticalLeverMatch] = []
        for reason, reward_key, details, record_audit in (
            (
                "redundant_stealthrock",
                "penalty_redundant_stealthrock",
                self._evaluate_redundant_stealthrock(battle, move),
                False,
            ),
            (
                "redundant_stickyweb",
                "penalty_redundant_stickyweb",
                self._evaluate_redundant_stickyweb(battle, move),
                False,
            ),
            (
                "redundant_spikes",
                "penalty_redundant_spikes",
                self._evaluate_redundant_spikes(battle, move),
                False,
            ),
            (
                "redundant_status",
                "penalty_redundant_status",
                self._evaluate_redundant_status(battle, move),
                False,
            ),
            (
                "bad_encore",
                "penalty_bad_encore",
                self._evaluate_bad_encore(battle, move),
                False,
            ),
            (
                "ineffective_heal",
                "penalty_ineffective_heal",
                self._evaluate_ineffective_heal(battle, move),
                False,
            ),
            (
                "wasteful_heal_overflow",
                "penalty_wasteful_heal_overflow",
                self._evaluate_wasteful_heal_overflow(battle, move),
                True,
            ),
            (
                "redundant_self_drop_move",
                "penalty_redundant_self_drop_move",
                self._evaluate_redundant_self_drop_move(battle, move),
                True,
            ),
            (
                "unsafe_stay_in_with_fast_ko_switch",
                "penalty_unsafe_stay_in_with_fast_ko_switch",
                self._evaluate_unsafe_stay_in_with_fast_ko_switch(battle, move),
                True,
            ),
            (
                "good_heal_timing",
                "bonus_good_heal_timing",
                self._evaluate_good_heal_timing(battle, move),
                False,
            ),
            (
                "heal_satiation",
                "penalty_heal_satiation",
                self._evaluate_heal_satiation(battle, move),
                True,
            ),
            (
                "good_attack_selection",
                "bonus_good_attack_selection",
                self._evaluate_good_attack_selection(battle, move),
                False,
            ),
        ):
            match = self._make_tactical_match(
                reason,
                reward_key,
                details,
                record_audit=record_audit,
            )
            if match is not None:
                matches.append(match)
        return matches

    def _evaluate_switch_tactical_levers(
        self,
        battle: AbstractBattle,
        switch_target: Pokemon,
    ) -> list[TacticalLeverMatch]:
        matches: list[TacticalLeverMatch] = []
        for reason, reward_key, details, record_audit in (
            (
                "good_safe_switch",
                "bonus_good_safe_switch",
                self._evaluate_good_safe_switch(battle, switch_target),
                False,
            ),
            (
                "abandon_boosted_mon",
                "penalty_abandon_boosted_mon",
                self._evaluate_abandon_boosted_mon(battle),
                True,
            ),
            (
                "wasted_free_switch",
                "penalty_wasted_free_switch",
                self._evaluate_wasted_free_switch(battle),
                True,
            ),
        ):
            match = self._make_tactical_match(reason, reward_key, details, record_audit=record_audit)
            if match is not None:
                matches.append(match)
        return matches

    def _evaluate_good_tera(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        """Reward terastallizing when it gains a defensive immunity to the
        opponent's last move, or when tera enables a KO that wouldn't happen
        without it (tera STAB pushes damage past remaining HP)."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return None

        tera_type = getattr(active, "tera_type", None)
        if tera_type is None:
            return None

        # Check defensive immunity: tera type is immune to opponent's last move type
        opp_last_move = opponent.last_move if hasattr(opponent, "last_move") else None
        gains_immunity = False
        if opp_last_move is not None:
            opp_move_type = getattr(opp_last_move, "type", None)
            if opp_move_type is not None:
                pre_tera_mult = opp_move_type.damage_multiplier(
                    active.type_1, active.type_2,
                    type_chart=self.vector_builder._gen_data.type_chart,
                )
                post_tera_mult = opp_move_type.damage_multiplier(
                    tera_type, None,
                    type_chart=self.vector_builder._gen_data.type_chart,
                )
                gains_immunity = pre_tera_mult > 0.0 and post_tera_mult == 0.0

        # Check offensive: tera enables a KO that wouldn't happen without it
        enables_ko = False
        move_type = self.vector_builder._resolve_move_type(active, move, battle)
        if move_type == tera_type and move.base_power and move.base_power > 0:
            opp_hp = _safe_hp_fraction(opponent)
            if opp_hp > 0.0:
                # Compute STAB without tera vs with tera
                pre_stab = 1.5 if move_type in active.types else 1.0
                post_stab = _stab_multiplier(active, move_type)
                if post_stab > pre_stab:
                    # Get base damage range (uses pre-tera STAB internally)
                    try:
                        _, max_pct = self.vector_builder._damage_range_percent(
                            battle, active, opponent, move, None, None,
                        )
                        # Scale max_pct by the STAB upgrade ratio
                        boosted_max_pct = max_pct * (post_stab / pre_stab) if pre_stab > 0 else max_pct
                        # Tera enables KO: without tera can't KO, with tera can
                        enables_ko = max_pct < opp_hp and boosted_max_pct >= opp_hp
                    except Exception:
                        pass

        if gains_immunity or enables_ko:
            return {
                "tera_type": str(tera_type),
                "gains_immunity": gains_immunity,
                "enables_ko": enables_ko,
            }
        return None

    def _update_action_tracking(
        self,
        battle: AbstractBattle,
        action: Move | Pokemon,
    ) -> None:
        """Update per-battle state for heal-satiation and free-switch tracking."""
        active = battle.active_pokemon
        active_species = getattr(active, "species", None) if active else None

        # Detect free switch entry: if last active fainted and current active changed
        if self._last_active_fainted and active_species and active_species != self._last_active_species:
            self._entered_after_faint.add(active_species)
            self._last_active_fainted = False

        # Track consecutive heals per active mon
        if isinstance(action, Move) and active_species:
            is_heal = self._move_has_heal(action)
            if is_heal:
                self._consecutive_heal_count[active_species] = (
                    self._consecutive_heal_count.get(active_species, 0) + 1
                )
            else:
                self._consecutive_heal_count[active_species] = 0

        # If switching, reset heal count for the mon being switched out
        if isinstance(action, Pokemon) and active_species:
            self._consecutive_heal_count[active_species] = 0

        # Track faint status for next turn's free-switch detection
        if active:
            self._last_active_species = active_species
            self._last_active_fainted = bool(getattr(active, "fainted", False))

    def _evaluate_abandon_boosted_mon(
        self,
        battle: AbstractBattle,
    ) -> Optional[Dict[str, Any]]:
        """Penalize switching out a pokemon with significant offensive boosts
        that is not under pressure (high HP and outspeeds opponent)."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return None

        # Check for significant offensive boosts (≥+2 in atk, spa, or spe)
        atk_boost = active.boosts.get("atk", 0)
        spa_boost = active.boosts.get("spa", 0)
        spe_boost = active.boosts.get("spe", 0)
        max_offensive_boost = max(atk_boost, spa_boost)
        total_boost = atk_boost + spa_boost + spe_boost

        if max_offensive_boost < 2 and total_boost < 3:
            return None

        # Check not under pressure: >50% HP
        hp = _safe_hp_fraction(active)
        if hp <= 0.5:
            return None

        # Check outspeeds (not in danger of being revenge-killed)
        speed_adv = self.vector_builder._speed_advantage(battle)
        if speed_adv < 0.5:
            return None

        return {
            "species": active.species,
            "atk_boost": atk_boost,
            "spa_boost": spa_boost,
            "spe_boost": spe_boost,
            "hp_fraction": round(hp, 3),
            "speed_advantage": round(speed_adv, 3),
        }

    def _evaluate_heal_satiation(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        """Penalize using a heal move 3+ times consecutively with the same mon."""
        if not self._move_has_heal(move):
            return None

        active = battle.active_pokemon
        if active is None:
            return None

        species = getattr(active, "species", None)
        if species is None:
            return None

        # Count is updated BEFORE eval, so current count includes this heal
        count = self._consecutive_heal_count.get(species, 0)
        if count < 3:
            return None

        return {
            "species": species,
            "consecutive_heals": count,
            "hp_fraction": round(_safe_hp_fraction(active), 3),
        }

    def _evaluate_wasted_free_switch(
        self,
        battle: AbstractBattle,
    ) -> Optional[Dict[str, Any]]:
        """Penalize switching out a mon on its very first turn after entering
        via forced switch (after a teammate fainted). You should have brought
        the other mon in directly on the free switch."""
        active = battle.active_pokemon
        if active is None:
            return None

        species = getattr(active, "species", None)
        if species is None:
            return None

        if species not in self._entered_after_faint:
            return None

        # This mon entered after a faint and is now switching out on turn 1
        self._entered_after_faint.discard(species)
        return {
            "species": species,
            "hp_fraction": round(_safe_hp_fraction(active), 3),
        }

    def _move_has_status(self, move: Move) -> bool:
        try:
            return move.status is not None
        except Exception:
            entry = getattr(move, "entry", {}) or {}
            return entry.get("status") is not None

    def _move_has_heal(self, move: Move) -> bool:
        if move.id in WEATHER_HEAL_MOVES:
            return True
        try:
            return float(move.heal) > 0.0
        except Exception:
            entry = getattr(move, "entry", {}) or {}
            raw_heal = entry.get("heal")
            if isinstance(raw_heal, (list, tuple)) and len(raw_heal) == 2 and raw_heal[1]:
                return float(raw_heal[0]) > 0.0
            return False

    def _heal_summary(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        active = battle.active_pokemon
        if active is None:
            return None

        raw_heal = self.vector_builder._move_heal_amount(move, battle)
        if raw_heal <= 0.0:
            return None

        effective_heal = self.vector_builder._effective_move_heal_amount(
            move,
            active,
            battle,
        )
        hp_fraction = _safe_hp_fraction(active)
        return {
            "active_hp_fraction": round(hp_fraction, 3),
            "raw_heal": round(raw_heal, 3),
            "effective_heal": round(effective_heal, 3),
            "overflow": round(max(0.0, raw_heal - effective_heal), 3),
        }

    def _evaluate_redundant_stealthrock(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        if move.id != "stealthrock" or SideCondition.STEALTH_ROCK not in battle.opponent_side_conditions:
            return None
        return {"opponent_has_stealthrock": True}

    def _evaluate_redundant_stickyweb(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        if move.id != "stickyweb" or SideCondition.STICKY_WEB not in battle.opponent_side_conditions:
            return None
        return {"opponent_has_stickyweb": True}

    def _evaluate_redundant_spikes(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        layers = int(battle.opponent_side_conditions.get(SideCondition.SPIKES, 0))
        if move.id != "spikes" or layers < 3:
            return None
        return {"opponent_spikes_layers": layers}

    def _evaluate_redundant_status(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        opponent = battle.opponent_active_pokemon
        if not self._move_has_status(move) or opponent is None or opponent.status is None:
            return None
        status_name = getattr(opponent.status, "name", str(opponent.status))
        return {"opponent_status": status_name}

    def _evaluate_bad_encore(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        opponent = battle.opponent_active_pokemon
        if move.id != "encore" or opponent is None or getattr(opponent, "last_move", None):
            return None
        return {"opponent_had_last_move": False}

    def _evaluate_ineffective_heal(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        heal = self._heal_summary(battle, move)
        if heal is None or float(heal["active_hp_fraction"]) <= 0.9:
            return None
        return heal

    def _evaluate_wasteful_heal_overflow(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        heal = self._heal_summary(battle, move)
        if heal is None:
            return None

        hp_fraction = float(heal["active_hp_fraction"])
        overflow = float(heal["overflow"])
        effective_heal = float(heal["effective_heal"])
        raw_heal = max(float(heal["raw_heal"]), 1e-6)
        if hp_fraction < 0.75 or overflow < 0.15:
            return None
        if effective_heal > raw_heal * 0.4:
            return None
        return heal

    def _evaluate_good_heal_timing(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        heal = self._heal_summary(battle, move)
        if heal is None:
            return None

        hp_fraction = float(heal["active_hp_fraction"])
        effective_heal = float(heal["effective_heal"])
        overflow = float(heal["overflow"])
        threat = self._assess_battle_threats(battle)
        under_pressure = (
            hp_fraction <= 0.35
            or threat.active_max_threat >= 0.45
            or threat.active_ohko_risk >= 0.15
        )
        if hp_fraction > 0.6 or effective_heal < 0.2 or overflow > 0.15 or not under_pressure:
            return None

        return {
            **heal,
            "active_max_threat": round(threat.active_max_threat, 3),
            "active_ohko_risk": round(threat.active_ohko_risk, 3),
        }

    def _evaluate_redundant_self_drop_move(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return None

        self_boosts = self._move_self_boosts(move)
        if not self_boosts:
            return None

        relevant_stat = "spa" if move.category == MoveCategory.SPECIAL else "atk"
        if self_boosts.get(relevant_stat, 0) >= 0:
            return None

        current_stage = int(active.boosts.get(relevant_stat, 0))
        if current_stage > -2:
            return None

        chosen_ev, chosen_max = self._move_expected_metrics(battle, active, opponent, move)
        opponent_hp = _safe_hp_fraction(opponent)
        if opponent_hp > 0.0 and chosen_max >= opponent_hp:
            return None

        best_alt = self._best_move_choice(
            battle,
            active,
            opponent,
            exclude_move_id=move.id,
            damaging_only=True,
        )
        best_alt_ev = best_alt["expected_value"] if best_alt is not None else 0.0

        if chosen_ev >= 0.25 and best_alt_ev < chosen_ev * 0.85:
            return None

        return {
            "stat": relevant_stat,
            "current_stage": current_stage,
            "chosen_expected_value": round(chosen_ev, 3),
            "chosen_max_pct": round(chosen_max, 3),
            "opponent_hp_fraction": round(opponent_hp, 3),
            "best_alt_move": best_alt["move_id"] if best_alt is not None else None,
            "best_alt_expected_value": round(max(best_alt_ev, 0.0), 3),
        }

    def _evaluate_good_attack_selection(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None or move.category == MoveCategory.STATUS:
            return None

        chosen_ev, chosen_max = self._move_expected_metrics(battle, active, opponent, move)
        opponent_hp = _safe_hp_fraction(opponent)
        if chosen_ev < 0.25 and chosen_max < opponent_hp:
            return None
        if self._move_has_unjustified_tradeoff(battle, active, opponent, move, chosen_ev, chosen_max):
            return None

        best_move = self._best_move_choice(
            battle,
            active,
            opponent,
            damaging_only=True,
        )
        if best_move is None:
            return None

        best_ev = float(best_move["expected_value"])
        ko_pressure = opponent_hp > 0.0 and chosen_max >= opponent_hp
        near_best = best_ev <= 0.0 or chosen_ev >= best_ev * 0.9
        if not ko_pressure and not near_best:
            return None

        return {
            "chosen_move": move.id,
            "chosen_expected_value": round(chosen_ev, 3),
            "chosen_max_pct": round(chosen_max, 3),
            "opponent_hp_fraction": round(opponent_hp, 3),
            "best_move": best_move["move_id"],
            "best_expected_value": round(best_ev, 3),
            "best_max_pct": round(float(best_move["max_pct"]), 3),
            "secured_ko": bool(ko_pressure),
        }

    def _evaluate_unsafe_stay_in_with_fast_ko_switch(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return None

        chosen_ev, chosen_max = self._move_expected_metrics(battle, active, opponent, move)
        opponent_hp = _safe_hp_fraction(opponent)
        if opponent_hp > 0.0 and chosen_max >= opponent_hp:
            return None

        threat = self._assess_battle_threats(battle)
        if not threat.threat_entries:
            return None
        if threat.active_max_threat < 0.6 and threat.active_ohko_risk < 0.25:
            return None
        if threat.opponent_speed is None:
            return None

        best_candidate: Optional[Dict[str, Any]] = None
        for candidate in self._candidate_switches(battle):
            candidate_metrics = self._switch_candidate_metrics(battle, candidate, threat)
            if candidate_metrics is None:
                continue
            if not candidate_metrics["faster_or_equal"] or not candidate_metrics["resists_all_threats"]:
                continue
            if candidate_metrics["switch_max_threat"] > 0.35 or candidate_metrics["switch_ohko_risk"] > 0.1:
                continue
            if float(candidate_metrics["best_reply_max_pct"] or 0.0) < opponent_hp:
                continue

            best_candidate = candidate_metrics
            break

        if best_candidate is None:
            return None

        return {
            "chosen_move": move.id,
            "chosen_expected_value": round(chosen_ev, 3),
            "chosen_max_pct": round(chosen_max, 3),
            "active_max_threat": round(threat.active_max_threat, 3),
            "active_ohko_risk": round(threat.active_ohko_risk, 3),
            "opponent_hp_fraction": round(opponent_hp, 3),
            **best_candidate,
        }

    def _evaluate_good_safe_switch(
        self,
        battle: AbstractBattle,
        switch_target: Pokemon,
    ) -> Optional[Dict[str, Any]]:
        opponent = battle.opponent_active_pokemon
        if opponent is None:
            return None

        threat = self._assess_battle_threats(battle)
        if not threat.threat_entries:
            return None
        if threat.active_max_threat < 0.55 and threat.active_ohko_risk < 0.25:
            return None

        candidate_metrics = self._switch_candidate_metrics(battle, switch_target, threat)
        if candidate_metrics is None:
            return None
        if not candidate_metrics["resists_all_threats"]:
            return None
        if candidate_metrics["switch_max_threat"] > 0.45 or candidate_metrics["switch_ohko_risk"] > 0.15:
            return None

        opponent_hp = _safe_hp_fraction(opponent)
        if not self._has_credible_switch_reply(candidate_metrics, opponent_hp):
            return None

        improves_board = (
            candidate_metrics["switch_max_threat"] + 0.15 < threat.active_max_threat
            or candidate_metrics["switch_ohko_risk"] + 0.15 < threat.active_ohko_risk
        )
        if not improves_board:
            return None

        return {
            "active_max_threat": round(threat.active_max_threat, 3),
            "active_ohko_risk": round(threat.active_ohko_risk, 3),
            "opponent_hp_fraction": round(opponent_hp, 3),
            **candidate_metrics,
        }

    def _move_self_boosts(self, move: Move) -> Dict[str, int]:
        try:
            boosts = move.self_boost
        except Exception:
            boosts = None
        if isinstance(boosts, dict):
            return {str(stat): int(amount) for stat, amount in boosts.items()}

        entry = getattr(move, "entry", {}) or {}
        if isinstance(entry.get("self"), dict) and isinstance(entry["self"].get("boosts"), dict):
            return {str(stat): int(amount) for stat, amount in entry["self"]["boosts"].items()}
        if isinstance(entry.get("selfBoost"), dict) and isinstance(entry["selfBoost"].get("boosts"), dict):
            return {str(stat): int(amount) for stat, amount in entry["selfBoost"]["boosts"].items()}
        return {}

    def _move_expected_metrics(
        self,
        battle: AbstractBattle,
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
    ) -> tuple[float, float]:
        min_pct, max_pct = self.vector_builder._damage_range_percent(
            battle,
            attacker,
            defender,
            move,
            battle.player_role,
            battle.opponent_role,
        )
        accuracy = _clamp01(float(getattr(move, "accuracy", 1.0) or 1.0))
        expected_value = _clamp01(((min_pct + max_pct) / 2.0) * accuracy)
        return expected_value, max_pct

    def _best_move_choice(
        self,
        battle: AbstractBattle,
        attacker: Pokemon,
        defender: Pokemon,
        *,
        exclude_move_id: str | None = None,
        damaging_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        best_choice: Optional[Dict[str, Any]] = None
        for candidate in tuple(battle.available_moves):
            if candidate.id == exclude_move_id:
                continue

            expected_value, max_pct = self._move_expected_metrics(battle, attacker, defender, candidate)
            if damaging_only and expected_value <= 0.0 and max_pct <= 0.0:
                continue
            if best_choice is None or expected_value > best_choice["expected_value"] or (
                expected_value == best_choice["expected_value"] and max_pct > best_choice["max_pct"]
            ):
                best_choice = {
                    "move_id": candidate.id,
                    "expected_value": expected_value,
                    "max_pct": max_pct,
                }
        return best_choice

    def _move_has_unjustified_tradeoff(
        self,
        battle: AbstractBattle,
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        expected_value: float,
        max_pct: float,
    ) -> bool:
        opponent_hp = _safe_hp_fraction(defender)
        if opponent_hp > 0.0 and max_pct >= opponent_hp:
            return False

        self_boosts = self._move_self_boosts(move)
        relevant_stat = "spa" if move.category == MoveCategory.SPECIAL else "atk"
        if self_boosts.get(relevant_stat, 0) < 0:
            return True
        if self.vector_builder._move_causes_recharge(move) >= 1.0:
            return True

        recoil = self.vector_builder._estimated_recoil_fraction(
            attacker,
            defender,
            move,
            expected_value,
        )
        return recoil > 0.25 and expected_value < 0.35

    def _candidate_switches(self, battle: AbstractBattle) -> list[Pokemon]:
        available_switches = getattr(battle, "available_switches", None)
        if available_switches:
            return [mon for mon in available_switches if mon is not None and not mon.fainted]
        return [
            mon
            for mon in battle.team.values()
            if mon is not None and not mon.active and not mon.fainted
        ]

    def _resists_all_threat_moves(
        self,
        battle: AbstractBattle,
        attacker: Pokemon,
        defender: Pokemon,
        threat_entries: Sequence[OpponentThreatEntry],
    ) -> bool:
        for entry in threat_entries:
            move_type = self.vector_builder._resolve_move_type(attacker, entry.move, battle)
            type_mult = _defender_type_mult(
                move_type, defender, self.vector_builder._gen_data.type_chart,
            )
            if type_mult > 0.5:
                return False
        return True

    def _assess_battle_threats(self, battle: AbstractBattle) -> ThreatAssessment:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return ThreatAssessment(
                posterior={},
                threat_entries=tuple(),
                active_max_threat=0.0,
                active_ohko_risk=0.0,
                active_speed=None,
                opponent_speed=None,
            )

        posterior = self.vector_builder._opponent_role_posterior(opponent)
        threat_entries = tuple(
            entry
            for entry in self.vector_builder._select_opponent_threat_entries(opponent, posterior)
            if entry.move.category != MoveCategory.STATUS and float(getattr(entry.move, "base_power", 0) or 0) > 0.0
        )
        active_max_threat = 0.0
        if threat_entries:
            active_max_threat = max(
                self.vector_builder._move_expected_value(
                    battle,
                    opponent,
                    active,
                    entry.move,
                    battle.opponent_role,
                    battle.player_role,
                )
                for entry in threat_entries
            )

        active_ohko_risk = self.vector_builder._estimate_ohko_risk(
            battle,
            opponent,
            active,
            posterior,
        )
        active_speed = self.vector_builder._effective_speed(active, battle.side_conditions)
        opponent_speed = self.vector_builder._effective_speed(
            opponent,
            battle.opponent_side_conditions,
            role_posterior=posterior,
        )
        return ThreatAssessment(
            posterior=posterior,
            threat_entries=threat_entries,
            active_max_threat=active_max_threat,
            active_ohko_risk=active_ohko_risk,
            active_speed=active_speed,
            opponent_speed=opponent_speed,
        )

    def _best_known_bench_reply(
        self,
        battle: AbstractBattle,
        attacker: Pokemon,
        defender: Pokemon,
    ) -> Optional[Dict[str, Any]]:
        best_reply = None
        for move in tuple(attacker.moves.values())[:4]:
            _, max_pct = self._move_expected_metrics(battle, attacker, defender, move)
            if best_reply is None or max_pct > best_reply["max_pct"]:
                best_reply = {"move_id": move.id, "max_pct": max_pct}
        return best_reply

    def _switch_candidate_metrics(
        self,
        battle: AbstractBattle,
        candidate: Pokemon,
        threat: ThreatAssessment,
    ) -> Optional[Dict[str, Any]]:
        opponent = battle.opponent_active_pokemon
        if opponent is None:
            return None

        candidate_speed = self.vector_builder._effective_speed(candidate, battle.side_conditions)
        faster = (
            candidate_speed is not None
            and threat.opponent_speed is not None
            and candidate_speed >= threat.opponent_speed
        )
        candidate_max_threat = 0.0
        candidate_ohko_risk = 0.0
        if threat.threat_entries:
            candidate_max_threat = max(
                self.vector_builder._move_expected_value(
                    battle,
                    opponent,
                    candidate,
                    entry.move,
                    battle.opponent_role,
                    battle.player_role,
                )
                for entry in threat.threat_entries
            )
            candidate_ohko_risk = self.vector_builder._estimate_ohko_risk(
                battle,
                opponent,
                candidate,
                threat.posterior,
            )

        best_reply = self._best_known_bench_reply(battle, candidate, opponent)
        return {
            "switch_species": getattr(candidate, "species", "<unknown>"),
            "switch_hp_fraction": round(_safe_hp_fraction(candidate), 3),
            "switch_speed": round(candidate_speed, 1) if candidate_speed is not None else None,
            "faster_or_equal": bool(faster),
            "resists_all_threats": self._resists_all_threat_moves(
                battle,
                opponent,
                candidate,
                threat.threat_entries,
            )
            if threat.threat_entries
            else False,
            "switch_max_threat": round(candidate_max_threat, 3),
            "switch_ohko_risk": round(candidate_ohko_risk, 3),
            "best_reply_move": best_reply["move_id"] if best_reply is not None else None,
            "best_reply_max_pct": round(best_reply["max_pct"], 3) if best_reply is not None else None,
        }

    def _has_credible_switch_reply(
        self,
        candidate_metrics: Dict[str, Any],
        opponent_hp_fraction: float,
    ) -> bool:
        best_reply_max_pct = float(candidate_metrics.get("best_reply_max_pct") or 0.0)
        return best_reply_max_pct >= min(0.5, opponent_hp_fraction)

    def _record_decision_audit(
        self,
        category: str,
        battle: AbstractBattle,
        order: BattleOrder,
        move: Move,
        details: Dict[str, Any],
    ) -> None:
        self._decision_audit_counts[category] = self._decision_audit_counts.get(category, 0) + 1
        samples = self._decision_audit_samples.setdefault(category, [])
        if len(samples) >= DECISION_AUDIT_SAMPLE_LIMIT:
            return

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        threat = self._assess_battle_threats(battle)
        feature_snapshot = self._decision_feature_snapshot(battle, move, threat)
        samples.append(
            {
                "battle_tag": _battle_tag(battle),
                "turn": getattr(battle, "turn", None),
                "category": category,
                "chosen_move": move.id,
                "chosen_order": str(order),
                "terastallize": bool(getattr(order, "terastallize", False)),
                "active_species": getattr(active, "species", None),
                "opponent_species": getattr(opponent, "species", None),
                "active_hp_fraction": round(_safe_hp_fraction(active), 3),
                "opponent_hp_fraction": round(_safe_hp_fraction(opponent), 3),
                "active_boosts": {k: v for k, v in getattr(active, "boosts", {}).items() if v},
                "opponent_boosts": {k: v for k, v in getattr(opponent, "boosts", {}).items() if v},
                "bench_summary": [
                    {
                        "species": getattr(mon, "species", None),
                        "hp_fraction": round(_safe_hp_fraction(mon), 3),
                        "active": bool(getattr(mon, "active", False)),
                        "fainted": bool(getattr(mon, "fainted", False)),
                    }
                    for mon in battle.team.values()
                    if mon is not None and not getattr(mon, "active", False)
                ][:5],
                "top_opp_threats": [
                    {
                        "move_id": entry.move.id,
                        "move_prob": round(entry.move_prob, 3),
                        "revealed": bool(entry.revealed_flag),
                    }
                    for entry in threat.threat_entries[:4]
                ],
                "feature_snapshot": feature_snapshot,
                "details": details,
            }
        )

    def _decision_feature_snapshot(
        self,
        battle: AbstractBattle,
        chosen_move: Move,
        threat: ThreatAssessment,
    ) -> Dict[str, Any]:
        vector = self.vector_builder.embed_battle(battle)

        move_block = self._move_block_snapshot(battle, vector, chosen_move)
        return {
            "speed": {
                "speed_advantage_feature": round(float(vector[SPEED_ADVANTAGE_INDEX]), 3),
                "active_speed_est": round(float(threat.active_speed), 1) if threat.active_speed is not None else None,
                "opponent_speed_est": round(float(threat.opponent_speed), 1) if threat.opponent_speed is not None else None,
            },
            "chosen_move_block": move_block,
            "threat_summary": {
                "active_max_threat": round(threat.active_max_threat, 3),
                "active_ohko_risk": round(threat.active_ohko_risk, 3),
                "opponent_top_role_mass": round(float(vector[OPP_THREAT_CONFIDENCE_START]), 3),
                "opponent_role_entropy_norm": round(float(vector[OPP_THREAT_CONFIDENCE_START + 1]), 3),
                "top_moves": [
                    {
                        "move_id": entry.move.id,
                        "move_prob": round(entry.move_prob, 3),
                        "revealed": bool(entry.revealed_flag),
                    }
                    for entry in threat.threat_entries[:4]
                ],
            },
            "switch_options": self._safe_switch_snapshot(battle, threat),
        }

    def _move_block_snapshot(
        self,
        battle: AbstractBattle,
        vector: np.ndarray,
        chosen_move: Move,
    ) -> Dict[str, Any]:
        for slot, move in enumerate(tuple(battle.available_moves)[:4]):
            if move.id != chosen_move.id:
                continue
            start = MY_MOVES_START + slot * MOVE_BLOCK_SIZE
            block = vector[start : start + MOVE_BLOCK_SIZE]
            return {
                "slot": slot,
                "move_id": move.id,
                "min_pct": round(float(block[0]), 3),
                "max_pct": round(float(block[1]), 3),
                "accuracy": round(float(block[2]), 3),
                "expected_value": round(float(block[3]), 3),
                "ko_flag": int(block[4]),
                "heal": round(float(block[12]), 3),
                "drain": round(float(block[13]), 3),
                "self_atk_delta": round(float(block[14]), 3),
                "self_spa_delta": round(float(block[15]), 3),
                "self_spe_delta": round(float(block[16]), 3),
                "recoil": round(float(block[17]), 3),
                "recharge": int(block[18]),
                "brn": round(float(block[19]), 3),
                "par": round(float(block[20]), 3),
                "psn": round(float(block[21]), 3),
                "frz": round(float(block[22]), 3),
                "slp": round(float(block[23]), 3),
                "conf": round(float(block[24]), 3),
            }
        return {"move_id": chosen_move.id, "slot": None}

    def _safe_switch_snapshot(
        self,
        battle: AbstractBattle,
        threat: ThreatAssessment,
    ) -> list[Dict[str, Any]]:
        if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
            return []

        candidates: list[Dict[str, Any]] = []
        for candidate in self._candidate_switches(battle):
            candidate_metrics = self._switch_candidate_metrics(battle, candidate, threat)
            if candidate_metrics is None:
                continue
            candidates.append(
                {
                    "species": candidate_metrics["switch_species"],
                    "hp_fraction": candidate_metrics["switch_hp_fraction"],
                    "faster_or_equal": candidate_metrics["faster_or_equal"],
                    "resists_all_threats": candidate_metrics["resists_all_threats"],
                    "max_threat": candidate_metrics["switch_max_threat"],
                    "ohko_risk": candidate_metrics["switch_ohko_risk"],
                    "best_reply_move": candidate_metrics["best_reply_move"],
                    "best_reply_max_pct": candidate_metrics["best_reply_max_pct"],
                }
            )

        candidates.sort(
            key=lambda item: (
                -int(item["faster_or_equal"]),
                -int(item["resists_all_threats"]),
                item["ohko_risk"],
                -float(item["best_reply_max_pct"] or 0.0),
            )
        )
        return candidates[:3]

    def get_tactical_shaping_report(self) -> dict[str, Any]:
        return {
            "decision_count": self._decision_count,
            "move_checks": self._tactical_shaping_move_checks,
            "switch_actions": self._switch_action_count,
            "switch_rate": (
                self._switch_action_count / self._decision_count if self._decision_count else 0.0
            ),
            "shaped_actions": self._tactical_shaping_shaped_actions,
            "rewarded_actions": self._tactical_shaping_rewarded_actions,
            "penalized_actions": self._tactical_shaping_penalized_actions,
            "shaped_action_rate": (
                self._tactical_shaping_shaped_actions / self._decision_count if self._decision_count else 0.0
            ),
            "total_shaping": self._tactical_shaping_total,
            "positive_total": self._tactical_positive_total,
            "negative_total": self._tactical_negative_total,
            "counts": dict(sorted(self._tactical_shaping_counts.items())),
            "totals": {
                category: round(total, 3)
                for category, total in sorted(self._tactical_shaping_totals.items())
            },
        }

    def get_strategic_penalty_report(self) -> dict[str, Any]:
        return {
            "decision_count": self._decision_count,
            "move_checks": self._strategic_penalty_move_checks,
            "switch_actions": self._switch_action_count,
            "switch_rate": (
                self._switch_action_count / self._decision_count if self._decision_count else 0.0
            ),
            "penalized_actions": self._strategic_penalty_penalized_actions,
            "total_penalty": self._strategic_penalty_total,
            "penalized_action_rate": (
                self._strategic_penalty_penalized_actions / self._strategic_penalty_move_checks
                if self._strategic_penalty_move_checks
                else 0.0
            ),
            "counts": dict(sorted(self._strategic_penalty_counts.items())),
        }

    def get_decision_audit_report(self) -> dict[str, Any]:
        return {
            "decision_count": self._decision_count,
            "move_checks": self._decision_audit_move_checks,
            "switch_actions": self._switch_action_count,
            "switch_rate": (
                self._switch_action_count / self._decision_count if self._decision_count else 0.0
            ),
            "flagged_actions": self._decision_audit_flagged_actions,
            "flagged_action_rate": (
                self._decision_audit_flagged_actions / self._decision_audit_move_checks
                if self._decision_audit_move_checks
                else 0.0
            ),
            "counts": dict(sorted(self._decision_audit_counts.items())),
            "samples": {category: list(samples) for category, samples in sorted(self._decision_audit_samples.items())},
        }

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return self.vector_builder.embed_battle(battle)


assert MY_MOVES_START + 4 * MOVE_BLOCK_SIZE == MY_BENCH_START
assert MY_BENCH_START + 5 * MY_BENCH_SLOT_SIZE == OPP_BENCH_START
assert OPP_BENCH_START + 5 * OPP_BENCH_SLOT_SIZE == TARGETING_START
assert TARGETING_START + 20 == MY_TEAM_REVEALED_START
assert MY_TEAM_REVEALED_START + 6 == OPP_THREAT_START
assert OPP_THREAT_START + OPP_THREAT_ROWS * OPP_THREAT_ROW_SIZE == OPP_THREAT_OHKO_START
assert OPP_THREAT_OHKO_START + 6 == OPP_THREAT_CONFIDENCE_START
assert OPP_THREAT_CONFIDENCE_START + 2 == ON_RECHARGE_INDEX
assert ON_RECHARGE_INDEX + 1 == VECTOR_LENGTH