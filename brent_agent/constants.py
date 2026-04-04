from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
    "victory_value": 15.0,
    "penalty_redundant_stealthrock": -0.15,
    "penalty_redundant_stickyweb": -0.15,
    "penalty_redundant_spikes": -0.15,
    "penalty_redundant_status": -0.05,
    "penalty_bad_encore": -0.05,
    "penalty_ineffective_heal": -0.05,
    "penalty_wasteful_heal_overflow": -0.025,
    "penalty_redundant_self_drop_move": -0.05,
    "penalty_unsafe_stay_in_with_fast_ko_switch": -0.2,
    "bonus_good_heal_timing": 0.2,
    "bonus_good_attack_selection": 0.0,
    "bonus_good_safe_switch": 0.2,
    "bonus_good_tera": 0.5,
    "penalty_abandon_boosted_mon": -0.05,
    "penalty_heal_satiation": -0.15,
    "penalty_wasted_free_switch": 0.0,
    "bonus_good_setup": 0.25,
    "bonus_pivot_into_advantage": 0.15,
    "bonus_predicted_switch": 0.3,
    "penalty_redundant_taunt": -0.15,
}

DECISION_AUDIT_SAMPLE_LIMIT = 20

MOVE_BLOCK_SIZE = 38  # 37 base + 1 PP fraction
MY_BENCH_SLOT_SIZE = 63  # 53 base + 5 matchup + 5 status (psn_sev, par, brn, slp, frz)
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
MY_BENCH_START = MY_MOVES_START + 4 * MOVE_BLOCK_SIZE  # 278
OPP_BENCH_START = MY_BENCH_START + 5 * MY_BENCH_SLOT_SIZE  # 593
TARGETING_START = OPP_BENCH_START + 5 * OPP_BENCH_SLOT_SIZE  # 693
MY_TEAM_REVEALED_START = TARGETING_START + 20  # 713
OPP_THREAT_START = MY_TEAM_REVEALED_START + 6  # 719
OPP_MOVES_VS_ME_START = OPP_THREAT_START + 2
OPP_THREAT_OHKO_START = OPP_THREAT_START + OPP_THREAT_ROWS * OPP_THREAT_ROW_SIZE
OPP_THREAT_CONFIDENCE_START = OPP_THREAT_OHKO_START + 6
ON_RECHARGE_INDEX = OPP_THREAT_CONFIDENCE_START + 2
ALIVE_DIFF_INDEX = ON_RECHARGE_INDEX + 1
FORCE_SWITCH_INDEX = ALIVE_DIFF_INDEX + 1
# Extended tail features
SPEED_RATIO_INDEX = FORCE_SWITCH_INDEX + 1
MY_TAILWIND_INDEX = SPEED_RATIO_INDEX + 1
OPP_TAILWIND_INDEX = MY_TAILWIND_INDEX + 1
OPP_ABILITY_START = OPP_TAILWIND_INDEX + 1
OPP_ABILITY_SIZE = 15
VECTOR_LENGTH = OPP_ABILITY_START + OPP_ABILITY_SIZE

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

OPP_ABILITY_ORDER = (
    "levitate", "flashfire", "voltabsorb", "waterabsorb",
    "lightningrod", "stormdrain", "motordrive", "sapsipper",
    "dryskin", "sturdy", "multiscale", "shadowshield",
    "intimidate", "magicbounce", "unaware",
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


assert MY_MOVES_START + 4 * MOVE_BLOCK_SIZE == MY_BENCH_START
assert MY_BENCH_START + 5 * MY_BENCH_SLOT_SIZE == OPP_BENCH_START
assert OPP_BENCH_START + 5 * OPP_BENCH_SLOT_SIZE == TARGETING_START
assert TARGETING_START + 20 == MY_TEAM_REVEALED_START
assert MY_TEAM_REVEALED_START + 6 == OPP_THREAT_START
assert OPP_THREAT_START + OPP_THREAT_ROWS * OPP_THREAT_ROW_SIZE == OPP_THREAT_OHKO_START
assert OPP_THREAT_OHKO_START + 6 == OPP_THREAT_CONFIDENCE_START
assert OPP_THREAT_CONFIDENCE_START + 2 == ON_RECHARGE_INDEX
assert ON_RECHARGE_INDEX + 1 == ALIVE_DIFF_INDEX
assert ALIVE_DIFF_INDEX + 1 == FORCE_SWITCH_INDEX
assert FORCE_SWITCH_INDEX + 1 == SPEED_RATIO_INDEX
assert OPP_ABILITY_START + OPP_ABILITY_SIZE == VECTOR_LENGTH