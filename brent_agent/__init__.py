"""brent_agent package — re-exports for backward compatibility.

All consumer files (policies.py, train_ppo.py, test_runner.py, etc.)
can continue to ``from brent_agent import X`` without changes.
"""
from .constants import (
    ALIVE_DIFF_INDEX,
    FORCE_SWITCH_INDEX,
    BENCH_MOVE_FLAG_SIZE,
    BOOST_ORDER,
    DAMAGE_BOOST_ITEMS,
    DECISION_AUDIT_SAMPLE_LIMIT,
    MOVE_BLOCK_SIZE,
    MY_ACTIVE_BLOCK_SIZE,
    MY_ACTIVE_START,
    MY_BENCH_SLOT_SIZE,
    MY_BENCH_START,
    MY_MOVES_START,
    MY_SIDE_START,
    MY_TEAM_REVEALED_START,
    ON_RECHARGE_INDEX,
    OPP_ACTIVE_BLOCK_SIZE,
    OPP_ACTIVE_START,
    OPP_BENCH_SLOT_SIZE,
    OPP_BENCH_START,
    OPP_MOVES_VS_ME_START,
    OPP_SIDE_START,
    OPP_THREAT_CONFIDENCE_START,
    OPP_THREAT_OHKO_START,
    OPP_THREAT_ROW_SIZE,
    OPP_THREAT_ROWS,
    OPP_THREAT_START,
    PIVOT_MOVES,
    POKE_ENV_REWARD_KEYS,
    RECOVERY_ITEMS,
    REWARD_CONFIG,
    SIDE_CONDITION_ORDER,
    SPEED_ADVANTAGE_INDEX,
    SPEED_BOOST_ITEMS,
    SPEED_DROP_ITEMS,
    STATUS_ORDER,
    TARGETING_START,
    TERRAIN_ORDER,
    TRICK_ROOM_INDEX,
    TURN_INDEX,
    TYPE_ORDER,
    VECTOR_LENGTH,
    VOLATILE_ORDER,
    WEATHER_ORDER,
    WEATHER_HEAL_MOVES,
    ABILITY_TYPE_IMMUNITIES,
    ATE_ABILITIES,
    OpponentThreatEntry,
    TacticalLeverMatch,
    TacticalRewardContext,
    ThreatAssessment,
    ability_immune,
    battle_tag,
    clamp,
    clamp01,
    defender_type_mult,
    effective_types,
    mon_key,
    normalize_ability,
    poison_severity,
    safe_hp_fraction,
    safe_identifier,
    stab_multiplier,
    stat_stage_multiplier,
)
from .mechanics import PokemonMechanics
from .observation import BrentObservationVectorBuilder
from .agent import BrentsRLAgent

# Backward-compat aliases (test_runner.py imports these with underscore prefix)
_stab_multiplier = stab_multiplier
_defender_type_mult = defender_type_mult
_effective_types = effective_types
_clamp01 = clamp01
_clamp = clamp
_safe_hp_fraction = safe_hp_fraction
_stat_stage_multiplier = stat_stage_multiplier
_ability_immune = ability_immune
_battle_tag = battle_tag
_mon_key = mon_key
_poison_severity = poison_severity
_safe_identifier = safe_identifier
