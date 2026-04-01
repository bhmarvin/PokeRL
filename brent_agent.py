from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

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
from poke_env.environment.singles_env import SinglesEnv
from poke_env.ps_client import AccountConfiguration

REWARD_CONFIG = {
    "fainted_value": 1.0,
    "hp_value": 1.0,
    "status_value": 0.25,
    "victory_value": 10.0,
}

VECTOR_LENGTH = 572
MOVE_BLOCK_SIZE = 14
MY_BENCH_SLOT_SIZE = 52
OPP_BENCH_SLOT_SIZE = 20
BENCH_MOVE_FLAG_SIZE = 8

TURN_INDEX = 0
WEATHER_START = 1
TERRAIN_START = 5
TRICK_ROOM_INDEX = 9
MY_SIDE_START = 10
OPP_SIDE_START = 17
MY_ACTIVE_START = 24
OPP_ACTIVE_START = 64
SPEED_ADVANTAGE_INDEX = 104
MY_MOVES_START = 105
MY_BENCH_START = 161
OPP_BENCH_START = 421
TARGETING_START = 521
MY_TEAM_REVEALED_START = 541
OPP_MOVES_VS_ME_START = 547
ON_RECHARGE_INDEX = 571

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


class BrentObservationVectorBuilder:
    def __init__(self) -> None:
        self._my_team_revealed_memory: Dict[str, set[str]] = {}
        self._damage_cache: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
        self._fallback_counts: Dict[Tuple[str, str, str], int] = {}
        self._fallback_samples: list[Dict[str, Any]] = []

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        self._damage_cache = {}
        self._update_reveal_memory(battle)
        try:
            vector = np.zeros(VECTOR_LENGTH, dtype=np.float32)

            self._fill_global_features(vector, battle)
            self._fill_side_conditions(vector, MY_SIDE_START, battle.side_conditions)
            self._fill_side_conditions(vector, OPP_SIDE_START, battle.opponent_side_conditions)
            self._fill_active_block(vector, MY_ACTIVE_START, battle.active_pokemon, battle.available_moves, battle)
            self._fill_active_block(
                vector,
                OPP_ACTIVE_START,
                battle.opponent_active_pokemon,
                tuple(battle.opponent_active_pokemon.moves.values()) if battle.opponent_active_pokemon else (),
                battle,
            )
            vector[SPEED_ADVANTAGE_INDEX] = self._speed_advantage(battle)
            self._fill_available_move_blocks(vector, battle)
            self._fill_my_bench(vector, battle)
            self._fill_opponent_bench(vector, battle)
            self._fill_targeting_matrix(vector, battle)
            self._fill_theory_of_mind(vector, battle)
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
        self._verify_active_block(issues, vector, MY_ACTIVE_START, battle.active_pokemon, "my_active")
        self._verify_active_block(
            issues,
            vector,
            OPP_ACTIVE_START,
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
        self._verify_theory_of_mind_leaks(issues, vector, battle)
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
        self._fill_type_one_hot(vector, start + 1, mon.types)

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
        mon: Optional[Pokemon],
        prefix: str,
    ) -> None:
        if mon is None:
            expected = np.zeros(40, dtype=np.float32)
            observed = vector[start : start + 40]
            if not np.allclose(observed, expected):
                issues.append(f"{prefix} expected zero block when pokemon is None")
            return

        self._verify_scalar(issues, f"{prefix}.hp", vector[start], _safe_hp_fraction(mon))
        type_set = set(mon.types)
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

    def _verify_theory_of_mind_leaks(
        self,
        issues: list[str],
        vector: np.ndarray,
        battle: AbstractBattle,
    ) -> None:
        opponent = battle.opponent_active_pokemon
        revealed_count = len(tuple(opponent.moves.values())[:4]) if opponent is not None else 0
        for move_idx in range(revealed_count, 4):
            start = OPP_MOVES_VS_ME_START + move_idx * 6
            if not np.allclose(vector[start : start + 6], 0.0):
                issues.append(f"theory_of_mind row {move_idx} leaked unrevealed opponent move data")

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
        vector[start + 12] = _clamp01(self._move_heal_amount(move, battle))
        vector[start + 13] = _clamp01(self._move_drain(move))

    def _fill_my_bench(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        bench = [mon for mon in battle.team.values() if not mon.active][:5]
        for slot, mon in enumerate(bench):
            start = MY_BENCH_START + slot * MY_BENCH_SLOT_SIZE
            vector[start] = 0.0 if mon.fainted else 1.0
            vector[start + 1] = _safe_hp_fraction(mon)
            self._fill_type_one_hot(vector, start + 2, mon.types)
            for move_idx, move in enumerate(tuple(mon.moves.values())[:4]):
                move_start = start + 20 + move_idx * BENCH_MOVE_FLAG_SIZE
                self._fill_bench_move_flags(vector, move_start, move, battle)

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
        vector[start + 6] = _clamp01(self._move_heal_amount(move, battle))
        vector[start + 7] = _clamp01(self._move_drain(move))

    def _fill_opponent_bench(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        bench = [mon for mon in battle.opponent_team.values() if not mon.active][:5]
        for slot, mon in enumerate(bench):
            if not mon.revealed:
                continue
            start = OPP_BENCH_START + slot * OPP_BENCH_SLOT_SIZE
            vector[start] = 1.0
            vector[start + 1] = _safe_hp_fraction(mon)
            self._fill_type_one_hot(vector, start + 2, mon.types)

    def _fill_targeting_matrix(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        attacker = battle.active_pokemon
        if attacker is None:
            return

        opponent_bench = [mon for mon in battle.opponent_team.values() if not mon.active][:5]
        my_moves = tuple(battle.available_moves)[:4]
        for move_idx, move in enumerate(my_moves):
            for target_idx, target in enumerate(opponent_bench):
                if not target.revealed:
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

    def _fill_theory_of_mind(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        my_team_order = self._my_team_order(battle)
        memory = self._my_team_revealed_memory.get(_battle_tag(battle), set())
        for idx, mon in enumerate(my_team_order):
            if mon is None:
                continue
            vector[MY_TEAM_REVEALED_START + idx] = 1.0 if _mon_key(mon) in memory else 0.0

        opponent = battle.opponent_active_pokemon
        if opponent is None:
            return

        revealed_moves = tuple(opponent.moves.values())[:4]
        for move_idx, move in enumerate(revealed_moves):
            for target_idx, target in enumerate(my_team_order):
                if target is None:
                    continue
                min_pct, max_pct = self._damage_range_percent(
                    battle,
                    opponent,
                    target,
                    move,
                    battle.opponent_role,
                    battle.player_role,
                )
                ev = _clamp01(((min_pct + max_pct) / 2.0) * _clamp01(float(move.accuracy)))
                vector[OPP_MOVES_VS_ME_START + move_idx * 6 + target_idx] = ev

    def _my_team_order(self, battle: AbstractBattle) -> Tuple[Optional[Pokemon], ...]:
        bench = [mon for mon in battle.team.values() if not mon.active][:5]
        ordered: list[Optional[Pokemon]] = [battle.active_pokemon]
        ordered.extend(bench)
        while len(ordered) < 6:
            ordered.append(None)
        return tuple(ordered[:6])

    def _speed_advantage(self, battle: AbstractBattle) -> float:
        my_speed = self._effective_speed(battle.active_pokemon, battle.side_conditions)
        opp_speed = self._effective_speed(
            battle.opponent_active_pokemon,
            battle.opponent_side_conditions,
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
    ) -> Optional[float]:
        if mon is None:
            return None
        base_speed = mon.stats.get("spe")
        if not isinstance(base_speed, (int, float)):
            return None

        speed = float(base_speed) * _stat_stage_multiplier(mon.boosts.get("spe", 0))

        if mon.status == Status.PAR:
            speed *= 0.5
        if SideCondition.TAILWIND in side_conditions:
            speed *= 2.0
        if mon.item in SPEED_BOOST_ITEMS:
            speed *= 1.5
        if mon.item in SPEED_DROP_ITEMS:
            speed *= 0.5
        if Effect.QUARKDRIVESPE in mon.effects or Effect.PROTOSYNTHESISSPE in mon.effects:
            speed *= 1.5
        return speed

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
        attacker_id = _safe_identifier(attacker, attacker_role)
        defender_id = _safe_identifier(defender, defender_role)
        if attacker_id is None or defender_id is None:
            return 0.0, 0.0
        if not self._stats_defined(attacker) or not self._stats_defined(defender):
            return 0.0, 0.0

        cache_key = (attacker_id, defender_id, move.id)
        cached = self._damage_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            min_damage, max_damage = calculate_damage(attacker_id, defender_id, move, battle)
        except Exception:
            return 0.0, 0.0

        damage_range = (float(min_damage), float(max_damage))
        self._damage_cache[cache_key] = damage_range
        return damage_range

    def _stats_defined(self, mon: Pokemon) -> bool:
        return all(isinstance(value, (int, float)) for value in mon.stats.values())

    def _defender_hp_scale(self, mon: Pokemon) -> float:
        hp_stat = mon.stats.get("hp")
        if isinstance(hp_stat, (int, float)) and hp_stat > 0:
            return float(hp_stat)
        max_hp = getattr(mon, "max_hp", 0)
        if isinstance(max_hp, (int, float)) and max_hp > 0:
            return float(max_hp)
        return 100.0

    def _is_stab(self, attacker: Pokemon, move: Move, battle: AbstractBattle) -> float:
        move_type = self._resolve_move_type(attacker, move, battle)
        return 1.0 if move_type in attacker.types else 0.0

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
        return self.reward_computing_helper(battle, **REWARD_CONFIG)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return self.vector_builder.embed_battle(battle)


assert MY_MOVES_START + 4 * MOVE_BLOCK_SIZE == MY_BENCH_START
assert MY_BENCH_START + 5 * MY_BENCH_SLOT_SIZE == OPP_BENCH_START
assert OPP_BENCH_START + 5 * OPP_BENCH_SLOT_SIZE == TARGETING_START
assert TARGETING_START + 20 == MY_TEAM_REVEALED_START
assert MY_TEAM_REVEALED_START + 6 == OPP_MOVES_VS_ME_START
assert OPP_MOVES_VS_ME_START + 24 == ON_RECHARGE_INDEX
assert ON_RECHARGE_INDEX + 1 == VECTOR_LENGTH