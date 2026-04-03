from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
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

from randbats_data import RandbatsMeta

from .constants import (
    ALIVE_DIFF_INDEX,
    BENCH_MOVE_FLAG_SIZE,
    BOOST_ORDER,
    DAMAGE_BOOST_ITEMS,
    FORCE_SWITCH_INDEX,
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
    OPP_SIDE_START,
    OPP_BENCH_START,
    OPP_THREAT_CONFIDENCE_START,
    OPP_THREAT_OHKO_START,
    OPP_THREAT_ROW_SIZE,
    OPP_THREAT_ROWS,
    OPP_THREAT_START,
    PIVOT_MOVES,
    RECOVERY_ITEMS,
    SIDE_CONDITION_ORDER,
    SPEED_ADVANTAGE_INDEX,
    SPEED_BOOST_ITEMS,
    SPEED_DROP_ITEMS,
    STATUS_ORDER,
    TARGETING_START,
    TERRAIN_ORDER,
    TERRAIN_START,
    TRICK_ROOM_INDEX,
    TURN_INDEX,
    WEATHER_START,
    TYPE_ORDER,
    VECTOR_LENGTH,
    VOLATILE_ORDER,
    WEATHER_HEAL_MOVES,
    WEATHER_ORDER,
    OpponentThreatEntry,
    _ability_immune,
    _battle_tag,
    _clamp,
    _clamp01,
    _defender_type_mult,
    _effective_types,
    _mon_key,
    _poison_severity,
    _safe_hp_fraction,
    _stat_stage_multiplier,
    _stab_multiplier,
)

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
            # Alive count differential: (my_alive - opp_alive) / 6, range [-1, 1]
            my_alive = sum(1 for m in battle.team.values() if not m.fainted)
            opp_alive = sum(1 for m in battle.opponent_team.values() if not m.fainted)
            vector[ALIVE_DIFF_INDEX] = (my_alive - opp_alive) / 6.0
            vector[FORCE_SWITCH_INDEX] = 1.0 if getattr(battle, "force_switch", False) else 0.0
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
        # Status effect chances (Serene Grace / Sheer Force aware)
        vector[start + 19] = self._move_effect_chance(move, statuses=(Status.BRN,), attacker=attacker)
        vector[start + 20] = self._move_effect_chance(move, statuses=(Status.PAR,), attacker=attacker)
        vector[start + 21] = self._move_effect_chance(move, statuses=(Status.PSN, Status.TOX), attacker=attacker)
        vector[start + 22] = self._move_effect_chance(move, statuses=(Status.FRZ,), attacker=attacker)
        vector[start + 23] = self._move_effect_chance(move, statuses=(Status.SLP,), attacker=attacker)
        vector[start + 24] = self._move_effect_chance(move, volatile_effect=Effect.CONFUSION, attacker=attacker)
        # Status-move visibility features
        vector[start + 25] = self._move_self_delta(move, "def")
        vector[start + 26] = self._move_self_delta(move, "spd")
        vector[start + 27] = self._move_is_setup(move)
        vector[start + 28] = self._move_is_hazard(move)
        vector[start + 29] = self._move_is_recovery(move)
        # Volatile / secondary effect features (Serene Grace / Sheer Force aware)
        vector[start + 30] = self._move_effect_chance(move, volatile_effect=Effect.FLINCH, attacker=attacker)
        vector[start + 31] = self._move_target_stat_drop_chance(move, "def", attacker=attacker)
        vector[start + 32] = self._move_target_stat_drop_chance(move, "spa", attacker=attacker)
        vector[start + 33] = self._move_target_stat_drop_chance(move, "spd", attacker=attacker)
        vector[start + 34] = self._move_target_stat_drop_chance(move, "spe", attacker=attacker)
        vector[start + 35] = self._move_target_stat_drop_chance(move, "accuracy", attacker=attacker)
        # Multi-hit flag
        vector[start + 36] = 1.0 if getattr(move, "n_hit", (1, 1)) != (1, 1) else 0.0

    def _fill_my_bench(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        bench = [mon for mon in battle.team.values() if not mon.active][:5]
        opponent = battle.opponent_active_pokemon
        opp_posterior = (
            self._opponent_role_posterior(opponent) if opponent is not None else None
        )
        opp_speed = (
            self._effective_speed(opponent, battle.opponent_side_conditions, role_posterior=opp_posterior)
            if opponent is not None else None
        )
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
            # Offensive matchup features vs opponent active
            self._fill_bench_matchup(vector, start + 53, mon, battle, opponent, opp_speed, opp_posterior)
            # Status conditions (mirror active block pattern)
            status = mon.status
            vector[start + 58] = _poison_severity(status)
            vector[start + 59] = 1.0 if status == Status.PAR else 0.0
            vector[start + 60] = 1.0 if status == Status.BRN else 0.0
            vector[start + 61] = 1.0 if status == Status.SLP else 0.0
            vector[start + 62] = 1.0 if status == Status.FRZ else 0.0

    def _fill_bench_matchup(
        self,
        vector: np.ndarray,
        start: int,
        mon: Pokemon,
        battle: AbstractBattle,
        opponent: Optional[Pokemon],
        opp_speed: Optional[float],
        opp_posterior: Optional[Dict[str, float]],
    ) -> None:
        """Fill per-bench-slot matchup features vs opponent active.
        [53] best_move_ev  - best move's expected damage % vs opponent
        [54] can_ohko       - 1.0 if best move can KO opponent
        [55] outspeeds      - 1.0 if this bench mon outspeeds opponent
        [56] max_incoming   - max damage % opponent can do to this bench mon
        [57] hazard_entry   - % HP lost on switch-in from hazards
        """
        if mon.fainted or opponent is None:
            return

        # Offensive: best move EV and OHKO flag
        best_ev = 0.0
        can_ohko = False
        opp_hp = _safe_hp_fraction(opponent)
        for move in mon.moves.values():
            if self._move_category(move) == MoveCategory.STATUS:
                continue
            min_pct, max_pct = self._damage_range_percent(
                battle, mon, opponent, move,
                battle.player_role, battle.opponent_role,
            )
            ev = _clamp01(((min_pct + max_pct) / 2.0) * _clamp01(float(move.accuracy)))
            if ev > best_ev:
                best_ev = ev
            if opp_hp > 0.0 and max_pct >= opp_hp:
                can_ohko = True
        vector[start] = _clamp01(best_ev)
        vector[start + 1] = 1.0 if can_ohko else 0.0

        # Speed comparison
        mon_speed = self._effective_speed(mon, battle.side_conditions)
        if mon_speed is not None and opp_speed is not None:
            vector[start + 2] = 1.0 if mon_speed > opp_speed else 0.0
        else:
            vector[start + 2] = 0.5  # unknown

        # Defensive: max incoming damage from opponent's moves
        max_incoming = 0.0
        if opp_posterior is not None:
            threat_entries = self._select_opponent_threat_entries(opponent, opp_posterior)
            for entry in threat_entries:
                ev = self._move_expected_value(
                    battle, opponent, mon,
                    entry.move, battle.opponent_role, battle.player_role,
                )
                if ev > max_incoming:
                    max_incoming = ev
        else:
            for opp_move in opponent.moves.values():
                min_pct, max_pct = self._damage_range_percent(
                    battle, opponent, mon, opp_move,
                    battle.opponent_role, battle.player_role,
                )
                ev = _clamp01(((min_pct + max_pct) / 2.0) * _clamp01(float(opp_move.accuracy)))
                if ev > max_incoming:
                    max_incoming = ev
        vector[start + 3] = _clamp01(max_incoming)

        # Hazard entry damage
        vector[start + 4] = self._hazard_entry_damage(mon, battle.side_conditions)

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

    def _hazard_entry_damage(self, mon: Pokemon, side_conditions: Dict[SideCondition, int]) -> float:
        """Estimate fraction of HP lost when switching this mon in due to hazards."""
        # Heavy-Duty Boots block all entry hazards
        item = getattr(mon, "item", None)
        if item and item.lower().replace(" ", "").replace("-", "") == "heavydutyboots":
            return 0.0
        damage = 0.0
        # Stealth Rock: type-based (0.5x to 4x of 12.5%)
        if SideCondition.STEALTH_ROCK in side_conditions:
            rock_type = PokemonType.ROCK
            types = _effective_types(mon)
            mult = 1.0
            type_chart = self._gen_data.type_chart if hasattr(self, "_gen_data") else None
            if type_chart is not None:
                for t in types:
                    if t is not None:
                        mult *= rock_type.damage_multiplier(t, None, type_chart=type_chart)
            damage += 0.125 * mult
        # Spikes: layers (1=12.5%, 2=16.7%, 3=25%)
        spikes = side_conditions.get(SideCondition.SPIKES, 0)
        if spikes > 0:
            types = _effective_types(mon)
            is_flying = any(t == PokemonType.FLYING for t in types if t is not None)
            is_levitate = _ability_immune(mon, PokemonType.GROUND)
            if not is_flying and not is_levitate:
                spikes_dmg = {1: 0.125, 2: 0.167, 3: 0.25}
                damage += spikes_dmg.get(spikes, 0.25)
        return _clamp01(damage)

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
        # Modifiers drawn from poke_env.calc.damage_calc_gen9 but adapted for
        # hypothetical stat sets (Bayesian role-weighted calcs).
        level = attacker.level or 80
        fixed_damage = self._fixed_damage_amount(attacker, defender, move, battle)
        if fixed_damage is not None:
            return fixed_damage, fixed_damage

        bp = float(getattr(move, "base_power", 0))
        if bp == 0:
            bp = 5.0  # Min signal for status moves to differentiate types

        category = move.category
        is_body_press = move.id == "bodypress"
        hits_physical = (
            category == MoveCategory.PHYSICAL
            or getattr(move, "entry", {}).get("overrideDefensiveStat", "") == "def"
        )
        if category == MoveCategory.PHYSICAL:
            a = float(att_stats.get("def" if is_body_press else "atk") or 100)
            d = float(def_stats.get("def") or 100)
        else:
            a = float(att_stats.get("spa") or 100)
            d = float(def_stats.get("spd") or 100)

        # Foul Play uses defender's Attack
        if move.id == "foulplay":
            a = float(def_stats.get("atk") or 100)

        # Boosts
        att_ability = getattr(attacker, "ability", None) or ""
        def_ability = getattr(defender, "ability", None) or ""
        if is_body_press:
            a_boost = _stat_stage_multiplier(attacker.boosts.get("def", 0))
        elif def_ability == "unaware":
            a_boost = 1.0  # Unaware ignores offensive boosts
        else:
            a_boost = _stat_stage_multiplier(attacker.boosts.get("atk" if category == MoveCategory.PHYSICAL else "spa", 0))
        if att_ability == "unaware" or getattr(move, "ignore_defensive", False):
            d_boost = 1.0  # Unaware/Chip Away ignores defensive boosts
        else:
            d_boost = _stat_stage_multiplier(defender.boosts.get("def" if hits_physical else "spd", 0))

        a *= a_boost
        d *= d_boost

        # Hustle: +50% physical attack, already baked into stat for our mons
        if att_ability == "hustle" and category == MoveCategory.PHYSICAL:
            a *= 1.5

        # ---- Base power modifiers ----
        bp_mult = 1.0
        move_type = self._resolve_move_type(attacker, move, battle)
        try:
            move_flags = set(move.flags)
        except Exception:
            move_flags = set()

        # Ability-based BP modifiers (attacker)
        if att_ability == "technician" and bp <= 60:
            bp_mult *= 1.5
        elif att_ability == "sheerforce" and (getattr(move, "secondary", None) or move.id == "orderup"):
            bp_mult *= 1.3
        elif att_ability == "toughclaws" and "contact" in move_flags:
            bp_mult *= 1.3
        elif att_ability == "strongjaw" and "bite" in move_flags:
            bp_mult *= 1.5
        elif att_ability == "megalauncher" and "pulse" in move_flags:
            bp_mult *= 1.5
        elif att_ability == "sharpness" and "slicing" in move_flags:
            bp_mult *= 1.5
        elif att_ability == "punkrock" and "sound" in move_flags:
            bp_mult *= 1.3
        elif att_ability == "ironfist" and "punch" in move_flags:
            bp_mult *= 1.2
        elif att_ability == "reckless" and getattr(move, "recoil", 0) > 0:
            bp_mult *= 1.2
        elif att_ability == "sandforce" and Weather.SANDSTORM in battle.weather and move_type in (PokemonType.ROCK, PokemonType.GROUND, PokemonType.STEEL):
            bp_mult *= 1.3
        # -ate abilities: 1.2x BP boost (check original type, not resolved move_type)
        if att_ability in ("pixilate", "aerilate", "refrigerate", "galvanize") and move.type == PokemonType.NORMAL and move.category != MoveCategory.STATUS:
            bp_mult *= 1.2

        # Terrain BP boosts (grounded attacker)
        is_grounded_att = not (PokemonType.FLYING in getattr(attacker, "types", ()) or att_ability == "levitate")
        if is_grounded_att:
            if Field.ELECTRIC_TERRAIN in battle.fields and move_type == PokemonType.ELECTRIC:
                bp_mult *= 1.3
            elif Field.GRASSY_TERRAIN in battle.fields and move_type == PokemonType.GRASS:
                bp_mult *= 1.3
            elif Field.PSYCHIC_TERRAIN in battle.fields and move_type == PokemonType.PSYCHIC:
                bp_mult *= 1.3

        # Terrain BP reductions (grounded defender)
        is_grounded_def = not (PokemonType.FLYING in getattr(defender, "types", ()) or def_ability == "levitate")
        if is_grounded_def:
            if Field.MISTY_TERRAIN in battle.fields and move_type == PokemonType.DRAGON:
                bp_mult *= 0.5
            elif Field.GRASSY_TERRAIN in battle.fields and move.id in ("bulldoze", "earthquake"):
                bp_mult *= 0.5

        # Move-specific BP doublings
        if (
            (move.id == "facade" and attacker.status in (Status.BRN, Status.PAR, Status.PSN, Status.TOX))
            or (move.id == "brine" and _safe_hp_fraction(defender) <= 0.5)
            or (move.id == "venoshock" and defender.status in (Status.PSN, Status.TOX))
            or (move.id == "hex" and defender.status is not None)
        ):
            bp_mult *= 2.0
        elif move.id == "knockoff" and defender.item is not None:
            bp_mult *= 1.5

        # Dry Skin takes 25% extra from Fire
        if def_ability == "dryskin" and move_type == PokemonType.FIRE:
            bp_mult *= 1.25

        bp *= bp_mult

        # ---- Attack stat modifiers ----
        atk_mult = 1.0
        # Choice items
        att_item = getattr(attacker, "item", None) or ""
        if att_item == "choiceband" and category == MoveCategory.PHYSICAL:
            atk_mult *= 1.5
        elif att_item == "choicespecs" and category == MoveCategory.SPECIAL:
            atk_mult *= 1.5

        # Guts overrides burn penalty and boosts physical
        if att_ability == "guts" and attacker.status is not None and category == MoveCategory.PHYSICAL:
            atk_mult *= 1.5
        # Pinch abilities (Overgrow, Blaze, Torrent, Swarm)
        elif _safe_hp_fraction(attacker) <= 1.0 / 3:
            if (att_ability == "overgrow" and move_type == PokemonType.GRASS) or \
               (att_ability == "blaze" and move_type == PokemonType.FIRE) or \
               (att_ability == "torrent" and move_type == PokemonType.WATER) or \
               (att_ability == "swarm" and move_type == PokemonType.BUG):
                atk_mult *= 1.5

        # Flash Fire
        if att_ability == "flashfire" and Effect.FLASH_FIRE in getattr(attacker, "effects", {}) and move_type == PokemonType.FIRE:
            atk_mult *= 1.5

        # Huge Power / Pure Power
        if att_ability in ("hugepower", "purepower") and category == MoveCategory.PHYSICAL:
            atk_mult *= 2.0

        # Water Bubble (water moves)
        if att_ability == "waterbubble" and move_type == PokemonType.WATER:
            atk_mult *= 2.0

        # Solar Power (special in sun)
        if att_ability == "solarpower" and Weather.SUNNYDAY in battle.weather and category == MoveCategory.SPECIAL:
            atk_mult *= 1.5

        # Gorilla Tactics (physical)
        if att_ability == "gorillatactics" and category == MoveCategory.PHYSICAL:
            atk_mult *= 1.5

        # Type-boosting abilities (Steelworker, Dragon's Maw, Rocky Payload)
        if (att_ability == "steelworker" and move_type == PokemonType.STEEL) or \
           (att_ability == "dragonsmaw" and move_type == PokemonType.DRAGON) or \
           (att_ability == "rockypayload" and move_type == PokemonType.ROCK):
            atk_mult *= 1.5
        elif att_ability == "transistor" and move_type == PokemonType.ELECTRIC:
            atk_mult *= 1.3

        # Thick Fat (defender, reduces Fire/Ice attack)
        if def_ability == "thickfat" and move_type in (PokemonType.FIRE, PokemonType.ICE):
            atk_mult *= 0.5
        # Water Bubble (defender, reduces Fire attack)
        elif def_ability == "waterbubble" and move_type == PokemonType.FIRE:
            atk_mult *= 0.5
        # Heatproof
        elif def_ability == "heatproof" and move_type == PokemonType.FIRE:
            atk_mult *= 0.5

        a *= atk_mult

        # ---- Defense stat modifiers ----
        def_mult = 1.0
        def_item = getattr(defender, "item", None) or ""

        # Sandstorm SpD boost for Rock types
        if Weather.SANDSTORM in battle.weather and PokemonType.ROCK in getattr(defender, "types", ()) and not hits_physical:
            def_mult *= 1.5
        # Snow Def boost for Ice types
        if Weather.SNOW in battle.weather and PokemonType.ICE in getattr(defender, "types", ()) and hits_physical:
            def_mult *= 1.5

        # Eviolite
        if def_item == "eviolite":
            def_mult *= 1.5
        # Assault Vest (SpD only)
        elif def_item == "assaultvest" and not hits_physical:
            def_mult *= 1.5

        # Fur Coat (physical defense doubled)
        if def_ability == "furcoat" and hits_physical:
            def_mult *= 2.0
        # Ice Scales (special defense halved incoming)
        elif def_ability == "icescales" and not hits_physical:
            def_mult *= 2.0
        # Marvel Scale
        elif def_ability == "marvelscale" and defender.status is not None and hits_physical:
            def_mult *= 1.5

        d *= def_mult

        # ---- Base damage ----
        base_damage = (((2 * level / 5 + 2) * bp * a / d) / 50) + 2

        # ---- Final modifiers ----
        type_mult = _defender_type_mult(move_type, defender, self._gen_data.type_chart)
        stab = _stab_multiplier(attacker, move_type)

        # Adaptability (already partially handled in _stab_multiplier for tera,
        # but non-tera adaptability gives 2.0x STAB)
        if att_ability == "adaptability" and not getattr(attacker, "is_terastallized", False):
            if move_type in attacker.types:
                stab = 2.0

        # Weather
        weather_mult = 1.0
        att_umbrella = att_item == "utilityumbrella"
        def_umbrella = def_item == "utilityumbrella"
        if not def_umbrella:
            if move_type == PokemonType.FIRE:
                if Weather.SUNNYDAY in battle.weather:
                    weather_mult = 1.5
                elif Weather.RAINDANCE in battle.weather:
                    weather_mult = 0.5
            elif move_type == PokemonType.WATER:
                if Weather.RAINDANCE in battle.weather:
                    weather_mult = 1.5
                elif Weather.SUNNYDAY in battle.weather:
                    weather_mult = 0.5
        # Hydro Steam special case
        if move.id == "hydrosteam" and Weather.SUNNYDAY in battle.weather and not att_umbrella:
            weather_mult = 1.5

        # Burn (Guts prevents burn penalty)
        burn_mult = 1.0
        if attacker.status == Status.BRN and category == MoveCategory.PHYSICAL:
            if att_ability != "guts" and move.id != "facade":
                burn_mult = 0.5

        # Screens
        screen_mult = 1.0
        def_side = getattr(battle, "side_conditions", {}) if defender == battle.active_pokemon else getattr(battle, "opponent_side_conditions", {})
        if SideCondition.AURORA_VEIL in def_side:
            screen_mult = 0.5
        elif SideCondition.REFLECT in def_side and hits_physical:
            screen_mult = 0.5
        elif SideCondition.LIGHT_SCREEN in def_side and not hits_physical:
            screen_mult = 0.5

        # Solid Rock / Filter / Prism Armor
        if def_ability in ("solidrock", "filter", "prismarmor") and type_mult > 1.0:
            type_mult *= 0.75

        # Multiscale / Shadow Shield (full HP)
        if def_ability in ("multiscale", "shadowshield") and _safe_hp_fraction(defender) >= 1.0:
            screen_mult *= 0.5

        # Tinted Lens (attacker, resisted moves hit harder)
        tinted = 1.0
        if att_ability == "tintedlens" and type_mult < 1.0:
            tinted = 2.0
        # Neuroforce (super-effective bonus)
        elif att_ability == "neuroforce" and type_mult > 1.0:
            tinted = 1.25

        # Item final modifiers
        item_mult = 1.0
        if att_item == "lifeorb":
            item_mult = 1.3
        elif att_item == "expertbelt" and type_mult > 1.0:
            item_mult = 1.2

        total_mult = type_mult * stab * weather_mult * burn_mult * screen_mult * tinted * item_mult

        final_min = base_damage * total_mult * 0.85
        final_max = base_damage * total_mult * 1.0

        # Multi-hit moves: scale by expected number of hits
        expected_hits = getattr(move, "expected_hits", 1) or 1
        if expected_hits > 1:
            final_min *= expected_hits
            final_max *= expected_hits

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
            return attacker.tera_type or move.type
        # -ate abilities: Normal → ability type
        _ATE_MAP = {
            "pixilate": PokemonType.FAIRY,
            "aerilate": PokemonType.FLYING,
            "refrigerate": PokemonType.ICE,
            "galvanize": PokemonType.ELECTRIC,
        }
        att_ability = (getattr(attacker, "ability", None) or "").lower().replace(" ", "")
        if move.type == PokemonType.NORMAL and att_ability in _ATE_MAP and move.category != MoveCategory.STATUS:
            return _ATE_MAP[att_ability]
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

    # ---------- Status-move visibility helpers (move block 25-29) ----------

    _SETUP_MOVES: set[str] = {
        "swordsdance", "nastyplot", "dragondance", "calmmind", "quiverdance",
        "shellsmash", "irondefense", "amnesia", "bulkup", "coil",
        "honeclaws", "workup", "tidyup", "tailglow", "growth",
        "agility", "autotomize", "rockpolish", "shiftgear", "cottonguard",
        "cosmicpower", "acidarmor", "barrier", "curse", "bellydrum",
        "filletaway", "victorydance", "clangoroussoul", "noretreat",
        "geomancy", "extremeevoboost",
    }
    _HAZARD_MOVES: set[str] = {
        "stealthrock", "spikes", "toxicspikes", "stickyweb", "cometshards",
    }
    _RECOVERY_MOVES: set[str] = {
        "recover", "roost", "softboiled", "moonlight", "morningsun",
        "synthesis", "slackoff", "milkdrink", "wish", "healorder",
        "shoreup", "junglehealing", "lifedew", "lunarblessing",
        "rest", "healingwish", "lunardance", "strengthsap",
    }

    def _move_is_setup(self, move: Move) -> float:
        """1.0 if the move is a stat-boosting setup move."""
        if move.id in self._SETUP_MOVES:
            return 1.0
        # Fallback: any move with net positive self-boosts and no damage
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
        if isinstance(boosts, dict) and sum(boosts.values()) > 0 and float(getattr(move, "base_power", 0)) == 0:
            return 1.0
        return 0.0

    def _move_is_hazard(self, move: Move) -> float:
        """1.0 if the move sets entry hazards."""
        return 1.0 if move.id in self._HAZARD_MOVES else 0.0

    def _move_is_recovery(self, move: Move) -> float:
        """1.0 if the move is a dedicated recovery move."""
        if move.id in self._RECOVERY_MOVES:
            return 1.0
        # Also flag moves with significant self-healing and no damage
        heal = float(getattr(move, "heal", 0.0) or 0.0)
        if heal >= 0.25 and float(getattr(move, "base_power", 0)) == 0:
            return 1.0
        return 0.0

    def _move_effect_chance(
        self,
        move: Move,
        *,
        statuses: Tuple[Status, ...] = (),
        volatile_effect: Optional[Effect] = None,
        attacker: Optional[Pokemon] = None,
    ) -> float:
        att_ability = (getattr(attacker, "ability", None) or "").lower().replace(" ", "") if attacker else ""
        is_serene = att_ability == "serenegrace"
        is_sheer = att_ability == "sheerforce"

        best = 0.0

        # Direct status/volatile (100% guaranteed, not secondary — unaffected by abilities)
        direct_status = getattr(move, "status", None)
        if direct_status in statuses:
            best = 1.0

        entry = getattr(move, "entry", {}) or {}
        if volatile_effect is not None and self._entry_effect(entry.get("volatileStatus")) == volatile_effect:
            best = 1.0

        # Sheer Force removes secondary effects entirely (but not direct status above)
        if is_sheer:
            return best

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
            if is_serene:
                chance = min(chance * 2.0, 1.0)
            if self._entry_status(secondary.get("status")) in statuses:
                best = max(best, chance)
            if volatile_effect is not None and self._entry_effect(secondary.get("volatileStatus")) == volatile_effect:
                best = max(best, chance)
        return best

    def _move_target_stat_drop_chance(
        self,
        move: Move,
        stat: str,
        attacker: Optional[Pokemon] = None,
    ) -> float:
        """Return the chance (0-1) that this move drops the target's stat."""
        att_ability = (getattr(attacker, "ability", None) or "").lower().replace(" ", "") if attacker else ""
        if att_ability == "sheerforce":
            return 0.0

        is_serene = att_ability == "serenegrace"
        entry = getattr(move, "entry", {}) or {}

        secondary_effects = getattr(move, "secondary", None)
        if not isinstance(secondary_effects, list):
            if isinstance(entry.get("secondary"), dict):
                secondary_effects = [entry["secondary"]]
            elif isinstance(entry.get("secondaries"), list):
                secondary_effects = entry["secondaries"]
            else:
                secondary_effects = []

        best = 0.0
        for secondary in secondary_effects:
            if not isinstance(secondary, dict):
                continue
            boosts = secondary.get("boosts")
            if not isinstance(boosts, dict):
                continue
            drop = boosts.get(stat, 0)
            if drop < 0:  # negative = target stat drop
                chance = _clamp01(float(secondary.get("chance", 100)) / 100.0)
                if is_serene:
                    chance = min(chance * 2.0, 1.0)
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


