"""Observation vector encoding: BrentObservationVectorBuilder."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

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
from poke_env.calc.damage_calc_gen9 import get_item_boost_type

from .constants import (
    BENCH_MOVE_FLAG_SIZE,
    BOOST_ORDER,
    DAMAGE_BOOST_ITEMS,
    MOVE_BLOCK_SIZE,
    MY_ACTIVE_BLOCK_SIZE,
    MY_ACTIVE_START,
    MY_BENCH_SLOT_SIZE,
    MY_BENCH_START,
    MY_MOVES_START,
    MY_TEAM_REVEALED_START,
    ON_RECHARGE_INDEX,
    OPP_ACTIVE_START,
    OPP_ACTIVE_BLOCK_SIZE,
    OPP_BENCH_SLOT_SIZE,
    OPP_BENCH_START,
    OPP_THREAT_CONFIDENCE_START,
    OPP_THREAT_OHKO_START,
    OPP_THREAT_ROW_SIZE,
    OPP_THREAT_ROWS,
    OPP_THREAT_START,
    ALIVE_DIFF_INDEX,
    FORCE_SWITCH_INDEX,
    RECOVERY_ITEMS,
    SPEED_ADVANTAGE_INDEX,
    SPEED_BOOST_ITEMS,
    STATUS_ORDER,
    TARGETING_START,
    TERRAIN_ORDER,
    TRICK_ROOM_INDEX,
    TURN_INDEX,
    TYPE_ORDER,
    VECTOR_LENGTH,
    VOLATILE_ORDER,
    WEATHER_ORDER,
    MY_SIDE_START,
    OPP_SIDE_START,
    OPP_BENCH_SLOT_SIZE,
    battle_tag,
    clamp,
    clamp01,
    effective_types,
    mon_key,
    poison_severity,
    safe_hp_fraction,
)
from .mechanics import PokemonMechanics


class BrentObservationVectorBuilder:
    def __init__(self) -> None:
        self._my_team_revealed_memory: Dict[str, set[str]] = {}
        self.mechanics = PokemonMechanics()

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        self.mechanics._damage_cache = {}
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
            vector[SPEED_ADVANTAGE_INDEX] = self.mechanics.speed_advantage(battle)
            self._fill_available_move_blocks(vector, battle)
            self._fill_my_bench(vector, battle)
            self._fill_opponent_bench(vector, battle)
            self._fill_targeting_matrix(vector, battle)
            self._fill_opponent_threat_features(vector, battle)
            vector[ON_RECHARGE_INDEX] = self.mechanics.on_recharge(battle.active_pokemon)
            # Alive count differential: (my_alive - opp_alive) / 6, range [-1, 1]
            my_alive = sum(1 for m in battle.team.values() if not m.fainted)
            opp_alive = sum(1 for m in battle.opponent_team.values() if not m.fainted)
            vector[ALIVE_DIFF_INDEX] = (my_alive - opp_alive) / 6.0
            vector[FORCE_SWITCH_INDEX] = 1.0 if getattr(battle, "force_switch", False) else 0.0
            return vector
        finally:
            self.mechanics._damage_cache.clear()
            if getattr(battle, "finished", False):
                self._my_team_revealed_memory.pop(battle_tag(battle), None)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

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
        for idx, weather in enumerate(WEATHER_ORDER, start=1):
            self._verify_scalar(
                issues,
                f"weather:{weather.name}",
                vector[idx],
                1.0 if weather in battle.weather else 0.0,
            )
        for idx, terrain in enumerate(TERRAIN_ORDER, start=5):
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
            self.mechanics.speed_advantage(battle),
        )
        self._verify_scalar(
            issues,
            "on_recharge",
            vector[ON_RECHARGE_INDEX],
            self.mechanics.on_recharge(battle.active_pokemon),
        )
        self._verify_opponent_bench_leaks(issues, vector, battle)
        self._verify_opponent_threat_ranges(issues, vector, battle)
        return issues

    # ------------------------------------------------------------------
    # Fill methods
    # ------------------------------------------------------------------

    def _update_reveal_memory(self, battle: AbstractBattle) -> None:
        memory = self._my_team_revealed_memory.setdefault(battle_tag(battle), set())
        for mon in battle.team.values():
            if mon.active or mon.revealed:
                memory.add(mon_key(mon))

    def _fill_global_features(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        vector[TURN_INDEX] = float(getattr(battle, "turn", 0)) / 100.0
        for idx, weather in enumerate(WEATHER_ORDER, start=1):
            vector[idx] = 1.0 if weather in battle.weather else 0.0
        for idx, terrain in enumerate(TERRAIN_ORDER, start=5):
            vector[idx] = 1.0 if terrain in battle.fields else 0.0
        vector[TRICK_ROOM_INDEX] = 1.0 if Field.TRICK_ROOM in battle.fields else 0.0

    def _fill_side_conditions(
        self,
        vector: np.ndarray,
        start: int,
        side_conditions: Dict[SideCondition, int],
    ) -> None:
        vector[start] = 1.0 if SideCondition.STEALTH_ROCK in side_conditions else 0.0
        vector[start + 1] = clamp01(side_conditions.get(SideCondition.SPIKES, 0) / 3.0)
        vector[start + 2] = clamp01(side_conditions.get(SideCondition.TOXIC_SPIKES, 0) / 2.0)
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

        vector[start] = safe_hp_fraction(mon)
        self._fill_type_one_hot(vector, start + 1, effective_types(mon))

        for offset, stat in enumerate(BOOST_ORDER):
            vector[start + 19 + offset] = clamp(float(mon.boosts.get(stat, 0)) / 6.0, -1.0, 1.0)

        for offset, status in enumerate(STATUS_ORDER):
            vector[start + 26 + offset] = 1.0 if mon.status == status else 0.0
        vector[start + 30] = poison_severity(mon.status)

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
            or any(self.mechanics.move_heal_amount(move, battle) > 0.0 or move.drain > 0.0 for move in moves)
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

        m = self.mechanics
        min_pct, max_pct = m.damage_range_percent(
            battle, attacker, defender, move, attacker_role, defender_role,
        )
        accuracy = clamp01(float(move.accuracy))
        expected_value = clamp01(((min_pct + max_pct) / 2.0) * accuracy)
        remaining_hp = clamp01(safe_hp_fraction(defender))
        flags = m.move_flags(move)
        category = m.move_category(move)
        priority = m.move_priority(move)

        vector[start] = min_pct
        vector[start + 1] = max_pct
        vector[start + 2] = accuracy
        vector[start + 3] = expected_value
        vector[start + 4] = 1.0 if max_pct >= remaining_hp and remaining_hp > 0.0 else 0.0
        vector[start + 5] = m.is_stab(attacker, move, battle)
        vector[start + 6] = 1.0 if category == MoveCategory.PHYSICAL else 0.0
        vector[start + 7] = 1.0 if category == MoveCategory.SPECIAL else 0.0
        vector[start + 8] = 1.0 if "contact" in flags else 0.0
        vector[start + 9] = 1.0 if "sound" in flags else 0.0
        vector[start + 10] = 1.0 if priority > 0 else 0.0
        vector[start + 11] = 1.0 if m.is_pivot(move) else 0.0
        vector[start + 12] = clamp01(m.effective_move_heal_amount(move, attacker, battle))
        vector[start + 13] = clamp01(m.move_drain(move))
        vector[start + 14] = m.move_self_delta(move, "atk")
        vector[start + 15] = m.move_self_delta(move, "spa")
        vector[start + 16] = m.move_self_delta(move, "spe")
        vector[start + 17] = m.estimated_recoil_fraction(attacker, defender, move, expected_value)
        vector[start + 18] = m.move_causes_recharge(move)
        # Status effect chances (Serene Grace / Sheer Force aware)
        vector[start + 19] = m.move_effect_chance(move, statuses=(Status.BRN,), attacker=attacker)
        vector[start + 20] = m.move_effect_chance(move, statuses=(Status.PAR,), attacker=attacker)
        vector[start + 21] = m.move_effect_chance(move, statuses=(Status.PSN, Status.TOX), attacker=attacker)
        vector[start + 22] = m.move_effect_chance(move, statuses=(Status.FRZ,), attacker=attacker)
        vector[start + 23] = m.move_effect_chance(move, statuses=(Status.SLP,), attacker=attacker)
        vector[start + 24] = m.move_effect_chance(move, volatile_effect=Effect.CONFUSION, attacker=attacker)
        # Status-move visibility features
        vector[start + 25] = m.move_self_delta(move, "def")
        vector[start + 26] = m.move_self_delta(move, "spd")
        vector[start + 27] = m.move_is_setup(move)
        vector[start + 28] = m.move_is_hazard(move)
        vector[start + 29] = m.move_is_recovery(move)
        # Volatile / secondary effect features (Serene Grace / Sheer Force aware)
        vector[start + 30] = m.move_effect_chance(move, volatile_effect=Effect.FLINCH, attacker=attacker)
        vector[start + 31] = m.move_target_stat_drop_chance(move, "def", attacker=attacker)
        vector[start + 32] = m.move_target_stat_drop_chance(move, "spa", attacker=attacker)
        vector[start + 33] = m.move_target_stat_drop_chance(move, "spd", attacker=attacker)
        vector[start + 34] = m.move_target_stat_drop_chance(move, "spe", attacker=attacker)
        vector[start + 35] = m.move_target_stat_drop_chance(move, "accuracy", attacker=attacker)
        # Multi-hit flag
        vector[start + 36] = 1.0 if getattr(move, "n_hit", (1, 1)) != (1, 1) else 0.0

    def _fill_my_bench(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        bench = [mon for mon in battle.team.values() if not mon.active][:5]
        opponent = battle.opponent_active_pokemon
        m = self.mechanics
        opp_posterior = (
            m.opponent_role_posterior(opponent) if opponent is not None else None
        )
        opp_speed = (
            m.effective_speed(opponent, battle.opponent_side_conditions, role_posterior=opp_posterior)
            if opponent is not None else None
        )
        for slot, mon in enumerate(bench):
            start = MY_BENCH_START + slot * MY_BENCH_SLOT_SIZE
            vector[start] = 0.0 if mon.fainted else 1.0
            vector[start + 1] = safe_hp_fraction(mon)
            self._fill_type_one_hot(vector, start + 2, effective_types(mon))
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
            vector[start + 58] = poison_severity(status)
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
        if mon.fainted or opponent is None:
            return

        m = self.mechanics
        # Offensive: best move EV and OHKO flag
        best_ev = 0.0
        can_ohko = False
        opp_hp = safe_hp_fraction(opponent)
        for move in mon.moves.values():
            if m.move_category(move) == MoveCategory.STATUS:
                continue
            min_pct, max_pct = m.damage_range_percent(
                battle, mon, opponent, move,
                battle.player_role, battle.opponent_role,
            )
            ev = clamp01(((min_pct + max_pct) / 2.0) * clamp01(float(move.accuracy)))
            if ev > best_ev:
                best_ev = ev
            if opp_hp > 0.0 and max_pct >= opp_hp:
                can_ohko = True
        vector[start] = clamp01(best_ev)
        vector[start + 1] = 1.0 if can_ohko else 0.0

        # Speed comparison
        mon_speed = m.effective_speed(mon, battle.side_conditions)
        if mon_speed is not None and opp_speed is not None:
            vector[start + 2] = 1.0 if mon_speed > opp_speed else 0.0
        else:
            vector[start + 2] = 0.5  # unknown

        # Defensive: max incoming damage from opponent's moves
        max_incoming = 0.0
        if opp_posterior is not None:
            threat_entries = m.select_opponent_threat_entries(opponent, opp_posterior)
            for entry in threat_entries:
                ev = m.move_expected_value(
                    battle, opponent, mon,
                    entry.move, battle.opponent_role, battle.player_role,
                )
                if ev > max_incoming:
                    max_incoming = ev
        else:
            for opp_move in opponent.moves.values():
                min_pct, max_pct = m.damage_range_percent(
                    battle, opponent, mon, opp_move,
                    battle.opponent_role, battle.player_role,
                )
                ev = clamp01(((min_pct + max_pct) / 2.0) * clamp01(float(opp_move.accuracy)))
                if ev > max_incoming:
                    max_incoming = ev
        vector[start + 3] = clamp01(max_incoming)

        # Hazard entry damage
        vector[start + 4] = m.hazard_entry_damage(mon, battle.side_conditions)

    def _fill_bench_move_flags(
        self,
        vector: np.ndarray,
        start: int,
        move: Move,
        battle: AbstractBattle,
    ) -> None:
        m = self.mechanics
        flags = m.move_flags(move)
        category = m.move_category(move)
        priority = m.move_priority(move)
        vector[start] = 1.0 if category == MoveCategory.PHYSICAL else 0.0
        vector[start + 1] = 1.0 if category == MoveCategory.SPECIAL else 0.0
        vector[start + 2] = 1.0 if "contact" in flags else 0.0
        vector[start + 3] = 1.0 if "sound" in flags else 0.0
        vector[start + 4] = 1.0 if priority > 0 else 0.0
        vector[start + 5] = 1.0 if m.is_pivot(move) else 0.0
        vector[start + 6] = clamp01(m.effective_move_heal_amount(move, None, battle))
        vector[start + 7] = clamp01(m.move_drain(move))

    def _fill_opponent_bench(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        bench = [mon for mon in battle.opponent_team.values() if not mon.active][:5]
        for slot, mon in enumerate(bench):
            if not mon.revealed:
                continue
            start = OPP_BENCH_START + slot * OPP_BENCH_SLOT_SIZE
            vector[start] = 1.0
            vector[start + 1] = safe_hp_fraction(mon)
            self._fill_type_one_hot(vector, start + 2, effective_types(mon))

    def _fill_targeting_matrix(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        attacker = battle.active_pokemon
        if attacker is None:
            return

        m = self.mechanics
        opponent_bench = [mon for mon in battle.opponent_team.values() if not mon.active][:5]
        my_moves = tuple(battle.available_moves)[:4]
        for move_idx, move in enumerate(my_moves):
            for target_idx, target in enumerate(opponent_bench):
                if not target.revealed or target.fainted:
                    continue
                min_pct, max_pct = m.damage_range_percent(
                    battle, attacker, target, move,
                    battle.player_role, battle.opponent_role,
                )
                ev = clamp01(((min_pct + max_pct) / 2.0) * clamp01(float(move.accuracy)))
                vector[TARGETING_START + move_idx * 5 + target_idx] = ev

    def _fill_opponent_threat_features(self, vector: np.ndarray, battle: AbstractBattle) -> None:
        my_team_order = self._my_team_order(battle)
        memory = self._my_team_revealed_memory.get(battle_tag(battle), set())
        for idx, mon in enumerate(my_team_order):
            if mon is None:
                continue
            vector[MY_TEAM_REVEALED_START + idx] = 1.0 if mon_key(mon) in memory else 0.0

        opponent = battle.opponent_active_pokemon
        if opponent is None:
            return

        m = self.mechanics
        posterior = m.opponent_role_posterior(opponent)
        threat_entries = m.select_opponent_threat_entries(opponent, posterior)
        for move_idx, entry in enumerate(threat_entries):
            row_start = OPP_THREAT_START + move_idx * OPP_THREAT_ROW_SIZE
            vector[row_start] = entry.move_prob
            vector[row_start + 1] = entry.revealed_flag
            for target_idx, target in enumerate(my_team_order):
                if target is None or target.fainted:
                    continue
                vector[row_start + 2 + target_idx] = m.move_expected_value(
                    battle, opponent, target, entry.move,
                    battle.opponent_role, battle.player_role,
                )
        for target_idx, target in enumerate(my_team_order):
            if target is None or target.fainted:
                continue
            vector[OPP_THREAT_OHKO_START + target_idx] = m.estimate_ohko_risk(
                battle, opponent, target, posterior,
            )
        top_role_mass, role_entropy_norm = m.role_posterior_summary(posterior)
        vector[OPP_THREAT_CONFIDENCE_START] = top_role_mass
        vector[OPP_THREAT_CONFIDENCE_START + 1] = role_entropy_norm

    def _my_team_order(self, battle: AbstractBattle) -> Tuple[Optional[Pokemon], ...]:
        bench = [mon for mon in battle.team.values() if not mon.active][:5]
        ordered: list[Optional[Pokemon]] = [battle.active_pokemon]
        ordered.extend(bench)
        while len(ordered) < 6:
            ordered.append(None)
        return tuple(ordered[:6])

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

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
            clamp01(side_conditions.get(SideCondition.SPIKES, 0) / 3.0),
            clamp01(side_conditions.get(SideCondition.TOXIC_SPIKES, 0) / 2.0),
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

        self._verify_scalar(issues, f"{prefix}.hp", vector[start], safe_hp_fraction(mon))
        type_set = set(effective_types(mon))
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
                clamp(float(mon.boosts.get(stat, 0)) / 6.0, -1.0, 1.0),
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
            poison_severity(mon.status),
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

    def get_fallback_report(self) -> dict[str, Any]:
        return self.mechanics.get_fallback_report()
