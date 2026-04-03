from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.effect import Effect
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.battle_order import BattleOrder
from poke_env.ps_client import AccountConfiguration

from .constants import (
    DECISION_AUDIT_SAMPLE_LIMIT,
    MOVE_BLOCK_SIZE,
    MY_MOVES_START,
    OPP_THREAT_CONFIDENCE_START,
    OPP_THREAT_ROWS,
    POKE_ENV_REWARD_KEYS,
    REWARD_CONFIG,
    SPEED_ADVANTAGE_INDEX,
    VECTOR_LENGTH,
    WEATHER_HEAL_MOVES,
    OpponentThreatEntry,
    TacticalLeverMatch,
    TacticalRewardContext,
    ThreatAssessment,
    _battle_tag,
    _clamp01,
    _defender_type_mult,
    _effective_types,
    _mon_key,
    _poison_severity,
    _safe_hp_fraction,
    _safe_identifier,
    _stab_multiplier,
    _stat_stage_multiplier,
    _ABILITY_TYPE_IMMUNITIES,
    _ability_immune,
)
from .observation import BrentObservationVectorBuilder

class BrentsRLAgent(SinglesEnv):
    def __init__(self, *args, **kwargs) -> None:
        import os, random, string
        _uid = f"{os.getpid()}{''.join(random.choices(string.ascii_lowercase, k=4))}"
        kwargs.setdefault(
            "account_configuration1",
            AccountConfiguration.generate(f"RL{_uid}", rand=False),
        )
        kwargs.setdefault(
            "account_configuration2",
            AccountConfiguration.generate(f"RLO{_uid}", rand=False),
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
        self._entered_after_faint: Dict[str, int] = {}
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

    @staticmethod
    def get_action_mask(battle) -> list[int]:
        """Override poke_env's get_action_mask to fix the 5-move collision bug.

        The upstream implementation iterates over active_pokemon.moves (all known
        moves, can exceed 4) instead of checking against available_moves.
        When a mon knows 5+ moves, move indices spill into the mega zone (10-13).

        Fix: iterate active_pokemon.moves (to keep index mapping consistent with
        action_to_order) but only mark a move as legal if its id is in
        available_moves AND its index is < 4.
        """
        from poke_env.environment.singles_env import SPECIAL_MOVES

        switch_space = [
            i
            for i, pokemon in enumerate(battle.team.values())
            if not battle.trapped
            and pokemon.base_species
            in [p.base_species for p in battle.available_switches]
        ]

        if battle._wait:
            actions = [0]
        elif battle.active_pokemon is None:
            actions = switch_space
        else:
            # Use active_pokemon.moves for index mapping (matches action_to_order)
            # but filter to only available moves AND cap at index 4
            available_ids = {m.id for m in battle.available_moves}
            move_space = [
                i + 6
                for i, move in enumerate(battle.active_pokemon.moves.values())
                if move.id in available_ids and i < 4
            ]

            mega_space = [i + 4 for i in move_space if battle.can_mega_evolve]
            # Z-moves
            zmove_space = []
            if battle.can_z_move and hasattr(battle.active_pokemon, "available_z_moves"):
                avail_z_ids = {m.id for m in battle.active_pokemon.available_z_moves}
                for i, move in enumerate(battle.active_pokemon.moves.values()):
                    if i < 4 and move.id in avail_z_ids:
                        zmove_space.append(6 + i + 8)
            dynamax_space = [i + 12 for i in move_space if battle.can_dynamax]
            tera_space = [i + 16 for i in move_space if battle.can_tera]

            if (
                not move_space
                and len(battle.available_moves) == 1
                and battle.available_moves[0].id in SPECIAL_MOVES
            ):
                move_space = [6]

            actions = (
                switch_space
                + move_space
                + mega_space
                + zmove_space
                + dynamax_space
                + tera_space
            )

        action_mask = [
            int(i in actions)
            for i in range(SinglesEnv.get_action_space_size(battle.gen))
        ]
        return action_mask

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
            (
                "good_setup",
                "bonus_good_setup",
                self._evaluate_good_setup(battle, move),
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
            (
                "pivot_into_advantage",
                "bonus_pivot_into_advantage",
                self._evaluate_pivot_into_advantage(battle, switch_target),
                False,
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
            self._entered_after_faint[active_species] = battle.turn
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

        entry_turn = self._entered_after_faint.pop(species, None)
        if entry_turn is None:
            return None

        # Only penalize if switching out on the very next turn after free entry
        if battle.turn > entry_turn + 1:
            return None

        return {
            "species": species,
            "entry_turn": entry_turn,
            "switch_turn": battle.turn,
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

        if best_alt_ev < chosen_ev:
            return None  # self-drop move is still the best damaging option

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

    def _evaluate_good_setup(
        self,
        battle: AbstractBattle,
        move: Move,
    ) -> Optional[Dict[str, Any]]:
        """Reward using a stat-boosting setup move when safe to do so:
        HP > 60%, not outsped for KO, and in a favorable or neutral matchup."""
        if self.vector_builder._move_is_setup(move) < 1.0:
            return None

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return None

        hp = _safe_hp_fraction(active)
        if hp <= 0.6:
            return None

        # Check we're not about to be KO'd
        threat = self._assess_battle_threats(battle)
        if threat.active_max_threat >= 0.8 or threat.active_ohko_risk >= 0.3:
            return None

        # Check speed: either we outspeed or we're bulky enough to take a hit
        outspeeds = (
            threat.active_speed is not None
            and threat.opponent_speed is not None
            and threat.active_speed > threat.opponent_speed
        )
        bulky_enough = hp > 0.8 and threat.active_max_threat < 0.5

        if not outspeeds and not bulky_enough:
            return None

        return {
            "move": move.id,
            "hp_fraction": round(hp, 3),
            "max_threat": round(threat.active_max_threat, 3),
            "outspeeds": outspeeds,
        }

    def _evaluate_pivot_into_advantage(
        self,
        battle: AbstractBattle,
        switch_target: Pokemon,
    ) -> Optional[Dict[str, Any]]:
        """Reward switching via a pivot move (U-turn/Volt Switch/Flip Turn)
        into a mon that resists the opponent's STAB and has a credible reply.
        Only fires when the PREVIOUS move was a pivot move."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return None

        # Check if active just used a pivot move
        last_move = active.last_move if hasattr(active, "last_move") else None
        if last_move is None:
            return None
        is_pivot = last_move.id in ("uturn", "voltswitch", "flipturn", "teleport")
        if not is_pivot:
            # Also check self_switch flag
            try:
                is_pivot = bool(last_move.self_switch)
            except Exception:
                pass
        if not is_pivot:
            return None

        # Check switch target resists opponent's STAB types
        opp_types = _effective_types(opponent)
        type_chart = self.vector_builder._gen_data.type_chart
        resists_stab = all(
            _defender_type_mult(opp_type, switch_target, type_chart) <= 0.5
            for opp_type in opp_types
            if opp_type is not None
        )
        if not resists_stab:
            return None

        # Check switch target isn't immediately threatened
        threat = self._assess_battle_threats(battle)
        # Estimate threat to switch target
        max_incoming = 0.0
        for entry in threat.threat_entries:
            ev = self.vector_builder._move_expected_value(
                battle, opponent, switch_target, entry.move,
                battle.opponent_role, battle.player_role,
            )
            if ev > max_incoming:
                max_incoming = ev
        if max_incoming > 0.45:
            return None

        return {
            "pivot_move": last_move.id,
            "switch_target": getattr(switch_target, "species", "?"),
            "resists_stab": True,
            "max_incoming": round(max_incoming, 3),
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
                "self_def_delta": round(float(block[25]), 3),
                "self_spd_delta": round(float(block[26]), 3),
                "is_setup": int(block[27]),
                "is_hazard": int(block[28]),
                "is_recovery": int(block[29]),
                "flinch": round(float(block[30]), 3),
                "tgt_def_drop": round(float(block[31]), 3),
                "tgt_spa_drop": round(float(block[32]), 3),
                "tgt_spd_drop": round(float(block[33]), 3),
                "tgt_spe_drop": round(float(block[34]), 3),
                "tgt_acc_drop": round(float(block[35]), 3),
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


