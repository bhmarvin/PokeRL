# Reward Function

## Summary
The environment reward is the sum of:

1. `poke_env`'s state-delta reward from `reward_computing_helper(...)`
2. Tactical shaping from positive and negative lever evaluators
3. Head hunter bonus for KOing high-threat opponent mons

The reward entry point is `BrentsRLAgent.calc_reward(battle)`.

## Current Config
```python
REWARD_CONFIG = {
    "fainted_value": 1.0,
    "hp_value": 0.5,
    "status_value": 0.25,
    "victory_value": 15.0,
    "penalty_redundant_stealthrock": -0.05,
    "penalty_redundant_stickyweb": -0.05,
    "penalty_redundant_spikes": -0.05,
    "penalty_redundant_status": -0.05,
    "penalty_bad_encore": -0.05,
    "penalty_ineffective_heal": -0.05,
    "penalty_wasteful_heal_overflow": -0.025,
    "penalty_redundant_self_drop_move": -0.05,
    "penalty_unsafe_stay_in_with_fast_ko_switch": -0.1,
    "bonus_good_heal_timing": 0.2,
    "bonus_good_attack_selection": 0.0,
    "bonus_good_safe_switch": 0.2,
    "bonus_good_tera": 0.5,
    "penalty_abandon_boosted_mon": -0.05,
    "penalty_heal_satiation": -0.05,
    "penalty_wasted_free_switch": 0.0,
}
```

## Base Reward
`calc_reward()` passes only the native `poke_env` keys (`fainted_value`, `hp_value`, `status_value`, `victory_value`) into `reward_computing_helper(...)`.

That helper returns the **change** in board score since the previous step, not an absolute score.

| Term | Weight | Meaning |
|------|--------|---------|
| `hp_value` | 0.5 | Delta of team HP fractions (yours minus opponent's) |
| `fainted_value` | 1.0 | +1.0 per opponent faint, -1.0 per yours |
| `status_value` | 0.25 | +0.25 when opponent gets statused, -0.25 when yours does |
| `victory_value` | 15.0 | +15.0 on win, -15.0 on loss (terminal, fires once) |

The victory value is deliberately large relative to incremental HP/faint deltas so the agent optimizes for winning rather than accumulating tactical bonuses.

## Tactical Shaping
Custom shaping applied on top of the base reward. Driven by pre-action evaluation captured in `action_to_order(...)`, consumed once in `calc_reward(...)`.

### Bonuses

**good_heal_timing (+0.2):** Recovery used when HP ≤ 60%, effective heal ≥ 20%, overflow ≤ 15%, and under pressure (HP ≤ 35% or opponent max threat ≥ 45% or OHKO risk ≥ 15%).

**good_safe_switch (+0.2):** Switch target resists all opponent threatening moves (type mult ≤ 0.5), isn't itself threatened (max incoming ≤ 45%, OHKO risk ≤ 15%), has a credible reply, and improves board position by ≥ 0.15 on threat or OHKO risk.

**good_tera (+0.5):** Terastallizing gains a defensive immunity to the opponent's last move type, or enables a KO by upgrading STAB past the opponent's remaining HP.

**head_hunter_bonus (+0.25/teammate, capped +0.75):** Extra reward for KOing an opponent mon that could OHKO multiple teammates. Scales with how many of your team it threatened.

### Penalties

**unsafe_stay_in (-0.1):** Attacking when outsped for KO and a safe switch exists that is faster, resists all threats, and can KO the opponent.

**abandon_boosted_mon (-0.05):** Switching out a mon with ≥+2 offensive boosts that has >50% HP and outspeeds the opponent.

**heal_satiation (-0.05):** Using a heal move 3+ consecutive turns on the same mon.

**wasted_free_switch (0.0):** Disabled. Was -0.1 for switching out a mon on the turn after it entered via forced switch, but has bad credit assignment (penalizes current decision for a previous turn's mistake).

**redundant_hazards (-0.05):** Setting Stealth Rock/Sticky Web when already up, or Spikes at 3 layers.

**redundant_status (-0.05):** Using a status move on an already-statused opponent.

**redundant_self_drop (-0.05):** Using a self-stat-dropping move when the stat is already at -2 or lower, the move can't KO, and a better alternative exists.

**Other:** bad_encore (-0.05), ineffective_heal (-0.05), wasteful_heal_overflow (-0.025).

## Design Notes
- Tactical shaping is based on **pre-action state**, not post-action state.
- Penalties are intentionally small (half their original values) to prevent the agent from learning to avoid penalties rather than learning to win.
- The pos/neg shaping ratio should stay close to 1:1 during healthy training. If negatives dominate 3:1+, the agent becomes risk-averse.
- `wasted_free_switch` is kept at 0.0 (audit-only) because it requires multi-turn credit assignment that confuses early training.
- The environment exposes `get_strategic_penalty_report()` and `get_tactical_shaping_report()` for monitoring shaping health.

## Main Symbols
```python
def calc_reward(self, battle: AbstractBattle) -> float
def action_to_order(self, action: np.int64, battle: AbstractBattle, fake: bool = False, strict: bool = True) -> BattleOrder
def get_strategic_penalty_report(self) -> dict[str, Any]
def get_tactical_shaping_report(self) -> dict[str, Any]
```
