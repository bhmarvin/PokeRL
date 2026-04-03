# Observation Tensor

## Summary
The policy consumes a single flat observation vector of length `780`.

The raw tensor is built by `BrentObservationVectorBuilder.embed_battle(battle)` and is exposed to PPO as:

```python
{
    "observation": np.ndarray(shape=(780,), dtype=np.float32),
    "action_mask": np.ndarray(shape=(action_space,), dtype=np.int64),
}
```

All features are pre-normalized to [0, 1] (or [-1, 1] for stat boosts). No runtime normalization layer is applied — the extractor receives raw values directly.

## Global Layout
Index ranges:

| Indices | Size | Block |
|---------|------|-------|
| 0 | 1 | Turn number / 100 |
| 1–4 | 4 | Weather one-hot (Rain, Sun, Sand, Snow) |
| 5–8 | 4 | Terrain one-hot (Electric, Grassy, Psychic, Misty) |
| 9 | 1 | Trick Room flag |
| 10–16 | 7 | My side conditions |
| 17–23 | 7 | Opponent side conditions |
| 24–83 | 60 | My active Pokemon block |
| 84–124 | 41 | Opponent active Pokemon block |
| 125 | 1 | Speed advantage flag |
| 126–277 | 152 | My available move blocks (4 × 38) |
| 278–592 | 315 | My bench blocks (5 × 63) |
| 593–692 | 100 | Opponent bench blocks (5 × 20) |
| 693–712 | 20 | Targeting matrix (4 moves × 5 bench targets) |
| 713–718 | 6 | My team reveal-memory flags |
| 719–750 | 32 | Opponent threat rows (4 × 8) |
| 751–756 | 6 | Opponent OHKO risk vs my team (6 slots) |
| 757–758 | 2 | Role posterior confidence summary |
| 759 | 1 | On-recharge flag |
| 760 | 1 | Alive count differential |
| 761 | 1 | Force switch flag |
| 762 | 1 | Speed ratio (continuous) |
| 763 | 1 | My tailwind flag |
| 764 | 1 | Opponent tailwind flag |
| 765–779 | 15 | Opponent ability one-hot |

## Side Conditions Block
Each side (my and opponent) has 7 features:

- Stealth Rock (binary)
- Spikes (layers / 3, clamped [0, 1])
- Toxic Spikes (layers / 2, clamped [0, 1])
- Sticky Web (binary)
- Reflect (binary)
- Light Screen (binary)
- Aurora Veil (binary)

## Active Pokemon Block
My active block is 60 features. Opponent active block is 41 features.

Shared base (40 features):

| Offset | Feature | Range |
|--------|---------|-------|
| +0 | HP fraction | [0, 1] |
| +1–18 | Type one-hot (18 types, tera-aware via `_effective_types`) | {0, 1} |
| +19–25 | Stat boosts / 6 (atk, def, spa, spd, spe, evasion, accuracy) | [-1, 1] |
| +26–29 | Status one-hot (BRN, PAR, SLP, FRZ) | {0, 1} |
| +30 | Poison severity (1.0 = TOX, 0.5 = PSN, 0.0 = none) | {0, 0.5, 1} |
| +31–34 | Volatile one-hot (Substitute, Taunt, Encore, Confusion) | {0, 1} |
| +35–39 | Item capabilities (choice-locked, recovery, dmg boost, spe boost, boots) | {0, 1} |

My active tera extension (+20 features, offsets 40–59):

| Offset | Feature | Range |
|--------|---------|-------|
| +40 | is_terastallized | {0, 1} |
| +41 | can_tera (battle-level flag) | {0, 1} |
| +42–59 | Tera type one-hot (18 types) | {0, 1} |

Opponent active tera extension (+1 feature, offset 40):

| Offset | Feature | Range |
|--------|---------|-------|
| +40 | is_terastallized | {0, 1} |

Opponent tera type is already reflected in the type one-hot via `_effective_types`.

## Available Move Block
Each move block is 38 features. 4 move slots, unfilled slots stay zero.

| Offset | Feature | Range |
|--------|---------|-------|
| +0 | Min damage % | [0, 1] |
| +1 | Max damage % | [0, 1] |
| +2 | Accuracy | [0, 1] |
| +3 | Expected value (avg damage × accuracy) | [0, 1] |
| +4 | KO flag (max_pct >= defender HP) | {0, 1} |
| +5 | STAB (tera-aware) | {0, 1} |
| +6 | Physical | {0, 1} |
| +7 | Special | {0, 1} |
| +8 | Contact | {0, 1} |
| +9 | Sound | {0, 1} |
| +10 | Priority (priority > 0) | {0, 1} |
| +11 | Pivot (U-turn, Volt Switch, etc.) | {0, 1} |
| +12 | Heal amount | [0, 1] |
| +13 | Drain ratio | [0, 1] |
| +14 | Self atk delta / 2 | [-1, 1] |
| +15 | Self spa delta / 2 | [-1, 1] |
| +16 | Self spe delta / 2 | [-1, 1] |
| +17 | Estimated recoil fraction | [0, 1] |
| +18 | Recharge flag | {0, 1} |
| +19 | Burn chance | [0, 1] |
| +20 | Paralysis chance | [0, 1] |
| +21 | Poison chance (PSN or TOX) | [0, 1] |
| +22 | Freeze chance | [0, 1] |
| +23 | Sleep chance | [0, 1] |
| +24 | Confusion chance | [0, 1] |
| +25 | Self def delta / 2 | [-1, 1] |
| +26 | Self spd delta / 2 | [-1, 1] |
| +27 | Is setup move | {0, 1} |
| +28 | Is hazard move | {0, 1} |
| +29 | Is recovery move | {0, 1} |
| +30 | Flinch chance | [0, 1] |
| +31 | Target def drop chance | [0, 1] |
| +32 | Target spa drop chance | [0, 1] |
| +33 | Target spd drop chance | [0, 1] |
| +34 | Target spe drop chance | [0, 1] |
| +35 | Target accuracy drop chance | [0, 1] |
| +36 | Multi-hit flag | {0, 1} |
| +37 | PP fraction (current / max) | [0, 1] |

Status/flinch/stat-drop chances are **ability-aware**: Serene Grace doubles secondary chances; Sheer Force zeroes them.

## My Bench Block
Each bench slot is 63 features. 5 slots for non-active team members.

| Offset | Feature | Range |
|--------|---------|-------|
| +0 | Alive flag | {0, 1} |
| +1 | HP fraction | [0, 1] |
| +2–19 | Type one-hot (18 types, tera-aware) | {0, 1} |
| +20–51 | Move flags (4 moves × 8 flags each) | — |
| +52 | Has Intimidate | {0, 1} |
| +53 | Best move EV vs opponent active | [0, 1] |
| +54 | Can OHKO opponent | {0, 1} |
| +55 | Outspeeds opponent (0.5 = unknown) | {0, 0.5, 1} |
| +56 | Max incoming damage from opponent | [0, 1] |
| +57 | Hazard entry damage on switch-in | [0, 1] |
| +58 | Poison severity | {0, 0.5, 1} |
| +59 | Paralysis | {0, 1} |
| +60 | Burn | {0, 1} |
| +61 | Sleep | {0, 1} |
| +62 | Freeze | {0, 1} |

Bench move flags (8 per move):

- physical, special, contact, sound, priority, pivot, heal amount, drain

## Opponent Bench Block
Each opponent bench slot is 20 features. Only revealed opponents are populated.

| Offset | Feature | Range |
|--------|---------|-------|
| +0 | Revealed flag | {0, 1} |
| +1 | HP fraction | [0, 1] |
| +2–19 | Type one-hot (18 types) | {0, 1} |

## Targeting Matrix
20 values total (4 moves × 5 bench targets).

For each of your up-to-4 available moves, the expected damage % against each of up to 5 revealed opponent bench targets. Only filled for revealed, non-fainted mons.

## Opponent Threat Block

**Team revealed flags** (6 features): one flag per team slot, 1.0 if that mon has been on the field.

**Threat rows** (4 × 8 = 32 features): four inferred move rows for the opponent active, selected via Bayesian role posterior:

| Offset | Feature | Range |
|--------|---------|-------|
| +0 | Move probability under role posterior | [0, 1] |
| +1 | Revealed flag (1.0 = seen, 0.0 = inferred) | {0, 1} |
| +2–7 | Expected damage % vs each of 6 team slots | [0, 1] |

**OHKO risk** (6 features): estimated probability the opponent can OHKO each team slot, weighted across role posterior stat spreads.

**Confidence summary** (2 features):

- Top role posterior mass [0, 1]
- Normalized role entropy [0, 1]

## Tail Features

| Index | Feature | Range |
|-------|---------|-------|
| 759 | On recharge (must recharge volatile) | {0, 1} |
| 760 | Alive count differential ((mine - theirs) / 6) | [-1, 1] |
| 761 | Force switch (forced to switch due to faint) | {0, 1} |

## Extended Tail Features

| Index | Feature | Range |
|-------|---------|-------|
| 762 | Speed ratio (my speed / (my speed + opp speed)) | [0, 1] |
| 763 | My tailwind active | {0, 1} |
| 764 | Opponent tailwind active | {0, 1} |
| 765–779 | Opponent ability one-hot (15 tracked abilities) | {0, 1} |

## Damage Source
Damage percentages come from `_damage_range_percent(...)`, which uses:

- current stats for the player's own active
- randbats role priors for hidden opponent stats when possible
- a manual Gen 9 damage formula with full modifier coverage
- inferred opponent move rows reuse the same damage path, combined with role-weighted move marginals from randbats metadata

## Main Symbols
```python
VECTOR_LENGTH = 780
def embed_battle(self, battle: AbstractBattle) -> np.ndarray
def verify_battle_embedding(self, battle: AbstractBattle, vector: np.ndarray) -> list[str]
```

## Related Debug Tool
Use `inspect_battle_debug.py` to inspect:

- active move damage blocks
- targeting matrix values
- opponent threat priors and OHKO-risk values
- legal action mapping
