# PokeRL

PokeRL is a reinforcement-learning project built on top of `poke_env` and Stable Baselines3 PPO for Pokemon Showdown Gen 9 random battles.

The project focuses on:

- a 758-dim structured observation tensor with battle-state, move-value, bench matchup, targeting, and theory-of-mind features
- a structured feature extractor that processes semantic blocks (active mons, moves, bench slots, threats) through shared-weight encoders before feeding a [512,256,128] MLP
- reward shaping on top of `poke_env`'s state-delta helper with tactical levers
- masked PPO so the policy never assigns probability mass to illegal actions
- repeatable training, checkpointing, evaluation, and summary output

## Main Files

- `brent_agent.py`: observation vector (758 dims), reward shaping, damage estimation, type/ability mechanics, tactical lever evaluation
- `train_ppo.py`: training entry point, checkpoint loading, eval callback, summary writing
- `policies.py`: structured observation extractor (9 semantic block encoders, 705-dim output) and masked PPO policy
- `opponents.py`: opponent selection (`random`, `max_base_power`, `simple_heuristic`)
- `randbats_data.py`: randbats role priors and stat inference
- `test_runner.py`: observation vector, tera, damage calc, volatile effect, and reward lever tests
- `inspect_battle_debug.py`: interactive battle and embedding inspector

## Observation Vector (758 dims)

| Block | Size | Description |
|-------|------|-------------|
| Global | 24 | Turn, weather, terrain, trick room, side conditions |
| My Active | 60 | HP, types, boosts, status, volatiles, items, tera (type + flags) |
| Opp Active | 41 | HP, types, boosts, status, volatiles, items, tera flag |
| Speed | 1 | Speed advantage vs opponent active |
| My Moves (4x37) | 148 | Damage range, accuracy, EV, KO flag, STAB, category, effects, setup/hazard/recovery flags, flinch, target stat drops, multi-hit flag |
| My Bench (5x63) | 315 | HP, types, move flags, intimidate, offensive matchup vs opp active (best EV, OHKO, speed), defensive matchup (max incoming), hazard entry damage, status conditions |
| Opp Bench (5x20) | 100 | Revealed flag, HP, types |
| Targeting (4x5) | 20 | My moves damage EV vs each opp bench slot |
| Threat/Meta | 49 | Team revealed, opp threat rows (4 moves x EV vs team), OHKO risk, role confidence, recharge, alive count diff, force switch |

### Move Block Detail (37 features per move)

| Index | Feature | Range |
|-------|---------|-------|
| 0-1 | Damage range (min%, max%) | [0, 1] |
| 2 | Accuracy | [0, 1] |
| 3 | Expected value (avg damage x accuracy) | [0, 1] |
| 4 | KO flag (max_pct >= remaining HP) | {0, 1} |
| 5 | STAB | {0, 1} |
| 6-7 | Category (physical, special) | {0, 1} |
| 8-9 | Flags (contact, sound) | {0, 1} |
| 10 | Priority move | {0, 1} |
| 11 | Pivot move (U-turn, Volt Switch) | {0, 1} |
| 12 | Heal amount | [0, 1] |
| 13 | Drain ratio | [0, 1] |
| 14-16 | Self stat deltas (atk, spa, spe) | [-1, 1] |
| 17 | Recoil fraction | [0, 1] |
| 18 | Recharge flag | {0, 1} |
| 19-24 | Status chances (brn, par, psn, frz, slp, confusion) | [0, 1] |
| 25-26 | Self stat deltas (def, spd) | [-1, 1] |
| 27 | Is setup move (Swords Dance, Calm Mind, etc.) | {0, 1} |
| 28 | Is hazard move (Stealth Rock, Spikes, etc.) | {0, 1} |
| 29 | Is recovery move (Recover, Roost, etc.) | {0, 1} |
| 30 | Flinch chance | [0, 1] |
| 31-35 | Target stat drop chances (def, spa, spd, spe, accuracy) | [0, 1] |
| 36 | Multi-hit move | {0, 1} |

Status/flinch/stat-drop chances are **ability-aware**: Serene Grace doubles secondary chances; Sheer Force zeroes them.

## Damage Calculator

The manual damage calculator (`_manual_damage_calc`) implements the Gen 9 damage formula with Bayesian role-weighted stat inference for opponent Pokemon. Modifier coverage includes:

**Base Power:** Technician, Sheer Force, Tough Claws, Strong Jaw, Mega Launcher, Sharpness, Punk Rock, Iron Fist, Reckless, Sand Force, Pixilate/Aerilate/Refrigerate/Galvanize (1.2x), terrain boosts (Electric/Grassy/Psychic), terrain reductions (Misty/Grassy), move-specific (Facade, Brine, Venoshock, Hex, Knockoff), Dry Skin

**Attack Stat:** Choice Band/Specs, Guts, Huge Power/Pure Power, Water Bubble, Solar Power, Gorilla Tactics, Steelworker/Dragon's Maw/Rocky Payload, Transistor, Flash Fire, pinch abilities (Overgrow/Blaze/Torrent/Swarm), Thick Fat/Heatproof (defender)

**Defense Stat:** Sandstorm SpD (Rock types), Snow Def (Ice types), Eviolite, Assault Vest, Fur Coat, Ice Scales, Marvel Scale

**Final Modifiers:** STAB (including tera + Adaptability), weather (with Utility Umbrella), burn (with Guts/Facade exceptions), screens (Reflect/Light Screen/Aurora Veil), Solid Rock/Filter/Prism Armor, Multiscale/Shadow Shield, Tinted Lens, Neuroforce, Life Orb, Expert Belt

**Special Moves:** Body Press (uses Def stat), Foul Play (uses target's Atk), Unaware (ignores boosts), Hustle, Hydro Steam, multi-hit scaling (expected hits)

**Type Resolution:** Weather Ball, Judgment, Nature Power, Terrain Pulse, Revelation Dance, Tera Blast (tera type), Pixilate/Aerilate/Refrigerate/Galvanize (Normal → ability type)

## Structured Extractor

The `StructuredObservationExtractor` in `policies.py` slices the pre-normalized observation vector into 9 semantic blocks. Each block passes through a small Linear+ReLU encoder. Repeated structures (moves, bench slots) use **shared-weight encoders** -- the same network processes each slot. Output is 705 dims fed to a [512,256,128] actor-critic MLP. Block sizes auto-derive from `brent_agent.py` constants.

## Reward Shaping

Core rewards from poke_env: `hp_value=0.5`, `fainted_value=1.0`, `status_value=0.25`, `victory_value=15.0`

Tactical levers (penalties/bonuses applied per-decision):

| Lever | Value | Triggers When |
|-------|-------|---------------|
| good_heal_timing | +0.2 | Healing below 40% HP when not outsped for KO |
| good_safe_switch | +0.2 | Switching to a mon that resists opponent's STAB |
| good_tera | +0.5 | Tera grants immunity or enables a KO |
| head_hunter_bonus | +0.25/teammate | KOing a mon that threatened multiple teammates |
| unsafe_stay_in | -0.1 | Staying in when outsped for KO with safe switch available |
| abandon_boosted_mon | -0.05 | Switching out a +2 mon not under pressure |
| heal_satiation | -0.05 | 3+ consecutive heals on same mon |
| wasted_free_switch | 0.0 | Disabled — bad credit assignment at early training |
| redundant_hazards | -0.05 | Setting hazards already on field |
| redundant_self_drop | -0.05 | Using self-stat-drop move when better option exists |

## Training

Recommended training pipeline (progressive difficulty, multi-env):

```powershell
# Stage 1: Random opponent (100k steps, 4 envs)
.\venv\Scripts\python.exe train_ppo.py --train-timesteps 100000 --train-opponents "random,random,random,random" --n-envs 4 --eval-opponent random --eval-freq 10000 --eval-battles 100 --device cuda --run-name stage1_random

# Stage 2: Max base power (200k steps, resume from stage 1)
.\venv\Scripts\python.exe train_ppo.py --train-timesteps 200000 --train-opponents "max_base_power,max_base_power,max_base_power,max_base_power" --n-envs 4 --eval-opponent max_base_power --eval-freq 10000 --eval-battles 100 --device cuda --learning-rate 1e-4 --run-name stage2_maxpower --resume-from results\ppo\stage1_random\best_model\best_model.zip

# Stage 3: Simple heuristic (300k steps, 8 envs, mixed opponents)
.\venv\Scripts\python.exe train_ppo.py --train-timesteps 300000 --train-opponents "simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,max_base_power" --n-envs 8 --eval-opponent simple_heuristic --eval-freq 10000 --eval-battles 100 --device cuda --learning-rate 1e-4 --run-name stage3_heuristic --resume-from results\ppo\stage2_maxpower\best_model\best_model.zip
```

## Outputs

Each run writes under `results/ppo/<run_name>/`:

- `checkpoints/`: periodic checkpoints
- `best_model/`: best eval model by callback reward
- `model.zip`: final saved model
- `summary.json`: final summary payload with eval records

TensorBoard logs are written under `results/ppo/logs/`.

```powershell
.\venv\Scripts\tensorboard.exe --logdir results\ppo\logs --port 6006
```

## Debugging

Use the inspector to step through live battles and compare embedding values:

```powershell
.\venv\Scripts\python.exe inspect_battle_debug.py --opponent random
.\venv\Scripts\python.exe inspect_battle_debug.py --opponent random --checkpoint "results\ppo\some_run\best_model\best_model.zip" --device cuda
```

## Tests

```powershell
.\venv\Scripts\python.exe test_runner.py
```

Tests cover: observation vector shape and content, tera observation encoding, tera immunity reward, tera damage calcs, damage calc modifiers (17 tests covering abilities/items/terrain/screens/weather), multi-hit damage scaling, -ate ability type resolution and BP boost, move block features (setup/hazard/recovery/multi-hit), volatile effects (flinch/stat drops with Serene Grace/Sheer Force awareness), bench status encoding, and reward lever assertions.
