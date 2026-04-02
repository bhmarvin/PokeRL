# PokeRL

PokeRL is a reinforcement-learning project built on top of `poke_env` and Stable Baselines3 PPO for Pokemon Showdown Gen 9 random battles.

The project focuses on:

- a 684-dim structured observation tensor with battle-state, move-value, bench matchup, targeting, and theory-of-mind features
- a structured feature extractor that processes semantic blocks (active mons, moves, bench slots, threats) through shared-weight encoders before feeding a [256,128] MLP
- reward shaping on top of `poke_env`'s state-delta helper with tactical levers
- masked PPO so the policy never assigns probability mass to illegal actions
- Welford online observation normalization (running mean/variance saved with model checkpoint)
- repeatable training, checkpointing, evaluation, and summary output

## Main Files

- `brent_agent.py`: observation vector (684 dims), reward shaping, damage estimation, type/ability mechanics, tactical lever evaluation
- `train_ppo.py`: training entry point, checkpoint loading, eval callback, summary writing
- `policies.py`: structured observation extractor (9 semantic block encoders) and masked PPO policy
- `opponents.py`: opponent selection (`random`, `max_base_power`, `simple_heuristic`)
- `randbats_data.py`: randbats role priors and stat inference
- `test_runner.py`: observation vector, tera, and reward lever tests
- `inspect_battle_debug.py`: interactive battle and embedding inspector

## Observation Vector (684 dims)

| Block | Size | Description |
|-------|------|-------------|
| Global | 24 | Turn, weather, terrain, trick room, side conditions |
| My Active | 60 | HP, types, boosts, status, volatiles, items, tera (type + flags) |
| Opp Active | 41 | HP, types, boosts, status, volatiles, items, tera flag |
| Speed | 1 | Speed advantage vs opponent active |
| My Moves (4x25) | 100 | Damage range, accuracy, EV, KO flag, STAB, category, effects |
| My Bench (5x58) | 290 | HP, types, move flags, intimidate, offensive matchup vs opp active (best EV, OHKO, speed), defensive matchup (max incoming), hazard entry damage |
| Opp Bench (5x20) | 100 | Revealed flag, HP, types |
| Targeting (4x5) | 20 | My moves damage EV vs each opp bench slot |
| Threat/Meta | 48 | Team revealed, opp threat rows (4 moves x EV vs team), OHKO risk, role confidence, recharge, alive count diff |

## Structured Extractor

The `StructuredObservationExtractor` in `policies.py` normalizes the full vector (Welford), then slices into 9 semantic blocks. Each block passes through a small Linear+ReLU encoder. Repeated structures (moves, bench slots) use **shared-weight encoders** -- the same network processes each slot. Output is 561 dims fed to a [256,128] actor-critic MLP.

## Reward Shaping

Core rewards from poke_env: `hp_value=0.5`, `fainted_value=1.0`, `status_value=0.25`, `victory_value=30.0`

Tactical levers (penalties/bonuses applied per-decision):

| Lever | Value | Triggers When |
|-------|-------|---------------|
| good_heal_timing | +0.2 | Healing below 40% HP when not outsped for KO |
| good_safe_switch | +0.2 | Switching to a mon that resists opponent's STAB |
| good_tera | +0.5 | Tera grants immunity or enables a KO |
| head_hunter_bonus | +0.25/teammate | KOing a mon that threatened multiple teammates |
| unsafe_stay_in | -0.2 | Staying in when outsped for KO with safe switch available |
| abandon_boosted_mon | -0.1 | Switching out a +2 mon not under pressure |
| heal_satiation | -0.1 | 3+ consecutive heals on same mon |
| wasted_free_switch | -0.1 | Switching out immediately after entering via faint (next turn only) |
| redundant_hazards | -0.1 | Setting hazards already on field |
| redundant_self_drop | -0.1 | Using self-stat-drop move at -2 when better option exists |

## Training

Recommended training pipeline (progressive difficulty):

```powershell
# Stage 1: Random opponent (100k steps)
.\venv\Scripts\python.exe train_ppo.py --train-timesteps 100000 --train-opponent random --eval-opponent random --eval-freq 10000 --eval-battles 100 --device cuda --run-name stage1_random

# Stage 2: Max base power (200k steps, resume from stage 1)
.\venv\Scripts\python.exe train_ppo.py --train-timesteps 200000 --train-opponent max_base_power --eval-opponent max_base_power --eval-freq 10000 --eval-battles 100 --device cuda --run-name stage2_maxpower --resume-from results\ppo\stage1_random\model.zip

# Stage 3: Simple heuristic (300k steps, resume from stage 2)
.\venv\Scripts\python.exe train_ppo.py --train-timesteps 300000 --train-opponent simple_heuristic --eval-opponent simple_heuristic --eval-freq 10000 --eval-battles 100 --device cuda --run-name stage3_heuristic --resume-from results\ppo\stage2_maxpower\model.zip
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

Tests cover: observation vector shape and content, tera observation encoding, tera immunity reward, tera damage calcs, and reward lever assertions.
