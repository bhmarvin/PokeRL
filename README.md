# PokeRL

PokeRL is a reinforcement-learning project built on top of `poke_env` and Stable Baselines3 PPO for Pokemon Showdown Gen 9 random battles.

The project focuses on:

- a custom observation tensor with battle-state, move-value, targeting, and theory-of-mind features
- reward shaping on top of `poke_env`'s state-delta helper
- masked PPO so the policy never assigns probability mass to illegal actions
- repeatable training, checkpointing, evaluation, and summary output

## Main Files

- `brent_agent.py`: environment features, reward shaping, damage estimation, strategic penalties
- `train_ppo.py`: training entry point, checkpoint loading, eval callback, summary writing
- `policies.py`: masked PPO policy and observation extractor
- `opponents.py`: opponent selection (`random`, `max_base_power`, `simple_heuristic`)
- `randbats_data.py`: randbats role priors and stat inference
- `inspect_battle_debug.py`: interactive battle and embedding inspector

## Training

Example fresh run against `max_base_power`:

```powershell
.\venv\Scripts\python.exe train_ppo.py --train-timesteps 150000 --train-opponent max_base_power --eval-opponent max_base_power --eval-freq 20000 --save-freq 20000 --eval-battles 100 --device cuda --run-name ppo_fresh_maxpower_150k_eval20k
```

Example fresh run against `simple_heuristic`:

```powershell
.\venv\Scripts\python.exe train_ppo.py --train-timesteps 300000 --train-opponent simple_heuristic --eval-opponent simple_heuristic --eval-freq 20000 --save-freq 20000 --eval-battles 100 --device cuda --run-name ppo_fresh_simpleheuristic_300k
```

Example resume run:

```powershell
.\venv\Scripts\python.exe train_ppo.py --resume-from "results\ppo\some_run\checkpoints\rl_model_40000_steps.zip" --train-timesteps 60000 --train-opponent max_base_power --eval-opponent max_base_power --eval-freq 10000 --save-freq 10000 --eval-battles 100 --device cuda --run-name resumed_run
```

## Outputs

Each run writes under `results/ppo/<run_name>/`:

- `checkpoints/`: periodic checkpoints
- `best_model/`: best eval model by callback reward
- `model.zip`: final saved model
- `summary.json`: final summary payload with eval records

TensorBoard logs are written under `results/ppo/logs/`.

Start TensorBoard with:

```powershell
.\venv\Scripts\tensorboard.exe --logdir results\ppo\logs --port 6006
```

## Debugging

Use the inspector to step through live battles and compare embedding values against recomputed damage estimates:

```powershell
.\venv\Scripts\python.exe inspect_battle_debug.py --opponent random
```

Load a trained model into the inspector:

```powershell
.\venv\Scripts\python.exe inspect_battle_debug.py --opponent random --checkpoint "results\ppo\some_run\best_model\best_model.zip" --device cuda
```

The inspector currently prints:

- active-target move damage blocks
- revealed bench targeting values
- theory-of-mind values for revealed enemy moves
- legal action decoding
- step-tracing around policy action submission

## Current Behavior Notes

- Reward shaping uses `victory_value = 15.0`.
- Strategic penalties are based on pre-move snapshots, not post-move board state.
- Damage estimation uses randbats priors for hidden opponent roles where possible.
- Name normalization for species, items, and moves is applied before randbats role filtering.
- Eval callback logging includes win/loss/draw counts, win rate, mean reward, average steps, switch rate, and strategic-penalty aggregates for new runs.

## Docs

See `docs/README.md` for the documentation index.
