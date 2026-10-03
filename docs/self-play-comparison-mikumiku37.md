# Self-Play Comparison: PokeRL vs mikumiku37 / Jaxcalibur

Sources: the mikumiku37 Smogon thread (Reg M-C VGC doubles, #1 on ladder, Oct 1 2026),
the Jaxcalibur writeup (Gen 9 randbats singles, #1 on ladder, Aug 2026) which
mikumiku37 says it adapted its design from. Jaxcalibur is the same format as PokeRL,
so it is the closer reference for most points below.

## Side-by-side

| Axis | mikumiku37 (VGC) | Jaxcalibur (randbats) | PokeRL (this repo) |
|---|---|---|---|
| Simulator | Custom sim, ~650x faster than Showdown | JAX reimplementation of the battle engine, many games in parallel on GPU | Live Showdown server via poke_env websockets, SubprocVecEnv |
| Training scale | ~330M games, 48.3 h on one RTX 5090 | ~100M games, ~1 week on one H100 | Curriculum totals 8.3M env steps (roughly a few hundred thousand games) |
| Algorithm | PPO | PPO + GAE | SB3 PPO (n_steps 4096, batch 512, 10 epochs, gamma 0.99, ent 0.02), optional RecurrentPPO |
| Network | ~8.7M param transformer, trained from scratch | ~8.5M param non-causal transformer, position-based MoE (pokemon vs move vs event tokens use different MLP weights) | ~1.1M params: 9 block encoders -> [512,256,128] actor/critic MLP |
| Input | Own-side game state + static dex data (types, base stats, move data). No damage calc, no usage stats, no speed resolver | Tokens: 1 field, 2 active, 5 moves, 6 own team, 6 opp team, 16 recent events, 24 matchup tokens (incl. damage calcs) | 780-dim handcrafted vector with full Gen 9 damage calc, Bayesian randbats set inference, threat matrices. No event history, no species/move identity embeddings |
| Reward | Terminal only (win/loss) | Win-probability objective (terminal) | victory 15 + hp 0.5 + faint 1.0 + status 0.25 deltas + ~20 hand-coded tactical levers |
| Auxiliary heads | Value head used by search | Win prob, opponent item/ability/moves/tera, opponent next action ("substantially stronger") | None |
| Opponents | League of its own past versions, from scratch, no human replays, no scripted bots | Pure self-play | Scripted curriculum (random -> max power -> heuristic) then a checkpoint pool (max 8, oldest evicted) with PFSP weights |
| Teams | ~1,260 public tournament teams, 20-team rotation on ladder | Randbats (format-provided) | Randbats (format-provided) |
| Search at inference | 1 turn ahead, 16 sampled worlds, top-8 x top-8 joint-action payoff table solved for a mixed strategy. Policy trained without search | pUCT, depth 4 (~2 turns), 20,480 rollouts over 32 sampled worlds | None |
| Evaluation | Real ladder: 1857 Elo, 81.2% GXE; 557-218 over 775 games | Real ladder: peak 2557 Elo, 95.3 GXE | 100 battles vs scripted eval opponent; Elo anchored to scripted bots (random 400, heuristic 1000) |

## Where the gap actually is

### 1. Games played (biggest gap by far)
Both reference bots trained on 100M-330M games. The PokeRL curriculum plays on the
order of a few hundred thousand. That is a ~1000x difference, and it explains more
than any architecture choice. Every other item below is second-order until games/sec
goes up. Neither reference used a live Showdown server for training.

Realistic options, in increasing cost:
- Measure games/sec now and make it a tracked metric. Without the number there is
  nothing to optimize against.
- Run several Showdown server processes (`--server-ports` already exists) and more
  envs per GPU. The policy forward pass is not the bottleneck at 1.1M params; the
  websocket round trip and the JS sim are.
- Batch inference: one process drives N battles and runs one batched forward pass
  per decision round, instead of one SB3 env per subprocess.
- Replace the simulator for training only (keep poke_env for ladder play). This is
  what both reference bots did and it is a large project. Candidates to look at:
  spktrm/minimon (JAX Showdown distillation) or writing a Gen 9 randbats-only engine.

### 2. Opponent sampling: the PFSP in this repo is uniform
`CheckpointPool.sample_pfsp` weights entries by `(1 - win_rate)^2`, but
`CheckpointPool.update_result` is never called anywhere in the codebase. Every entry
stays at `games < 5` forever, so every weight is 1.0 and sampling is uniform. Fix
before anything else, since it is cheap: have `SelfPlayPlayer` (which sees
`battle.won` from the opponent side) append outcomes to a per-worker results file,
and have `PokeEnvEvalCallback` fold them into the pool manifest each eval cycle.

Other league differences vs the reference bots:
- Admission: a checkpoint only enters the pool when it sets a new best *mean shaped
  reward vs the scripted eval opponent*. The league should admit on a schedule
  (every N steps) and track quality inside the league, not via an outside bot.
- Eviction: max 8 with oldest-first eviction throws away exactly the opponents most
  likely to expose forgetting. AlphaStar-style leagues keep everything and let
  sampling weights handle relevance. Disk is cheap; keep all, cap sampling.
- Assignment granularity: each worker gets one opponent per eval cycle (50k steps),
  so a PPO batch sees at most 8 distinct opponents. Sample a new opponent per
  episode instead (SelfPlayPlayer can hold several loaded models and pick at
  `battle` start).
- The scripted-bot curriculum is a crutch the reference bots did not need, but at
  PokeRL's game count it is probably still correct. The risk is that stage 3 and 4
  (6M steps vs SimpleHeuristics) overfit to one scripted opponent before self-play
  starts. Consider moving to the league earlier and mixing in the scripted bots as
  fixed league members rather than as the main diet.

### 3. Reward: shaped vs terminal
mikumiku37 used terminal reward only; Jaxcalibur optimizes win probability. PokeRL
adds ~20 tactical levers on top of hp/faint/status deltas. Two consequences:
- Model selection and pool admission use mean shaped reward, not win rate. A policy
  that farms `good_safe_switch` and `good_heal_timing` can out-score one that wins
  more. Switch `best_model` selection to win rate (tie-break on reward).
- The levers encode the author's strategy priors. They help early with few games,
  but they cap the ceiling at that prior. The docs already list "automatic reward
  annealing" as future work; implement it as a multiplier on the lever block that
  decays to zero by the time the agent reaches the self-play tier. Keep the
  hp/faint/status deltas, which are close to potential-based shaping and mostly
  harmless.

### 4. Auxiliary prediction heads (cheap, high value)
Jaxcalibur trains the same trunk to also predict (a) the opponent's hidden set
(item, ability, moves, tera) and (b) the opponent's next action, and reports the
next-action head made the bot "substantially stronger". Both labels are free in
self-play: after a turn resolves you know what the opponent chose, and at battle end
you know the opponent's full sets. PokeRL already infers sets with a Bayesian prior
in `randbats_data.py`; a learned head would replace that with something trained on
actual outcomes. In SB3 this means subclassing `PPO.train` to add the extra losses
and storing the labels in the rollout buffer.

### 5. Observation: what to keep and what is missing
Keep the damage calc. Jaxcalibur's 24 matchup tokens include damage calcs, so
hand-computed matchup features are not the problem. What PokeRL lacks:
- Event history. Jaxcalibur feeds the last 16 events as tokens. PokeRL has no turn
  history at all unless `--use-lstm` is on, and the LSTM has to recover it from
  state deltas. Even a fixed window of the last 4-8 (actor, move, target, result)
  tuples as an extra block would help switch prediction and set inference.
- Identity embeddings. The vector never says *which* species, move, item or ability
  it is looking at, only derived numbers (plus a 15-way opponent ability one-hot).
  Small learned embedding tables (species, move, item, ability ids) concatenated
  into the active/bench/move blocks let the net learn set correlations the
  hand-features miss.

### 6. Search at inference
Both reference bots gain a lot from test-time search over sampled hidden-info
worlds. This is blocked in PokeRL by the simulator: poke_env cannot fork a battle
state and roll it forward. It only becomes possible after item 1's simulator
replacement. Not worth attempting before then.

### 7. Evaluation
The reference bots measure on the real ladder. PokeRL measures 100 games vs a
scripted bot and anchors Elo on scripted bots. Add: (a) internal league Elo between
pool checkpoints (round-robin of ~50 games per pair at each admission), and (b) a
periodic ladder run with the current best checkpoint. Win rate vs
SimpleHeuristics saturates and stops being informative long before the ladder does.

## Prioritized list

| Priority | Change | Effort | Why |
|---|---|---|---|
| P0 | Feed pool win rates (`update_result` is never called) | ~50 lines | PFSP is currently uniform |
| P0 | Instrument games/sec; scale servers/envs; batch inference | days | 1000x fewer games than the reference bots |
| P1 | Best-model and pool admission by win rate, not shaped reward | ~20 lines | Selection currently rewards lever-farming |
| P1 | Anneal tactical levers to zero by the self-play tier | ~40 lines | Hand priors cap the ceiling |
| P1 | Per-episode opponent sampling; keep all checkpoints; scheduled admission | ~150 lines | League diversity per batch |
| P1 | Auxiliary heads: opponent next action, opponent set | ~300 lines (custom PPO.train) | Jaxcalibur's biggest single reported gain |
| P2 | Event-history block + species/move/item/ability embeddings | ~200 lines + obs version bump | Only format-knowledge the reference nets have that this one lacks |
| P2 | Internal league Elo + periodic ladder eval | ~150 lines | Scripted-bot win rate saturates |
| P3 | Training-only fast simulator; then 1-turn payoff-matrix search | weeks | Unlocks search and 100x throughput |

## What PokeRL already does that matches or exceeds the references
- Action masking at the logit level (both references do this too).
- Hidden-info handling via Bayesian randbats set inference; the references handle
  it with sampled worlds at search time and a learned set-prediction head.
- A full Gen 9 damage calc with ability/item/terrain coverage; Jaxcalibur feeds
  equivalent matchup tokens.
- Multi-stage curriculum and adaptive tiers, which the references skipped only
  because they could afford 100M+ games.
