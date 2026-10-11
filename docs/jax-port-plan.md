# PokeRL v2: JAX engine, transformer policy, self-play league, search

This is the full plan for rebuilding PokeRL along the lines of Jaxcalibur and
mikumiku37: a Gen 9 randbats battle engine in JAX, a token transformer with
auxiliary heads, a PureJaxRL-style PPO self-play league, differential testing
against the real Showdown simulator, and test-time search over sampled hidden
information. It is written to be executed collaboratively: the mechanic
transcription (abilities, moves, items) is the long tail and is explicitly set
up as shared, trackable work.

Everything below was checked against the current Showdown source
(`smogon/pokemon-showdown` master, October 2026) and current package versions.
Numbers that came from measuring are marked as such.

---

## 0. Decisions already made, and the three I need from you

### Decisions this plan commits to

| Decision | Choice | Why |
|---|---|---|
| Engine language | Pure JAX (`jax.numpy` + `lax`), no Python control flow inside `step` | The whole point: thousands of battles per `vmap`, trajectories in `lax.scan`, env and policy on the same GPU |
| Engine scope | Gen 9 randbats singles only, exactly the set pool Showdown generates | Measured pool is 509 species, 348 moves, 203 abilities, 38 items. Nothing outside that pool needs to exist |
| Team generation | Pre-generate teams with Showdown itself, store as arrays, sample at `reset` | Avoids porting 1,900 lines of `teams.ts` culling logic and guarantees the exact ladder distribution |
| Correctness oracle | Showdown in-process via Node (`sim/battle.ts`), seeded PRNG, turn-by-turn state diff | It is the only spec that exists |
| Model library | Flax NNX 0.12 + Optax 0.2.8 + Orbax for checkpoints | Current recommended Flax API; NNX is PyTorch-shaped so the learning curve from your SB3 code is small |
| Training loop | Hand-written PPO in the PureJaxRL pattern (single `lax.scan` over rollout + update) | SB3 cannot drive a batched JAX env or add auxiliary losses cleanly |
| Reward | Terminal win/loss only, plus optional hp/faint delta with a decay-to-zero schedule | Both reference bots used terminal only. We keep the delta as a training-speed knob, not a strategy prior |
| Search | Stage A: one-turn payoff matrix over sampled worlds solved for a mixed strategy (mikumiku37). Stage B: depth-limited decoupled pUCT (Jaxcalibur) | Stage A is a few hundred lines on top of a batched `step`. Stage B needs the tree machinery |
| Deployment | poke_env stays as the Showdown client. A thin adapter maps poke_env's battle object to the engine's observation and, for search, to sampled engine states | Your ladder plumbing, replay viewer, Elo and benchmark tooling all survive |

### What I need from you before Phase 2

1. **GPU and OS.** Your scripts use `venv\Scripts\python.exe`, so you are on
   Windows. JAX has no native CUDA build for Windows. GPU JAX runs on Linux or
   on Windows through WSL2 with the CUDA driver. If you are on Windows, the
   plan assumes WSL2 (Ubuntu) with `pip install "jax[cuda12]"`. Confirm which
   GPU you have. Throughput targets below assume a 4090/5090-class card.
2. **Time budget.** The milestone schedule assumes roughly 10 focused hours a
   week from you plus my transcription and scaffolding between sessions. Tell
   me if that is wrong and I will rescale.
3. **Showdown version pin.** We pin one commit of `smogon/pokemon-showdown` for
   the whole project and regenerate the team bank and data tables only when we
   deliberately bump it. Randbats sets change monthly. I propose pinning the
   commit at the time Phase 0 lands.

---

## 1. Scope, measured

I ran `tools/mechanics_inventory.py` (now in the repo) against Showdown master.
It reads `data/random-battles/gen9/sets.json`, the data files and the sim, and
reports what the randbats pool actually touches.

| Category | In pool | Data-only (no logic) | With event handlers | Handler lines (TS) |
|---|---|---|---|---|
| Species | 509 (879 role sets) | all | 0 | 0 |
| Moves | 348 | 233 | 115 | 6,571 |
| Abilities | 203 | 5 | 198 | 3,199 |
| Items | 38 | 3 | 35 | 654 |
| Conditions (status, weather, terrain, volatiles in `conditions.ts`) | 35 | 0 | 35 | ~900 |
| Engine events the sim fires | 72 | | | |

So the job is: port the core turn pipeline (one-time, the hard design work),
then transcribe about **350 effects totalling about 11,000 lines of
TypeScript** into predicated array code. The 233 data-only moves and the
species table are pure data loads. This is what "months" meant, and now it has
a denominator. Every effect gets a row in `coverage.yaml` (Section 6) and we
can measure the percentage of real randbats teams the engine can run at any
point.

Largest single effects by line count, which are also the ones to schedule
deliberately rather than discover late:

- Moves: Substitute (75), Court Change (65), Encore (58), Weather Ball (52),
  Rapid Spin (51), Throat Chop (47), Aurora Veil (46), Mortal Spin (46),
  Trick/Switcheroo (45 each), Protect (43), Destiny Bond (42), Mirror Coat
  (42), Taunt (41), Curse (41).
- Abilities: Natural Cure (81), Protosynthesis (65), Quark Drive (64), Ice Face
  (55), Disguise (45), Shields Down (43), Water Bubble (39), Illusion (39),
  Flash Fire (39), Flower Veil (37), Slow Start (34), Trace (32).
- Items: White Herb (53), Air Balloon (28), Booster Energy (26), Custap Berry
  (24), Leppa Berry (23), the Choice items (22 each), Clear Amulet (22).

Illusion, Trace, Transform-style and Commander-style effects are the ones that
break fixed-shape assumptions. They get a design note in Phase 2 before any
code.

---

## 2. Tooling and references, located

### Python side

| Package | Version (checked) | Role |
|---|---|---|
| `jax`, `jaxlib` | 0.11.2 | Core |
| `flax` (NNX API) | 0.12.10 | Model |
| `optax` | 0.2.8 | Optimizers, schedules, PPO loss helpers |
| `orbax-checkpoint` | current | Checkpoints and the league bank |
| `chex` | 0.1.92 | Shape/dtype assertions in tests, `chex.dataclass` for state |
| `jaxtyping` | current | Shape annotations on every engine function |
| `mctx` | 0.0.71 | Reference for pUCT batching. Not directly usable (assumes alternating perfect-information moves); we write our own simultaneous-move search |
| `minimon-jax` | current | Reference only. Shows one working layout of a jittable turn resolver, forced-switch phases and legal masks. Its mechanics are procedural analogues, not Gen 9 |
| `poke-env` 0.13 | already pinned | Ladder client and deployment adapter |
| `numpy`, `pytest`, `hypothesis` | | Tests, fuzzing |

### Node side (oracle and data export)

| Thing | Where | Use |
|---|---|---|
| Showdown simulator | `smogon/pokemon-showdown`, `sim/battle.ts` | In-process battles with `new Battle({formatid, seed, p1, p2})`. No server |
| PRNG injection | `sim/prng.ts`, `Battle` options `seed` and `prng` | Seeds: `gen5,xxxx...` selects the 64-bit LCG; `sodium,...` selects ChaCha20. `prng` accepts any object with `next()`, so a stub PRNG is a one-liner |
| State serialization | `sim/state.ts`, `State.serializeBattle` | Full battle state as JSON after every turn for diffing |
| Team generation | `Teams.generate('gen9randombattle', {seed})` | Builds the team bank |
| Set pool | `data/random-battles/gen9/sets.json` | Role sets per species (already parsed by the inventory script) |
| Data tables | `@pkmn/dex` (JSON-encoded, no runtime deps) or Showdown's `data/*.ts` via a small export script | Species, moves, abilities, items, type chart to flat arrays |
| Damage calc reference | `@smogon/calc` | Spot-check oracle for the damage formula port in Phase 1 |

### Reference designs

- Jaxcalibur writeup (jaxcalibur.github.io): token layout (1 field, 2 active,
  5 moves, 6 team, 6 opp team, 16 history events, 24 matchup), 8.5M params,
  non-causal transformer with position-type MoE, heads for policy, win prob,
  opponent next action, opponent item/ability/moves/tera; PPO+GAE; pUCT depth 4,
  20,480 rollouts over 32 sampled worlds; ~100M games in a week on one H100.
- mikumiku37 thread: 8.7M transformer, terminal reward only, league of past
  versions, ~330M games in 48h on one 5090 in a custom sim 650x faster than
  Showdown, one-turn 16-world top-8 x top-8 payoff matrix solved for a mixed
  strategy.
- PureJaxRL: the single-file PPO pattern we copy.

---

## 3. Repository layout

New package beside the existing code. Nothing in `brent_agent` is deleted
during the port; it is the spec for the damage calc and the source of the
matchup features.

```
pokejax/
  data/
    export_showdown.mjs        # Node: dumps dex tables + type chart + sets to JSON
    build_tables.py            # JSON -> packed .npz arrays with stable integer ids
    build_team_bank.mjs        # Node: Teams.generate x N -> teams.jsonl
    pack_team_bank.py          # teams.jsonl -> teams.npz (int arrays)
    tables.npz, teams.npz, ids.json
  engine/
    state.py                   # BattleState / SideState / MonState chex dataclasses
    tables.py                  # loads tables.npz into jnp arrays (module-level constants)
    stats.py                   # stat formula, boosts, speed with modifiers
    damage.py                  # Phase 1: vectorized damage formula (ported from brent_agent)
    types.py                   # effectiveness, STAB, tera type resolution
    legal.py                   # legal-action mask
    turn.py                    # step(): the turn pipeline (Section 5)
    passes/                    # one file per engine pass, table-driven effect handlers
      modify_priority.py, before_move.py, accuracy.py, base_power.py,
      modify_atk_def.py, modify_damage.py, on_hit.py, secondaries.py,
      after_move.py, switch_in.py, residual.py, faint.py
    effects/                   # per-effect predicates/handlers registered into passes
      abilities.py, items.py, moves.py, conditions.py
    prng.py                    # Gen5 LCG port + stub modes
    env.py                     # reset / step / observe / legal, the public API
  oracle/
    showdown_worker.mjs        # Node: runs a battle with given seed/teams/actions, emits state per turn
    diff.py                    # canonicalizes both states, reports first divergence
    fuzz.py                    # random-policy fuzzing driver, triage by effect id
    canon.py                   # Showdown state JSON -> canonical dict
  obs/
    tokens.py                  # BattleState + knowledge mask -> token tensors
    knowledge.py               # what each side has revealed (shared with poke_env adapter)
    history.py                 # last-K event ring buffer
  model/
    transformer.py             # Flax NNX model, heads
    losses.py
  train/
    ppo.py                     # PureJaxRL-style loop
    league.py                  # checkpoint bank, PFSP sampling, Elo
    config.py
  search/
    worlds.py                  # determinization: sample hidden sets consistent with knowledge
    payoff.py                  # Stage A one-turn matrix + Nash solve
    puct.py                    # Stage B
  deploy/
    poke_env_adapter.py        # poke_env Battle -> knowledge + obs; -> sampled BattleState
    ladder_player.py
  coverage.yaml                # every effect, its status, owner, notes
tests/
  test_damage_parity.py        # brent_agent calc vs engine calc on the 17 existing cases + fuzz
  test_engine_units.py
  test_oracle_diff.py
```

---

## 4. Phase 0: data pipeline and oracle harness (week 1 to 2)

Goal: a Node worker that plays a seeded Showdown battle from given teams and
actions and emits canonical state per turn, plus packed data tables and a
team bank. No JAX yet. This is pure plumbing and I can do nearly all of it.

1. **Pin Showdown.** `git submodule add` or a vendored checkout at a fixed
   commit under `third_party/pokemon-showdown`. `npm ci && node build`.
2. **Export tables.** `export_showdown.mjs` loads `Dex` and writes:
   `species.json` (id, num, types, base stats, weight, forme links),
   `moves.json` (id, type, category, base power, accuracy, priority, pp, flags,
   target, secondary chances, self boosts, drain/recoil fractions, multihit,
   crit ratio, and the raw list of handler hook names for coverage),
   `abilities.json`, `items.json`, `typechart.json`, `conditions.json`.
3. **Assign stable integer ids.** `build_tables.py` sorts ids, writes
   `ids.json` (bidirectional maps) and `tables.npz` with one row per id.
   Shape conventions: species `[S, ...]`, moves `[M, ...]`, type chart
   `[19, 19]` float32 including the "typeless" row. These ids are the same
   ones the observation embeddings use later.
4. **Team bank.** `build_team_bank.mjs` calls `Teams.generate` 1,000,000 times
   with sequential seeds and writes JSONL. `pack_team_bank.py` packs to
   `teams.npz`: `species [N,6] int16`, `level [N,6] int8`, `moves [N,6,4]
   int16`, `ability [N,6] int16`, `item [N,6] int16`, `tera [N,6] int8`,
   `evs/ivs/nature` as the resulting final stats `[N,6,6] int16`, `gender`,
   `hp`. A million teams is about 150 MB packed. Sampling a battle is
   `teams[key1], teams[key2]`.
5. **Oracle worker.** `showdown_worker.mjs` reads a JSON job
   `{seed, team1, team2, actions: [[p1choice, p2choice], ...]}` and emits
   `State.serializeBattle` after every turn. It accepts `prng_mode`:
   `gen5` (real seed) or `stub` (an object whose `next()` returns a fixed
   sequence, used for deterministic diffing). Runs as a persistent process
   reading jobs from stdin so fuzzing does not pay Node startup per game.
6. **Canonicalizer.** `canon.py` reduces the serialized battle to the fields
   our state has: per mon hp, status, boosts, volatiles (as a sorted id list),
   item, ability, moves and pp, tera state, active flag; per side conditions
   with counters; field weather/terrain/trick room with turns left; turn
   number; whose decision is pending. Anything not in our state is dropped,
   so a diff never fails on message logs.

Exit criteria: `python -m pokejax.oracle.fuzz --games 100 --policy random`
runs 100 Showdown games through the worker and writes canonical traces.
`tables.npz` and `teams.npz` load and round-trip ids.

---

## 5. Phase 1: JAX fundamentals by porting the damage calculator (week 2 to 3)

This is your JAX on-ramp and the first engine module. `brent_agent`'s
`_manual_damage_calc` is already a pure function of (attacker, defender, move,
field). It becomes `pokejax/engine/damage.py`.

Design:

```python
@jaxtyped
def damage_roll(
    atk: MonState, dfn: MonState, move_id: Int[Array, ""], field: FieldState,
    roll: Int[Array, ""],          # 0..15, the 85..100% roll index
    crit: Bool[Array, ""],
    tables: Tables,
) -> Int[Array, ""]: ...
```

Every `if` in the Python version becomes `jnp.where(cond, a, b)` or a table
lookup indexed by the effect id. The Gen 9 formula's rounding order matters
(Showdown uses `chainModify` with 4096-based fixed-point and specific
rounding), so the port follows `sim/battle-actions.ts: getDamage` and
`modifyDamage` line by line, not the Python version. The Python version is
the test oracle for the 17 existing cases; `@smogon/calc` through the Node
worker is the second oracle for a few thousand fuzzed cases.

What you learn here, and need for the rest: `jnp.where` instead of `if`,
`vmap` over batches, integer fixed-point care, `chex.dataclass` pytrees,
`jax.jit` tracing errors when a Python `if` sneaks in, and `jaxtyping`
shape discipline.

Exit criteria: `tests/test_damage_parity.py` passes on all existing cases and
on 5,000 fuzzed (attacker, defender, move, field) tuples against `@smogon/calc`
with zero mismatches, under `jit` and `vmap`.

---

## 6. Phase 2: engine core design (week 3 to 5)

This is the one-time hard part: the state layout, the turn pipeline, and the
handler architecture. We agree on this before any effect is transcribed.

### 6.1 State layout (struct of arrays, fixed shapes)

```python
@chex.dataclass
class MonState:                 # leading dims [..., 2 sides, 6 slots]
    species: int16
    level: int8
    hp: int16; max_hp: int16
    stats: int16[6]             # hp atk def spa spd spe (final, after level/nature)
    boosts: int8[7]             # atk def spa spd spe acc eva, -6..6
    status: int8                # 0 none, brn par slp frz psn tox
    status_counter: int8        # sleep turns, toxic counter
    types: int8[2]              # current types (tera overrides)
    base_types: int8[2]
    tera_type: int8; terastallized: bool
    ability: int16; base_ability: int16
    item: int16; item_consumed: bool
    moves: int16[4]; pp: int8[4]; move_disabled: bool[4]
    volatiles: bool[V]          # V ~ 48 named volatiles used by the pool
    volatile_counters: int8[V]  # durations / stacks for the ones that need it
    last_move: int16; last_move_target: int8; move_this_turn: int16
    times_attacked: int8
    fainted: bool; active: bool
    revealed_moves: bool[4]; revealed_item: bool; revealed_ability: bool   # knowledge
    forme: int16                # for Shields Down, Ice Face, Disguise, etc.
    substitute_hp: int16
    protect_count: int8         # consecutive protect counter
    illusion_as: int8           # slot index shown to the opponent, -1 if none

@chex.dataclass
class SideState:                # [..., 2]
    active_slot: int8
    conditions: int8[C]         # C ~ 12: SR, spikes layers, tspikes, web, reflect, lightscreen, veil, tailwind, safeguard, mist... with turns-left
    must_switch: bool; can_tera: bool
    last_move_used: int16; future_sight: (turns int8, damage int16, attacker int8)
    wish: (turns int8, hp int16)

@chex.dataclass
class FieldState:
    weather: int8; weather_turns: int8
    terrain: int8; terrain_turns: int8
    trick_room: int8; gravity: int8; ...

@chex.dataclass
class BattleState:
    mons: MonState; sides: SideState; field: FieldState
    turn: int16; phase: int8          # 0 choose, 1 forced-switch p1, 2 forced-switch p2, 3 both
    rng: uint32[2]                    # Gen5 LCG state (or PRNGKey in stub mode)
    done: bool; winner: int8
    history: EventRing                # last 16 events for the observation
```

Design rules: every array has a fixed shape; "a mon has no item" is id 0, not
an absence; every list in Showdown that can vary (volatiles, side conditions)
is a fixed bool/int vector over the enumerated set the pool uses; every
"duration" is a counter decremented in the residual pass.

Three effects need explicit designs before transcription because they fight
fixed shapes: Illusion (observation-side only, the engine state is truthful),
Trace/Transform-style ability copying (write into `ability`, keep
`base_ability`), and forme changes (Shields Down, Ice Face, Disguise,
Protosynthesis/Quark Drive stat selection) via `forme` plus a stats-override
table. Commander and Zacian/Zamazenta conditions are doubles or non-randbats
and are out of scope.

### 6.2 Action space and legal mask

Single integer per side per step, same layout for both the engine and the
ladder adapter: `0..3` move, `4..7` move + tera, `8..13` switch to slot, `14`
no-op (only legal when the other side is in a forced switch). Legal mask
computed from state: PP, Disable/Encore/Taunt/Choice lock/Assault Vest,
trapped flags, fainted, `can_tera`, forced-switch phase. This matches
minimon's convention and is a strict subset of poke_env's 26-slot gen 9
action space, so the adapter is a table.

### 6.3 Turn pipeline as fixed passes

Showdown fires 72 distinct events. The ones the randbats pool subscribes to
(from the inventory) are around 45. The engine runs them as a fixed sequence
of passes, each a pure function `(state, ctx) -> (state, ctx)`:

1. **Switch resolution** for any chosen switches (both sides), with
   `BeforeSwitchOut`, `SwitchOut`, `SwitchIn` (hazards, Intimidate, weather and
   terrain setters, Booster Energy, Air Balloon announce) in speed order.
2. **Tera** for sides that chose it.
3. **Action ordering**: `ModifyPriority` and `FractionalPriority` (Gale Wings,
   Prankster, Triage, Quick Claw, Custap, Quick Draw are not in pool; check
   with coverage), then speed with `ModifySpe` (Chlorophyll, Swift Swim, Sand
   Rush, Slush Rush, Surge Surfer, Unburden, Quick Feet, Protosynthesis/Quark
   speed, paralysis, Tailwind, Trick Room inversion, Choice Scarf), speed ties
   via the PRNG.
4. **For each of the two move executions in order**, a sub-pipeline:
   `BeforeMove` (sleep, freeze, paralysis, flinch, confusion, Taunt, recharge,
   truant-style), `TryMove`/`Try`, `ModifyMove`/`ModifyType` (the -ate
   abilities, Weather Ball, Tera Blast, Judgment-style), target redirection
   (none in singles except Protect/semi-invuln), `Invulnerability`,
   `Accuracy`/`ModifyAccuracy`, `TryHit`/`TryImmunity` (type immunity, ability
   immunities like Levitate, Flash Fire, Water Absorb, Volt Absorb, Sap
   Sipper, Earth Eater, Well-Baked Body), `PrepareHit`, `CriticalHit`,
   `BasePower`, `ModifyAtk/Def/SpA/SpD`, damage formula, `ModifyDamage`,
   `WeatherModifyDamage`, `Damage` (Substitute, Disguise, Ice Face, Sturdy,
   Focus Sash, Multiscale), apply HP, `DamagingHit` (Rough Skin, Rocky Helmet,
   contact statuses, Stamina, Weak Armor, Anger Shell), `Hit`/`AfterHit`
   (move's own `onHit`: Knock Off, Trick, Rapid Spin), `ModifySecondaries`
   and secondary rolls (Serene Grace, Sheer Force, Shield Dust, Covert Cloak),
   drain/recoil, `AfterMoveSecondarySelf` (Life Orb, Throat Spray),
   `AfterMove`, faint check, `AfterFaint` (Moxie, Beast Boost, Chilling Neigh,
   As One, Grim Neigh), `EmergencyExit`/Eject-style.
5. **Residual** in Showdown's documented order (weather damage, Leftovers/Black
   Sludge, status damage, Leech Seed, Wish, Aqua Ring, Ingrain, side
   condition timers, weather/terrain timers, Slow Start, Speed Boost, Moody,
   Hydration, Shed Skin, Harvest, Pickup are coverage items) with
   `onResidualOrder` and `SubOrder` values copied from the data files into a
   sort key table.
6. **Faint handling and phase transition**: set `phase` to the forced-switch
   state for whichever sides fainted; `done` when a side has no mons.
7. **Turn counter, history ring update.**

Within each pass, every effect in the pool that hooks that event is evaluated
for every mon and masked by "is this effect active on this mon right now".
That is the predication pattern: compute all, select one. Cost is linear in
pool size per pass and irrelevant on a GPU.

### 6.4 Effect handler architecture

Each effect declares which passes it participates in and provides a handler
per pass:

```python
# pokejax/engine/effects/abilities.py
@ability("guts")
class Guts:
    @hook("modify_atk", priority=5)
    def modify_atk(ctx, st, mon_idx, active):
        has_status = st.mons.status[mon_idx] != 0
        mult = jnp.where(active & has_status, 1.5, 1.0)
        return ctx.replace(atk_mult=ctx.atk_mult * mult)
```

A registry turns this into, per pass, a sorted list of (priority, handler)
that `turn.py` folds over. Handlers are plain functions, so every one is unit
testable by constructing a state and calling the pass. Priorities come from
the `onXPriority` values in Showdown's data; they are the order Showdown
resolves conflicting modifiers, and the inventory captures them.

### 6.5 PRNG

Two modes, selected at build time (static arg):

- **gen5**: a port of `Gen5RNG` (64-bit LCG, the constants are in
  `sim/prng.ts`) over `uint32[2]`, with `random(m,n)`, `randomChance(a,b)`,
  `shuffle` (Fisher-Yates, how speed ties resolve) implemented exactly as the
  TS does. This gives bit-exact traces against Showdown *only if* we also
  consume randomness in the same order. We do not promise that in Phase 3; it
  is a Phase 4 stretch goal checked per effect.
- **stub**: a PRNGKey-driven mode where every random site has its own
  deterministic policy (max roll, no crit, always hit, no secondary, speed
  ties by slot). The oracle worker runs the matching stub on the Showdown
  side. This is what differential testing uses first, because it removes the
  consumption-order problem entirely.
- **training**: PRNGKey-driven real randomness. Same code as stub with the
  policies replaced by draws.

### 6.6 Knowledge and observation hooks

The engine state is ground truth. `obs/knowledge.py` computes the per-side
"revealed" view: species of the active and anything that has been on the
field; moves once used; item once it visibly triggers or is announced
(Air Balloon, Booster Energy, Choice lock inferred is *not* revealed, the
policy learns that); ability once announced (Intimidate, weather setters,
Protosynthesis, immunities that message). These rules mirror what
Showdown's protocol sends and what poke_env exposes, because the same
`knowledge.py` runs on the ladder adapter.

Exit criteria for Phase 2: `turn.py` runs a complete battle with the data-only
subset (233 moves with no handlers, the 5 data-only abilities, no items), under
`jit` and `vmap(4096)`; the oracle diff agrees for 1,000 fuzzed stub-mode
games restricted to that subset. That restriction is enforced by the team
bank filter in Section 7.

---

## 7. Phase 3: mechanic transcription, the collaborative part (week 5 to 16)

### 7.1 Coverage manifest

`pokejax/coverage.yaml`, one entry per effect, generated by the inventory
script and then maintained by hand:

```yaml
abilities:
  guts:
    hooks: [onModifyAtk, onModifyAtkPriority]
    showdown_lines: 12
    status: verified          # todo | ported | verified | wontfix
    owner: claude
    passes: [modify_atk]
    tests: [tests/test_abilities.py::test_guts]
    notes: "Burn attack halving is skipped in modify_damage when Guts active"
```

`status` moves `todo -> ported` when the handler and a unit test land, and
`ported -> verified` only when a fuzz run of at least 500 oracle games that
include the effect shows zero divergences attributed to it.

### 7.2 The team bank filter is what makes this incremental

`pack_team_bank.py` tags each team with the set of effect ids it uses.
`env.reset` samples only from teams whose every effect is `ported` or better.
So the engine is trainable from the first week of Phase 3 on a small subset
of the real distribution, and the subset grows monotonically. We report
**coverage = fraction of the 1M-team bank fully supported** as the headline
metric. At 95% we start the real training runs; at 99% we stop worrying.

Because randbats sets are role-based, coverage grows fast with the common
effects: Leftovers, Heavy-Duty Boots, Choice items, Life Orb, the stat-boost
abilities, Intimidate, Regenerator, Protect, Knock Off, U-turn, Stealth Rock,
Swords Dance and the recovery moves cover a large fraction of sets on their
own.

### 7.3 Work split

- **Me, per batch of ~20 effects:** read the Showdown source for each, write
  the handler into the right passes, write a unit test that sets up the exact
  situation (e.g. Guts + burn + physical move), update `coverage.yaml`, run
  the fuzz diff on a bank filtered to include the batch, triage any
  divergence. One PR per batch, with the inventory diff in the description.
- **You, per batch:** review semantic correctness from your own knowledge of
  the mechanics (this is where your `brent_agent` work pays off; you have
  already hand-implemented much of the damage-side logic), pick the next
  batch by frequency in the bank, and run the GPU-side regression
  (`jit` + `vmap` speed and the differential fuzz) on your machine.
- **Both:** when the fuzz finds a divergence that is a Showdown subtlety
  (ordering, rounding, an undocumented interaction), we document it in the
  `notes` field so it is never rediscovered.

### 7.4 Batch order (by bank frequency, then dependency)

1. Items used by the most sets: Leftovers, Heavy-Duty Boots, Choice Band /
   Specs / Scarf, Life Orb, Assault Vest, Eviolite, Focus Sash, Sitrus/Lum/
   Chesto, Rocky Helmet, Black Sludge, Booster Energy, Air Balloon, White
   Herb, Loaded Dice, Clear Amulet, Covert Cloak, Throat Spray, Weakness
   Policy, Custap, Expert Belt, Light Ball, Thick Club, Leppa, Flame/Toxic
   Orb, Wide Lens, Shed Shell, Red Card, Eject Button (that is the 38).
2. Stat-modifying and immunity abilities (about 60): the `onModifyAtk/SpA/Spe/
   Def/SpD` and `onTryHit/onImmunity` families. Mechanically similar, so they
   go fast once the pattern is set.
3. Switch-in abilities (about 25): Intimidate, weather and terrain setters,
   Protosynthesis/Quark Drive, Screen Cleaner, Download, Frisk (knowledge
   side), Pressure, Unnerve, Supreme Overlord.
4. Core move handlers (about 40): Protect family, Substitute, hazards and
   removal (Stealth Rock, Spikes, Toxic Spikes, Sticky Web, Rapid Spin,
   Mortal Spin, Defog, Tidy Up), pivots, Knock Off, Trick/Switcheroo,
   recovery moves, Rest/Sleep Talk, Encore, Taunt, Disable, Leech Seed,
   Will-O-Wisp/Toxic/Thunder Wave semantics, Curse, Wish, Future Sight, Roar/
   Whirlwind/Dragon Tail, Sucker Punch, Counter/Mirror Coat, Destiny Bond,
   Endeavor, Body Press, Foul Play, Weather Ball, Tera Blast.
5. Damage-side and contact abilities (about 40): Rough Skin, Iron Barbs,
   Static/Flame Body/Poison Point/Effect Spore, Stamina, Weak Armor, Anger
   Shell, Berserk, Cursed Body, Aftermath, Innards Out, Mummy/Lingering Aroma,
   Wandering Spirit, Gooey/Tangling Hair, Cotton Down, Toxic Debris.
6. Residual and status abilities (about 25): Regenerator, Natural Cure,
   Shed Skin, Hydration, Speed Boost, Moody, Poison Heal, Magic Guard, Bad
   Dreams, Ice Body/Rain Dish/Dry Skin, Harvest, Pickup, Slow Start,
   Emergency Exit/Wimp Out.
7. The forme and copy effects (about 10) with their design notes: Illusion,
   Disguise, Ice Face, Shields Down, Zen Mode-style, Trace, Imposter (if in
   pool; Ditto is), Mimicry, Power of Alchemy-style.
8. Everything else, by whatever the bank says is left.

### 7.5 Throughput checkpoints

At the end of batches 2, 4 and 6 we measure `vmap(4096)` steps per second
under `jit`. Target on a 4090/5090: above 100k decisions per second end to
end with a random policy, i.e. roughly 2,000 games per second. If a batch
regresses throughput by more than 20% we look at the handler (usually a
`lax.switch` that should have been a table gather, or a dynamic-shape leak).

---

## 8. Phase 4: differential testing at scale (continuous from week 5, hardening week 16 to 18)

The oracle harness from Phase 0 is run three ways:

1. **Stub-mode fuzz, per batch** (continuous): random legal actions on both
   sides, 500 to 5,000 games, first-divergence report with the effects active
   in that turn. This is the main correctness loop.
2. **Distributional parity, per milestone**: real randomness on both sides,
   100k games, same team bank. Compare win-rate matrices by species, average
   game length, KO counts, status rates, crit rates. A chi-square over these
   catches "I forgot the crit immunity" bugs that no single trace shows.
3. **Bit-exact traces, stretch**: gen5 mode, with RNG consumption order
   matched. We attempt it for the core pipeline only (damage roll, crit,
   accuracy, secondary, speed tie). If it works it becomes the fastest
   possible regression test. If consumption order proves too fragile we stop
   and rely on 1 and 2.

Gate for Phase 5: 0 divergences over 20k stub games on the 95%-coverage bank,
and distributional parity within noise on 100k games.

---

## 9. Phase 5: observation tokens and the model (week 10 to 14, parallel with transcription)

This can start as soon as Phase 2 lands, because it only needs state shapes.

### 9.1 Tokens

Mirroring Jaxcalibur's layout, adapted to what your observation code already
computes well:

| Token type | Count | Content |
|---|---|---|
| Field | 1 | weather, terrain, trick room, turn, both sides' conditions and timers |
| Active (own, opp) | 2 | species emb + item emb + ability emb + tera emb, hp frac, status, boosts, volatiles, types, knowledge flags |
| Own moves | 4 | move emb, pp, disabled, plus the full 38-feature move block from `brent_agent` (damage range vs active, KO flag, EV, flags) |
| Own team | 6 | species/item/ability/tera emb, hp, status, revealed flags, plus bench matchup features from `_fill_bench_matchup` |
| Opp team | 6 | same, with unknown ids as the special "unknown" embedding and the randbats role posterior from `randbats_data` as a dense feature |
| History | 16 | event ring: (actor slot emb, move emb, target emb, outcome bucket, damage frac, turn delta) |
| Matchup | 24 | own 6 x opp 4-known-or-inferred-moves damage EV and OHKO flags, as in the targeting and threat matrices you already build |

Each token type has its own input projection (position-type MoE in
Jaxcalibur's words). Identity embeddings are learned tables over the stable
ids from Phase 0, 64 dims each. Everything numeric is pre-normalized exactly
as `brent_agent` does today.

### 9.2 Architecture

Non-causal encoder: 59 tokens, d_model 256, 8 heads, 6 layers, per-token-type
MLP weights, pre-LN, GELU. That is about 5M params in the trunk; with
embeddings and heads about 8M, in line with both references. Heads from a
pooled (CLS) token plus the own-move and own-team tokens:

- **policy**: 15 logits, masked with the legal mask before softmax.
- **value**: win probability, tanh, trained on the terminal result.
- **opp next action**: 15-way over the opponent's action space, trained on
  what they actually did. Jaxcalibur's biggest single win.
- **opp set**: per opp slot, multi-label over moves (348), item (38),
  ability (203), tera (19), trained on the ground truth the engine has.
  Masked to slots that are revealed-species.

Flax NNX, `nnx.MultiHeadAttention`, `nnx.Linear`, `nnx.LayerNorm`. Mixed
precision bf16 for attention, fp32 for the heads and losses.

Exit criteria: forward and backward under `jit` on a batch of 8192
observations in under 20 ms on your GPU; a supervised smoke test that the
opp-set head learns randbats priors from engine-generated data.

---

## 10. Phase 6: PPO self-play league (week 14 to 20)

### 10.1 Loop

PureJaxRL shape, two-sided:

```
for update in range(U):                      # Python loop, each body is one jit
    (state, obs, key), traj = lax.scan(rollout_step, carry, None, length=T)   # T=32 steps, B=4096 battles
    adv, ret = gae(traj)
    params, opt = lax.scan(epoch, (params, opt), None, length=E)              # E=4 epochs, minibatch 8192
    league.maybe_add(params); metrics
```

Both players in every battle are controlled inside the same `vmap`: slot 0 by
the learner, slot 1 by an opponent whose params are gathered from a stacked
checkpoint bank by an index sampled at `reset`. Only slot 0's transitions
enter the PPO buffer. Opponent action sampling uses the opponent's own
policy with temperature 1 so it is not deterministic.

### 10.2 League

- Bank: stacked params `[K, ...]` in device memory, K up to 32 recent plus
  16 "anchor" checkpoints spaced over training, saved with Orbax.
- Sampling: 50% latest self, 35% PFSP over the bank with weight
  `(1 - winrate)^2` from a running per-opponent win-rate table that is
  actually updated every rollout (the bug in the current repo), 15% uniform
  over anchors so nothing is forgotten.
- Elo: internal round-robin of 200 games per pair at every bank addition,
  tracked as the main training curve alongside win rate vs the newest anchor.
- Admission: every N updates unconditionally, plus whenever the learner's Elo
  over the bank passes a threshold.

### 10.3 Hyperparameters to start from

| Param | Start | Notes |
|---|---|---|
| Battles in parallel | 4096 | fit to GPU memory |
| Rollout length | 32 steps | ~ one game is 25 to 60 decisions, so partial episodes with bootstrapping |
| Gamma | 1.0 | terminal reward, episodic |
| GAE lambda | 0.95 | |
| Clip | 0.2 | |
| LR | 3e-4 cosine to 3e-5 | Adam, b2 0.95, grad clip 1.0 |
| Entropy | 0.01 decaying to 0.001 | |
| Value coef | 0.5 | |
| Aux coefs | next-action 0.3, opp-set 0.1 | tune on validation |
| Epochs / minibatch | 4 / 8192 | |

Throughput math to set expectations: at 2,000 games/s the engine alone does
about 170M games a day. mikumiku37's #1 checkpoint saw 330M. Jaxcalibur's
saw 100M. Training will be the bottleneck, not the env, which is the right
place to be.

Exit criteria: a 24-hour run on your GPU that shows monotone internal Elo,
beats `SimpleHeuristicsPlayer` above 95% through the poke_env adapter, and
reaches a ladder GXE you are happy to report.

---

## 11. Phase 7: search (week 20 onward)

Stage A, mikumiku37-style, because it needs nothing but batched `step`:

1. Sample W=16 worlds: hidden opp sets drawn from the randbats posterior
   (`randbats_data` + `sets.json`) consistent with everything revealed.
2. For each world, take the policy's top-8 own actions and top-8 predicted
   opponent actions (the next-action head makes this cheap), roll each of
   the 64 joint actions one turn with real randomness R=4 times, evaluate
   the resulting states with the value head, average: a 8x8 payoff matrix
   per world, averaged over worlds.
3. Solve the zero-sum matrix game for the own mixed strategy (fictitious
   play or a tiny LP; 8x8 is trivial) and sample from it.

That is 16 x 64 x 4 = 4,096 engine steps and 4,096 value evaluations per
decision, well under a second on a GPU, and it directly addresses the
simultaneous-move problem that single-agent search ignores.

Stage B, Jaxcalibur-style depth-4 pUCT with decoupled per-player statistics
over sampled worlds, only if Stage A shows a clear gain and you want more.

---

## 12. Phase 8: deployment adapter (week 18 to 20, parallel)

`deploy/poke_env_adapter.py`:

- poke_env `Battle` -> `knowledge.py` view -> tokens. Same code path as
  training, so no train/deploy observation skew. The existing
  `BrentObservationVectorBuilder` features that feed the move, bench and
  matchup tokens are reused through a thin wrapper.
- For search: `Battle` -> sampled `BattleState` worlds. This needs the hidden
  parts filled (opp sets from the posterior, own exact state from the client)
  and the visible parts transcribed (hp, boosts, status, volatiles, side
  conditions, field). This is the one place the engine's state must be
  constructible from protocol messages, which also makes it a second
  correctness check.
- Latency budget: 150 ms per decision without search, 1 s with Stage A.

`ladder_player.py` replaces `SelfPlayPlayer` for ladder and challenge play.
`benchmark_model.py`, `elo.py` and `replay_viewer.py` keep working with a
small interface change.

---

## 13. What is reused from PokeRL, concretely

| PokeRL piece | Fate |
|---|---|
| `brent_agent/observation.py` damage calc and move/bench/matchup features | Spec for `engine/damage.py`; feature code reused inside the token builder |
| `randbats_data.py` | Posterior for opp-set tokens and for search determinization |
| `test_runner.py` damage and mechanics cases | Seed corpus for `tests/test_damage_parity.py` |
| Action masking fix (5-move overflow) | Carried into the adapter table |
| `benchmark_model.py`, `elo.py`, `replay_viewer.py`, `dashboard.py` | Kept, pointed at the new player |
| `train_ppo.py`, `policies.py`, `self_play.py`, `checkpoint_pool.py`, `adaptive_opponent.py`, `run_curriculum.py` | Retired once Phase 6 trains |
| Reward levers | Retired. hp/faint delta survives as an annealed knob |

---

## 14. Timeline and milestones

Assumes WSL2 or Linux with a 4090/5090-class GPU and about 10 hours a week
from you. Weeks are calendar weeks.

| Milestone | Week | Exit criterion |
|---|---|---|
| M0 Oracle + data | 2 | 100 seeded Showdown games produce canonical traces; tables and team bank packed |
| M1 Damage parity | 3 | 0 mismatches on existing cases plus 5k fuzzed vs `@smogon/calc`, under jit/vmap |
| M2 Engine core | 5 | Full battles on the data-only subset, 1k stub games diff-clean, vmap(4096) jit compiles |
| M3 Coverage 60% | 9 | Items + stat/immunity abilities + switch-in abilities + core moves ported and verified |
| M4 Model + obs | 12 | Transformer forward/backward at target speed; opp-set head learns priors |
| M5 Coverage 90% | 14 | Batches 5 to 7 done; throughput above 100k steps/s |
| M6 First training run | 16 | PPO league runs 24h, Elo rises, beats SimpleHeuristics > 95% via adapter |
| M7 Coverage 99% + parity | 18 | 20k stub games diff-clean; distributional parity on 100k |
| M8 Ladder | 20 | Bot on ladder with Stage A search; GXE reported |
| M9 Stage B search | later | Only if A shows a gain |

If the GPU or time budget is smaller, the order holds and the weeks stretch;
nothing in the plan depends on finishing fast.

---

## 15. Risks and what we do about them

| Risk | Mitigation |
|---|---|
| Windows: no CUDA JAX | WSL2 from day one; all scripts assume a POSIX shell |
| Consumption-order PRNG matching never works | Stub mode and distributional parity are the primary gates; bit-exact is a stretch, not a dependency |
| A fixed-shape-breaking effect (Illusion, Trace, formes) stalls the core | Designed in Phase 2 before transcription; `wontfix` is allowed in `coverage.yaml` and the bank filter simply excludes those sets, which costs coverage, not correctness |
| The bot learns engine bugs | Coverage status only reaches `verified` through fuzz; distributional parity at every milestone; ladder games are replayed through the adapter and diffed |
| Showdown randbats sets drift | Pinned commit; a documented regen procedure; a ladder-time check that the opponent's revealed set exists in our bank |
| jit compile times explode as handlers grow | Table-driven handlers, not `lax.switch` chains; measure compile time per batch |
| Transformer throughput below target | d_model 192 / 4 layers fallback; both references are "small" models for a reason |

---

## 16. First concrete actions

1. You: confirm GPU, OS (WSL2?), and weekly hours. Install
   `jax[cuda12]`, `flax`, `optax`, `chex`, `jaxtyping`, `orbax-checkpoint`,
   Node 20, and clone Showdown at a pinned commit. Run
   `python tools/mechanics_inventory.py third_party/pokemon-showdown` and
   commit the resulting `docs/mechanics_inventory.json`.
2. Me: Phase 0 in full (export script, table packer, team bank generator and
   packer, oracle worker, canonicalizer, fuzz driver), as one PR with its own
   tests, so M0 is done by the time you have the environment up.
3. Together: Phase 1, the damage calc port, as the JAX on-ramp. I scaffold
   `engine/damage.py` with the first few modifiers and the parity test; you
   port the next modifiers from your own `_manual_damage_calc` and watch the
   test go green.
4. Me: Phase 2 design doc for state, passes and the three fixed-shape
   breakers, for your review before any effect is transcribed.
