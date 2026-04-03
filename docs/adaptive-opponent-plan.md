# Adaptive Opponent Selection Plan

## Problem

Fixed opponent mixes waste training signal. Easy opponents (random at 97% WR) produce
low-variance "free wins" that don't teach anything. Hard opponents the agent can't beat
yet produce all-loss batches that collapse entropy. The sweet spot is ~40-60% win rate —
the agent is challenged but can learn from both wins and losses.

## Core Design: AdaptivePlayer

A single `Player` subclass that wraps multiple opponent strategies and dynamically
rotates based on recent performance. Lives inside each SubprocVecEnv worker —
no IPC needed.

```python
class AdaptivePlayer(Player):
    """Dynamically selects opponent difficulty based on recent win rate."""

    def __init__(self, opponents: list[Player], window: int = 20):
        self.opponents = opponents          # easy → hard
        self.current_idx = 0
        self.recent_results: deque[bool]    # last N game outcomes
        self.window = window

    def choose_move(self, battle):
        return self.opponents[self.current_idx].choose_move(battle)

    def record_result(self, won: bool):
        self.recent_results.append(won)
        if len(self.recent_results) >= self.window:
            wr = sum(self.recent_results) / len(self.recent_results)
            if wr > 0.70 and self.current_idx < len(self.opponents) - 1:
                self.current_idx += 1   # promote to harder opponent
                self.recent_results.clear()
            elif wr < 0.30 and self.current_idx > 0:
                self.current_idx -= 1   # demote to easier opponent
                self.recent_results.clear()
```

### Opponent ladder (easy → hard):
0. `RandomPlayer`
1. `MaxBasePowerPlayer`
2. `SimpleHeuristicsPlayer`
3. `SelfPlayPlayer` (frozen or refreshing checkpoint)

### How result tracking works:
- poke_env battles have `battle.won` / `battle.lost` after each episode
- Hook into `SingleAgentWrapper` or the eval callback to call `record_result()`
- Could also override `BrentsRLAgent.calc_reward` to track — it already sees `battle.won`

## Implementation Steps (DONE)

1. ~~**Create `adaptive_opponent.py`** with `AdaptivePlayer` class~~
2. ~~**Add `"adaptive"` to `OPPONENT_CHOICES`** in `opponents.py`~~
3. ~~**Wire result tracking** — uses battle object reference, checks `battle.won` on battle transition~~
4. ~~**CLI args**: `--self-play-checkpoint` for the self-play tier~~
5. ~~**Logging**: prints PROMOTED/DEMOTED messages with `flush=True` for SubprocVecEnv visibility~~

### Implementation Notes
- Result tracking uses `_check_prev_battle_result()` which detects battle transitions via `battle_tag` changes, rather than hooking into `calc_reward`.
- `--adaptive-start-tier` CLI arg added to skip easy tiers for late-stage training (e.g., `--adaptive-start-tier 2` starts at heuristic).
- SubprocVecEnv workers use `print(..., flush=True)` to ensure PROMOTED/DEMOTED messages are visible in the parent process.

## Wiring into SubprocVecEnv

Each worker gets its own `AdaptivePlayer` instance. No shared state needed — each
worker independently climbs the ladder based on its own games. This naturally creates
diversity: some workers will be on tier 2, others on tier 3, giving the PPO batch a
mix of difficulty levels.

```python
def _make_env_fn(battle_format, log_level, checkpoint_path=None):
    def _init():
        env = BrentsRLAgent(...)
        opponent = AdaptivePlayer([
            RandomPlayer(...),
            MaxBasePowerPlayer(...),
            SimpleHeuristicsPlayer(...),
            SelfPlayPlayer(checkpoint_path, ...),  # if provided
        ])
        return Monitor(SingleAgentWrapper(env, opponent))
    return _init
```

## Thresholds

| Metric | Promote (→ harder) | Demote (→ easier) | Window |
|--------|-------------------|-------------------|--------|
| Win rate | > 70% | < 30% | 20 games |

These are conservative — agent needs to convincingly beat an opponent before
moving up, and has to badly struggle before moving down. Hysteresis prevents
rapid oscillation.

---

## Reach Features (Future)

### Elo-based matchmaking
Instead of a fixed ladder, sample opponents weighted by Elo gap. Target
opponents where the agent has ~45-55% expected win rate (max information gain).
Requires maintaining per-opponent Elo in real-time, not just at eval time.
**Effort: ~150 lines. Depends on Elo tracker (done).**

### Population-based training (PBT)
Run N agents in parallel, each with different hyperparams (LR, ent_coef, gamma).
Every K steps, the worst performers copy weights+hyperparams from the best and
mutate slightly. Discovers good hyperparams automatically.
**Effort: significant. Needs Ray or custom multi-process orchestration. Multi-GPU ideal.**

### Prioritized Fictitious Self-Play (PFSP)
From OpenAI Five / AlphaStar. Maintain a league of past checkpoints. When sampling
an opponent, prioritize ones the agent has the worst win rate against (not random
sampling). Prevents "strategy forgetting" where the agent learns to beat the latest
opponent but forgets how to beat older ones.
**Effort: ~200 lines on top of CheckpointPool. Needs per-opponent win rate tracking.**

### Curriculum with automatic reward annealing
As the agent improves, reduce shaping reward magnitudes toward zero. The
terminal win/loss signal should eventually dominate. Prevents reward hacking
at high skill levels where shaping heuristics become inaccurate.
**Effort: ~30 lines. Add a decay multiplier to REWARD_CONFIG values based on
current Elo or win rate.**

### Opponent modeling / theory of mind
Train a separate network to predict the opponent's next action from battle state.
Feed this prediction into the policy as an extra feature. The agent learns to
"read" opponents and counter-predict.
**Effort: significant. New network head, new training loop, auxiliary loss.**
