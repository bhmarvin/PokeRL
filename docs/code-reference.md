# Code Reference

## Core Files

### `brent_agent.py`
Main environment customization file.

Contains:

- reward shaping config (`REWARD_CONFIG`)
- observation tensor builder (`BrentObservationVectorBuilder`, 758 dims)
- damage estimation logic (Gen 9 manual damage formula)
- opponent threat prior logic (Bayesian role posterior)
- tactical shaping and strategic penalty tracking
- the custom `SinglesEnv` subclass

Main symbols:

```python
class BrentObservationVectorBuilder:
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray
    def verify_battle_embedding(self, battle: AbstractBattle, vector: np.ndarray) -> list[str]

class BrentsRLAgent(SinglesEnv):
    def calc_reward(self, battle: AbstractBattle) -> float
    def action_to_order(self, action: np.int64, battle: AbstractBattle, fake: bool = False, strict: bool = True) -> BattleOrder
    def get_strategic_penalty_report(self) -> dict[str, Any]
    def get_tactical_shaping_report(self) -> dict[str, Any]
```

Inputs / outputs:

- `embed_battle(...)`:
  - input: live `AbstractBattle`
  - output: `np.ndarray` of shape `(758,)`
- `calc_reward(...)`:
  - input: live `AbstractBattle`
  - output: scalar reward = base delta + tactical shaping + head hunter bonus

### `train_ppo.py`
Main training entry point.

Contains:

- CLI argument parsing
- multi-env creation (`SubprocVecEnv` with `--n-envs` and `--train-opponents`)
- PPO construction and checkpoint loading
- periodic evaluation callback
- final summary writing

Main symbols:

```python
def parse_args() -> argparse.Namespace
def create_subproc_env(args: argparse.Namespace) -> SubprocVecEnv
def evaluate_policy(model: PPO, args: argparse.Namespace) -> tuple[EvalSummary, list[EvalBattleRecord], dict[str, Any], dict[str, Any]]
def build_model(args: argparse.Namespace, env: SubprocVecEnv) -> PPO
def main() -> None

class PokeEnvEvalCallback(BaseCallback):
    def _on_step(self) -> bool
```

Key CLI knobs:

- `--train-timesteps`
- `--resume-from`
- `--train-opponents` (comma-separated, one per env)
- `--n-envs`
- `--eval-opponent`
- `--eval-freq`
- `--save-freq`
- `--eval-battles`
- `--device`
- `--run-name`
- `--learning-rate`

### `policies.py`
Custom masked PPO policy with structured feature extraction.

Main symbols:

```python
class StructuredObservationExtractor(BaseFeaturesExtractor):
    # 9 semantic block encoders with shared weights for repeated structures
    # Output: 705 dims (was 561)
    # Block encoder sizes: global=32, my_active=64, opp_active=48, speed=1(passthrough),
    #   move=48×4=192, my_bench=48×5=240, opp_bench=16×5=80, targeting=16, threat=32
    def forward(self, obs: dict[str, th.Tensor]) -> th.Tensor

class MaskedActorCriticPolicy(ActorCriticPolicy):
    # net_arch=[512, 256, 128]
    # Action masking via masked_fill with finfo.min on illegal logits
    def forward(self, obs: dict[str, th.Tensor], deterministic: bool = False)
    def evaluate_actions(self, obs: dict[str, th.Tensor], actions: th.Tensor)
    def get_distribution(self, obs: dict[str, th.Tensor])
```

### `opponents.py`
Maps string names to baseline opponents.

```python
OPPONENT_CHOICES = ("random", "max_base_power", "simple_heuristic")
def create_opponent(opponent_name: str, *, battle_format: str, log_level: int, start_listening: bool = False) -> Player
```

### `randbats_data.py`
Randbats prior database wrapper.

```python
class RandbatsMeta:
    def get_species_data(self, species: str) -> Optional[Dict[str, Any]]
    def calculate_stat(self, base: int, level: int, ev: int = 0, iv: int = 31, nature: float = 1.0, is_hp: bool = False) -> int
    def get_role_stats(self, species_name: str, role_name: str, base_stats: Dict[str, int]) -> Dict[str, int]
    def filter_roles(self, species_name: str, revealed_moves: List[str], revealed_item: Optional[str] = None) -> Dict[str, float]
    def get_move_marginals(self, species_name: str, role_weights: Dict[str, float]) -> Dict[str, float]
```

### `inspect_battle_debug.py`
Interactive debugging utility for stepping through live battles and comparing embedding values.

### `obs_audit.py`
Observation-space audit utility for inspecting per-index min/max/mean/std and sparsity across live battles.

### `benchmark_model.py`
Benchmark entry point for evaluating trained checkpoints against multiple opponent types.

## Data Flow

### Training Flow
1. `train_ppo.py` parses CLI args.
2. `create_subproc_env(...)` builds `SubprocVecEnv` with N `BrentsRLAgent` instances plus chosen opponents.
3. `build_model(...)` creates PPO or loads from checkpoint.
4. PPO requests observations from `embed_battle(...)` — raw [0,1] features, no runtime normalization.
5. `StructuredObservationExtractor` slices into 9 blocks, encodes each, concatenates to 705 dims.
6. `MaskedActorCriticPolicy` masks illegal actions and feeds [512,256,128] MLP.
7. `calc_reward(...)` returns base delta + tactical shaping + head hunter bonus.
8. `PokeEnvEvalCallback` evaluates every `eval_freq` steps.

### Observation Flow
1. Raw battle state enters `embed_battle(...)`
2. Vector builder fills global, active, move, bench, targeting, reveal-memory, threat, and tail sections (758 dims)
3. All features pre-normalized to [0,1] or [-1,1] — no Welford or runtime normalization
4. Policy reads `obs["observation"]`, action mask removes illegal logits

## Common Commands

Fresh training (multi-env):

```powershell
.\venv\Scripts\python.exe train_ppo.py --train-timesteps 100000 --train-opponents "random,random,random,random" --n-envs 4 --eval-opponent random --eval-freq 10000 --eval-battles 100 --device cuda --run-name fresh_random
```

Resume with curriculum transition:

```powershell
.\venv\Scripts\python.exe train_ppo.py --train-timesteps 200000 --train-opponents "max_base_power,max_base_power,max_base_power,max_base_power" --n-envs 4 --eval-opponent max_base_power --eval-freq 10000 --eval-battles 100 --device cuda --learning-rate 1e-4 --run-name stage2 --resume-from results\ppo\stage1\best_model\best_model.zip
```

Mixed opponent training (8 envs):

```powershell
.\venv\Scripts\python.exe train_ppo.py --train-timesteps 300000 --train-opponents "simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,simple_heuristic,max_base_power" --n-envs 8 --eval-opponent simple_heuristic --eval-freq 10000 --eval-battles 100 --device cuda --learning-rate 1e-4 --run-name stage3 --resume-from results\ppo\stage2\best_model\best_model.zip
```

TensorBoard:

```powershell
.\venv\Scripts\tensorboard.exe --logdir results\ppo\logs --port 6006
```

Benchmark:

```powershell
.\venv\Scripts\python.exe benchmark_model.py --checkpoint "results\ppo\some_run\best_model\best_model.zip" --algo ppo --n-battles 100 --device cuda
```
