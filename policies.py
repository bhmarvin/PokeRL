from __future__ import annotations

from typing import Any

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from brent_agent import (
    VECTOR_LENGTH,
    MY_ACTIVE_START,
    MY_ACTIVE_BLOCK_SIZE,
    OPP_ACTIVE_START,
    OPP_ACTIVE_BLOCK_SIZE,
    SPEED_ADVANTAGE_INDEX,
    MY_MOVES_START,
    MOVE_BLOCK_SIZE,
    MY_BENCH_START,
    MY_BENCH_SLOT_SIZE,
    OPP_BENCH_START,
    OPP_BENCH_SLOT_SIZE,
    TARGETING_START,
    MY_TEAM_REVEALED_START,
    ON_RECHARGE_INDEX,
    ALIVE_DIFF_INDEX,
)

# Block boundaries (derived from brent_agent constants)
_GLOBAL_END = MY_ACTIVE_START                          # 0:24
_MY_ACTIVE_END = OPP_ACTIVE_START                      # 24:84
_OPP_ACTIVE_END = SPEED_ADVANTAGE_INDEX                # 84:125
_SPEED_END = SPEED_ADVANTAGE_INDEX + 1                 # 125:126
_MOVES_END = MY_BENCH_START                            # 126:226
_MY_BENCH_END = OPP_BENCH_START                        # 226:516
_OPP_BENCH_END = TARGETING_START                       # 516:616
_TARGETING_END = MY_TEAM_REVEALED_START                # 616:636
_THREAT_END = ALIVE_DIFF_INDEX + 1                     # 636:684

# Block sizes
_GLOBAL_SIZE = _GLOBAL_END                             # 24
_MY_ACTIVE_SIZE = MY_ACTIVE_BLOCK_SIZE                 # 60
_OPP_ACTIVE_SIZE = OPP_ACTIVE_BLOCK_SIZE               # 41
_TARGETING_SIZE = _TARGETING_END - TARGETING_START      # 20
_THREAT_SIZE = _THREAT_END - _TARGETING_END             # 48 (was 47, +1 alive_diff)


class StructuredObservationExtractor(BaseFeaturesExtractor):
    """Structured feature extractor that processes semantic blocks separately.

    The observation vector is sliced into 9 semantic blocks:
    global state, my active, opp active, speed, moves(×4), my bench(×5),
    opp bench(×5), targeting matrix, and threat/meta features.

    Repeated structures (moves, bench slots) use shared-weight encoders.
    All block embeddings are concatenated for the downstream MLP.
    """

    # Encoder output sizes
    _GLOBAL_OUT = 32
    _MY_ACTIVE_OUT = 64
    _OPP_ACTIVE_OUT = 48
    _SPEED_OUT = 1  # passthrough
    _MOVE_OUT = 32  # per move, ×4 = 128
    _MY_BENCH_OUT = 32  # per slot, ×5 = 160
    _OPP_BENCH_OUT = 16  # per slot, ×5 = 80
    _TARGETING_OUT = 16
    _THREAT_OUT = 32

    FEATURES_DIM = (
        _GLOBAL_OUT + _MY_ACTIVE_OUT + _OPP_ACTIVE_OUT + _SPEED_OUT
        + _MOVE_OUT * 4 + _MY_BENCH_OUT * 5 + _OPP_BENCH_OUT * 5
        + _TARGETING_OUT + _THREAT_OUT
    )  # = 561

    def __init__(self, observation_space: Any):
        super().__init__(observation_space, features_dim=self.FEATURES_DIM)

        # Welford RunningMeanStd normalization (applied to full vector before slicing)
        self.register_buffer("running_mean", th.zeros(VECTOR_LENGTH))
        self.register_buffer("running_var", th.ones(VECTOR_LENGTH))
        self.register_buffer("count", th.tensor(1e-4))

        # Block encoders
        self.global_enc = nn.Sequential(nn.Linear(_GLOBAL_SIZE, self._GLOBAL_OUT), nn.ReLU())
        self.my_active_enc = nn.Sequential(nn.Linear(_MY_ACTIVE_SIZE, self._MY_ACTIVE_OUT), nn.ReLU())
        self.opp_active_enc = nn.Sequential(nn.Linear(_OPP_ACTIVE_SIZE, self._OPP_ACTIVE_OUT), nn.ReLU())
        self.move_enc = nn.Sequential(nn.Linear(MOVE_BLOCK_SIZE, self._MOVE_OUT), nn.ReLU())
        self.my_bench_enc = nn.Sequential(nn.Linear(MY_BENCH_SLOT_SIZE, self._MY_BENCH_OUT), nn.ReLU())
        self.opp_bench_enc = nn.Sequential(nn.Linear(OPP_BENCH_SLOT_SIZE, self._OPP_BENCH_OUT), nn.ReLU())
        self.targeting_enc = nn.Sequential(nn.Linear(_TARGETING_SIZE, self._TARGETING_OUT), nn.ReLU())
        self.threat_enc = nn.Sequential(nn.Linear(_THREAT_SIZE, self._THREAT_OUT), nn.ReLU())

    def _welford_normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalize using running mean/variance (Welford's algorithm)."""
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_count = x.shape[0]
            delta = batch_mean - self.running_mean
            total = self.count + batch_count
            self.running_mean += delta * batch_count / total
            m_a = self.running_var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta**2 * self.count * batch_count / total
            self.running_var = m2 / total
            self.count = total
        std = (self.running_var + 1e-8).sqrt()
        return ((x - self.running_mean) / std).clamp(-10.0, 10.0)

    def forward(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        x = self._welford_normalize(obs["observation"])  # (batch, VECTOR_LENGTH)

        # Slice and encode each semantic block
        global_f = self.global_enc(x[:, :_GLOBAL_END])
        my_act_f = self.my_active_enc(x[:, MY_ACTIVE_START:_MY_ACTIVE_END])
        opp_act_f = self.opp_active_enc(x[:, OPP_ACTIVE_START:_OPP_ACTIVE_END])
        speed_f = x[:, SPEED_ADVANTAGE_INDEX:_SPEED_END]

        # Shared move encoder: 4 moves × MOVE_BLOCK_SIZE features each
        move_embeds = []
        for i in range(4):
            start = MY_MOVES_START + i * MOVE_BLOCK_SIZE
            end = start + MOVE_BLOCK_SIZE
            move_embeds.append(self.move_enc(x[:, start:end]))
        moves_f = th.cat(move_embeds, dim=1)  # (batch, 128)

        # Shared my-bench encoder: 5 slots × MY_BENCH_SLOT_SIZE features each
        my_bench_embeds = []
        for i in range(5):
            start = MY_BENCH_START + i * MY_BENCH_SLOT_SIZE
            end = start + MY_BENCH_SLOT_SIZE
            my_bench_embeds.append(self.my_bench_enc(x[:, start:end]))
        my_bench_f = th.cat(my_bench_embeds, dim=1)  # (batch, 160)

        # Shared opp-bench encoder: 5 slots × 20 features each
        opp_bench_embeds = []
        for i in range(5):
            start = OPP_BENCH_START + i * OPP_BENCH_SLOT_SIZE
            end = start + OPP_BENCH_SLOT_SIZE
            opp_bench_embeds.append(self.opp_bench_enc(x[:, start:end]))
        opp_bench_f = th.cat(opp_bench_embeds, dim=1)  # (batch, 80)

        target_f = self.targeting_enc(x[:, TARGETING_START:_TARGETING_END])
        threat_f = self.threat_enc(x[:, _TARGETING_END:_THREAT_END])

        return th.cat([
            global_f, my_act_f, opp_act_f, speed_f,
            moves_f, my_bench_f, opp_bench_f,
            target_f, threat_f,
        ], dim=1)  # (batch, 561)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[256, 128],
            features_extractor_class=StructuredObservationExtractor,
        )
        self._mask: th.Tensor | None = None

    def _set_action_mask(self, obs: dict[str, th.Tensor]) -> None:
        self._mask = obs["action_mask"]

    def forward(self, obs: dict[str, th.Tensor], deterministic: bool = False):
        self._set_action_mask(obs)
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs: dict[str, th.Tensor], actions: th.Tensor):
        self._set_action_mask(obs)
        return super().evaluate_actions(obs, actions)

    def get_distribution(self, obs: dict[str, th.Tensor]):
        self._set_action_mask(obs)
        return super().get_distribution(obs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        action_logits = self.action_net(latent_pi)
        if self._mask is None:
            raise RuntimeError("Action mask not set before distribution construction.")
        legal_mask = self._mask.to(dtype=th.bool)
        masked_logits = action_logits.masked_fill(
            ~legal_mask,
            th.finfo(action_logits.dtype).min,
        )
        return self.action_dist.proba_distribution(masked_logits)
