from __future__ import annotations

from typing import Any

import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from brent_agent import VECTOR_LENGTH


class ObservationExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Any):
        super().__init__(observation_space, features_dim=VECTOR_LENGTH)
        self.register_buffer("running_mean", th.zeros(VECTOR_LENGTH))
        self.register_buffer("running_var", th.ones(VECTOR_LENGTH))
        self.register_buffer("count", th.tensor(1e-4))

    def forward(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        x = obs["observation"]
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


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[128, 128],
            features_extractor_class=ObservationExtractor,
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
