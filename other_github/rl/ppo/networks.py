"""Actor (Gaussian policy) and Critic networks for custom PPO."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


OBS_DIM = 16
ACT_DIM = 1
HIDDEN = 64

# std ≈ 0.3 so most samples land inside action_space [-1, 1] before clipping.
LOG_STD_INIT = float(np.log(0.3))
# Floor std at ~0.1: on tiny correlated robot batches a free log_std can ratchet
# to near-deterministic before the mean converges, killing exploration (audit K10;
# pairs with the locked ent_coef=0.1 decision).
LOG_STD_MIN = -2.3
LOG_STD_MAX = 2.0


def _orthogonal(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp_trunk(obs_dim: int = OBS_DIM) -> nn.Sequential:
    return nn.Sequential(
        _orthogonal(nn.Linear(obs_dim, HIDDEN), gain=np.sqrt(2)),
        nn.ReLU(),
        _orthogonal(nn.Linear(HIDDEN, HIDDEN), gain=np.sqrt(2)),
        nn.ReLU(),
    )


class Actor(nn.Module):
    """16-dim obs → Gaussian over 1-dim delta_force. State-independent log_std."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
        super().__init__()
        self.trunk = _mlp_trunk(obs_dim)
        # Small gain on the mean head so early policy is near zero (tiny residual).
        self.mean_head = _orthogonal(nn.Linear(HIDDEN, act_dim), gain=0.01)
        self.log_std = nn.Parameter(
            torch.full((act_dim,), LOG_STD_INIT, dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_head(self.trunk(obs))
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class Critic(nn.Module):
    """16-dim obs → scalar value."""

    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.trunk = _mlp_trunk(obs_dim)
        self.value_head = _orthogonal(nn.Linear(HIDDEN, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(obs)).squeeze(-1)
