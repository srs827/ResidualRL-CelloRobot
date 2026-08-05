"""On-policy rollout buffer with GAE-lambda advantage estimation."""

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.values = np.zeros(self.n_steps, dtype=np.float32)
        self.rewards = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros(self.n_steps, dtype=np.float32)
        self.advantages = np.zeros(self.n_steps, dtype=np.float32)
        self.returns = np.zeros(self.n_steps, dtype=np.float32)
        self.ptr = 0

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def add(self, obs, action, log_prob, value, reward, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ):
        # dones[t] marks the boundary between step t and t+1; mask stops both the
        # value bootstrap and the GAE recursion across episode resets.
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size: int):
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(self.returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(adv, dtype=torch.float32, device=self.device)

        idx = np.arange(self.n_steps)
        np.random.shuffle(idx)

        for start in range(0, self.n_steps, batch_size):
            mb = idx[start : start + batch_size]
            yield {
                "obs": obs[mb],
                "actions": actions[mb],
                "old_log_probs": old_log_probs[mb],
                "returns": returns[mb],
                "advantages": advantages[mb],
            }
