"""Custom PPO agent — pure PyTorch, no stable_baselines3.

Collection and update are split into separate methods so the RTDE loop can
drive env.step() itself and only call `learn_from_buffer()` during natural
pauses (e.g. between bow strokes) without blocking the 500Hz control loop.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.ppo.networks import ACT_DIM, OBS_DIM, Actor, Critic
from rl.ppo.rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(
        self,
        env,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        lr: float = 3e-4,
        n_steps: int = 256,
        n_epochs: int = 4,
        batch_size: int = 64,
        clip_range: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.env = env
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        # Double-buffering: the RTDE thread reads these frozen copies via
        # sample(); the RL thread trains self.actor/self.critic and calls
        # _sync_inference_copies() at the end of each update. Keeps concurrent
        # reads stable without any locking.
        self.actor_infer = Actor(obs_dim, act_dim).to(self.device).eval()
        self.critic_infer = Critic(obs_dim).to(self.device).eval()
        for p in self.actor_infer.parameters():
            p.requires_grad_(False)
        for p in self.critic_infer.parameters():
            p.requires_grad_(False)

        self.buffer = RolloutBuffer(n_steps, obs_dim, act_dim, device=str(self.device))

        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl   # early-stop epochs when |approx_kl| exceeds this

        self._last_obs: Optional[np.ndarray] = None

        self._sync_inference_copies()

    # ------------------------------------------------------------------
    def _sync_inference_copies(self):
        """Copy trained weights into the frozen inference copies. Called at the
        end of every update so the RTDE thread's next sample() reads a fresh,
        consistent snapshot."""
        self.actor_infer.load_state_dict(self.actor.state_dict())
        self.critic_infer.load_state_dict(self.critic.state_dict())

    # ------------------------------------------------------------------
    # Inference — signature matches sb3 for drop-in infer_ppo.py usage.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, obs, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor_infer.act(obs_t, deterministic=deterministic)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        action_np = np.clip(action_np, -1.0, 1.0)
        return action_np, None

    # ------------------------------------------------------------------
    # RTDE-side sampling: one call per tick. Reads only the frozen inference
    # copies, so it is safe to call concurrently with a running update().
    # Returns everything the external loop will need to pass back into update().
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob = self.actor_infer.act(obs_t, deterministic=False)
        value = self.critic_infer(obs_t)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        clipped_action = np.clip(action_np, -1.0, 1.0)
        return {
            "action": action_np,
            "clipped_action": clipped_action,
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
        }

    # ------------------------------------------------------------------
    # High-level update — what Arvind's RL thread calls. Hides GAE, buffer,
    # epochs, and weight sync. transitions is a list of dicts produced by
    # sample() + the external env loop; last_obs is the observation after the
    # final transition (needed to bootstrap the last value).
    # ------------------------------------------------------------------
    def update(self, transitions, last_obs):
        # Robot episodes vary in length (note timing vs tick count), and the
        # robot loop may accumulate SEVERAL strokes per update — size the buffer
        # to what actually arrived instead of silently requiring n_steps.
        if len(transitions) != self.buffer.n_steps:
            self.buffer = RolloutBuffer(len(transitions), self.buffer.obs_dim,
                                        self.buffer.act_dim, device=str(self.device))
        self.buffer.reset()
        for t in transitions:
            self.buffer.add(
                obs=t["obs"],
                action=t["action"],
                log_prob=t["log_prob"],
                value=t["value"],
                reward=t["reward"],
                done=t["done"],
            )
        with torch.no_grad():
            last_obs_t = torch.as_tensor(
                last_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            last_value = float(self.critic_infer(last_obs_t).item())
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        return self.learn_from_buffer()

    # ------------------------------------------------------------------
    # Rollout collection — env I/O only, no gradient work.
    # ------------------------------------------------------------------
    def collect_rollout(self):
        self.buffer.reset()

        if self._last_obs is None:
            obs, _ = self.env.reset()
            self._last_obs = obs

        ep_rewards, ep_lengths = [], []
        running_reward, running_length = 0.0, 0

        for _ in range(self.n_steps):
            obs_t = torch.as_tensor(
                self._last_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = self.actor_infer.act(obs_t, deterministic=False)
                value = self.critic_infer(obs_t)

            action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
            clipped_action = np.clip(action_np, -1.0, 1.0)

            next_obs, reward, terminated, truncated, _ = self.env.step(clipped_action)
            done = bool(terminated or truncated)

            # Store pre-clip action so ratios stay consistent with the sampled density.
            self.buffer.add(
                obs=self._last_obs,
                action=action_np,
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                reward=float(reward),
                done=done,
            )

            running_reward += reward
            running_length += 1

            if done:
                ep_rewards.append(running_reward)
                ep_lengths.append(running_length)
                running_reward, running_length = 0.0, 0
                next_obs, _ = self.env.reset()

            self._last_obs = next_obs

        with torch.no_grad():
            last_obs_t = torch.as_tensor(
                self._last_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            last_value = float(self.critic_infer(last_obs_t).item())

        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        return {"ep_rewards": ep_rewards, "ep_lengths": ep_lengths}

    # ------------------------------------------------------------------
    # Update — pure compute. Safe to call between bow strokes.
    # ------------------------------------------------------------------
    def learn_from_buffer(self):
        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_frac": [],
        }

        stop_early = False
        for _ in range(self.n_epochs):
            if stop_early:
                break
            for batch in self.buffer.get_minibatches(self.batch_size):
                dist = self.actor(batch["obs"])
                new_log_probs = dist.log_prob(batch["actions"]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                adv = batch["advantages"]

                pg1 = ratio * adv
                pg2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                policy_loss = -torch.min(pg1, pg2).mean()

                values = self.critic(batch["obs"])
                value_loss = ((values - batch["returns"]) ** 2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                # Clip actor and critic SEPARATELY (2026-07-10 review rl-soundness-2):
                # a joint norm let the critic's large early-training gradients consume
                # the whole budget and scale the ACTOR's gradient toward zero in exactly
                # the informative batches (measured x0.063 on the hacked run).
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch["old_log_probs"] - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_frac"].append(clip_frac)

                if self.target_kl is not None and abs(approx_kl) > self.target_kl:
                    stop_early = True   # policy moved far enough this update
                    break

        self._sync_inference_copies()
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Simulation training convenience loop.
    # ------------------------------------------------------------------
    def learn(self, total_timesteps: int, log_interval: int = 1):
        n_updates = max(1, total_timesteps // self.n_steps)
        history = []
        for update in range(1, n_updates + 1):
            rollout_info = self.collect_rollout()
            train_info = self.learn_from_buffer()

            ep_rews = rollout_info["ep_rewards"]
            mean_ep_reward = float(np.mean(ep_rews)) if ep_rews else float("nan")
            entry = {
                "update": update,
                "timesteps": update * self.n_steps,
                "mean_ep_reward": mean_ep_reward,
                **train_info,
            }
            history.append(entry)

            if update % log_interval == 0:
                print(
                    f"update {update:4d} | steps {entry['timesteps']:7d} "
                    f"| ep_rew {mean_ep_reward:+.3f} "
                    f"| pi {entry['policy_loss']:+.4f} "
                    f"| v {entry['value_loss']:.4f} "
                    f"| H {entry['entropy']:.3f} "
                    f"| kl {entry['approx_kl']:+.4f} "
                    f"| clip {entry['clip_frac']:.2f}"
                )
        return history

    # ------------------------------------------------------------------
    def save(self, path, meta=None):
        """meta: optional dict of run context (nominal_mm, obs_dim, bow_speed, reward
        version...) stored INSIDE the checkpoint so eval/ab can refuse a mismatched
        deployment (2026-07-10 review provenance-4: a nominal-0 policy evaluated at the
        default +2mm nominal shifts every commanded depth silently)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "meta": dict(meta) if meta else {},
            },
            str(path),
        )

    def load(self, path):
        """Returns the checkpoint's meta dict ({} for pre-2026-07-10 checkpoints)."""
        ckpt = torch.load(str(path), map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                pass
        self._sync_inference_copies()
        return ckpt.get("meta", {})
