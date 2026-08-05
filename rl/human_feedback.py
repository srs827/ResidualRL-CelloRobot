"""
rl/human_feedback.py

Human-in-the-loop feedback for RL training.

Design:
    During episode:  human presses key → timestep + state logged
    After episode:   flags applied retroactively to rollout buffer
                     before PPO gradient update
    Offline:         flagged audio reviewed and categorized
                     quality flags added to classifier dataset

The human is NOT in the real-time RL loop — flags are
applied between episodes, not per timestep.
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
import numpy as np


class HumanFeedbackLogger:
    """
    Monitors keyboard for human mistake flags during RL training.

    Usage:
        logger = HumanFeedbackLogger(flag_key='f', log_path='flags.jsonl')
        logger.start()

        # At start of each episode:
        logger.set_episode(episode_id=42, step_ref=step_counter)

        # After each episode:
        flags = logger.get_episode_flags()
        logger.clear_episode()

        logger.stop()
    """

    def __init__(
        self,
        flag_key       = 'f',
        log_path       = 'rl_flags.jsonl',
        penalty_before = 3,    # timesteps before flag to penalize
        penalty_after  = 4,    # timesteps after flag to penalize
    ):
        self.flag_key       = flag_key
        self.log_path       = Path(log_path)
        self.penalty_before = penalty_before
        self.penalty_after  = penalty_after

        self._episode_id    = None
        self._step_ref      = [0]       # mutable ref — updated externally
        self._episode_flags = []
        self._all_flags     = []
        self._listener      = None
        self._running       = False

    def start(self):
        """Start keyboard monitoring in background thread."""
        try:
            from pynput import keyboard
            self._listener = keyboard.Listener(on_press=self._on_key)
            self._listener.start()
            print(f"Human feedback active. Press '{self.flag_key}' to flag mistake.")
        except ImportError:
            print("pynput not installed — human feedback disabled.")
            print("Install with: pip install pynput")

    def stop(self):
        if self._listener:
            self._listener.stop()

    def set_episode(self, episode_id: int, step_ref: list):
        """
        Call at start of each episode.
        step_ref: single-element list [0] — update step_ref[0] each timestep
                  so the logger always knows the current step.
        """
        self._episode_id    = episode_id
        self._step_ref      = step_ref
        self._episode_flags = []

    def get_episode_flags(self) -> list:
        """Returns flags collected during this episode."""
        return self._episode_flags.copy()

    def has_flags(self) -> bool:
        return len(self._episode_flags) > 0

    def clear_episode(self):
        self._episode_flags = []

    def total_flags(self) -> int:
        return len(self._all_flags)

    def flag_rate(self, last_n_episodes: int = 100) -> float:
        """Flags per episode — should decrease as policy improves."""
        if self._episode_id is None:
            return 0.0
        recent = [f for f in self._all_flags
                  if f['episode_id'] >= self._episode_id - last_n_episodes]
        return len(recent) / max(last_n_episodes, 1)

    def _on_key(self, key):
        try:
            if hasattr(key, 'char') and key.char == self.flag_key:
                self._register_flag()
        except Exception:
            pass

    def _register_flag(self):
        step = self._step_ref[0]

        flag = {
            'episode_id':      self._episode_id,
            'step':            step,
            'timestamp':       datetime.now().isoformat(),
            'penalize_steps':  list(range(
                max(0, step - self.penalty_before),
                step + self.penalty_after + 1
            )),
        }

        self._episode_flags.append(flag)
        self._all_flags.append(flag)

        # Log immediately — don't lose flags on crash
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(flag) + '\n')

        print(f"  [HUMAN FLAG] episode={self._episode_id} step={step}")


def apply_human_flags_to_buffer(buffer, episode_flags: list, penalty: float = -1.0):
    """
    Retroactively apply human penalty to flagged timesteps in rollout buffer.

    Called AFTER episode ends, BEFORE learn_from_buffer() / PPO update.
    Modifies buffer.rewards in-place, then recomputes advantages.

    Args:
        buffer:         RolloutBuffer instance (from ppo.py)
        episode_flags:  list of flag dicts from HumanFeedbackLogger
        penalty:        additive reward penalty (negative)
    """
    if not episode_flags:
        return buffer

    penalize_steps = set()
    for flag in episode_flags:
        for step in flag.get('penalize_steps', []):
            if 0 <= step < buffer.ptr:
                penalize_steps.add(step)

    if not penalize_steps:
        return buffer

    for step in penalize_steps:
        buffer.rewards[step] += penalty

    # Recompute advantages with modified rewards
    buffer.compute_returns_and_advantages(
        last_value = 0.0,
        gamma      = 0.99,
        gae_lambda = 0.95,
    )

    print(f"  Applied human penalty to {len(penalize_steps)} timesteps")
    return buffer