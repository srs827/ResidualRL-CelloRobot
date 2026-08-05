"""
reward/reward_fn.py

Complete reward function for single-string residual RL.

Components (weighted sum):
    quality   (0.45) — smoothed classifier score
    trend     (0.10) — is quality improving within episode?
    dynamic   (0.20) — does force match score's dynamic marking?
    bow       (0.15) — bow management (not running out)
    timing    (0.05) — note duration accuracy
    smooth    (0.05) — penalize jerky force changes

Human penalty is additive (not weighted) — always applied at full strength.

Usage:
    reward_fn = CompleteCelloReward(classifier_model=mock_classifier)
    reward_fn.reset()

    # each timestep:
    reward, components = reward_fn.compute(
        audio_chunk, physical_params, robot_state, score_context
    )
"""

import numpy as np
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from constants import BASELINE_FORCE, DYNAMIC_MAP, REWARD_WEIGHTS
from reward.sliding_window import SlidingWindowClassifier


class CompleteCelloReward:

    HUMAN_PENALTY = -1.0   # applied additively per flagged timestep

    def __init__(
        self,
        classifier_model,
        sr             = 44100,
        window_sec     = 0.5,
        ema_alpha      = 0.7,    # smoothing factor — higher = smoother
        trend_window   = 10,     # timesteps for trend computation
    ):
        self.window_clf    = SlidingWindowClassifier(
            model      = classifier_model,
            sr         = sr,
            window_sec = window_sec,
        )
        self.ema_alpha     = ema_alpha
        self.ema_score     = None
        self.score_history = deque(maxlen=trend_window)

        # Human feedback state
        self._human_flag_steps = 0   # timesteps remaining for penalty

        # Previous force — for smoothness calculation
        self._prev_force = None

        # Episode logging
        self.reward_log  = []

    # ── Main call ────────────────────────────────────────────────

    def compute(
        self,
        audio_chunk,       # np.ndarray or None (mock)
        physical_params,   # np.ndarray (6,) — measured values
        robot_state,       # dict from baseline.get_simulated_state()
        score_context,     # dict from baseline.get_score_context()
    ) -> tuple:
        """
        Compute reward for one timestep.
        Returns (total_reward: float, components: dict)
        """
        string = score_context.get('current_string', 'A') or 'A'
        if '-' in string:
            string = string.split('-')[-1]   # use destination string

        r_quality, r_trend = self._quality_reward(audio_chunk, physical_params, string)
        r_dynamic          = self._dynamic_reward(robot_state, score_context)
        r_bow              = self._bow_reward(robot_state)
        r_timing           = self._timing_reward(robot_state, score_context)
        r_smooth           = self._smooth_reward(robot_state)
        r_human            = self._human_penalty()

        w = REWARD_WEIGHTS
        total = (
            w['quality'] * r_quality +
            w['trend']   * r_trend   +
            w['dynamic'] * r_dynamic +
            w['bow']     * r_bow     +
            w['timing']  * r_timing  +
            w['smooth']  * r_smooth  +
            r_human                   # additive
        )

        components = {
            'total':   float(total),
            'quality': float(r_quality),
            'trend':   float(r_trend),
            'dynamic': float(r_dynamic),
            'bow':     float(r_bow),
            'timing':  float(r_timing),
            'smooth':  float(r_smooth),
            'human':   float(r_human),
        }
        self.reward_log.append(components)

        return float(total), components

    # ── Component implementations ─────────────────────────────────

    def _quality_reward(self, audio_chunk, physical_params, string):
        """
        Primary reward: smoothed classifier score.
        Also returns trend bonus for improving quality.
        """
        raw_score = self.window_clf.push(audio_chunk, physical_params, string)

        # Buffer still filling — return neutral
        if raw_score is None:
            return 0.5, 0.0

        # EMA smoothing
        if self.ema_score is None:
            self.ema_score = raw_score
        else:
            self.ema_score = (self.ema_alpha * self.ema_score
                              + (1 - self.ema_alpha) * raw_score)

        # Trend — is quality improving over recent history?
        self.score_history.append(self.ema_score)
        r_trend = 0.0
        if len(self.score_history) >= 5:
            x     = np.arange(len(self.score_history))
            slope = np.polyfit(x, list(self.score_history), 1)[0]
            # Scale: slope of 0.01/step = +1.0, -0.01/step = -1.0
            r_trend = float(np.clip(slope * 100, -1.0, 1.0))

        return float(self.ema_score), r_trend

    def _dynamic_reward(self, robot_state, score_context):
        """
        Reward for matching the score's dynamic marking.
        Maps dynamic (pp/mf/ff) → target force multiplier.
        """
        string  = score_context.get('current_string', 'A') or 'A'
        if '-' in string:
            return 0.0   # no dynamic target during transition

        dynamic    = score_context.get('current_dynamic', 'mf') or 'mf'
        multiplier = DYNAMIC_MAP.get(dynamic, 0.6) / 0.6   # normalize around mf=1.0
        target_f   = BASELINE_FORCE.get(string, 3.0) * multiplier
        actual_f   = robot_state.get('measured_force', target_f)

        error    = abs(actual_f - target_f) / max(target_f, 0.1)
        r_dynamic = float(np.clip(1.0 - error, 0.0, 1.0))
        return r_dynamic

    def _bow_reward(self, robot_state):
        """
        Penalize approaching bow limits.
        Returns 0.0 when bow is healthy, negative when running low.
        """
        bow_pos   = robot_state.get('bow_position', 0.5)
        bow_dir   = robot_state.get('bow_direction', 1.0)

        # Remaining bow in current direction
        remaining = (1.0 - bow_pos) if bow_dir > 0 else bow_pos

        if remaining > 0.30:
            return 0.0
        elif remaining > 0.10:
            # Soft linear penalty: 0 at 30%, -0.5 at 10%
            return -0.5 * (0.30 - remaining) / 0.20
        elif remaining > 0.0:
            # Stronger penalty: -0.5 at 10%, -1.0 at 0%
            return -0.5 - 0.5 * (0.10 - remaining) / 0.10
        else:
            return -1.0   # completely out — episode should terminate

    def _timing_reward(self, robot_state, score_context):
        """
        Penalize note duration errors.
        Only active near end of note (last 30% of duration).
        """
        expected = score_context.get('note_duration', None)
        if expected is None or expected <= 0:
            return 0.0

        elapsed  = robot_state.get('time_in_current_note', 0.0)
        progress = elapsed / max(expected, 0.01)

        # Too early to judge timing
        if progress < 0.70:
            return 0.0

        error    = abs(elapsed - expected) / max(expected, 0.01)
        r_timing = -float(np.clip(error, 0.0, 1.0))
        return r_timing

    def _smooth_reward(self, robot_state):
        """
        Penalize large force changes between timesteps.
        Jerky force changes cause audible artifacts.
        """
        current_force = robot_state.get('measured_force', 3.0)

        if self._prev_force is None:
            self._prev_force = current_force
            return 0.0

        delta            = abs(current_force - self._prev_force)
        self._prev_force = current_force

        MAX_DELTA = 0.3   # N — full penalty at this delta
        r_smooth  = -float(np.clip(delta / MAX_DELTA, 0.0, 1.0))
        return r_smooth

    def _human_penalty(self):
        """
        Additive penalty when human flags a mistake.
        Lasts for N timesteps around the flag.
        """
        if self._human_flag_steps > 0:
            self._human_flag_steps -= 1
            return self.HUMAN_PENALTY
        return 0.0

    # ── Human feedback interface ──────────────────────────────────

    def register_human_flag(self, n_timesteps: int = 7):
        """
        Call when human presses mistake button.
        Activates penalty for n_timesteps.
        """
        self._human_flag_steps = n_timesteps

    # ── Episode utilities ─────────────────────────────────────────

    def reset(self):
        """Call at start of each episode."""
        self.window_clf.reset()
        self.ema_score         = None
        self.score_history.clear()
        self._human_flag_steps = 0
        self._prev_force       = None
        self.reward_log        = []

    def episode_summary(self) -> dict:
        """Call at end of episode for logging."""
        if not self.reward_log:
            return {}
        keys = self.reward_log[0].keys()
        return {
            f'{k}_mean': float(np.mean([r[k] for r in self.reward_log]))
            for k in keys
        }