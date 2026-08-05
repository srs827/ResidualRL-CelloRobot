"""
rl/env.py

Single-string residual RL environment.
Implements the Gymnasium interface.

The environment:
    - Gets baseline action from BaselineController each timestep
    - Policy outputs small residual correction
    - Combined action sent to robot (or mock)
    - Reward computed from classifier + other components
    - Episode ends when stroke sequence completes or bow runs out

Works identically with mock or real components.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from constants import (
    OBS_DIM, ACTION_DIM,
    BASELINE_FORCE, DYNAMIC_MAP, ARTICULATION_MAP,
    TIMESTEP_SEC, STEPS_PER_STROKE,
)


class SingleStringResidualEnv(gym.Env):
    """
    Gymnasium environment for single-string residual RL.

    Observation (20-dim): see _get_obs() for full breakdown
    Action (3-dim):       [dx, dy, dz] normalized [-1, 1]
                          scaled to [±3mm, ±3mm, ±2mm] TCP corrections

    Episode = one MIDI phrase on one string.
    Terminates when sequence completes or bow runs out.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        baseline,         # MockBaselineController or real BaselineController
        reward_fn,        # CompleteCelloReward
        string  = 'A',    # which string this episode runs on
        max_steps = 500,  # safety truncation
        mock_mic  = True, # True when no real microphone available
    ):
        super().__init__()

        self.baseline  = baseline
        self.reward_fn = reward_fn
        self.string    = string
        self.max_steps = max_steps
        self.mock_mic  = mock_mic

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (OBS_DIM,),
            dtype = np.float32,
        )
        self.action_space = spaces.Box(
            low   = -1.0,
            high   = 1.0,
            shape = (ACTION_DIM,),
            dtype = np.float32,
        )

        self._step_count    = 0
        self._episode_count = 0

        # Stored for obs building
        self._last_baseline_action = None
        self._last_robot_state     = None
        self._last_score_context   = None

    # ── Gymnasium API ─────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.baseline.reset(string=self.string)
        self.reward_fn.reset()

        self._step_count    = 0
        self._episode_count += 1

        # Advance baseline one step to get initial state
        self._last_baseline_action = self.baseline.get_baseline_action()
        self._last_robot_state     = self.baseline.get_simulated_state()
        self._last_score_context   = self.baseline.get_score_context()

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        """
        action: np.ndarray shape (ACTION_DIM,) in [-1, 1]
        """
        # 1. Get baseline for this timestep
        baseline_action = self._last_baseline_action
        if baseline_action is None:
            obs = self._get_obs()
            return obs, 0.0, True, False, {'reason': 'sequence_complete'}

        # 2. Apply residual to baseline
        tcp_target, force = self.baseline.apply_residual(baseline_action, action)

        # 3. Simulate execution (mock: just update baseline state)
        #    Real robot: rtde_c.servoL(tcp_target, ...) here
        robot_state = self.baseline.get_simulated_state()

        # Inject corrected force into robot state for reward computation
        robot_state['measured_force'] += (force - baseline_action['force_nominal'])
        robot_state['time_in_current_note'] = (
            baseline_action['step'] * TIMESTEP_SEC
        )

        # 4. Get audio chunk (mock: None, real: mic.record())
        audio_chunk = None if self.mock_mic else self._record_audio()

        # 5. Get physical params for classifier
        physical_params = self._get_physical_params(robot_state)

        # 6. Compute reward
        score_context = self.baseline.get_score_context()
        reward, components = self.reward_fn.compute(
            audio_chunk     = audio_chunk,
            physical_params = physical_params,
            robot_state     = robot_state,
            score_context   = score_context,
        )

        # 7. Advance baseline to next timestep
        self._last_baseline_action = self.baseline.get_baseline_action()
        self._last_robot_state     = self.baseline.get_simulated_state()
        self._last_score_context   = score_context

        self._step_count += 1

        # 8. Check termination
        terminated = (
            self.baseline.is_complete() or
            robot_state.get('bow_position', 0.5) <= 0.01 or
            robot_state.get('bow_position', 0.5) >= 0.99
        )
        truncated = self._step_count >= self.max_steps

        obs = self._get_obs()

        info = {
            **components,
            'step':     self._step_count,
            'string':   self.string,
            'force':    robot_state.get('measured_force', 0.0),
            'bow_pos':  robot_state.get('bow_position', 0.5),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    # ── Observation builder ───────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Build 20-dimensional observation vector.
        All values normalized to approximately [-1, 1].

        Breakdown:
            [0]     measured_force / 5.0
            [1]     bow_speed / 0.25
            [2]     bow_position            (already [0,1])
            [3]     bow_accel / 1.0
            [4]     force_deviation / 0.3
            [5:11]  ft sensor / [10,10,10,1,1,1]
            [11]    current_dynamic normalized
            [12]    current_articulation normalized
            [13]    time_remaining / 2.0
            [14]    beat_position           (already [0,1])
            [15]    next_dynamic normalized
            [16]    next_articulation normalized
            [17]    ema_quality_score       (already [0,1])
            [18]    quality_trending_up     (binary 0/1)
            [19]    bow_direction           (+1 or -1)
        """
        rs = self._last_robot_state   or {}
        sc = self._last_score_context or {}
        ba = self._last_baseline_action or {}

        ft = rs.get('ft', np.zeros(6))

        # Score context
        cur_dyn   = sc.get('current_dynamic', 'mf')   or 'mf'
        cur_art   = sc.get('current_articulation', 'legato') or 'legato'
        nxt_dyn   = sc.get('next_dynamic', 'mf')      or 'mf'
        nxt_art   = sc.get('next_articulation', 'legato') or 'legato'

        # Quality history
        ema   = self.reward_fn.ema_score or 0.5
        hist  = list(self.reward_fn.score_history)
        trend = float(hist[-1] > hist[0]) if len(hist) >= 2 else 0.0

        obs = np.array([
            # Robot physical state
            rs.get('measured_force', 3.0) / 5.0,
            rs.get('bow_speed', 0.0)      / 0.25,
            rs.get('bow_position', 0.5),
            rs.get('bow_accel', 0.0)      / 1.0,
            rs.get('force_deviation', 0.0)/ 0.3,

            # F/T sensor
            float(ft[0]) / 10.0,
            float(ft[1]) / 10.0,
            float(ft[2]) / 10.0,
            float(ft[3]) / 1.0,
            float(ft[4]) / 1.0,
            float(ft[5]) / 1.0,

            # Score context — current
            DYNAMIC_MAP.get(cur_dyn, 0.6),
            ARTICULATION_MAP.get(cur_art, 0.0),
            sc.get('time_remaining', 1.0) / 2.0,
            0.5,   # beat_position placeholder — implement from MIDI parser

            # Score context — lookahead
            DYNAMIC_MAP.get(nxt_dyn, 0.6),
            ARTICULATION_MAP.get(nxt_art, 0.0),

            # Quality history
            float(ema),
            float(trend),

            # Bow direction
            rs.get('bow_direction', 1.0),

        ], dtype=np.float32)

        assert obs.shape == (OBS_DIM,), (
            f"Obs shape mismatch: {obs.shape} != ({OBS_DIM},). "
            f"Check OBS_DIM in constants.py."
        )
        return obs

    # ── Helpers ───────────────────────────────────────────────────

    def _get_physical_params(self, robot_state: dict) -> np.ndarray:
        """
        6-dim physical conditioning for classifier.
        All MEASURED values.
        """
        ft = robot_state.get('ft', np.zeros(6))
        return np.array([
            robot_state.get('measured_force', 3.0),
            abs(robot_state.get('force_deviation', 0.0)),
            robot_state.get('bow_speed', 0.0),
            robot_state.get('bow_position', 0.5),
            float(ft[0]),   # lateral force
            float(ft[3]),   # torque
        ], dtype=np.float32)

    def _record_audio(self) -> np.ndarray:
        """
        Record 50ms of audio from real microphone.
        Implement when real mic is connected.
        Returns silence for now.
        """
        import sounddevice as sd
        audio = sd.rec(
            int(TIMESTEP_SEC * 44100),
            samplerate = 44100,
            channels   = 1,
            dtype      = 'float32',
        )
        sd.wait()
        return audio.flatten()