"""
cello_env.py

Custom Gym environment for SAC-based bow force control on the UR5e.

The sound classifier is baked in as the reward source inside step().
SAC never sees the classifier directly — it just optimizes the reward signal
that the classifier produces. Swap MockSoundClassifier for your real one.

Observation (16-dim):
    tcp_pose       [0:6]   — x, y, z (m), rx, ry, rz (rad)
    force_torque   [6:12]  — fx, fy, fz (N), tx, ty, tz (Nm)  [F/T sensor]
    bow_speed      [12]    — m/s
    bow_accel      [13]    — m/s²
    bow_position   [14]    — 0.0 (frog) → 1.0 (tip)
    current_force  [15]    — commanded normal force (N)

Action (1-dim):
    delta_force    [0]     — normalised [-1, 1], scaled to ±MAX_DELTA_N per step
                            agent outputs how much to change the bow force

Episode = one note stroke (max_steps timesteps).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Classifier interface — swap in your real model here
# ---------------------------------------------------------------------------

class SoundClassifierBase(ABC):
    """
    Abstract base class for the sound classifier.
    predict() receives a mono float32 audio chunk and returns
    (label, confidence) where label ∈ {"Good", "Bad"} and confidence ∈ [0, 1].

    The optional current_force kwarg is used by MockSoundClassifier to shape
    the reward around the applied force. Real classifiers should accept and
    ignore it (signature compatibility).
    """
    @abstractmethod
    def predict(self, audio: np.ndarray, current_force: float = None) -> tuple[str, float]:
        ...


class MockSoundClassifier(SoundClassifierBase):
    """
    Simulates a classifier during training/testing without hardware.

    Reward shape: force near OPTIMAL_FORCE → high Good confidence.
    This gives the SAC agent a meaningful signal to learn from.
    """
    OPTIMAL_FORCE = 3.5   # N  — the "ideal" bow pressure for the mock
    SIGMA = 1.5           # controls how quickly quality degrades away from optimal

    def __init__(self, noise: float = 0.05):
        self.noise = noise  # adds realistic stochasticity

    def predict(self, audio: np.ndarray, current_force: float = 3.5) -> tuple[str, float]:
        # Gaussian-shaped quality centred on optimal force
        quality = np.exp(-0.5 * ((current_force - self.OPTIMAL_FORCE) / self.SIGMA) ** 2)
        quality = float(np.clip(quality + self.noise * np.random.randn(), 0.0, 1.0))
        label = "Good" if quality >= 0.5 else "Bad"
        confidence = quality if label == "Good" else 1.0 - quality
        return label, confidence


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CelloEnv(gym.Env):
    """
    SAC environment for bow force control.

    Hardware methods (_get_obs, _get_audio, _apply_force) are stubs here.
    Subclass and override them to connect real RTDE + audio capture.
    """

    metadata = {"render_modes": []}

    # Observation indices (for readability elsewhere)
    IDX_TCP   = slice(0, 6)
    IDX_FT    = slice(6, 12)
    IDX_SPEED = 12
    IDX_ACCEL = 13
    IDX_BOW   = 14
    IDX_FORCE = 15
    OBS_DIM   = 16

    # Force limits
    FORCE_MIN     = 0.3   # N
    FORCE_MAX     = 8.0   # N
    FORCE_INIT    = 3.0   # N  — reset value
    MAX_DELTA_N   = 0.3   # N  — max force change per step (action = 1.0)

    # Reward shaping weights
    W_CLASSIFIER  = 1.0
    W_FORCE_DELTA = 0.02  # penalty for large force jumps (smooth bowing)
    W_FORCE_RANGE = 0.01  # penalty for drifting far from FORCE_INIT

    def __init__(
        self,
        classifier: SoundClassifierBase,
        max_steps: int = 150,
        audio_samples: int = 2205,   # ~50 ms @ 44100 Hz
    ):
        super().__init__()
        self.classifier   = classifier
        self.max_steps    = max_steps
        self.audio_samples = audio_samples

        # --- observation space ---
        low  = np.array([
            -1.0, -1.0, -1.0,                     # tcp xyz  (m)
            -np.pi, -np.pi, -np.pi,               # tcp rpy  (rad)
            -50., -50., -50.,                      # force xyz (N)
            -5.,  -5.,  -5.,                       # torque xyz (Nm)
            0.,                                    # bow speed (m/s)
            -10.,                                  # bow accel (m/s²)
            0.,                                    # bow position
            self.FORCE_MIN,                        # current commanded force
        ], dtype=np.float32)

        high = np.array([
            1.0,  1.0,  1.0,
            np.pi, np.pi, np.pi,
            50.,  50.,  50.,
            5.,   5.,   5.,
            2.0,
            10.,
            1.0,
            self.FORCE_MAX,
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # --- action space: normalised delta force ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._force  = self.FORCE_INIT
        self._step   = 0
        self._obs    = np.zeros(self.OBS_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    # Hardware stubs — override in a HardwareCelloEnv subclass
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return a 15-dim vector from RTDE: [tcp(6), ft(6), speed, accel, bow_pos]."""
        # Stub: return zeros (replace with real RTDE reads)
        return np.zeros(15, dtype=np.float32)

    def _get_audio(self) -> np.ndarray:
        """Return a float32 mono audio buffer for the classifier."""
        # Stub: return silence (replace with real mic/DAQ read)
        return np.zeros(self.audio_samples, dtype=np.float32)

    def _apply_force(self, force: float):
        """Send the commanded normal force to the robot via RTDE."""
        # Stub: no-op (replace with rtde.set_force(...))
        pass

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        label: str,
        confidence: float,
        prev_force: float,
    ) -> float:
        sign     = 1.0 if label == "Good" else -1.0
        r_class  = self.W_CLASSIFIER * sign * confidence
        r_smooth = -self.W_FORCE_DELTA * abs(self._force - prev_force)
        r_range  = -self.W_FORCE_RANGE * abs(self._force - self.FORCE_INIT)
        return r_class + r_smooth + r_range

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step  = 0
        # Randomise starting force so the agent sees a wide range of states
        # during training, not just FORCE_INIT. This is essential for PPO to
        # learn that high force needs a *negative* action.
        self._force = float(self.np_random.uniform(self.FORCE_MIN, self.FORCE_MAX))
        hw_obs      = self._get_obs()                      # shape (15,)
        self._obs   = np.append(hw_obs, self._force).astype(np.float32)
        return self._obs.copy(), {}

    def step(self, action: np.ndarray):
        prev_force   = self._force
        delta        = float(action[0]) * self.MAX_DELTA_N
        self._force  = float(np.clip(self._force + delta, self.FORCE_MIN, self.FORCE_MAX))

        # 1. Send force to robot
        self._apply_force(self._force)

        # 2. Get hardware observation
        hw_obs     = self._get_obs()                       # (15,)
        self._obs  = np.append(hw_obs, self._force).astype(np.float32)

        # 3. Classify sound — reward is baked in here
        # NOTE: pass current_force so MockSoundClassifier can shape the reward
        # around the actually-applied force, not the default value.
        audio              = self._get_audio()
        label, confidence  = self.classifier.predict(audio, current_force=self._force)
        reward             = self._compute_reward(label, confidence, prev_force)

        self._step  += 1
        truncated    = self._step >= self.max_steps
        terminated   = False

        info = {
            "step":       self._step,
            "force":      self._force,
            "label":      label,
            "confidence": confidence,
            "reward":     reward,
        }

        return self._obs.copy(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
