"""
zero_baseline.py — a NO-ROBOT stand-in for Samantha's TargetStringBaselineController,
for OFFLINE dry-runs of the residual-RL loop.

Same interface (reset / get_baseline_action / execute_timestep / get_full_state /
is_complete / close), but it never touches RTDE: it returns a fixed nominal action
and a synthetic mechanical state, advancing a tick counter so an "episode" is just
N ticks. Swap this for the real baseline_controller on hardware.

Author: Claude (for Zixian, 2026-06-04). Offline only.
"""


class ZeroBaseline:
    NOMINAL_DEPTH = 0.0015   # m past light contact (matches recording_script 'medium')
    NOMINAL_SPEED = 0.10     # m/s

    def __init__(self, n_ticks: int = 30):
        self.n_ticks = n_ticks
        self._t = 0

    def reset(self):
        self._t = 0

    def is_complete(self) -> bool:
        return self._t >= self.n_ticks

    def get_baseline_action(self) -> dict:
        """Nominal action the policy adds a residual to."""
        bow_pos = self._t / max(self.n_ticks - 1, 1)
        return {"depth": self.NOMINAL_DEPTH,
                "speed": self.NOMINAL_SPEED,
                "bow_position": float(bow_pos)}

    def execute_timestep(self, action: dict):
        """No robot — just advance the tick counter."""
        self._t += 1

    def get_full_state(self) -> dict:
        """Synthetic mechanical state (stand-in for RTDE reads). Provides the v1 obs keys."""
        return {"depth": self.NOMINAL_DEPTH,
                "contact_force_magnitude": 0.0,
                "bow_speed": self.NOMINAL_SPEED,
                "bow_position": self._t / max(self.n_ticks - 1, 1),
                "bow_direction": 1.0,
                "dev_depth": 0.0,
                "bow_accel": 0.0,
                "j4": 0.0, "j5": 0.0, "j6": 0.0}

    def close(self):
        pass
