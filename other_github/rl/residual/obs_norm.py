"""
obs_norm.py — observation normalization for the residual-RL policy.

The 10 v1 obs features have wildly different magnitudes (depth ~1e-3 m, force ~3 N,
joints ~3 rad). Feeding raw values to the net lets large-magnitude features dominate
and hurts learning, so we divide each by a rough characteristic scale -> each ~O(1).

These scales are FIRST-GUESS magnitudes; refine from real robot data once we have it.
Order matches residual_runner.OBS_FEATURES:
  depth, contact_force_magnitude, bow_speed, bow_position,
  bow_direction, dev_depth, bow_accel, j4, j5, j6
"""

import numpy as np

OBS_SCALES = np.array(
    [0.003,   # depth (m), max press ~3 mm
     5.0,     # contact_force_magnitude (N)
     0.15,    # bow_speed (m/s)
     1.0,     # bow_position (already 0..1)
     1.0,     # bow_direction (+-1)
     0.001,   # dev_depth (m)
     1.0,     # bow_accel (m/s^2, rough)
     3.14, 3.14, 3.14],   # wrist joints j4/j5/j6 (rad)
    dtype=np.float32,
)


def normalize(obs: np.ndarray) -> np.ndarray:
    return (np.asarray(obs, dtype=np.float32) / OBS_SCALES).astype(np.float32)
