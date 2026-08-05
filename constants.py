"""

Constants to use across the project. (Placeholders for now)
"""

import numpy as np

# ── Observation / Action dimensions ───────────────────────────────

OBS_DIM    = 20   # see get_observation() in env.py
ACTION_DIM = 3    # [dx, dy, dz] — TCP position corrections only (V1)
                  # expand to 5 [dx, dy, dz, d_force, d_speed] when
                  # force-controlled baseline is validated 

# ── Residual action bounds ─────────────────────────────────────────
# How much the policy can deviate from the baseline.
# Conservative for V1 — expand after validating stability.

MAX_DELTA_XY  = 0.003   # m  (±3mm lateral correction)
MAX_DELTA_Z   = 0.002   # m  (±2mm into/away from string)
MAX_DELTA_F   = 0.30    # N  (reserved for V2, not used in V1)
MAX_DELTA_V   = 0.05    # m/s (reserved for V2, not used in V1)

# ── Force limits (hardware safety) ────────────────────────────────
FORCE_MIN = 0.3    # N — below this: no string contact
FORCE_MAX = 8.0    # N — above this: raucous / hardware risk

# ── Baseline force per string ──────────────────────────────────────
# Placeholder values — collaborator fills in calibrated values
# from real robot force measurements.
BASELINE_FORCE = {
    'A': 2.8,
    'D': 3.1,
    'G': 3.5,
    'C': 4.0,
}

# ── Bow poses per string ──
BOW_POSES = {
    'A': {
        'frog': np.array([.336637615375, .773335607743, .103937252349,
                          -1.369835518384, -2.336199267621,  1.326965437172]),
        'tip':  np.array([.525205911288, .350983193771, .214779688012,
                          -1.369835518377, -2.336199267606,  1.326965437192]),
    },
    'D': {
        'frog': np.array([.3028, .7498, .1173, -1.6641, -2.0843, 1.0380]),
        'tip':  np.array([.3404, .2802, .1763, -1.6146, -2.0448, 1.0423]),
    },
    'G': {
        'frog': np.array([.2812, .6817, .1047, -1.8122, -1.9402, 0.4937]),
        'tip':  np.array([.1620, .2013, .0594, -1.9298, -1.9313, 0.5551]),
    },
    'C': {
        'frog': np.array([.2567, .6101, .0626, -1.7432, -1.5245, 0.1638]),
        'tip':  np.array([.0798, .2852, -.0867, -1.8196, -1.6583, 0.1809]),
    },
}

# ── String ordering (for crossing distance calculation) ────────────
STRING_ORDER = ['C', 'G', 'D', 'A']   # low to high

# ── RL timestep ────────────────────────────────────────────────────
TIMESTEP_SEC  = 0.05    # 50ms per RL step
STEPS_PER_STROKE = 150  # max timesteps per bow stroke

# ── Dynamic markings → normalized value ───────────────────────────
DYNAMIC_MAP = {
    'pp':  0.20,
    'mp':  0.40,
    'mf':  0.60,
    'f':   0.80,
    'ff':  1.00,
}

# ── Articulation → normalized value ───────────────────────────────
ARTICULATION_MAP = {
    'legato':   0.0,
    'detache':  0.5,
    'martele':  1.0,
}

# ── TCP offset (from a_full_bow.script — must match real robot) ────
TCP_OFFSET  = [0.028210348281514253, -0.09610723587300697,
               -0.09969041498611403, 0.0, 0.0, 0.0]
PAYLOAD_KG  = 0.260000
PAYLOAD_COG = [0.050000, -0.008000, 0.024000]

# ── Reward weights ─────────────────────────────────────────────────
# Must sum to 1.0 (human penalty is additive, not weighted)
REWARD_WEIGHTS = {
    'quality':   0.45,
    'trend':     0.10,
    'dynamic':   0.20,
    'bow':       0.15,
    'timing':    0.05,
    'smooth':    0.05,
}