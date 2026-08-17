"""
rl/driver_piece.py

Full driver: closed-loop LOUDNESS + DEPTH regulation for PieceResidualEnv.

Adds five observation dims carrying the previous stroke's measured state, and
a two-channel training curriculum that randomizes, per episode, (a) a virtual
gain offset on the loudness reading and (b) a hidden depth offset on the
executed strokes — so the policy can only keep notes in their loudness zones
AND near the tone sweet spot by reading its sensors and countering with its
two residuals:

    obs[18]  prev-stroke signed zone error, dB / 3.0      (loudness channel)
    obs[19]  EMA of the signed zone error, dB / 3.0
    obs[20]  prev-stroke measured dBFS, zone-centred, / 10.0
    obs[21]  prev-stroke torque magnitude mean / 0.5      (depth channel)
    obs[22]  prev-stroke torque magnitude max  / 1.0

Injection points (both at the executor boundary, invisible to the planner and
to the policy except through consequences):
  - gain offset: added to StrokeResult.measured_dbfs after execution;
  - depth offset: added to the commanded stroke/segment depths BEFORE
    execution, then re-clamped to the stock safety envelope
    [-MAX_OUTWARD_DEPTH, +MAX_INWARD_DEPTH] — the robot never reaches
    anywhere the stock system could not already command.

Curriculum bounds: gain ±1.5 dB (speed residual authority +1.79/−2.19 dB);
depth ±0.5 mm (residual authority ±0.5 mm; observed natural contact drift
~1.3 mm/session motivates the channel).

Blind flags train capacity-matched ablation arms: --blind-dyn-obs zeroes the
three loudness dims, --blind-mech-obs the two torque dims (network shape and
curriculum identical; only information differs).

Obs 18→23 requires a fresh SAC model; no old checkpoint or replay buffer is
ever reusable. 2026-08-13: the upstream action space became a 6-dim
per-segment envelope (piece_env.ACTION_DIM) — the driver is action-space
agnostic (it only adds obs dims and executor-boundary offsets), but any
checkpoint trained before the change is dead on the new env for that reason
too.

Zixian Liu, 2026-08-10.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
from gymnasium import spaces

from rl.piece_env import PieceResidualEnv, OBS_DIM, DEPTH_LO, DEPTH_HI

LOUD_DIMS = 3
MECH_DIMS = 2
DRIVER_EXTRA_DIMS = LOUD_DIMS + MECH_DIMS
ERR_NORM_DB = 3.0
DBFS_NORM_DB = 10.0
TORQUE_MEAN_NORM = 0.5
TORQUE_MAX_NORM = 1.0
DEFAULT_GAIN_JITTER_DB = 1.5
DEFAULT_DEPTH_JITTER_MM = 0.5
ERR_EMA_ALPHA = 0.7


def _seg_with_depth(seg, depth):
    try:
        return dataclasses.replace(seg, depth=depth)
    except TypeError:
        return seg._replace(depth=depth)


class CurriculumExecutor:
    """Injects the two per-episode offsets at the executor boundary.

    depth: commanded stroke/segment depths get offset_m added and are
    re-clamped to the stock envelope before the inner executor sees them.
    gain: measured_dbfs gets offset_db added after execution.
    """

    def __init__(self, inner, calib_db: float = 0.0):
        self.inner = inner
        self.offset_db = 0.0
        self.depth_offset_m = 0.0
        # one-time measurement-chain calibration (measured minus model frame),
        # SUBTRACTED from every reading before the curriculum offset is added —
        # like taring a scale; estimated from long-stroke model residuals.
        self.calib_db = float(calib_db)

    def _shift(self, stroke):
        if not self.depth_offset_m:
            return stroke
        d = float(np.clip(stroke.depth + self.depth_offset_m,
                          DEPTH_LO, DEPTH_HI))
        s = copy.copy(stroke)
        segs = getattr(stroke, "segments", None)
        if segs:
            s = dataclasses.replace(s, depth=d, segments=[
                _seg_with_depth(seg, float(np.clip(
                    seg.depth + self.depth_offset_m, DEPTH_LO, DEPTH_HI)))
                for seg in segs])
        else:
            s = dataclasses.replace(s, depth=d)
        return s

    def begin_episode(self, first):
        return self.inner.begin_episode(self._shift(first))

    def execute(self, stroke):
        result = self.inner.execute(self._shift(stroke))
        if result.measured_dbfs is not None and (self.offset_db or self.calib_db):
            result.measured_dbfs = (float(result.measured_dbfs)
                                    - self.calib_db + self.offset_db)
        return result

    def end_episode(self):
        return self.inner.end_episode()

    def close(self):
        return self.inner.close()


class DriverPieceEnv(PieceResidualEnv):
    """PieceResidualEnv + two-channel sensing + two-channel curriculum."""

    def __init__(self, *args, gain_jitter_db: float = 0.0,
                 gain_fixed_db: float | None = None,
                 depth_jitter_mm: float = 0.0,
                 depth_fixed_mm: float | None = None,
                 blind_dyn_obs: bool = False,
                 blind_mech_obs: bool = False,
                 dbfs_calib_db: float = 0.0,
                 mock_seed: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.loudness is None:
            raise RuntimeError(
                "DriverPieceEnv needs the closed-loop loudness model: the "
                "loudness obs dims do not exist without it.")
        self.dbfs_calib_db = float(dbfs_calib_db)
        # Since 2026-08-12 loudness_model.json can carry its own gain offset,
        # applied inside zone_reward(). Exactly ONE correction may be active:
        # both at once would subtract the miscalibration twice and judge every
        # stroke several dB too quiet.
        self.model_gain_offset_db = float(
            getattr(self.loudness, "gain_offset_db", 0.0))
        if self.model_gain_offset_db and self.dbfs_calib_db:
            raise RuntimeError(
                f"double gain correction: loudness_model.json has "
                f"gain_offset_db={self.model_gain_offset_db:+.2f} dB AND "
                f"--dbfs-calib-db={self.dbfs_calib_db:+.2f} dB was passed. "
                "Keep exactly one: zero the json field or drop the flag.")
        self.gain_jitter_db = float(gain_jitter_db)
        self.gain_fixed_db = gain_fixed_db
        self.depth_jitter_mm = float(depth_jitter_mm)
        self.depth_fixed_mm = depth_fixed_mm
        self.blind_dyn_obs = bool(blind_dyn_obs)
        self.blind_mech_obs = bool(blind_mech_obs)

        self.executor = CurriculumExecutor(self.executor,
                                           calib_db=self.dbfs_calib_db)
        if mock_seed is not None and hasattr(self.executor.inner, "rng"):
            self.executor.inner.rng = np.random.default_rng(mock_seed)

        lo, _ = self.loudness.zone_bounds("p")
        _, hi = self.loudness.zone_bounds("f")
        self._zones_mid = (lo + hi) / 2.0

        self._last_err_db = 0.0
        self._err_ema_db = 0.0
        self._last_dbfs = None
        self._last_torque_mean = 0.0
        self._last_torque_max = 0.0

        self.observation_space = spaces.Box(
            -np.inf, np.inf, (OBS_DIM + DRIVER_EXTRA_DIMS,), dtype=np.float32)

    # ── episode lifecycle ─────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        self._last_err_db = 0.0
        self._err_ema_db = 0.0
        self._last_dbfs = None
        self._last_torque_mean = 0.0
        self._last_torque_max = 0.0
        obs, info = super().reset(seed=seed, options=options)

        # Known quirk (reviewed 2026-08-13, deliberately kept): super().reset
        # calls executor.begin_episode BEFORE the new offsets are drawn below,
        # so on hardware the initial set-down uses the PREVIOUS episode's
        # depth offset (<=0.5 mm; the first stroke corrects it). Sampling
        # earlier would need np_random before super() seeds it — not worth
        # breaking seed determinism over a one-stroke set-down transient.
        if self.gain_fixed_db is not None:
            self.executor.offset_db = float(self.gain_fixed_db)
        elif self.gain_jitter_db > 0.0:
            self.executor.offset_db = float(self.np_random.uniform(
                -self.gain_jitter_db, self.gain_jitter_db))
        else:
            self.executor.offset_db = 0.0

        if self.depth_fixed_mm is not None:
            self.executor.depth_offset_m = float(self.depth_fixed_mm) / 1000.0
        elif self.depth_jitter_mm > 0.0:
            self.executor.depth_offset_m = float(self.np_random.uniform(
                -self.depth_jitter_mm, self.depth_jitter_mm)) / 1000.0
        else:
            self.executor.depth_offset_m = 0.0
        return obs, info

    # ── reward hook doubles as the sensor-storage hook ────────────

    def _reward(self, plan_stroke, exec_stroke, result, shortfall, action):
        total, components = super()._reward(plan_stroke, exec_stroke, result,
                                            shortfall, action)
        components["gain_offset_db"] = self.executor.offset_db
        components["depth_offset_mm"] = self.executor.depth_offset_m * 1000.0
        if components.get("zone") is not None:
            # The sensor dims must live in the MODEL frame — the same frame
            # zone_reward grades in. Stock err_db is logged in the raw mic
            # frame (r_dynamic corrects internally, err_db does not), so
            # subtract the model's gain offset here. With the v0.1.0-style
            # executor-boundary calibration (--dbfs-calib-db) the offset is 0
            # and this is a no-op.
            err_model = (float(components["err_db"])
                         - self.model_gain_offset_db)
            components["err_db_model"] = err_model
            self._last_err_db = err_model
            self._err_ema_db = (ERR_EMA_ALPHA * self._err_ema_db
                                + (1 - ERR_EMA_ALPHA) * self._last_err_db)
            self._last_dbfs = (
                float(result.measured_dbfs) - self.model_gain_offset_db
                if result.measured_dbfs is not None else None)
        phys = np.asarray(result.physical, dtype=float)
        if phys.shape[0] >= 6 and np.isfinite(phys[4]) and np.isfinite(phys[5]):
            self._last_torque_mean = float(phys[4])
            self._last_torque_max = float(phys[5])
        return total, components

    # ── observation ───────────────────────────────────────────────

    def _obs(self):
        base = super()._obs()
        if self.blind_dyn_obs:
            loud = np.zeros(LOUD_DIMS, dtype=np.float32)
        else:
            dbfs_c = (0.0 if self._last_dbfs is None
                      else (self._last_dbfs - self._zones_mid) / DBFS_NORM_DB)
            loud = np.array([
                self._last_err_db / ERR_NORM_DB,
                self._err_ema_db / ERR_NORM_DB,
                dbfs_c,
            ], dtype=np.float32)
        if self.blind_mech_obs:
            mech = np.zeros(MECH_DIMS, dtype=np.float32)
        else:
            mech = np.array([
                self._last_torque_mean / TORQUE_MEAN_NORM,
                self._last_torque_max / TORQUE_MAX_NORM,
            ], dtype=np.float32)
        return np.concatenate([base, loud, mech])


def make_factory(gain_jitter_db=0.0, gain_fixed_db=None,
                 depth_jitter_mm=0.0, depth_fixed_mm=None,
                 blind_dyn_obs=False, blind_mech_obs=False,
                 dbfs_calib_db=0.0, mock_seed=None):
    """Drop-in replacement for the PieceResidualEnv constructor."""
    def factory(*args, **kwargs):
        return DriverPieceEnv(*args, gain_jitter_db=gain_jitter_db,
                              gain_fixed_db=gain_fixed_db,
                              depth_jitter_mm=depth_jitter_mm,
                              depth_fixed_mm=depth_fixed_mm,
                              blind_dyn_obs=blind_dyn_obs,
                              blind_mech_obs=blind_mech_obs,
                              dbfs_calib_db=dbfs_calib_db,
                              mock_seed=mock_seed, **kwargs)
    return factory
