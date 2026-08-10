"""
rl/loudness.py

Closed-loop dynamics: measured loudness instead of assumed loudness.

The planner maps a written dynamic to a bow speed with volume_to_speed(), a
linear rule that squeezes p..f into the middle of an absolute ppp..fff axis
and then never checks what actually came out of the instrument. Measured on
t1 that is 3.5 dB of contrast -- audibly no dynamics at all.

This module replaces the assumption with a fit to the 500 recordings in
dataset_a_final (see SoundClassifier/checkpoints/loudness_model.json):

    dBFS = a * 20log10(bow_speed) + b * depth_mm + d
    standard config, n=100, R^2 = 0.892, residual sd 1.14 dB

a ~ 1.13 confirms the calibration docstring's "amplitude proportional to bow
speed"; b ~ 0.42 dB/mm makes depth worth about 1 dB across the whole safety
envelope, i.e. a timbral trim rather than a dynamic lever.

Inverting it turns a TARGET LOUDNESS into the speed that actually produces
it, which spreads p..f over the full ~10 dB the instrument can reach instead
of 3.5 dB.

Dynamics are then judged in four discrete ZONES (p, mp, mf, f), each an equal
2.5 dB slice of that reachable span. A stroke inside its zone is fully
correct; outside, reward falls off with the dB distance to the nearest edge.
Zones rather than a point target because the residual sd is 1.14 dB -- asking
a stroke to hit an exact dBFS would be grading noise.

CAVEATS, all real:
  - bow_position is NOT in the model. Every dataset recording is a full-bow
    stroke, so its midpoint is 0.535 +/- 0.0002; there is no variation to fit.
    The RL plays partial strokes anywhere in u = 0.2..0.8, where leverage
    genuinely differs, and that error is NOT in the 1.14 dB residual.
  - The dataset is sustained full-bow strokes; RL strokes run 7..233 mm.
  - dBFS is absolute and therefore tied to THIS microphone and gain setting.
    Change the interface, the gain, or the mic position and the zones must be
    re-fitted or they will be silently wrong.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "SoundClassifier" / "checkpoints" / "loudness_model.json"

# Written dynamic -> one of the four zones. MuseScore's scale has eight
# levels; the instrument can only resolve about four, so the extremes fold in.
DYNAMIC_TO_ZONE = {
    "ppp": "p", "pp": "p", "p": "p",
    "mp": "mp",
    "mf": "mf",
    "f": "f", "ff": "f", "fff": "f",
}
ZONE_ORDER = ["p", "mp", "mf", "f"]


class LoudnessModel:
    """Fitted dBFS(speed, depth) with its inverse and the dynamic zones."""

    def __init__(self, path: Path | str | None = None):
        path = Path(path or MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"No loudness model at {path}. Fit one from dataset_a_final "
                f"before using closed-loop dynamics.")
        data = json.loads(path.read_text())
        c = data["coef"]
        self.a, self.b, self.d = float(c["a"]), float(c["b"]), float(c["d"])
        self.zones = data["zones"]
        self.residual_sd_db = float(data["fit"]["residual_sd_db"])
        self.speed_min, self.speed_max = data["speed_range"]
        self.path = path

    # ── forward / inverse ────────────────────────────────────────

    def predict_dbfs(self, speed: float, depth_m: float = 0.0005) -> float:
        speed = max(float(speed), 1e-4)
        return self.a * 20.0 * math.log10(speed) + self.b * (depth_m * 1000.0) + self.d

    def invert_speed(self, target_dbfs: float, depth_m: float = 0.0005) -> float:
        """Bow speed that produces `target_dbfs` at this depth, clamped to the
        calibrated range so the answer is always interpolation."""
        exponent = (target_dbfs - self.b * (depth_m * 1000.0) - self.d) / (20.0 * self.a)
        speed = 10.0 ** exponent
        return float(min(max(speed, self.speed_min), self.speed_max))

    # ── zones ────────────────────────────────────────────────────

    @staticmethod
    def zone_for_dynamic(dynamic: str) -> str:
        return DYNAMIC_TO_ZONE.get(dynamic, "mf")

    def zone_bounds(self, zone: str) -> tuple[float, float]:
        z = self.zones[zone]
        return float(z["db_lo"]), float(z["db_hi"])

    def zone_centre(self, zone: str) -> float:
        return float(self.zones[zone]["db_centre"])

    def target_speed_for_dynamic(self, dynamic: str,
                                 depth_m: float = 0.0005) -> float:
        """Feed-forward: the speed that should land this dynamic in its zone."""
        return self.invert_speed(self.zone_centre(self.zone_for_dynamic(dynamic)),
                                 depth_m)

    def zone_reward(self, measured_dbfs: float, zone: str,
                    falloff_db: float = 3.0) -> float:
        """
        1.0 anywhere inside the zone, falling linearly to 0 once the stroke is
        `falloff_db` outside the nearest edge.

        Flat inside rather than peaked at the centre because the model's own
        residual is 1.14 dB: rewarding sub-decibel precision would be
        rewarding measurement noise, and would also punish the tone-driven
        micro-adjustments the policy is supposed to be free to make.
        """
        lo, hi = self.zone_bounds(zone)
        if lo <= measured_dbfs <= hi:
            return 1.0
        distance = (lo - measured_dbfs) if measured_dbfs < lo else (measured_dbfs - hi)
        return float(max(0.0, 1.0 - distance / falloff_db))


_MODEL: LoudnessModel | None = None


def get_model() -> LoudnessModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = LoudnessModel()
    return _MODEL


def measure_dbfs(audio) -> float | None:
    """
    RMS level of a stroke, in dBFS.

    MUST be given the raw capture. The classifier window is peak-normalised
    (that is what the model was trained on), and normalising destroys exactly
    the absolute level this is trying to measure.
    """
    import numpy as np
    if audio is None or len(audio) == 0:
        return None
    a = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(a * a)))
    return 20.0 * math.log10(max(rms, 1e-9))
