"""
hardware_baseline.py — REAL-ROBOT adapter for residual depth-RL. v2 (Batch-1
redesign, 2026-06-10; v1 had the audit's R1/R2 critical depth-reference bugs).

Same interface as ZeroBaseline: reset / is_complete / get_baseline_action /
execute_timestep(action) / get_full_state / close.

DESIGN (one taught string line + feed-forward depth):
  The FROG/TIP waypoints are TAUGHT AT light contact (teach_tip_on_axis, bow on a
  string tape mark), so the recording line IS the contact line — no per-frac
  offset to measure. Every tick the commanded pose is CONSTRUCTED as
      bow_pose_full(frac) + depth_cmd * e_press,
      depth_cmd = clip(nominal_depth + clip(residual, ±DEPTH_RESIDUAL_CLAMP), 0, MAX_PRESS_DEPTH)
  (constants below — currently ±5mm on nominal 2mm, ceiling 8mm; the paper's tighter
  bounded-residual claim requires re-tightening once nominal sits in the good band)
  so no feedback term can ever lunge (v1's R2). Measured depth is reported in
  the observation ONLY — if feedback is useful, the POLICY learns it.
  frac comes from an OWN counter advanced by bow_speed*TICK/SPAN_M per tick —
  collect_tone's exact convention (train==deploy on the speed axis, 2026-06-30);
  the controller note supplies direction/bookkeeping only.

What this kills from the audit: R1 (scalar z reference vs 110.8mm tilted
string), R2 (unclamped correction), K8 (pure-Z press vs the validated normal),
K3 (force=3.0 -> baseline's hidden z-offset = 0), R10 (geometry from the
validated recording waypoints), R11/K11 (clamp_reach on every command), K4+lit
(abort at 5N AND retract before raising), R17 (reset = hover + force-guarded
descent, never a blind moveL onto a stale on-string waypoint), R3 (safety-mode
checked every tick and at reset).

CALIBRATION (2026-06-21): waypoints RE-TAUGHT at light contact after the cello
move + bow swap, so contact = bow_pose_full directly. Anchors RETIRED — they were
a patch for old waypoints that sat 2-10mm off true contact; points taught on the
string need no offset. After ANY re-teach, run verify_contact_line.py to confirm
the straight line stays on the string and the press normal builds force.

RUNS ONLY in an env with ur_rtde (+ sounddevice via recording_grid import).
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, os.path.join(_REPO, "Baseline-Runners")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import recording_grid as R                                       # noqa: E402
from baseline_controller import TargetStringBaselineController, velocity_to_speed   # noqa: E402

# ---- waypoints taught AT light contact (2026-06-21) -> the recording line IS the
#      contact line; no per-frac offset (anchors retired) ----
NOMINAL_DEPTH = 0.002       # m past the CAPTURED line. Zixian 2026-06-27: set in the TOO-LIGHT region
                            # (reward ~0.49) so the RL has clear room to climb to the good band d~4-5.
DEPTH_RESIDUAL_CLAMP = 0.005    # m; keep == residual_runner.DEPTH_RESIDUAL_CLAMP. ±5mm (swept-safe; gives
                                # the RL near-full depth control d∈[0,7] from nominal 2). TIGHTEN to
                                # ±0.5-1mm for the paper's bounded-residual claim once baseline sits in-band.
MAX_PRESS_DEPTH = 0.008     # m, depth ceiling (nominal 2 + clamp 5 = 7 <= 8; d=7 ~4N < 5N abort)
DEPTH_MIN = -0.006          # m, LIFT floor (2026-07-10): negative depth = bow lifts ABOVE the taught
                            # line. Lifting only REDUCES force (safe direction). Lets us re-zero to a
                            # contact line that has DRIFTED FIRMER (e.g. today: 0mm command landed 3.3N)
                            # by using a NEGATIVE nominal, without re-teaching FROG/TIP.
FORCE_ABORT_N = 5.5         # N — aligned to collect_tone's 5.5 (Zixian 2026-07-05); still under the
                            # bow's 3-6N near-tip structural margin (Wofford 2018). Do NOT raise past 6.
DEPTH_RATE_LIMIT = 0.001    # m, max |change of depth_cmd| per 50ms tick (R2 backstop)

# ---- attack primitive v1' (2026-08-02, lit check ~/Desktop/Research/lit_check_20260802.md §三:
#      human on-string attacks ramp VELOCITY, not force — "during attacks, acceleration is the
#      operative word" (Guettler); a long depth ramp dwells below the measured minimum-bow-force
#      floor and CAUSES multiple-slip surface sound). Staircase speed ramp over the first 3 ticks
#      (150ms; effective accel ≈ v/0.15s = 0.6-1.4 m/s² for our speeds, inside the measured cello
#      Guettler wedge 0.15-3.15 m/s², Lampis 2024) + a light-side depth bias for 2 ticks (~100ms;
#      tolerance asymmetry: light-side failures allowed 90ms vs heavy-side 50ms).
#      OFF by default; enable via env ATTACK_RAMP=1, trainer --attack-ramp, or hb.ATTACK_RAMP=True. ----
ATTACK_RAMP = os.environ.get("ATTACK_RAMP", "0").strip().lower() in ("1", "true", "yes")
ATTACK_SPEED_SCALE = (1.0 / 3.0, 2.0 / 3.0)   # ticks 0-1 (tick 2 onward = full commanded speed)
ATTACK_DEPTH_SCALE = (0.90, 0.95)             # ticks 0-1 (tick 2 onward = full commanded depth)

# ---- attack primitive v2 "LAND AT TARGET" (2026-08-03). Blind test same day: press-in strokes
#      (land at nominal 0, press the residual in DURING the stroke) scored 14/16 "明显刮" by ear
#      at 4mm target, while collect_tone-style landed strokes at the SAME 4mm/0.11 scored 3/3
#      clean; ramp v1' did NOT help (machine + ear both flat) because the scratch lives in the
#      below-minimum-force press-in transit, not in the speed buildup. Literature agrees
#      (lit_check_20260802 §三: on-string attacks start with force at working value —
#      Guettler/Woodhouse; Schoonderwaldt measured attack force over the first 10ms).
#      When ON, the caller passes the stroke's TARGET depth to reset(depth_override_m=...) and
#      the guarded descent lands there — the stroke then starts with force already established.
#      OFF by default; enable via env LAND_AT_TARGET=1/true/yes or trainer --land-at-target. ----
LAND_AT_TARGET = os.environ.get("LAND_AT_TARGET", "0").strip().lower() in ("1", "true", "yes")
LAND_MAX_M = 0.0045    # landing-depth ceiling (review H1): ~4.0-5.0N at measured stiffness,
                       # keeps every guarded-descent step >=0.5N under FORCE_ABORT_N


def landing_band(target_depth_m):
    """Landed-force sanity band for a guarded descent to target_depth_m. The historic flat
    0.05-2.5N band was calibrated for nominals <=2mm; deeper deliberate landings scale with
    depth (measured 2026-08-03: 4mm landings 2.81/2.96/3.56N on the lighter line; contact
    stiffness ~0.9-1.1 N/mm). Upper = min(5.0, max(2.5, 1.2 * depth_mm)) keeps <=2.08mm
    behavior identical and stays under FORCE_ABORT_N=5.5."""
    upper = min(5.0, max(2.5, 1.2 * float(target_depth_m) * 1000.0))
    return 0.05, upper

# ---- stroke timing = collect_tone's convention EXACTLY (train==deploy on the speed axis).
#      The bow frac is OUR OWN counter advanced by speed*TICK/SPAN per tick; the controller
#      note supplies bookkeeping only. speed_residual=0 at NOMINAL_SPEED reproduces the
#      depth-only runs byte-for-byte (2026-06-30, speed-plan Stage 1). ----
TICK = 0.05                                   # = TIMESTEP_SEC = collect_tone.TICK
STROKE_END_FRAC = 0.80                        # stroke = start_frac(0.20) -> here (collect_tone FRAC_END)
SPAN_M = float(np.linalg.norm(R.TIP_TARGET_STR[:3] - R.FROG_TARGET_STR[:3]))   # frog->tip length
NOMINAL_SPEED = float(velocity_to_speed(80))  # 0.1093 m/s = the note vel=80 the depth runs used


def commanded_depth_mm(action0, nominal=NOMINAL_DEPTH):
    """STEADY-STATE commanded depth (mm) for a raw policy action[0]: [-1,1] clip -> residual clamp ->
    nominal -> [0, MAX_PRESS] ceiling. Single source of truth for logging/figures (no 2+a*5).
    Pass the RUN's actual nominal (hw.nominal_depth) — a --nominal-mm override else mislogs by the
    difference. NOTE: excludes the <=5-tick DEPTH_RATE_LIMIT ramp at stroke start."""
    r = float(np.clip(action0, -1.0, 1.0)) * DEPTH_RESIDUAL_CLAMP
    return float(np.clip(float(nominal) + r, DEPTH_MIN, MAX_PRESS_DEPTH)) * 1000.0
ACCEL_LIMIT = 5.0           # m/s^2, clamp differentiated bow accel in obs
HOVER_M = 0.012             # reset hover height above the contact line
CLEAR_M = 0.10              # tare clearance (12mm hover was graze-prone for taring)
TARE_MAX_N = 1.0            # free-air |F| after zeroFtSensor must be below this

E_PRESS = -R.RETRACT_DIR_BASE                          # unit vector INTO the string
_BOW_A = R.TIP_TARGET_STR[:3] - R.FROG_TARGET_STR[:3]  # validated recording line

# fail-loudly sanity: the taught line IS the contact line, so the only invariants are
# (a) the waypoints haven't drifted from what verify_contact_line.py validated, and
# (b) FROG/TIP share one orientation (bow_pose_full lerps the rotvec).
assert np.allclose([R.FROG_TARGET_STR[0], R.FROG_TARGET_STR[2],
                    R.TIP_TARGET_STR[0], R.TIP_TARGET_STR[2]],
                   [0.346316666, 0.100027869, 0.537882319, 0.243073479],
                   atol=1e-9), \
    "recording waypoints CHANGED since verify_contact_line — re-run it before RL"
assert np.allclose(R.FROG_TARGET_STR[3:], R.TIP_TARGET_STR[3:], atol=1e-6), \
    "FROG/TIP orientations differ — bow_pose_full's rotvec lerp is no longer valid"


# E_PRESS is NOT perpendicular to the bow axis (66 deg apart), so a naive
# projection lets press displacement leak into the frac estimate (~65% of the
# press read as frac shift — caught by the offline test). Decompose OBLIQUELY:
#   p - FROG = f * _BOW_A + d * E_PRESS   (f = bow fraction, d = press-axis dist)
_M = np.array([[float(np.dot(_BOW_A, _BOW_A)), float(np.dot(_BOW_A, E_PRESS))],
               [float(np.dot(_BOW_A, E_PRESS)), 1.0]])
_Minv = np.linalg.inv(_M)


def line_coords(p3):
    """(frac, press_dist) of a base-frame point in the oblique line/press basis."""
    rel = np.asarray(p3, dtype=float) - R.FROG_TARGET_STR[:3]
    f, d = _Minv @ np.array([float(np.dot(rel, _BOW_A)), float(np.dot(rel, E_PRESS))])
    return float(np.clip(f, 0.0, 1.0)), float(d)


def frac_of(p3):
    """Bow fraction (0=frog, 1=tip), insensitive to press-axis displacement."""
    return line_coords(p3)[0]


def contact_pose(frac):
    """Full 6D pose of light contact at a bow fraction. The waypoints were taught
    AT light contact, so this is just bow_pose_full — no per-frac offset."""
    return R.bow_pose_full(float(frac)).copy()


# usable bow region = where the bow-tip reach stays under MAX_SAFE_REACH. The frog end exceeds
# it (~0.84m); only frac >~0.06 is safe. Computed once at import (don't re-scan every tick).
_SAFE_REACH = R.MAX_SAFE_REACH - 0.01
_grid = np.linspace(0.0, 1.0, 201)
_safe_fr = _grid[np.array([float(np.linalg.norm(R.bow_pose_full(f)[:3])) for f in _grid]) < _SAFE_REACH]
_COVERAGE = (round(float(_safe_fr.min()) + 0.02, 3), float(_safe_fr.max())) if len(_safe_fr) else (0.30, 1.0)


def coverage():
    return _COVERAGE


def safety_ok(ctrl):
    """Normal safety mode + Running (lesson 2026-06-09: in protective stop every
    command fails silently and the receive buffer freezes)."""
    try:
        if ctrl.rtde_r.isProtectiveStopped():
            return False
        if int(ctrl.rtde_r.getSafetyMode()) != 1:
            return False
        if int(ctrl.rtde_r.getRobotMode()) != 7:
            return False
    except Exception:
        return False   # FAIL CLOSED: a dead/frozen receive interface is exactly
                       # the protective-stop signature (2026-06-09) — never "safe"
    return True


class HardwareBaseline:
    def __init__(self, note_events, nominal_depth=NOMINAL_DEPTH, start_frac=0.30, bow_speed=None):
        lo, hi = coverage()
        if not (lo <= start_frac <= hi):
            raise ValueError(f"start_frac {start_frac} outside calibrated coverage [{lo:.3f},{hi:.3f}]")
        if not (DEPTH_MIN <= nominal_depth <= MAX_PRESS_DEPTH - DEPTH_RESIDUAL_CLAMP):
            raise ValueError(f"nominal_depth {nominal_depth} outside [{DEPTH_MIN}, "
                             f"{MAX_PRESS_DEPTH - DEPTH_RESIDUAL_CLAMP}] (negative = lift to re-zero a "
                             f"drifted-firmer line; keep residual headroom below the ceiling)")
        if len(note_events) > 1 or (note_events and note_events[0].get("bow_dir", "down") != "down"):
            raise ValueError("the frac-window stroke is a SINGLE DOWN-bow: the own frac counter has "
                             "no note boundaries / up-bow direction (yet). Pass exactly one down note.")
        self.ctrl = TargetStringBaselineController(note_events)
        self.nominal_depth = float(nominal_depth)
        self.start_frac = float(start_frac)
        self.bow_speed = float(bow_speed) if bow_speed is not None else NOMINAL_SPEED
        if not (0.02 <= self.bow_speed <= 0.30):
            raise ValueError(f"bow_speed {self.bow_speed} outside sane band [0.02, 0.30] m/s")
        self._frac = None                  # OWN frac counter (collect_tone convention); set by reset()
        self._stroke_tick = 0              # ticks since stroke start (attack-ramp v1' window)
        self._last_depth_cmd = None
        self._last_residual = 0.0
        self._clip_warned = False
        self._prev_speed, self._prev_t = 0.0, None

    def set_bow_speed(self, v):
        """Per-episode speed change for CONDITIONED training (2026-07-19): swap the stroke
        speed without rebuilding the RTDE connection. Call BEFORE reset() — mid-stroke the
        frac counter is advancing at the old rate."""
        v = float(v)
        if not (0.02 <= v <= 0.30):
            raise ValueError(f"bow_speed {v} outside sane band [0.02, 0.30] m/s")
        self.bow_speed = v

    # ------------------------------------------------------------------ guards
    def _require_safe(self, what):
        if not safety_ok(self.ctrl):
            raise RuntimeError(f"ROBOT NOT NORMAL/RUNNING ({what}) — recover on the pendant, rerun.")

    def _abort_retract(self, why):
        """Get OFF the string before raising (K4): proven from on-string poses."""
        try:
            self.ctrl.rtde_c.servoStop()
            R.retract(self.ctrl)
        except Exception:
            pass
        raise RuntimeError(why)

    def _resolve_descent_target(self, depth_override_m):
        """Descent target for reset(): nominal, or (LAND_AT_TARGET v2) the stroke's own
        target depth. Override requires the flag ON — a caller passing overrides with the
        flag OFF is a wiring bug, fail loudly before any motion."""
        if depth_override_m is None:
            return self.nominal_depth
        if not LAND_AT_TARGET:
            raise ValueError("reset(depth_override_m=...) passed but LAND_AT_TARGET is OFF "
                             "— caller/flag wiring bug")
        # LAND_MAX: landings are force-checked only every step — at the measured stiffness
        # (0.9-1.1 N/mm) targets >4.5mm land at 4.0-5.0N, and >=6mm would GUARANTEE a 5.5N
        # descent abort (review H1). Deeper presses remain reachable IN-STROKE (clipped at
        # MAX_PRESS_DEPTH there), where the per-tick force abort + abort penalty handle them.
        return float(np.clip(float(depth_override_m), DEPTH_MIN, LAND_MAX_M))

    # ------------------------------------------------------------------- reset
    def reset(self, depth_override_m=None):
        """Guarded approach: (conditional travel) -> hover above the contact line
        at start_frac -> tare -> force-guarded descent to the DESCENT TARGET
        (nominal, or the stroke's target depth under LAND_AT_TARGET v2). NEVER
        calls ctrl.reset() (blind moveL onto a stale on-string waypoint, R17)."""
        target = self._resolve_descent_target(depth_override_m)
        try:
            self.ctrl.rtde_c.servoStop()   # kill the previous episode's servo thread:
        except Exception:                  # moveL while servoL streams = control-script
            pass                           # fault (BLOCKER; rtde_v3's 'servoStop() #New')
        self._require_safe("reset preflight")
        self._last_depth_cmd, self._last_residual = None, 0.0
        self._prev_speed, self._prev_t = 0.0, None
        # reset the note sequence WITHOUT the controller's unguarded motion;
        # seed _current_note so the first obs reports the correct bow_direction
        self.ctrl.current_note_idx = 0
        self.ctrl.current_step = 0
        self.ctrl._trajectory = []
        self.ctrl._current_note = (dict(self.ctrl.note_events[0])
                                   if getattr(self.ctrl, "note_events", None) else {})

        clear = R.clamp_reach(R.apply_force_offset(contact_pose(self.start_frac), -CLEAR_M))
        hover = R.clamp_reach(R.apply_force_offset(contact_pose(self.start_frac), -HOVER_M))
        tcp = np.array(self.ctrl.rtde_r.getActualTCPPose())
        if tcp[2] >= 0.30:
            # parked high: direct inward moveL to clearance (retract/reset_to from a
            # parked pose caused the 06-09 C306A3 fault)
            ok = self.ctrl.rtde_c.moveL(clear.tolist(), self.ctrl.SERVO_VEL, self.ctrl.SERVO_ACCEL)
            if ok is False:
                raise RuntimeError("moveL to clearance failed — check robot state.")
        else:
            R.reset_to(self.ctrl, clear)   # proven from on/near-string poses
        time.sleep(0.3)
        self._require_safe("after travel to clearance")

        # tare FAR from the string with the pendant-measured payload (12mm hover
        # taring risked grazing the string and zeroing-in real contact force)
        try:
            self.ctrl.rtde_c.setPayload(float(R.FT_PAYLOAD_KG), list(R.FT_PAYLOAD_COG))
            time.sleep(0.2)
        except Exception:
            pass
        ok = self.ctrl.rtde_c.zeroFtSensor()
        time.sleep(0.3)
        f0 = float(np.linalg.norm(self.ctrl.rtde_r.getActualTCPForce()[:3]))
        if ok is False or f0 > TARE_MAX_N:
            raise RuntimeError(f"TARE FAILED at clearance (ok={ok}, |F|={f0:.2f}N) — do not trust force.")

        # free-air descent clearance -> hover (colinear along the press axis)
        ok = self.ctrl.rtde_c.moveL(hover.tolist(), self.ctrl.SERVO_VEL, self.ctrl.SERVO_ACCEL)
        if ok is False or not safety_ok(self.ctrl):
            raise RuntimeError("descent to hover failed / robot left Normal state.")
        time.sleep(0.2)
        # FAST moveL through FREE AIR to SLOW_BAND above the line, then SLOW force-guarded only for
        # the last bit into contact (Zixian 2026-06-27: the full 0.5mm crawl wasted time in free air).
        SLOW_BAND = 0.003
        # start the crawl SLOW_BAND above the (possibly NEGATIVE) target, so a lifted re-zero
        # still gets a real force-guarded approach (not an empty descent range).
        descent_start = min(-SLOW_BAND, target - SLOW_BAND)
        near = R.clamp_reach(R.apply_force_offset(contact_pose(self.start_frac), descent_start))
        ok = self.ctrl.rtde_c.moveL(near.tolist(), 0.08, 0.4)
        if ok is False or not safety_ok(self.ctrl):
            raise RuntimeError("fast free-air approach failed / robot left Normal state.")
        if float(np.linalg.norm(self.ctrl.rtde_r.getActualTCPForce()[:3])) > FORCE_ABORT_N:
            self._abort_retract("FORCE ABORT after fast approach (contact higher than expected?)")
        # force-guarded descent: descent_start -> contact -> descent target. Dynamic stepping
        # (review M1+H1c): 0.5mm steps in free air / light contact, HALVED to 0.25mm once
        # |F|>3.0N (limits the pre-check force transient near the abort line), and the loop
        # ends EXACTLY at the target — np.arange's last-multiple-of-step undershoot left up to
        # 0.5mm to be pressed in by tick 0 at full bow speed (the press-in scratch, miniature).
        d = float(descent_start)
        while True:
            pose = R.clamp_reach(R.apply_force_offset(contact_pose(self.start_frac), float(d)))
            ok = self.ctrl.rtde_c.moveL(pose.tolist(), self.ctrl.SERVO_VEL, self.ctrl.SERVO_ACCEL)
            if ok is False or not safety_ok(self.ctrl):
                raise RuntimeError("descent moveL failed / robot left Normal state.")
            time.sleep(0.15)
            f = float(np.linalg.norm(self.ctrl.rtde_r.getActualTCPForce()[:3]))
            if f > FORCE_ABORT_N:
                self._abort_retract(f"FORCE ABORT during descent: |F|={f:.1f}N at offset {d*1000:+.2f}mm")
            if d >= target - 1e-9:
                break
            step = 0.0005 if f <= 3.0 else 0.00025
            d = min(d + step, float(target))
        # _last_depth_cmd starts AT the landed target: under LAND_AT_TARGET the first tick's
        # depth_cmd equals the same target (caller passes the matching residual), so the rate
        # limiter no-ops and the stroke never dips back toward nominal. Flag OFF: target ==
        # nominal, identical to the historic init.
        self._last_depth_cmd = target
        self._frac = self.start_frac                       # stroke frac counter starts here
        self._stroke_tick = 0                              # attack-ramp window restarts per stroke
        f_land = float(np.linalg.norm(self.ctrl.rtde_r.getActualTCPForce()[:3]))
        band_lo, band_hi = landing_band(target)            # depth-aware sanity band (2026-08-03)
        print(f"  reset: at contact+{target*1000:.1f}mm, frac={self.start_frac:.2f}, "
              f"|F|={f_land:.2f}N (band {band_lo:.2f}-{band_hi:.2f}N)")
        if not (band_lo < f_land < band_hi):
            if f_land <= band_lo and target < 0 and depth_override_m is not None:
                # The POLICY chose a lift (override < 0): landing off-string is legitimate —
                # lifting is the safe direction and quiet strokes are the reward gate's job,
                # not an abort (review M2: the calibration-stale diagnosis was wrong here).
                print(f"  (override lift {target*1000:+.1f}mm landed |F|={f_land:.2f}N ≈ "
                      f"off-string — proceeding as a quiet stroke)")
                return
            if f_land <= band_lo and target < 0:
                # Hovering OFF-string at a NEGATIVE target: the line was likely RESTORED
                # (or re-taught) and this lifted nominal is now STALE — the line is fine.
                # (2026-07-10 review neg-depth-1: the generic message sent the operator
                # chasing a calibration problem that didn't exist.)
                self._abort_retract(f"landed |F|={f_land:.2f}N ≈ no contact at NEGATIVE target "
                                    f"({target * 1000:.1f}mm): the contact line has "
                                    f"probably been RESTORED and this lifted nominal is STALE — "
                                    f"re-probe the line and re-zero nominal toward 0.")
            self._abort_retract(f"landed |F|={f_land:.2f}N outside sanity band "
                                f"({band_lo:.2f}-{band_hi:.2f}N for a {target*1000:.1f}mm landing): "
                                f"contact has drifted — re-run verify_contact_line.py "
                                f"(waypoints/normal may need re-teaching).")

    # ------------------------------------------------------------- passthrough
    def is_complete(self):
        if self._frac is not None and self._frac > STROKE_END_FRAC + 1e-9:
            return True                                    # frac window done (collect_tone convention)
        return self.ctrl.is_complete()                     # backstop: note bookkeeping exhausted

    def get_baseline_action(self):
        return self.ctrl.get_baseline_action()

    # ----------------------------------------------------------------- execute
    def execute_timestep(self, action):
        """Build the command from the calibrated line + bounded feed-forward depth."""
        self._require_safe("tick")                                    # R3
        fmag = float(np.linalg.norm(self.ctrl.rtde_r.getActualTCPForce()[:3]))
        if fmag > FORCE_ABORT_N:                                      # 5N + retract (K4)
            self._abort_retract(f"FORCE ABORT: |F|={fmag:.1f}N > {FORCE_ABORT_N}N")

        # OWN frac counter (collect_tone convention: frac += speed*TICK/SPAN per tick). The
        # controller's tcp_target/note timing no longer drives frac — it diverged from the
        # collection convention at any speed != nominal (fixed-dur vs fixed-window, 2026-06-30).
        lo, hi = coverage()
        if self._frac is None:
            self._frac = self.start_frac
            self._stroke_tick = 0
        raw_frac = self._frac
        adv = self.bow_speed * TICK / SPAN_M
        if ATTACK_RAMP and self._stroke_tick < len(ATTACK_SPEED_SCALE):
            adv *= ATTACK_SPEED_SCALE[self._stroke_tick]           # v1' onset speed ramp
        self._frac = raw_frac + adv                                # advance the RAW counter
        if (raw_frac < lo or raw_frac > hi) and not self._clip_warned:
            print(f"  !! bow frac {raw_frac:.2f} outside reachable coverage [{lo:.2f},{hi:.2f}] "
                  f"— CLIPPING to the edge (frog end exceeds reach; re-teach if you need more span).")
            self._clip_warned = True
        frac = float(np.clip(raw_frac, lo, hi))

        resid = float(np.clip(action.get("depth_residual", 0.0),
                              -DEPTH_RESIDUAL_CLAMP, DEPTH_RESIDUAL_CLAMP))
        depth_cmd = float(np.clip(self.nominal_depth + resid, DEPTH_MIN, MAX_PRESS_DEPTH))
        if (ATTACK_RAMP and depth_cmd > 0.0
                and self._stroke_tick < len(ATTACK_DEPTH_SCALE)):
            depth_cmd *= ATTACK_DEPTH_SCALE[self._stroke_tick]     # v1' light-side onset bias
            # (only when pressing; scaling a NEGATIVE depth would mean MORE contact at onset)
        if self._last_depth_cmd is not None:                          # R2 backstop
            depth_cmd = self._last_depth_cmd + float(np.clip(depth_cmd - self._last_depth_cmd,
                                                             -DEPTH_RATE_LIMIT, DEPTH_RATE_LIMIT))
        self._last_depth_cmd = depth_cmd
        # APPLIED residual (post clip + rate limit), not the requested one —
        # dev_depth in the obs must reflect what the robot actually did
        self._last_residual = float(depth_cmd - self.nominal_depth)

        pose = contact_pose(frac)
        pose[:3] += depth_cmd * E_PRESS
        pose = R.clamp_reach(pose)                                    # R11
        # force = BASELINE_FORCE_A makes the controller's hidden z-offset exactly 0 (K3)
        self.ctrl.execute_timestep(pose, self.ctrl.BASELINE_FORCE_A)
        self._stroke_tick += 1

    # --------------------------------------------------------------------- obs
    def get_full_state(self):
        s = self.ctrl.get_full_state()
        tcp = np.asarray(s["tcp"], dtype=float)
        _, d_raw = line_coords(tcp[:3])               # oblique: press dist clean of frac leak
        depth = d_raw                                 # taught line IS contact -> d_raw is depth past it

        now = time.time()
        accel = 0.0
        if self._prev_t is not None:
            dt = now - self._prev_t
            if dt > 1e-4:
                accel = float(np.clip((s["bow_speed"] - self._prev_speed) / dt,
                                      -ACCEL_LIMIT, ACCEL_LIMIT))
        self._prev_speed, self._prev_t = s["bow_speed"], now

        return {"depth": depth,
                "contact_force_magnitude": float(s["contact_force_magnitude"]),
                "bow_speed": float(s["bow_speed"]),
                "bow_position": float(s["bow_position"]),
                "bow_direction": float(s["bow_direction"]),
                "dev_depth": float(self._last_residual),
                "bow_accel": float(accel),
                "j4": float(s["j4"]), "j5": float(s["j5"]), "j6": float(s["j6"])}

    def close(self):
        """Stop streaming and, if we're near the string, get OFF it before
        disconnecting (don't leave the bow pressed; retract only from the
        proven near-string regime — never from a parked pose)."""
        try:
            self.ctrl.rtde_c.servoStop()
            tcp = np.array(self.ctrl.rtde_r.getActualTCPPose())
            if tcp[2] < 0.30 and safety_ok(self.ctrl):
                R.retract(self.ctrl)
        except Exception:
            pass
        self.ctrl.close()
