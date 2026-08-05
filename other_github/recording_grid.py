"""
recording_script.py

Data collection script for building the sound quality classifier dataset.

Records systematic bow strokes across force / speed / bow-zone conditions on a
SINGLE string, capturing synchronized audio + full RTDE state time series.

This script is intentionally intertwined with baseline_controller.py:
    - It reuses the SAME robot connection, TCP/payload calibration, frog/tip
      waypoints, z-offset force model (Z_PER_NEWTON), and StateLogger.
    - It does NOT parse a MIDI file. Instead it SYNTHESIZES note events directly
      from the parameter grid, so every commanded (force, speed, bow_zone) value
      is hit precisely rather than being quantized through MIDI velocity.
    - The bow waypoints (FROG_TARGET_STR / TIP_TARGET_STR) are defined LOCALLY
      in this file, taken from the working URScript and calibrated together with
      the 'out' retract vector. To record a different string, swap these for that
      string's frog/tip values (and TCP_OFFSET still comes from the controller).
      One run = one string.

Stroke mechanics:
    - Each stroke's trajectory is built EXPLICITLY from the bow-zone fractions,
      so the commanded bow direction is always honored (no note-sequence
      alternation). Up-bow reverses the zone fractions (tip side -> frog side).
    - Between strokes the bow is LIFTED off the string, travels while lifted,
      and is lowered back onto the exact start point — so the reposition move
      never drags across the string and never appears in the recorded audio.
    - The on-string bowing window is marked by wall-clock timestamps so the
      state summary excludes the lift/reposition moves (the full .npy time
      series still contains everything for debugging).

Usage:
    python recording_script.py

Testing mode:
    Set TESTING = True below to record ONE condition only (defined in TEST_*),
    bypassing the full grid. Test recordings are prefixed TEST_ so they don't
    pollute the real dataset.

Physical setup requires:
    - Microphone fixed position relative to cello body (not robot)
      - ~20-30cm from top plate, angled toward bridge
      - Mark with tape — must be identical every session
    - Audio chain: mic -> audio interface -> computer
      - Check no clipping (peak < 0.9)
      - Check SNR acceptable (quiet room)
    - Robot: tare F/T sensor at session start
    - Bow: rosin consistently
    - Record ambient noise level before starting
"""

import sys
from pathlib import Path

# OUR COPY (recording_grid.py): the controller lives in Baseline-Runners/ next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent / "Baseline-Runners"))

import numpy as np
import sounddevice as sd
import soundfile as sf
import json
import time
import random
from datetime import datetime

# Everything physical comes from the baseline controller so the two stay in sync.
# EXCEPTION: the bow waypoints. We override them with the values from the working
# URScript (song()), which are calibrated together with the 'out' retract vector
# and use distinct frog/tip orientations (the wrist rotates along the stroke) —
# a better-conditioned geometry than the controller's shared-orientation pair.
from baseline_controller import (
    TargetStringBaselineController,
    velocity_to_dynamic,
    FORCE_MIN,
    FORCE_MAX,
    SPEED_MIN,
    SPEED_MAX,
)


# ── Bow waypoints (A string, from the working paganini URScript) ──
# Order: [x, y, z, rx, ry, rz]. These are the VALIDATED A-string poses from the
# working paganini program (which bows a_tip<->a_frog cleanly). Frog and tip
# share orientation. These match baseline_controller's FROG/TIP_TARGET_STR.
# NOTE: a_frog is at ~0.85 m reach (the UR5e limit), so lifting OUTWARD near the
# frog will exceed reach — lifting is only safe in the tip half of the bow.
# RE-TAUGHT 2026-06-21 after cello move + bow swap; CORRECTED 2026-06-22 from the flange frame
# to the bow-tip TCP frame (pendant installation TCP was [0,0,0]=flange, so the read-only capture
# was 141.3mm off; converted p_tip = p_flange + R(ori)@TCP_OFFSET). z -0.018->+0.0995 ~ old 0.104.
# RE-CAPTURED 2026-06-27 (capture_waypoints_local.py: READ-ONLY LOCAL mode, NEW angle, pendant
# translation-jog -> orientation held to 0.00deg). The robot TCP was ALREADY the bow-tip (left by
# teach_tip_on_axis's setTcp), so getActualTCPPose was already bow-tip — NO flange->tip conversion
# needed. frog bow-tip reach 0.817, z 0.100 (~old 0.0995); cello drifted slightly closer.
FROG_TARGET_STR = np.array([0.346316666, 0.733120691, 0.100027869,
                            -1.423855578, -2.172165015, 1.374086581])
TIP_TARGET_STR  = np.array([0.537882319, 0.304981426, 0.243073479,
                            -1.423855578, -2.172165015, 1.374086581])

# Physical bow length (meters) = positional distance frog<->tip.
BOW_LENGTH_TARGET_STR = float(np.linalg.norm(TIP_TARGET_STR[:3] - FROG_TARGET_STR[:3]))


# ══════════════════════════════════════════════════════════════════
# TESTING MODE
# ══════════════════════════════════════════════════════════════════
# When True, record ONLY the single condition defined below, then exit.
# Useful for verifying the audio chain, robot motion, lift/reset, and the
# save pipeline without running the full grid.
TESTING = False

TEST_DEPTH    = 0.0015           # m of press past light contact (dynamic level)
TEST_SPEED    = 0.10             # m/s
TEST_ZONE     = (0.45, 0.75)     # (start_fraction, end_fraction) frog=0 tip=1
TEST_BOW_DIR  = 'down'           # 'down' (frog->tip) or 'up' (tip->frog)
TEST_REPEATS  = 2                # how many recordings of this one condition
TEST_SKIP_CONSISTENCY = True     # skip the 5-stroke consistency check in testing


# ── String identity ───────────────────────────────────────────────
# The baseline controller is calibrated for ONE string at a time. Set this to
# match whatever FROG/TIP waypoints are loaded there so filenames and metadata
# are labelled correctly. (Cosmetic — the physical string is determined entirely
# by the controller's waypoints.)
RECORDING_STRING = 'A'


# ── Audio config ──────────────────────────────────────────────────
AUDIO_DEVICE  = None
SAMPLE_RATE   = 44100
PRE_BUFFER    = 0.2     # silence before stroke
POST_BUFFER   = 0.3     # silence after stroke
N_REPEATS     = 5       # recordings per condition (full session)
CHANNELS      = 1       # mono audio


# ── Reset / lift config ───────────────────────────────────────────
# To reset between strokes WITHOUT dragging on the string, the bow retracts
# along RETRACT_DIR_BASE (validated with find_liftoff) to this clearance, travels
# at clearance to above the next start point, then descends. find_liftoff
# confirmed 50mm clears cleanly on the real robot; 60mm adds margin and stays
# well within reach in the tip-half / upper_middle test zone.
LIFT_DISTANCE   = 0.060   # 60mm retract along the validated direction
SETTLE_SEC      = 0.3     # let contact settle after lowering before bowing
MARK_SETTLE_SEC = 0.05    # let the logger capture a settled sample or two


OUTPUT_DIR    = Path("dataset")
AUDIO_DIR     = OUTPUT_DIR / "audio"
STATE_DIR     = OUTPUT_DIR / "states"
META_FILE     = OUTPUT_DIR / "metadata.jsonl"
SESSION_LOG   = OUTPUT_DIR / "sessions.jsonl"


# ── Parameter grid ────────────────────────────────────────────────
# DEPTH is the dynamic control: how far the bow presses into the string past
# light contact (deeper = louder/heavier tone). Specified directly in METERS,
# NOT converted from Newtons — we vary depth as the experimental knob and let the
# resulting force/tone be whatever it is. Kept small and safe (sub-mm to a few
# mm); the working motion used light contact (~0 depth) cleanly.
# Speeds in m/s (realized through stroke DURATION = distance / speed).

DEPTH_LEVELS = {            # meters of press past light contact
    'very_light': 0.0000,   # just touching (lightest tone)
    'light':      0.0008,
    'medium':     0.0015,
    'firm':       0.0025,
    'heavy':      0.0035,   # near the safe ceiling (MAX_PRESS_DEPTH)
}

SPEED_LEVELS = {
    'very_slow': 0.05,  # m/s (near SPEED_MIN)
    'slow':      0.07,
    'medium':    0.10,
    'fast':      0.13,
    'very_fast': 0.15,  # near SPEED_MAX
}

# Bow zones as (start_fraction, end_fraction) along frog(0) -> tip(1).
BOW_ZONES = {
    'frog':         (0.05, 0.35),   # lower third
    'lower_middle': (0.25, 0.55),
    'upper_middle': (0.45, 0.75),
    'tip':          (0.65, 0.95),   # upper third
}

# Deliberately bad conditions — extreme depth/speed to capture poor tone.
# depth in meters; clamped to MAX_PRESS_DEPTH at runtime for safety.
BAD_CONDITIONS = [
    {'depth': 0.0040, 'speed': 0.10, 'label': 'too_much_press'},
    {'depth': 0.0000, 'speed': 0.10, 'label': 'barely_touching'},
    {'depth': 0.0015, 'speed': 0.02, 'label': 'too_slow'},
    {'depth': 0.0015, 'speed': 0.30, 'label': 'too_fast'},
    {'depth': 0.0040, 'speed': 0.03, 'label': 'heavy_press_slow'},
    {'depth': 0.0000, 'speed': 0.28, 'label': 'light_fast'},
]


# ── Synthetic note construction ───────────────────────────────────

def make_stroke_note(depth, speed, bow_zone, bow_dir):
    """
    Build a single synthetic stroke descriptor.

    depth: meters the bow presses into the string past light contact — the
    dynamic control (deeper = heavier tone). Clamped to MAX_PRESS_DEPTH.
    Speed is realized via duration: the bow traverses the zone's physical
    distance at the target speed, so duration = distance / speed.
    """
    zone_fraction = abs(bow_zone[1] - bow_zone[0])
    distance      = zone_fraction * BOW_LENGTH_TARGET_STR   # meters
    duration      = distance / speed                        # seconds

    depth = float(np.clip(depth, 0.0, MAX_PRESS_DEPTH))

    # Dynamic label from depth (just a descriptive tag for metadata).
    frac_depth = depth / MAX_PRESS_DEPTH if MAX_PRESS_DEPTH > 0 else 0.0
    velocity   = int(np.clip(1 + frac_depth * 126, 1, 127))

    return {
        'duration':        float(duration),
        'velocity':        velocity,
        'dynamic':         velocity_to_dynamic(velocity),
        'bow_dir':         bow_dir,
        'string':          RECORDING_STRING,
        'commanded_depth': depth,
        'commanded_speed': float(speed),
        'bow_start':       float(bow_zone[0]),
        'bow_end':         float(bow_zone[1]),
    }


# ── Audio helpers ─────────────────────────────────────────────────

def list_audio_devices():
    """Print all available audio devices with their indices."""
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print()


def find_focusrite_device() -> int:
    """Auto-detect Focusrite Scarlett device index."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if 'Scarlett' in dev['name'] or 'Focusrite' in dev['name']:
            if dev['max_input_channels'] >= 1:
                print(f"Found Focusrite at index {i}: {dev['name']}")
                return i
    raise RuntimeError(
        "Focusrite not found. Run list_audio_devices() to see available devices "
        "and set AUDIO_DEVICE manually."
    )


# ── Geometry: retract direction (validated on hardware) + reach guard ─
# RE-DERIVED 2026-06-22 by find_normal.py at frac 0.5: force-vector and lift-off-gradient
# methods AGREED within 5.6deg. It came out 66deg off the old value -> the cello "move" was a
# big ROTATION, not a translation (consistent with the 170mm re-teach inconsistency + the large
# lateral contact force). Now mostly +Y (the string normal faces sideways, not up).
#   OLD (pre-rotation, find_liftoff, "strongly upward" z=0.72): [0.68861, 0.031962, 0.724428]
# Press/lift axis = the BOW's TOOL Z (NOT base Z). Zixian (2026-06-27) jogged TOOL +Z in LOCAL and
# the bow pressed INTO the string; TOOL -Z lifted off cleanly. Converted to base via the captured
# bow orientation [-1.4239,-2.1722,1.3741]: RETRACT/lift-off = R(ori)@[0,0,-1], E_PRESS/press =
# R(ori)@[0,0,+1]. (base +-Z was 57deg wrong; the old +Y normal 33deg off; find_normal too noisy.)
RETRACT_DIR_BASE = np.array([0.597784964, 0.586472995, 0.546536882])
RETRACT_DIR_BASE = RETRACT_DIR_BASE / np.linalg.norm(RETRACT_DIR_BASE)

# Reach guard: never command a target whose distance from base exceeds this.
# UR5e max reach is ~0.85 m; stay under with margin.
MAX_SAFE_REACH = 0.83


def clamp_reach(pose, max_reach=MAX_SAFE_REACH):
    """
    If pose's position is farther than max_reach from the base, pull it radially
    inward to sit exactly at max_reach. Orientation unchanged. Prevents reach /
    joint-limit faults from any move target.
    """
    out = np.array(pose, dtype=float).copy()
    r = np.linalg.norm(out[:3])
    if r > max_reach:
        out[:3] = out[:3] * (max_reach / r)
    return out


def offset_along(pose, base_dir, distance):
    """Translate pose by `distance` m along a BASE-frame direction, then reach-clamp."""
    out = np.array(pose, dtype=float).copy()
    out[:3] = out[:3] + distance * np.asarray(base_dir, dtype=float)
    return clamp_reach(out)


def offset_along_normal(pose, distance):
    """
    Translate pose by `distance` m along the retract direction (away from the
    string for +distance), reach-clamped. Orientation unchanged.
    """
    return offset_along(pose, RETRACT_DIR_BASE, distance)


# ── Force <-> contact-depth mapping ───────────────────────────────
# The baseline model assumes the WAYPOINT already gives BASELINE_FORCE_A (3 N)
# and Z_PER_NEWTON adjusts around it. In reality the tared sensor shows the
# waypoint gives only ~0.3 N, so that model is offset by ~2.7 N and the slope is
# unverified. calibrate_force_depth() measures the true relationship and fills
# these globals; until then we fall back to the controller's (wrong) model.
#
#   force ≈ FORCE_AT_WAYPOINT + (depth_into_string / CAL_M_PER_N_inverse)
# We store it as: depth(force) = (force - FORCE_AT_WAYPOINT) * CAL_M_PER_N
FORCE_AT_WAYPOINT = None   # N, measured contact force at the raw waypoint
CAL_M_PER_N       = None   # m of press per additional N (measured slope)

# HARD SAFETY CAP: never press deeper than this into the string, no matter what
# the model says. A bad calibration once commanded 46-94mm; that overloads the
# arm and protective-stops it. Real bow press is sub-millimeter to a few mm.
MAX_PRESS_DEPTH = 0.004    # 4mm absolute ceiling on press depth


def force_to_z_offset(ctrl, force):
    """
    Commanded force (N) -> press depth (m) into the string from the waypoint.
    Uses the MEASURED calibration if available; else the controller's model.
    ALWAYS clamped to [0, MAX_PRESS_DEPTH] for safety.
    """
    if FORCE_AT_WAYPOINT is not None and CAL_M_PER_N is not None:
        depth = (force - FORCE_AT_WAYPOINT) * CAL_M_PER_N
    else:
        depth = (force - ctrl.BASELINE_FORCE_A) * ctrl.Z_PER_NEWTON
    # Clamp: never press beyond the safety ceiling, never pull off the string.
    return float(np.clip(depth, 0.0, MAX_PRESS_DEPTH))


def apply_force_offset(pose, force_depth):
    """
    Apply force depth along the retract direction. Positive force_depth presses
    INTO the string (opposite the away-from-string retract direction).
    """
    return offset_along_normal(pose, -force_depth)


def calibrate_force_depth(ctrl, at_frac=0.60, step=0.00025, max_depth=0.006,
                          max_force=6.0):
    """
    Measure the true force<->depth relationship at a bow fraction, using the
    (per-zone tared) F/T sensor. Presses into the string in small steps, reads
    tared contact force at each depth, fits a line, and fills FORCE_AT_WAYPOINT
    and CAL_M_PER_N so commanded forces map to real Newtons.

    Run AFTER a per-zone tare at the same fraction. Stops at max_depth or
    max_force, whichever first (safety).
    """
    global FORCE_AT_WAYPOINT, CAL_M_PER_N
    print(f"\nCalibrating force<->depth at bow frac {at_frac:.2f}...")

    # Tare at this fraction first so contact force is meaningful here.
    zero_ft_in_free_air(ctrl, at_frac=at_frac, set_payload=False)

    # Descend onto the waypoint (no press) to establish the depth=0 reference.
    waypoint = bow_pose_full(at_frac)
    reset_to(ctrl, clamp_reach(waypoint))
    time.sleep(0.4)

    depths, forces = [], []
    depth = 0.0
    while depth <= max_depth:
        pressed = clamp_reach(apply_force_offset(waypoint, depth))
        ctrl.rtde_c.moveL(pressed.tolist(), ctrl.SERVO_VEL, ctrl.SERVO_ACCEL)
        time.sleep(0.3)
        f = float(np.linalg.norm(ctrl.rtde_r.getActualTCPForce()[:3]))
        depths.append(depth); forces.append(f)
        print(f"  depth={depth*1000:5.2f}mm  force={f:5.3f}N")
        if f >= max_force:
            print("  reached max_force cap.")
            break
        depth += step

    # Retract to safety.
    retract(ctrl)

    if len(depths) < 3:
        print("  Not enough points to fit. Aborting calibration.")
        return None

    depths = np.array(depths); forces = np.array(forces)

    # Fit force = a*depth + b.
    a, b = np.polyfit(depths, forces, 1)   # a = N per m, b = force at depth 0

    # SANITY CHECKS — reject a fit where force doesn't really respond to depth.
    force_range = float(forces.max() - forces.min())
    # correlation between depth and force
    if np.std(depths) > 0 and np.std(forces) > 0:
        corr = float(np.corrcoef(depths, forces)[0, 1])
    else:
        corr = 0.0
    print(f"\n  Fit: force(N) = {a:.1f} * depth(m) + {b:.3f}")
    print(f"  force range over sweep: {force_range:.3f} N, depth-force corr: {corr:.2f}")

    if force_range < 1.0 or corr < 0.5 or a <= 0:
        print("  REJECTED — force barely responded to depth (range < 1 N or weak")
        print("  correlation). Position-based force control is NOT working here:")
        print("  pressing the TCP in does not build predictable contact force")
        print("  over this range. NOT storing calibration; force offset stays 0.")
        print("  => Need a different force approach (see notes). Strokes will use")
        print("     the raw waypoint (light contact) until then.")
        # Leave FORCE_AT_WAYPOINT / CAL_M_PER_N as None so force_to_z_offset
        # falls back, and pin commanded force to baseline so offset is ~0.
        return None

    FORCE_AT_WAYPOINT = float(b)
    CAL_M_PER_N       = float(1.0 / a)
    print(f"  => FORCE_AT_WAYPOINT = {FORCE_AT_WAYPOINT:.3f} N")
    print(f"  => CAL_M_PER_N       = {CAL_M_PER_N:.6f} m/N")
    return FORCE_AT_WAYPOINT, CAL_M_PER_N




# ── Lift / reposition (no string contact) ─────────────────────────

def retract(ctrl):
    """
    Retract the bow off the string. OUR CHANGE: lift STRAIGHT UP first to clear the
    string vertically (no drag/scrape), then retract along the calibrated direction.
    """
    pose = np.array(ctrl.rtde_r.getActualTCPPose())
    pose[2] += 0.03                                   # +3 cm pure vertical -> off the string
    ctrl.rtde_c.moveL(pose.tolist(), ctrl.SERVO_VEL, ctrl.SERVO_ACCEL)
    retracted = offset_along_normal(np.array(ctrl.rtde_r.getActualTCPPose()), LIFT_DISTANCE)
    ctrl.rtde_c.moveL(retracted.tolist(), ctrl.SERVO_VEL, ctrl.SERVO_ACCEL)


def reset_to(ctrl, target_pose):
    """
    Move the bow to target_pose (ON the string) WITHOUT ever dragging across or
    dipping onto the string during the move:

      1. retract from current position to clearance (away from string)
      2. travel AT CLEARANCE to directly above the target (target + clearance
         along the target's own normal) — stays off the string the whole time
      3. descend along the target's normal onto the target

    Only step 3 approaches the string, and it approaches perpendicular (along
    the normal), so there is no lateral drag. Steps 1-2 keep the bow hovered.
    """
    # 1. retract straight off the string from where we are now.
    retract(ctrl)

    # 2. travel at clearance to directly above the target. Computing the
    #    clearance point from the TARGET's normal guarantees the descent in
    #    step 3 is purely perpendicular and the travel stays above the string.
    above_target = offset_along_normal(target_pose, LIFT_DISTANCE)
    ctrl.rtde_c.moveL(above_target.tolist(), ctrl.SERVO_VEL, ctrl.SERVO_ACCEL)

    # 3. descend perpendicular onto the target.
    ctrl.rtde_c.moveL(list(target_pose), ctrl.SERVO_VEL, ctrl.SERVO_ACCEL)
    time.sleep(SETTLE_SEC)


# ── F/T sensor zeroing (gravity/bias compensation) ────────────────

# Payload of everything mounted past the F/T sensor (bow + holder), in kg, and
# its center of gravity (m) in the tool frame. Measured on the UR teach pendant
# (Payload -> Measure): 0.190 kg, COG [128, -6, 19] mm -> [0.128, -0.006, 0.019] m.
# NOTE: pendant reports COG in mm; setPayload expects METERS.
FT_PAYLOAD_KG  = 0.190
FT_PAYLOAD_COG = [0.128, -0.006, 0.019]


def zero_ft_in_free_air(ctrl, clearance=0.120, at_frac=None, set_payload=True):
    """
    Zero the F/T sensor in free air so readings reflect STRING CONTACT, not tool
    weight. Order: (set payload) -> move well clear of the string -> zero.

    at_frac: if given, the bow first moves to a clearance pose ABOVE that bow
    fraction before zeroing, so the wrist ORIENTATION at tare-time matches the
    orientation during that zone's strokes. Because the residual gravity-comp
    error is ~constant within a zone (orientation barely changes), taring at the
    zone's orientation cancels it for that zone. This is the per-zone tare.

    set_payload: set the payload first (do this at least once per session).

    Uses a LARGE clearance (default 120mm) so the bow is unambiguously off the
    string when zeroing.
    """
    print("\nZeroing F/T sensor"
          + (f" for bow frac {at_frac:.2f}..." if at_frac is not None else "..."))

    if set_payload:
        ctrl.rtde_c.setPayload(FT_PAYLOAD_KG, FT_PAYLOAD_COG)
        time.sleep(0.2)
        try:
            m_now   = ctrl.rtde_r.getPayload()
            cog_now = ctrl.rtde_r.getPayloadCog()
            print(f"  payload readback: mass={m_now:.3f} kg (want {FT_PAYLOAD_KG:.3f}), "
                  f"cog={np.round(cog_now,4).tolist()} (want {FT_PAYLOAD_COG})")
            if abs(m_now - FT_PAYLOAD_KG) > 0.005:
                print("  !! Robot mass != requested — a stale payload is overriding.")
        except Exception as e:
            print(f"  (payload readback unavailable: {e})")

    # Move to a clearance pose. If at_frac given, position above that fraction so
    # the orientation matches the zone; else lift from the current pose.
    if at_frac is not None:
        base_pose = bow_pose_full(at_frac)
    else:
        base_pose = np.array(ctrl.rtde_r.getActualTCPPose())
    clear = clamp_reach(offset_along_normal(base_pose, clearance))
    ctrl.rtde_c.moveL(clear.tolist(), ctrl.SERVO_VEL, ctrl.SERVO_ACCEL)
    time.sleep(0.6)

    ok = ctrl.rtde_c.zeroFtSensor()
    time.sleep(0.3)
    f = float(np.linalg.norm(ctrl.rtde_r.getActualTCPForce()[:3]))
    print(f"  zeroFtSensor() returned {ok}; free-air force now {f:.3f} N (≈0).")
    return f


def verify_ft_at_poses(ctrl, n=4, check_clearance=0.120):
    """
    Sanity check: after zeroing, read free-air force at several wrist poses along
    the bow. All should be near 0 if gravity compensation is correct; a large
    spread means the payload/COG is still off.

    Uses a LARGE clearance (default 120mm, double the normal lift) so the bow is
    unambiguously off the string at every pose — a smaller clearance can graze on
    the tilted string and contaminate the reading with intermittent contact,
    which looks like (but isn't) a payload error. Retracts fully between poses so
    the travel never approaches the string.
    """
    print(f"\nChecking free-air force at several poses "
          f"(clearance {check_clearance*1000:.0f}mm, all should be ~0):")

    # Start from a guaranteed-clear pose.
    retract(ctrl)
    time.sleep(0.3)

    fracs  = np.linspace(0.45, 0.75, n)   # tip-half, reach-safe
    forces = []
    for fr in fracs:
        # Big clearance, reach-clamped. Move there via a clearly-airborne path.
        pose = clamp_reach(offset_along_normal(bow_pose_full(fr), check_clearance))
        ctrl.rtde_c.moveL(pose.tolist(), ctrl.SERVO_VEL, ctrl.SERVO_ACCEL)
        time.sleep(0.5)
        f = float(np.linalg.norm(ctrl.rtde_r.getActualTCPForce()[:3]))
        forces.append(f)
        print(f"  frac={fr:.2f}: {f:.3f} N")
    spread = float(np.max(forces) - np.min(forces))
    avg    = float(np.mean(forces))
    print(f"  spread across poses: {spread:.3f} N, mean {avg:.3f} N")
    if spread < 0.4 and avg < 0.4:
        print("  OK — pose-independent and near zero. Gravity comp is good.")
    elif spread < 0.4:
        print("  Pose-INDEPENDENT but not near zero: a constant offset remains —")
        print("  re-zero (zeroFtSensor) while clearly in free air.")
    else:
        print("  Pose-DEPENDENT: likely still grazing the string (increase")
        print("  check_clearance) OR payload/COG is off. Confirm the bow is fully")
        print("  clear at each pose by eye; if it is, recheck the payload values.")
    return forces


# EAR-GUIDED LINE TRIM (2026-07-20, Claude per Zixian's diagnosis): at low depths the stroke
# died in its second half -> the taught TIP end sits SHALLOWER than the string. The force
# probe could NOT measure this (0.8N contact threshold drowns in F/T tare drift — three
# back-to-back runs disagreed by 7mm); the EAR plus the one physically-consistent probe run
# both said "tip needs ~1.5mm more". Applied linearly along the span (frog 0 -> tip
# TIP_TRIM_M) INTO the string. Tune by LISTENING: a low-depth stroke must sound evenly
# across the whole span. Re-zero this at any anchor re-teach.
TIP_TRIM_M = 0.0015


def bow_pose_full(frac):
    """Full 6-DOF pose on the string at a bow fraction (for FT pose checks)."""
    p = FROG_TARGET_STR + frac * (TIP_TARGET_STR - FROG_TARGET_STR)
    p[:3] += float(frac) * TIP_TRIM_M * (-RETRACT_DIR_BASE)   # deepen toward the tip
    return p


# ── Verify the calibrated retract clears the string ───────────────

def probe_lift_axis(ctrl):
    """
    Verify the retract direction clears the string. Places the bow on the string
    at a nominal force, reads contact force, retracts along the 'out' vector,
    reads again. Contact force should drop toward free-air levels after retract.
    """
    mid_note = make_stroke_note(depth=0.0015, speed=0.10,
                                bow_zone=BOW_ZONES['upper_middle'], bow_dir='down')
    start_tcp, _ = stroke_endpoints(mid_note)

    on     = apply_force_offset(start_tcp, mid_note['commanded_depth'])
    reset_to(ctrl, on)                  # retract, travel at clearance, descend

    time.sleep(0.4)
    f_on = float(np.linalg.norm(ctrl.rtde_r.getActualTCPForce()[:3]))

    retract(ctrl)                       # retract along calibrated direction
    time.sleep(0.4)
    f_off = float(np.linalg.norm(ctrl.rtde_r.getActualTCPForce()[:3]))

    print(f"\nRetract verification (LIFT_DISTANCE={LIFT_DISTANCE*1000:.0f}mm):")
    print(f"  contact force on string:  {f_on:.3f} N")
    print(f"  contact force after retract: {f_off:.3f} N")
    print("  NOTE: the F/T sensor on this setup reads tool weight / wrist angle,")
    print("  not true string contact (it has read higher AFTER lifting). Do NOT")
    print("  trust these numbers for pass/fail. JUDGE BY WATCHING THE BOW:")
    print("    - Did it lift up-and-off the string at retract? -> good")
    print("    - Does it stay clear while traveling, only touching at descent? -> good")
    print("  (Tare the F/T sensor in baseline_controller to make these meaningful.)")
    return f_on, f_off


# ── Single-stroke execution ───────────────────────────────────────

# Force-mode config.
FORCE_MODE_TYPE       = 2          # 2 = task frame not transformed (we orient it)
FORCE_COMPLIANT_SPEED = 0.05       # m/s max TCP speed along the compliant (press) axis
FORCE_MAX_SAFE        = 8.0        # N — hard cap on commanded contact force
USE_FORCE_MODE        = False      # FALSE = position/depth stroke (the working one)
FORCE_SETTLE_S        = 0.6        # s to let force mode reach steady state before bowing

# Commanded->measured force correction. Force mode settles to a fraction of the
# commanded wrench against a compliant surface (measured ~0.68 of commanded).
# To achieve a TARGET contact force, command target / FORCE_MODE_GAIN. This is
# refined by calibrate_force_mode(); the default is the stationary-test estimate.
FORCE_MODE_GAIN       = 0.68       # measured / commanded (steady state)


def corrected_wrench_force(target_force):
    """Commanded wrench needed to ACHIEVE target_force, given the steady-state gain."""
    cmd = target_force / max(FORCE_MODE_GAIN, 1e-3)
    return float(np.clip(cmd, 0.0, FORCE_MAX_SAFE))


def _press_task_frame(ctrl):
    """
    Build a task-frame pose [x,y,z,rx,ry,rz] (base frame) whose Z-AXIS points
    INTO the string (= -RETRACT_DIR_BASE), so forceMode can push along it.
    Origin is the current TCP position. Returns the 6-vector pose.

    We construct a rotation whose third column (tool/task Z) equals the press
    direction, then convert to a rotation vector.
    """
    press = -RETRACT_DIR_BASE / np.linalg.norm(RETRACT_DIR_BASE)   # into string

    # Build an orthonormal frame with z = press. Pick an arbitrary perpendicular.
    z = press
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = helper - np.dot(helper, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])           # columns are frame axes in base

    # Rotation matrix -> rotation vector (axis-angle).
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-9:
        rotvec = np.array([0.0, 0.0, 0.0])
    else:
        rx = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
        ry = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
        rz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
        rotvec = theta * np.array([rx, ry, rz])

    pos = np.array(ctrl.rtde_r.getActualTCPPose())[:3]
    return [pos[0], pos[1], pos[2], rotvec[0], rotvec[1], rotvec[2]]


def calibrate_force_mode(ctrl, at_frac=0.60, test_forces=(2.0, 3.0, 4.0),
                         settle_s=1.5):
    """
    Measure the steady-state ratio (measured / commanded) for force mode at a
    bow fraction, across a few commanded levels, and update FORCE_MODE_GAIN so
    commanded target forces map to real measured Newtons.

    Stationary only (no bowing) — safe, since stationary force mode is stable
    here. Holds each commanded force for settle_s, averages the last half.
    """
    global FORCE_MODE_GAIN
    print(f"\nCalibrating force-mode gain at frac {at_frac:.2f}...")
    zero_ft_in_free_air(ctrl, at_frac=at_frac, set_payload=False)

    note = make_stroke_note(depth=0.0, speed=0.10,
                            bow_zone=(at_frac-0.05, at_frac+0.05), bow_dir='down')
    place_bow_at_stroke_start(ctrl, note)

    task_frame = _press_task_frame(ctrl)
    selection  = [0, 0, 1, 0, 0, 0]
    limits     = [0.1, 0.1, FORCE_COMPLIANT_SPEED, 0.1, 0.1, 0.1]

    ratios = []
    for cmd in test_forces:
        cmd_c = float(np.clip(cmd, 0.0, FORCE_MAX_SAFE))
        wrench = [0, 0, cmd_c, 0, 0, 0]
        samples = []
        t0 = time.time()
        while time.time() - t0 < settle_s:
            ctrl.rtde_c.forceMode(task_frame, selection, wrench, FORCE_MODE_TYPE, limits)
            time.sleep(0.05)
            if time.time() - t0 > settle_s * 0.5:   # average the settled half
                samples.append(float(np.linalg.norm(ctrl.rtde_r.getActualTCPForce()[:3])))
        meas = float(np.mean(samples)) if samples else 0.0
        ratios.append(meas / cmd_c if cmd_c > 0 else 0.0)
        print(f"  commanded {cmd_c:.1f} N -> measured {meas:.2f} N "
              f"(ratio {ratios[-1]:.2f})")

    ctrl.rtde_c.forceModeStop()
    time.sleep(0.1)
    retract(ctrl)

    gain = float(np.mean(ratios))
    if 0.2 < gain < 1.5:
        FORCE_MODE_GAIN = gain
        print(f"  => FORCE_MODE_GAIN = {FORCE_MODE_GAIN:.3f} "
              f"(commanded force will be divided by this to hit the target)")
    else:
        print(f"  gain {gain:.2f} out of expected range; keeping "
              f"{FORCE_MODE_GAIN:.2f}. Inspect readings.")
    return FORCE_MODE_GAIN


def test_force_mode_stationary(ctrl, force=3.0, at_frac=0.60, hold_s=2.0):
    """
    SAFETY TEST: engage force mode while STATIONARY (no bowing) and watch whether
    the contact force settles to the target without lunging. Run this before any
    force-mode bowing. Place at light contact, tare, engage force mode, hold, and
    print the measured force over a couple seconds.
    """
    print(f"\nStationary force-mode test: target {force:.1f} N at frac {at_frac:.2f}")
    zero_ft_in_free_air(ctrl, at_frac=at_frac, set_payload=False)

    # Light contact at the test point.
    note = make_stroke_note(depth=0.0, speed=0.10,
                            bow_zone=(at_frac-0.05, at_frac+0.05), bow_dir='down')
    place_bow_at_stroke_start(ctrl, note)

    f_cmd      = float(np.clip(force, 0.0, FORCE_MAX_SAFE))
    task_frame = _press_task_frame(ctrl)
    selection  = [0, 0, 1, 0, 0, 0]
    wrench     = [0, 0, f_cmd, 0, 0, 0]
    limits     = [0.1, 0.1, FORCE_COMPLIANT_SPEED, 0.1, 0.1, 0.1]

    print("  engaging force mode (watch the bow — it should settle, not lunge)...")
    t0 = time.time()
    while time.time() - t0 < hold_s:
        ctrl.rtde_c.forceMode(task_frame, selection, wrench, FORCE_MODE_TYPE, limits)
        time.sleep(0.05)
        f = float(np.linalg.norm(ctrl.rtde_r.getActualTCPForce()[:3]))
        print(f"    t={time.time()-t0:4.1f}s  measured |F|={f:.2f} N (target {f_cmd:.1f})")

    ctrl.rtde_c.forceModeStop()
    time.sleep(0.1)
    retract(ctrl)
    print("  force mode released, retracted.")


def execute_stroke_force_mode(ctrl: TargetStringBaselineController, note: dict):
    """
    Execute ONE stroke while maintaining a commanded CONTACT FORCE via forceMode.

    Force mode makes the robot compliant ONLY along the press axis (task-frame
    Z = into the string) and holds the commanded force there. The lateral bowing
    is driven by speedL (a streaming Cartesian VELOCITY command), NOT moveL.
    moveL is a blocking point-to-point move that fights force mode and faults the
    controller; speedL is designed to run alongside force mode.

    Sequence: forceMode on -> speedL along the bow until the stroke distance is
    covered -> speedL stop -> forceModeStop -> retract.

    PRECONDITION: bow already lightly placed on the string at the stroke start.
    """
    start_tcp, end_tcp = stroke_endpoints(note)
    target_force = float(np.clip(note['commanded_force'], 0.0, FORCE_MAX_SAFE))
    force = corrected_wrench_force(target_force)   # command more to achieve target
    speed = float(note['commanded_speed'])

    # Bow direction (unit) and total distance in base frame.
    bow_vec  = np.array(end_tcp[:3]) - np.array(start_tcp[:3])
    distance = float(np.linalg.norm(bow_vec))
    if distance < 1e-6:
        return {}
    bow_unit = bow_vec / distance

    # Cartesian velocity vector along the bow — but FIRST remove any component
    # along the press axis. The press axis (into the string) is force-mode
    # compliant; if speedL also commands motion along it, the two controllers
    # FIGHT and the bow drives into the string (crash). On this tilted string the
    # raw bow direction has ~41% along the press axis, so this projection is
    # essential. After projection, speedL only drives the bowing plane and force
    # mode owns the press axis alone.
    press_unit = -RETRACT_DIR_BASE / np.linalg.norm(RETRACT_DIR_BASE)
    bow_in_plane = bow_unit - np.dot(bow_unit, press_unit) * press_unit
    n = np.linalg.norm(bow_in_plane)
    if n < 1e-6:
        print("  (bow direction is entirely along press axis?! aborting stroke)")
        return {}
    bow_in_plane /= n
    vel = [speed * bow_in_plane[0], speed * bow_in_plane[1], speed * bow_in_plane[2],
           0, 0, 0]

    # Force-mode setup: compliant only on the press axis (task-frame Z).
    task_frame = _press_task_frame(ctrl)
    selection  = [0, 0, 1, 0, 0, 0]
    wrench     = [0, 0, force, 0, 0, 0]
    limits     = [0.1, 0.1, FORCE_COMPLIANT_SPEED, 0.1, 0.1, 0.1]

    start_pos = np.array(ctrl.rtde_r.getActualTCPPose())[:3]

    ctrl.state_logger.start()
    time.sleep(MARK_SETTLE_SEC)
    bow_t_start = time.time()

    # Engage force mode and let it reach steady state before bowing.
    t_settle = time.time()
    while time.time() - t_settle < FORCE_SETTLE_S:
        ctrl.rtde_c.forceMode(task_frame, selection, wrench, FORCE_MODE_TYPE, limits)
        time.sleep(0.05)

    # Bow with speedL, refreshing force mode each tick, until distance covered.
    # Progress is measured as displacement PROJECTED onto the in-plane bow
    # direction (the press axis is owned by force mode and excluded). The target
    # in-plane distance is the full stroke length projected the same way.
    target_in_plane = distance * float(np.dot(bow_unit, bow_in_plane))
    dt          = 0.02                       # 50 Hz control tick
    accel       = 0.5
    timeout     = (target_in_plane / max(speed, 1e-3)) * 2.0 + 1.0   # safety
    t0          = time.time()
    while True:
        ctrl.rtde_c.forceMode(task_frame, selection, wrench, FORCE_MODE_TYPE, limits)
        ctrl.rtde_c.speedL(vel, accel, dt)
        disp = np.array(ctrl.rtde_r.getActualTCPPose())[:3] - start_pos
        traveled = float(np.dot(disp, bow_in_plane))   # progress along bow plane
        if traveled >= target_in_plane:
            break
        if time.time() - t0 > timeout:
            print("  (stroke timeout — stopping)")
            break
        time.sleep(dt)

    bow_t_end = time.time()

    # Stop motion, release force mode, retract.
    ctrl.rtde_c.speedL([0, 0, 0, 0, 0, 0], accel, 0.1)
    ctrl.rtde_c.forceModeStop()
    time.sleep(0.1)
    retract(ctrl)

    ctrl.state_logger.stop()
    return ctrl.state_logger.get_summary()  # OUR CHANGE: no-arg (her StateLogger has no window)


def execute_stroke_via_controller(ctrl: TargetStringBaselineController, note: dict):
    """
    Execute exactly ONE stroke in the commanded direction.

    Dispatches to force-mode (command real contact force) or the older
    position-only stroke based on USE_FORCE_MODE.

    PRECONDITION: the bow is already placed ON the string at the stroke's start
    position (the caller handles placement via place_bow_at_stroke_start()).
    """
    if USE_FORCE_MODE:
        return execute_stroke_force_mode(ctrl, note)

    # ---- position stroke: depth is the dynamic control (the working path) ----
    start_tcp, end_tcp = stroke_endpoints(note)

    # Press the bow into the string by the commanded DEPTH (the dynamic level).
    # Same depth at start and end so contact is constant across the stroke.
    z_off     = note['commanded_depth']
    end_press = apply_force_offset(end_tcp, z_off)

    # Bow speed: moveL travels start->end at this linear speed (m/s). This is the
    # same effect the URScript bow() gets from movel(..., t=note_dur), since
    # commanded_speed = stroke_distance / note_duration.
    speed = note['commanded_speed']
    accel = 0.5   # m/s^2 — gentle accel so onset isn't a transient thunk

    # Start logging, mark the bowing window, run the single stroke move.
    ctrl.state_logger.start()
    time.sleep(MARK_SETTLE_SEC)
    bow_t_start = time.time()        # window opens

    # The bow is already placed at start_press (by place_bow_at_stroke_start),
    # so the stroke is a SINGLE clean moveL to the end at commanded speed.
    ctrl.rtde_c.moveL(end_press.tolist(), speed, accel)

    bow_t_end = time.time()          # window closes (before retract)

    ctrl.state_logger.stop()         # OUR CHANGE: stop BEFORE retract so the summary is bow-only
    # Retract so the bow doesn't drag during the post-buffer audio.
    retract(ctrl)

    return ctrl.state_logger.get_summary()  # OUR CHANGE: no-arg (her StateLogger has no window)


def stroke_endpoints(note: dict):
    """Return (start_tcp, end_tcp) for a stroke, honoring bow direction. Reach-clamped."""
    bow_start = note['bow_start']
    bow_end   = note['bow_end']
    if note['bow_dir'] == 'up':
        bow_start, bow_end = 1.0 - bow_start, 1.0 - bow_end
    start_tcp = FROG_TARGET_STR + bow_start * (TIP_TARGET_STR - FROG_TARGET_STR)
    end_tcp   = FROG_TARGET_STR + bow_end   * (TIP_TARGET_STR - FROG_TARGET_STR)
    return clamp_reach(start_tcp), clamp_reach(end_tcp)


def place_bow_at_stroke_start(ctrl, note: dict):
    """
    Place the bow on the string at this stroke's start position, pressed in by
    the commanded depth (the dynamic level), so contact is set before bowing.

    Uses reset_to: retract from wherever the bow is, travel at clearance to above
    the start, then descend perpendicular onto it.
    """
    start_tcp, _ = stroke_endpoints(note)
    if USE_FORCE_MODE:
        target = start_tcp                       # light contact; force mode presses
    else:
        target = apply_force_offset(start_tcp, note['commanded_depth'])
    reset_to(ctrl, clamp_reach(target))


# ── Recording function ────────────────────────────────────────────

def record_stroke(
    ctrl: TargetStringBaselineController,
    depth: float,
    speed: float,
    bow_zone: tuple,
    condition_label: str,
    stroke_id: str,
    session_id: str,
    bow_dir: str = 'down',
    n_repeats: int = N_REPEATS,
    device: int = AUDIO_DEVICE,
    tare_per_condition: bool = True,
) -> list:
    """
    Execute one condition N times and save each recording (audio + state).
    Returns list of metadata dicts saved.

    tare_per_condition: re-zero the F/T sensor in free air at this zone's
    orientation before recording. The gravity-comp residual is ~constant within
    a zone, so a tare at the zone orientation cancels it, making the contact
    force readings for this condition trustworthy. Payload is NOT re-set here
    (done once at session start); we only re-zero.
    """
    saved = []

    # Per-condition tare at the zone midpoint orientation.
    if tare_per_condition:
        zone_mid = 0.5 * (bow_zone[0] + bow_zone[1])
        zero_ft_in_free_air(ctrl, at_frac=zone_mid, set_payload=False)

    for repeat in range(n_repeats):
        note = make_stroke_note(depth, speed, bow_zone, bow_dir)

        # ── Reposition BEFORE audio so the reset move is silent/off-record ──
        # place_bow_at_stroke_start -> reset_to retracts from wherever the bow
        # is (on the string for repeat 0, already retracted after a stroke),
        # travels at clearance, and descends perpendicular onto the start. No
        # separate lift needed.
        place_bow_at_stroke_start(ctrl, note)

        # Audio length scales with the (speed-dependent) stroke duration.
        stroke_dur    = note['duration']
        total_samples = int((PRE_BUFFER + stroke_dur + POST_BUFFER) * SAMPLE_RATE)

        audio_buf = sd.rec(
            total_samples,
            samplerate = SAMPLE_RATE,
            channels   = CHANNELS,
            dtype      = 'float32',
            device     = device,
        )

        time.sleep(PRE_BUFFER)
        stroke_start = time.time()

        # Bow is already placed on the string; this only bows + lifts off.
        summary = execute_stroke_via_controller(ctrl, note)

        stroke_end = time.time()
        time.sleep(POST_BUFFER)
        sd.wait()

        fname_audio = f"{stroke_id}_r{repeat:02d}.wav"
        fname_state = f"{stroke_id}_r{repeat:02d}_state.npy"

        sf.write(str(AUDIO_DIR / fname_audio), audio_buf, SAMPLE_RATE)
        ctrl.state_logger.save(str(STATE_DIR / fname_state))

        peak = float(np.max(np.abs(audio_buf)))
        if peak > 0.95:
            print(f"  Audio clipping detected: peak={peak:.3f}")

        meta = {
            'stroke_id':        stroke_id,
            'repeat':           repeat,
            'session_id':       session_id,
            'timestamp':        datetime.now().isoformat(),
            'audio_file':       fname_audio,
            'state_file':       fname_state,
            'string':           RECORDING_STRING,

            'commanded': {
                'depth_m':     float(depth),
                'speed':       float(speed),
                'bow_start':   float(bow_zone[0]),
                'bow_end':     float(bow_zone[1]),
                'bow_dir':     bow_dir,
                'string':      RECORDING_STRING,
                'duration':    float(stroke_dur),
                'dynamic_est': note['dynamic'],
            },

            'measured': summary,

            'stroke_duration':  stroke_end - stroke_start,
            'audio_peak':       float(peak),

            'condition_label':  condition_label,
            'condition_type':   ('test' if condition_label.startswith('TEST_')
                                 else 'bad' if condition_label.startswith('bad_')
                                 else 'systematic'),

            'annotations':      [],
        }

        with open(META_FILE, 'a') as f:
            f.write(json.dumps(meta) + '\n')

        saved.append(meta)

        print(f"  {fname_audio} | "
              f"depth_cmd={depth*1000:.2f}mm  "
              f"F_meas={summary.get('force_mean', 0):.2f}±{summary.get('force_std', 0):.2f}N | "
              f"v_cmd={speed:.3f}  v_meas={summary.get('speed_mean', 0):.3f}m/s | "
              f"bow={summary.get('bow_pos_start', 0):.2f}→{summary.get('bow_pos_end', 0):.2f} | "
              f"dur={stroke_dur:.2f}s  peak={peak:.3f}")

        time.sleep(0.3)

    return saved


# ── Testing mode: one condition ───────────────────────────────────

def run_test_recording(ctrl: TargetStringBaselineController):
    """
    Record a SINGLE condition (TEST_* settings) for TEST_REPEATS repeats.
    Files/metadata are prefixed TEST_ so they don't mix with the real dataset.
    """
    session_id = "TEST_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    label     = f"TEST_D{TEST_DEPTH*1000:.1f}mm_S{TEST_SPEED}_Z{TEST_ZONE[0]}-{TEST_ZONE[1]}_{TEST_BOW_DIR}"
    stroke_id = f"{session_id}_{RECORDING_STRING}_{label}"

    print(f"\n{'='*60}")
    print(f"TESTING MODE — single condition")
    print(f"{'='*60}")
    print(f"  String:   {RECORDING_STRING}")
    print(f"  Depth:    {TEST_DEPTH*1000:.2f} mm (dynamic level)")
    print(f"  Speed:    {TEST_SPEED} m/s")
    print(f"  Zone:     {TEST_ZONE}  ({TEST_BOW_DIR} bow)")
    print(f"  Repeats:  {TEST_REPEATS}")
    print(f"{'='*60}\n")

    record_stroke(
        ctrl            = ctrl,
        depth           = TEST_DEPTH,
        speed           = TEST_SPEED,
        bow_zone        = TEST_ZONE,
        condition_label = label,
        stroke_id       = stroke_id,
        session_id      = session_id,
        bow_dir         = TEST_BOW_DIR,
        n_repeats       = TEST_REPEATS,
    )

    print(f"\n{'='*60}")
    print(f"Test complete: {TEST_REPEATS} recording(s)")
    print(f"Audio:    {AUDIO_DIR}")
    print(f"States:   {STATE_DIR}")
    print(f"Metadata: {META_FILE}  (prefixed TEST_)")
    print(f"{'='*60}\n")


# ── Full session ──────────────────────────────────────────────────

def run_collection_session(ctrl: TargetStringBaselineController, randomize: bool = True):
    """
    Run one complete data collection session for the current string.
    Records all systematic conditions + bad conditions.
    """
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Session: {session_id}   String: {RECORDING_STRING}")
    print(f"{'='*60}")

    conditions = []
    bow_dir    = 'down'

    for depth_name, depth_val in DEPTH_LEVELS.items():
        for speed_name, speed_val in SPEED_LEVELS.items():
            for zone_name, zone in BOW_ZONES.items():
                conditions.append({
                    'depth':   depth_val,
                    'speed':   speed_val,
                    'zone':    zone,
                    'bow_dir': bow_dir,
                    'label':   f"D{depth_name}_S{speed_name}_Z{zone_name}",
                    'type':    'systematic',
                })
                bow_dir = 'up' if bow_dir == 'down' else 'down'

    if randomize:
        random.shuffle(conditions)

    for bad in BAD_CONDITIONS:
        conditions.append({
            'depth':   bad['depth'],
            'speed':   bad['speed'],
            'zone':    BOW_ZONES['upper_middle'],
            'bow_dir': 'down',
            'label':   f"bad_{bad['label']}",
            'type':    'bad',
        })

    total_recordings = len(conditions) * N_REPEATS
    print(f"Conditions: {len(conditions)} ({len(BAD_CONDITIONS)} bad)")
    print(f"Recordings: {total_recordings}")
    print()

    stroke_count = 0

    for i, cond in enumerate(conditions):
        stroke_id = f"{session_id}_{RECORDING_STRING}_{cond['label']}_{i:03d}"
        print(f"\n[{i+1}/{len(conditions)}] {cond['label']}")

        record_stroke(
            ctrl            = ctrl,
            depth           = cond['depth'],
            speed           = cond['speed'],
            bow_zone        = cond['zone'],
            condition_label = cond['label'],
            stroke_id       = stroke_id,
            session_id      = session_id,
            bow_dir         = cond['bow_dir'],
            n_repeats       = N_REPEATS,
        )

        stroke_count += N_REPEATS

        rest = 30 if i % 5 == 4 else 5
        if i < len(conditions) - 1:
            print(f"  (resting {rest}s)")
            time.sleep(rest)

    session_log = {
        'session_id':   session_id,
        'string':       RECORDING_STRING,
        'date':         datetime.now().isoformat(),
        'n_conditions': len(conditions),
        'n_recordings': stroke_count,
        'depth_levels': DEPTH_LEVELS,
        'speed_levels': SPEED_LEVELS,
        'bow_zones':    {k: list(v) for k, v in BOW_ZONES.items()},
        'notes':        input("\nSession notes (ENTER to skip): ").strip(),
    }
    with open(SESSION_LOG, 'a') as f:
        f.write(json.dumps(session_log) + '\n')

    print(f"\n{'='*60}")
    print(f"Session complete: {stroke_count} recordings")
    print(f"Audio:    {AUDIO_DIR}")
    print(f"States:   {STATE_DIR}")
    print(f"Metadata: {META_FILE}")
    print(f"{'='*60}\n")


# ── Audio chain validation ────────────────────────────────────────

def validate_audio_chain(device: int):
    """Record 3 seconds of silence and check for clipping/noise."""
    print("Validating audio chain (3s recording)...")
    audio = sd.rec(
        int(3.0 * SAMPLE_RATE),
        samplerate = SAMPLE_RATE,
        channels   = CHANNELS,
        dtype      = 'float32',
        device     = device,
    )
    sd.wait()
    audio = audio.flatten()

    peak    = float(np.max(np.abs(audio)))
    rms     = float(np.sqrt(np.mean(audio**2)))
    db_peak = 20 * np.log10(peak + 1e-10)
    db_rms  = 20 * np.log10(rms + 1e-10)

    print(f"  Peak: {peak:.4f} ({db_peak:.1f} dBFS)  "
          f"RMS: {rms:.4f} ({db_rms:.1f} dBFS)")

    if peak > 0.1:
        print("  Silence is too loud: check for noise/hum")
    elif peak < 0.0001:
        print("  No signal detected: check mic connection")
    else:
        print("  Audio chain OK")

    return peak, rms


# ── Consistency check ─────────────────────────────────────────────

def check_robot_consistency(ctrl: TargetStringBaselineController, n_strokes: int = 5):
    """
    Run N identical strokes (medium force/speed, mid-bow) and check force
    consistency. If force_std > 0.3N, something is wrong.
    """
    print(f"\nConsistency check ({n_strokes} identical strokes)...")
    force_means = []

    test_zone = BOW_ZONES['upper_middle']
    for i in range(n_strokes):
        note    = make_stroke_note(depth=0.0015, speed=0.10, bow_zone=test_zone, bow_dir='down')
        place_bow_at_stroke_start(ctrl, note)
        summary = execute_stroke_via_controller(ctrl, note)
        fm      = summary.get('force_mean', 0)
        force_means.append(fm)
        print(f"  Stroke {i+1}: force_mean={fm:.3f}N")
        time.sleep(3)

    std = float(np.std(force_means))
    print(f"  Force consistency std: {std:.3f}N")

    if std > 0.3:
        print("  High force variance: check rosin, bow contact, F/T sensor")
    else:
        print("  Robot consistency OK")

    return std


# ── Entry point ───────────────────────────────────────────────────

if __name__ == '__main__':

    print("\nRobot Cello: Data Collection")
    print(f"String: {RECORDING_STRING}  (set by baseline_controller waypoints)")
    if TESTING:
        print("*** TESTING MODE ENABLED — single condition only ***")
    print("=" * 40)

    list_audio_devices()
    audio_device = find_focusrite_device()

    # One controller instance = the configured string's full calibration.
    # Empty event list; strokes are driven explicitly, not via note sequencing.
    ctrl = TargetStringBaselineController([])

    try:
        ctrl.reset()  # move to frog, verify contact

        validate_audio_chain(audio_device)

        # Session start: set payload once and do a baseline zero + diagnostic.
        # Per-CONDITION tares (in record_stroke) handle the pose-dependent
        # residual during actual recording, so verify_ft_at_poses here is just a
        # health check — a large spread is expected and is cancelled per-zone.
        input("\nPress ENTER to set payload and do a baseline F/T zero...")
        zero_ft_in_free_air(ctrl, set_payload=True)
        verify_ft_at_poses(ctrl)

        # Force-depth calibration is only for the legacy position stroke. With
        # force mode, the robot servos to the commanded force directly, so we
        # skip it.
        if not USE_FORCE_MODE:
            input("\nPress ENTER to calibrate force<->depth "
                  "(bow will press into the string in small steps)...")
            calibrate_force_depth(ctrl, at_frac=0.60)
        else:
            print("\nUSE_FORCE_MODE=True: skipping force-depth calibration "
                  "(force is servoed directly).")

        if TESTING:
            input("\nPress ENTER to verify the retract clears the string...")
            probe_lift_axis(ctrl)
            if not TEST_SKIP_CONSISTENCY:
                input("\nPress ENTER to continue to robot consistency check...")
                check_robot_consistency(ctrl)
            input("\nPress ENTER to begin TEST recording...")
            run_test_recording(ctrl)
        else:
            input("\nPress ENTER to continue to robot consistency check...")
            check_robot_consistency(ctrl)
            input("\nPress ENTER to begin data collection session...")
            run_collection_session(ctrl, randomize=True)

    finally:
        try:
            ctrl.rtde_c.forceModeStop()   # ensure we never exit still in force mode
        except Exception:
            pass
        ctrl.close()