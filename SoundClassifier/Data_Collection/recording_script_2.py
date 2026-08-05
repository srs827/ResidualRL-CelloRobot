"""
multi_string_recording.py

Multi-string bow-stroke data collection with a calibrated, hand-defined
safe-reset protocol.

PROTOCOL (per angle, per string)
    CALIBRATION (interactive, once per angle — saved to disk and reused):
      1. Free drive ON. User hovers the bow above the FROG area (a safe
         clearance height). Press ENTER. → frog_hover captured (pose + joints).
      2. Robot moveJ's slowly DOWN to the exact frog joint target (q_frog).
         (Small joint move from the hover → no snapping.)
      3. Robot moveL's frog → tip (one full bow) to confirm the stroke line.
      4. Free drive ON. User hovers above the TIP area. Press ENTER.
         → tip_hover captured.
      frog_hover + tip_hover define the safe "upper line" for resets.

    RECORDING (automatic, for that angle's depth×speed×bow_dir conditions):
      reset  = moveJ to start-hover  →  moveJ down to start (q_frog or q_tip)
      place  = small moveL applying press depth into the string
      stroke = single moveL along the string (frog↔tip), same as baseline
      lift   = moveJ up to the end-hover

WHY THIS WORKS
    Every large repositioning is a moveJ between hover/target JOINT configs
    that were physically reached during calibration — guaranteed reachable,
    same IK branch. The only moveL calls are: the tiny press, the stroke
    itself, and the calibration bow — all short and within one IK branch.

ANGLE_MODE = 'mid_only'   — mid angle per string  (4 calibrations)
ANGLE_MODE = 'all_angles' — top/mid/bottom        (up to 12 calibrations)

Calibration persists in calibration.json. Delete it to re-calibrate
(e.g. if the physical setup changed). Recording progress checkpoints
separately; Ctrl+C saves and re-running resumes.
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sounddevice as sd
import soundfile as sf

from BaselineControls.baseline_controller import (
    TargetStringBaselineController,
    velocity_to_dynamic,
)


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

TESTING      = False
STRING_ORDER = ['A', 'D', 'G', 'C']
ANGLE_MODE   = 'all_angles'   # 'mid_only' | 'all_angles'

AUDIO_DEVICE    = None
SAMPLE_RATE     = 44100
PRE_BUFFER      = 0.2
POST_BUFFER     = 0.3
N_REPEATS       = 5
CHANNELS        = 1

SETTLE_SEC      = 0.3
MARK_SETTLE_SEC = 0.05

# moveJ motion params (joint space). accel=1.0 rad/s^2 is a standard UR value
# (0.3 was rejected as invalid). Speed kept slow for safety/visibility.
MOVEJ_SPEED     = 0.25   # rad/s  (~14 deg/s)
MOVEJ_ACCEL     = 1.0    # rad/s^2
MOVEJ_SPEED_BIG = 0.15   # rad/s, used automatically for large joint moves
MOVEJ_BIG_THRESH = 1.2   # rad (L2 over joints) → "large move", slow down + warn

# Stroke moveL params — same as baseline controller / original script.
STROKE_ACCEL    = 0.5    # m/s^2

MAX_PRESS_DEPTH = 0.004
BOW_FULL_START  = 0.05
BOW_FULL_END    = 0.95

DEPTH_LEVELS = {
    'very_light': 0.0000,
    'light':      0.0008,
    'medium':     0.0015,
    'firm':       0.0025,
    'heavy':      0.0035,
}
SPEED_LEVELS = {
    'very_slow': 0.05,
    'slow':      0.07,
    'medium':    0.10,
    'fast':      0.13,
    'very_fast': 0.15,
}
BAD_CONDITIONS = [
    {'depth': 0.0040, 'speed': 0.10, 'label': 'too_much_press'},
    {'depth': 0.0000, 'speed': 0.10, 'label': 'barely_touching'},
    {'depth': 0.0015, 'speed': 0.02, 'label': 'too_slow'},
    {'depth': 0.0015, 'speed': 0.30, 'label': 'too_fast'},
]

_DIR_SUFFIX = {'mid_only': 'mid', 'all_angles': 'all'}
OUTPUT_DIR  = Path(f"dataset_multi_string_{_DIR_SUFFIX[ANGLE_MODE]}")
AUDIO_DIR   = OUTPUT_DIR / "audio"
STATE_DIR   = OUTPUT_DIR / "states"
META_FILE   = OUTPUT_DIR / "metadata.jsonl"
SESSION_LOG = OUTPUT_DIR / "sessions.jsonl"
CHECKPOINT  = OUTPUT_DIR / "progress_checkpoint.json"
CALIB_FILE  = OUTPUT_DIR / "calibration.json"

# Press direction (into the string) — only used for the tiny depth offset,
# exactly as the baseline controller does. Not used for large moves.
RETRACT_DIR_BASE = np.array([0.68861, 0.031962, 0.724428])
RETRACT_DIR_BASE = RETRACT_DIR_BASE / np.linalg.norm(RETRACT_DIR_BASE)
MAX_SAFE_REACH   = 0.89


# ══════════════════════════════════════════════════════════════════
# BOW ANGLE WAYPOINTS  (from bowing_regions.txt)
# ══════════════════════════════════════════════════════════════════

A_TOP_F    = np.array([.43613, .69256, -.29752, 1.041, 2.850, -1.655])
A_TOP_T    = np.array([.654442, .37533, .13275, 1.041, 2.850, -1.655])
A_MID_F    = np.array([.27184, .78359, .12283, 1.620, 2.492, -1.119])
A_MID_T    = np.array([.43098, .30660, .27219, 1.653, 2.510, -1.229])
A_BOTTOM_F = np.array([.35351, .79398, .10399, 1.900, 2.221, -1.728])
A_BOTTOM_T = np.array([.44659, .38004, -.07432, 1.905, 2.196, -1.791])

D_TOP_F    = np.array([.42075, .70016, .07254, 1.284, 2.923, -1.959])
D_TOP_T    = np.array([.59077, .28077, .06251, 1.284, 2.923, -1.959])
D_MID_F    = np.array([.39034, .69286, .08219, 2.000, 2.779, -1.665])
D_MID_T    = np.array([.39053, .22339, .08530, 2.000, 2.779, -1.665])
D_BOTTOM_F = np.array([.37416, .71903, .10273, 2.556, 2.153, -1.856])
D_BOTTOM_T = np.array([.24806, .27354, .21650, 2.556, 2.153, -1.856])

G_TOP_F    = np.array([.38029, .63846, .06457, 2.061, 3.128, -1.493])
G_TOP_T    = np.array([.35515, .18951, -.05136, 2.061, 3.128, -1.493])
G_MID_F    = np.array([.32869, .64247, .09041, 2.462, 2.632, -1.192])
G_MID_T    = np.array([.23545, .22704, .05865, 2.462, 2.632, -1.192])
G_BOTTOM_F = np.array([.36775, .65103, .07817, 2.827, 2.305, -1.771])
G_BOTTOM_T = np.array([.16714, .20374, .11027, 2.827, 2.305, -1.771])

C_TOP_F    = np.array([.35300, .60179, .05862, 2.559, 3.040, -1.359])
C_TOP_T    = np.array([.21635, .20883, -0.09254, 2.559, 3.040, -1.359])
C_MID_F    = np.array([.29998, .58924, .07835, 2.873, 2.569, -1.046])
C_MID_T    = np.array([.15512, .29383, .01174, 2.873, 2.569, -1.046])
C_BOTTOM_F = np.array([.31978, .63974, .07771, 3.123, 2.168, -1.131])
C_BOTTOM_T = np.array([.05780, .27254, .06287, 3.123, 2.168, -1.131])

# Joint configs (radians) for frog and tip of each angle (Polyscope degrees).
A_TOP_Q_F    = np.radians([-133.55, -5.82, 20.44, -44.92, -94.26, -180.87])
A_TOP_Q_T    = np.radians([-160.49, -21.25, 49.16, -62.38, -117.08, -195.75])
A_MID_Q_F    = np.radians([-123.73, -15.96, 36.24, -68.95, -88.42, -148.13])
A_MID_Q_T    = np.radians([-162.55, -61.12, 100.79, -89.21, -111.21, -177.77])
A_BOTTOM_Q_F = np.radians([-127.93, -5.06, 15.80, -39.01, -75.06, -145.71])
A_BOTTOM_Q_T = np.radians([-158.76, -60.43, 88.26, -53.40, -101.68, -159.46])

D_TOP_Q_F    = np.radians([-132.09, -4.54, 22.76, -37.51, -78.11, -183.02])
D_TOP_Q_T    = np.radians([-166.68, -26.90, 75.03, -68.39, -110.66, -194.56])
D_MID_Q_F    = np.radians([-132.28, -15.66, 45.61, -60.20, -61.64, -167.18])
D_MID_Q_T    = np.radians([-170.10, -44.50, 116.48, -98.47, -95.11, -185.17])
D_BOTTOM_Q_F = np.radians([-131.17, -21.64, 53.90, -60.61, -48.57, -146.67])
D_BOTTOM_Q_T = np.radians([-162.99, -74.77, 136.07, -82.73, -77.57, -161.54])

G_TOP_Q_F    = np.radians([-133.09, -19.37, 57.96, -69.60, -51.61, -178.44])
G_TOP_Q_T    = np.radians([-171.19, -24.74, 115.53, -114.74, -85.68, -196.86])
G_MID_Q_F    = np.radians([-132.62, -28.14, 74.65, -95.90, -54.17, -149.87])
G_MID_Q_T    = np.radians([-167.00, -42.39, 138.49, -134.95, -79.33, -175.59])
G_BOTTOM_Q_F = np.radians([-133.08, -25.58, 69.35, -73.51, -37.49, -153.20])
G_BOTTOM_Q_T = np.radians([-170.44, -52.39, 153.44, -119.67, -71.90, -171.51])

C_TOP_Q_F    = np.radians([-133.49, -24.56, 72.54, -81.58, -36.99, -173.02])
C_TOP_Q_T    = np.radians([-164.21, -14.40, 126.87, -134.28, -64.49, -191.00])
C_MID_Q_F    = np.radians([-133.39, -31.41, 90.04, -115.59, -40.42, -142.33])
C_MID_Q_T    = np.radians([-150.52, -29.11, 138.47, -153.55, -51.51, -160.52])
C_BOTTOM_Q_F = np.radians([-132.16, -28.64, 84.18, -117.81, -37.66, -125.26])
C_BOTTOM_Q_T = np.radians([-145.42, -26.06, 155.24, -179.14, -45.32, -141.62])

STRING_ANGLES = {
    'A': [
        {'name': 'top',    'frog': A_TOP_F,    'tip': A_TOP_T,    'q_frog': A_TOP_Q_F,    'q_tip': A_TOP_Q_T},
        {'name': 'mid',    'frog': A_MID_F,    'tip': A_MID_T,    'q_frog': A_MID_Q_F,    'q_tip': A_MID_Q_T},
        {'name': 'bottom', 'frog': A_BOTTOM_F, 'tip': A_BOTTOM_T, 'q_frog': A_BOTTOM_Q_F, 'q_tip': A_BOTTOM_Q_T},
    ],
    'D': [
        {'name': 'top',    'frog': D_TOP_F,    'tip': D_TOP_T,    'q_frog': D_TOP_Q_F,    'q_tip': D_TOP_Q_T},
        {'name': 'mid',    'frog': D_MID_F,    'tip': D_MID_T,    'q_frog': D_MID_Q_F,    'q_tip': D_MID_Q_T},
        {'name': 'bottom', 'frog': D_BOTTOM_F, 'tip': D_BOTTOM_T, 'q_frog': D_BOTTOM_Q_F, 'q_tip': D_BOTTOM_Q_T},
    ],
    'G': [
        {'name': 'top',    'frog': G_TOP_F,    'tip': G_TOP_T,    'q_frog': G_TOP_Q_F,    'q_tip': G_TOP_Q_T},
        {'name': 'mid',    'frog': G_MID_F,    'tip': G_MID_T,    'q_frog': G_MID_Q_F,    'q_tip': G_MID_Q_T},
        {'name': 'bottom', 'frog': G_BOTTOM_F, 'tip': G_BOTTOM_T, 'q_frog': G_BOTTOM_Q_F, 'q_tip': G_BOTTOM_Q_T},
    ],
    'C': [
        {'name': 'top',    'frog': C_TOP_F,    'tip': C_TOP_T,    'q_frog': C_TOP_Q_F,    'q_tip': C_TOP_Q_T},
        {'name': 'mid',    'frog': C_MID_F,    'tip': C_MID_T,    'q_frog': C_MID_Q_F,    'q_tip': C_MID_Q_T},
        {'name': 'bottom', 'frog': C_BOTTOM_F, 'tip': C_BOTTOM_T, 'q_frog': C_BOTTOM_Q_F, 'q_tip': C_BOTTOM_Q_T},
    ],
}

_ANGLE_FILTER = {
    'mid_only':   {'mid'},
    'all_angles': {'top', 'mid', 'bottom'},
}


# ══════════════════════════════════════════════════════════════════
# GEOMETRY
# ══════════════════════════════════════════════════════════════════

def clamp_reach(pose, max_reach=MAX_SAFE_REACH):
    out = np.array(pose, dtype=float).copy()
    r   = np.linalg.norm(out[:3])
    if r > max_reach:
        out[:3] = out[:3] * (max_reach / r)
    return out

def apply_press(pose, depth):
    """Offset pose into the string by depth m (along -RETRACT_DIR_BASE)."""
    depth = float(np.clip(depth, 0.0, MAX_PRESS_DEPTH))
    out = np.array(pose, dtype=float).copy()
    out[:3] += -depth * RETRACT_DIR_BASE
    return clamp_reach(out)


# ══════════════════════════════════════════════════════════════════
# MOTION PRIMITIVES
# ══════════════════════════════════════════════════════════════════

def wait_until_stopped(ctrl, timeout=10.0):
    """Block until joint velocities are ~0 (guards against async returns)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if float(np.linalg.norm(ctrl.rtde_r.getActualQd())) < 0.01:
                return
        except Exception:
            return
        time.sleep(0.05)

def movej(ctrl, q, label=""):
    """
    moveJ to a joint target, blocking. Auto-slows for large moves and warns.
    Used for all repositioning (reset, descend, lift) between known configs.
    """
    q   = np.asarray(q, dtype=float)
    cur = np.array(ctrl.rtde_r.getActualQ())
    dist = float(np.linalg.norm(q - cur))
    speed = MOVEJ_SPEED
    if dist > MOVEJ_BIG_THRESH:
        speed = MOVEJ_SPEED_BIG
        print(f"  [movej {label}] large move {np.degrees(dist):.0f}° — going slow, watch robot")
    ctrl.rtde_c.moveJ(q.tolist(), speed, MOVEJ_ACCEL)
    wait_until_stopped(ctrl)

def movel(ctrl, pose, speed, accel):
    """moveL to a Cartesian pose, blocking."""
    ctrl.rtde_c.moveL(clamp_reach(pose).tolist(), speed, accel)
    wait_until_stopped(ctrl)


# ══════════════════════════════════════════════════════════════════
# FREE DRIVE + CALIBRATION
# ══════════════════════════════════════════════════════════════════

def freedrive_capture(ctrl, prompt):
    """Enter free drive, wait for ENTER, capture pose+joints, exit free drive."""
    print(f"\n  {prompt}")
    print( "  FREE DRIVE ON — push the robot by hand. Press ENTER when positioned.")
    ctrl.rtde_c.freedriveMode()
    input("  [free drive active] ")
    ctrl.rtde_c.endFreedriveMode()
    time.sleep(0.5)
    pose = list(ctrl.rtde_r.getActualTCPPose())
    q    = list(ctrl.rtde_r.getActualQ())
    print(f"  Captured: TCP=[{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}]")
    return pose, q

def calibrate_angle(ctrl, string, angle):
    """
    Interactive calibration for one angle. Returns the calibration dict.
    Captures frog_hover and tip_hover; descends to frog via moveJ; bows to tip.
    """
    key = f"{string}_{angle['name']}"
    print(f"\n{'═'*60}")
    print(f"  CALIBRATION — {key}")
    print(f"{'═'*60}")

    # 1. Hover above frog
    frog_hover_pose, frog_hover_q = freedrive_capture(
        ctrl, f"Hover the bow a SAFE distance ABOVE the {key} FROG area.")

    # Sanity: how far is the hover from the frog joint target?
    dist = float(np.linalg.norm(np.array(frog_hover_q) - angle['q_frog']))
    print(f"  Hover is {np.degrees(dist):.0f}° (joint L2) from frog target.")
    if dist > MOVEJ_BIG_THRESH:
        print("  NOTE: that's a large move. Make sure the hover is directly above")
        print("  the frog with the bow oriented as it will sit on the string.")
        if input("  Re-do this hover? (y/N): ").strip().lower() == 'y':
            return calibrate_angle(ctrl, string, angle)

    # 2. Descend to exact frog joint target (small, slow moveJ)
    print("  Descending to frog (moveJ to q_frog)...")
    movej(ctrl, angle['q_frog'], label=f"{key} frog")

    # 3. One full bow frog -> tip (moveL) to confirm the stroke line
    print("  Bowing frog -> tip (moveL)...")
    movel(ctrl, angle['tip'], 0.08, STROKE_ACCEL)

    # 4. Hover above tip
    tip_hover_pose, tip_hover_q = freedrive_capture(
        ctrl, f"Now hover the bow a SAFE distance ABOVE the {key} TIP area.")

    print(f"  {key} calibration complete.\n")
    return {
        'frog_hover_pose': frog_hover_pose, 'frog_hover_q': frog_hover_q,
        'tip_hover_pose':  tip_hover_pose,  'tip_hover_q':  tip_hover_q,
    }


def load_calibration():
    if CALIB_FILE.exists():
        try:
            return json.loads(CALIB_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_calibration(calib):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CALIB_FILE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(calib, indent=2))
    tmp.replace(CALIB_FILE)


# ══════════════════════════════════════════════════════════════════
# RESET + STROKE  (uses captured hovers + target joints)
# ══════════════════════════════════════════════════════════════════

def stroke_poses(angle, bow_dir):
    """Return (start_pose, end_pose, start_q, end_q, start_hover_q, end_hover_q,
    cal) given bow direction. Frog=start for down-bow, tip=start for up-bow."""
    # Filled by caller with calibration; this only resolves frog/tip ordering.
    if bow_dir == 'down':
        return ('frog', 'tip')
    else:
        return ('tip', 'frog')

def reset_and_stroke(ctrl, angle, cal, depth, speed, bow_dir,
                     audio_record_fn):
    """
    Full single-stroke motion using the calibrated reset map:
      reset : moveJ to start-hover  → moveJ down to start joints
      place : small moveL applying press depth
      stroke: single moveL along the string  (audio recorded here)
      lift  : moveJ up to end-hover
    Returns the StateLogger summary for the stroke.
    """
    start_name, end_name = stroke_poses(angle, bow_dir)

    start_pose   = angle['frog']   if start_name == 'frog' else angle['tip']
    end_pose     = angle['tip']    if end_name   == 'tip'  else angle['frog']
    start_q      = angle['q_frog'] if start_name == 'frog' else angle['q_tip']
    start_hover  = np.array(cal['frog_hover_q'] if start_name == 'frog'
                            else cal['tip_hover_q'])
    end_hover    = np.array(cal['tip_hover_q']  if end_name == 'tip'
                            else cal['frog_hover_q'])

    # RESET: travel along the upper line to the start hover, then descend.
    movej(ctrl, start_hover, label="to start-hover")
    movej(ctrl, start_q,     label="descend to start")

    # PLACE: apply press depth (tiny moveL into the string).
    pressed_start = apply_press(start_pose, depth)
    movel(ctrl, pressed_start, 0.05, STROKE_ACCEL)

    # STROKE: single moveL along the string at commanded speed.
    pressed_end = apply_press(end_pose, depth)
    summary = audio_record_fn(lambda: _do_stroke(ctrl, pressed_end, speed))

    # LIFT: moveJ up to the end hover (off the string).
    movej(ctrl, end_hover, label="lift to end-hover")
    return summary

def _do_stroke(ctrl, pressed_end, speed):
    """The recorded motion: one moveL along the string."""
    ctrl.state_logger.start()
    time.sleep(MARK_SETTLE_SEC)
    t0 = time.time()
    ctrl.rtde_c.moveL(clamp_reach(pressed_end).tolist(), speed, STROKE_ACCEL)
    t1 = time.time()
    wait_until_stopped(ctrl)
    ctrl.state_logger.stop()
    return ctrl.state_logger.get_summary(t_start=t0, t_end=t1)


# ══════════════════════════════════════════════════════════════════
# AUDIO
# ══════════════════════════════════════════════════════════════════

def find_focusrite_device():
    for i, dev in enumerate(sd.query_devices()):
        if ('Scarlett' in dev['name'] or 'Focusrite' in dev['name']) \
                and dev['max_input_channels'] >= 1:
            print(f"Found Focusrite at index {i}: {dev['name']}")
            return i
    raise RuntimeError("Focusrite not found.")

def validate_audio_chain(device):
    print("Validating audio chain (3s)...")
    audio = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype='float32', device=device)
    sd.wait()
    peak = float(np.max(np.abs(audio.flatten())))
    print(f"  Peak: {peak:.4f} ({20*np.log10(peak+1e-10):.1f} dBFS) — "
          + ("OK" if 1e-4 < peak < 0.1 else "CHECK LEVELS"))
    return peak


# ══════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════

def load_checkpoint():
    if CHECKPOINT.exists():
        try:
            return json.loads(CHECKPOINT.read_text())
        except Exception:
            return None
    return None

def save_checkpoint(state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(CHECKPOINT)


# ══════════════════════════════════════════════════════════════════
# CONDITIONS
# ══════════════════════════════════════════════════════════════════

def build_conditions():
    include    = _ANGLE_FILTER[ANGLE_MODE]
    conditions = []
    counter    = 0
    for string in STRING_ORDER:
        for angle in STRING_ANGLES[string]:
            if angle['name'] not in include:
                continue
            for d_name, d_val in DEPTH_LEVELS.items():
                for s_name, s_val in SPEED_LEVELS.items():
                    bow_dir = 'down' if counter % 2 == 0 else 'up'
                    counter += 1
                    conditions.append({
                        'cond_id': f"{string}_{angle['name']}_D{d_name}_S{s_name}",
                        'string':  string, 'angle': angle,
                        'depth':   d_val, 'speed': s_val,
                        'bow_dir': bow_dir, 'type': 'systematic',
                    })
    for string in STRING_ORDER:
        mid = next((a for a in STRING_ANGLES[string] if a['name'] == 'mid'), None)
        if mid is None:
            continue
        for bad in BAD_CONDITIONS:
            conditions.append({
                'cond_id': f"{string}_bad_{bad['label']}",
                'string':  string, 'angle': mid,
                'depth':   bad['depth'], 'speed': bad['speed'],
                'bow_dir': 'down', 'type': 'bad',
            })
    return conditions


# ══════════════════════════════════════════════════════════════════
# RECORD ONE CONDITION
# ══════════════════════════════════════════════════════════════════

def record_condition(ctrl, cond, cal, session_id, device):
    angle  = cond['angle']
    string = cond['string']
    ctrl.state_logger.set_bow_reference(angle['frog'], angle['tip'])

    stroke_id = f"{session_id}_{cond['cond_id']}"
    metas     = []

    for repeat in range(N_REPEATS):
        # Duration for the audio buffer from physical bow length / speed.
        bow_len    = float(np.linalg.norm(angle['tip'][:3] - angle['frog'][:3])) \
                     * abs(BOW_FULL_END - BOW_FULL_START)
        stroke_dur = bow_len / max(cond['speed'], 1e-6)
        total_samples = int((PRE_BUFFER + stroke_dur + POST_BUFFER) * SAMPLE_RATE)

        # Recording wrapper: starts audio, runs the stroke fn, returns summary.
        captured = {}
        def audio_record_fn(stroke_fn):
            buf = sd.rec(total_samples, samplerate=SAMPLE_RATE,
                         channels=CHANNELS, dtype='float32', device=device)
            time.sleep(PRE_BUFFER)
            summary = stroke_fn()
            time.sleep(POST_BUFFER)
            sd.wait()
            captured['buf'] = buf
            return summary

        summary = reset_and_stroke(ctrl, angle, cal, cond['depth'],
                                   cond['speed'], cond['bow_dir'], audio_record_fn)
        audio_buf = captured['buf']

        fname_audio = f"{stroke_id}_r{repeat:02d}.wav"
        fname_state = f"{stroke_id}_r{repeat:02d}_state.npy"
        sf.write(str(AUDIO_DIR / fname_audio), audio_buf, SAMPLE_RATE)
        ctrl.state_logger.save(str(STATE_DIR / fname_state))

        peak = float(np.max(np.abs(audio_buf)))
        if peak > 0.95:
            print(f"  CLIPPING: peak={peak:.3f}")

        metas.append({
            'stroke_id': stroke_id, 'cond_id': cond['cond_id'], 'repeat': repeat,
            'session_id': session_id, 'timestamp': datetime.now().isoformat(),
            'experiment': 'multi_string_tilt_yaw',
            'audio_file': fname_audio, 'state_file': fname_state, 'string': string,
            'commanded': {
                'depth_m': float(cond['depth']), 'speed': float(cond['speed']),
                'angle_name': angle['name'],
                'frog_pose': angle['frog'].tolist(), 'tip_pose': angle['tip'].tolist(),
                'bow_start': BOW_FULL_START, 'bow_end': BOW_FULL_END,
                'bow_dir': cond['bow_dir'],
            },
            'measured': summary, 'audio_peak': peak,
            'condition_label': cond['cond_id'], 'condition_type': cond['type'],
        })

        print(f"  {fname_audio} | {string} {angle['name']} {cond['bow_dir']} | "
              f"v={cond['speed']:.2f} d={cond['depth']*1000:.1f}mm peak={peak:.3f}")
        time.sleep(0.3)

    return metas


# ══════════════════════════════════════════════════════════════════
# MAIN COLLECTION
# ══════════════════════════════════════════════════════════════════

def run_collection(ctrl, device):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    all_conds = build_conditions()
    calib     = load_calibration()

    ckpt = load_checkpoint()
    if ckpt and ckpt.get('completed') is not None:
        session_id = ckpt['session_id']
        completed  = set(ckpt['completed'])
        print(f"\nResuming {session_id}: {len(completed)}/{len(all_conds)} done.")
    else:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        completed  = set()
        ckpt = {'session_id': session_id, 'angle_mode': ANGLE_MODE,
                'completed': [], 'started': datetime.now().isoformat(),
                'last_updated': ''}
        save_checkpoint(ckpt)
        print(f"\nNew session {session_id}: {len(all_conds)} conditions.")

    remaining = [c for c in all_conds if c['cond_id'] not in completed]
    print(f"{len(remaining)} remaining. Ctrl+C saves and stops.\n")

    current_key   = None
    done_this_run = 0

    try:
        for idx, cond in enumerate(remaining):
            angle = cond['angle']
            key   = f"{cond['string']}_{angle['name']}"

            # New angle → ensure calibrated (interactive if not in file)
            if key != current_key:
                if key not in calib:
                    calib[key] = calibrate_angle(ctrl, cond['string'], angle)
                    save_calibration(calib)
                else:
                    print(f"\n  Using saved calibration for {key}. "
                          f"Moving to its frog-hover...")
                    movej(ctrl, np.array(calib[key]['frog_hover_q']),
                          label=f"{key} frog-hover")
                current_key = key

            print(f"\n[{len(completed)+1}/{len(all_conds)}] {cond['cond_id']}")
            metas = record_condition(ctrl, cond, calib[key], session_id, device)

            with open(META_FILE, 'a') as f:
                for m in metas:
                    f.write(json.dumps(m) + '\n')
            completed.add(cond['cond_id'])
            ckpt['completed']    = sorted(completed)
            ckpt['last_updated'] = datetime.now().isoformat()
            save_checkpoint(ckpt)
            done_this_run += 1

            if idx < len(remaining) - 1:
                rest = 30 if idx % 5 == 4 else 5
                print(f"  (resting {rest}s)")
                time.sleep(rest)

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving progress (robot left where it is).")
        ckpt['completed']    = sorted(completed)
        ckpt['last_updated'] = datetime.now().isoformat()
        save_checkpoint(ckpt)
        print(f"Saved: {len(completed)}/{len(all_conds)} done ({done_this_run} this run).")
        print("Re-run to resume.")
        return

    session_log = {
        'session_id': session_id, 'angle_mode': ANGLE_MODE,
        'date': datetime.now().isoformat(),
        'n_conditions': len(all_conds), 'n_recordings': len(all_conds) * N_REPEATS,
        'notes': input("\nSession notes (ENTER to skip): ").strip(),
    }
    with open(SESSION_LOG, 'a') as f:
        f.write(json.dumps(session_log) + '\n')
    try:
        CHECKPOINT.replace(CHECKPOINT.with_suffix('.json.done'))
    except Exception:
        pass
    print(f"\nComplete: {len(all_conds)*N_REPEATS} recordings in {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\nRobot Cello: Multi-String Data Collection (calibrated reset)")
    print(f"Strings: {STRING_ORDER}  Mode: {ANGLE_MODE}")
    if TESTING:
        print("*** TESTING MODE ***")
    print("=" * 50)

    audio_device = find_focusrite_device()
    ctrl         = TargetStringBaselineController([])

    try:
        ctrl.reset()
        validate_audio_chain(audio_device)
        input("\nPress ENTER to begin / resume collection...")

        if TESTING:
            AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            calib  = load_calibration()
            string = STRING_ORDER[0]
            angle  = next(a for a in STRING_ANGLES[string]
                          if a['name'] in _ANGLE_FILTER[ANGLE_MODE])
            key = f"{string}_{angle['name']}"
            if key not in calib:
                calib[key] = calibrate_angle(ctrl, string, angle)
                save_calibration(calib)
            cond = {'cond_id': f"TEST_{key}", 'string': string, 'angle': angle,
                    'depth': 0.0015, 'speed': 0.10, 'bow_dir': 'down', 'type': 'test'}
            session_id = "TEST_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            metas = record_condition(ctrl, cond, calib[key], session_id, audio_device)
            with open(META_FILE, 'a') as f:
                for m in metas: f.write(json.dumps(m) + '\n')
            print("Test complete.")
        else:
            run_collection(ctrl, audio_device)

    finally:
        try:
            ctrl.rtde_c.forceModeStop()
        except Exception:
            pass
        ctrl.close()