"""
recording_script_a_configs.py

A-string data collection across the FIVE bowing configurations from the updated
stringa_closer.script. Each config now carries its OWN local safe/reset points
(fsafe near frog, tsafe near tip) instead of two shared generic reset points —
this keeps every reset move small (37-84mm) and close to where the bow already
is, instead of one big jump to a distant shared point (the ~157mm move that
caused the earlier fault). Sweeps signed depth offset and speed only.

DESIGN PRINCIPLE: every motion is a moveL straight to one of YOUR defined
points (frog, tip, fsafe, tsafe per config) — the same pattern
stringa_closer.script uses on the pendant (movel(point, a=1.2, v=0.25)). No
computed retract directions, no derived clearance offsets, no shared/generic
reset points. If it isn't a point you defined for THIS config, the robot
doesn't go there. Signed depth offset and speed are the only things varied.

Per stroke (config X, down bow):
    1. moveL(X_fsafe)                       lift, clear, near frog
    2. moveL(X_frog + signed depth offset)  onto the string at the start
    3. moveL(X_tip + signed depth, speed=v) THE STROKE
    4. moveL(X_tsafe)                       lift off, near tip
(up bow reverses: start at tip, end at frog, fsafe/tsafe swap roles accordingly)

Installation (TCP/payload/gravity) is set explicitly to match
stringa_closer.script's own set_tcp/set_target_payload/set_gravity lines, so
every point here is interpreted in the same frame it was verified in.

RESUMABLE: stop any time, re-run, it continues from the last completed
(config, depth, speed, dir, repeat).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sounddevice as sd
import soundfile as sf
import json
import time
import random
from datetime import datetime

from BaselineControls.baseline_controller import (
    TargetStringBaselineController,
    velocity_to_dynamic,
)


RECORDING_STRING = 'A'


class RobotFaultStop(RuntimeError):
    """Raised immediately if a moveL fails or the control script stops running.
    Never caught silently. Stops the script rather than continuing as if a
    failed move succeeded."""
    pass


def safe_moveL(ctrl, pose, speed, accel, what=""):
    """The ONLY way this script moves the robot. Goes straight to `pose` — no
    modification, no computed offsets beyond what the caller already applied.
    Stops the script immediately on any failure."""
    try:
        ok = ctrl.rtde_c.moveL(list(pose), speed, accel)
    except Exception as e:
        raise RobotFaultStop(f"moveL raised during '{what}': {e}. STOPPING.") from e
    if ok is False:
        raise RobotFaultStop(f"moveL returned False during '{what}'. STOPPING.")
    try:
        running = ctrl.rtde_c.isProgramRunning()
    except Exception:
        running = None
    if running is False:
        raise RobotFaultStop(
            f"RTDE control script not running after '{what}'. STOPPING. "
            f"Check the robot by eye and on the teach pendant."
        )


# ══════════════════════════════════════════════════════════════════
# YOUR DEFINED POINTS — exactly as in the updated stringa_closer.script.
# Each config has its OWN frog/tip/fsafe/tsafe — no shared reset points.
# ══════════════════════════════════════════════════════════════════
BOW_CONFIGS = {
    'standard': {
        'frog':  np.array([.328964663774, .703229462539, -.009258082371,
                           1.501488908519, 2.294264795278, -1.521353569816]),
        'tip':   np.array([.557923990118, .279876247922, .200314083410,
                           -1.446439182207, -2.240158754246, 1.575901583680]),
        'fsafe': np.array([.328964663671, .703229462391, .045333550563,
                           1.501488908520, 2.294264795279, -1.521353569816]),
        'tsafe': np.array([.557923989897, .279876247854, .255567513319,
                           -1.446439179803, -2.240158755503, 1.575901584151]),
    },
    'bridge': {
        'frog':  np.array([.359288490162, .697109874572, -.037989803078,
                           -1.454279199729, -2.266388048891, 1.536760224005]),
        'tip':   np.array([.586461738590, .282502051645, .164701776524,
                           -1.446439182239, -2.240158754244, 1.575901583647]),
        'fsafe': np.array([.359288490131, .697109874486, -.001067425999,
                           -1.454279200049, -2.266388049537, 1.536760222734]),
        'tsafe': np.array([.586461738476, .282502051555, .217703154389,
                           -1.446439181781, -2.240158754326, 1.575901583927]),
    },
    'board': {
        'frog':  np.array([.340805201884, .662776270539, .002576205786,
                           -1.352622850133, -2.125738220444, 1.489970485961]),
        'tip':   np.array([.544882776920, .265820111445, .144044784492,
                           -1.351471826274, -2.124442034566, 1.490717340742]),
        'fsafe': np.array([.302077525730, .700121956050, .049438367194,
                           1.578563955137, 2.373499757692, -1.317137839599]),
        'tsafe': np.array([.556820561715, .261105329878, .296519711916,
                           -1.474441171643, -2.232009306923, 1.513428520367]),
    },
    'topangle': {
        'frog':  np.array([.323342050286, .675895581573, .007426293674,
                           -1.681562550000, -1.855899529465, 1.628173418626]),
        'tip':   np.array([.437751370491, .285154715032, .237082469560,
                           -1.681562550049, -1.855899529511, 1.628173418552]),
        'fsafe': np.array([.313606910359, .699665763872, .045524839604,
                           -1.688715651329, -1.887063157556, 1.664053021292]),
        'tsafe': np.array([.452893533959, .279140634505, .294094096621,
                           -1.674748829176, -1.873859951296, 1.678075624897]),
    },
    'botangle': {
        'frog':  np.array([.354749185881, .708893152215, -.050831555492,
                           -1.033198855435, -2.489950727989, 1.270004512054]),
        'tip':   np.array([.669656684572, .302367815533, .026684655516,
                           -1.039222465315, -2.496995506148, 1.264917666173]),
        'fsafe': np.array([.375991499569, .708893151877, .008153071661,
                           -1.033198855367, -2.489950728075, 1.270004512018]),
        'tsafe': np.array([.683581126997, .302367815453, .089648590214,
                           -1.039222444665, -2.496995494610, 1.264917705729]),
    },
}


# ══════════════════════════════════════════════════════════════════
# DEPTH (the dynamic control)
# ══════════════════════════════════════════════════════════════════
# Depth is a signed offset from the taught frog/tip waypoints. The taught
# waypoint is the pendant-good baseline: negative moves OUT toward fsafe for
# lighter contact, positive moves IN away from fsafe for heavier contact.
# Built only from points already in stringa_closer.script, per config — not a
# generic/shared direction.
MAX_OUTWARD_DEPTH = 0.0015   # 1.5mm outward/light safety ceiling
MAX_INWARD_DEPTH  = 0.0020   # 2.0mm inward/heavy safety ceiling


def press_direction(cfg):
    """Unit vector from this config's frog toward its OWN fsafe (the lift
    direction already defined for this config)."""
    d = cfg['fsafe'][:3] - cfg['frog'][:3]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.zeros(3)


def apply_depth(pose, cfg, depth):
    """Offset `pose` by signed `depth` meters from the taught baseline.

    depth < 0 moves outward toward fsafe; depth > 0 presses inward.
    """
    depth = float(np.clip(depth, -MAX_OUTWARD_DEPTH, MAX_INWARD_DEPTH))
    out = np.array(pose, dtype=float).copy()
    out[:3] = out[:3] - depth * press_direction(cfg)
    return out


# ══════════════════════════════════════════════════════════════════
# Installation — EXACTLY what stringa_closer.script sets
# ══════════════════════════════════════════════════════════════════
VERIFIED_TCP         = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VERIFIED_PAYLOAD_KG  = 0.100000
VERIFIED_PAYLOAD_COG = [0.000000, 0.000000, 0.000000]
VERIFIED_GRAVITY     = [0.0, 0.0, 9.82]


def apply_verified_installation(ctrl):
    """
    Set TCP and payload to EXACTLY what stringa_closer.script uses. Call this
    BEFORE any motion. Every point in BOW_CONFIGS was hand-verified on the
    pendant under these values; using anything else (e.g. baseline_controller's
    own TCP_OFFSET/payload) reinterprets the same numbers as a different
    physical pose.
    """
    ctrl.rtde_c.setTcp(VERIFIED_TCP)
    ctrl.rtde_c.setPayload(VERIFIED_PAYLOAD_KG, VERIFIED_PAYLOAD_COG)
    time.sleep(0.2)
    print("Installation set to match stringa_closer.script:")
    print(f"  TCP:     {VERIFIED_TCP}")
    print(f"  Payload: {VERIFIED_PAYLOAD_KG} kg, CoG {VERIFIED_PAYLOAD_COG}")
    try:
        tcp_now = ctrl.rtde_r.getActualTCPPose()
        m_now   = ctrl.rtde_r.getPayload()
        print(f"  readback: payload={m_now:.3f} kg  current TCP pose={np.round(tcp_now,4).tolist()}")
    except Exception as e:
        print(f"  (readback unavailable: {e})")


# ══════════════════════════════════════════════════════════════════
# TESTING MODE
# ══════════════════════════════════════════════════════════════════
TESTING = False
TEST_CONFIG  = 'standard'
TEST_DEPTH   = 0.0
TEST_SPEED   = 0.25
TEST_BOW_DIR = 'down'
TEST_REPEATS = 2


# ── Audio config ──────────────────────────────────────────────────
AUDIO_DEVICE  = None
SAMPLE_RATE   = 44100
PRE_BUFFER    = 0.2
POST_BUFFER   = 0.3
N_REPEATS     = 5
CHANNELS      = 1

# Which PHYSICAL input of the interface to record, 1-based. 0 means "find the
# one that actually has signal" at startup.
#
# This is not the same thing as CHANNELS. Asking for one channel records
# input 1 whatever happens to be plugged into it, and an interface will return
# a perfectly valid stream of its own noise floor from an empty socket — no
# error, right file length, silent contents. A whole session was recorded that
# way once, at -90 dBFS, because the mic had been moved to input 2 while the
# code still took input 1. The channel is verified before anything is written.
AUDIO_CHANNEL = 0

# Below this a channel is judged to be carrying nothing. A live input picks up
# room tone far above it; an empty socket sits near -90 dBFS.
AUDIO_LIVE_DBFS = -75.0
MOVE_ACCEL    = 1.2     # matches a=1.2 in stringa_closer.script
RESET_SPEED   = 0.25    # matches v=0.25 in stringa_closer.script
PENDANT_REFERENCE_DEPTH = 0.0
PENDANT_REFERENCE_SPEED = 0.25

# Audio buffer is sized for PRE_BUFFER + (kinematic stroke estimate) +
# POST_BUFFER, but moveL's real execution time can still overrun that
# estimate (network/RTDE jitter, etc). Record this many extra seconds as a
# safety net, then trim back to the actual measured duration afterward.
AUDIO_SAFETY_MARGIN = 5.0

# Spacing, in seconds, between the state-timeline points written into each
# metadata.jsonl record (in addition to the full-rate log saved per-stroke
# via state_logger.save()).
STATE_SAMPLE_INTERVAL = 0.1


# ── Parameter grids (depth + speed only) ──────────────────────────
DEPTH_LEVELS = {
    'very_light': -0.0020,
    'light':      -0.0010,
    'medium':      0.0000,
    'firm':        0.0008,
}
SPEED_LEVELS = {
    'very_slow':   0.09,
    'slow':        0.12,
    'medium':      0.15,
    'fast':        0.20,
    'pendant_ref': PENDANT_REFERENCE_SPEED,
}
BAD_CONDITIONS = []


# ── Output / resume ───────────────────────────────────────────────
OUTPUT_DIR  = Path("dataset_a_final")
AUDIO_DIR   = OUTPUT_DIR / "audio"
STATE_DIR   = OUTPUT_DIR / "states"
META_FILE   = OUTPUT_DIR / "metadata.jsonl"
SESSION_LOG = OUTPUT_DIR / "sessions.jsonl"
_SESSION_MARKER = OUTPUT_DIR / ".session_id"
STROKE_RECORD_TYPE = "stroke"
STATE_SAMPLE_RECORD_TYPE = "state_sample"


def get_session_id():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if _SESSION_MARKER.exists():
        return _SESSION_MARKER.read_text().strip()
    sid = datetime.now().strftime("%Y%m%d_%H%M%S")
    _SESSION_MARKER.write_text(sid)
    return sid


def load_completed_keys():
    done = set()
    if not META_FILE.exists():
        return done
    with open(META_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get('record_type', STROKE_RECORD_TYPE) != STROKE_RECORD_TYPE:
                    continue
                label = r.get('condition_label')
                if label:
                    done.add((label, int(r['repeat'])))
            except Exception:
                continue
    return done


def condition_key(condition_label, repeat):
    return (condition_label, int(repeat))


# ══════════════════════════════════════════════════════════════════
# F/T zeroing — at THIS config's own fsafe point, no shared/generic point
# ══════════════════════════════════════════════════════════════════

def zero_ft_at_config_safe(ctrl, config_name):
    """Zero the F/T sensor sitting at this config's own fsafe (clear of the
    string, local to this config — not a shared generic point)."""
    cfg = BOW_CONFIGS[config_name]
    print(f"\nZeroing F/T sensor at {config_name}_fsafe...")
    safe_moveL(ctrl, cfg['fsafe'], RESET_SPEED, MOVE_ACCEL,
              what=f"FT zero: {config_name}_fsafe")
    time.sleep(0.5)
    ok = ctrl.rtde_c.zeroFtSensor()
    time.sleep(0.3)
    f = float(np.linalg.norm(ctrl.rtde_r.getActualTCPForce()[:3]))
    print(f"  zeroFtSensor() returned {ok}; force now {f:.3f} N (≈0 expected).")


# ══════════════════════════════════════════════════════════════════
# Stroke: ONLY moves between this config's own points
# ══════════════════════════════════════════════════════════════════

def stroke_kinematic_duration(distance, speed, accel):
    """
    Real moveL duration for a straight-line move of `distance` ramping up to
    `speed` under `accel` (trapezoidal profile), not the naive distance/speed
    figure. distance/speed ignores the accel/decel ramps entirely, which
    under-estimated the real stroke time by speed/accel seconds — the audio
    buffer sized off that estimate then auto-stopped before the robot (and
    the post-buffer) actually finished, cutting off the tail of every note.
    """
    if accel <= 0 or speed <= 0:
        return distance / speed
    ramp_distance = speed ** 2 / accel
    if ramp_distance >= distance:
        # Triangular profile: too short to ever reach commanded speed.
        return 2.0 * np.sqrt(distance / accel)
    return speed / accel + distance / speed


def make_stroke_note(config_name, depth, speed, bow_dir):
    cfg = BOW_CONFIGS[config_name]
    distance = float(np.linalg.norm(cfg['tip'][:3] - cfg['frog'][:3]))
    duration = stroke_kinematic_duration(distance, speed, MOVE_ACCEL)
    depth = float(np.clip(depth, -MAX_OUTWARD_DEPTH, MAX_INWARD_DEPTH))
    depth_span = MAX_OUTWARD_DEPTH + MAX_INWARD_DEPTH
    frac = (depth + MAX_OUTWARD_DEPTH) / depth_span if depth_span > 0 else 0.0
    velocity = int(np.clip(1 + frac * 126, 1, 127))
    return {
        'config': config_name, 'duration': float(duration), 'velocity': velocity,
        'dynamic': velocity_to_dynamic(velocity), 'bow_dir': bow_dir,
        'commanded_depth': depth, 'commanded_speed': float(speed),
    }


def stroke_targets(note):
    """
    Resolve this note into its start/end/reset poses.

    'down' (frog->tip):  fsafe -> frog(+depth) -> tip(+depth, speed=v) -> tsafe
    'up'   (tip->frog):  tsafe -> tip(+depth)   -> frog(+depth, speed=v) -> fsafe
    """
    cfg = BOW_CONFIGS[note['config']]
    if note['bow_dir'] == 'up':
        start_raw, end_raw   = cfg['tip'], cfg['frog']
        reset_before, reset_after = cfg['tsafe'], cfg['fsafe']
    else:
        start_raw, end_raw   = cfg['frog'], cfg['tip']
        reset_before, reset_after = cfg['fsafe'], cfg['tsafe']

    start_pressed = apply_depth(start_raw, cfg, note['commanded_depth'])
    end_pressed   = apply_depth(end_raw, cfg, note['commanded_depth'])
    return reset_before, start_pressed, end_pressed, reset_after


def prepare_stroke_start(ctrl, note):
    """
    Move to this stroke's start pose before audio recording begins. These setup
    moves are intentionally outside the recorded buffer; otherwise slow reset
    and placement moves can consume the buffer before the note finishes.
    """
    reset_before, start_pressed, end_pressed, reset_after = stroke_targets(note)

    # 1. this config's own reset point near the stroke start (lift, clear)
    safe_moveL(ctrl, reset_before, RESET_SPEED, MOVE_ACCEL, what="reset before stroke")

    # 2. onto the string at the stroke start, pressed in by depth
    safe_moveL(ctrl, start_pressed, RESET_SPEED, MOVE_ACCEL, what="place at stroke start")

    return end_pressed, reset_after


def execute_recorded_stroke(ctrl, note, end_pressed):
    """Run only the actual bow stroke while audio/state logging are active."""
    speed = note['commanded_speed']

    ctrl.state_logger.start()
    time.sleep(0.05)
    t0 = time.time()

    # 3. THE STROKE — the only move at commanded speed.
    safe_moveL(ctrl, end_pressed, speed, MOVE_ACCEL, what=f"bow stroke ({note['config']})")

    t1 = time.time()
    ctrl.state_logger.stop()

    return ctrl.state_logger.get_summary(t_start=t0, t_end=t1), t0, t1


def finish_stroke(ctrl, reset_after):
    # 4. lift off to this config's own reset point near the end.
    safe_moveL(ctrl, reset_after, RESET_SPEED, MOVE_ACCEL, what="reset after stroke")


def execute_full_stroke(ctrl, note):
    """
    Full unrecorded/sanity stroke: setup, stroke, lift. Recording uses the same
    pieces separately so the audio stream covers only the musical stroke.
    """
    end_pressed, reset_after = prepare_stroke_start(ctrl, note)
    summary, t0, t1 = execute_recorded_stroke(ctrl, note, end_pressed)
    finish_stroke(ctrl, reset_after)
    return summary, t0, t1


# ── Audio helpers ─────────────────────────────────────────────────

def find_focusrite_device():
    for i, dev in enumerate(sd.query_devices()):
        if ('Scarlett' in dev['name'] or 'Focusrite' in dev['name']) and dev['max_input_channels'] >= 1:
            print(f"Found Focusrite at index {i}: {dev['name']}")
            return i
    raise RuntimeError("Focusrite not found; set AUDIO_DEVICE manually.")


def probe_audio_channels(device, seconds: float = 1.5):
    """Level, in dBFS, on every input channel of `device`."""
    n_channels = int(sd.query_devices(device)['max_input_channels'])
    if n_channels < 1:
        return []
    buffer = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                    channels=n_channels, dtype='float32', device=device)
    sd.wait()
    return [20.0 * np.log10(float(np.sqrt(np.mean(buffer[:, i] ** 2))) + 1e-12)
            for i in range(n_channels)]


def select_audio_channel(device, preferred=None):
    """
    Pick the input channel to record, and prove it carries signal.

    Returns (channel, levels) with channel 1-based, or (0, levels) when nothing
    on the device is live. An explicit `preferred` is honoured even if it looks
    dead, so a deliberately quiet setup can still be recorded; the levels are
    reported either way.
    """
    levels = probe_audio_channels(device)
    if not levels:
        return 0, levels
    if preferred:
        return preferred, levels
    best = max(range(len(levels)), key=lambda i: levels[i])
    return (best + 1 if levels[best] > AUDIO_LIVE_DBFS else 0), levels


def report_audio_channels(levels, chosen):
    for i, level in enumerate(levels):
        mark = "  <- recording this one" if i + 1 == chosen else ""
        state = "live" if level > AUDIO_LIVE_DBFS else "no signal"
        print(f"    channel {i + 1}: {level:7.1f} dBFS  {state:9s}{mark}")


def validate_audio_chain(device, preferred=None):
    """
    Check every input before any recording happens, and settle on a channel.

    Returns the 1-based channel to record, or 0 if nothing is live. Replaces
    the old single-channel peak check, which could only ever look at input 1
    and so reported "CHECK SIGNAL" without saying that the signal was sitting
    on a different socket the whole time.
    """
    print("Validating audio chain...")
    channel, levels = select_audio_channel(device, preferred)
    report_audio_channels(levels, channel)
    if not channel:
        print("  ##  No input is carrying signal. Check the mic is plugged in,")
        print("  ##  the gain is up, 48V is on for a condenser, and macOS has")
        print("  ##  granted microphone access to the terminal.")
    return channel


# ── Recording ─────────────────────────────────────────────────────

def sample_state_timeline(log, t_start, t_end, interval=STATE_SAMPLE_INTERVAL):
    """
    Down-sample the logger's full-rate (100Hz) log to one point every
    `interval` seconds across [t_start, t_end] — the SAME window used for
    get_summary() — so metadata.jsonl carries points throughout the note,
    not just the start/midpoint/aggregate stats. The full-rate log is still
    saved separately per-stroke via state_logger.save().
    """
    window = [s for s in log if t_start <= s['t'] <= t_end]
    if not window:
        return []

    points = []
    idx = 0
    t = t_start
    while t <= t_end + 1e-9:
        while idx < len(window) - 1 and abs(window[idx + 1]['t'] - t) < abs(window[idx]['t'] - t):
            idx += 1
        s = window[idx]
        torque_mag = float(np.sqrt(s['ft_tx']**2 + s['ft_ty']**2 + s['ft_tz']**2))
        points.append({
            't_rel':        round(s['t'] - t_start, 4),
            'bow_position': s['bow_position'],
            'bow_speed':    s['bow_speed'],
            'torque_mag':   torque_mag,
            'tcp_x': s['tcp_x'], 'tcp_y': s['tcp_y'], 'tcp_z': s['tcp_z'],
        })
        t += interval
    return points


def make_state_sample_records(meta):
    """Flatten the per-stroke state timeline into JSONL-friendly sample rows."""
    timeline = meta.get('state_timeline', [])
    duration = float(meta.get('stroke_duration') or 0.0)
    rows = []
    for i, sample in enumerate(timeline):
        t_rel = float(sample.get('t_rel', 0.0))
        rows.append({
            'record_type': STATE_SAMPLE_RECORD_TYPE,
            'stroke_id': meta['stroke_id'],
            'repeat': meta['repeat'],
            'session_id': meta['session_id'],
            'timestamp': meta['timestamp'],
            'audio_file': meta['audio_file'],
            'state_file': meta['state_file'],
            'string': meta['string'],
            'config': meta['config'],
            'condition_label': meta['condition_label'],
            'condition_type': meta['condition_type'],
            'sample_index': i,
            'sample_interval_s': STATE_SAMPLE_INTERVAL,
            't_rel': t_rel,
            'stroke_phase': t_rel / duration if duration > 1e-9 else 0.0,
            'commanded': meta['commanded'],
            'state': sample,
        })
    return rows


def record_one(ctrl, note, condition_label, stroke_id, session_id, repeat, device=AUDIO_DEVICE):
    end_pressed, reset_after = prepare_stroke_start(ctrl, note)

    stroke_dur = note['duration']
    nominal_total_s = PRE_BUFFER + stroke_dur + POST_BUFFER
    total_samples = int((nominal_total_s + AUDIO_SAFETY_MARGIN) * SAMPLE_RATE)
    # mapping selects the PHYSICAL input. Passing channels=CHANNELS instead
    # would take input 1 regardless of which socket the mic is in.
    audio_buf = sd.rec(total_samples, samplerate=SAMPLE_RATE,
                       mapping=[AUDIO_CHANNEL or 1],
                       dtype='float32', device=device)
    rec_start = time.time()
    time.sleep(PRE_BUFFER)
    summary, t0, t1 = execute_recorded_stroke(ctrl, note, end_pressed)
    finish_stroke(ctrl, reset_after)
    time.sleep(POST_BUFFER)

    # Stop and trim to what actually elapsed in real time, instead of
    # trusting the pre-stroke distance/speed estimate. The safety-margin
    # buffer above guarantees recording never auto-stopped early; this just
    # cuts the unused trailing margin back off.
    elapsed_s = time.time() - rec_start
    sd.stop()
    sd.wait()
    n_valid = min(int(round(elapsed_s * SAMPLE_RATE)), total_samples)
    audio_buf = audio_buf[:n_valid]
    if n_valid >= total_samples:
        print("  WARNING: audio buffer filled before recording stopped; "
              "increase AUDIO_SAFETY_MARGIN if this file is still cut off.")

    state_timeline = sample_state_timeline(ctrl.state_logger.log, t0, t1)

    fname_audio = f"{stroke_id}_r{repeat:02d}.wav"
    fname_state = f"{stroke_id}_r{repeat:02d}_state.npy"
    sf.write(str(AUDIO_DIR / fname_audio), audio_buf, SAMPLE_RATE)
    ctrl.state_logger.save(str(STATE_DIR / fname_state))
    peak = float(np.max(np.abs(audio_buf)))
    level_dbfs = 20.0 * np.log10(float(np.sqrt(np.mean(audio_buf ** 2))) + 1e-12)
    if level_dbfs < AUDIO_LIVE_DBFS:
        print(f"  ##  {fname_audio} came back SILENT ({level_dbfs:.1f} dBFS) — "
              f"check input {AUDIO_CHANNEL or 1}")

    cfg = BOW_CONFIGS[note['config']]
    speed = note['commanded_speed']
    meta = {
        'record_type': STROKE_RECORD_TYPE,
        'stroke_id': stroke_id, 'repeat': repeat, 'session_id': session_id,
        'timestamp': datetime.now().isoformat(),
        'audio_file': fname_audio, 'state_file': fname_state,
        'string': RECORDING_STRING, 'config': note['config'],
        'commanded': {
            'config': note['config'], 'depth_m': float(note['commanded_depth']),
            'depth_offset_m': float(note['commanded_depth']),
            'depth_reference': '0.0 is taught pendant waypoint; negative is outward/lighter',
            'speed': float(speed), 'bow_dir': note['bow_dir'],
            'duration': float(stroke_dur), 'dynamic_est': note['dynamic'],
            'frog_pose': cfg['frog'].tolist(), 'tip_pose': cfg['tip'].tolist(),
            'fsafe_pose': cfg['fsafe'].tolist(), 'tsafe_pose': cfg['tsafe'].tolist(),
        },
        'measured': summary,
        'force_contact': {
            'force_mean': summary.get('force_mean'), 'force_std': summary.get('force_std'),
            'force_min': summary.get('force_min'), 'force_max': summary.get('force_max'),
            'contact_mag_mean': summary.get('contact_mag_mean'),
            'contact_mag_std': summary.get('contact_mag_std'),
            'contact_mag_max': summary.get('contact_mag_max'),
            'fx_mean': summary.get('fx_mean'), 'fy_mean': summary.get('fy_mean'),
        },
        'speed_accuracy': {
            'commanded': float(speed), 'meas_mean': float(summary.get('speed_mean', 0.0)),
            'meas_max': float(summary.get('speed_max', 0.0)),
            'max_ratio': float(summary.get('speed_max', 0.0) / speed) if speed else 0.0,
        },
        'audio_channel': AUDIO_CHANNEL or 1,
        'audio_timing': {
            # Actual measured times (relative to when recording started),
            # not the pre-stroke distance/speed estimate — moveL's real
            # duration regularly overruns that estimate, which used to leave
            # the audio buffer too short and cut off the tail of the note.
            'sample_rate': SAMPLE_RATE, 'pre_buffer_s': PRE_BUFFER,
            'post_buffer_s': POST_BUFFER, 'stroke_start_s': t0 - rec_start,
            'stroke_end_s': t1 - rec_start, 'duration_s': t1 - t0,
            'recorded_duration_s': n_valid / SAMPLE_RATE,
        },
        'state_timeline': state_timeline,
        'state_timeline_interval_s': STATE_SAMPLE_INTERVAL,
        'stroke_duration': t1 - t0, 'audio_peak': peak,
        'condition_label': condition_label,
        'condition_type': ('test' if condition_label.startswith('TEST_')
                           else 'bad' if condition_label.startswith('bad_') else 'systematic'),
        'annotations': [],
    }
    sample_rows = make_state_sample_records(meta)
    with open(META_FILE, 'a') as f:
        f.write(json.dumps(meta) + '\n')
        for row in sample_rows:
            f.write(json.dumps(row) + '\n')

    print(f"  {fname_audio} | cfg={note['config']:8s} depth_offset={note['commanded_depth']*1000:+.1f}mm "
          f"F={summary.get('force_mean',0):.2f}N contact={summary.get('contact_mag_mean',0):.2f}N | "
          f"v_cmd={speed:.3f} v_max={summary.get('speed_max',0):.3f} | peak={peak:.3f} | "
          f"state_rows={len(sample_rows)}")
    return meta


# ── Full session (resumable) ──────────────────────────────────────

def build_conditions(randomize=True, seed=42):
    conditions = []
    bow_dir = 'down'
    for config_name in BOW_CONFIGS:
        for depth_name, depth_val in DEPTH_LEVELS.items():
            for speed_name, speed_val in SPEED_LEVELS.items():
                conditions.append({
                    'config': config_name, 'depth': depth_val, 'speed': speed_val,
                    'bow_dir': bow_dir, 'label': f"{config_name}_D{depth_name}_S{speed_name}",
                    'type': 'systematic',
                })
                bow_dir = 'up' if bow_dir == 'down' else 'down'
    if randomize:
        random.Random(seed).shuffle(conditions)
    for bad in BAD_CONDITIONS:
        conditions.append({
            'config': 'standard', 'depth': bad['depth'], 'speed': bad['speed'],
            'bow_dir': 'down', 'label': f"bad_{bad['label']}", 'type': 'bad',
        })
    return conditions


def run_collection_session(ctrl, randomize=True):
    session_id = get_session_id()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions(randomize=randomize)
    done = load_completed_keys()
    total = len(conditions) * N_REPEATS
    target_keys = {
        condition_key(cond['label'], repeat)
        for cond in conditions
        for repeat in range(N_REPEATS)
    }
    done_in_target = done & target_keys

    print(f"\n{'='*60}\nSession {session_id}  (A string, {len(BOW_CONFIGS)} configs)")
    print(f"Conditions: {len(conditions)}   Target recordings: {total}")
    print(f"Already completed: {len(done_in_target)}   Remaining: {total - len(done_in_target)}\n{'='*60}")

    last_config = None
    for ci, cond in enumerate(conditions):
        stroke_id = f"{session_id}_{RECORDING_STRING}_{cond['label']}_{ci:03d}"
        remaining = [r for r in range(N_REPEATS)
                    if condition_key(cond['label'], r) not in done]
        if not remaining:
            continue

        print(f"\n[{ci+1}/{len(conditions)}] {cond['label']}  ({len(remaining)} to do)")
        # Re-tare whenever the config changes (orientation differs by config).
        if cond['config'] != last_config:
            zero_ft_at_config_safe(ctrl, cond['config'])
            last_config = cond['config']

        for repeat in remaining:
            note = make_stroke_note(cond['config'], cond['depth'], cond['speed'], cond['bow_dir'])
            record_one(ctrl, note, cond['label'], stroke_id, session_id, repeat)
            done.add(condition_key(cond['label'], repeat))

        if ci < len(conditions) - 1:
            time.sleep(30 if ci % 5 == 4 else 5)

    with open(SESSION_LOG, 'a') as f:
        f.write(json.dumps({
            'session_id': session_id, 'string': RECORDING_STRING,
            'date': datetime.now().isoformat(), 'configs': list(BOW_CONFIGS.keys()),
            'depth_levels': DEPTH_LEVELS, 'speed_levels': SPEED_LEVELS,
            'completed_after_run': len(done & target_keys), 'target_total': total,
        }) + '\n')

    print(f"\n{'='*60}\nRun complete: {len(done)}/{total} recordings.")
    if len(done) < total:
        print("Re-run this script to continue where it left off.")
    print('='*60)


def run_test_recording(ctrl):
    session_id = "TEST_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    label = f"TEST_{TEST_CONFIG}_D{TEST_DEPTH*1000:.1f}mm_S{TEST_SPEED}_{TEST_BOW_DIR}"
    stroke_id = f"{session_id}_{RECORDING_STRING}_{label}"

    print(f"\n{'='*60}\nTESTING — {TEST_CONFIG}  depth {TEST_DEPTH*1000:.1f}mm  "
          f"speed {TEST_SPEED}  {TEST_BOW_DIR}\n{'='*60}")
    zero_ft_at_config_safe(ctrl, TEST_CONFIG)
    for repeat in range(TEST_REPEATS):
        note = make_stroke_note(TEST_CONFIG, TEST_DEPTH, TEST_SPEED, TEST_BOW_DIR)
        record_one(ctrl, note, label, stroke_id, session_id, repeat)
    print(f"\nTest complete: {TEST_REPEATS} recording(s) in {AUDIO_DIR}")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == '__main__':
    print("\nRobot Cello: A-string bowing-config data collection (resumable)")
    print("Motion is restricted to your defined points only, PER CONFIG:")
    print("frog/tip/fsafe/tsafe — each config uses its OWN local safe points,")
    print("not a shared/generic reset point. No invented motion.")
    if TESTING:
        print("*** TESTING MODE — single condition ***")
    print("=" * 50)

    audio_device = find_focusrite_device()
    ctrl = TargetStringBaselineController([])

    # Set installation BEFORE any motion (must match stringa_closer.script).
    apply_verified_installation(ctrl)

    try:
        AUDIO_CHANNEL = validate_audio_chain(audio_device, AUDIO_CHANNEL or None)
        if not AUDIO_CHANNEL:
            answer = input("\n  Continue anyway, recording silence? [y/N] ")
            if answer.strip().lower() not in ('y', 'yes'):
                print("Aborted — the robot has not been touched.")
                raise SystemExit(1)
            AUDIO_CHANNEL = 1

        input(f"\nPress ENTER to do ONE sanity stroke at '{TEST_CONFIG}' "
              f"(uses ONLY {TEST_CONFIG}'s own points)...")
        sanity_note = make_stroke_note(
            TEST_CONFIG,
            PENDANT_REFERENCE_DEPTH,
            PENDANT_REFERENCE_SPEED,
            'down',
        )
        zero_ft_at_config_safe(ctrl, TEST_CONFIG)
        execute_full_stroke(ctrl, sanity_note)
        print("  sanity stroke done — check the robot before continuing.")

        if TESTING:
            input("\nPress ENTER to begin TEST recording...")
            run_test_recording(ctrl)
        else:
            input("\nPress ENTER to begin/resume data collection...")
            run_collection_session(ctrl, randomize=True)

    except RobotFaultStop as e:
        print("\n" + "!"*60)
        print("ROBOT FAULT — SCRIPT STOPPED")
        print("!"*60)
        print(str(e))
        print("Check the robot by eye and on the teach pendant before doing")
        print("anything else.")
        print("!"*60)
        raise

    finally:
        ctrl.close()
