#!/usr/bin/env python3

# rhythm section 
# slightly slower tempo 

# increase bow speed and make the bow lighter and reduce tension at the end of the bow 
# use slightly more bow less weight

"""
play_midi_pieces.py

Play a whole piece of open-A-string music on the cello robot, from a MIDI file.

This is the performance counterpart to SoundClassifier/Data_Collection/
recording_a_only.py. That script plays ONE isolated stroke per condition,
always across the FULL bow, always starting from a taught endpoint. A piece
needs the opposite: many strokes of DIFFERENT lengths, chained back to back,
each one starting wherever the previous one happened to stop. So the one thing
this file adds over the recording script is a *bow budget* — it tracks where
the bow is and decides, per note, whether the intended stroke actually fits.

Everything else is deliberately inherited, not reinvented:

    - the waypoints            BOW_CONFIGS['standard'] (frog/tip/fsafe/tsafe)
    - the installation         VERIFIED_TCP / payload (set BEFORE any motion)
    - the depth convention     apply_depth(), signed offset along +z
    - the motion primitive     moveL, via safe_moveL() with fault checking
    - the state logging        StateLogger at 100 Hz

all imported from recording_a_only so the numbers cannot drift apart. Poses
here are interpreted in exactly the frame those points were verified in.


THE ONE GEOMETRIC EXTENSION
───────────────────────────
recording_a_only only ever commands taught points. A piece needs poses part
way along the bow, so pose_at(u) interpolates frog -> tip for u in [0, 1].
The translation is a straight lerp along the same line the validated full-bow
stroke already traverses. The ROTATION needs care: the taught frog and tip
rotation vectors sit in OPPOSITE hemispheres of the axis-angle double cover
(|r| ~ 3.136 and ~3.097, both just under pi, with flipped signs). They are
only 4.28 degrees apart physically, but lerping the raw vectors passes through
r ~ 0 — a 178.5 degree error at the midpoint, i.e. a full wrist flip mid-stroke.
pose_at() canonicalises tip into frog's hemisphere first. _verify_geometry()
asserts this at startup and is also run by --dry-run.


DYNAMICS: WHAT THE DATA SAYS
────────────────────────────
Measured over the 100 'standard' strokes in dataset_a_final (RMS of the audio
inside the logged stroke window, grouped by commanded depth and speed):

    bow speed  0.09 -> 0.25 m/s      -29.4 -> -20.6 dBFS     8.8 dB, monotonic
    depth     -1.5 -> +0.8 mm        -24.2 -> -23.4 dBFS     0.8 dB, not monotonic

and 20*log10(0.25/0.09) = 8.87 dB against 8.84 dB measured — i.e. over this
range the radiated amplitude is, to within the measurement noise, directly
PROPORTIONAL TO BOW SPEED. Depth over the +-2 mm safety envelope is worth
under a decibel and does not order cleanly.

So dynamics are driven by bow speed, and depth is mapped from volume as a
secondary/timbral trim (deeper = louder, as requested) rather than as the main
lever. Because loudness is speed and speed x duration is bow length, dynamics
and the bow budget are the SAME constraint: a loud long note eats bow. That
coupling is the whole reason the planner below exists.

If you widen the depth envelope later (MAX_INWARD_DEPTH is a 2 mm safety
ceiling, not a physical limit), re-run the calibration and raise DEPTH_WEIGHT.


BOW PROTOCOL
────────────
Bow position u runs 0.0 (frog) to 1.0 (tip); down-bow increases u, up-bow
decreases it. Usable range is [U_MIN, U_MAX] with a hard margin at each end.
Per note, in order:

    1. direction   = user override, else score bowing, else alternate
    2. volume      -> mean bow speed -> stroke length L = v * duration
    3. does L fit in the remaining bow in that direction?
         fits                    -> play it
         overshoots comfort band -> shrink within BOW_SLACK (bow distribution)
         still doesn't fit, dir was automatic
                                 -> FLIP and play the note the other way
         still doesn't fit, dir was forced by the user or the score
                                 -> RETAKE: lift to +z, traverse in the air,
                                    set down again, keeping BOTH the bowing
                                    and the dynamic intact
         longer than the whole bow
                                 -> SPLIT into k alternating sub-strokes
                                    (a bow change mid-note, what a cellist does)

FLIP is free and silent. RETAKE costs time and is scheduled into the gap before
the note; if the gap is too short the planner says so instead of silently
running late. Nothing is ever commanded past U_MIN/U_MAX.


SCORE -> ROBOT: WHAT CAN AND CANNOT BE AUTO-TRANSLATED
──────────────────────────────────────────────────────
DYNAMICS: yes, automatically, from the MIDI alone. Note velocity carries the
dynamic marking (MuseScore's standard scale, ppp=16 ... fff=126) and CC2
carries the continuous curve, so hairpins are recoverable — including
crescendos that happen WITHIN one long note. t1.mid uses exactly this: 25
notes, velocities 49..96, with 250 CC2 events shaping them.

BOWINGS: no, not from MIDI — the format has no channel for up-bow/down-bow.
Standard MIDI carries pitch, time and velocity, and that is all. There are
three real ways to get bowings in, all supported here:

    a .mxl / .musicxml file  MuseScore's own export format DOES carry bowings,
                            as <technical><up-bow/>/<down-bow/>, along with
                            slurs, dynamic marks and hairpins. This is the
                            honest answer to "auto-translate the bowings in the
                            music": if they are written in the score, this
                            reads them. Compressed .mxl is read directly, no
                            need to unzip or convert.

                            PREFER PASSING THE .mxl AS THE ONLY INPUT. It
                            carries the notes, the rhythm, the dynamics AND the
                            bowings, so nothing has to be matched up between
                            two files:
                                play_midi_pieces.py piece.mxl

    --score FILE.mxl        Overlay a score's bowings and slurs onto a MIDI,
                            keeping the MIDI's timing and CC2 dynamics. Notes
                            are matched BY POSITION, so it is only safe when
                            both files came from the same engraving; the note
                            counts are compared and any mismatch is reported
                            loudly, because past the first discrepancy every
                            bowing would land on the wrong note.

                            The usual cause of a mismatch is repeats: MuseScore
                            PLAYS THEM OUT in a MIDI export but leaves one
                            written copy in MusicXML. Repeat barlines and
                            voltas are therefore expanded by default (see
                            --no-repeats) so the two agree. Da capo and dal
                            segno are not expanded, and say so when present.

    --annotations FILE.json Per-note overrides by index, for bowings that are
                            not in any file and dynamics you want to force.

    --bowing rule-of-downbow
                            Infer bowings from the metre when the score has
                            none: strong beats take a down-bow, the rest
                            alternate. This is the actual convention cellists
                            use, and it is derivable from the time signature
                            the MIDI already contains. It will produce
                            repeated down-bows, which the planner turns into
                            retakes — again, what a player does.

Slurs (from MusicXML) are honoured as one bow across the group. On an open A
that is literally one sustained stroke, since a slur adds no articulation the
bow can hear.


USAGE
─────
    # look at the plan without touching the robot — always do this first
    python BaselineControls/play_midi_pieces.py MIDI-Files/t1.mid --dry-run

    # play it
    python BaselineControls/play_midi_pieces.py MIDI-Files/t1.mid

    # BEST: play the MuseScore export directly, and get the written bowings,
    # slurs and dynamics with nothing to match up between files
    python BaselineControls/play_midi_pieces.py piece.mxl

    # or keep the MIDI's timing and take only the bowings from the score
    python BaselineControls/play_midi_pieces.py MIDI-Files/t1.mid \
        --score piece.mxl

    # infer bowings from the metre, record the performance
    python BaselineControls/play_midi_pieces.py MIDI-Files/t1.mid \
        --bowing rule-of-downbow --record

Run it with .venv311 (mido, ur_rtde, sounddevice all live there).
"""

import argparse
import importlib.util
import json
import math
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIDI = REPO_ROOT / "MIDI-Files" / "t1.mid"


# ══════════════════════════════════════════════════════════════════
# Inherit the verified setup from the recording script
# ══════════════════════════════════════════════════════════════════
# Loaded by path: SoundClassifier/Data_Collection has no __init__.py, so it is
# not importable as a package. Importing it (rather than copying the numbers in)
# is the point — the waypoints, the installation and the depth convention stay
# identical to the ones that were verified on the pendant and used to record
# dataset_a_final. If they ever change there, they change here too.

def _load_recording_module():
    path = REPO_ROOT / "SoundClassifier" / "Data_Collection" / "recording_a_only.py"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find the recording script at {path}")
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("recording_a_only", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        # NOTE: the interpreter named here used to be .venv311, which now lives
        # in an iCloud-synced directory and had its 60k files evicted — it can
        # no longer import anything. ~/venvs/cello311 is the working
        # environment (outside iCloud, with ur_rtde 1.6.3 built against
        # boost@1.85). Pointing at the dead one sent people in circles.
        raise ImportError(
            f"Could not import recording_a_only ({e}).\n"
            f"It pulls in mido, ur_rtde, sounddevice and soundfile. "
            f"Run this script with\n"
            f"  ~/venvs/cello311/bin/python "
            f"BaselineControls/{Path(__file__).name} ...\n"
            f"(NOT .venv or .venv311 — those lack the dependencies.)\n"
        ) from e
    return module


_rec = _load_recording_module()

BOW_CONFIGS          = _rec.BOW_CONFIGS
MAX_OUTWARD_DEPTH    = _rec.MAX_OUTWARD_DEPTH      # 1.5 mm out (lighter)
MAX_INWARD_DEPTH     = _rec.MAX_INWARD_DEPTH       # 2.0 mm in  (heavier)
apply_depth          = _rec.apply_depth
press_direction      = _rec.press_direction
safe_moveL           = _rec.safe_moveL
RobotFaultStop       = _rec.RobotFaultStop
MOVE_ACCEL           = _rec.MOVE_ACCEL             # 1.2 m/s^2, matches pendant
RESET_SPEED          = _rec.RESET_SPEED            # 0.25 m/s, matches pendant
SAMPLE_RATE          = _rec.SAMPLE_RATE


# ══════════════════════════════════════════════════════════════════
# Bow geometry
# ══════════════════════════════════════════════════════════════════

CONFIG_NAME = "standard"

# Safety margin at each end of the bow. The planner will never command a pose
# outside [U_MIN, U_MAX], so the bow physically cannot be run off either end.
U_MIN = 0.04
U_MAX = 0.96

# Home band — where strokes ought to be landing. Anything ending outside it
# gets nudged back toward it, lengthening or shortening within BOW_SLACK.
#
# The pull has to be able to LENGTHEN a stroke, not only shorten it. An earlier
# version clipped the landing point to the edge of a band and only ever
# shortened, which made that edge an attractor: once the bow arrived there,
# alternating strokes of equal length left it stuck, and the rest of the piece
# was played in one small patch of hair near the frog. A symmetric pull toward
# the middle has no such fixed point.
#
# Width was chosen by measurement, not taste. Swept over t1.mxl, batman,
# minuet_no_2v2, allegro and rach_2, counting strokes landing within 3% of a
# hard limit and dynamic shortfalls caused by the correction itself:
#
#   [0.32,0.68]   9 shortfalls on t1   — too tight, trims healthy strokes
#   [0.22,0.78]   5 shortfalls on t1, and rach_2 reached u=0.053
#   [0.18,0.82]   2 shortfalls on t1, nothing within 3% of a limit except
#                 batman at 0.132 — still 0.09 clear
#   [0.12,0.88]   0 shortfalls, but batman reached u=0.072
#
# Tighter is not safer: a narrow band spends the slack budget on constant small
# corrections, leaving none when a real one is needed, which is why 0.22 did
# worse at the limits than 0.18 did.
U_HOME_LO = 0.18
U_HOME_HI = 0.82

# How much a stroke's length may be modulated, as a fraction, purely to keep
# the bow in a usable part of its range. This is bow distribution: cellists
# routinely spend a bit more or less bow to stay out of trouble. +-20% of
# length is +-20% of mean speed, which by the calibration above is about
# +-1.6 dB — audible as shading, not as a wrong dynamic.
BOW_SLACK = 0.20


def _bow_config():
    cfg = BOW_CONFIGS[CONFIG_NAME]
    return cfg


CFG   = _bow_config()
FROG  = np.asarray(CFG["frog"],  dtype=float)
TIP   = np.asarray(CFG["tip"],   dtype=float)
FSAFE = np.asarray(CFG["fsafe"], dtype=float)
TSAFE = np.asarray(CFG["tsafe"], dtype=float)

BOW_LENGTH   = float(np.linalg.norm(TIP[:3] - FROG[:3]))          # ~0.525 m
USABLE_BOW   = BOW_LENGTH * (U_MAX - U_MIN)                        # ~0.494 m

# Speed for the lift/traverse/set-down moves of a retake.
#
# RESET_SPEED (0.25 m/s) is the pendant-verified figure the recording sessions
# used, and it is the safe default. But it makes a retake take about two
# seconds, which is far longer than the silence between notes in most music, so
# a forced bowing that needs one will run late. A retake happens with the bow
# LIFTED CLEAR of the string, so the string cannot be struck by going faster;
# the limit is how hard you want the arm accelerating near the instrument.
# Raise it with --retake-speed, in steps, watching the robot.
RETAKE_SPEED = RESET_SPEED

# Lift-off distance for retakes. fsafe is exactly frog + 54.6 mm in +z and
# tsafe is exactly tip + 55.3 mm in +z, so "lift" is pure world +z here and a
# safe point above ANY u is just pose_at(u) + LIFT_M. Use the smaller of the
# two taught clearances so a mid-bow lift is never more generous than the two
# clearances that were actually verified.
LIFT_M = float(min(np.linalg.norm(FSAFE[:3] - FROG[:3]),
                   np.linalg.norm(TSAFE[:3] - TIP[:3])))


def _rv_to_matrix(r):
    """Axis-angle rotation vector -> rotation matrix (Rodrigues)."""
    r = np.asarray(r, dtype=float)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _canonicalise_rotvec(r, reference):
    """
    Return whichever of the two equivalent axis-angle representations of `r`
    lies nearer `reference`.

    A rotation vector r and r * (|r| - 2pi) / |r| describe the SAME rotation
    (rotating by |r| about k equals rotating by |r| - 2pi about k). Near
    |r| = pi the two are far apart numerically while being physically
    identical, and UR poses land there constantly. The taught frog and tip
    rotations are exactly this case, so tip must be pulled into frog's
    hemisphere before anything interpolates between them.
    """
    r = np.asarray(r, dtype=float)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return r
    alternative = r * (theta - 2.0 * math.pi) / theta
    if np.linalg.norm(alternative - reference) < np.linalg.norm(r - reference):
        return alternative
    return r


# Tip rotation expressed in the same hemisphere as frog's, computed once.
_TIP_ROT_CANON = _canonicalise_rotvec(TIP[3:], FROG[3:])


def pose_at(u: float) -> np.ndarray:
    """
    Pose at bow fraction u, 0.0 = frog, 1.0 = tip.

    Position is a straight lerp along the frog -> tip line: the same line the
    validated full-bow stroke already travels, so a partial stroke stays on the
    trajectory the taught points describe. Orientation lerps between the taught
    rotations AFTER canonicalisation; the two are 4.28 degrees apart, so over
    that span a normalised lerp and a true slerp agree to far better than a
    thousandth of a degree (checked by _verify_geometry).

    Exactly at the ends the taught pose is returned verbatim, so a full stroke
    is bit-for-bit the move recording_a_only makes.
    """
    u = float(np.clip(u, 0.0, 1.0))
    if u <= 1e-9:
        return FROG.copy()
    if u >= 1.0 - 1e-9:
        return TIP.copy()

    pose = np.empty(6, dtype=float)
    pose[:3] = FROG[:3] + u * (TIP[:3] - FROG[:3])
    pose[3:] = FROG[3:] + u * (_TIP_ROT_CANON - FROG[3:])
    return pose


def lifted(pose) -> np.ndarray:
    """Pose lifted clear of the string, in the same +z sense as fsafe/tsafe."""
    out = np.asarray(pose, dtype=float).copy()
    out[:3] = out[:3] + LIFT_M * press_direction(CFG)
    return out


def _verify_geometry(verbose: bool = False):
    """
    Startup self-check on the interpolation. Cheap, and it guards the one place
    where a silent error would drive the wrist through a 178 degree flip in the
    middle of a stroke rather than just sounding wrong.
    """
    assert np.allclose(pose_at(0.0), FROG, atol=1e-12), "pose_at(0) != taught frog"
    assert np.allclose(pose_at(1.0), TIP,  atol=1e-12), "pose_at(1) != taught tip"

    R_frog, R_tip = _rv_to_matrix(FROG[3:]), _rv_to_matrix(TIP[3:])

    def angle_between(Ra, Rb):
        return math.degrees(math.acos(
            float(np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0))))

    span = angle_between(R_frog, R_tip)
    assert span < 10.0, f"frog/tip orientations {span:.2f} deg apart, too far to lerp"

    # Canonicalised lerp vs the exact geodesic, sampled along the bow.
    worst = 0.0
    for u in np.linspace(0.0, 1.0, 51):
        R_lerp = _rv_to_matrix(pose_at(u)[3:])
        expected = u * span                       # constant-rate geodesic
        worst = max(worst, abs(angle_between(R_frog, R_lerp) - expected))
    assert worst < 0.05, f"lerp deviates {worst:.4f} deg from the geodesic"

    # And the check that actually matters: the naive version is catastrophic.
    naive_mid = _rv_to_matrix((FROG[3:] + TIP[3:]) / 2.0)
    naive_err = angle_between(R_frog, naive_mid)

    if verbose:
        print("Geometry self-check")
        print(f"  bow length            {BOW_LENGTH*1000:7.1f} mm "
              f"(usable {USABLE_BOW*1000:.1f} mm over u in [{U_MIN}, {U_MAX}])")
        print(f"  lift-off clearance    {LIFT_M*1000:7.1f} mm along +z")
        print(f"  frog->tip orientation {span:7.2f} deg apart")
        print(f"  canonicalised lerp    {worst:7.4f} deg worst-case error   OK")
        print(f"  (naive lerp would be  {naive_err:7.2f} deg off at the midpoint)")
    return True


# ══════════════════════════════════════════════════════════════════
# Dynamics: volume -> bow speed (primary) and depth (secondary)
# ══════════════════════════════════════════════════════════════════
# SPEED_MIN/MAX are the endpoints of the calibration grid in dataset_a_final,
# not guesses: every speed in between was actually recorded on this instrument
# in this configuration. Staying inside the measured range means the mapping is
# interpolation rather than extrapolation.

SPEED_MIN = 0.09    # m/s  measured -29.4 dBFS
SPEED_MAX = 0.25    # m/s  measured -20.6 dBFS
SPEED_FLOOR = 0.05  # m/s  below this the tone breaks up; used only as a clamp

# Depth still tracks volume as requested (less depth quieter, more depth
# louder), spanning the same safety envelope the recording script enforces.
# Worth about 0.8 dB by measurement, so it shades the tone rather than setting
# the dynamic. DEPTH_WEIGHT scales how much of the envelope volume is allowed
# to sweep; raise it if the envelope is ever widened and recalibrated.
DEPTH_WEIGHT = 1.0

# Acceleration. MOVE_ACCEL (1.2 m/s^2) is the pendant-matched default and is
# used whenever it is enough. Short notes physically cannot be played at it:
# covering L metres in T seconds needs at least 4L/T^2, so a 0.08 s note needs
# several m/s^2 no matter how short the stroke. ACCEL_MAX is the ceiling the
# planner is allowed to raise to, and stroke lengths are cut before it.
ACCEL_MAX = 6.0     # m/s^2
# Raised from 4.0 on 2026-08-17. 4.0 was a conservative choice, not a hardware
# limit (a UR5e does far more at the tool), and it was costing two things:
#
#   1. It ate 46% of the residual policy's UPWARD speed authority. The policy
#      asks for length * (1 + a*0.35); solve_stroke then cuts the length back
#      to what 4LT^-2 <= accel_max allows, so on short notes the request was
#      silently discarded. Measured on yunpiece.mxl: a=+1 delivered a median
#      0.1109 m/s where the residual asked for 0.1381. At 6.0 and above the
#      full range arrives. Every training run so far trained with half its
#      speed range clipped away.
#   2. Bow REVERSAL costs 2v/a, and this piece reverses 178 times in 42 s. At
#      4.0 that is ~51 ms of turnaround inside a 111 ms note -- nearly half of
#      each fast note spent not sounding, which is why the fast passage would
#      not hold its rhythm at any tempo.
#
# 8.0 was tried first and MEASURED WORSE on hardware 2026-08-17: period_corr
# fell 0.911 -> 0.846 overall and 0.896 -> 0.813 on the fast 100-200 ms notes,
# while long notes were untouched (0.970 -> 0.967). Damage confined to the
# notes that actually spend the extra acceleration is the bow-bounce
# signature -- above ~8 m/s^2 the bow chatters at note starts instead of
# gripping. (The CNN judge called that take BETTER, 0.479 vs 0.473, which is
# the third time it disagreed with the physical measures and the ear.)
#
# 6.0 is the smallest value that still buys the FULL speed range -- the
# residual's +-35% saturates exactly here -- so it takes the whole policy
# benefit while staying as far from the bounce threshold as that allows.
# Turnaround is ~34 ms rather than 4.0's ~51 ms. Do not raise without
# re-measuring period_corr on the fast passage.
#
# NOTE: the reward's price for harsh attacks used to be normalised by this
# same constant, so raising it made harsh attacks cheaper. That is now
# pinned separately as piece_env.ONSET_ACCEL_SCALE.

# Target ratio of commanded PEAK speed to the stroke's mean speed.
#
# Loudness follows the mean speed (length/duration), but moveL is commanded
# with the peak, and the ramps mean the peak always exceeds the mean. How much
# is set by the acceleration: at the bare minimum accel that fits the note
# (4L/T^2) the profile is triangular and the peak is TWICE the mean, which both
# swoops audibly within the note and pushes the commanded speed past the
# 0.25 m/s ceiling the dataset actually calibrated. Solving v/a + L/v = T for a
# given ratio r = v*T/L gives
#
#     a = r^2 / (r - 1) * L / T^2
#
# so r = 1.25 costs 6.25 L/T^2, about 1.6x the minimum, and keeps the commanded
# speed inside the measured range with a much steadier bow. Accel is still
# capped by ACCEL_MAX, so short notes fall back to whatever ratio they can get.
PEAK_SPEED_RATIO = 1.25

# Commanded peak speeds above this are extrapolating past the calibration grid
# in dataset_a_final; the planner reports how many strokes do it.
CALIBRATED_SPEED_MAX = 0.25

# MuseScore's velocity scale for dynamic marks. Used both to name a dynamic and
# to place a velocity on a continuous 0..1 volume axis.
DYNAMIC_VELOCITIES = [
    ("ppp",  16), ("pp",  33), ("p",   49), ("mp",  64),
    ("mf",   80), ("f",   96), ("ff", 112), ("fff", 126),
]
DYNAMIC_BY_NAME = {name: vel for name, vel in DYNAMIC_VELOCITIES}


def velocity_to_volume(velocity: float) -> float:
    """MIDI velocity -> volume on a 0..1 axis where 0 = ppp and 1 = fff."""
    lo, hi = DYNAMIC_VELOCITIES[0][1], DYNAMIC_VELOCITIES[-1][1]
    return float(np.clip((float(velocity) - lo) / (hi - lo), 0.0, 1.0))


def volume_to_velocity(volume: float) -> int:
    lo, hi = DYNAMIC_VELOCITIES[0][1], DYNAMIC_VELOCITIES[-1][1]
    return int(round(lo + float(np.clip(volume, 0.0, 1.0)) * (hi - lo)))


def velocity_to_dynamic(velocity: float) -> str:
    """Nearest dynamic marking to a MIDI velocity, on MuseScore's scale."""
    return min(DYNAMIC_VELOCITIES, key=lambda kv: abs(kv[1] - float(velocity)))[0]


def volume_to_speed(volume: float) -> float:
    """
    Volume -> mean bow speed.

    Linear in speed because the calibration found amplitude proportional to
    speed, so equal steps in volume give equal steps in amplitude. The stroke
    LENGTH that comes out of this (speed x duration) is what the bow planner
    then has to fit into the remaining bow.
    """
    v = float(np.clip(volume, 0.0, 1.0))
    return SPEED_MIN + v * (SPEED_MAX - SPEED_MIN)


def speed_to_volume(speed: float) -> float:
    """Inverse of volume_to_speed, for reporting what a clipped stroke sounds like."""
    return float(np.clip((speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN), 0.0, 1.0))


def volume_to_depth(volume: float) -> float:
    """
    Volume -> signed depth offset, in the recording script's convention:
    negative is outward/lighter, positive presses in/heavier, 0.0 is the taught
    waypoint. Clamped by apply_depth() to the same +-safety envelope.
    """
    v = float(np.clip(volume, 0.0, 1.0))
    span = (MAX_OUTWARD_DEPTH + MAX_INWARD_DEPTH) * DEPTH_WEIGHT
    return -MAX_OUTWARD_DEPTH * DEPTH_WEIGHT + v * span


def speed_to_dbfs(speed: float) -> float:
    """
    Predicted stroke loudness, from the fit to dataset_a_final: amplitude is
    proportional to bow speed, anchored at the measured 0.25 m/s = -19.9 dBFS.
    Only used to report expected dynamics and shortfalls.
    """
    return 20.0 * math.log10(max(speed, 1e-6) / SPEED_MAX) - 19.9


# ══════════════════════════════════════════════════════════════════
# Trapezoidal motion solving
# ══════════════════════════════════════════════════════════════════
# moveL ramps up at `accel`, cruises at `speed`, ramps down. Duration is not
# distance/speed — ignoring the ramps is what made the recording script's
# original audio buffers too short. The planner needs the inverse: given a
# stroke length and the number of seconds the note lasts, what speed and accel
# make moveL take exactly that long?

def trapezoid_duration(length: float, speed: float, accel: float) -> float:
    """How long moveL actually takes for `length` at `speed` under `accel`."""
    if length <= 0.0:
        return 0.0
    if speed <= 0.0 or accel <= 0.0:
        return float("inf")
    if speed * speed / accel >= length:
        return 2.0 * math.sqrt(length / accel)     # triangular: never cruises
    return speed / accel + length / speed          # trapezoidal


@dataclass
class MotionSolution:
    """A commanded (speed, accel) plus what it actually achieves."""
    length: float          # m, after any reduction
    speed: float           # m/s, commanded peak
    accel: float           # m/s^2, commanded
    duration: float        # s, what moveL will really take
    mean_speed: float      # m/s, length/duration — this is what sets loudness
    length_capped: bool    # length had to be cut to make the timing
    too_slow: bool         # could not be done in time even after cutting


def solve_stroke(length: float, duration: float,
                 accel_nominal: float = MOVE_ACCEL,
                 accel_max: float = ACCEL_MAX,
                 speed_max: float = 0.6) -> MotionSolution:
    """
    Find the moveL parameters that cover `length` in `duration` seconds.

    Duration falls monotonically with speed and bottoms out at the triangular
    profile 2*sqrt(L/a), so the note is playable iff a >= 4L/T^2. Given a
    feasible accel, the speed that hits T exactly is the smaller root of
    v^2 - a*T*v + a*L = 0.

    When 4L/T^2 exceeds accel_max the note is too short for that much bow at
    any achievable acceleration, and the LENGTH is cut instead of the timing
    being missed — rhythm is preserved, the note comes out a little quieter,
    and length_capped is set so the planner can report the shortfall.
    """
    length = max(float(length), 1e-4)
    duration = max(float(duration), 1e-3)

    capped = False

    # Minimum acceleration that can cover this length in this time at all.
    accel_required = 4.0 * length / (duration * duration)

    if accel_required > accel_max:
        # Not enough acceleration available: shorten the stroke instead of
        # overrunning the beat. L = a*T^2/4 is exactly what fits.
        length = accel_max * duration * duration / 4.0
        capped = True
        accel_required = 4.0 * length / (duration * duration)

    # Prefer the acceleration that holds the peak near PEAK_SPEED_RATIO x mean
    # rather than the bare minimum that fits, but never go below the
    # pendant-verified nominal or above the ceiling.
    ratio = PEAK_SPEED_RATIO
    accel_for_ratio = (ratio * ratio / (ratio - 1.0)) * length / (duration * duration)
    accel = float(np.clip(max(accel_required, accel_for_ratio),
                          accel_nominal, accel_max))

    # Smaller root of v^2 - a*T*v + a*L = 0.
    disc = accel * accel * duration * duration - 4.0 * accel * length
    if disc < 0.0:
        # Only reachable through floating point slop at the boundary; fall back
        # to the triangular profile, which is that boundary.
        speed = math.sqrt(length * accel)
    else:
        speed = (accel * duration - math.sqrt(disc)) / 2.0

    too_slow = False
    if speed > speed_max:
        # Would need to cruise faster than allowed. Cut the length to what
        # speed_max can deliver inside the note.
        usable = duration - speed_max / accel
        if usable > 0.0:
            length = min(length, speed_max * usable)
            capped = True
        speed = speed_max
        too_slow = trapezoid_duration(length, speed, accel) > duration * 1.02

    speed = max(speed, 1e-3)
    achieved = trapezoid_duration(length, speed, accel)

    return MotionSolution(
        length=length,
        speed=speed,
        accel=accel,
        duration=achieved,
        mean_speed=length / achieved if achieved > 0 else 0.0,
        length_capped=capped,
        too_slow=too_slow,
    )


# ══════════════════════════════════════════════════════════════════
# Score model
# ══════════════════════════════════════════════════════════════════

# ── Articulation ──────────────────────────────────────────────────
# Articulations that SEPARATE the notes under a slur — portato, or louré.
#
# A plain slur on an open A really is one continuous sound: the notes share a
# pitch and there is no left-hand change, so nothing marks where one ends and
# the next begins. Dots under that slur mean something different, though. They
# ask for one bow DIRECTION with the notes articulated inside it, and on an
# open string that separation is the only thing making them audible as separate
# notes at all. Merging such a group into one stroke throws the dots away.
#
# So: slur alone -> one stroke. Slur plus any of these -> keep the notes apart
# but hold the bow in one direction across the whole group.
SEPARATING_ARTICULATIONS = frozenset({
    "staccato",         # the dot
    "staccatissimo",    # the wedge
    "detached-legato",  # MusicXML's own name for a dot under a slur
})

# Fraction of its written time a marked note actually sounds for; the rest is
# silence with the bow stopped. This is what makes the separation audible —
# without it two adjacent strokes in the same direction run together.
ARTICULATION_DURATION = {
    "staccatissimo":   0.35,
    "staccato":        0.55,
    "detached-legato": 0.80,   # portato: a pulse, barely detached
    "tenuto":          0.95,   # full length, the merest lift
}

# Note length at and above which a written articulation is taken literally.
#
# Applied as a flat fraction, staccato removes 45% of EVERY marked note however
# short. On challengepiece's 0.25 s staccato eighths that is 112 ms of silence
# — the notes come out as detached blips next to a fully sustained quarter, and
# the pulse reads as uneven rather than articulated.
#
# Players do not work that way. Separation is closer to absolute than
# proportional: as the notes get faster the dot becomes an accent, not a gap.
# So the silence a mark asks for is scaled by duration/ARTICULATION_REF, capped
# at 1. A 0.5 s staccato quarter is unchanged (0.55 as written); a 0.25 s
# eighth keeps ~78% of its length instead of 55%.
#
# Raise it to detach short notes more, lower it toward 0 to take every mark
# literally (the previous behaviour). Exposed as --articulation-ref.
ARTICULATION_REF = 0.5


@dataclass
class ScoreNote:
    """
    One note as read from the score, before any robot considerations.

    bow_dir is None unless the score or the user actually specified one; the
    planner treats None as "your call" and alternates. volume is on the same
    0..1 axis as velocity_to_volume.
    """
    index: int
    onset: float                       # s from the start of the piece
    duration: float                    # s, sounding length
    volume: float                      # 0..1
    dynamic: str                       # nearest marking, for display
    bow_dir: str | None = None         # 'down' | 'up' | None
    bow_dir_source: str = "auto"       # auto | score | metre | annotation
    volume_source: str = "midi"        # midi | score | annotation
    volume_end: float | None = None    # for within-note hairpins
    slur_id: int | None = None         # notes sharing an id share one bow
    articulations: frozenset = frozenset()
    measure: int = 0
    beat: float = 0.0                  # position within the measure, in beats
    is_strong_beat: bool = False
    # Per-note depth override, in metres, from --annotations. Depth is normally
    # derived from volume, which means it cannot be adjusted without also
    # changing the loudness. Set here it wins outright, so a note can be
    # lightened for tone without being made quieter.
    depth_override: float | None = None

    @property
    def end(self) -> float:
        return self.onset + self.duration


# ── MIDI ──────────────────────────────────────────────────────────

def parse_midi(midi_path: str, tempo_scale: float = 1.0) -> tuple[list[ScoreNote], dict]:
    """
    Read note events from a MIDI file, all treated as open A.

    Pitch is ignored by design — the piece is assumed to be all open A string,
    so every note becomes the same stroke and only rhythm and dynamics matter.
    A warning is printed if the file contains pitches other than A, since that
    means the piece is not actually playable as written on one open string.

    Dynamics come from two places:
      - note velocity, which carries the written dynamic mark
      - CC2, which MuseScore writes as a continuous curve, so hairpins survive
        the export. Sampling CC2 at the start and end of each note recovers a
        crescendo or diminuendo happening WITHIN a single long note, which
        velocity alone cannot express.

    Absolute times are resolved through every tempo change in the file, not
    just the first one.
    """
    import mido

    mid = mido.MidiFile(str(midi_path))
    ticks_per_beat = mid.ticks_per_beat

    # Merge all tracks onto one absolute-tick timeline.
    events = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            events.append((tick, msg))
    events.sort(key=lambda item: item[0])

    # Tempo map, so ticks -> seconds is right across tempo changes.
    tempo_changes = [(0, 500000)]                 # default 120 bpm
    for tick, msg in events:
        if msg.type == "set_tempo":
            if tick == 0:
                tempo_changes[0] = (0, msg.tempo)
            else:
                tempo_changes.append((tick, msg.tempo))

    def tick_to_seconds(target_tick: int) -> float:
        seconds = 0.0
        prev_tick, prev_tempo = tempo_changes[0]
        for change_tick, tempo in tempo_changes[1:]:
            if change_tick >= target_tick:
                break
            seconds += (change_tick - prev_tick) * prev_tempo / (ticks_per_beat * 1e6)
            prev_tick, prev_tempo = change_tick, tempo
        seconds += (target_tick - prev_tick) * prev_tempo / (ticks_per_beat * 1e6)
        return seconds * tempo_scale

    # Time signature, for beat/measure positions (needed by rule-of-downbow).
    numerator, denominator = 4, 4
    for tick, msg in events:
        if msg.type == "time_signature":
            numerator, denominator = msg.numerator, msg.denominator
            break
    beats_per_measure = numerator * (4.0 / denominator)

    # CC2 curve (MuseScore's expression channel).
    cc2 = [(tick, msg.value) for tick, msg in events
           if msg.type == "control_change" and msg.control == 2]

    def cc2_at(tick: int):
        if not cc2:
            return None
        value = None
        for change_tick, val in cc2:
            if change_tick > tick:
                break
            value = val
        return value if value is not None else cc2[0][1]

    # Note on/off pairing.
    raw_notes = []
    pending = {}
    pitches = set()
    for tick, msg in events:
        if msg.type == "note_on" and msg.velocity > 0:
            pending.setdefault((msg.channel, msg.note), []).append((tick, msg.velocity))
            pitches.add(msg.note)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if pending.get(key):
                start_tick, velocity = pending[key].pop(0)
                if tick > start_tick:
                    raw_notes.append((start_tick, tick, velocity))
    raw_notes.sort()

    notes = []
    for start_tick, end_tick, velocity in raw_notes:
        onset = tick_to_seconds(start_tick)
        duration = tick_to_seconds(end_tick) - onset
        if duration < 1e-3:
            continue

        # Velocity sets the level; CC2 refines it and gives the within-note shape.
        volume = velocity_to_volume(velocity)
        volume_end = None
        start_cc = cc2_at(start_tick)
        end_cc = cc2_at(max(end_tick - 1, start_tick))
        if start_cc is not None and end_cc is not None and abs(end_cc - start_cc) >= 4:
            volume = velocity_to_volume(start_cc)
            volume_end = velocity_to_volume(end_cc)

        beat_position = start_tick / ticks_per_beat
        measure = int(beat_position // beats_per_measure)
        beat_in_measure = beat_position - measure * beats_per_measure

        notes.append(ScoreNote(
            index=len(notes),
            onset=onset,
            duration=duration,
            volume=volume,
            dynamic=velocity_to_dynamic(volume_to_velocity(volume)),
            volume_end=volume_end,
            measure=measure,
            beat=beat_in_measure,
            is_strong_beat=abs(beat_in_measure - round(beat_in_measure)) < 0.05
                           and _is_strong(round(beat_in_measure), numerator, denominator),
        ))

    meta = {
        "source": str(midi_path),
        "format": "midi",
        "time_signature": f"{numerator}/{denominator}",
        "beats_per_measure": beats_per_measure,
        "tempo_bpm": round(60e6 / tempo_changes[0][1] / tempo_scale, 2),
        "n_notes": len(notes),
        "cc2_events": len(cc2),
        "pitches": sorted(pitches),
        "duration": max((n.end for n in notes), default=0.0),
    }
    return notes, meta


def _is_strong(beat: int, numerator: int, denominator: int) -> bool:
    """
    Strong beats for the rule of the down-bow.

    Beat 1 is always strong. In compound metres (6/8, 9/8, 12/8) the strong
    beats are the starts of the dotted-beat groups; in simple duple/quadruple
    the half-bar is secondary-strong; in 3/4 only beat 1.
    """
    if beat == 0:
        return True
    if denominator == 8 and numerator in (6, 9, 12):
        return beat % 3 == 0
    if numerator == 4:
        return beat == 2
    return False


# ── MusicXML ──────────────────────────────────────────────────────

_MUSICXML_DYNAMICS = {name for name, _ in DYNAMIC_VELOCITIES} | {
    "sf", "sfz", "fp", "rf", "rfz", "sffz", "fz"
}


def _strip_namespace(root):
    for element in root.iter():
        if isinstance(element.tag, str) and "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]
    return root


def _read_musicxml_root(path: Path):
    """Read .musicxml/.xml directly, or pull the root score out of a .mxl zip."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            target = None
            if "META-INF/container.xml" in archive.namelist():
                container = _strip_namespace(
                    ElementTree.fromstring(archive.read("META-INF/container.xml")))
                rootfile = container.find(".//rootfile")
                if rootfile is not None:
                    target = rootfile.get("full-path")
            if target is None:
                candidates = [n for n in archive.namelist()
                              if n.endswith((".xml", ".musicxml"))
                              and not n.startswith("META-INF")]
                if not candidates:
                    raise ValueError(f"No score found inside {path}")
                target = candidates[0]
            return _strip_namespace(ElementTree.fromstring(archive.read(target)))
    return _strip_namespace(ElementTree.parse(path).getroot())


def _ending_numbers(element) -> set:
    """The pass numbers a volta applies to, from number="1" or number="1,2"."""
    raw = (element.get("number") or "").replace(" ", "")
    return {int(token) for token in raw.split(",") if token.isdigit()}


def expand_repeat_order(measures) -> tuple[list[int], list[str]]:
    """
    Work out the order the measures are actually played in, following repeat
    barlines and voltas.

    This matters whenever a MusicXML file is compared against a MIDI export of
    the same piece: MuseScore's MIDI export PLAYS THE REPEATS OUT, writing the
    repeated bars twice, while the MusicXML keeps one written copy behind a
    repeat barline. Left unexpanded, a piece with a single repeat yields half
    as many notes from the MusicXML as from the MIDI, and every bowing after
    the repeat lines up against the wrong note.

    Handles forward/backward repeat barlines, repeats taken more than twice
    (times="3"), and numbered endings. Does NOT handle da capo, dal segno or
    coda jumps; those are reported so the mismatch is visible rather than
    silent.

    Returns the play order as measure indices, plus any notes for the caller.
    """
    order: list[int] = []
    notes: list[str] = []

    index = 0
    repeat_start = 0
    pass_number = 1
    times_taken: dict[int, int] = {}
    # True only on the iteration immediately after jumping back to a repeat
    # start. Without it, re-entering the measure that carries the forward
    # repeat would reset pass_number to 1 every time round, so a first-time
    # ending would be chosen on every pass and the second ending never played.
    jumped_back = False

    # Backstop against a malformed repeat structure looping forever.
    guard, guard_limit = 0, len(measures) * 64 + 1000

    while index < len(measures):
        guard += 1
        if guard > guard_limit:
            notes.append("repeat structure did not resolve; played straight through")
            return list(range(len(measures))), notes

        measure = measures[index]

        for sound in measure.findall(".//sound"):
            if any(sound.get(k) for k in ("dacapo", "dalsegno", "tocoda")):
                notes.append("score uses da capo / dal segno / coda, which is not "
                             "expanded — note counts may not match the MIDI")

        if measure.find('.//repeat[@direction="forward"]') is not None:
            repeat_start = index
            if not jumped_back:
                pass_number = 1
        jumped_back = False

        # A volta that does not apply on this pass is skipped, along with
        # everything up to the next volta.
        starts = [e for e in measure.findall(".//ending")
                  if e.get("type") == "start"]
        if starts:
            applies = _ending_numbers(starts[0])
            if applies and pass_number not in applies:
                next_index = index + 1
                while next_index < len(measures):
                    if any(e.get("type") == "start"
                           for e in measures[next_index].findall(".//ending")):
                        break
                    next_index += 1
                index = next_index
                continue

        order.append(index)

        backward = measure.find('.//repeat[@direction="backward"]')
        if backward is not None:
            try:
                times = max(2, int(backward.get("times", 2)))
            except ValueError:
                times = 2
            taken = times_taken.get(index, 1)
            if taken < times:
                times_taken[index] = taken + 1
                pass_number += 1
                index = repeat_start
                jumped_back = True
                continue

        index += 1

    return order, notes


def parse_musicxml(path: str, tempo_scale: float = 1.0,
                   expand_repeats: bool = True) -> tuple[list[ScoreNote], dict]:
    """
    Read a MusicXML score, including the things MIDI cannot carry.

    This is the answer to "can the bowings written in the music be translated
    automatically": in MusicXML they can, because they are actually there.
    Extracted here:

        <technical><down-bow/> / <up-bow/>   the engraved bowings
        <notations><slur type="start|stop">  one bow across the group
        <direction><dynamics><f/> etc.       written dynamic marks
        <direction><wedge type="crescendo">  hairpins, applied across the span
        <sound tempo="...">                  tempo, including changes

    Only the first part is read (a solo cello line). Chords and rests are
    skipped; tied notes are merged into one sounding note.
    """
    root = _read_musicxml_root(Path(path))

    part = root.find(".//part")
    if part is None:
        raise ValueError(f"{path} has no <part> — is it really MusicXML?")

    divisions = 1
    tempo_bpm = 120.0
    numerator, denominator = 4, 4
    current_dynamic_volume = velocity_to_volume(DYNAMIC_BY_NAME["mf"])
    # The opening dynamic is frequently engraved AFTER the first note of the
    # bar rather than before it, so notes read before the first explicit mark
    # would otherwise be stuck on the mf default. Remember which ones those
    # were and back-fill them once the real opening dynamic is known.
    first_explicit_volume = None
    pending_default: list[int] = []
    wedge = None                       # (kind, start_beat, start_volume)

    notes: list[ScoreNote] = []
    position = 0.0                     # quarter notes from the start
    measure_start = 0.0
    n_grace_skipped = [0]
    slur_counter = 0
    open_slurs: dict[str, int] = {}
    n_bowings = 0
    wedge_spans: list[tuple[float, float, str, float]] = []

    all_measures = part.findall("measure")
    if expand_repeats:
        play_order, repeat_notes = expand_repeat_order(all_measures)
    else:
        play_order, repeat_notes = list(range(len(all_measures))), []

    for measure_index, source_index in enumerate(play_order):
        measure = all_measures[source_index]
        measure_start = position

        for element in measure:
            tag = element.tag

            if tag == "attributes":
                if element.find("divisions") is not None:
                    divisions = int(element.findtext("divisions", "1"))
                time_element = element.find("time")
                if time_element is not None:
                    numerator = int(time_element.findtext("beats", "4"))
                    denominator = int(time_element.findtext("beat-type", "4"))

            elif tag == "direction":
                sound = element.find(".//sound")
                if sound is not None and sound.get("tempo"):
                    tempo_bpm = float(sound.get("tempo"))

                dynamics = element.find(".//dynamics")
                if dynamics is not None:
                    for child in dynamics:
                        if child.tag in DYNAMIC_BY_NAME:
                            current_dynamic_volume = velocity_to_volume(
                                DYNAMIC_BY_NAME[child.tag])
                            if first_explicit_volume is None:
                                first_explicit_volume = current_dynamic_volume
                            break
                        if child.tag in _MUSICXML_DYNAMICS:
                            current_dynamic_volume = velocity_to_volume(
                                DYNAMIC_BY_NAME["f"])
                            if first_explicit_volume is None:
                                first_explicit_volume = current_dynamic_volume
                            break

                wedge_element = element.find(".//wedge")
                if wedge_element is not None:
                    kind = wedge_element.get("type")
                    if kind in ("crescendo", "diminuendo"):
                        wedge = (kind, position)
                    elif kind == "stop" and wedge is not None:
                        wedge_spans.append((wedge[1], position, wedge[0]))
                        wedge = None

            elif tag == "backup":
                position -= float(element.findtext("duration", "0")) / divisions
            elif tag == "forward":
                position += float(element.findtext("duration", "0")) / divisions

            elif tag == "note":
                # Grace notes carry no <duration> and steal no time from the
                # bar. Reading one would add a zero-length note that the
                # planner then has to make a stroke for, and would push every
                # later note out of step with the MIDI. On an open A an
                # ornament is inaudible anyway — there is no pitch to ornament.
                if element.find("grace") is not None:
                    n_grace_skipped[0] += 1
                    continue

                length = float(element.findtext("duration", "0")) / divisions
                is_chord = element.find("chord") is not None
                is_rest = element.find("rest") is not None

                if is_chord:
                    continue           # same onset as the previous note; skip
                if is_rest:
                    position += length
                    continue

                tie_stop = any(t.get("type") == "stop"
                               for t in element.findall("tie"))

                if tie_stop and notes:
                    notes[-1].duration += length * 60.0 / tempo_bpm * tempo_scale
                    position += length
                    continue

                bow_dir = None
                articulations: set[str] = set()
                notations = element.find("notations")
                if notations is not None:
                    if notations.find(".//down-bow") is not None:
                        bow_dir = "down"
                    elif notations.find(".//up-bow") is not None:
                        bow_dir = "up"
                    if bow_dir:
                        n_bowings += 1

                    # Dots, wedges, lines. These are what tell a slurred group
                    # to be played as separate notes rather than one sound.
                    for group in notations.findall("articulations"):
                        for mark in group:
                            if isinstance(mark.tag, str):
                                articulations.add(mark.tag)

                    for slur in notations.findall("slur"):
                        number = slur.get("number", "1")
                        if slur.get("type") == "start":
                            slur_counter += 1
                            open_slurs[number] = slur_counter

                slur_id = next(iter(open_slurs.values()), None) if open_slurs else None

                beat_in_measure = position - measure_start
                if first_explicit_volume is None:
                    pending_default.append(len(notes))
                notes.append(ScoreNote(
                    index=len(notes),
                    onset=position * 60.0 / tempo_bpm * tempo_scale,
                    duration=length * 60.0 / tempo_bpm * tempo_scale,
                    volume=current_dynamic_volume,
                    dynamic=velocity_to_dynamic(
                        volume_to_velocity(current_dynamic_volume)),
                    bow_dir=bow_dir,
                    bow_dir_source="score" if bow_dir else "auto",
                    volume_source="score",
                    slur_id=slur_id,
                    articulations=frozenset(articulations),
                    measure=measure_index,
                    beat=beat_in_measure,
                    is_strong_beat=abs(beat_in_measure - round(beat_in_measure)) < 0.05
                                   and _is_strong(int(round(beat_in_measure)),
                                                  numerator, denominator),
                ))

                if notations is not None:
                    for slur in notations.findall("slur"):
                        if slur.get("type") == "stop":
                            open_slurs.pop(slur.get("number", "1"), None)

                position += length

    # Back-fill the notes that were read before the piece's first dynamic mark.
    # Without this a score whose opening f sits just after the first note leaves
    # exactly one note on the mf default — invisible on its own, but enough to
    # look like a dynamic range that isn't there.
    if first_explicit_volume is not None and pending_default:
        for index in pending_default:
            notes[index].volume = first_explicit_volume
            notes[index].dynamic = velocity_to_dynamic(
                volume_to_velocity(first_explicit_volume))

    # Apply hairpins across the spans they cover, in quarter notes.
    #
    # The starting level is taken from the FIRST NOTE inside the span, not from
    # whatever dynamic happened to be current when the <wedge> element was
    # read. Engravers put the dynamic mark and the hairpin at the same musical
    # point in either order, so snapshotting at wedge-start silently loses the
    # mark whenever it is written second — a "p <" would come out starting at
    # the previous forte.
    for start_beat, end_beat, kind in wedge_spans:
        span = max(end_beat - start_beat, 1e-6)
        inside = [n for n in notes
                  if start_beat <= n.onset * tempo_bpm / 60.0 / tempo_scale <= end_beat]
        if not inside:
            continue

        start_volume = inside[0].volume
        target = min(start_volume + 0.25, 1.0) if kind == "crescendo" \
            else max(start_volume - 0.25, 0.0)

        for note in inside:
            beat_position = note.onset * tempo_bpm / 60.0 / tempo_scale
            t = (beat_position - start_beat) / span
            note.volume = start_volume + t * (target - start_volume)
            note.dynamic = velocity_to_dynamic(volume_to_velocity(note.volume))

    meta = {
        "source": str(path),
        "format": "musicxml",
        "time_signature": f"{numerator}/{denominator}",
        "tempo_bpm": tempo_bpm / tempo_scale,
        "n_notes": len(notes),
        "n_bowings": n_bowings,
        "n_slurs": slur_counter,
        "n_hairpins": len(wedge_spans),
        "n_measures_written": len(all_measures),
        "n_measures_played": len(play_order),
        "repeats_expanded": len(play_order) > len(all_measures),
        "n_grace_skipped": n_grace_skipped[0],
        "score_notes": list(repeat_notes),
        "duration": max((n.end for n in notes), default=0.0),
    }
    return notes, meta


def describe_dynamic_range(notes: list[ScoreNote]) -> dict:
    """
    Report the dynamic range the score writes, WITHOUT changing anything.

    This is the default. It exists because knowing a piece has no dynamics is
    worth saying out loud — but see expand_dynamic_range below for why the
    range is deliberately not stretched to fit.
    """
    values = [note.volume for note in notes]
    values += [note.volume_end for note in notes if note.volume_end is not None]
    if not values:
        return {"expanded": False, "reason": "no notes"}

    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))
    if high - low < 0.05:
        return {"expanded": False,
                "reason": "the score has effectively one dynamic",
                "written_low": low, "written_high": high}
    return {"expanded": False, "described": True,
            "written_low": low, "written_high": high}


def expand_dynamic_range(notes: list[ScoreNote]) -> dict:
    """
    Stretch the dynamics the piece actually uses across the full usable range.

    OFF BY DEFAULT — it measurably hurts tone. Enable with --expand-dynamics.

    The reasoning was sound on paper: the instrument has only ~8.8 dB between
    its quietest and loudest usable bow speed, real music sits in the middle of
    the ppp..fff axis, so a p-to-f piece was using barely a third of the range.
    Stretching it to fit gave the biggest dynamic contrast the hardware can
    produce.

    In practice it sounds worse. Reaching the bottom of the range means the
    slowest bow speed at the lightest depth simultaneously, and that corner is
    where the string stops speaking cleanly — long quiet notes come out thin
    and unsteady rather than soft. A narrower range of good tone beats a wider
    range that includes bad tone, so the literal mapping is the default and
    ordinary music stays in the well-behaved middle of the calibration.

    Worth revisiting if the usable envelope is ever recalibrated — more depth
    at low speed is the classical way to keep a quiet note speaking, and the
    current 2 mm ceiling leaves no room to trade one against the other.

    Volume is read on an absolute ppp..fff axis, but almost no piece uses all
    of it — real music lives between p and f, which is the middle 43% of that
    axis. Mapped straight through, a p-to-f piece asks for 0.138..0.207 m/s of
    bow, and by the speed calibration that is 3.5 dB. Played back to back with
    everything else going on, 3.5 dB reads as no dynamics at all.

    The instrument only has about 8.8 dB between its quietest and loudest
    usable bow speed, against the 30-plus dB of a cello played by hand. Since
    the score's intent is already being compressed that hard, there is no
    reason to also leave part of the range unused: whatever spread the piece
    writes gets mapped onto the whole of it, so p..f uses every decibel there
    is rather than a third of them.

    A piece with one dynamic mark, or none, has no spread to stretch. Nothing
    is invented for it — the caller is told instead, because the fix there is
    to write some dynamics in, not to have them synthesised.
    """
    values = [note.volume for note in notes]
    values += [note.volume_end for note in notes if note.volume_end is not None]
    if not values:
        return {"expanded": False, "reason": "no notes"}

    # Percentiles rather than min/max, so one stray note cannot define the
    # range. A single note left on the default level — which happens whenever
    # an engraver puts the opening dynamic after the first note — would
    # otherwise look like a whole extra dynamic step and get stretched into a
    # dramatic contrast that is nowhere in the score.
    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))
    spread = high - low

    # Under about a third of a dynamic step there is nothing meaningful to
    # stretch, and scaling it up would just amplify rounding.
    if spread < 0.05:
        return {
            "expanded": False,
            "reason": "the score has effectively one dynamic",
            "written_low": low,
            "written_high": high,
        }

    def rescale(value):
        return float(np.clip((value - low) / spread, 0.0, 1.0))

    for note in notes:
        note.volume = rescale(note.volume)
        if note.volume_end is not None:
            note.volume_end = rescale(note.volume_end)
        note.dynamic = velocity_to_dynamic(volume_to_velocity(note.volume))

    return {
        "expanded": True,
        "written_low": low,
        "written_high": high,
        "gain_db": (speed_to_dbfs(volume_to_speed(1.0))
                    - speed_to_dbfs(volume_to_speed(0.0)))
                   - (speed_to_dbfs(volume_to_speed(high))
                      - speed_to_dbfs(volume_to_speed(low))),
    }


def merge_score_into_midi(midi_notes: list[ScoreNote],
                          score_notes: list[ScoreNote]) -> int:
    """
    Overlay a MusicXML score's bowings and slurs onto MIDI-derived notes.

    Only the things MIDI cannot express are taken from the MusicXML. Timing
    stays with the MIDI, whose tempo map is authoritative, and so do dynamics:
    the exported velocities plus the CC2 curve carry more detail than the
    written marks do, since they already have the hairpins interpolated.

    Matching is positional, which is correct when both files came from the same
    engraving. If the counts differ only the common prefix is merged, and the
    caller is told how many notes matched.
    """
    merged = 0
    for midi_note, score_note in zip(midi_notes, score_notes):
        if score_note.bow_dir:
            midi_note.bow_dir = score_note.bow_dir
            midi_note.bow_dir_source = "score"
        if score_note.slur_id is not None:
            midi_note.slur_id = score_note.slur_id
        merged += 1
    return merged


# ── Per-note annotations ──────────────────────────────────────────

def load_annotations(path: str, notes: list[ScoreNote]) -> int:
    """
    Apply per-note user overrides from a JSON file.

    Format — a list of objects, or an object with a "notes" list. Each entry
    keys on a note index (0-based, in score order) and may set a bowing, a
    volume, or both:

        {"notes": [
          {"index": 0,  "bow": "down", "volume": "ff"},
          {"index": 1,  "bow": "down"},
          {"index": 4,  "volume": 0.25},
          {"index": 12, "volume": "pp", "volume_end": "mf"}
        ]}

    "bow" is "down" or "up". "volume" is a dynamic marking ("pp".."fff"), a
    0..1 number, or a MIDI velocity (any number above 1). "volume_end" makes
    the note swell or fade across its own length.

    A forced bowing is honoured exactly, even where it costs a retake — that
    is the point of forcing it.
    """
    data = json.loads(Path(path).read_text())
    entries = data["notes"] if isinstance(data, dict) else data
    by_index = {note.index: note for note in notes}

    applied = 0
    for entry in entries:
        index = int(entry["index"])
        note = by_index.get(index)
        if note is None:
            print(f"  annotation for note {index} ignored — the score has "
                  f"{len(notes)} notes")
            continue

        if "bow" in entry:
            bow = str(entry["bow"]).lower()
            if bow not in ("down", "up"):
                raise ValueError(f"note {index}: bow must be 'down' or 'up', got {bow!r}")
            note.bow_dir = bow
            note.bow_dir_source = "annotation"

        if "depth_mm" in entry:
            # Signed, same convention as the recording script: negative is
            # outward/lighter, positive presses in. Clamped by apply_depth()
            # to the safety envelope regardless of what is written here.
            note.depth_override = float(entry["depth_mm"]) / 1000.0

        if "volume" in entry:
            note.volume = _parse_volume(entry["volume"])
            note.volume_source = "annotation"
            # Drop any hairpin inherited from the MIDI's CC2 curve unless this
            # entry asks for one. Otherwise setting "volume": "ff" would be
            # quietly averaged against a crescendo the user never asked for,
            # and the note would come out well short of ff.
            if "volume_end" not in entry:
                note.volume_end = None

        if "volume_end" in entry:
            note.volume_end = _parse_volume(entry["volume_end"])
            note.volume_source = "annotation"

        note.dynamic = velocity_to_dynamic(volume_to_velocity(note.volume))
        applied += 1

    return applied


def _parse_volume(value) -> float:
    """Accept a dynamic marking, a 0..1 volume, or a MIDI velocity."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in DYNAMIC_BY_NAME:
            raise ValueError(f"unknown dynamic {value!r}; "
                             f"expected one of {sorted(DYNAMIC_BY_NAME)}")
        return velocity_to_volume(DYNAMIC_BY_NAME[key])
    number = float(value)
    return velocity_to_volume(number) if number > 1.0 else float(np.clip(number, 0.0, 1.0))


# ── Bowing assignment ─────────────────────────────────────────────

def assign_bowings(notes: list[ScoreNote], rule: str) -> None:
    """
    Fill in bow_dir wherever nothing has specified one.

    'alternate'        leave it to the planner, which alternates and flips
                       whenever the bow would run out. Nothing is forced, so
                       no retakes are ever needed for bowing reasons.

    'rule-of-downbow'  the actual convention: strong beats take a down-bow, and
                       everything between them alternates. This IS a real
                       auto-translation from the score — the metre it needs is
                       already in the MIDI's time signature. It deliberately
                       produces repeated down-bows at bar lines, which the
                       planner turns into retakes, exactly as a player would.

    Notes under one slur share a bowing: the first note's direction wins.
    """
    if rule == "rule-of-downbow":
        previous = None
        for note in notes:
            if note.bow_dir is not None:
                previous = note.bow_dir
                continue
            if note.is_strong_beat:
                note.bow_dir = "down"
                note.bow_dir_source = "metre"
            elif previous is not None:
                note.bow_dir = "up" if previous == "down" else "down"
                note.bow_dir_source = "metre"
            else:
                note.bow_dir = "down"
                note.bow_dir_source = "metre"
            previous = note.bow_dir
    elif rule != "alternate":
        raise ValueError(f"unknown bowing rule {rule!r}")

    # One bow per slur group.
    group_direction: dict[int, str | None] = {}
    for note in notes:
        if note.slur_id is None:
            continue
        if note.slur_id not in group_direction:
            group_direction[note.slur_id] = note.bow_dir
        elif group_direction[note.slur_id] is not None:
            note.bow_dir = group_direction[note.slur_id]
            note.bow_dir_source = "slur"


def portato_groups(notes: list[ScoreNote]) -> set:
    """
    Slur groups carrying dots, wedges or portato marks — one bow, but separate
    notes. These are the groups merge_slurs must leave alone.
    """
    return {note.slur_id for note in notes
            if note.slur_id is not None
            and (note.articulations & SEPARATING_ARTICULATIONS)}


def merge_slurs(notes: list[ScoreNote],
                merge: bool = True) -> tuple[list[ScoreNote], int, int]:
    """
    Collapse each LEGATO slur group into one sustained note.

    On an open A a plain slur genuinely is one continuous sound: the notes
    under it share a pitch and there is no left-hand change, so nothing
    distinguishes them to the ear. Playing them as separate strokes would
    insert bow changes the score explicitly asks not to have.

    A slur carrying staccato dots is the opposite case and is NOT merged. That
    is portato: one bow direction, notes separated inside it. Merging it would
    turn two articulated notes into one long tone and silently discard the
    dots. Those groups stay as separate notes; the planner holds them to a
    single bow direction and ARTICULATION_DURATION puts a real gap between
    them, so they speak as two notes drawn in one bow.

    Returns the notes, how many groups were merged, and how many were kept
    apart as portato.
    """
    if not any(note.slur_id is not None for note in notes):
        return notes, 0, 0

    separated = portato_groups(notes)

    # merge=False treats EVERY slur as portato: the notes stay separate and the
    # planner still holds them to one bow direction. The merging default argues
    # that on an open A a plain slur is one continuous sound, which is true of
    # the sound but throws the written note away — a slur ending on a short
    # note loses that note entirely. When the score's articulation matters more
    # than the letter of the legato, this keeps it. See --no-merge-slurs.
    if not merge:
        separated = {note.slur_id for note in notes if note.slur_id is not None}

    merged: list[ScoreNote] = []
    merged_groups: set = set()
    for note in notes:
        if (merged and note.slur_id is not None
                and note.slur_id not in separated
                and merged[-1].slur_id == note.slur_id):
            previous = merged[-1]
            previous.duration = note.end - previous.onset
            previous.volume_end = note.volume_end if note.volume_end is not None \
                else note.volume
            merged_groups.add(note.slur_id)
            continue
        merged.append(note)

    for i, note in enumerate(merged):
        note.index = i
    return merged, len(merged_groups), len(separated)


# ══════════════════════════════════════════════════════════════════
# The bow planner
# ══════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    """One moveL. A stroke is several of these only when it swells."""
    u_start: float
    u_end: float
    speed: float
    accel: float
    depth: float
    duration: float


@dataclass
class Stroke:
    """A planned bow stroke, plus everything needed to explain it afterwards."""
    note_index: int
    onset: float                     # s, scheduled start
    duration: float                  # s, planned playing time
    direction: str                   # 'down' | 'up'
    direction_source: str
    u_start: float
    u_end: float
    length: float                    # m of bow used
    mean_speed: float                # m/s — sets the loudness
    depth: float                     # m, signed
    volume_target: float             # what the score asked for
    volume_actual: float             # what the stroke will actually deliver
    dynamic: str
    segments: list[Segment] = field(default_factory=list)
    retake_from: float | None = None  # u the bow lifts from, if a retake
    retake_time: float = 0.0          # s the retake needs
    events: list[str] = field(default_factory=list)
    part_of: int | None = None        # note index, when a note was split
    warnings: list[str] = field(default_factory=list)


class BowPlanner:
    """
    Turn score notes into strokes that fit on a real bow.

    The planner holds one piece of state — u, where the bow is — and answers
    one question per note: does the stroke the music asks for fit in what is
    left, and if not, what is the least damaging way out? The escalation is
    ordered by how much it costs musically: shrink (a shade quieter), flip (an
    inaudible change of bowing), retake (a silent gap), split (an audible bow
    change mid-note).
    """

    # Below this many metres a stroke is not worth playing; treat the bow as
    # exhausted in that direction.
    MIN_STROKE = 0.008

    # Gap between notes, as a fraction of the note, that gets absorbed into the
    # stroke rather than left as a stop. MuseScore writes a 1-beat note as
    # 0.998 beats, and honouring that literally would put a pointless 1 ms halt
    # before every bow change.
    GAP_ABSORB_FRAC = 0.25

    # Shortest silence worth repositioning the bow in. A rest is the one place
    # a reposition costs nothing musically — the bow is off the string and
    # nothing is sounding — so this is deliberately short. Below it the move
    # could not complete anyway.
    MIN_RESET_GAP = 0.25

    # How many notes ahead to treat as one phrase. A phrase runs until the next
    # rest long enough to reset in, so this is only a tractability backstop for
    # music with no rests at all.
    MAX_LOOKAHEAD = 128

    # Smallest reposition worth making, as a fraction of the bow (~10 mm).
    #
    # A reset is never just a slide: it is a lift clear of the string, a
    # traverse and a set-down, so it costs two vertical moves however short the
    # horizontal one is. Without a floor here the planner would happily spend a
    # full second of robot motion to shift the bow by a millimetre, which buys
    # nothing and puts an extra lift on and off the string for no reason.
    MIN_RESET_TRAVEL = 0.02

    # A within-note volume change of at least this much (0..1) is worth
    # rendering as a real swell rather than averaging away.
    SWELL_THRESHOLD = 0.06
    SWELL_SEGMENTS = 4
    SWELL_MIN_DURATION = 0.5          # s; shorter notes are not worth segmenting

    def __init__(self, start_u: float = 0.5, enable_swell: bool = True,
                 accel_max: float = ACCEL_MAX,
                 retake_speed: float = RETAKE_SPEED,
                 retake_accel: float = None,
                 lookahead: bool = True,
                 articulation_ref: float = ARTICULATION_REF,
                 depth_offset: float = 0.0,
                 speed_scale: float = 1.0):
        retake_accel = MOVE_ACCEL if retake_accel is None else retake_accel
        self.u = float(np.clip(start_u, U_MIN, U_MAX))
        self.enable_swell = enable_swell
        self.articulation_ref = articulation_ref
        self.depth_offset = depth_offset
        self.speed_scale = speed_scale
        self.accel_max = accel_max
        self.retake_speed = retake_speed
        self.retake_accel = retake_accel
        self.lookahead = lookahead
        self.last_direction: str | None = None
        self.last_slur_id: int | None = None
        # False until the first stroke is planned. Until then the bow is still
        # in the air and can be set down anywhere for free.
        self.placed = False
        # When the previous stroke finishes. A retake has to fit in the silence
        # between that and the next onset, or the note starts late.
        self.free_from = 0.0

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _sign(direction: str) -> float:
        return 1.0 if direction == "down" else -1.0

    def _depth_for(self, volume: float) -> float:
        """
        Depth for a volume, plus the global --depth-offset trim.

        Depth is normally a pure function of volume, which makes the whole
        piece's contact weight a consequence of its dynamics: a piece marked pp
        throughout sits at -0.96 mm everywhere, slow bow AND light depth at
        once. That corner is where the string stops speaking, so being able to
        lean the whole piece in or out without rewriting its dynamics is worth
        a knob.

        A per-note depth_mm annotation is NOT shifted by this — an explicit
        value means exactly what it says. Clamped to the same safety envelope
        apply_depth() enforces, so the printed plan shows what will be sent.
        """
        return float(np.clip(volume_to_depth(volume) + self.depth_offset,
                             -MAX_OUTWARD_DEPTH, MAX_INWARD_DEPTH))

    def _available(self, u: float, direction: str) -> float:
        """Metres of bow left from u in `direction` before the hard margin."""
        if direction == "down":
            return max(0.0, (U_MAX - u) * BOW_LENGTH)
        return max(0.0, (u - U_MIN) * BOW_LENGTH)

    @staticmethod
    def _place_centred(direction: str, length: float) -> float:
        """
        Where to set the bow down for a retake.

        Centre the stroke in the usable bow. That keeps the air move as short
        as it can be while still leaving room on both sides, so a retake never
        sets up the next note to need another one.
        """
        fraction = min(length / BOW_LENGTH, U_MAX - U_MIN)
        centre = (U_MIN + U_MAX) / 2.0
        if direction == "down":
            return float(np.clip(centre - fraction / 2.0, U_MIN, U_MAX - fraction))
        return float(np.clip(centre + fraction / 2.0, U_MIN + fraction, U_MAX))

    def _retake_seconds(self, u_from: float, u_to: float) -> float:
        """
        Time for lift -> traverse -> set down, at the pendant reset speed.

        These are three separate moveL calls, so each one accelerates from and
        decelerates back to a standstill. Summing the three durations is the
        honest figure; treating them as one move of the combined distance would
        credit a cruise that never happens and under-estimate by a good half
        second. That matters because this number decides whether the planner
        says a retake fits in the silence before the note.
        """
        traverse = abs(u_to - u_from) * BOW_LENGTH
        speed, accel = self.retake_speed, self.retake_accel
        return (trapezoid_duration(LIFT_M,     speed, accel)
                + trapezoid_duration(traverse, speed, accel)
                + trapezoid_duration(LIFT_M,   speed, accel)
                + 0.10)   # RTDE round trips and settling

    # ── main entry ───────────────────────────────────────────────

    @staticmethod
    def budget_volume(note: ScoreNote) -> float:
        """The level a stroke is sized on — the mean, when the note swells."""
        volume = float(np.clip(note.volume, 0.0, 1.0))
        if note.volume_end is None:
            return volume
        return 0.5 * (volume + float(np.clip(note.volume_end, 0.0, 1.0)))

    def plan(self, notes: list[ScoreNote]) -> list[Stroke]:
        strokes: list[Stroke] = []

        # Effective duration of every note, resolved up front so the bow
        # reserved for a slur group below is computed from the real numbers.
        durations = [self._effective_duration(notes, i) for i in range(len(notes))]

        # How much bow must remain in the current direction from each note to
        # the END of its slur group.
        #
        # Without this the planner sizes each note alone, and the first note of
        # a portato group can leave the bow near the tip with a second down-bow
        # still to come — which forces a retake, lifting the bow off in the
        # middle of the slur that exists precisely to keep it down. Reserving
        # the whole group up front means the group is placed where all of it
        # fits. Computed backwards so each note carries its own length plus
        # everything still to come under the same slur.
        reserves = [0.0] * len(notes)
        for i in range(len(notes) - 1, -1, -1):
            own = volume_to_speed(self.budget_volume(notes[i])) * durations[i]
            continues = (notes[i].slur_id is not None
                         and i + 1 < len(notes)
                         and notes[i + 1].slur_id == notes[i].slur_id)
            reserves[i] = own + (reserves[i + 1] if continues else 0.0)

        for i, note in enumerate(notes):
            # A rest is the one moment repositioning the bow is free: nothing
            # is sounding and the bow is off the string anyway. So before every
            # phrase, look ahead over the whole of it and, if the bow is not
            # somewhere the phrase can be played from, move it now rather than
            # discovering the problem several notes in and having to lift out
            # of the middle of a line.
            pre_reset = None
            gap_before = note.onset - self.free_from
            if self.lookahead and self.placed and gap_before >= self.MIN_RESET_GAP:
                pre_reset = self._plan_phrase_reset(notes, durations, i, gap_before)

            strokes.extend(self._plan_note(note, durations[i], reserves[i],
                                           pre_reset=pre_reset,
                                           gap_before=gap_before))

        return strokes

    def _phrase_bow_window(self, notes: list[ScoreNote], durations: list[float],
                           start: int) -> tuple | None:
        """
        The bow positions a phrase could be started from and played right
        through, without running past either end.

        A phrase here is the run of notes from `start` up to the next rest long
        enough to reset in — i.e. exactly the stretch during which the bow
        CANNOT be repositioned, so it is the stretch that has to fit.

        Each note contributes a signed displacement along the bow. Track the
        running total and its extremes: if the phrase starts at u, the bow
        visits u + every partial sum, so it stays legal precisely when

            u + min(partial sums) >= U_MIN     and
            u + max(partial sums) <= U_MAX

        which is a single interval of feasible starting points. That is the
        whole calculation — no search, no simulation.

        Returns (lo, hi, n_notes, span) or None when the interval is empty,
        meaning the phrase simply needs more bow than exists and will have to
        break somewhere no matter where it starts.
        """
        displacements = []
        previous_direction = self.last_direction

        index = start
        while index < len(notes) and (index - start) < self.MAX_LOOKAHEAD:
            if index > start:
                gap = notes[index].onset - (notes[index - 1].onset
                                            + durations[index - 1])
                if gap >= self.MIN_RESET_GAP:
                    break              # the phrase ends; the bow can reset there

            note = notes[index]
            direction = note.bow_dir or (
                self._flip(previous_direction) if previous_direction else "down")
            # A slur holds its direction rather than alternating.
            if (index > start and note.slur_id is not None
                    and notes[index - 1].slur_id == note.slur_id
                    and previous_direction is not None):
                direction = previous_direction

            length = min(volume_to_speed(self.budget_volume(note)) * durations[index],
                         USABLE_BOW)
            displacements.append(self._sign(direction) * length / BOW_LENGTH)
            previous_direction = direction
            index += 1

        if not displacements:
            return None

        running = lowest = highest = 0.0
        for step in displacements:
            running += step
            lowest = min(lowest, running)
            highest = max(highest, running)

        lo = U_MIN - lowest
        hi = U_MAX - highest
        if lo > hi + 1e-9:
            return None

        return lo, hi, len(displacements), highest - lowest

    def _plan_phrase_reset(self, notes: list[ScoreNote], durations: list[float],
                           start: int, gap: float) -> tuple | None:
        """
        Decide whether to spend this rest repositioning, and where to move to.

        Returns (from_u, seconds, note) for the caller to attach to the first
        stroke of the phrase, or None to stay put. Staying put is the common
        answer — the bow is only moved when the phrase genuinely could not be
        played from where it already is.
        """
        window = self._phrase_bow_window(notes, durations, start)
        if window is None:
            return ("infeasible",)

        lo, hi, count, span = window

        # Keep a little clearance inside the window so the small corrections
        # the planner makes later have somewhere to go.
        margin = min(0.04, max(0.0, (hi - lo) / 4.0))
        target = float(np.clip(self.u, lo + margin, hi - margin))

        if abs(target - self.u) <= self.MIN_RESET_TRAVEL:
            return None       # close enough that a lift-and-replace buys nothing

        seconds = self._retake_seconds(self.u, target)
        partial = False

        if seconds > gap:
            # The rest is not long enough for the whole move. Repositioning
            # anyway would push the phrase late, which is worse than the
            # compromise the ordinary shrink/flip machinery would make — so go
            # only as far as the rest genuinely allows. Getting partway still
            # leaves the phrase better placed than not moving at all.
            #
            # Two lift-offs alone may already exceed the rest, in which case
            # there is no move worth making and the bow stays put.
            reachable = self._reachable_within(self.u, target, gap)
            if reachable is None:
                return None
            target, partial = reachable, True
            seconds = self._retake_seconds(self.u, target)

        return (self.u, target, seconds, count, span, gap, partial)

    def _reachable_within(self, u_from: float, u_to: float,
                          budget: float) -> float | None:
        """
        How far toward u_to the bow can get inside `budget` seconds.

        Duration rises monotonically with distance, so a short bisection on the
        fraction of the way there is exact enough and needs no algebra against
        the trapezoidal profile. Returns None when even a stationary lift and
        set-down would overrun, since then no move is worth making.
        """
        if self._retake_seconds(u_from, u_from) > budget:
            return None

        low, high = 0.0, 1.0
        for _ in range(24):
            mid = (low + high) / 2.0
            candidate = u_from + mid * (u_to - u_from)
            if self._retake_seconds(u_from, candidate) <= budget:
                low = mid
            else:
                high = mid

        moved = u_from + low * (u_to - u_from)
        return moved if abs(moved - u_from) > self.MIN_RESET_TRAVEL else None

    def _effective_duration(self, notes: list[ScoreNote], i: int) -> float:
        """How long note i actually plays for, after gaps and articulation."""
        note = notes[i]
        next_onset = notes[i + 1].onset if i + 1 < len(notes) else None

        # Absorb a tiny notated gap so the bow does not stop pointlessly
        # between notes that are effectively continuous.
        duration = note.duration
        if next_onset is not None:
            gap = next_onset - note.end
            if 0.0 < gap <= self.GAP_ABSORB_FRAC * note.duration:
                duration = next_onset - note.onset

        # An articulation mark overrides that: a dotted note is meant to be
        # short, and the silence after it is the articulation. Without this the
        # notes of a portato group would run straight into each other and sound
        # like the single slurred tone the dots rule out.
        marks = [ARTICULATION_DURATION[a] for a in note.articulations
                 if a in ARTICULATION_DURATION]
        if marks:
            # Scale the silence the mark asks for by how long the note is, so
            # fast notes are articulated rather than chopped (ARTICULATION_REF).
            silence = note.duration * (1.0 - min(marks))
            if self.articulation_ref > 0.0:
                silence *= min(1.0, note.duration / self.articulation_ref)
            duration = note.duration - silence

        return duration

    def _plan_note(self, note: ScoreNote, duration: float,
                   reserve: float = 0.0, pre_reset: tuple | None = None,
                   gap_before: float | None = None) -> list[Stroke]:
        # A note that swells has to be budgeted on its MEAN speed, not its
        # starting one: length is mean_speed * duration, so sizing a pp->ff
        # crescendo as if it were pp would squeeze the whole shape into a pp
        # amount of bow and the loud end would never arrive.
        volume = float(np.clip(note.volume, 0.0, 1.0))
        budget_volume = self.budget_volume(note)

        # A reposition planned into the rest before this phrase. Applying it
        # here means the phrase starts from a position the whole of it fits
        # from, so nothing downstream has to lift the bow mid-line.
        planned_reset = None
        partial_reset = False
        phrase_warning = None
        if pre_reset is not None:
            if pre_reset[0] == "infeasible":
                phrase_warning = (
                    "this phrase needs more bow than exists in one span — it "
                    "will have to break somewhere whatever the starting point")
            else:
                from_u, target_u, seconds, count, span, gap, partial = pre_reset
                self.u = target_u
                planned_reset = (from_u, seconds)
                partial_reset = partial

        # speed_scale trades bow budget for tone. Amplitude is proportional to
        # bow speed and depth is worth under 1 dB, so a given loudness can be
        # made with a fast light bow or a slow heavy one — and the fast light
        # one sounds clearer. Scaling here raises the speed AND the bow used,
        # because length = speed x duration and the rhythm is fixed. The cost
        # is the bow budget: every stroke eats more hair, so retakes, flips and
        # dynamic shortfalls all become more likely. Pair it with a negative
        # --depth-offset to keep the loudness roughly where the score asked.
        target_speed = volume_to_speed(budget_volume) * self.speed_scale
        want_length = target_speed * duration

        # Longer than the whole bow at the dynamic the music asks for: no
        # single stroke can do it, so change bow mid-note like a player would.
        if want_length > USABLE_BOW:
            return self._split_note(note, duration, want_length)

        forced = note.bow_dir is not None
        direction = note.bow_dir or (
            self._flip(self.last_direction) if self.last_direction else "down")

        # Still inside a portato group: the notes are separate, but the bow is
        # not. Carry on in the same direction instead of alternating, and treat
        # that as forced so the planner cannot flip out of it — a bow change
        # mid-group is precisely what the slur forbids.
        continuing = (note.slur_id is not None
                      and note.slur_id == self.last_slur_id
                      and self.last_direction is not None)
        if continuing:
            direction = self.last_direction
            forced = True

        # Before the first stroke the bow is not on the string yet — prepare()
        # sets it down wherever the first stroke begins. So if the opening note
        # does not fit from the requested start, move the start rather than
        # planning a retake that would only be lifting a bow already in the
        # air. --start-u is honoured whenever the first note does fit from it.
        # What has to fit in this direction: this note, plus whatever else is
        # still to come under the same slur. For an unslurred note the two are
        # the same; for the first note of a portato group `need` covers the
        # whole group, so the group gets placed where all of it fits instead of
        # stranding the bow near an end with more of the slur still to play.
        need = max(want_length, reserve)

        if not self.placed:
            if self._available(self.u, direction) < need:
                self.u = self._place_centred(direction, min(need, USABLE_BOW))
            self.placed = True

        events: list[str] = []
        warnings: list[str] = []
        retake_from = None
        retake_time = 0.0

        # The reposition planned into the preceding rest, if there was one.
        # This uses the same lift/traverse/set-down machinery as an emergency
        # retake, but it happens in silence that already existed rather than
        # stealing time from the music.
        if planned_reset is not None:
            retake_from, retake_time = planned_reset
            events.append("RESET~" if partial_reset else "RESET")
        if phrase_warning:
            warnings.append(phrase_warning)

        # Where the bow physically is right now. A planned reset has already
        # moved self.u to where the phrase should begin, but the robot has not
        # gone there yet — so any repositioning below must still start from
        # here, not from the position the plan is aiming at.
        physical_u = planned_reset[0] if planned_reset is not None else self.u

        u_start = self.u

        available = self._available(u_start, direction)
        length = want_length

        if need > available:
            if not forced:
                # Free to bow it the other way. Costs nothing musically, so try
                # this before compromising the dynamic.
                other = self._flip(direction)
                if self._available(u_start, other) >= need:
                    direction = other
                    available = self._available(u_start, other)
                    events.append("FLIP")

            if need > available:
                # Shrinking is only on the table when nothing else in this slur
                # is waiting on the bow; shortening a note to fit while the rest
                # of the group still needs room only defers the problem.
                shrinkable = reserve <= want_length + 1e-9
                if (shrinkable and available >= want_length * (1.0 - BOW_SLACK)
                        and available >= self.MIN_STROKE):
                    # Close enough: spend the bow that is there. Shading, not a
                    # wrong dynamic.
                    length = available
                    events.append("SHRINK")
                else:
                    # Not close enough. Lift, reposition, come back down; the
                    # bowing and the dynamic both survive intact.
                    #
                    # Anchored at physical_u, so if a reset was already planned
                    # into the preceding rest this supersedes it as ONE move
                    # from where the bow actually is, rather than leaving a
                    # move that starts from a place the bow never reached.
                    retake_from = physical_u
                    u_start = self._place_centred(direction, min(need, USABLE_BOW))
                    retake_time = self._retake_seconds(retake_from, u_start)
                    if "RESET" in events:
                        events.remove("RESET")
                    events.append("RETAKE")
                    available = self._available(u_start, direction)
                    if length > available:
                        length = available
                        events.append("SHRINK")
                    if continuing:
                        warnings.append(
                            "slur broken: the group needs more bow than is left, "
                            "so the bow lifts mid-slur")

        # Bow distribution: pull a stroke that would end outside the comfort
        # band back inside, within the slack budget, before it gets near the
        # hard margin.
        length = self._distribute(u_start, direction, length, want_length, events)

        solution = solve_stroke(length, duration, accel_max=self.accel_max)
        if solution.length_capped:
            events.append("CAP")
            warnings.append(
                f"note is {duration*1000:.0f} ms — {length*1000:.0f} mm of bow "
                f"needs {4*length/duration**2:.1f} m/s^2, above the "
                f"{self.accel_max:.1f} limit; shortened to {solution.length*1000:.0f} mm")
        if solution.too_slow:
            warnings.append("cannot be played in the time available")

        length = solution.length
        u_end = u_start + self._sign(direction) * length / BOW_LENGTH
        u_end = float(np.clip(u_end, U_MIN, U_MAX))

        actual_volume = speed_to_volume(solution.mean_speed)
        if actual_volume < budget_volume - 0.08:
            warnings.append(
                f"dynamic shortfall: asked {note.dynamic}, will sound "
                f"{velocity_to_dynamic(volume_to_velocity(actual_volume))} "
                f"({speed_to_dbfs(solution.mean_speed) - speed_to_dbfs(target_speed):+.1f} dB)")

        # Depth follows the level the stroke actually delivers, so a stroke
        # that had to be shortened is not left pressing as if it were still
        # loud. Swelling strokes override this per segment in _build_segments.
        depth = (note.depth_override if note.depth_override is not None
                 else self._depth_for(budget_volume))
        stroke = Stroke(
            note_index=note.index,
            onset=note.onset,
            duration=solution.duration,
            direction=direction,
            direction_source=note.bow_dir_source,
            u_start=u_start,
            u_end=u_end,
            length=length,
            mean_speed=solution.mean_speed,
            depth=depth,
            volume_target=volume,
            volume_actual=actual_volume,
            dynamic=note.dynamic,
            retake_from=retake_from,
            retake_time=retake_time,
            events=events,
            warnings=warnings,
        )
        stroke.segments = self._build_segments(stroke, note, solution)

        # A retake is silent, but it is not instantaneous. It has to fit in the
        # gap between the end of the previous stroke and this onset; if it does
        # not, the note starts late and everything after it is pushed back, so
        # say so at planning time rather than discovering it in performance.
        if retake_from is not None:
            gap = note.onset - self.free_from
            if retake_time > gap + 1e-3:
                warnings.append(
                    f"retake needs {retake_time:.2f} s but only {max(gap, 0.0):.2f} s "
                    f"of silence precedes this note — it will start "
                    f"{retake_time - gap:.2f} s late")

        self.u = u_end
        self.last_direction = direction
        self.last_slur_id = note.slur_id
        self.free_from = note.onset + solution.duration
        return [stroke]

    def _distribute(self, u_start: float, direction: str,
                    length: float, want_length: float,
                    events: list[str]) -> float:
        """
        Nudge a stroke so it lands back in the home band, spending at most
        BOW_SLACK of its length either way.

        This is bow distribution, and it is what stops the bow migrating to one
        end of itself over a long piece. A stroke heading away from the middle
        gets trimmed; one heading back gets extended. Because it can do both,
        there is no length the bow can settle into and get stuck at — the
        earlier shrink-only version pinned the last third of t1 at u = 0.12,
        playing it all in one patch of hair at the frog.

        A stroke already landing inside the home band is left exactly alone, so
        well-placed notes never get their dynamic modulated for no reason.
        """
        sign = self._sign(direction)
        u_end = u_start + sign * length / BOW_LENGTH
        if U_HOME_LO <= u_end <= U_HOME_HI:
            return length

        # Aim for the near edge of the home band rather than dead centre: the
        # smallest correction that does the job costs the least dynamic.
        target_end = float(np.clip(u_end, U_HOME_LO, U_HOME_HI))

        # Signed, so a target "behind" the bow yields a negative length and
        # gets floored by the slack limit rather than reversing the stroke.
        wanted = (target_end - u_start) * sign * BOW_LENGTH

        adjusted = float(np.clip(wanted,
                                 want_length * (1.0 - BOW_SLACK),
                                 want_length * (1.0 + BOW_SLACK)))
        adjusted = min(adjusted, self._available(u_start, direction))
        adjusted = max(adjusted, self.MIN_STROKE)

        # Only worth reporting when it actually changed something audible.
        if abs(adjusted - length) > 0.05 * max(length, 1e-9) and "SHRINK" not in events:
            events.append("DISTRIB")
        return adjusted

    def _split_note(self, note: ScoreNote, duration: float,
                    want_length: float) -> list[Stroke]:
        """
        A note too long for one bow becomes k alternating sub-strokes.

        The bow change is audible, which is unavoidable — the alternative is
        slowing the bow until it fits, which would make a long fortissimo note
        quiet. Real players change bow here too.
        """
        parts = max(2, math.ceil(want_length / (USABLE_BOW * 0.9)))
        part_duration = duration / parts
        strokes: list[Stroke] = []

        for part in range(parts):
            # Spread a swell ACROSS the parts rather than repeating it in each
            # one: a pp->ff note split in two goes pp->mf then mf->ff, not
            # pp->ff twice.
            if note.volume_end is None:
                part_volume, part_volume_end = note.volume, None
            else:
                def at(fraction):
                    return note.volume + (note.volume_end - note.volume) * fraction
                part_volume = at(part / parts)
                part_volume_end = at((part + 1) / parts)

            sub_note = ScoreNote(
                index=note.index,
                onset=note.onset + part * part_duration,
                duration=part_duration,
                volume=part_volume,
                dynamic=note.dynamic,
                bow_dir=None if part else note.bow_dir,
                bow_dir_source=note.bow_dir_source,
                volume_end=part_volume_end,
                articulations=note.articulations,
                measure=note.measure,
                beat=note.beat,
            )
            produced = self._plan_note(sub_note, part_duration)
            for stroke in produced:
                stroke.part_of = note.index
                stroke.events.append(f"SPLIT {part+1}/{parts}")
            strokes.extend(produced)

        return strokes

    def _build_segments(self, stroke: Stroke, note: ScoreNote,
                        solution: MotionSolution) -> list[Segment]:
        """
        Split a stroke into sub-moves when the score asks for a swell inside it.

        A single moveL is one constant commanded speed, so a crescendo written
        across one long note is invisible to it. Splitting the stroke into a
        few blended sub-moves at rising speeds renders it — and because they
        all continue in the same direction along the bow, there is no bow
        change and the note stays one sound.
        """
        volume_end = note.volume_end
        swell = (self.enable_swell and volume_end is not None
                 and abs(volume_end - stroke.volume_target) >= self.SWELL_THRESHOLD
                 and stroke.duration >= self.SWELL_MIN_DURATION)

        if not swell:
            return [Segment(
                u_start=stroke.u_start, u_end=stroke.u_end,
                speed=solution.speed, accel=solution.accel,
                depth=stroke.depth, duration=solution.duration,
            )]

        count = self.SWELL_SEGMENTS
        weights = np.array([
            volume_to_speed(stroke.volume_target
                            + (volume_end - stroke.volume_target) * (i + 0.5) / count)
            for i in range(count)
        ])
        # Distribute the bow in proportion to speed so each sub-move takes the
        # same time and the swell is even.
        fractions = weights / weights.sum()
        segment_duration = stroke.duration / count

        segments: list[Segment] = []
        u = stroke.u_start
        travel = stroke.u_end - stroke.u_start
        for i in range(count):
            u_next = u + travel * float(fractions[i])
            volume = stroke.volume_target + (volume_end - stroke.volume_target) * (i + 0.5) / count
            sub = solve_stroke(abs(u_next - u) * BOW_LENGTH, segment_duration)
            segments.append(Segment(
                u_start=u, u_end=u_next,
                speed=sub.speed, accel=sub.accel,
                depth=self._depth_for(volume), duration=sub.duration,
            ))
            u = u_next

        stroke.events.append("SWELL")
        return segments

    @staticmethod
    def _flip(direction: str) -> str:
        return "up" if direction == "down" else "down"


# ══════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════

def print_plan(strokes: list[Stroke], notes: list[ScoreNote], meta: dict):
    print()
    print("=" * 100)
    print(f"PLAN  {meta['source']}")
    print(f"  {meta['format']}  {meta.get('time_signature','?')}  "
          f"{meta.get('tempo_bpm','?')} bpm  "
          f"{meta['n_notes']} notes  {meta.get('duration',0):.1f} s")
    if meta.get("n_bowings"):
        print(f"  bowings read from the score: {meta['n_bowings']}"
              + (f", slurs: {meta['n_slurs']}" if meta.get("n_slurs") else "")
              + (f", hairpins: {meta['n_hairpins']}" if meta.get("n_hairpins") else ""))
    if meta.get("repeats_expanded"):
        print(f"  repeats expanded: {meta['n_measures_written']} written measures "
              f"-> {meta['n_measures_played']} played")
    if meta.get("n_grace_skipped"):
        print(f"  grace notes skipped: {meta['n_grace_skipped']} "
              f"(inaudible on an open string)")
    if meta.get("n_slurs_merged"):
        print(f"  legato slur groups merged into one stroke each: "
              f"{meta['n_slurs_merged']} (one bow, one sound on an open A)")
    if meta.get("n_portato_groups"):
        print(f"  portato groups kept as separate notes: "
              f"{meta['n_portato_groups']} (dots under a slur: one bow "
              f"direction, notes articulated inside it)")
    for note in meta.get("score_notes", []):
        print(f"  NOTE: {note}")
    dyn = meta.get("dynamics")
    if dyn is not None:
        lo, hi = dyn.get("written_low"), dyn.get("written_high")
        if dyn.get("expanded"):
            print(f"  dynamics: score spans "
                  f"{velocity_to_dynamic(volume_to_velocity(lo))}"
                  f"..{velocity_to_dynamic(volume_to_velocity(hi))}, "
                  f"expanded to the full usable range "
                  f"(+{dyn['gain_db']:.1f} dB of contrast)")
        elif dyn.get("described"):
            print(f"  dynamics: score spans "
                  f"{velocity_to_dynamic(volume_to_velocity(lo))}"
                  f"..{velocity_to_dynamic(volume_to_velocity(hi))}, "
                  f"played as written")
        else:
            print(f"  ##  DYNAMICS: {dyn['reason']} — every note will sound the")
            print(f"  ##  same. Nothing here can fix that: write dynamics into")
            print(f"  ##  the score in MuseScore and re-export, or set them per")
            print(f"  ##  note with --annotations.")
    if meta.get("cc2_events"):
        print(f"  CC2 expression events: {meta['cc2_events']} "
              f"(hairpins recovered from the curve)")
    print("=" * 100)
    print(f"{'#':>3} {'onset s':>8} {'dur s':>6} {'dyn':>4} {'dir':>4} {'src':>10} "
          f"{'u start':>8} {'u end':>7} {'bow mm':>7} {'v m/s':>6} {'depth mm':>9} "
          f"{'dBFS':>7}  events")
    print("-" * 100)

    for stroke in strokes:
        events = " ".join(stroke.events)
        print(f"{stroke.note_index:3d} {stroke.onset:8.3f} {stroke.duration:6.3f} "
              f"{stroke.dynamic:>4} {stroke.direction:>4} {stroke.direction_source:>10} "
              f"{stroke.u_start:8.3f} {stroke.u_end:7.3f} "
              f"{stroke.length*1000:7.1f} {stroke.mean_speed:6.3f} "
              f"{stroke.depth*1000:+9.2f} {speed_to_dbfs(stroke.mean_speed):7.1f}  {events}")
        for warning in stroke.warnings:
            print(f"      ! {warning}")

    print("-" * 100)

    counts: dict[str, int] = {}
    for stroke in strokes:
        for event in stroke.events:
            key = event.split()[0]
            counts[key] = counts.get(key, 0) + 1

    u_values = [s.u_start for s in strokes] + [s.u_end for s in strokes]
    total_warnings = sum(len(s.warnings) for s in strokes)

    print(f"strokes           {len(strokes)}  from {len(notes)} notes")
    print(f"bow position      {min(u_values):.3f} .. {max(u_values):.3f}  "
          f"(hard limits {U_MIN} .. {U_MAX})")
    print(f"bow used          {sum(s.length for s in strokes)*1000:.0f} mm total, "
          f"{np.mean([s.length for s in strokes])*1000:.0f} mm mean")
    print(f"mean speed        {min(s.mean_speed for s in strokes):.3f} .. "
          f"{max(s.mean_speed for s in strokes):.3f} m/s  "
          f"({speed_to_dbfs(min(s.mean_speed for s in strokes)):.1f} .. "
          f"{speed_to_dbfs(max(s.mean_speed for s in strokes)):.1f} dBFS)")

    # Loudness follows the mean, but moveL is commanded with the peak, and only
    # speeds up to CALIBRATED_SPEED_MAX were ever measured. Anything above that
    # is extrapolation, so say so rather than burying it.
    peaks = [segment.speed for stroke in strokes for segment in stroke.segments]
    accels = [segment.accel for stroke in strokes for segment in stroke.segments]
    over = sum(1 for p in peaks if p > CALIBRATED_SPEED_MAX)
    print(f"peak speed        {min(peaks):.3f} .. {max(peaks):.3f} m/s commanded"
          + (f"   [{over}/{len(peaks)} above the {CALIBRATED_SPEED_MAX} m/s "
             f"calibration ceiling]" if over else "   (all within calibration)"))
    print(f"accel             {min(accels):.2f} .. {max(accels):.2f} m/s^2 "
          f"(pendant nominal {MOVE_ACCEL})")
    resets = [s for s in strokes if "RESET" in s.events]
    if resets:
        moved = sum(abs(s.u_start - s.retake_from) for s in resets
                    if s.retake_from is not None) * BOW_LENGTH
        print(f"rest repositioning {len(resets)} reset(s) during rests, "
              f"{moved*1000:.0f} mm of bow travel in silence, "
              f"{sum(s.retake_time for s in resets):.2f} s total")
    print(f"events            {counts if counts else 'none'}")
    print(f"warnings          {total_warnings}")
    print("  FLIP    bow direction reversed because the intended one had no room")
    print("  SHRINK  stroke shortened to the bow available (slightly quieter)")
    print("  DISTRIB stroke trimmed to keep the bow inside its comfort band")
    print("  RESET   bow repositioned during a rest, so the coming phrase fits")
    print("  RESET~  as RESET, but the rest only allowed part of the move")
    print("  RETAKE  bow lifted, repositioned and set down to keep a forced bowing")
    print("  SPLIT   note too long for one bow, played as several")
    print("  CAP     note too short for that much bow at the accel limit")
    print("  SWELL   rendered as sub-moves at rising/falling speed")
    print("=" * 100)


# ══════════════════════════════════════════════════════════════════
# Audio input
# ══════════════════════════════════════════════════════════════════
# An interface with several inputs will happily hand back a perfectly valid
# stream of the converter's own noise floor from a socket with nothing in it.
# Nothing errors, the file is the right length, and it is silent — which is a
# very expensive thing to discover after a session's worth of runs. So the
# channel is checked for signal BEFORE the robot is allowed to move.

# Seconds of silence between pressing ENTER and the first note. Long enough to
# step back from the instrument and let the room settle, and it sits INSIDE the
# recording, so every take carries a noise-floor reference ahead of the music
# and the input stream is demonstrably running before anything sounds.
LEAD_IN_SEC = 0.0

# Inherited from recording_a_only rather than duplicated, the same way the
# waypoints and depth convention are, so the performance script and the data
# collection script can never disagree about which input they are listening to.
AUDIO_LIVE_DBFS      = _rec.AUDIO_LIVE_DBFS
probe_audio_channels = _rec.probe_audio_channels
select_audio_channel = _rec.select_audio_channel
report_audio_channels = _rec.report_audio_channels


# ══════════════════════════════════════════════════════════════════
# Execution
# ══════════════════════════════════════════════════════════════════

class PiecePlayer:
    """
    Drives the robot through a planned list of strokes.

    Every motion is a moveL through safe_moveL, so a failed move or a stopped
    control script stops the script rather than being ignored — the same fault
    discipline the recording sessions ran under. Poses are only ever pose_at(u)
    for u inside [U_MIN, U_MAX], with the depth offset and the +z lift the
    taught safe points already define.
    """

    def __init__(self, record_audio: bool = False, output_dir: Path | None = None,
                 retake_speed: float = RETAKE_SPEED,
                 retake_accel: float = MOVE_ACCEL,
                 audio_device=None, audio_channel: int = 1,
                 servo: bool = False, servo_threshold: float = None,
                 servo_lookahead: float = None):
        from BaselineControls.baseline_controller import TargetStringBaselineController

        self.controller = TargetStringBaselineController([])
        _rec.apply_verified_installation(self.controller)

        # Hybrid execution. Off by default: the moveL path is the verified one,
        # and servoL only wins on notes long enough to shape.
        self.servo = None
        self.servo_mod = None
        self.servo_threshold = 0.0
        self.servo_log: list[dict] = []
        if servo:
            import importlib.util as _il
            _spec = _il.spec_from_file_location(
                "servo_player", Path(__file__).parent / "servo_player.py")
            self.servo_mod = _il.module_from_spec(_spec)
            _spec.loader.exec_module(self.servo_mod)
            self.servo_threshold = (self.servo_mod.HYBRID_THRESHOLD
                                    if servo_threshold is None else servo_threshold)
            self.servo = self.servo_mod.ServoStrokePlayer(
                self.controller, sys.modules[__name__],
                lookahead=(self.servo_mod.SERVO_LOOKAHEAD
                           if servo_lookahead is None else servo_lookahead))
            print(f"Hybrid execution: servoL for notes >= {self.servo_threshold:.2f}s, "
                  f"moveL below (lookahead {self.servo.lookahead:.3f}s, "
                  f"accel limit {self.servo.accel_limit:.1f})")

        self.rtde_c = self.controller.rtde_c
        self.rtde_r = self.controller.rtde_r
        self.logger = self.controller.state_logger
        self.logger.set_bow_reference(FROG, TIP)

        self.retake_speed = retake_speed
        self.retake_accel = retake_accel
        self.record_audio = record_audio
        self.output_dir = output_dir or (REPO_ROOT / "BaselineControls" / "performances")
        self.audio_buffer = None
        self.audio_device = audio_device
        self.audio_channel = audio_channel
        self.audio_start = None
        self.lead_in = 0.0
        self.timeline: list[dict] = []

    # ── setup ────────────────────────────────────────────────────

    def prepare(self, first: Stroke):
        """Zero the F/T sensor at the taught safe point, then set the bow down."""
        _rec.zero_ft_at_config_safe(self.controller, CONFIG_NAME)

        start_pose = apply_depth(pose_at(first.u_start), CFG, first.depth)
        print(f"Placing the bow at u={first.u_start:.3f} "
              f"(depth {first.depth*1000:+.2f} mm)...")
        safe_moveL(self.controller, lifted(start_pose), RESET_SPEED, MOVE_ACCEL,
                   what="move above the first note")
        safe_moveL(self.controller, start_pose, RESET_SPEED, MOVE_ACCEL,
                   what="set the bow down for the first note")

        force = float(np.linalg.norm(self.rtde_r.getActualTCPForce()[:3]))
        print(f"  contact {force:.3f} N")

    # ── motion primitives ────────────────────────────────────────

    def _retake(self, u_from: float, u_to: float, depth: float):
        """Lift clear, traverse in the air, set down. The bow makes no sound."""
        here = apply_depth(pose_at(u_from), CFG, depth)
        there = apply_depth(pose_at(u_to), CFG, depth)
        speed, accel = self.retake_speed, self.retake_accel
        safe_moveL(self.controller, lifted(here), speed, accel,
                   what=f"retake: lift at u={u_from:.3f}")
        safe_moveL(self.controller, lifted(there), speed, accel,
                   what=f"retake: traverse to u={u_to:.3f}")
        safe_moveL(self.controller, there, speed, accel,
                   what=f"retake: set down at u={u_to:.3f}")

    def _play(self, stroke: Stroke):
        """
        Execute one stroke.

        A single segment is one moveL, identical in form to the validated
        recording stroke. Several segments (a swell) go out as one blended
        moveL path so the bow does not stop between them and the note stays a
        single sound.

        With --servo, notes at or above the hybrid threshold are STREAMED with
        servoL instead, so their speed and depth can be changed while they
        sound. Short notes stay on moveL, where the controller plans the whole
        trapezoid itself and measurably tracks better — see servo_player's
        HYBRID_THRESHOLD for the numbers.
        """
        if self.servo is not None and stroke.duration >= self.servo_threshold:
            self._play_servo(stroke)
            return

        if len(stroke.segments) == 1:
            segment = stroke.segments[0]
            target = apply_depth(pose_at(segment.u_end), CFG, segment.depth)
            safe_moveL(self.controller, target, segment.speed, segment.accel,
                       what=f"note {stroke.note_index} ({stroke.direction})")
            return

        path = []
        for i, segment in enumerate(stroke.segments):
            pose = apply_depth(pose_at(segment.u_end), CFG, segment.depth)
            length = abs(segment.u_end - segment.u_start) * BOW_LENGTH
            # Blend must stay well under half a segment, and the last one must
            # be zero so the stroke actually stops where it is supposed to.
            #
            # The 0.3*length term already protects short segments; the absolute
            # cap is what decides whether an interior junction is a genuine
            # transition or a rounded-off stop. At the old 5 mm, a ~56 mm RL
            # envelope segment blended over 9% of its length and the bow still
            # slowed to near zero twice per note -- audible as the note
            # breaking into pieces. 25 mm lets the controller carry velocity
            # through the junction, which is the whole point of the envelope.
            # 0.3 -> 0.45 of the segment, 2026-08-17. The 25 mm cap was
            # raised for ~56 mm envelope segments, where 0.3*length = 17 mm.
            # The RL env's segments are 16-35 mm, so 0.3*length gives only
            # 4.8-10.6 mm -- back at the 5 mm this comment already records as
            # too small to carry velocity. Measured: 3-segment notes overran
            # their duration by 120 ms, cut to 55 ms by raising junction
            # accel; this chases the rest. 0.45 stays under the geometric
            # limit of half a segment, and is additionally clamped by the NEXT
            # segment so two adjacent blends cannot overlap.
            nxt = stroke.segments[i + 1] if i + 1 < len(stroke.segments) else None
            nxt_len = (abs(nxt.u_end - nxt.u_start) * BOW_LENGTH) if nxt else length
            blend = (0.0 if i == len(stroke.segments) - 1
                     else min(0.45 * length, 0.45 * nxt_len, 0.025))
            path.append(list(pose) + [segment.speed, segment.accel, blend])

        try:
            ok = self.rtde_c.moveL(path)
        except Exception as e:
            raise RobotFaultStop(
                f"blended moveL raised on note {stroke.note_index}: {e}. STOPPING.") from e
        if ok is False:
            raise RobotFaultStop(
                f"blended moveL returned False on note {stroke.note_index}. STOPPING.")

    def _play_servo(self, stroke: Stroke):
        """
        Stream one long note with servoL, rendering any swell as a schedule of
        setpoint changes rather than a pre-blended path.

        This is where the servo path earns its place: the SAME mechanism a
        policy would use to react mid-note is used here to render the swell the
        score asked for. A blended moveL can only follow a shape decided before
        the stroke began; a setpoint can be rewritten while the note sounds.

        The scheduler runs on its own thread because the control loop must not
        block — a stalled tick is audible.
        """
        import threading as _threading

        segments = stroke.segments
        first = segments[0]
        span = abs(first.u_end - first.u_start) * BOW_LENGTH
        setpoint = self.servo_mod.Setpoint(
            speed=span / first.duration if first.duration > 0 else stroke.mean_speed,
            depth=first.depth)

        stop = _threading.Event()

        def schedule():
            # Walk the remaining segments in time, writing each one's mean
            # speed and depth as it becomes current.
            elapsed = first.duration
            for seg in segments[1:]:
                if stop.wait(elapsed):
                    return
                length = abs(seg.u_end - seg.u_start) * BOW_LENGTH
                setpoint.write(
                    speed=length / seg.duration if seg.duration > 0 else None,
                    depth=seg.depth)
                elapsed = seg.duration

        worker = None
        if len(segments) > 1:
            worker = _threading.Thread(target=schedule, daemon=True)
            worker.start()

        try:
            result = self.servo.play_stroke(
                stroke.u_start, stroke.direction, stroke.duration, setpoint,
                u_limit=stroke.u_end)
        finally:
            stop.set()
            if worker is not None:
                worker.join(timeout=0.2)

        self.servo_log.append({"note_index": stroke.note_index, **result})

    # ── the performance ──────────────────────────────────────────

    def play(self, strokes: list[Stroke], total_duration: float,
             lead_in: float = 0.0):
        # Audio and state logging both start BEFORE the lead-in, so the wait is
        # captured rather than skipped. That gives a stretch of room tone ahead
        # of the first note to measure the noise floor against, and guarantees
        # the input stream and the F/T log are both settled well before
        # anything sounds. audio_lead_in_s in the performance JSON says where
        # the music actually begins.
        if self.record_audio:
            self._start_audio(total_duration + lead_in)

        self.logger.start()
        self.lead_in = lead_in
        if lead_in > 0:
            self._count_down(lead_in)

        t_zero = time.time()

        for stroke in strokes:
            # A retake happens in the silence before the note. Start it early
            # enough that the note itself still begins on time.
            if stroke.retake_from is not None:
                lead = stroke.retake_time
                self._sleep_until(t_zero + stroke.onset - lead)
                self._retake(stroke.retake_from, stroke.u_start, stroke.depth)

            self._sleep_until(t_zero + stroke.onset)

            t_start = time.time()
            self._play(stroke)
            t_end = time.time()

            drift = t_start - t_zero - stroke.onset
            self.timeline.append({
                "note_index": stroke.note_index,
                "scheduled_onset": stroke.onset,
                "actual_onset": t_start - t_zero,
                "drift": drift,
                "planned_duration": stroke.duration,
                "actual_duration": t_end - t_start,
                "direction": stroke.direction,
                "u_start": stroke.u_start,
                "u_end": stroke.u_end,
                "length_m": stroke.length,
                "mean_speed": stroke.mean_speed,
                "depth_m": stroke.depth,
                "dynamic": stroke.dynamic,
                "events": stroke.events,
            })

            print(f"  [{stroke.note_index:3d}] {stroke.direction:4s} "
                  f"u {stroke.u_start:.3f}->{stroke.u_end:.3f}  "
                  f"{stroke.length*1000:5.1f} mm  {stroke.mean_speed:.3f} m/s  "
                  f"{stroke.dynamic:>3}  "
                  f"plan {stroke.duration:.3f}s / real {t_end-t_start:.3f}s  "
                  f"drift {drift:+.3f}s  {' '.join(stroke.events)}")

        # Lift clear of the string at the end, the same way the recording
        # script finishes every stroke on a safe point. Leaving the bow resting
        # on the string holds the last note damped and leaves the arm loaded
        # against the instrument until someone moves it.
        last = strokes[-1]
        final_pose = apply_depth(pose_at(last.u_end), CFG, last.depth)
        safe_moveL(self.controller, lifted(final_pose), RESET_SPEED, MOVE_ACCEL,
                   what="lift off at the end of the piece")

        self.logger.stop()
        return time.time() - t_zero

    @staticmethod
    def _sleep_until(target: float):
        remaining = target - time.time()
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _count_down(seconds: float):
        """
        Wait before the first note, printing as it goes.

        The printing is the point. A silent thirty-second pause between "press
        ENTER" and the first sound is indistinguishable from a hung script, and
        this one has several places it can legitimately block. Counting down
        out loud says which of those it is.
        """
        print(f"\nLead-in: {seconds:.0f} s before the first note. "
              f"Audio and state logging are already running.")
        end = time.time() + seconds

        while True:
            remaining = end - time.time()
            if remaining <= 0.05:
                break
            # Every five seconds, then every second over the last five.
            step = 1.0 if remaining <= 5.0 else 5.0
            print(f"  {remaining:5.1f} s", flush=True)
            time.sleep(min(step, remaining))

        print("  playing.\n", flush=True)

    # ── audio ────────────────────────────────────────────────────

    def _start_audio(self, total_duration: float):
        import sounddevice as sd

        if self.audio_device is None:
            try:
                self.audio_device = _rec.find_focusrite_device()
            except Exception as e:
                print(f"  audio disabled: {e}")
                self.record_audio = False
                return

        # mapping selects the physical input to record. Passing channels=1
        # instead would silently take input 1 whatever is plugged into it,
        # which is exactly how a session of runs can come back empty.
        samples = int((total_duration + 5.0) * SAMPLE_RATE)
        self.audio_buffer = sd.rec(samples, samplerate=SAMPLE_RATE,
                                   mapping=[self.audio_channel],
                                   dtype="float32", device=self.audio_device)
        self.audio_start = time.time()

    def save(self, meta: dict, elapsed: float):
        """Write the audio, the 100 Hz state log and the executed timeline."""
        from datetime import datetime

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{Path(meta['source']).stem}_{stamp}"

        if self.record_audio and self.audio_buffer is not None:
            import sounddevice as sd
            import soundfile as sf
            sd.stop()
            sd.wait()
            valid = min(int((time.time() - self.audio_start) * SAMPLE_RATE),
                        len(self.audio_buffer))
            audio = self.audio_buffer[:valid]
            sf.write(str(self.output_dir / f"{stem}.wav"), audio, SAMPLE_RATE)
            rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=float) ** 2)))
            level = 20.0 * math.log10(rms + 1e-12)
            print(f"  audio  {stem}.wav  ch{self.audio_channel}  "
                  f"peak {float(np.max(np.abs(audio))):.4f}  {level:.1f} dBFS"
                  + ("   ## SILENT — check the input" if level < AUDIO_LIVE_DBFS else ""))

        self.logger.save(str(self.output_dir / f"{stem}_state.npy"))

        record = {
            "score": meta,
            "config": CONFIG_NAME,
            "bow_length_m": BOW_LENGTH,
            "elapsed_s": elapsed,
            "audio_lead_in_s": self.lead_in,
            "calibration": {
                "speed_min": SPEED_MIN, "speed_max": SPEED_MAX,
                "depth_min": -MAX_OUTWARD_DEPTH, "depth_max": MAX_INWARD_DEPTH,
                "note": "amplitude proportional to bow speed; "
                        "measured over dataset_a_final/standard",
            },
            "timeline": self.timeline,
        }
        (self.output_dir / f"{stem}_performance.json").write_text(
            json.dumps(record, indent=2))
        print(f"  state  {stem}_state.npy")
        print(f"  plan   {stem}_performance.json")

    def close(self):
        self.controller.close()


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def build_score(args) -> tuple[list[ScoreNote], dict]:
    """Load the score, merge in any extra sources, and settle the bowings."""
    path = Path(args.midi)
    if path.suffix.lower() in (".xml", ".musicxml", ".mxl"):
        notes, meta = parse_musicxml(str(path), args.tempo_scale,
                                     expand_repeats=not args.no_repeats)
    else:
        notes, meta = parse_midi(str(path), args.tempo_scale)

        if meta["pitches"] and set(meta["pitches"]) != {57}:
            others = [p for p in meta["pitches"] if p != 57]
            print(f"  NOTE: the file contains pitches other than open A "
                  f"(MIDI 57): {others}. All of them will be played as open A.")

    if args.score:
        score_notes, score_meta = parse_musicxml(
            args.score, args.tempo_scale, expand_repeats=not args.no_repeats)
        merged = merge_score_into_midi(notes, score_notes)
        for key in ("n_bowings", "n_slurs", "n_hairpins",
                    "repeats_expanded", "n_measures_written",
                    "n_measures_played", "n_grace_skipped", "score_notes"):
            if key in score_meta:
                meta[key] = score_meta[key]
        meta["score_overlay"] = args.score
        print(f"  merged {merged} notes from {Path(args.score).name}: "
              f"{score_meta.get('n_bowings',0)} bowings, "
              f"{score_meta.get('n_slurs',0)} slurs")

        # Positional matching is only valid if both files describe the same
        # sequence of notes. A count mismatch means every bowing past the first
        # difference lands on the wrong note, so this is an error worth
        # shouting about rather than a footnote.
        if len(score_notes) != len(notes):
            print()
            print(f"  ##  WARNING: {Path(args.score).name} has "
                  f"{len(score_notes)} notes, the MIDI has {len(notes)}.")
            print(f"  ##  Bowings are matched by position, so only the first "
                  f"{min(len(notes), len(score_notes))} are trustworthy and")
            print(f"  ##  anything after the first discrepancy is on the wrong "
                  f"note.")
            if not score_meta.get("repeats_expanded") and len(notes) > len(score_notes):
                print(f"  ##  The MIDI has MORE notes and the score has no "
                      f"expanded repeats — if the piece repeats, MuseScore's")
                print(f"  ##  MIDI export plays them out. Playing the .mxl "
                      f"directly avoids the whole problem.")
            print()

    if args.annotations:
        applied = load_annotations(args.annotations, notes)
        print(f"  applied {applied} per-note overrides from {args.annotations}")

    # Literal by default: expansion reaches the extremes of the calibrated
    # envelope, and the quiet end of it does not sound good on long notes.
    meta["dynamics"] = (expand_dynamic_range(notes) if args.expand_dynamics
                        else describe_dynamic_range(notes))

    assign_bowings(notes, args.bowing)
    notes, n_merged, n_portato = merge_slurs(
        notes, merge=not getattr(args, "no_merge_slurs", False))
    meta["n_slurs_merged"] = n_merged
    meta["n_portato_groups"] = n_portato
    meta["n_notes"] = len(notes)
    return notes, meta


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Play an open-A-string piece on the cello robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Always run --dry-run first and read the plan.")
    parser.add_argument("midi", nargs="?", default=str(DEFAULT_MIDI),
                        help="MIDI (or MusicXML) file to play. "
                             f"Default: {DEFAULT_MIDI}")
    parser.add_argument("--score", metavar="FILE.musicxml",
                        help="MusicXML score to take bowings, slurs and "
                             "dynamics from; timing still comes from the MIDI")
    parser.add_argument("--annotations", metavar="FILE.json",
                        help="per-note bowing/volume overrides")
    parser.add_argument("--bowing", choices=["alternate", "rule-of-downbow"],
                        default="alternate",
                        help="how to bow notes nothing has specified "
                             "(default: alternate)")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan and print, without connecting to the robot")
    parser.add_argument("--record", action="store_true",
                        help="record the performance to a wav")
    parser.add_argument("--tempo-scale", type=float, default=1.0,
                        help="stretch (>1) or compress (<1) all timings")
    parser.add_argument("--start-u", type=float, default=0.5,
                        help="where on the bow to start, 0=frog 1=tip "
                             "(default: 0.5, the middle)")
    parser.add_argument("--lead-in", type=float, default=LEAD_IN_SEC,
                        metavar="SECONDS",
                        help=f"silence between pressing ENTER and the first "
                             f"note (default: {LEAD_IN_SEC:.0f}). Recording and "
                             f"state logging run throughout, so the take gets a "
                             f"noise-floor reference and the stream is settled "
                             f"before anything sounds. Use 0 to start at once")
    parser.add_argument("--audio-channel", type=int, default=0, metavar="N",
                        help="which input channel of the interface to record "
                             "(1-based). Default 0 = pick the channel that "
                             "actually has signal. An interface returns a valid "
                             "but silent stream from an empty socket, so the "
                             "channel is always verified before the robot moves")
    parser.add_argument("--expand-dynamics", action="store_true",
                        help="stretch the range the score uses across the whole "
                             "usable one for maximum contrast. OFF by default: "
                             "it reaches the slowest bow at the lightest depth, "
                             "where long quiet notes stop speaking cleanly. A "
                             "narrower range of good tone beats a wider range "
                             "with bad tone in it")
    parser.add_argument("--no-lookahead", action="store_true",
                        help="do not look ahead over the coming phrase during "
                             "rests. By default the planner uses any rest of "
                             f"{BowPlanner.MIN_RESET_GAP}s or more to move the "
                             "bow somewhere the whole next phrase can be played "
                             "from, which is free — the bow is off the string "
                             "and nothing is sounding")
    parser.add_argument("--no-repeats", action="store_true",
                        help="do not expand repeat barlines and voltas in a "
                             "MusicXML/.mxl score. Repeats ARE expanded by "
                             "default, because MuseScore's MIDI export plays "
                             "them out and the two must agree")
    parser.add_argument("--no-swell", action="store_true",
                        help="do not render within-note hairpins as sub-moves")
    parser.add_argument("--retake-speed", type=float, default=RETAKE_SPEED,
                        metavar="M_S",
                        help="speed of the lift/traverse/set-down moves of a "
                             f"retake (default: {RETAKE_SPEED}, the pendant "
                             "reset speed). The bow is off the string for all "
                             "three, so this can go higher to fit retakes into "
                             "shorter gaps")
    parser.add_argument("--retake-accel", type=float, default=MOVE_ACCEL,
                        metavar="M_S2",
                        help="acceleration for the three moves of a retake "
                             f"(default: {MOVE_ACCEL}, the pendant figure). "
                             "This, not --retake-speed, is what actually makes "
                             "retakes quicker: the lift and traverse are so "
                             "short that they never reach cruising speed, so "
                             "their duration is 2*sqrt(distance/accel) and the "
                             "speed limit never binds")
    parser.add_argument("--accel-max", type=float, default=ACCEL_MAX,
                        metavar="M_S2",
                        help="ceiling on commanded tool acceleration "
                             f"(default: {ACCEL_MAX}, against the pendant-verified "
                             f"{MOVE_ACCEL}). Short notes are limited by this: "
                             "covering L metres in T seconds needs 4L/T^2, so "
                             "raising it is what lets fast notes keep their "
                             "dynamic. Raise it in small steps and listen")
    parser.add_argument("--servo", action="store_true",
                        help="HYBRID execution: stream long notes with servoL "
                             "so their speed and depth can change mid-note, "
                             "keeping moveL for short ones. Measured in air, "
                             "servoL tracks to -4%% at 1.0s and -8%% at 0.5s but "
                             "degrades below that, so the split is a win on "
                             "both sides. Off by default — moveL is the "
                             "verified path")
    parser.add_argument("--servo-threshold", type=float, default=None,
                        metavar="SEC",
                        help="note length at and above which --servo streams "
                             "(default: 0.5). Note a correction based on the "
                             "note's own sound cannot land before ~0.6s, so "
                             "notes under ~1s cannot really be reacted to")
    parser.add_argument("--servo-lookahead", type=float, default=None,
                        metavar="SEC",
                        help="servoL lookahead_time (default: 0.10). Measured "
                             "0.03 tracks markedly better: the bow lags by "
                             "about speed x lookahead of distance")
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        metavar="X",
                        help="multiply every note's bow speed, and therefore "
                             "the bow it uses, by this (default: 1.0). Faster "
                             "and lighter is the clearer way to make a given "
                             "loudness — the residual RL converged on exactly "
                             "that (+25%% speed, lighter depth) — but each "
                             "stroke then eats more hair, so watch the plan "
                             "for retakes and shortfalls. Pair with a negative "
                             "--depth-offset to hold the loudness steady")
    parser.add_argument("--depth-offset", type=float, default=0.0,
                        metavar="MM",
                        help="shift every note's depth by this many mm, "
                             "negative for lighter contact, positive to lean "
                             "in (default: 0). Depth is otherwise a pure "
                             "function of volume, so a piece marked pp "
                             f"throughout sits near "
                             f"{-MAX_OUTWARD_DEPTH*1000:+.1f}..{MAX_INWARD_DEPTH*1000:+.1f} mm "
                             "with no way to trim it. Clamped to the safety "
                             "envelope; per-note depth_mm annotations are NOT "
                             "shifted")
    parser.add_argument("--no-merge-slurs", action="store_true",
                        help="keep slurred notes separate instead of collapsing "
                             "each group into one sustained tone. They still "
                             "share a bow direction (portato). Use when a slur "
                             "ends on a short note and you want to hear that "
                             "note rather than have it absorbed")
    parser.add_argument("--articulation-ref", type=float,
                        default=ARTICULATION_REF, metavar="SEC",
                        help="note length at and above which a written "
                             f"articulation is taken literally (default: "
                             f"{ARTICULATION_REF}). Shorter notes get "
                             "proportionally less separation, so fast staccato "
                             "is articulated rather than chopped into blips. "
                             "0 takes every mark literally at any tempo")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt before playing")
    args = parser.parse_args(argv)

    print("\nRobot Cello — open A string, whole piece")
    print(f"  waypoints    BOW_CONFIGS['{CONFIG_NAME}'] from recording_a_only.py")
    print(f"  installation VERIFIED_TCP / {_rec.VERIFIED_PAYLOAD_KG} kg payload")
    print()

    _verify_geometry(verbose=True)
    print()

    notes, meta = build_score(args)
    if not notes:
        print("No notes found.")
        return 1

    planner = BowPlanner(start_u=args.start_u, enable_swell=not args.no_swell,
                         accel_max=args.accel_max,
                         retake_speed=args.retake_speed,
                         retake_accel=args.retake_accel,
                         lookahead=not args.no_lookahead,
                         articulation_ref=args.articulation_ref,
                         depth_offset=args.depth_offset / 1000.0,
                         speed_scale=args.speed_scale)
    strokes = planner.plan(notes)
    print_plan(strokes, notes, meta)

    if args.dry_run:
        print("\nDry run — the robot was not contacted.")
        return 0

    total = max(s.onset + s.duration for s in strokes) + 2.0

    # Verify the audio chain BEFORE the robot is asked to do anything. A run
    # that plays perfectly and records nothing is a wasted session, and an
    # interface gives no error for a channel with nothing plugged into it.
    audio_device = None
    audio_channel = args.audio_channel or 1
    if args.record:
        print("\nChecking the audio input...")
        try:
            audio_device = _rec.find_focusrite_device()
            audio_channel, levels = select_audio_channel(
                audio_device, args.audio_channel or None)
            report_audio_channels(levels, audio_channel)
        except Exception as e:
            print(f"  could not check the audio input: {e}")
            levels = []

        if not audio_channel:
            print("\n  ##  No input channel is carrying any signal.")
            print("  ##  Check: the mic is plugged in, the interface gain is up,")
            print("  ##  48V phantom power is on for a condenser mic, and macOS")
            print("  ##  has granted microphone access to the terminal.")
            print("  ##  Recording now would produce a silent file.")
            answer = input("  Continue anyway, without usable audio? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Aborted — the robot has not been touched.")
                return 1
            audio_channel = args.audio_channel or 1

    if not args.yes:
        answer = input(f"\nPlay {len(strokes)} strokes over {total:.1f} s? "
                       f"Check the robot by eye first. [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 0

    player = PiecePlayer(record_audio=args.record,
                         retake_speed=args.retake_speed,
                         retake_accel=args.retake_accel,
                         audio_device=audio_device,
                         audio_channel=audio_channel,
                         servo=args.servo,
                         servo_threshold=args.servo_threshold,
                         servo_lookahead=args.servo_lookahead)
    try:
        player.prepare(strokes[0])
        input("\nPress ENTER to play (Ctrl+C to abort)...")
        elapsed = player.play(strokes, total, lead_in=args.lead_in)

        print(f"\nDone in {elapsed:.2f} s (planned {total-2.0:.2f} s).")
        drifts = [entry["drift"] for entry in player.timeline]
        if drifts:
            print(f"  timing drift  {min(drifts):+.3f} .. {max(drifts):+.3f} s, "
                  f"final {drifts[-1]:+.3f} s")
        player.save(meta, elapsed)

    except KeyboardInterrupt:
        print("\nAborted by the user.")
        player.logger.stop()
    except RobotFaultStop as e:
        print("\n" + "!" * 60)
        print("ROBOT FAULT — STOPPED")
        print("!" * 60)
        print(str(e))
        print("Check the robot by eye and on the teach pendant before doing")
        print("anything else.")
        print("!" * 60)
        raise
    finally:
        player.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
