"""
rl/piece_env.py

Residual RL over a whole piece, one bow stroke per RL step.

The baseline is BaselineControls/play_midi_pieces.py: its score parsing
(MIDI / MusicXML, bowings, slurs, dynamics) and its BowPlanner produce the
strokes the piece *should* be played with. This environment lets a policy
apply a small residual to each stroke — a speed scale and a depth offset,
the two knobs the one-stroke RL (other_github/rl/sac) plugs in — and is
rewarded primarily by the sound classifier's tone-quality score.

The score's intent is never overridden, only shaded:
    - bow DIRECTIONS (the written/planned bowings) are kept exactly
    - note ONSETS and DURATIONS are kept exactly (rhythm is sacred)
    - mean speed may move ±SPEED_RESIDUAL_FRAC around the planned value,
      which by the dataset calibration is about ±1.6 dB of shading
    - depth may move ±DEPTH_RESIDUAL_M, clamped to the recording script's
      safety envelope [-MAX_OUTWARD_DEPTH, +MAX_INWARD_DEPTH]

Because speed × duration = bow length, a speed residual changes how much bow
a stroke consumes, so the environment re-tracks the bow position u stroke by
stroke and clamps any stroke that would run past U_MIN/U_MAX (with a reward
penalty), exactly mirroring the planner's own budget discipline. Retakes
re-anchor the bow at their planned landing point, so drift cannot compound
across a rest.

Episode  = one pass through the piece.
Step     = one stroke:  obs -> action (residual) -> execute -> classify -> reward.

Executors:
    MockExecutor      no hardware; synthesizes physical params. Pair with
                      MockScorer, whose sweet spot in (speed, depth) gives
                      SAC a learnable signal for end-to-end testing —
                      the same idea as the partner code's MockSoundClassifier,
                      extended to both knobs.
    HardwareExecutor  rl/piece_hardware.py — real UR5e via PiecePlayer's
                      verified motion primitives + microphone capture.

Scorers:
    RealScorer        SoundClassifier/classifier.py BowingQualityClassifier:
                      predict(audio_500ms, physical_params_6, string) -> [0,1].
                      Falls back to its own heuristic when no checkpoint.
    MockScorer        physics-only stand-in for training without a mic.

Run from the repository environment created by scripts/setup.sh or
scripts/setup.ps1 (importing the baseline pulls in sounddevice; ur_rtde is
stubbed in mock mode when the optional hardware profile is not installed).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ══════════════════════════════════════════════════════════════════
# Baseline import — the planner IS the baseline, loaded, not copied
# ══════════════════════════════════════════════════════════════════

def _stub_rtde_if_missing():
    """
    Mock-mode fallback: baseline_controller does `import rtde_control` /
    `import rtde_receive` at module level, but never touches the robot until
    a controller is constructed. On a machine without the ur_rtde build the
    mock pipeline is still perfectly usable, so install loud stub modules
    rather than dying at import. Anything that actually reaches for the robot
    raises immediately, and piece_hardware refuses to run against a stub.
    """
    import types
    stubbed = []
    for name in ("rtde_control", "rtde_receive", "rtde_io"):
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            module._IS_CELLO_RL_STUB = True

            def _raise(*_a, _name=name, **_k):
                raise RuntimeError(
                    f"{_name} is a mock-mode stub — the real ur_rtde build is "
                    f"not importable in this environment, so the robot cannot "
                    f"be used. Mock training is unaffected.")

            def _getattr(attr, _r=_raise):
                # Behave like a normal module for dunder probes (__file__,
                # __spec__, ...) so inspect/pickle machinery is not confused;
                # only robot-API lookups get the raiser.
                if attr.startswith("__") and attr.endswith("__"):
                    raise AttributeError(attr)
                return _r

            module.__getattr__ = _getattr
            sys.modules[name] = module
            stubbed.append(name)
    if stubbed:
        print(f"(ur_rtde not importable — stubbed {', '.join(stubbed)}; "
              f"mock mode only, hardware runs will refuse to start)")


def load_baseline_module():
    """
    Load BaselineControls/play_midi_pieces.py by path (it is not a package).

    This transitively imports the verified setup from recording_a_only
    (waypoints, depth convention, safe_moveL, sounddevice), which is
    the point: the RL baseline cannot drift from what was verified on the
    pendant. Without the ur_rtde build only mock mode is available.
    """
    _stub_rtde_if_missing()
    path = REPO_ROOT / "BaselineControls" / "play_midi_pieces.py"
    spec = importlib.util.spec_from_file_location("play_midi_pieces", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        raise ImportError(
            f"Could not import the baseline planner ({e}).\n"
            f"It pulls in ur_rtde and sounddevice — run with\n"
            f"  python rl/play_piece.py ..."
        ) from e
    sys.modules.setdefault("play_midi_pieces", module)
    return module


PMP = load_baseline_module()


# ══════════════════════════════════════════════════════════════════
# Residual bounds
# ══════════════════════════════════════════════════════════════════

# Bow speed residual, as a fraction of the planned mean speed.
#
# Was 0.20 (±1.6 dB), chosen to match the planner's BOW_SLACK so a residual
# could never outrun the bow budget. That turned out to be too timid: the
# instrument only spans 8.9 dB in total, real music sits in the middle of it,
# and on t1 the dynamics term was near-saturated because the policy could not
# move far enough to matter. ±35% is ±2.6 dB — still inside the calibrated
# 0.09..0.25 m/s grid for any mid-range note, so the mapping stays
# interpolation rather than extrapolation.
#
# The cost is bow budget: a 35% longer stroke eats 35% more hair, so expect
# more DISTRIB/FLIP events. r_bow already prices that in.
SPEED_RESIDUAL_FRAC = 0.35

# ±1.0 mm of depth around the planned value (was ±0.5). Depth is worth only
# ~0.42 dB/mm so this is not a dynamics lever — it is the timbral one, and the
# difference between a note that speaks and one that does not. Widening it
# matters most on slow quiet notes, where more weight is the classical way to
# keep the string speaking.
DEPTH_RESIDUAL_M = 0.0010

DEPTH_LO = -PMP.MAX_OUTWARD_DEPTH   # -1.5 mm (lighter)
DEPTH_HI = PMP.MAX_INWARD_DEPTH     # +2.0 mm (heavier)

# Reward weights. Quality is the headline; dynamic accuracy anchors the
# policy to the written dynamics so "best tone" cannot be bought by playing
# everything at one comfortable loudness.
W_QUALITY = 0.50
W_DYNAMIC = 0.25
W_BOW     = 0.15
W_SMOOTH  = 0.10

# ── Multi-dimensional tone reward ─────────────────────────────────
# The classifier predicts six human-rated dimensions but the reward only ever
# used `overall`. That was measurably the wrong choice: on challengepiece,
# `overall` ranked the runs almost exactly opposite to `tone_quality`, and
# opposite to the ear.
#
#     run              overall   tone_quality
#     original           0.838      0.389
#     tempo fixed        0.684      0.736     <- ear preferred, tone agreed
#
# `attack_quality` moved the same way (0.22 -> 0.55) when the articulation was
# fixed, so it is carrying real signal about exactly the harsh bow changes the
# reward was blind to. Weights sum to 1.0 and are applied to whichever heads
# the checkpoint actually provides.
TONE_HEAD_WEIGHTS = {
    "tone_quality":   0.45,
    "attack_quality": 0.25,
    "release_quality": 0.15,
    "overall":        0.15,
}

# ── Objective defect penalties ────────────────────────────────────
# The 40 hand-engineered features are computed on every stroke and then thrown
# away — only the CNN's scalar survives. These four are monotonic (more or less
# is unambiguously better), need no human labels, and work at ANY window
# length, which is what makes them useful exactly where the CNN is weakest:
# notes too short to fill its 500 ms window.
#
# Deliberately excluded: spectral flatness (you want SOME noise — bow noise is
# part of cello timbre, and driving it to zero gives a sterile sound) and
# envelope_cv (a swell has high CV by design). heuristic_quality_score puts
# 0.30 weight on minimising flatness, which is the most questionable part of it.
#
# (feature, direction, good_value, bad_value) — direction +1 means higher is
# better. Scored linearly between good and bad, clipped to [0, 1].
DEFECT_FEATURES = [
    # (feature, direction, good, bad, weight) — weights let one defect be
    # priced harder than the others without re-scaling W_DEFECT: the
    # dynamics-vs-scratchiness imbalance (writeup-aug12 §4: ~6:1 in favour
    # of staying quiet and scratchy) is tuned here, via hnr's weight.
    # (weight column added by Zixian, 2026-08-13; defaults unchanged.)
    ("hnr_db_mean",        +1, 12.0,  3.0, 1.0),   # scratchy when low
    ("voiced_fraction",    +1,  0.98, 0.70, 1.0),  # string not speaking
    ("attack_overshoot",   -1,  0.85, 1.40, 1.0),  # harsh attack
    ("f0_stability_cents", -1,  1.0, 20.0, 1.0),   # pitch wobble
]
W_DEFECT = 0.15   # scaled against the tone term; applied as a penalty

# ── Onset acceleration penalty ────────────────────────────────────
# Nothing in the reward has ever priced the harshness of a bow change. Each
# stroke starts from rest and solve_stroke raises acceleration up to ACCEL_MAX
# to make short notes fit — and higher acceleration IS a harder attack. This
# makes that trade explicit instead of letting the planner choose harshness
# silently. Zero penalty at the pendant-verified MOVE_ACCEL, full at ACCEL_MAX.
W_ONSET_ACCEL = 0.05

# Envelope jerk. Small on purpose: it should discourage a jagged shape without
# flattening the envelope into the single value it replaced.
W_ENVELOPE = 0.05
# Segment speed below SPEED_MIN -> the string stops speaking.
#
# DORMANT at the current bounds: SPEED_RESIDUAL_FRAC 0.35 makes the most
# lopsided envelope w = [0.65, 1.35, 1.35], whose slowest segment still clears
# SPEED_MIN. Measured 0 firings over 40 enveloped strokes of t1 and
# string_crossings under the worst-case action. It is insurance for a wider
# residual range, not a term that currently does anything -- and note it prices
# COMMANDED speed, so it would not catch an amplitude collapse arriving by any
# other route. The reward is still blind on the audio side (RL_METHOD.md #2).
W_SPEAK = 0.12

# A dynamic error of this many dB zeroes the dynamic reward term.
DYNAMIC_FULL_ERR_DB = 3.0

# The classifier scores a fixed 0.5 s window (SoundClassifier's convention, and
# what dataset_a_final's sustained full-bow strokes filled). A note SHORTER
# than that cannot fill it: the rest of the window is neighbouring notes and
# silence, which the model never saw in training. Scoring an 0.08 s note this
# way is mostly measuring its neighbours.
#
# So the tone term is shrunk toward neutral in proportion to how much of the
# window the note actually occupies:
#
#     quality_eff = 0.5 + fill * (quality - 0.5),   fill = min(1, dur / 0.5)
#
# A full-length note is unaffected (fill=1). An 0.08 s note contributes ~0.16
# of its deviation, so it can neither reward nor punish the policy much on
# evidence that is mostly not about it. Reward SCALE is unchanged, so returns
# stay comparable across pieces; only the misleading gradient is removed. The
# dynamic and bow terms are untouched — those are measured from motion, not
# audio, and are perfectly valid on a short note.
CLASSIFIER_WINDOW_SEC = 0.5

# How much of the tone signal survives on the SHORTEST notes.
#
# What this term means CHANGED once the executor started cutting the analysis
# window to the note (piece_hardware.NOTE_MATCHED_WINDOW). It used to stand for
# "most of this 500 ms window is other notes, so distrust it" — a fixed window
# around an 80 ms note is 420 ms of its neighbours, and scoring that is scoring
# the wrong notes. That contamination is now gone: the window IS the note.
#
# What is left is only that the hand-engineered features get less reliable as
# the window shrinks — fewer pitch periods for F0 stability and HNR. Measured
# on the standard config that costs little: rho 0.824 at 0.10 s against 0.835
# at 0.50 s. Hence a light floor rather than the heavy shrink this started as.
WINDOW_FILL_FLOOR = 0.85

# The acceleration ceiling solve_stroke plans against. Used to work out the
# fastest MEAN speed a note of a given duration can reach (accel_max*T/4), so
# the dynamic target is never set beyond what physics allows.
ACCEL_MAX_FOR_DYNAMICS = PMP.ACCEL_MAX

# 22 = the original 16 score/state dims + the FULL previous action (6). It
# was 18 with only prev_action[0:2] — sensible for the old [speed, depth]
# action, but after the envelope change those two slots held segment-0/1
# SPEED residuals and the depth half of the action was invisible to the
# policy. That made r_smooth's depth component unpredictable from (s, a) —
# reward noise whose only deterministic fix is a CONSTANT depth action
# ("depth shaping went unused" is that attractor's signature) — and blocked
# any incremental depth correction. (Zixian, 2026-08-13; checkpoints were
# already invalidated by the 2->6 action change, so this costs nothing now.)
OBS_DIM = 22

# ── Per-segment envelope control ──────────────────────────────────
# The policy used to emit ONE (speed, depth) for a whole note, which meant it
# could not shape anything within a stroke — no ramped attack, no pressure
# moving through the note. That is the expressive limit that motivated the
# servoL work, and servoL turned out to cost real tone (0.840 -> 0.617
# tone_quality on string_crossings, no lookahead setting recovering it).
#
# But moveL can already shape within a note: a blended moveL PATH carries a
# speed AND a depth per waypoint and executes as one continuous motion. That is
# how the planner renders swells. So the expressiveness servoL was wanted for
# is available feed-forward, at moveL's tone quality, on EVERY note rather than
# only the ~2% long enough for closed-loop feedback to arrive in time.
#
# The action is therefore an envelope: a (speed, depth) residual per segment.
# Segment 0 is always the attack — for a note too short to segment it is the
# whole note, so dim 0 means "the attack" consistently regardless of length.
N_ENVELOPE_SEGMENTS = 3

# Below this the note cannot usefully be split. Three segments of a 0.25 s note
# are ~12 mm of bow each, and the blend radius (up to 5 mm) would be a large
# fraction of that — the segments would smear into each other. Short notes take
# segment 0's residual applied to the whole stroke, which is the right reading
# anyway: a 0.08 s note IS its attack.
ENVELOPE_MIN_DURATION = 0.40

ACTION_DIM = 2 * N_ENVELOPE_SEGMENTS   # [spd x3, depth x3], each in [-1, 1]


# ══════════════════════════════════════════════════════════════════
# Executor / scorer interfaces
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExecStroke:
    """One stroke as actually commanded (baseline plan + residual applied)."""
    note_index: int
    direction: str                  # 'down' | 'up' — from the plan, never changed
    u_start: float
    u_end: float
    length: float                   # m
    duration: float                 # s, from the score (rhythm preserved)
    mean_speed: float               # m/s — what sets the loudness
    speed: float                    # m/s commanded peak
    accel: float                    # m/s^2 commanded
    depth: float                    # m signed, clamped to safety envelope
    volume_target: float            # 0..1, what the score asked for
    retake_from: float | None = None
    onset: float = 0.0
    gap_before: float = 0.0         # s of silence before this stroke
    segments: list = field(default_factory=list)   # PMP.Segment list


@dataclass
class StrokeResult:
    """What came back from executing one stroke."""
    audio: np.ndarray | None        # ~500 ms peak-normalized window, or None (mock)
    physical: np.ndarray            # (6,) in dataset.PHYSICAL_FEATURE_NAMES order
    measured_mean_speed: float
    achieved_u_end: float
    # RMS level of the RAW stroke capture, before peak normalisation. This is
    # what closes the dynamics loop acoustically; None in mock mode without a
    # loudness model, in which case the reward falls back to the speed proxy.
    measured_dbfs: float | None = None
    # Per-segment (speed, depth) actually executed. The real classifier does
    # not need this — it HEARS the shape. MockScorer does, because it scores
    # from commanded numbers and would otherwise be blind to any envelope,
    # making a flat envelope trivially optimal in mock training.
    segment_profile: list | None = None


class ExecutorBase:
    def begin_episode(self, first: ExecStroke):
        pass

    def execute(self, stroke: ExecStroke) -> StrokeResult:
        raise NotImplementedError

    def end_episode(self):
        pass

    def close(self):
        pass


class MockExecutor(ExecutorBase):
    """
    No hardware. Physical params are synthesized from the command:

        [depth_m, 0, mean_speed, mid_u, torque_est, torque_est*1.3]

    which mirrors dataset.build_physical_features for records without a
    force_contact block (slot 0 = commanded depth). Torque grows with depth
    so a depth-sensitive classifier sees a consistent signal.
    """

    def __init__(self, noise: float = 0.02, rng: np.random.Generator | None = None):
        self.noise = noise
        self.rng = rng or np.random.default_rng()
        # Synthesize a level from the fitted loudness model so the closed-loop
        # dynamics reward can be exercised without the robot. Optional: without
        # the model the env falls back to the bow-speed proxy.
        try:
            from rl.loudness import get_model
            self._loudness = get_model()
        except FileNotFoundError:
            self._loudness = None

    def execute(self, stroke: ExecStroke) -> StrokeResult:
        n = lambda s: 1.0 + self.noise * self.rng.standard_normal() * s
        mean_speed = stroke.mean_speed * n(1.0)
        mid_u = (stroke.u_start + stroke.u_end) / 2.0
        # crude contact-torque model: rises with depth into the string
        torque = max(0.0, 0.08 + 60.0 * (stroke.depth - DEPTH_LO)) * n(2.0)
        physical = np.array([
            stroke.depth, 0.0, mean_speed, mid_u, torque, torque * 1.3,
        ], dtype=np.float32)
        dbfs = None
        if self._loudness is not None:
            # Model prediction plus its own residual scatter, so the mock is
            # no more precise than the real measurement it stands in for.
            dbfs = (self._loudness.predict_dbfs(mean_speed, stroke.depth)
                    + self._loudness.residual_sd_db * self.rng.standard_normal())

        return StrokeResult(
            audio=None,
            physical=physical,
            measured_mean_speed=mean_speed,
            achieved_u_end=stroke.u_end,
            measured_dbfs=dbfs,
            segment_profile=[
                (abs(sg.u_end - sg.u_start) * PMP.BOW_LENGTH / max(sg.duration, 1e-6),
                 sg.depth) for sg in stroke.segments],
        )


class MockScorer:
    """
    Stand-in for the sound classifier: tone quality peaks at a sweet spot in
    (depth, speed) and degrades away from it, with noise. Gives SAC a real
    gradient to climb so the whole pipeline can be validated hardware-free.
    """

    DEPTH_STAR = 0.0004     # m — slightly into the string
    DEPTH_SIGMA = 0.0007
    SPEED_BREAKUP = 0.07    # m/s — tone breaks up below this

    def __init__(self, noise: float = 0.03, rng: np.random.Generator | None = None):
        self.noise = noise
        self.rng = rng or np.random.default_rng()

    def score_detailed(self, audio, physical, window_pos=0.5,
                       string="A", segment_profile=None) -> dict:
        """Mirror RealScorer's richer contract so mock runs exercise the same
        reward path. Heads are perturbations of the base score; defect features
        are synthesized near their good values so the mock does not fabricate
        defects the physics model knows nothing about.

        `segment_profile` makes the mock able to SEE an envelope. Without it a
        flat envelope is trivially optimal in mock training — the score comes
        from stroke-level means, so no shape can move it — and a mock run then
        proves only that the plumbing works, not that the action space is
        learnable. The preference encoded here is deliberately the textbook
        one: a note that starts a little slower and lighter than it ends
        speaks more cleanly than one struck at full speed."""
        base = self.score(audio, physical, window_pos, string)
        attack_bonus = 0.0
        if segment_profile and len(segment_profile) > 1:
            (v0, d0), (v1, d1) = segment_profile[0], segment_profile[-1]
            # Reward the first segment being gentler than the last, saturating
            # so an absurdly slow attack is not rewarded without limit.
            ramp = (v1 - v0) / max(abs(v1) + abs(v0), 1e-6)
            lighten = (d1 - d0) / max(DEPTH_RESIDUAL_M * 2, 1e-6)
            attack_bonus = 0.15 * float(np.clip(ramp, -1, 1)) \
                + 0.05 * float(np.clip(lighten, -1, 1))
        out = {h: float(np.clip(base + 0.05 * self.rng.standard_normal(), 0, 1))
               for h in ("overall", "tone_quality", "attack_quality",
                         "release_quality", "bow_control", "dynamic_accuracy")}
        # The envelope shows up where it would in reality: the attack head.
        for h in ("attack_quality", "overall"):
            out[h] = float(np.clip(out[h] + attack_bonus, 0.0, 1.0))
        for feat, direction, good, bad, _w in DEFECT_FEATURES:
            # Slide between good and bad with the base score, so a bad mock
            # stroke also looks defective.
            out[feat] = float(bad + (good - bad) * base)
        return out

    def score(self, audio, physical, window_pos=0.5, string="A") -> float:
        depth, speed = float(physical[0]), float(physical[2])
        q = np.exp(-0.5 * ((depth - self.DEPTH_STAR) / self.DEPTH_SIGMA) ** 2)
        if speed < self.SPEED_BREAKUP:
            q *= max(0.0, speed / self.SPEED_BREAKUP) ** 2
        q += self.noise * self.rng.standard_normal()
        return float(np.clip(q, 0.0, 1.0))


# Preferred RL reward checkpoint. SoundClassifier's own DEFAULT_CHECKPOINT is
# quality_cnn.pt, which is trained on pseudo_heuristic_v1 PSEUDO-LABELS — its
# near-perfect metrics (val_mse 0.0003, rho 0.99) are it re-learning a
# deterministic heuristic, not human judgement of tone. Optimising a policy
# against it would maximise that heuristic, which is not what "best tone"
# means here. The human-annotated checkpoint (399 ratings from 2 annotators,
# recording-level Spearman rho 0.70) is the honest reward signal, so it is
# preferred when present.
#
# Preference order, best first: the A1-A5 model is trained on all five
# annotators (500 recordings, 999 ratings, 499 of them double-rated) and beats
# the two-annotator model on every headline metric — recording-level Spearman
# 0.798 vs 0.702, val MSE 0.0242 vs 0.0355 — including 0.847 on the 'standard'
# bow config, which is the one CONFIG_NAME selects and therefore the only one
# the RL ever plays in.
# The multilength model is preferred now that the executor feeds NOTE-MATCHED
# windows: it was trained on 0.10/0.20/0.35/0.50 s windows, which is the
# distribution it now actually sees. Measured on the standard config the two
# are equal (mean rho 0.830 each), but across all five configs the multilength
# one is both better and far flatter with length (0.782 vs 0.769 mean, spread
# 0.006 vs 0.018) — so it is the safer choice when window length varies per
# note rather than being fixed.
HUMAN_CHECKPOINTS = [
    REPO_ROOT / "SoundClassifier" / "checkpoints" / "quality_cnn_multilength.pt",
    REPO_ROOT / "SoundClassifier" / "checkpoints" / "quality_cnn_human_A1_A5.pt",
    REPO_ROOT / "SoundClassifier" / "checkpoints" / "quality_cnn_human_current.pt",
]
HUMAN_CHECKPOINT = next((p for p in HUMAN_CHECKPOINTS if p.exists()),
                        HUMAN_CHECKPOINTS[-1])


class RealScorer:
    """
    Wraps SoundClassifier/classifier.py's BowingQualityClassifier — the real
    reward source. Without a checkpoint the classifier falls back to its own
    scalar-feature heuristic.

    Checkpoint choice is reported loudly, and pseudo-label checkpoints are
    warned about, because which one is loaded decides what the policy is
    actually being trained to maximise.
    """

    def __init__(self, checkpoint_path=None, device="cpu"):
        sc_dir = REPO_ROOT / "SoundClassifier"
        if str(sc_dir) not in sys.path:
            sys.path.insert(0, str(sc_dir))
        from classifier import BowingQualityClassifier, DEFAULT_CHECKPOINT

        if checkpoint_path is None:
            checkpoint_path = (HUMAN_CHECKPOINT if HUMAN_CHECKPOINT.exists()
                               else DEFAULT_CHECKPOINT)
        self.checkpoint_path = Path(checkpoint_path)
        self.clf = BowingQualityClassifier(checkpoint_path=checkpoint_path,
                                           device=device)

        self.label_source = "unknown"
        try:
            import torch
            meta = torch.load(checkpoint_path, map_location="cpu",
                              weights_only=False)
            self.label_source = meta.get("label_source", "unknown")
            if meta.get("pseudo_labels"):
                print(f"##  WARNING: {self.checkpoint_path.name} is trained on "
                      f"PSEUDO-LABELS ({self.label_source}). A policy trained "
                      f"against it maximises a heuristic, not human-judged "
                      f"tone. Prefer {HUMAN_CHECKPOINT.name}.")
            else:
                print(f"Reward checkpoint: {self.checkpoint_path.name} "
                      f"(label_source={self.label_source})")
        except Exception:
            pass

    def score(self, audio, physical, window_pos=0.5, string="A") -> float:
        return float(self.clf.predict(audio, physical, string=string,
                                      window_pos=window_pos))

    def score_detailed(self, audio, physical, window_pos=0.5,
                       string="A") -> dict:
        """
        Every head the checkpoint provides, plus the objective defect features.

        Two independent signals come back. The heads are human judgement and
        need the 500 ms window to be meaningful. The features are physics and
        work at any length — which is what lets a short note be graded for
        scratchiness and harsh attack even when its tone score cannot be
        trusted.
        """
        out = dict(self.clf.predict_detailed(audio, physical, string=string,
                                             window_pos=window_pos))
        if audio is not None and len(audio) > 0:
            import audio_features as af
            scalars = af.extract_scalar_features(
                np.asarray(audio, dtype=np.float32), sr=af.SR)
            names = af.SCALAR_FEATURE_NAMES
            for feat, *_ in DEFECT_FEATURES:
                out[feat] = float(scalars[names.index(feat)])
        return out


# ══════════════════════════════════════════════════════════════════
# Score loading + baseline planning
# ══════════════════════════════════════════════════════════════════

def calibrate_dynamics(notes) -> dict:
    """
    Re-aim every note at the loudness its dynamic is supposed to produce.

    The planner turns a written dynamic into a bow speed with
    volume_to_speed(), a linear rule across an absolute ppp..fff axis. Real
    music lives between p and f, the middle 43% of that axis, so a p..f piece
    asks for 0.138..0.206 m/s -- 3.5 dB, which is audibly no dynamics at all.
    Worse, it is open-loop: measured against the fitted model, the baseline
    plays p about 3 dB too loud.

    This inverts the loudness model instead: for each note, take the centre of
    its dynamic's zone, solve for the bow speed that actually produces that
    level, and rewrite the note's volume so the planner's own machinery
    (bow budgeting, stroke solving, swells) produces it. p..f then spans the
    full ~10 dB the instrument can reach rather than 3.5 dB.

    This is FEED-FORWARD -- it aims correctly before the stroke. The RL
    residual then only has to correct what the model got wrong, rather than
    discover the whole speed->loudness mapping from scratch.
    """
    from rl.loudness import get_model
    model = get_model()

    # The planner derives BOTH speed and depth from `volume`, so re-aiming
    # volume at a speed target drags depth along with it. Measured on t1 that
    # put 7 of 25 strokes into slow-bow-AND-light-depth simultaneously -- the
    # corner expand_dynamic_range()'s docstring warns is "where the string
    # stops speaking cleanly... thin and unsteady rather than soft". Depth is
    # worth only ~0.42 dB/mm by the fit, so it is a poor dynamic lever anyway.
    # Keep the original volumes for depth and let speed carry the dynamics.
    original_volumes = [float(n.volume) for n in notes]

    changed = 0
    for note in notes:
        zone = model.zone_for_dynamic(note.dynamic)
        target_speed = model.invert_speed(model.zone_centre(zone))
        volume = PMP.speed_to_volume(target_speed)
        if abs(volume - note.volume) > 1e-6:
            changed += 1
        # Keep any within-note swell by shifting its end by the same amount.
        if note.volume_end is not None:
            note.volume_end = float(np.clip(
                note.volume_end + (volume - note.volume), 0.0, 1.0))
        note.volume = volume
    return {"calibrated": True, "notes_reaimed": changed,
            "original_volumes": original_volumes}


def load_piece(path: str, tempo_scale: float = 1.0,
               bowing_rule: str | None = None,
               calibrated_dynamics: bool = False):
    """
    Parse a piece the same way play_midi_pieces.py does and plan it.

    Returns (notes, meta, strokes) where strokes is the baseline plan the
    residual policy shades. .mxl/.musicxml/.xml files carry written bowings,
    slurs and dynamics; .mid carries rhythm + velocities (+ CC2 hairpins).
    """
    p = Path(path)
    if p.suffix.lower() in (".mxl", ".musicxml", ".xml"):
        notes, meta = PMP.parse_musicxml(str(p), tempo_scale=tempo_scale)
    else:
        notes, meta = PMP.parse_midi(str(p), tempo_scale=tempo_scale)

    if bowing_rule:
        PMP.assign_bowings(notes, bowing_rule)

    notes, n_merged, n_portato = PMP.merge_slurs(notes)

    # AFTER merging, so original_volumes is indexed the same way the planner's
    # stroke.note_index will be.
    if calibrated_dynamics:
        meta["dynamics_calibration"] = calibrate_dynamics(notes)
    meta["n_slurs_merged"] = n_merged
    meta["dynamics"] = PMP.describe_dynamic_range(notes)

    planner = PMP.BowPlanner()
    strokes = planner.plan(notes)
    return notes, meta, strokes


# ══════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════

class PieceResidualEnv(gym.Env):
    """
    Observation (18,), roughly normalized to [-1, 1]:
        0  planned duration / 2 s
        1  planned mean speed / SPEED_MAX
        2  planned depth / DEPTH_HI
        3  direction (+1 down, -1 up)
        4  current bow position u
        5  available bow fraction in this stroke's direction
        6  volume target (0..1)
        7  volume at note end (== target when no swell)
        8  1 if this stroke starts with a retake
        9  1 if this stroke is part of a split note
        10 next stroke volume (0 at end)
        11 next stroke duration / 2 s
        12 next stroke direction (0 at end)
        13 last stroke's quality score
        14 EMA of quality this episode
        15 progress through the piece (0..1)
        16 previous action[0]
        17 previous action[1]

    Action (2,) in [-1, 1]:
        0  speed residual  -> mean speed × (1 + a0 · SPEED_RESIDUAL_FRAC)
        1  depth residual  -> depth + a1 · DEPTH_RESIDUAL_M (clamped to envelope)

    Reward per stroke:
        W_QUALITY · classifier score
      + W_DYNAMIC · (1 - |achieved-vs-written dB error| / 3 dB)
      - W_BOW     · bow-budget violation (shortfall + limit proximity)
      - W_SMOOTH  · action jerk between consecutive strokes
    """

    metadata = {"render_modes": []}

    def __init__(self, piece_path: str, executor: ExecutorBase | None = None,
                 scorer=None, tempo_scale: float = 1.0,
                 bowing_rule: str | None = None, ema_alpha: float = 0.7,
                 string: str = "A", closed_loop_dynamics: bool = True,
                 calibrated_dynamics: bool = False):
        super().__init__()
        self.piece_path = str(piece_path)
        self.executor = executor or MockExecutor()
        self.scorer = scorer or MockScorer()
        self.string = string
        self.ema_alpha = ema_alpha

        # Closed-loop dynamics need the fitted loudness model. Absent, the
        # reward silently falls back to the bow-speed proxy -- say so, rather
        # than let a run quietly grade dynamics the old way.
        self.loudness = None
        if closed_loop_dynamics:
            try:
                from rl.loudness import get_model
                self.loudness = get_model()
                print(f"Dynamics: closed loop on measured dBFS "
                      f"(zones from {self.loudness.path.name}, "
                      f"residual sd {self.loudness.residual_sd_db:.2f} dB)")
            except FileNotFoundError as e:
                print(f"Dynamics: OPEN loop on bow speed -- {e}")

        self.notes, self.meta, self.plan = load_piece(
            piece_path, tempo_scale=tempo_scale, bowing_rule=bowing_rule,
            calibrated_dynamics=calibrated_dynamics)
        if not self.plan:
            raise ValueError(f"{piece_path}: the planner produced no strokes")

        # Depth stays on the ORIGINAL written dynamics even when speed has
        # been re-aimed, so calibration cannot drag quiet notes into the
        # slow-and-light corner where the string stops speaking.
        cal = self.meta.get("dynamics_calibration") or {}
        self._depth_volumes = cal.get("original_volumes")

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,),
                                       dtype=np.float32)

        self._i = 0
        self._u = self.plan[0].u_start
        self._ema = 0.5
        self._last_q = 0.5
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._episode_log: list[dict] = []

    # ── residual application ──────────────────────────────────────

    def _build_envelope(self, s, u_start, u_end, length, solution, base_depth,
                        spd_res, dep_res, use_envelope, depth) -> list:
        """
        Turn the envelope residuals into a blended moveL path.

        The bow travels u_start -> u_end over the note's duration either way;
        what the envelope decides is HOW that travel is distributed. Segment i
        gets a speed weight of (1 + spd_res[i] * SPEED_RESIDUAL_FRAC), and the
        spans are allocated in proportion — so a policy that wants a soft
        attack asks for a slow first segment and the bow spends less hair
        there, arriving at the same place at the same time.

        Depth is set per segment outright, which is the part moveL can do and a
        single-value action never could: lean in to make the string speak, then
        lighten through the sustain.

        Total length and total duration are preserved exactly, so the envelope
        cannot smuggle in extra bow or break the rhythm — those stay the
        planner's decisions.
        """
        if not use_envelope:
            return [PMP.Segment(u_start=u_start, u_end=u_end,
                                speed=solution.speed, accel=solution.accel,
                                depth=depth, duration=solution.duration)]

        n = N_ENVELOPE_SEGMENTS
        weights = np.clip(1.0 + spd_res * SPEED_RESIDUAL_FRAC, 0.25, 4.0)
        # Equal time per segment; the speed weights decide how much bow each
        # one covers. Distributing by time (not distance) is what makes the
        # weights read as "speed during this part of the note".
        seg_duration = solution.duration / n
        fractions = weights / weights.sum()

        # Segment speeds must form a CONTINUOUS profile. Solving each segment
        # with solve_stroke() would give it a rest-to-rest trapezoid, so the bow
        # would decelerate to near zero at every interior boundary -- two dead
        # stops inside a 1 s note. That is audible: the string stops speaking
        # and re-articulates, and a listener hears the note break into three
        # pieces. It also went unpunished by the reward (r_onset/r_smooth pay
        # for a soft first segment; nothing priced the stalls), so the policy
        # learned to do it. Only the first and last boundaries are at rest.
        #
        # Each segment therefore gets its own CRUISE speed, scaled by the same
        # peak/mean ratio the whole-stroke solution uses so the two real end
        # ramps are still paid for and the note keeps its duration.
        mean_speed = length / max(solution.duration, 1e-6)
        peak_over_mean = (solution.speed / mean_speed) if mean_speed > 1e-6 else 1.0

        travel = u_end - u_start
        segments, u = [], u_start
        for i in range(n):
            u_next = u + travel * float(fractions[i]) if i < n - 1 else u_end
            span_m = abs(u_next - u) * PMP.BOW_LENGTH
            # Cap at the calibrated grid: beyond SPEED_MAX the loudness model
            # is extrapolating and the tone is uncharacterised. A capped
            # segment simply runs a little long rather than being cut short.
            cruise = (span_m / seg_duration) * peak_over_mean
            cruise = float(np.clip(cruise, 1e-4, PMP.SPEED_MAX))
            seg_depth = float(np.clip(
                base_depth + float(dep_res[i]) * DEPTH_RESIDUAL_M,
                DEPTH_LO, DEPTH_HI))
            segments.append(PMP.Segment(
                u_start=u, u_end=u_next, speed=cruise, accel=solution.accel,
                depth=seg_depth, duration=seg_duration))
            u = u_next
        return segments

    def _build_exec_stroke(self, plan_stroke, action) -> tuple[ExecStroke, float]:
        """
        Apply the residual to a planned stroke from the CURRENT bow position.

        Returns (exec_stroke, shortfall_frac) where shortfall_frac is how much
        of the desired length had to be cut to stay inside [U_MIN, U_MAX].
        """
        s = plan_stroke
        direction = s.direction
        sign = 1.0 if direction == "down" else -1.0

        # Retakes re-anchor the bow at the planned landing point.
        retake_from = None
        u_start = self._u
        if s.retake_from is not None:
            retake_from = self._u
            u_start = s.u_start

        # Envelope: a (speed, depth) residual per segment. The MEAN speed
        # residual sets the stroke's overall length — the bow budget is a
        # property of the whole note — while the per-segment values shape how
        # that length is distributed in time.
        spd_res = np.asarray(action[:N_ENVELOPE_SEGMENTS], dtype=float)
        dep_res = np.asarray(action[N_ENVELOPE_SEGMENTS:2 * N_ENVELOPE_SEGMENTS],
                             dtype=float)
        use_envelope = s.duration >= ENVELOPE_MIN_DURATION

        # A note too short to segment is its own attack, so segment 0's
        # residual applies to the whole thing. That keeps dim 0 meaning "the
        # attack" whatever the note length.
        speed_scale = 1.0 + (float(spd_res.mean()) if use_envelope
                             else float(spd_res[0])) * SPEED_RESIDUAL_FRAC

        # Base depth: the planner's, unless dynamics were re-aimed -- then use
        # the depth the ORIGINAL written dynamic implies, so speed carries the
        # dynamics and depth keeps its musical shaping.
        base_depth = s.depth
        if self._depth_volumes and s.note_index < len(self._depth_volumes):
            base_depth = PMP.volume_to_depth(self._depth_volumes[s.note_index])
        depth_res0 = float(dep_res.mean()) if use_envelope else float(dep_res[0])
        depth = float(np.clip(base_depth + depth_res0 * DEPTH_RESIDUAL_M,
                              DEPTH_LO, DEPTH_HI))

        desired = s.length * speed_scale
        if direction == "down":
            available = max(0.0, (PMP.U_MAX - u_start) * PMP.BOW_LENGTH)
        else:
            available = max(0.0, (u_start - PMP.U_MIN) * PMP.BOW_LENGTH)
        length = float(np.clip(desired, 1e-4, available))
        shortfall = (desired - length) / max(desired, 1e-9)

        solution = PMP.solve_stroke(length, s.duration)
        length = solution.length
        u_end = float(np.clip(u_start + sign * length / PMP.BOW_LENGTH,
                              PMP.U_MIN, PMP.U_MAX))

        segments = self._build_envelope(s, u_start, u_end, length, solution,
                                        base_depth, spd_res, dep_res,
                                        use_envelope, depth)

        prev_end = (self.plan[self._i - 1].onset + self.plan[self._i - 1].duration
                    if self._i > 0 else 0.0)
        exec_stroke = ExecStroke(
            note_index=s.note_index,
            direction=direction,
            u_start=u_start,
            u_end=u_end,
            length=length,
            duration=s.duration,
            mean_speed=solution.mean_speed,
            speed=solution.speed,
            accel=solution.accel,
            depth=depth,
            volume_target=s.volume_target,
            retake_from=retake_from,
            onset=s.onset,
            gap_before=max(0.0, s.onset - prev_end),
            segments=segments,
        )
        return exec_stroke, float(shortfall)

    # ── reward ────────────────────────────────────────────────────

    def _reward(self, plan_stroke, exec_stroke, result, shortfall,
                action) -> tuple[float, dict]:
        window_pos = 0.5
        detail = {}
        if hasattr(self.scorer, "score_detailed"):
            try:
                detail = self.scorer.score_detailed(
                    result.audio, result.physical, window_pos=window_pos,
                    string=self.string, segment_profile=result.segment_profile)
            except TypeError:
                # RealScorer hears the shape in the audio and takes no profile.
                detail = self.scorer.score_detailed(
                    result.audio, result.physical, window_pos=window_pos,
                    string=self.string)
        quality = float(detail.get("overall") if detail else
                        self.scorer.score(result.audio, result.physical,
                                          window_pos=window_pos,
                                          string=self.string))

        fill = float(np.clip(exec_stroke.duration / CLASSIFIER_WINDOW_SEC,
                             0.0, 1.0))

        # ── tone, from whichever heads the checkpoint provides ────
        # A short note is nearly all attack and release — there is barely any
        # sustain to judge — so as `fill` falls, weight shifts onto the heads
        # that describe short events. That is the point of using the detailed
        # output rather than `overall`: attack_quality is a HUMAN judgement
        # about something that actually fits inside an 80 ms note.
        tone = quality
        if detail:
            weights = dict(TONE_HEAD_WEIGHTS)
            if fill < 1.0:
                shift = (1.0 - fill)
                weights["attack_quality"] += 0.30 * shift
                weights["release_quality"] += 0.10 * shift
                weights["tone_quality"] = max(0.0, weights["tone_quality"] - 0.30 * shift)
                weights["overall"] = max(0.0, weights["overall"] - 0.10 * shift)
            usable = {h: w for h, w in weights.items() if h in detail and w > 0}
            if usable:
                total_w = sum(usable.values())
                tone = sum(detail[h] * w for h, w in usable.items()) / total_w

        # Only a light discount for short windows — and measurement, not
        # assumption, is why it is light.
        #
        # This started as a heavy shrink toward neutral (fill scaling the whole
        # deviation), on the assumption that a short note padded out to 500 ms
        # with spectrogram floor could not be scored. Tested directly on the
        # standard config, that assumption was WRONG: scoring the same
        # recordings at different window lengths gives
        #
        #     0.10s rho 0.824    0.35s rho 0.828
        #     0.20s rho 0.834    0.50s rho 0.835
        #
        # i.e. the ranking barely degrades down to 100 ms, because the trunk's
        # mean+std pooling is still driven by the real frames and the padding
        # only adds a constant. The heavy discount was suppressing real signal.
        #
        # A small residual discount is kept because rho measures ORDERING, not
        # calibration, and because these were clean slices of sustained strokes
        # — a short note in a real passage also has its neighbours bleeding
        # into the window, which none of this tested.
        quality_eff = 0.5 + (WINDOW_FILL_FLOOR
                             + (1.0 - WINDOW_FILL_FLOOR) * fill) * (tone - 0.5)

        # ── objective defects, valid at any window length ─────────
        r_defect = 0.0
        defect_detail = {}
        defect_wsum = 0.0
        for feat, direction, good, bad, w in DEFECT_FEATURES:
            if feat not in detail:
                continue
            value = detail[feat]
            score01 = (value - bad) / (good - bad) if good != bad else 1.0
            score01 = float(np.clip(score01, 0.0, 1.0))
            defect_detail[feat] = score01
            r_defect += w * (1.0 - score01)
            defect_wsum += w
        if defect_wsum > 0:
            r_defect /= defect_wsum

        # Dynamic accuracy: achieved mean speed vs what the score asked for --
        # but capped at what the note's DURATION physically allows.
        #
        # Covering length L in T seconds needs accel >= 4L/T^2, so the longest
        # stroke that fits is L = accel_max*T^2/4 and the fastest achievable
        # MEAN speed is L/T = accel_max*T/4. A 0.08 s note therefore tops out
        # near 0.08 m/s however loud the score marks it; solve_stroke already
        # shortens the stroke rather than miss the beat.
        #
        # Grading against the written dynamic anyway punished the policy by up
        # to 8 dB for the planner's physical limit -- a penalty no residual
        # could ever remove, so it was pure noise in the objective. Grade
        # against the best the note can actually do.
        written_speed = PMP.volume_to_speed(plan_stroke.volume_target)
        achievable_speed = ACCEL_MAX_FOR_DYNAMICS * exec_stroke.duration / 4.0
        target_speed = min(written_speed, achievable_speed)

        zone = None
        zone_cap_db = 0.0
        if self.loudness is not None and result.measured_dbfs is not None:
            # CLOSED LOOP: grade the level the instrument actually produced
            # against the zone this dynamic belongs in. Bow speed was only ever
            # a proxy for loudness, and an imperfect one -- it ignores how
            # depth, bow position and the state of the rosin change the
            # speed->level transfer. This measures the thing we actually care
            # about.
            zone = self.loudness.zone_for_dynamic(plan_stroke.dynamic)
            lo, hi = self.loudness.zone_bounds(zone)
            # Duration-aware grading (Zixian, 2026-08-13): a short note's mean
            # speed tops out at accel_max*T/4, so its level tops out at what
            # the loudness model predicts for that speed. Grading an absolute
            # zone above that cap charged an unremovable penalty — the same
            # disease the open-loop branch above already fixed. When the
            # written zone is unreachable, LOWER THE FLOOR to the reachable
            # region but KEEP THE WRITTEN CEILING: anything louder than the
            # model's cap is strictly closer to what the composer asked for,
            # so it must never be punished (the cap is a model estimate with
            # ~1.1 dB residual — half of all best-effort strokes read above
            # it). The cap is computed at the PLAN's depth, not the executed
            # one, so the policy cannot drag its own grading band down by
            # playing shallow.
            zone_cap_db = 0.0
            cap_dbfs = self.loudness.predict_dbfs(
                min(achievable_speed, self.loudness.speed_max),
                plan_stroke.depth)
            if cap_dbfs < lo:
                zone_cap_db = lo - cap_dbfs
                lo = cap_dbfs - (hi - lo)      # hi stays the written ceiling
            r_dynamic = self.loudness.zone_reward(result.measured_dbfs, zone,
                                                  bounds=(lo, hi))
            centre = (lo + hi) / 2.0
            err_db = float(result.measured_dbfs - centre)
        else:
            # OPEN LOOP fallback: no microphone or no fitted model, so infer
            # loudness from bow speed as before.
            err_db = abs(20.0 * np.log10(
                max(result.measured_mean_speed, 1e-3) / max(target_speed, 1e-3)))
            r_dynamic = float(np.clip(1.0 - err_db / DYNAMIC_FULL_ERR_DB, 0.0, 1.0))

        # Bow budget: cutting a stroke short, or parking near a hard limit,
        # is the residual policy's fault — the baseline plan always fits.
        u_end = result.achieved_u_end
        edge = min(u_end - PMP.U_MIN, PMP.U_MAX - u_end)
        r_bow = -float(np.clip(3.0 * shortfall, 0.0, 1.0))
        if edge < 0.03:
            r_bow -= 0.5 * (1.0 - edge / 0.03)

        r_smooth = -0.5 * float(np.mean(np.abs(action - self._prev_action)))

        # Envelope jerk: a shape that lurches between segments is a bow that
        # lurches within the note. Penalising the step between consecutive
        # segments pushes the policy toward shapes that ramp rather than jump —
        # which is the whole point of having an envelope instead of one value.
        spd_env = np.asarray(action[:N_ENVELOPE_SEGMENTS], dtype=float)
        dep_env = np.asarray(action[N_ENVELOPE_SEGMENTS:2 * N_ENVELOPE_SEGMENTS],
                             dtype=float)
        r_envelope = 0.0
        if exec_stroke.duration >= ENVELOPE_MIN_DURATION and N_ENVELOPE_SEGMENTS > 1:
            r_envelope = -0.5 * float(
                np.mean(np.abs(np.diff(spd_env))) + np.mean(np.abs(np.diff(dep_env))))

        # Harsh bow changes: acceleration above the pendant-verified nominal is
        # what makes an attack hard, and solve_stroke raises it freely to fit
        # short notes. Price it so the policy can trade attack smoothness
        # against the planner's rhythmic precision explicitly.
        #
        # Short-note fix (Zixian, 2026-08-13): penalise only the EXCESS over
        # the PLAN's own acceleration. A short loud note needs high accel to
        # reach its calibrated speed inside the beat — neither the planner
        # nor any residual can lower that, so grading against MOVE_ACCEL
        # charged short notes a constant unremovable penalty (the same
        # disease the duration-aware dynamics fix removed: an unsatisfiable
        # term is pure noise in the objective, and its gradient leaks into
        # behaviour). The zero-residual plan therefore scores exactly 0, and
        # a policy that asks for more bow than planned pays for exactly the
        # extra acceleration that choice demands. Softer-than-plan attacks
        # earn nothing here — attack_quality already hears them.
        accel = exec_stroke.accel
        plan_length = abs(plan_stroke.u_end - plan_stroke.u_start) * PMP.BOW_LENGTH
        accel_needed = max(
            PMP.MOVE_ACCEL,
            PMP.solve_stroke(plan_length, plan_stroke.duration).accel)
        # Fixed denominator: one m/s² of excess costs the same on every note.
        # Normalising by (max - needed) made the price explode as needed
        # approached the ceiling — a cliff on short notes, not a gradient.
        r_onset = -float(np.clip(
            (accel - accel_needed)
            / max(ACCEL_MAX_FOR_DYNAMICS - PMP.MOVE_ACCEL, 1e-6),
            0.0, 1.0))

        # A segment slower than the calibrated floor does not make the string
        # speak. This is the term that was missing when the rest-to-rest
        # renderer stalled the bow twice per note: the result was audibly
        # broken to a listener, and the reward called it an improvement (see
        # RL_METHOD.md 9.3). The renderer no longer stalls by construction, but
        # the speed weights clip at 0.25..4.0, so a lopsided envelope can still
        # starve a segment. Price the physical cause -- bow too slow to sustain
        # Helmholtz motion -- rather than the specific rendering bug.
        r_speak = 0.0
        if len(exec_stroke.segments) > 1:
            deficits = [
                np.clip((PMP.SPEED_MIN - sg.speed) / PMP.SPEED_MIN, 0.0, 1.0)
                for sg in exec_stroke.segments]
            r_speak = -float(np.mean(deficits))

        total = (W_QUALITY * quality_eff + W_DYNAMIC * r_dynamic
                 + W_BOW * r_bow + W_SMOOTH * r_smooth
                 - W_DEFECT * r_defect + W_ONSET_ACCEL * r_onset
                 + W_ENVELOPE * r_envelope + W_SPEAK * r_speak)
        components = {
            # `quality` stays the RAW classifier score so logs and the
            # episode summary report what was actually heard; quality_eff is
            # what the policy is graded on.
            "quality": quality, "quality_eff": quality_eff,
            "tone": float(tone), "window_fill": fill,
            "r_dynamic": r_dynamic, "r_bow": r_bow,
            "r_smooth": r_smooth, "err_db": float(err_db),
            "zone_cap_db": float(zone_cap_db),
            "r_defect": float(r_defect), "r_onset": float(r_onset),
            "r_envelope": float(r_envelope), "r_speak": float(r_speak),
            "n_segments": len(exec_stroke.segments),
            "accel": float(accel),
            "shortfall": shortfall, "total": float(total),
            "zone": zone, "dbfs": result.measured_dbfs,
            **{f"d_{k}": v for k, v in defect_detail.items()},
            **{f"h_{k}": detail[k] for k in
               ("tone_quality", "attack_quality", "release_quality")
               if k in detail},
        }
        return float(total), components

    # ── observation ───────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        if self._i < len(self.plan):
            s = self.plan[self._i]
        else:
            s = self.plan[-1]
        sign = 1.0 if s.direction == "down" else -1.0
        u = self._u if s.retake_from is None else s.u_start
        if s.direction == "down":
            avail = (PMP.U_MAX - u) / (PMP.U_MAX - PMP.U_MIN)
        else:
            avail = (u - PMP.U_MIN) / (PMP.U_MAX - PMP.U_MIN)

        if self._i + 1 < len(self.plan):
            nxt = self.plan[self._i + 1]
            nxt_vol, nxt_dur = nxt.volume_target, nxt.duration
            nxt_dir = 1.0 if nxt.direction == "down" else -1.0
        else:
            nxt_vol, nxt_dur, nxt_dir = 0.0, 0.0, 0.0

        vol_end = s.volume_target
        note = self.notes[s.note_index] if s.note_index < len(self.notes) else None
        if note is not None and note.volume_end is not None:
            vol_end = float(np.clip(note.volume_end, 0.0, 1.0))

        return np.array([
            s.duration / 2.0,
            s.mean_speed / PMP.SPEED_MAX,
            s.depth / DEPTH_HI,
            sign,
            u,
            float(np.clip(avail, 0.0, 1.0)),
            s.volume_target,
            vol_end,
            1.0 if s.retake_from is not None else 0.0,
            1.0 if s.part_of is not None else 0.0,
            nxt_vol,
            nxt_dur / 2.0,
            nxt_dir,
            self._last_q,
            self._ema,
            self._i / max(len(self.plan), 1),
            *[float(a) for a in self._prev_action],
        ], dtype=np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = 0
        self._u = self.plan[0].u_start
        self._ema = 0.5
        self._last_q = 0.5
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._episode_log = []

        first, _ = self._build_exec_stroke(self.plan[0],
                                           np.zeros(ACTION_DIM, dtype=np.float32))
        self.executor.begin_episode(first)
        return self._obs(), {"piece": self.piece_path, "n_strokes": len(self.plan)}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).flatten()
        plan_stroke = self.plan[self._i]

        exec_stroke, shortfall = self._build_exec_stroke(plan_stroke, action)
        result = self.executor.execute(exec_stroke)
        reward, components = self._reward(plan_stroke, exec_stroke, result,
                                          shortfall, action)

        self._u = result.achieved_u_end
        self._last_q = components["quality"]
        self._ema = self.ema_alpha * self._ema + (1 - self.ema_alpha) * self._last_q
        self._prev_action = action.copy()

        self._episode_log.append({
            "note_index": exec_stroke.note_index,
            "direction": exec_stroke.direction,
            "u_start": exec_stroke.u_start,
            "u_end": result.achieved_u_end,
            "mean_speed": exec_stroke.mean_speed,
            "depth_mm": exec_stroke.depth * 1000.0,
            "action": action.tolist(),
            **components,
        })

        self._i += 1
        terminated = self._i >= len(self.plan)
        if terminated:
            self.executor.end_episode()

        info = {"stroke": self._episode_log[-1]}
        if terminated:
            info["episode_log"] = self._episode_log
            info["mean_quality"] = float(np.mean(
                [s["quality"] for s in self._episode_log]))
            # Dynamics is half the objective, so report it alongside tone --
            # a run optimising dynamics is otherwise invisible in the log.
            info["mean_dynamic"] = float(np.mean(
                [s["r_dynamic"] for s in self._episode_log]))
            in_zone = [s for s in self._episode_log if s["r_dynamic"] >= 0.999]
            info["in_zone"] = f"{len(in_zone)}/{len(self._episode_log)}"

        return self._obs(), reward, terminated, False, info

    def close(self):
        self.executor.close()
