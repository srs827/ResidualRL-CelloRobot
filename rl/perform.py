"""
rl/perform.py

Play a piece at TRUE tempo with a trained policy. Playback + evaluation
only — nothing here changes training.

Implements writeup-aug12.md §5's spec:

  1. absolute-onset scheduling — before each stroke, wait until
     performance_t0 + written_onset, not "sleep the gap". Delays stop
     compounding, and short notes stop absorbing a disproportionate share
     (the training loop's ~50 ms constant per-note dead time stretches a
     0.111 s note 1.45x but a 0.5 s note only 1.10x — audible as wrong
     RHYTHM, not slow tempo).
  2. async scoring — the classifier runs in a worker thread; the main loop
     proceeds immediately with the most recent completed score. True
     per-stroke scores still land in the log, written by the worker.
  3. timing report — wall time vs written duration, worst per-note slip,
     stretch by duration bucket.
  4. when late: play immediately and log the slip. No catch-up by
     shortening notes.
  5. training untouched — this file only wraps the executor and scorer.

Checkpoint lineage is auto-detected from the saved observation space
(18 = stock, 23 = driver, zero curriculum offsets). Note: with async
scoring the policy's quality observation dims lag one extra stroke; the
loudness/torque driver dims are measured synchronously and do not lag.

Usage:
    .venv/bin/python rl/perform.py CKPT.zip MIDI-Files/t1.mid --real
    .venv/bin/python rl/perform.py CKPT.zip MIDI-Files/t1.mid --mock  # plumbing

Zixian Liu, 2026-08-13.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── async scoring ─────────────────────────────────────────────────

class AsyncScorer:
    """Submit scoring jobs to a worker; answer instantly with the most
    recent COMPLETED result. True scores are collected in .rows."""

    def __init__(self, inner):
        self.inner = inner
        self.rows = []
        self._latest_q = 0.5
        # Non-empty from the start: an empty dict makes piece_env fall
        # through to scorer.score() — a SECOND submission for the same
        # stroke, which shifts stroke_seq for every later row and
        # double-counts stroke 1 in the mean.
        self._latest_detail = {"overall": 0.5}
        self._n_submitted = 0
        self._q = queue.Queue()
        self._t = threading.Thread(target=self._work, daemon=True)
        self._t.start()

    def _work(self):
        while True:
            job = self._q.get()
            if job is None:
                self._q.task_done()
                return
            idx, audio, physical, window_pos, string = job
            try:
                if hasattr(self.inner, "score_detailed"):
                    detail = self.inner.score_detailed(
                        audio, physical, window_pos=window_pos, string=string)
                    q = float(detail.get("overall", 0.5))
                else:
                    detail = {}
                    q = float(self.inner.score(audio, physical,
                                               window_pos=window_pos,
                                               string=string))
                self._latest_detail = detail
                self._latest_q = q
                self.rows.append({"stroke_seq": idx, "quality": q,
                                  **{k: float(v) for k, v in detail.items()
                                     if isinstance(v, (int, float))}})
            except Exception as e:
                self.rows.append({"stroke_seq": idx, "error": repr(e)})
            finally:
                self._q.task_done()

    def _submit(self, audio, physical, window_pos, string):
        self._n_submitted += 1
        self._q.put((self._n_submitted, audio, physical, window_pos, string))

    # what the env calls — return immediately with the latest completed
    def score(self, audio, physical, window_pos=0.5, string="A", **kw):
        self._submit(audio, physical, window_pos, string)
        return self._latest_q

    def score_detailed(self, audio, physical, window_pos=0.5, string="A",
                       **kw):
        self._submit(audio, physical, window_pos, string)
        return dict(self._latest_detail)

    def drain(self, timeout=60.0):
        # bound the whole wait, including the in-flight job — a hung
        # classifier must not block the end-of-episode report forever
        t0 = time.time()
        while self._q.unfinished_tasks and time.time() - t0 < timeout:
            time.sleep(0.05)
        if self._q.unfinished_tasks:
            print(f"(scorer drain timed out after {timeout:.0f}s — "
                  f"{self._q.unfinished_tasks} job(s) unscored)")

    def close(self):
        self._q.put(None)


# ── compiled mode ─────────────────────────────────────────────────

# The UR controller executes a blended moveL path FASTER than solve_stroke's
# model (T = L/v + v/a) predicts. Measured 2026-08-17 from the bow's own
# reversals at 100 Hz, yunpiece.mxl fast passage: 128 reversals for 128
# written notes, core interval 117.8 ms against a written 125.0 ms -- every
# note 5.8% short, consistently (sd 8.1 ms, so the per-note rhythm itself is
# fine).
#
# That small constant error is what made the passage sound broken. It
# accumulates ~7 ms per note, and the re-sync at each dispatch boundary dumps
# the ~100 ms of banked rush as a single silent hole: exactly 7 holes were
# measured against exactly 7 dispatch boundaries, 741 ms of excess between
# them. Rush, hiccup, rush -- seven times.
#
# Scaling commanded speed by this factor makes the motion fill its written
# slot, so the sleep-to-target never fires and the holes cannot form.
#
# The value is FITTED, not derived. Two measured points -- gain 1.000 gave a
# 117.8 ms core interval, gain 0.942 gave 120.9 ms -- do not lie on an
# inverse-linear curve, so the controller's overhead is not purely a speed
# scaling. Fitting spacing = A + B/gain gives A ~ 67 ms of fixed per-note cost
# and B ~ 50 ms that scales. That model predicted 0.875, which OVERSHOT to a
# ratio of 1.024 (the piece then dragged, tempo_ratio 1.044). Three measured
# points -- gain 1.000/0.942/0.875 giving ratio 0.951/0.982/1.024 -- are very
# nearly linear, and interpolating them for ratio 1.000 gives 0.913.
#
# This is an empirical constant for THIS controller, blend radius and note
# length. Re-measure motion_fast_ratio in compiled_report.json after changing
# any of them; do not assume it transfers to other repertoire.
# Re-measure with `motion_fast_ratio` in compiled_report.json after any change
# to the controller, the blend radius, or ACCEL_MAX.
# Superseded by rl/timing_model.json (see rl/calibrate_timing.py). Kept only
# as the fallback when no calibration file is present.
MOVEL_TIMING_GAIN = 0.913


def _timing_model():
    """(c0, c1) for motion_time = c0 + c1 * commanded, from calibration.

    A single scalar cannot express this: measured 2026-08-17, the cost is
    ~51.7 ms FIXED plus a 0.9536 scale, so a 100 ms note runs 60% long while a
    700 ms note runs 3% long. Three scalar gains were fitted before measuring
    (0.942 / 0.875 / 0.913); each was right for one note length.
    """
    f = REPO_ROOT / "rl" / "timing_model.json"
    try:
        d = json.loads(f.read_text())
        return float(d["c0"]), float(d["c1"])
    except Exception:
        return 0.0, MOVEL_TIMING_GAIN

# Seconds between issuing moveL and the bow actually moving. This codebase
# already measured the per-dispatch cost at ~124 ms (2026-08-14) but only ever
# treated it as overhead to AMORTISE, never as latency to SCHEDULE AROUND.
#
# The loop below sleeps until t0 + written_onset and THEN issues the dispatch,
# so the motion starts ~124 ms after the note was due. Every run is late by
# that amount, and the scheduler can never recover it: it sleeps when early
# and does nothing when late. Measured 2026-08-17 on yunpiece.mxl -- the
# performance ran 1-3 s behind the written schedule for the whole piece
# (0.41 s behind at T=5 s, 3.07 s at T=20 s), worst in the slow opening where
# gaps exceed the 20 ms run-split threshold so nearly every note is its own
# dispatch and pays the latency individually.
#
# Issuing the dispatch EARLY by this much puts the MOTION on the beat instead
# of the COMMAND. Verify with the drift table, not with tempo_ratio.
DISPATCH_LEAD_S = 0.124

class CapturingExecutor:
    """Runs the env's full planning/obs machinery WITHOUT a robot: collects
    every ExecStroke (with its rendered segments) and returns silent
    results. Used by --compile to precompute the whole performance."""

    def __init__(self):
        self.strokes = []

    def begin_episode(self, first):
        pass

    def execute(self, stroke):
        from rl.piece_env import StrokeResult
        self.strokes.append(stroke)
        phys = np.array([stroke.depth, 0.0, stroke.mean_speed,
                         (stroke.u_start + stroke.u_end) / 2.0, 0.2, 0.26],
                        dtype=np.float32)
        return StrokeResult(audio=None, physical=phys, measured_dbfs=None,
                            measured_mean_speed=float(stroke.mean_speed),
                            achieved_u_end=float(stroke.u_end),
                            segment_profile=None)

    def end_episode(self):
        pass

    def close(self):
        pass


class NeutralScorer:
    """Constant scores for the compile pass (the classifier cannot hear a
    stroke that has not been played; quality obs sit at neutral, which is
    also what live perform's async scorer feeds until results arrive)."""

    def score(self, audio, physical, window_pos=0.5, string="A", **kw):
        return 0.5

    def score_detailed(self, audio, physical, window_pos=0.5, string="A",
                       **kw):
        return {"overall": 0.5}


def _parse_action(spec):
    if not spec:
        return None
    v = [float(x) for x in spec.split(",")]
    if len(v) != 6:
        sys.exit(f"--fixed-action needs 6 values, got {len(v)}")
    return v


def compile_performance(model, env, baseline, fixed_action=None,
                        half=None):
    """Deterministic rollout through the capturing env -> stroke list.

    `fixed_action` plays one constant action on every stroke. Used to generate
    deliberately different-sounding takes for ear judgement: the reward has to
    be shown to ORDER takes the way a listener does before another training run
    is worth the hardware time, and one pair is not enough evidence.

    `half` ablates one half of the 6-dim envelope action IN PLAY, with no
    retraining: "depth" zeroes the three SPEED residuals so the stroke keeps
    the planner's own bow speed and length while the policy still shapes
    depth; "speed" does the reverse. The action is [spd x3, depth x3] and the
    two halves are independent at the executor, so either can be dropped
    without touching the rest.

    Why it exists: on 2026-08-18 the trained policy was heard as more musical
    ("more character", interesting dynamics) but tonally worse, with the
    damage attributed by ear to within-note swells. Speed and depth shaping
    are the two candidates and they cost very different things -- the speed
    weights are normalised, so they redistribute bow WITHIN the note and are
    what produce a lurch, while depth is a gentle timbral trim (~0.42 dB/mm
    across the whole envelope). This separates them for the price of one take
    instead of a retrain.

    NOTE what this does NOT remove: a note whose segments differ in DEPTH is
    still dispatched as moveL(path) rather than a single pose (see
    render_baseline's collapse check, which tests speed AND depth), so
    depth-only shaping keeps the path-dispatch timing cost. Use
    --flatten-envelope to drop that too.
    """
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        if fixed_action is not None:
            action = np.asarray(fixed_action, dtype=np.float32)
        elif baseline or model is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        if half:
            from rl.piece_env import N_ENVELOPE_SEGMENTS as _NS
            action = np.asarray(action, dtype=np.float32).copy()
            if half == "depth":
                action[:_NS] = 0.0          # planner's speed, learned depth
            else:
                action[_NS:2 * _NS] = 0.0   # learned speed, planner's depth
        obs, r, done, _, info = env.step(action)
    return env.executor.strokes


def measure_rhythm(audio, sr, strokes, audio_t0_rel=0.0) -> dict:
    """Per-note timing, measured from the recording itself.

    Compiled mode reported only tempo_ratio -- a WHOLE-PIECE ratio -- and that
    statistic cannot see a rhythm problem by construction: a passage that
    rushes and then waits at the next re-sync averages out to 1.0. Measured
    2026-08-17 on yunpiece.mxl, tempo_ratio read 1.002 while the fast passage
    ran 11% fast with 50 ms of note-to-note jitter. Live mode has had
    per-note slip metrics all along; compiled mode, the one used to perform,
    had none.

    Reports spacing against the WRITTEN grid, and separately for the fastest
    quartile of notes, because that is where the error concentrates and where
    a mean over the whole piece hides it.
    """
    try:
        import librosa
    except ImportError:
        return {"rhythm_error": "librosa unavailable"}
    y = np.asarray(audio, dtype=np.float32).ravel()
    if len(y) < sr // 10 or len(strokes) < 4:
        return {}

    written, t = [], 0.0
    for s in strokes:
        t += s.gap_before
        written.append(t)
        t += s.duration
    written = np.asarray(written)

    a0 = max(float(audio_t0_rel), 0.0)
    det = np.asarray(librosa.onset.onset_detect(
        y=y, sr=sr, units="time", backtrack=True, hop_length=64,
        delta=0.05)) - a0
    if len(det) < 4:
        return {"onsets_detected": int(len(det))}

    out = {"onsets_detected": int(len(det)), "notes_written": len(written),
           "onset_surplus": int(len(det) - len(written))}

    # Note-to-note SPACING, not absolute onset: spacing is what a listener
    # hears as rhythm, and it does not need the two clocks to share an origin.
    wi = np.diff(written)
    di = np.diff(det)
    for name, mask in (("all", np.ones(len(wi), bool)),
                       ("fast", wi <= np.percentile(wi, 25) + 1e-9)):
        w_med = float(np.median(wi[mask])) if mask.any() else 0.0
        if w_med <= 0 or len(di) < 4:
            continue
        # Compare like with like: only intervals near the written spacing.
        near = di[(di > 0.4 * w_med) & (di < 2.5 * w_med)]
        if len(near) < 3:
            continue
        out[f"{name}_written_spacing_ms"] = round(1000 * w_med, 1)
        out[f"{name}_actual_spacing_ms"] = round(1000 * float(np.median(near)), 1)
        out[f"{name}_spacing_ratio"] = round(float(np.median(near)) / w_med, 3)
        out[f"{name}_jitter_sd_ms"] = round(1000 * float(np.std(near)), 1)
    return out


def measure_rhythm_from_motion(motion, strokes) -> dict:
    """Stroke boundaries from the bow's own reversals, not from audio.

    bow_speed changes sign exactly when the bow turns around, which is exactly
    a stroke boundary. No detector, no threshold, no phantoms. At the
    StateLogger's 100 Hz this resolves boundaries to ~10 ms, which is fine
    against a 125 ms note grid and far cleaner than onset detection, whose
    phantom rate on this material is 45-60%.
    """
    if not motion or len(motion) < 20 or len(strokes) < 4:
        return {}
    t = np.array([m["t"] for m in motion], dtype=float)
    v = np.array([m["bow_speed"] for m in motion], dtype=float)
    if not np.any(np.abs(v) > 1e-4):
        return {"motion_rhythm_error": "no bow motion logged"}

    # Sign changes of a lightly smoothed velocity, ignoring near-zero noise.
    k = 3
    vs = np.convolve(v, np.ones(k) / k, mode="same")
    live = np.abs(vs) > 0.15 * np.percentile(np.abs(vs), 90)
    rev = []
    last = None
    for i in range(len(vs)):
        if not live[i]:
            continue
        sgn = 1 if vs[i] > 0 else -1
        if last is not None and sgn != last:
            rev.append(t[i])
        last = sgn
    if len(rev) < 4:
        return {"motion_reversals": int(len(rev))}

    rev = np.asarray(rev)
    ioi = np.diff(rev)
    written, tw = [], 0.0
    for s in strokes:
        tw += s.gap_before
        written.append(tw)
        tw += s.duration
    wi = np.diff(np.asarray(written))
    fast_w = float(np.median(wi[wi <= np.percentile(wi, 25) + 1e-9]))

    near = ioi[(ioi > 0.4 * fast_w) & (ioi < 2.5 * fast_w)]
    out = {"motion_reversals": int(len(rev)),
           "motion_strokes_expected": len(strokes)}
    if len(near) >= 3 and fast_w > 0:
        out["motion_fast_spacing_ms"] = round(1000 * float(np.median(near)), 1)
        out["motion_fast_ratio"] = round(float(np.median(near)) / fast_w, 3)
        out["motion_fast_jitter_ms"] = round(1000 * float(np.std(near)), 1)
    return out


def render_baseline(strokes, report_dir, output_dir=None,
                    flatten_envelope: bool = False, piece_path=None):
    """Play the compiled strokes through the BASELINE player's loop.

    This exists because render_compiled's batching is what broke the rhythm.
    Measured 2026-08-17 on yunpiece.mxl, same rig, same piece:

        BaselineControls/play_midi_pieces.py   3 ms per note, final drift +3 ms
        perform.py --compile                   ~7 ms/note, dumped as 7 holes
        perform.py live                        52 ms fixed per note

    The baseline sends ONE single-pose moveL per single-segment stroke and
    only uses a blended path WITHIN a multi-segment note -- never across
    notes. Its own comment says why: "Short notes stay on moveL, where the
    controller plans the whole trapezoid itself and measurably tracks
    better." render_compiled batched across strokes to remove the dead time
    between dispatches (which is what makes live mode scratchy), and paid for
    it in timing.

    The baseline gets both: 114 ms of motion inside a 125 ms slot leaves only
    ~11 ms of gap, against live mode's ~64 ms. It is not a compromise between
    the two modes -- it beats both, and it has been sitting in the repo the
    whole time.

    So: keep compile_performance (it is what applies the policy's residuals
    deterministically), and hand the resulting strokes to the proven loop
    rather than re-implementing playback.
    """
    import rl.piece_hardware as ph
    import BaselineControls.play_midi_pieces as PMP

    plan = [ph._as_plan_stroke(s) for s in strokes]

    # Collapse FLAT multi-segment strokes to one segment.
    #
    # The env splits any note >= ENVELOPE_MIN_DURATION into 3 segments so the
    # policy can shape it. With a flat envelope -- always for --baseline, and
    # whenever the residuals happen to agree -- all three carry identical speed
    # and depth, so they describe exactly the same motion as one segment. But
    # _play routes >1 segment down its blended-PATH branch instead of the
    # single moveL, and that branch is what tracks worse: measured 2026-08-17,
    # long notes overran ~20% (planned 0.500 s -> actual 0.619) and the drift
    # accumulated to 1.67 s, where the pure baseline held 0.107 s on the same
    # piece. 25/25 of this piece's multi-segment notes were flat.
    #
    # Collapsing is exact -- same poses, same speed, same depth -- and only
    # changes which dispatch the controller gets.
    collapsed = 0
    for st_ in plan:
        segs = st_.segments
        flat = (len({round(g.speed, 6) for g in segs}) == 1
                and len({round(g.depth, 9) for g in segs}) == 1)
        if len(segs) > 1 and flatten_envelope and not flat:
            # Force the collapse using the mean, discarding the shaping.
            # Measured 2026-08-17: a 3-segment note overruns its planned
            # duration by a median 120 ms where a 1-segment note overruns by
            # 5.6 ms, because >1 segment routes to moveL(path) instead of the
            # single-pose moveL. The envelope is the policy's main lever, so
            # the loop charges ~120 ms of timing error for every note the
            # policy shapes -- during training as well as playback.
            mean_speed = sum(g.speed for g in segs) / len(segs)
            mean_depth = sum(g.depth for g in segs) / len(segs)
            segs = [dataclasses.replace(segs[0], speed=mean_speed,
                                        depth=mean_depth)]
            st_.segments = segs
            flat = True
        if len(segs) > 1 and flat:
            first, last = segs[0], segs[-1]
            st_.segments = [dataclasses.replace(
                first, u_end=last.u_end,
                duration=sum(g.duration for g in segs))]
            collapsed += 1
    if collapsed:
        print(f"collapsed {collapsed} flat multi-segment stroke(s) to a single "
              f"moveL (the branch the baseline tracks 3 ms/note on)")
    total = max(s.onset + s.duration for s in plan) + 2.0

    # Probe the input the way the baseline's main() does. PiecePlayer
    # defaults to audio_channel=1, and on this rig channel 1 is an empty
    # socket at -92 dBFS while channel 2 carries the mic -- taking the default
    # writes a silent wav that looks like a successful recording.
    audio_device, audio_channel = None, 1
    try:
        audio_device = PMP._rec.find_focusrite_device()
        audio_channel, levels = PMP.select_audio_channel(audio_device, None)
        PMP.report_audio_channels(levels, audio_channel)
    except Exception as e:
        print(f"  could not check the audio input: {e}")

    player = PMP.PiecePlayer(record_audio=True,
                             output_dir=Path(output_dir or report_dir),
                             audio_device=audio_device,
                             audio_channel=audio_channel)
    player.prepare(plan[0])
    elapsed = player.play(plan, total, lead_in=0.0)

    # play() only BUFFERS the audio; save() is what writes the wav, the
    # 100 Hz state log and the executed timeline. Missing this produced five
    # takes with reports and no recordings.
    try:
        player.save({"source": str(piece_path or "compiled")}, elapsed)
    except Exception as e:
        print(f"  (player.save failed: {type(e).__name__}: {e})")

    timeline = list(getattr(player, "timeline", []))
    if timeline:
        (report_dir / "timeline.json").write_text(json.dumps(timeline, indent=2))
    drifts = [e["drift"] for e in timeline] if timeline else []
    rep = {"mode": "baseline-render", "strokes": len(plan),
           "wall_s": round(elapsed, 2),
           "written_s": round(max(s.onset + s.duration for s in plan), 2)}
    if drifts:
        rep["drift_min_s"] = round(min(drifts), 3)
        rep["drift_max_s"] = round(max(drifts), 3)
        rep["drift_final_s"] = round(drifts[-1], 3)
        rep["drift_mean_ms"] = round(1000 * float(np.mean(drifts)), 1)
        rep["notes_over_20ms"] = int(sum(abs(d) > 0.020 for d in drifts))
    return rep


def render_compiled(strokes, report_dir, max_run_s: float = 2.0):
    """Play the piece as a FEW blended moveL paths — one dispatch per
    contiguous run of notes, split only at rests and retakes, so the
    constant ~124 ms per-dispatch cost (measured 2026-08-14) is paid once
    per run and absorbed by the rest that precedes it. This is the
    architecture that holds tempo_ratio ~1.01 vs ~1.16 stroke-by-stroke."""
    import soundfile as sf
    import rl.piece_hardware as ph
    import BaselineControls.play_midi_pieces as PMP
    # Report notes below the fastest motion the mechanism can produce
    # (2*sqrt(L/a), the triangular profile). Batched dispatch does not pay the
    # isolated stroke's fixed cost, so only the motion floor applies here.
    floor_hits = sum(1 for st_ in strokes
                     if st_.duration + 1e-9 < 2.0 * (st_.length / max(
                         st_.accel, 1e-6)) ** 0.5)
    if floor_hits:
        print(f"NOTE: {floor_hits}/{len(strokes)} notes are shorter than the "
              f"fastest motion their length allows; their rhythm cannot be "
              f"met without shorter strokes, more accel, or slower tempo.")

    # Split into contiguous runs at rests / retakes -- and then again whenever
    # a run would exceed max_run_s of WRITTEN time.
    #
    # Timing is enforced only at run boundaries: the loop below sleeps until
    # each run's scheduled onset and then hands the whole run to moveL as one
    # blended path. Inside a run nothing re-anchors the clock, so note
    # durations are whatever the commanded speeds and the corner-cutting
    # blends produce.
    #
    # That is harmless when runs are short. It is not harmless on a score with
    # real bowings: measured on yunpiece.mxl 2026-08-17, splitting only at
    # rests/retakes gave 28 runs whose median was 0.28 s but whose LARGEST was
    # 14.20 s covering 128 notes -- fourteen seconds of fast passage with no
    # re-sync, which drifts audibly even though the piece as a whole landed at
    # tempo_ratio 0.962.
    #
    # Re-splitting costs one ~124 ms dispatch each time, but the dispatch is
    # absorbed by the sleep-to-target whenever the run is running EARLY, which
    # is the failure direction here (0.962 => ~1.6 s of slack over the piece).
    # At 2.0 s the added overhead is ~1.5 s against that 1.6 s, so the clock
    # gets re-anchored eight times inside that long run for roughly free.
    runs, cur = [], []
    cur_written = 0.0
    for s in strokes:
        boundary = (s.retake_from is not None or s.gap_before > 0.02)
        if not boundary and cur and max_run_s > 0 and \
                cur_written + s.duration > max_run_s:
            boundary = True         # re-sync split: no retake, just a new dispatch
        if boundary and cur:
            runs.append(cur)
            cur = []
            cur_written = 0.0
        cur.append(s)
        cur_written += s.duration
    if cur:
        runs.append(cur)

    executor = ph.HardwareExecutor()
    executor.begin_episode(strokes[0])       # tare + set bow at first stroke
    time.sleep(0.3)                           # let the mic stream settle

    t0 = time.time()
    onset = 0.0
    run_starts = []                           # wall time each run began
    for run in runs:
        first = run[0]
        onset += first.gap_before
        if first.retake_from is not None:
            executor.player._retake(first.retake_from, first.u_start,
                                    first.depth)
        # Aim the MOTION at the written onset, not the command: moveL takes
        # DISPATCH_LEAD_S to start moving.
        target = t0 + onset - DISPATCH_LEAD_S
        now = time.time()
        if now < target:
            time.sleep(target - now)
        path = []
        for k, s in enumerate(run):
            n_seg = len(s.segments)
            # Fold the FOLLOWING gap into this stroke's motion.
            #
            # A run is dispatched as one continuous moveL path, so any
            # gap_before between strokes inside it simply does not happen --
            # the strokes are concatenated. The run boundary test only fires
            # above 20 ms, and this score's staccato gaps are 14.1 ms, so
            # every one of them was being dropped: measured 2026-08-17 on
            # yunpiece.mxl, written note spacing is 110.9 ms of motion + 14.1
            # ms of silence = 125.0 ms, and the robot played 111.7 ms. Each
            # note arrived 11% early, and after ~16 notes the 2 s re-sync
            # dumped the accumulated 226 ms as one audible hole -- heard as
            # the fast passage rushing and then lurching.
            #
            # Splitting the run at every gap instead would cost 128 dispatches
            # at ~124 ms, which is longer than the passage itself. So slow the
            # stroke to occupy its whole written slot. Onsets then land on the
            # written grid without any extra dispatch.
            #
            # What this trades: the notes become legato rather than detached,
            # since the silence becomes slower motion. They were already being
            # played legato -- the gap was discarded either way -- so this
            # buys correct rhythm at no articulation cost over the status quo.
            # Real staccato needs the bow to stop or lift, which a single
            # blended path cannot express.
            gap_after = (run[k + 1].gap_before if k + 1 < len(run) else 0.0)
            # Ask for the duration that RESULTS in the written slot, inverting
            # the measured motion_time = c0 + c1 * commanded. Clamped at 15 ms:
            # below that the request is nonsense and the note is already at the
            # mechanism's floor, which the caller is warned about separately.
            # NOTE: rl/timing_model.json is calibrated on ISOLATED strokes and
            # does NOT apply here. Inside one batched moveL the per-waypoint
            # cost is far smaller than the 52 ms measured for a standalone
            # dispatch -- batched playback runs ~6% FAST where the isolated
            # model predicts running long. Applying it here measured 2026-08-17
            # made things worse (fast ratio 1.000 -> 0.832). The scalar gain is
            # the right correction for THIS regime; the calibration is for the
            # stroke-at-a-time path.
            want = s.duration + gap_after
            stretch = want / max(s.duration, 1e-6)
            for i, seg in enumerate(s.segments):
                pose = PMP.apply_depth(PMP.pose_at(seg.u_end), PMP.CFG,
                                       seg.depth)
                length = abs(seg.u_end - seg.u_start) * PMP.BOW_LENGTH
                # zero blend at each stroke's final segment: the bow
                # reverses there, velocity is zero by physics — rounding
                # it off would smear the bow change
                blend = (0.0 if i == n_seg - 1
                         else min(0.3 * length, 0.025))
                path.append(list(pose) + [seg.speed / stretch
                                          * MOVEL_TIMING_GAIN,
                                          seg.accel, blend])
        run_starts.append(time.time() - t0)
        ok = executor.player.controller.rtde_c.moveL(path)
        if ok is False:
            print("WARNING: moveL(path) returned False — check the pendant")
        onset += sum(s.duration for s in run) + sum(
            s.gap_before for s in run[1:])
    wall = time.time() - t0

    # The bow's own motion, at StateLogger's 100 Hz. begin_episode() already
    # started this logger, so compiled mode has been collecting it all along
    # and discarding it.
    #
    # It is a far better rhythm instrument than the microphone. A stroke
    # boundary is a SIGN CHANGE in bow_speed -- the bow physically reverses --
    # which is exact and unambiguous. Audio onset detection on this material
    # returns 80-107 phantom events per 182 notes (bow noise, re-sync seams,
    # double attacks), and those phantoms land as short intervals that inflate
    # any jitter estimate. Motion has no such failure mode.
    motion = []
    lg = getattr(executor, "logger", None)
    if lg is not None:
        motion = [{"t": r["t"] - t0, "bow_position": r["bow_position"],
                   "bow_speed": r["bow_speed"]} for r in list(lg.log)]

    time.sleep(0.3)                           # audio tail
    executor._stop_audio()
    audio = (np.concatenate(executor._chunks)
             if executor._chunks else np.zeros(1))
    audio_t0_rel = (executor._audio_t0 - t0
                    if executor._audio_t0 else 0.0)
    sf.write(str(report_dir / "compiled_full.wav"), audio,
             executor.sample_rate)
    executor.close()
    print(f"{len(runs)} dispatch run(s) for {len(strokes)} strokes")
    if motion:
        (report_dir / "motion.json").write_text(json.dumps(motion))
    return (wall, audio, executor.sample_rate, run_starts, runs, audio_t0_rel,
            motion)


# ── absolute-onset scheduling ─────────────────────────────────────

def make_perform_executor():
    """Subclass HardwareExecutor lazily (hardware imports on real runs only)."""
    import rl.piece_hardware as ph

    class PerformExecutor(ph.HardwareExecutor):

        def begin_episode(self, first):
            super().begin_episode(first)
            self._sched = 0.0            # written onset accumulator (s)
            self._perform_t0 = time.time()
            self.timing = []             # (stroke, slip_s, written_onset)

        def execute(self, stroke):
            target = self._perform_t0 + self._sched + stroke.gap_before
            # a retake belongs to the rest before the note: do it now, then
            # wait out whatever written silence remains
            if stroke.retake_from is not None:
                self.player._retake(stroke.retake_from, stroke.u_start,
                                    stroke.depth)
            now = time.time()
            slip = 0.0
            if now < target:
                time.sleep(target - now)
            else:
                slip = now - target      # late: play NOW, never shorten
            self.timing.append({"stroke": self._stroke_n + 1,
                                "written_onset": self._sched + stroke.gap_before,
                                "slip_s": slip})
            self._sched += stroke.gap_before + stroke.duration
            s2 = dataclasses.replace(copy.copy(stroke), gap_before=0.0,
                                     retake_from=None)
            # Phase timing: where does the per-note overhead go? (measured
            # 2026-08-14: ~150 ms/note constant, ratio 1.163 on a contiguous
            # piece — gaps cannot absorb it, so it must be found and removed)
            play_box, slice_box = [0.0], [0.0]
            _op = self.player._play

            def _tp(ps, _o=_op, _b=play_box):
                t = time.time()
                r = _o(ps)
                _b[0] = time.time() - t
                return r

            _os = self._slice_window

            def _ts(a, b, _o=_os, _b=slice_box):
                t = time.time()
                r = _o(a, b)
                _b[0] = time.time() - t
                return r

            self.player._play = _tp
            self._slice_window = _ts
            t0 = time.time()
            try:
                result = super().execute(s2)
            finally:
                self.player._play = _op
                self._slice_window = _os
            dt = time.time() - t0
            self.timing[-1].update({
                "written_s": round(stroke.duration, 3),
                "exec_s": round(dt, 3),
                "play_s": round(play_box[0], 3),
                "slice_s": round(slice_box[0], 3),
                "other_s": round(dt - play_box[0] - slice_box[0], 3)})
            return result

    return PerformExecutor


# ── main ──────────────────────────────────────────────────────────

def _build_env(obs_dim, piece, mock, seed, executor, scorer,
               calibrated_dynamics=False, tempo_scale=1.0):
    from rl.piece_env import PieceResidualEnv, OBS_DIM
    if obs_dim == OBS_DIM:
        return PieceResidualEnv(piece_path=piece, executor=executor,
                                scorer=scorer, tempo_scale=tempo_scale,
                                calibrated_dynamics=calibrated_dynamics)
    from rl.driver_piece import DriverPieceEnv, DRIVER_EXTRA_DIMS
    if obs_dim != OBS_DIM + DRIVER_EXTRA_DIMS:
        sys.exit(f"checkpoint obs dim {obs_dim} matches neither stock "
                 f"({OBS_DIM}) nor driver ({OBS_DIM + DRIVER_EXTRA_DIMS})")
    return DriverPieceEnv(piece_path=piece, executor=executor, scorer=scorer,
                          tempo_scale=tempo_scale,
                          gain_fixed_db=0.0, depth_fixed_mm=0.0,
                          calibrated_dynamics=calibrated_dynamics)


def main():
    ap = argparse.ArgumentParser(description="true-tempo playback")
    ap.add_argument("checkpoint", nargs="?", default=None,
                    help="trained policy; optional with --baseline")
    ap.add_argument("piece")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--mock", action="store_true",
                    help="plumbing test (no timing semantics)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--baseline", action="store_true",
                    help="zero residuals instead of the policy")
    ap.add_argument("--max-run-s", type=float, default=2.0,
                    help="compiled mode: re-sync the clock by starting a new "
                         "moveL dispatch whenever a run would exceed this "
                         "much WRITTEN time. Timing is only enforced at run "
                         "boundaries, so one long run drifts: yunpiece.mxl "
                         "split only at rests/retakes produced a single 14.2 s "
                         "run of 128 notes. 0 disables re-splitting.")
    ap.add_argument("--tempo-scale", type=float, default=1.0,
                    help="stretch every note duration by this factor. The "
                         "planner's reachable mean speed is accel_max*T/4, so "
                         "very short notes cannot reach a speed that makes the "
                         "string SPEAK (PMP.SPEED_MIN=0.09 m/s) and rub "
                         "instead. Measured on yunpiece: at 1.0 70%% of "
                         "strokes command < 0.09 m/s, at 1.5 only 3%%. No "
                         "policy can fix this -- the residual cannot "
                         "manufacture speed the duration forbids.")
    ap.add_argument("--fixed-action", default=None,
                    help="six comma-separated values in [-1,1] "
                         "(spd x3, depth x3) played on EVERY stroke, for "
                         "generating deliberately different takes to judge")
    ap.add_argument("--only", choices=("depth", "speed"), default=None,
                    help="play only half of the policy's envelope action: "
                         "'depth' keeps the learned depth shaping and uses the "
                         "PLANNER's speed; 'speed' the reverse. Ablation with "
                         "no retraining (the halves are independent at the "
                         "executor). Ignored with --baseline.")
    ap.add_argument("--flatten-envelope", action="store_true",
                    help="average each note's envelope to one segment so it "
                         "dispatches as a single moveL. Costs the intra-note "
                         "shaping; saves the ~120 ms per-note overrun that "
                         "moveL(path) charges.")
    ap.add_argument("--render", choices=("baseline", "compiled"),
                    default="baseline",
                    help="how --compile dispatches. 'baseline' uses "
                         "PiecePlayer's per-stroke loop (3 ms/note measured); "
                         "'compiled' uses batched blended paths (better inter-"
                         "note continuity, measurably worse rhythm).")
    ap.add_argument("--compile", action="store_true",
                    help="precompute all actions, then dispatch each "
                         "contiguous run of notes as ONE blended moveL path "
                         "(tempo ~1.01x vs ~1.16x stroke-by-stroke; "
                         "stock/baseline checkpoints only)")
    ap.add_argument("--calibrated-dynamics", action="store_true", default=False,
                    help="MUST match how the checkpoint was trained "
                         "(our protocol: off)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.real == args.mock:
        sys.exit("choose exactly one of --real / --mock")

    from rl.piece_env import OBS_DIM
    model = None
    if args.checkpoint is not None:
        from stable_baselines3 import SAC
        model = SAC.load(args.checkpoint, env=None, device="cpu")
        obs_dim = model.observation_space.shape[0]
    elif args.baseline:
        obs_dim = OBS_DIM          # zero residuals need no policy
    else:
        sys.exit("a checkpoint is required unless --baseline")

    if args.compile:
        if obs_dim != OBS_DIM:
            sys.exit("--compile is stock/baseline only — the driver's "
                     "closed-loop sensing needs the stroke-by-stroke path")
        cap = CapturingExecutor()
        env = _build_env(obs_dim, args.piece, True, args.seed, cap,
                         NeutralScorer(), args.calibrated_dynamics,
                         tempo_scale=args.tempo_scale)
        strokes = compile_performance(model, env, args.baseline,
                                      fixed_action=_parse_action(
                                          args.fixed_action),
                                      half=args.only)
        written_total = max(s.onset + s.duration for s in env.plan)
        env.close()
        out_dir = REPO_ROOT / "rl" / "checkpoints_piece" / \
            f"perform_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"compiled {len(strokes)} strokes -> {out_dir}")
        if args.mock:
            print("[mock] compile plumbing OK; real render skipped")
            return
        if args.render == "baseline":
            rep = render_baseline(strokes, out_dir,
                                  flatten_envelope=args.flatten_envelope,
                                  piece_path=args.piece)
            print(json.dumps(rep, indent=2))
            (out_dir / "compiled_report.json").write_text(
                json.dumps({"report": rep}, indent=2))
            return
        wall, audio, sr, run_starts, runs, a0rel, motion = \
            render_compiled(strokes, out_dir, max_run_s=args.max_run_s)
        rep = {"mode": "compiled", "strokes": len(strokes),
               "wall_s": round(wall, 2),
               "written_s": round(written_total, 2),
               "tempo_ratio": round(wall / max(written_total, 1e-6), 3)}
        # tempo_ratio alone cannot see local rhythm error -- a rush and the
        # hole after it cancel. Measure the actual note grid.
        rep.update(measure_rhythm(audio, sr, strokes, a0rel))
        rep.update(measure_rhythm_from_motion(motion, strokes))
        try:
            from rl.piece_env import RealScorer
            judge = RealScorer()
            qs = []
            for rs, run in zip(run_starts, runs):
                tcur = rs - a0rel
                for s in run:
                    lo, hi = int(tcur * sr), int((tcur + s.duration) * sr)
                    seg = audio[lo:hi]
                    if len(seg) > 0:
                        phys = np.array(
                            [s.depth, 0.0, s.mean_speed,
                             (s.u_start + s.u_end) / 2.0, 0.2, 0.26],
                            dtype=np.float32)
                        qs.append(float(judge.score(seg, phys)))
                    tcur += s.duration
            if qs:
                rep["mean_quality_posthoc"] = round(float(np.mean(qs)), 3)
        except Exception as e:
            rep["posthoc_error"] = repr(e)
        print(json.dumps(rep, indent=2))
        (out_dir / "compiled_report.json").write_text(json.dumps(
            {"report": rep, "run_starts": run_starts,
             "audio_t0_rel": a0rel}, indent=2))
        return

    if args.real:
        from rl.piece_env import RealScorer
        executor = make_perform_executor()()
        scorer = AsyncScorer(RealScorer())
    else:
        from rl.piece_env import MockExecutor, MockScorer
        # MockExecutor emits MIC-frame dBFS natively (mic_offset_db), so the
        # mock grades on the same frame as the real chain and a driver
        # checkpoint sees unbiased err obs. The hot-mic wrapper that used to
        # live here would now apply the offset a second time.
        executor = MockExecutor(rng=np.random.default_rng(args.seed + 1))
        scorer = AsyncScorer(MockScorer(rng=np.random.default_rng(args.seed + 2)))

    env = _build_env(obs_dim, args.piece, args.mock, args.seed,
                     executor, scorer, args.calibrated_dynamics, tempo_scale=args.tempo_scale)
    if not args.baseline and model is not None and \
            env.action_space.shape != model.action_space.shape:
        sys.exit(f"checkpoint action space {model.action_space.shape} != env "
                 f"{env.action_space.shape} — wrong lineage (with --baseline "
                 "this would be allowed; the policy is never queried)")
    written_total = max(s.onset + s.duration for s in env.plan)

    out_dir = REPO_ROOT / "rl" / "checkpoints_piece" / \
        f"perform_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"perform -> {out_dir}")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + 100 * ep)
        t0 = time.time()
        done, n = False, 0
        while not done:
            if args.baseline:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(action)
            n += 1
        wall = time.time() - t0
        scorer.drain()
        true_q = [r["quality"] for r in scorer.rows if "quality" in r]
        rep = {"episode": ep + 1, "strokes": n,
               "wall_s": round(wall, 2),
               "written_s": round(written_total, 2),
               "tempo_ratio": round(wall / max(written_total, 1e-6), 3),
               "mean_quality_true": (round(float(np.mean(true_q)), 3)
                                     if true_q else None)}
        timing = getattr(env.executor, "timing",
                         getattr(getattr(env.executor, "inner", None),
                                 "timing", None))
        if timing:
            slips = [t["slip_s"] for t in timing]
            rep["worst_slip_ms"] = round(1000 * max(slips), 1)
            rep["mean_slip_ms"] = round(1000 * float(np.mean(slips)), 1)
            rep["late_notes"] = int(sum(s > 0.02 for s in slips))
            if timing and "exec_s" in timing[0]:
                rep["mean_motion_overhead_ms"] = round(1000 * float(np.mean(
                    [t["play_s"] - t["written_s"] for t in timing])), 1)
                rep["mean_slice_ms"] = round(1000 * float(np.mean(
                    [t["slice_s"] for t in timing])), 1)
                rep["mean_other_ms"] = round(1000 * float(np.mean(
                    [t["other_s"] for t in timing])), 1)
        print(json.dumps(rep, indent=2))
        (out_dir / f"ep{ep + 1:02d}_report.json").write_text(json.dumps(
            {"report": rep, "timing": timing, "scores": scorer.rows},
            indent=2))
        scorer.rows = []

    env.close()
    scorer.close()


if __name__ == "__main__":
    main()
