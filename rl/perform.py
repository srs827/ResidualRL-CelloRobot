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


def compile_performance(model, env, baseline):
    """Deterministic rollout through the capturing env -> stroke list."""
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        if baseline or model is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(action)
    return env.executor.strokes


def render_compiled(strokes, report_dir):
    """Play the piece as a FEW blended moveL paths — one dispatch per
    contiguous run of notes, split only at rests and retakes, so the
    constant ~124 ms per-dispatch cost (measured 2026-08-14) is paid once
    per run and absorbed by the rest that precedes it. This is the
    architecture that holds tempo_ratio ~1.01 vs ~1.16 stroke-by-stroke."""
    import soundfile as sf
    import rl.piece_hardware as ph
    import BaselineControls.play_midi_pieces as PMP

    # split into contiguous runs at rests / retakes
    runs, cur = [], []
    for s in strokes:
        boundary = (s.retake_from is not None or s.gap_before > 0.02)
        if boundary and cur:
            runs.append(cur)
            cur = []
        cur.append(s)
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
        target = t0 + onset
        now = time.time()
        if now < target:
            time.sleep(target - now)
        path = []
        for s in run:
            n_seg = len(s.segments)
            for i, seg in enumerate(s.segments):
                pose = PMP.apply_depth(PMP.pose_at(seg.u_end), PMP.CFG,
                                       seg.depth)
                length = abs(seg.u_end - seg.u_start) * PMP.BOW_LENGTH
                # zero blend at each stroke's final segment: the bow
                # reverses there, velocity is zero by physics — rounding
                # it off would smear the bow change
                blend = (0.0 if i == n_seg - 1
                         else min(0.3 * length, 0.025))
                path.append(list(pose) + [seg.speed, seg.accel, blend])
        run_starts.append(time.time() - t0)
        ok = executor.player.controller.rtde_c.moveL(path)
        if ok is False:
            print("WARNING: moveL(path) returned False — check the pendant")
        onset += sum(s.duration for s in run) + sum(
            s.gap_before for s in run[1:])
    wall = time.time() - t0

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
    return wall, audio, executor.sample_rate, run_starts, runs, audio_t0_rel


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
               calibrated_dynamics=False):
    from rl.piece_env import PieceResidualEnv, OBS_DIM
    if obs_dim == OBS_DIM:
        return PieceResidualEnv(piece_path=piece, executor=executor,
                                scorer=scorer,
                                calibrated_dynamics=calibrated_dynamics)
    from rl.driver_piece import DriverPieceEnv, DRIVER_EXTRA_DIMS
    if obs_dim != OBS_DIM + DRIVER_EXTRA_DIMS:
        sys.exit(f"checkpoint obs dim {obs_dim} matches neither stock "
                 f"({OBS_DIM}) nor driver ({OBS_DIM + DRIVER_EXTRA_DIMS})")
    return DriverPieceEnv(piece_path=piece, executor=executor, scorer=scorer,
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
                         NeutralScorer(), args.calibrated_dynamics)
        strokes = compile_performance(model, env, args.baseline)
        written_total = max(s.onset + s.duration for s in env.plan)
        env.close()
        out_dir = REPO_ROOT / "rl" / "checkpoints_piece" / \
            f"perform_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"compiled {len(strokes)} strokes -> {out_dir}")
        if args.mock:
            print("[mock] compile plumbing OK; real render skipped")
            return
        wall, audio, sr, run_starts, runs, a0rel = \
            render_compiled(strokes, out_dir)
        rep = {"mode": "compiled", "strokes": len(strokes),
               "wall_s": round(wall, 2),
               "written_s": round(written_total, 2),
               "tempo_ratio": round(wall / max(written_total, 1e-6), 3)}
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
        executor = MockExecutor(rng=np.random.default_rng(args.seed + 1))
        scorer = AsyncScorer(MockScorer(rng=np.random.default_rng(args.seed + 2)))
        # hot-mic parity with the real chain (see train_piece_logged):
        # without it every mock stroke grades gain_offset_db too quiet and
        # a driver checkpoint sees err obs biased by the same amount
        try:
            from rl.loudness import LoudnessModel
            _off = float(getattr(LoudnessModel(), "gain_offset_db", 0.0))
        except Exception:
            _off = 0.0
        if _off:
            _orig_exec = executor.execute

            def _hot(stroke, _o=_orig_exec, _v=_off):
                r = _o(stroke)
                if r.measured_dbfs is not None:
                    r.measured_dbfs = float(r.measured_dbfs) + _v
                return r
            executor.execute = _hot

    env = _build_env(obs_dim, args.piece, args.mock, args.seed,
                     executor, scorer, args.calibrated_dynamics)
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
