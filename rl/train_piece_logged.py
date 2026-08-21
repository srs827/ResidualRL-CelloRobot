"""
rl/train_piece_logged.py

Same as rl/train_piece.py, plus per-stroke logging and run isolation.

What it adds on top of the stock trainer (all OFF-by-default extras are
flag-gated; with no extra flags the training semantics are IDENTICAL to
rl/train_piece.py — only more data is recorded):

  * per-stroke jsonl record (note index, action, u, speed, depth, quality,
    dynamics terms, dBFS, zone, gain offset) written to a per-run directory,
    line-buffered so no strokes are lost however the process dies;
  * per-run directory isolation: checkpoints, stroke audio, logs all land in
    rl/checkpoints_piece/run_<timestamp>/ — a resumed run can never overwrite
    an earlier run's wavs or pick up a stale replay buffer by accident;
  * --episode-offset N: start episode numbering at N+1 (use when resuming so
    episode ids stay unique across a run's segments);
  * full-episode audio: the continuous mic stream of each episode is saved as
    ep####_full.wav with a matching ep####_times.json (per-stroke t0/t1/window
    boundaries and both dBFS readings), enabling any offline re-slicing;
  * --note-dbfs: measure the dynamics-reward dBFS over the note's own span
    [t_start, t_end] instead of the fixed 0.5 s classifier window. The window
    RMS under-reads a note of duration d by ~10*log10(d/0.5) — about -8 dB at
    80 ms — which pushes short notes below every loudness zone regardless of
    the action. Off by default to preserve stock semantics.

Usage (stock-faithful replication):
    .venv/bin/python rl/train_piece_logged.py MIDI-Files/t1.mid --real \
        --timesteps 4000 --save-stroke-audio --no-calibrated-dynamics

Zixian Liu, 2026-08-10.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── flags of OUR wrapper, peeled off before the stock argparse sees them ────

RAW_ARGV = list(sys.argv)   # pre-peel, for provenance in the log header


def _peel(flag: str, has_value: bool = False):
    for i, a in enumerate(sys.argv):
        if a == flag:
            sys.argv.pop(i)
            if has_value:
                if i >= len(sys.argv):
                    sys.exit(f"{flag} needs a value")
                return sys.argv.pop(i)
            return True
        if has_value and a.startswith(flag + "="):
            sys.argv.pop(i)
            val = a.split("=", 1)[1]
            if val == "":
                sys.exit(f"{flag}= given with an empty value — either pass a "
                         "value or drop the flag (silently defaulting an "
                         "explicit flag hides typos)")
            return val
    return False if not has_value else None


NOTE_DBFS = bool(_peel("--note-dbfs"))
# (3) alternate loop: play a zero-residual reference episode every N episodes
# so the baseline is plotted on the same axes as the learning curve.
REFERENCE_EVERY = int(_peel("--reference-every", has_value=True)
                      or os.environ.get("CELLO_REFERENCE_EVERY", 0)
                      or (10 if os.environ.get("CELLO_ALT_LOOP", "") not in ("", "0") else 0))
EPISODE_OFFSET = int(_peel("--episode-offset", has_value=True) or 0)

# ── --resume-run RUN_DIR: compute resume arguments FROM the run ─────────────
# Hand-typed resume commands silently rewound three runs on 2026-08-16/17:
# a stale --resume path resets the weights while --episode-offset merely
# relabels the log, so the run LOOKS continuous while training restarts from
# an early checkpoint (~45 min of robot time lost, three times). This flag
# derives both values from the run directory itself and prints them loudly:
#   * checkpoint = the run's *_latest.zip (falls back to newest ckpt_ep*.zip)
#   * episode offset = highest episode number actually recorded in the run's
#     stroke_log.jsonl (already includes any offset the run itself was
#     started with, so chained resumes stay consistent)
# Passing --resume or --episode-offset alongside --resume-run is refused.
RESUME_RUN = _peel("--resume-run", has_value=True)
if RESUME_RUN:
    import glob as _glob
    if EPISODE_OFFSET:
        sys.exit("--resume-run computes the episode offset itself; drop "
                 "--episode-offset")
    if any(a == "--resume" or a.startswith("--resume=") for a in sys.argv):
        sys.exit("--resume-run computes the checkpoint itself; drop --resume")
    _run = Path(RESUME_RUN)
    if not _run.is_dir():
        sys.exit(f"--resume-run: {_run} is not a directory")
    _latest = sorted(_run.glob("sac_piece_*_latest.zip"))
    _ckpt = _latest[-1] if _latest else None
    if _ckpt is None:
        _numbered = sorted(_run.glob("ckpt_ep*.zip"))
        _ckpt = _numbered[-1] if _numbered else None
    if _ckpt is None:
        sys.exit(f"--resume-run: no sac_piece_*_latest.zip or ckpt_ep*.zip "
                 f"in {_run}")
    _log = _run / "stroke_log.jsonl"
    if not _log.exists():
        sys.exit(f"--resume-run: {_log} missing — cannot derive the episode "
                 "offset; use manual --resume/--episode-offset if you are "
                 "certain of the numbers")
    _max_ep = 0
    for _line in open(_log):
        try:
            _rec = json.loads(_line)
        except json.JSONDecodeError:
            continue
        if not _rec.get("header") and "episode" in _rec:
            _max_ep = max(_max_ep, int(_rec["episode"]))
    if _max_ep == 0:
        sys.exit(f"--resume-run: no episode records in {_log}")
    EPISODE_OFFSET = _max_ep
    sys.argv += ["--resume", str(_ckpt.with_suffix(""))]
    print(f"--resume-run {_run.name}:")
    print(f"  checkpoint      -> {_ckpt.name}"
          + ("  (+ replay buffer)" if _ckpt.with_name(
              _ckpt.stem + "_buffer.pkl").exists() else "  (model only)"))
    print(f"  episode offset  -> {_max_ep} (highest episode in stroke_log)")
    print(f"  VERIFY the first resumed episode prints as episode "
          f"{_max_ep + 1} and its scores look continuous with the run — if "
          f"they reset to early-training levels, the checkpoint was stale.")
DRIVER = bool(_peel("--driver"))
GAIN_JITTER_DB = float(_peel("--gain-jitter-db", has_value=True) or 1.5)
DEPTH_JITTER_MM = float(_peel("--depth-jitter-mm", has_value=True) or 0.0)
DBFS_CALIB_DB = float(_peel("--dbfs-calib-db", has_value=True) or 0.0)
BLIND_DYN_OBS = bool(_peel("--blind-dyn-obs"))
BLIND_MECH_OBS = bool(_peel("--blind-mech-obs"))
MOCK_SEED = _peel("--mock-seed", has_value=True)
MOCK_SEED = int(MOCK_SEED) if MOCK_SEED is not None else None
# Keep a numbered checkpoint every N episodes (0 = off). Training-time
# averages are exploration-subsidised and can mask a bad deterministic mean
# (challengepiece: train 0.468, deterministic 0.015) — numbered checkpoints +
# rl/select_best.py replace "keep whatever the last step happened to be".
CKPT_EVERY = int(_peel("--ckpt-every", has_value=True) or 10)
# Reward-weight overrides for this run, as JSON, e.g.
#   --reward-weights '{"W_DEFECT":0.25,"DEFECT_WEIGHTS.hnr_db_mean":2.0,
#                      "TONE_HEAD_WEIGHTS.tone_quality":0.55}'
# Applied by setattr on rl.piece_env before training; recorded in the log
# header. This is the sweep mechanism for reward tuning — no code edits.
REWARD_WEIGHTS_JSON = _peel("--reward-weights", has_value=True)
REWARD_OVERRIDES: dict = {}

# Driver-only flags passed without --driver would be silently inert — and
# worse, --dbfs-calib-db would still be RECORDED in the header, so
# calibrate_gain --from-log would "undo" a correction that never ran and
# bake a wrong offset into loudness_model.json. Refuse loudly instead.
if not DRIVER:
    _inert = [(f, v) for f, v in [
        ("--dbfs-calib-db", DBFS_CALIB_DB), ("--depth-jitter-mm", DEPTH_JITTER_MM),
        ("--blind-dyn-obs", BLIND_DYN_OBS), ("--blind-mech-obs", BLIND_MECH_OBS),
        ("--mock-seed", MOCK_SEED)] if v]
    if GAIN_JITTER_DB != 1.5:
        _inert.append(("--gain-jitter-db", GAIN_JITTER_DB))
    if _inert:
        sys.exit("these flags do nothing without --driver: "
                 + ", ".join(f for f, _ in _inert))

# Honour --save-dir as the PARENT of the run directory. The run keeps its
# timestamped subdirectory either way (a resumed or repeated run must never
# overwrite an earlier one), but where it lands is the caller's choice —
# without this the flag was silently ignored here and every run, mock or
# real, piled into rl/checkpoints_piece/. The argv value is rewritten below
# so the stock executor's checkpoints and stroke audio land inside the same
# run directory as the jsonl log.
_SAVE_BASE = None
_SAVE_IDX = None
for _i, _a in enumerate(sys.argv):
    if _a == "--save-dir" and _i + 1 < len(sys.argv):
        _SAVE_BASE, _SAVE_IDX = Path(sys.argv[_i + 1]), _i + 1
        break
    if _a.startswith("--save-dir="):
        _SAVE_BASE, _SAVE_IDX = Path(_a.split("=", 1)[1]), _i
        break

_RUN_BASE = _SAVE_BASE if _SAVE_BASE is not None else (
    REPO_ROOT / "rl" / "checkpoints_piece")
RUN_DIR = _RUN_BASE / f"run_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RUN_DIR / "stroke_log.jsonl"
EP_AUDIO_DIR = RUN_DIR / "episode_audio"


def _suite_fingerprint():
    """Identity of the judging suite, recorded in every log header.

    The judge checkpoint lineage changed on 2026-08-12 (multilength) and
    loudness_model.json now carries a gain_offset_db — no two runs are
    comparable without knowing which judge and which offset scored them.
    """
    import hashlib
    out = {}
    ckpt_dir = REPO_ROOT / "SoundClassifier" / "checkpoints"
    for p in sorted(ckpt_dir.glob("*.pt")) + [ckpt_dir / "loudness_model.json"]:
        try:
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        except OSError:
            pass
    try:
        out["loudness_gain_offset_db"] = float(json.loads(
            (ckpt_dir / "loudness_model.json").read_text()
        ).get("gain_offset_db", 0.0))
    except Exception:
        out["loudness_gain_offset_db"] = 0.0
    # Which judge will actually score this run (the preference list changes
    # silently when a new checkpoint file appears), and which code graded it
    # (the reward function changed daily this week; argv can't capture it).
    try:
        import rl.piece_env as _pe
        out["judge_resolved"] = str(getattr(_pe, "HUMAN_CHECKPOINT", "n/a"))
    except Exception:
        out["judge_resolved"] = "import-failed"
    try:
        import subprocess
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO_ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               cwd=REPO_ROOT, capture_output=True, text=True)
        out["code"] = (head.stdout.strip()
                       + ("+dirty" if dirty.stdout.strip() else ""))
    except Exception:
        out["code"] = "unknown"
    return out


SUITE = _suite_fingerprint()

# Route the stock --save-dir (checkpoints + stroke_audio) into the run directory
# unless the caller chose one explicitly. Fixes the stock executor's
# resumed-run wav overwrite (its per-process episode counter restarts at 1
# inside a fixed directory).
if _SAVE_BASE is None:
    sys.argv += ["--save-dir", str(RUN_DIR)]
elif sys.argv[_SAVE_IDX].startswith("--save-dir="):
    sys.argv[_SAVE_IDX] = f"--save-dir={RUN_DIR}"
else:
    sys.argv[_SAVE_IDX] = str(RUN_DIR)

import rl.train_piece as tp


class LoggedEpisodeQualityLogger(tp.EpisodeQualityLogger):
    """Stock logger + one jsonl line per stroke, flushed per line."""

    def __init__(self):
        super().__init__()
        self._f = open(LOG_PATH, "a", buffering=1)
        self._f.write(json.dumps({
            "header": True, "argv": sys.argv, "raw_argv": RAW_ARGV,
            "t": time.time(),
            "note_dbfs": NOTE_DBFS, "episode_offset": EPISODE_OFFSET,
            "driver": DRIVER, "gain_jitter_db": GAIN_JITTER_DB if DRIVER else 0.0,
            "depth_jitter_mm": DEPTH_JITTER_MM if DRIVER else 0.0,
            "dbfs_calib_db": DBFS_CALIB_DB,
            "blind_dyn_obs": BLIND_DYN_OBS, "blind_mech_obs": BLIND_MECH_OBS,
            "mock_seed": MOCK_SEED,
            "suite": SUITE,
            "reward_overrides": REWARD_OVERRIDES,
        }) + "\n")
        self._episode = 1 + EPISODE_OFFSET
        self.episode_offset = EPISODE_OFFSET   # console print matches the log
        self._stroke_seq = 0
        self._stroke_in_ep = 0

    def __call__(self, locals_, globals_):
        for info in locals_.get("infos", []):
            s = info.get("stroke")
            if s is not None:
                self._stroke_seq += 1
                self._stroke_in_ep += 1
                rec = {"t": time.time(), "episode": self._episode,
                       "stroke": self._stroke_in_ep,
                       "stroke_seq": self._stroke_seq}
                rec.update(s)
                # err_db is logged in the raw mic frame while r_dynamic is
                # graded in the model frame (loudness_model gain_offset_db);
                # record the model-frame error too so analysis never mixes
                # frames. Driver runs already provide it.
                if ("err_db_model" not in s and s.get("zone") is not None
                        and SUITE.get("loudness_gain_offset_db")):
                    rec["err_db_model"] = (float(s["err_db"])
                                           - SUITE["loudness_gain_offset_db"])
                self._f.write(json.dumps(rec) + "\n")
            if "mean_quality" in info:      # episode boundary
                ep_done = self._episode
                self._episode += 1
                self._stroke_in_ep = 0
                if CKPT_EVERY and ep_done % CKPT_EVERY == 0:
                    model = locals_.get("self")
                    if model is not None:
                        p = RUN_DIR / f"ckpt_ep{ep_done:04d}"
                        model.save(str(p))     # params only, no replay buffer
                        print(f"[retention] episode {ep_done} -> {p.name}.zip")
        return super().__call__(locals_, globals_)


def _apply_reward_weights():
    """Override piece_env reward constants for this run (see flag above)."""
    if not REWARD_WEIGHTS_JSON:
        return {}
    import rl.piece_env as pe
    changes = json.loads(REWARD_WEIGHTS_JSON)
    for key, val in changes.items():
        if key.startswith(("TONE_HEAD_WEIGHTS.", "DEFECT_WEIGHTS.")) and \
                not hasattr(pe, key.split(".", 1)[0].replace(
                    "DEFECT_WEIGHTS", "DEFECT_FEATURES")):
            sys.exit(f"--reward-weights: this piece_env build (v0.1.0?) has "
                     f"no {key.split('.', 1)[0]} — dotted keys need the "
                     "2026-08-12+ reward")
        if key.startswith("TONE_HEAD_WEIGHTS."):
            head = key.split(".", 1)[1]
            if head not in pe.TONE_HEAD_WEIGHTS:
                sys.exit(f"--reward-weights: unknown tone head {head}")
            pe.TONE_HEAD_WEIGHTS[head] = float(val)
        elif key.startswith("DEFECT_WEIGHTS."):
            feat = key.split(".", 1)[1]
            for i, row in enumerate(pe.DEFECT_FEATURES):
                if row[0] == feat:
                    pe.DEFECT_FEATURES[i] = (*row[:4], float(val))
                    break
            else:
                sys.exit(f"--reward-weights: unknown defect feature {feat}")
        elif hasattr(pe, key) and isinstance(getattr(pe, key), (int, float)):
            setattr(pe, key, float(val))
        else:
            sys.exit(f"--reward-weights: piece_env has no scalar '{key}'")
        print(f"reward override: {key} = {val}")
    return changes


def _report_mock_frame():
    """MockExecutor emits MIC-frame dBFS natively since the frame fix.

    This used to install a HotMicMockExecutor subclass, because the mock
    generated MODEL-frame loudness while zone_reward() subtracts
    loudness_model.json's gain_offset_db from every reading — so an
    unpatched mock graded a constant gain_offset_db too quiet. That is now
    fixed at the source (MockExecutor.mic_offset_db), which makes the stock
    entry points correct in mock too; patching here as well would subtract
    the offset twice. Just record what the mock is simulating.
    """
    off = SUITE.get("loudness_gain_offset_db", 0.0)
    if not off:
        return
    print(f"mock executor simulates the hot mic (+{off:.2f} dB, from "
          f"loudness_model gain_offset_db) so zone grading matches the "
          f"real chain")


def _patch_hardware_executor():
    """Subclass HardwareExecutor: full-episode audio + note-span dBFS.

    Imported lazily — only a --real run needs (or can use) it.
    """
    import soundfile as sf
    import rl.piece_hardware as ph
    from rl.loudness import measure_dbfs

    if NOTE_DBFS and getattr(ph, "NOTE_MATCHED_WINDOW", False):
        print("--note-dbfs: this build note-matches the analysis window "
              "itself (piece_hardware.NOTE_MATCHED_WINDOW) — the flag is "
              "kept only for v0.1.0 worktrees; measured dBFS is unchanged "
              "here.")

    class InstrumentedExecutor(ph.HardwareExecutor):

        def begin_episode(self, first):
            super().begin_episode(first)
            self._stroke_times = []

        def _raw_slice(self, t0: float, t1: float):
            if self._audio_t0 is None or not self._chunks:
                return None
            audio = np.concatenate(self._chunks)
            sr = self.sample_rate
            lo = max(int((t0 - self._audio_t0) * sr), 0)
            hi = min(int((t1 - self._audio_t0) * sr), len(audio))
            return audio[lo:hi] if hi > lo else None

        def _slice_window(self, t_start: float, t_end: float):
            window, win_dbfs = super()._slice_window(t_start, t_end)
            dur = t_end - t_start
            skip = ph.PRE_ROLL_SEC if dur > 3 * ph.PRE_ROLL_SEC else 0.0
            note = self._raw_slice(t_start + skip, t_end)
            note_dbfs = measure_dbfs(note) if note is not None else None
            # Window bounds, version-agnostic: note-matched builds expose
            # _window_bounds; v0.1.0 has _window_centre + a fixed 0.5 s window.
            if hasattr(self, "_window_bounds"):
                w0, w1 = self._window_bounds(t_start, t_end)
            else:
                c = self._window_centre(t_start, t_end)
                w0, w1 = c - ph.WINDOW_SEC / 2.0, c + ph.WINDOW_SEC / 2.0
            a0 = self._audio_t0
            self._stroke_times.append({
                "stroke": self._stroke_n,
                "t0": t_start - a0 if a0 else None,
                "t1": t_end - a0 if a0 else None,
                "centre": (w0 + w1) / 2.0 - a0 if a0 else None,
                "win_t0": w0 - a0 if a0 else None,
                "win_t1": w1 - a0 if a0 else None,
                "win_dbfs": win_dbfs, "note_dbfs": note_dbfs,
            })
            if (NOTE_DBFS and note_dbfs is not None
                    and not getattr(ph, "NOTE_MATCHED_WINDOW", False)):
                return window, note_dbfs
            return window, win_dbfs

        def _dump_episode_audio(self, partial: bool = False):
            if not self._chunks:
                return
            EP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            tag = "_partial" if partial else ""
            ep = self._episode          # unoffset: matches stroke_audio wavs;
            sf.write(EP_AUDIO_DIR / f"ep{ep:04d}_full{tag}.wav",
                     np.concatenate(self._chunks), self.sample_rate)
            (EP_AUDIO_DIR / f"ep{ep:04d}_times{tag}.json").write_text(
                json.dumps({"episode_offset": EPISODE_OFFSET,
                            "global_episode": ep + EPISODE_OFFSET,
                            "strokes": self._stroke_times}))
            # Bow motion at 100 Hz, alongside the audio and the window bounds.
            #
            # Without this the analysis window cannot be checked offline. The
            # window is anchored to t_start, stamped just before moveL is
            # dispatched, and the arm then accelerates out of rest before the
            # string speaks -- so the window can sit off the note, and on
            # 2026-08-18 it measured a median 40 ms early (99% of strokes >20
            # ms, 79% >50 ms) worth a mean 1.74 dB under-read.
            #
            # The fix has to key on MOTION, not audio: on real takes, bow
            # motion onset resolves to an IQR of 5.5 ms where audio onset
            # detection gives 90 ms (and the writeup records it inventing
            # 80-107 false onsets per take). The perform path already saves
            # this; training runs did not, which is why the fix was written,
            # validated against the WRONG path (perform pre-starts strokes, so
            # its motion leads the onset while training's follows it), found
            # neutral, and reverted.
            #
            # What it is FOR: rl/reward_noise.py 2026-08-18 showed repeat sd
            # growing 11x through one episode (0.0115 -> 0.1278 on total
            # reward at strokes 36/90/145) while the action signal stayed
            # flat. The first hypothesis was that the analysis window drifts
            # off the note. This log DISPROVED that, which is what it is for.
            #
            # Measured on the first instrumented episode, matching each stroke
            # to the nearest bow DIRECTION REVERSAL (the exact method -- see
            # below): reversal minus commanded onset is a median -0.9 ms with
            # a drift of -0.02 ms/stroke and corr(offset, stroke index)
            # -0.062. The window tracks the real strokes to about a
            # millisecond and does not drift. Two earlier readings that said
            # otherwise (a "median 40 ms early", a "-0.50 ms/stroke drift")
            # were both artifacts of AUDIO-ENERGY onset detection, which
            # writeup-aug17 already records inventing 80-107 false onsets per
            # take. Do not re-derive them.
            #
            # Note the detector that matters. First-threshold-crossing of
            # |bow_speed| does NOT work here: through a continuous passage the
            # bow never stops between notes, so it latches onto the previous
            # stroke and reports a spurious ~-94 ms. bow_speed is SIGNED, so
            # use sign changes -- one reversal per bow stroke, exact.
            #
            # So the cause of the repeat-sd growth is still unknown. Note it
            # cannot be diagnosed from a single episode at all: reward_noise
            # measures spread across REPLAYS of one stroke, while an episode
            # gives within-episode spread across different strokes. Within one
            # episode the loudness residual does not grow with stroke index
            # (corr 0.008), so whatever it is, it is a between-replay effect.
            log = getattr(getattr(self, "logger", None), "log", None)
            if log:
                try:
                    np.save(EP_AUDIO_DIR / f"ep{ep:04d}_state{tag}.npy",
                            np.array(list(log), dtype=object),
                            allow_pickle=True)
                except Exception as e:
                    print(f"(episode state save failed: {e} — continuing)")
            self._chunks = []
            self._stroke_times = []

        def end_episode(self):
            try:
                self._stop_audio()          # stop the callback thread FIRST
                self._dump_episode_audio()
            except Exception as e:
                print(f"(episode-audio save failed: {e} — continuing)")
            super().end_episode()

        def close(self):
            # mid-episode death (fault / Ctrl-C): salvage the in-flight audio
            try:
                self._stop_audio()
                self._dump_episode_audio(partial=True)
            except Exception:
                pass
            super().close()

    ph.HardwareExecutor = InstrumentedExecutor


if __name__ == "__main__":
    print(f"run dir -> {RUN_DIR}")
    print(f"per-stroke log -> {LOG_PATH}"
          + ("   [note-span dBFS ACTIVE]" if NOTE_DBFS else "")
          + (f"   [episode offset {EPISODE_OFFSET}]" if EPISODE_OFFSET else ""))
    REWARD_OVERRIDES = _apply_reward_weights()
    if "--real" in sys.argv:
        _patch_hardware_executor()
    else:
        _report_mock_frame()
    if DRIVER:
        from rl.driver_piece import make_factory
        tp.PieceResidualEnv = make_factory(
            gain_jitter_db=GAIN_JITTER_DB, depth_jitter_mm=DEPTH_JITTER_MM,
            blind_dyn_obs=BLIND_DYN_OBS, blind_mech_obs=BLIND_MECH_OBS,
            dbfs_calib_db=DBFS_CALIB_DB, mock_seed=MOCK_SEED)
        from rl.piece_env import OBS_DIM as _OD
        from rl.driver_piece import DRIVER_EXTRA_DIMS as _DX
        print(f"DRIVER env active: obs {_OD}->{_OD + _DX}, gain jitter "
              f"+/-{GAIN_JITTER_DB} dB, depth jitter +/-{DEPTH_JITTER_MM} mm, "
              f"dBFS calib {DBFS_CALIB_DB:+.1f} dB"
              + ("  [BLIND-DYN]" if BLIND_DYN_OBS else "")
              + ("  [BLIND-MECH]" if BLIND_MECH_OBS else ""))
    tp.EpisodeQualityLogger = LoggedEpisodeQualityLogger
    tp.main()
