"""
rl/calibrate_gain.py

Measure the microphone-chain gain offset against the loudness model, and
optionally write it into SoundClassifier/checkpoints/loudness_model.json
(the mechanism piece_env's zone grading reads it from).

Why this exists: the loudness zones are ABSOLUTE dBFS, so any change of mic,
gain knob or interface shifts every reading bodily and silently kills the
dynamics reward (writeup-aug12.md §1: a 4.2 dB drift zeroed 25% of the reward
with no error message). The offset is a property of the audio chain, not of
the playing — re-measure at the start of every session and whenever the
audio path changes.

Two ways to get data:

  --from-log RUN_DIR [RUN_DIR ...]
      Reuse stroke logs written by rl/train_piece_logged.py. Driver runs are
      fine: the per-stroke curriculum gain offset and the executor-boundary
      calibration are both undone from the header/row fields, recovering the
      raw mic reading.

  --measure [--episodes N]
      Play N zero-residual episodes (the planner's nominal — same thing the
      baseline plays) and measure directly. REAL is the default (--real
      just makes it explicit); --mock exists only to test the plumbing.

The estimate is median residual (measured dBFS minus the offset-free model
prediction at the achieved speed and commanded depth), with sd = 1.4826*MAD,
reported per zone as well: a chain change moves every zone by the same
amount; per-zone disagreement means model nonlinearity, not gain drift.

Usage:
    .venv/bin/python rl/calibrate_gain.py --from-log rl/checkpoints_piece/run_X
    .venv/bin/python rl/calibrate_gain.py --measure --piece MIDI-Files/t1.mid
    ... add --write to update loudness_model.json in place.

Zixian Liu, 2026-08-13.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_JSON = REPO_ROOT / "SoundClassifier" / "checkpoints" / "loudness_model.json"

# --mock is a plumbing test: the mock is told to read exactly this much hotter
# than the model, and a correct run recovers it. Deliberately NOT the json's
# current gain_offset_db — a self-test whose expected answer tracks the value
# under test cannot fail.
KNOWN_ANSWER_DB = 4.2


def _offset_free_model():
    from rl.loudness import LoudnessModel
    m = LoudnessModel(MODEL_JSON)
    m.gain_offset_db = 0.0          # we are estimating it; do not pre-correct
    return m


def _residuals_from_log(run_dir: Path, model):
    """(residual, zone) pairs from one stroke log, raw-mic frame recovered."""
    rows = [json.loads(l) for l in open(run_dir / "stroke_log.jsonl")]
    header = rows[0] if rows and rows[0].get("header") else {}
    calib = float(header.get("dbfs_calib_db", 0.0) or 0.0)
    out = []
    for r in rows:
        if r.get("header") or r.get("dbfs") is None:
            continue
        speed = r.get("mean_speed")
        depth_mm = r.get("depth_mm")
        if speed is None or depth_mm is None:
            continue
        # Undo what the driver pipeline did to the raw reading:
        # logged = raw - calib + curriculum_gain_offset
        raw = float(r["dbfs"]) + calib - float(r.get("gain_offset_db", 0.0))
        pred = model.predict_dbfs(float(speed), float(depth_mm) / 1000.0)
        out.append((raw - pred, r.get("zone")))
    return out


def _residuals_from_episode(args, model):
    """Zero-residual episodes through the stock env; collect (resid, zone)."""
    from rl.piece_env import PieceResidualEnv, MockExecutor, MockScorer

    if args.mock:
        rng = np.random.default_rng(args.seed)
        # Known-answer run: the mock reads hotter than the model by exactly
        # KNOWN_ANSWER_DB, which is what this tool should recover. Pin it
        # explicitly rather than inheriting loudness_model.json's current
        # gain_offset_db — the whole point is a fixed right answer that does
        # not move when the json is re-measured (and the model offset is
        # already zeroed above, since we are estimating it).
        executor = MockExecutor(rng=np.random.default_rng(args.seed + 1),
                                mic_offset_db=KNOWN_ANSWER_DB)
        scorer = MockScorer(rng=np.random.default_rng(args.seed + 2))
    else:
        from rl.piece_hardware import HardwareExecutor
        from rl.piece_env import RealScorer
        executor = HardwareExecutor()
        scorer = RealScorer()

    env = PieceResidualEnv(piece_path=args.piece, executor=executor,
                           scorer=scorer, tempo_scale=args.tempo_scale)
    zeros = np.zeros(env.action_space.shape, dtype=np.float32)
    out = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, k = False, 0
        while not done:
            obs, r, done, _, info = env.step(zeros)
            s = info["stroke"]
            k += 1
            if s.get("dbfs") is None or s.get("mean_speed") is None:
                print(f"  stroke {k:2d}: NO AUDIO (dbfs=None) — "
                      f"speed {s.get('mean_speed')}, zone {s.get('zone')}")
                continue
            pred = model.predict_dbfs(float(s["mean_speed"]),
                                      float(s["depth_mm"]) / 1000.0)
            print(f"  stroke {k:2d}: dbfs {float(s['dbfs']):7.2f}  "
                  f"pred {pred:7.2f}  resid {float(s['dbfs']) - pred:+6.2f}  "
                  f"zone {s.get('zone')}")
            out.append((float(s["dbfs"]) - pred, s.get("zone")))
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser(description="measure mic-chain gain offset")
    ap.add_argument("--from-log", nargs="+", default=None, metavar="RUN_DIR")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--real", action="store_true",
                    help="explicit real-robot mode (the default when --mock "
                         "is absent; this flag just makes it visible)")
    ap.add_argument("--mock", action="store_true",
                    help=f"plumbing test only (known answer +{KNOWN_ANSWER_DB})")
    ap.add_argument("--piece", default="MIDI-Files/t1.mid")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--tempo-scale", type=float, default=1.0,
                    help=">1 slows the piece down. MUST match the tempo you "
                         "will TRAIN at: the offset is measured through the "
                         "same analysis window the reward grades through, and "
                         "note duration moves that window. Measured "
                         "2026-08-19, the residual runs +7.73 dB per decade of "
                         "note length, so vocalise at 120 bpm vs its written "
                         "50 (a 2.4x change) is worth ~2.9 dB -- more than a "
                         "2.5 dB zone.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true",
                    help="update loudness_model.json gain_offset_db")
    args = ap.parse_args()
    if bool(args.from_log) == bool(args.measure):
        sys.exit("choose exactly one of --from-log / --measure")
    if args.write and args.mock:
        sys.exit(f"--write with --mock refused: the mock's "
                 f"+{KNOWN_ANSWER_DB} dB is "
                 "synthetic (a known-answer plumbing test) and would "
                 "overwrite a REAL measured offset in loudness_model.json")
    if args.from_log:
        print("note: prefer zero-residual runs — on trained runs, enveloped "
              "loud notes execute slower than commanded (segment cruise caps "
              "at SPEED_MAX), which biases f-zone residuals low")

    model = _offset_free_model()
    if args.from_log:
        pairs, srcs = [], []
        for d in args.from_log:
            got = _residuals_from_log(Path(d), model)
            pairs += got
            srcs.append(f"{Path(d).name} (n={len(got)})")
        source = ", ".join(srcs)
    else:
        pairs = _residuals_from_episode(args, model)
        source = (f"{'mock' if args.mock else 'real'} zero-residual x"
                  f"{args.episodes} on {Path(args.piece).name}")

    if len(pairs) < 5:
        sys.exit(f"only {len(pairs)} usable strokes — not enough to calibrate")

    res = np.array([p[0] for p in pairs], dtype=float)
    offset = float(np.median(res))
    sd = float(1.4826 * np.median(np.abs(res - offset)))
    cur = json.loads(MODEL_JSON.read_text())
    old = float(cur.get("gain_offset_db", 0.0))
    print(f"source: {source}")
    print(f"gain offset = {offset:+.2f} dB   (robust sd {sd:.2f}, "
          f"n {len(res)}; json currently {old:+.2f})")
    zones = sorted({z for _, z in pairs if z})
    zone_medians = {}
    for z in zones:
        zr = np.array([p[0] for p in pairs if p[1] == z])
        print(f"  zone {z:3s}: {np.median(zr):+.2f} dB  (n {len(zr)})")
        if len(zr) >= 8:        # tiny zones spread past 1.5 dB by chance
            zone_medians[z] = float(np.median(zr))
    spread = (max(zone_medians.values()) - min(zone_medians.values())
              if len(zone_medians) > 1 else 0.0)
    if spread > 1.5:
        balanced = float(np.mean(list(zone_medians.values())))
        print(f"  WARNING: per-zone medians spread {spread:.2f} dB — that is "
              "model nonlinearity (or duration-dependent bias), not gain "
              "drift. Global median is dominated by the piece's zone mix; "
              f"balanced mean-of-zone-medians = {balanced:+.2f} dB. "
              "Investigate before writing anything.")
        if args.write:
            sys.exit("--write refused while zones disagree — a single offset "
                     "cannot represent this; fix the model or pick a "
                     "measurement piece the zones agree on.")

    if args.write:
        cur["gain_offset_db"] = round(offset, 2)
        cur["gain_offset_note"] = (
            f"Measured {date.today().isoformat()} via rl/calibrate_gain.py "
            f"({source}): median residual vs offset-free model, "
            f"sd {sd:.2f}, n {len(res)}. Previous value {old:+.2f}. "
            "Zones are absolute dBFS — re-measure whenever the audio path "
            "changes.")
        MODEL_JSON.write_text(json.dumps(cur, indent=2))
        print(f"wrote gain_offset_db={cur['gain_offset_db']:+.2f} "
              f"-> {MODEL_JSON.relative_to(REPO_ROOT)}")
    elif abs(offset - old) > 0.5:
        print(f"NOTE: differs from json by {offset - old:+.2f} dB — "
              "re-run with --write to update before training.")


if __name__ == "__main__":
    main()
