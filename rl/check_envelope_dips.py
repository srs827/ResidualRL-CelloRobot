"""
rl/check_envelope_dips.py

Offline check for mid-note volume dips (writeup-aug12.md to-do #7).

Why: the 3-segment renderer once solved each segment as an independent
rest-to-rest trapezoid, so the bow decelerated to near zero twice inside
every enveloped note — audible as the note breaking into three pieces, and
invisible to the reward (nothing measured it). The renderer is fixed, but
nothing PROVES a run is dip-free; this does, from the full-episode audio
that rl/train_piece_logged.py records (episode_audio/ep*_full.wav +
ep*_times.json), so it works on any past or future --real run.

Metric, per stroke longer than --min-duration: short-window RMS envelope
over the note's interior (attack and release trimmed), dip = median dB of
the interior minus its minimum dB. A clean sustained note stays within a
few dB; a bow stall shows up as a deep notch.

Usage:
    .venv/bin/python rl/check_envelope_dips.py RUN_DIR [--dip-db 6]

Zixian Liu, 2026-08-13.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def envelope_db(x, sr, win_s=0.03):
    n = max(int(win_s * sr), 1)
    k = len(x) // n
    if k < 3:
        return None
    frames = x[:k * n].reshape(k, n)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12
    return 20.0 * np.log10(rms)


def main():
    ap = argparse.ArgumentParser(description="mid-note dip check")
    ap.add_argument("run_dir")
    ap.add_argument("--min-duration", type=float, default=0.4)
    ap.add_argument("--dip-db", type=float, default=6.0,
                    help="flag a stroke when interior dips this far below "
                         "its median level")
    ap.add_argument("--trim", type=float, default=0.08,
                    help="s trimmed from each end (attack/release excluded)")
    args = ap.parse_args()

    audio_dir = Path(args.run_dir) / "episode_audio"
    jsons = sorted(audio_dir.glob("ep*_times*.json"))
    if not jsons:
        sys.exit(f"no episode_audio/ep*_times.json under {args.run_dir} — "
                 "needs a --real run made with rl/train_piece_logged.py")

    dips, flagged, checked = [], [], 0
    for jp in jsons:
        wav = jp.with_name(jp.name.replace("_times", "_full")
                           .replace(".json", ".wav"))
        if not wav.exists():
            continue
        meta = json.loads(jp.read_text())
        x, sr = sf.read(wav)
        if x.ndim > 1:
            x = x.mean(axis=1)
        for s in meta["strokes"]:
            # Prefer the ANALYSIS-window bounds: the stroke's t0..t1 are the
            # moveL call bounds and include ~124 ms of stationary (silent)
            # dispatch/settle time at the end, which reads as a huge fake
            # "dip" in every stroke. The scored window excludes it.
            t0 = s.get("win_t0", s.get("t0"))
            t1 = s.get("win_t1", s.get("t1"))
            if t0 is None or t1 is None or (t1 - t0) < args.min_duration:
                continue
            lo = int((t0 + args.trim) * sr)
            hi = int((t1 - args.trim) * sr)
            if hi - lo < sr * 0.1:
                continue
            env = envelope_db(x[lo:hi], sr)
            if env is None:
                continue
            checked += 1
            dip = float(np.median(env) - np.min(env))
            dips.append(dip)
            if dip > args.dip_db:
                flagged.append((jp.stem, s.get("stroke"), dip))

    if not checked:
        sys.exit("no strokes long enough to check")
    dips = np.array(dips)
    print(f"strokes checked (>= {args.min_duration:.2f}s): {checked}")
    print(f"interior dip dB: median {np.median(dips):.1f}   "
          f"p90 {np.percentile(dips, 90):.1f}   max {dips.max():.1f}")
    print(f"flagged (> {args.dip_db:.1f} dB): {len(flagged)}"
          f"  ({100.0 * len(flagged) / checked:.0f}%)")
    for ep, st, dip in sorted(flagged, key=lambda r: -r[2])[:10]:
        print(f"  {ep} stroke {st}: dip {dip:.1f} dB")


if __name__ == "__main__":
    main()
