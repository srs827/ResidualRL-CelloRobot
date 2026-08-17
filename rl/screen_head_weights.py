"""
rl/screen_head_weights.py

Offline screening of TONE_HEAD_WEIGHTS candidates — no retraining, no robot.

Every stroke logged by rl/train_piece_logged.py on the current build carries
the individual judge heads (h_tone_quality, h_attack_quality,
h_release_quality) plus `quality` (the overall head) and `window_fill`.
That is everything needed to re-compute the reward's tone mix under any
candidate head weighting, including the short-note fill shift, and compare
how the candidates would have SCORED the same playing.

Workflow:
  1. run on one or more stroke logs -> per-candidate episode means + rank
     agreement vs the current default;
  2. the tool prints a LISTENING LIST: the strokes where candidates disagree
     the most, with their stroke-audio wav names — ear-label just those few;
  3. re-run with --labels labels.csv (stroke_seq,score in [0,1]) -> per-
     candidate Spearman vs the ear, which is the actual selection criterion.

Head weights change only the JUDGING of a sound, so this screening is valid
without retraining. (Defect weights are different — they change BEHAVIOUR
and must be swept on hardware.)

Usage:
    .venv/bin/python rl/screen_head_weights.py RUN_DIR [RUN_DIR ...]
    .venv/bin/python rl/screen_head_weights.py RUN_DIR --labels ears.csv

Zixian Liu, 2026-08-13.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CANDIDATES = {
    "current":      {"tone_quality": 0.45, "attack_quality": 0.25,
                     "release_quality": 0.15, "overall": 0.15},
    "tq-heavy":     {"tone_quality": 0.60, "attack_quality": 0.20,
                     "release_quality": 0.10, "overall": 0.10},
    "tq-aq":        {"tone_quality": 0.50, "attack_quality": 0.35,
                     "release_quality": 0.10, "overall": 0.05},
    "balanced":     {"tone_quality": 0.25, "attack_quality": 0.25,
                     "release_quality": 0.25, "overall": 0.25},
    "tq-only":      {"tone_quality": 1.0, "attack_quality": 0.0,
                     "release_quality": 0.0, "overall": 0.0},
    "overall-only": {"tone_quality": 0.0, "attack_quality": 0.0,
                     "release_quality": 0.0, "overall": 1.0},  # old regime
}


def mix(row, weights):
    """Reproduce piece_env's tone mix incl. the short-note fill shift."""
    heads = {"tone_quality": row.get("h_tone_quality"),
             "attack_quality": row.get("h_attack_quality"),
             "release_quality": row.get("h_release_quality"),
             "overall": row.get("quality")}
    fill = float(row.get("window_fill", 1.0))
    w = dict(weights)
    if fill < 1.0:
        shift = 1.0 - fill
        w["attack_quality"] += 0.30 * shift
        w["release_quality"] += 0.10 * shift
        w["tone_quality"] = max(0.0, w["tone_quality"] - 0.30 * shift)
        w["overall"] = max(0.0, w["overall"] - 0.10 * shift)
    usable = {h: wt for h, wt in w.items()
              if heads.get(h) is not None and wt > 0}
    if not usable:
        return None
    tot = sum(usable.values())
    return sum(float(heads[h]) * wt for h, wt in usable.items()) / tot


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.mean(ra * rb))


def main():
    ap = argparse.ArgumentParser(description="screen head-weight candidates")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--labels", default=None,
                    help="csv: stroke_seq,score — ear labels")
    ap.add_argument("--listen", type=int, default=10,
                    help="size of the listening list")
    args = ap.parse_args()

    rows = []
    for d in args.run_dirs:
        for line in open(Path(d) / "stroke_log.jsonl"):
            r = json.loads(line)
            if r.get("header") or r.get("h_tone_quality") is None:
                continue
            r["_run"] = Path(d).name
            rows.append(r)
    if len(rows) < 20:
        sys.exit(f"only {len(rows)} strokes with per-head scores — need logs "
                 "from the current build (h_* fields)")
    print(f"{len(rows)} strokes from {len(args.run_dirs)} run(s)\n")

    scores = {}
    for name, w in CANDIDATES.items():
        s = np.array([mix(r, w) for r in rows], dtype=float)
        scores[name] = s
        print(f"{name:14s} mean {np.nanmean(s):.3f}   "
              f"rho vs current {spearman(s, scores['current']):+.3f}"
              if name != "current" else
              f"{name:14s} mean {np.nanmean(s):.3f}   (reference)")

    if args.labels:
        ears = {}
        for line in open(args.labels):
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0].isdigit():
                ears[int(parts[0])] = float(parts[1])
        idx = [i for i, r in enumerate(rows)
               if int(r.get("stroke_seq", -1)) in ears]
        if len(idx) < 5:
            sys.exit(f"only {len(idx)} labelled strokes matched — check "
                     "stroke_seq values")
        y = np.array([ears[int(rows[i]["stroke_seq"])] for i in idx])
        print(f"\nvs {len(idx)} ear labels (selection criterion):")
        ranked = sorted(CANDIDATES,
                        key=lambda n: -spearman(scores[n][idx], y))
        for name in ranked:
            print(f"  {name:14s} Spearman vs ear {spearman(scores[name][idx], y):+.3f}")
        print(f"\n=> best by ear: {ranked[0]}")
        return

    # No labels yet: print the strokes the candidates fight over the most —
    # ear-label THESE, not everything.
    live = [n for n in CANDIDATES if n != "overall-only"]
    mat = np.stack([scores[n] for n in live])
    disagreement = np.nanmax(mat, axis=0) - np.nanmin(mat, axis=0)
    order = np.argsort(-disagreement)[:args.listen]
    print(f"\nListening list (top {len(order)} disagreement strokes) — "
          "ear-label these, save as csv 'stroke_seq,score':")
    for i in order:
        r = rows[i]
        wav = f"ep{int(r['episode']):04d}_s{int(r['stroke']):03d}.wav"
        print(f"  seq {int(r.get('stroke_seq', -1)):5d}  {r['_run']}/"
              f"stroke_audio? {wav}   spread {disagreement[i]:.3f}  "
              f"heads tq {r.get('h_tone_quality'):.2f} "
              f"aq {r.get('h_attack_quality'):.2f} "
              f"ov {r.get('quality'):.2f}")


if __name__ == "__main__":
    main()
