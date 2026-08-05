"""
play_grid.py — ordered listening pass for EAR-LABELING a depth x speed grid (collect_tone output).

Plays one rep per (speed, depth) cell, ordered speed-major then depth ascending, announcing each
cell before it plays — so Zixian can walk the grid once and mark good / too-light / crushed per
cell. Repeat a cell later with:  afplay <printed path>

Usage:
  .venv-rtde/bin/python play_grid.py --data tone_grid            # 1 rep per cell
  .venv-rtde/bin/python play_grid.py --data tone_grid --reps 2   # 2 reps per cell
  .venv-rtde/bin/python play_grid.py --data tone_grid --speed 0.11   # one speed column only

Author: Claude (for Zixian, 2026-06-30).
"""
import argparse
import glob
import os
import re
import subprocess
import time
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="collect_tone output dir (has audio/*.wav)")
    ap.add_argument("--reps", type=int, default=1, help="reps to play per cell")
    ap.add_argument("--speed", type=float, default=None, help="only this speed column")
    ap.add_argument("--gap", type=float, default=0.4, help="pause between strokes, s")
    args = ap.parse_args()

    cells = defaultdict(list)   # (speed, depth) -> [files]
    for f in sorted(glob.glob(os.path.join(args.data, "audio", "*.wav"))):
        m = re.search(r"D(-?[\d.]+)mm_S([\d.]+)", f)
        if m:
            cells[(float(m.group(2)), float(m.group(1)))].append(f)
    if not cells:
        raise SystemExit(f"no D..mm_S.. wavs under {args.data}/audio")

    keys = sorted(cells)
    if args.speed is not None:
        keys = [k for k in keys if abs(k[0] - round(args.speed, 2)) < 5e-3]   # filenames quantize to .2f
        if not keys:
            raise SystemExit(f"no cells at speed {args.speed}; available: {sorted({k[0] for k in cells})}")
    speeds = sorted({k[0] for k in keys})
    print(f"{len(keys)} cells, speeds {speeds}. Mark each cell: good / light(轻) / crush(压死)\n")

    for s, d in keys:
        for f in cells[(s, d)][:args.reps]:
            print(f"▶  speed={s:.2f}  depth={d:g}mm   ({os.path.basename(f)})")
            subprocess.run(["afplay", f])
            time.sleep(args.gap)
    print("\ndone — send Claude the labels per cell (e.g. 's=0.07: good 3-5, light 1-2, crush 8-9').")


if __name__ == "__main__":
    main()
