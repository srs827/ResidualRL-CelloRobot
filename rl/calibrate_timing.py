"""
rl/calibrate_timing.py

Measure how long the UR controller ACTUALLY takes to execute a moveL stroke,
against how long solve_stroke's model says it should, and fit the correction.

Why this exists. The planner solves T = L/v + v/a for the motion parameters and
then trusts that the robot takes T. It does not. Measured 2026-08-17 on
yunpiece.mxl, the error is duration-dependent -- short notes run long, long
notes run short -- so a single scalar gain cannot fix it, which is exactly what
three rounds of tuning MOVEL_TIMING_GAIN discovered the hard way (0.942, 0.875,
0.913, each right for one note length and wrong for the others).

That timing error is the whole rhythm problem. Compiled playback batches notes
into one blended path for TONE -- it removes the ~124 ms of dead time where
live mode leaves the bow loaded and motionless between dispatches -- but inside
a path nothing corrects timing, so the per-note error accumulates until the
next re-sync dumps it as an audible hole. Get the commanded speeds right and
the re-sync becomes unnecessary: both properties at once.

Method. Play isolated strokes across the (length, duration) grid the repertoire
uses. Take actual duration from the StateLogger's bow_speed at 100 Hz -- motion
start and end are threshold crossings, which is exact and needs no onset
detection (audio onset detection on this material invents 45-60% phantom
events). Fit

    T_actual = c0 + c1 * T_model

where c0 is fixed per-dispatch cost and c1 scales the modelled motion. Writing
the inverse into the path generator makes commanded durations come true.

Usage:
    caffeinate -dims ~/venvs/cello311/bin/python rl/calibrate_timing.py
    ... --reps 2          repeats per grid point (default 2)
    ... --out PATH        where to write the fit (default rl/timing_model.json)
    ... --dry-run         print the grid and exit, no robot

Zixian's calibrate_gain.py does the same job for the microphone chain; this is
its counterpart for the motion chain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DEFAULT = REPO_ROOT / "rl" / "timing_model.json"

# Grid. Chosen to bracket what the repertoire actually asks for: yunpiece's
# strokes run ~3-40 mm of bow over 0.06-1.0 s, with the fast passage clustered
# near 11 mm / 0.11 s. Points outside what solve_stroke can deliver at the
# current ACCEL_MAX are skipped rather than silently length-capped, since a
# capped stroke measures the cap, not the controller.
LENGTHS_MM = (5.0, 10.0, 20.0, 35.0)
DURATIONS_S = (0.10, 0.15, 0.25, 0.40, 0.70)

MOTION_EPS = 0.004      # m/s; below this the bow is considered stopped


def measure_one(executor, PMP, u_start, direction, length_m, duration):
    """Dispatch one stroke as a single-waypoint moveL path; time it two ways.

    Returns (t_wall, t_motion, u_end) where t_wall includes dispatch latency
    and t_motion is the bow actually moving, from the 100 Hz log.
    """
    sign = 1.0 if direction == "down" else -1.0
    u_end = float(np.clip(u_start + sign * length_m / PMP.BOW_LENGTH,
                          PMP.U_MIN, PMP.U_MAX))
    actual_len = abs(u_end - u_start) * PMP.BOW_LENGTH
    if actual_len < 1e-4:
        return None
    sol = PMP.solve_stroke(actual_len, duration)
    depth = 0.0005
    pose = PMP.apply_depth(PMP.pose_at(u_end), PMP.CFG, depth)

    n0 = len(executor.logger.log)
    t0 = time.time()
    ok = executor.player.controller.rtde_c.moveL(
        [list(pose) + [sol.speed, sol.accel, 0.0]])
    t_wall = time.time() - t0
    if ok is False:
        print("    moveL returned False")
        return None
    time.sleep(0.05)                       # let the log catch up

    rows = list(executor.logger.log)[n0:]
    t_motion = float("nan")
    if rows:
        t = np.array([r["t"] for r in rows])
        v = np.abs(np.array([r["bow_speed"] for r in rows]))
        live = v > MOTION_EPS
        if live.any():
            t_motion = float(t[live][-1] - t[live][0])
    return {"u_start": u_start, "u_end": u_end, "direction": direction,
            "length_m": actual_len, "t_commanded": duration,
            "t_model": sol.duration if hasattr(sol, "duration") else duration,
            "speed": sol.speed, "accel": sol.accel,
            "length_capped": bool(getattr(sol, "length_capped", False)),
            "t_wall": t_wall, "t_motion": t_motion}


def main() -> None:
    ap = argparse.ArgumentParser(description="calibrate moveL execution timing")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import rl.piece_env as pe
    PMP = pe.PMP

    grid = []
    for L in LENGTHS_MM:
        for T in DURATIONS_S:
            sol = PMP.solve_stroke(L / 1000.0, T)
            capped = bool(getattr(sol, "length_capped", False))
            grid.append((L, T, sol.speed, sol.accel, capped))
    print(f"grid ({len(grid)} points, ACCEL_MAX={PMP.ACCEL_MAX}):")
    print(f"{'L mm':>7}{'T s':>7}{'speed':>9}{'accel':>8}{'capped':>8}")
    for L, T, v, a, c in grid:
        print(f"{L:>7.1f}{T:>7.2f}{v:>9.4f}{a:>8.2f}{'YES' if c else '':>8}")
    if args.dry_run:
        return

    import rl.piece_hardware as ph
    executor = ph.HardwareExecutor()

    # begin_episode does the F/T tare and sets the bow down; it wants a stroke,
    # so hand it a harmless short one at mid-bow.
    # Must NOT be swallowed: begin_episode is what tares the F/T sensor and
    # sets the bow down on the string. Skipping it would run the whole grid
    # with the bow wherever it happened to be left.
    from rl.piece_env import ExecStroke
    sol0 = PMP.solve_stroke(0.01, 0.2)
    first = ExecStroke(note_index=0, direction="down",
                       u_start=0.5, u_end=0.5 + 0.01 / PMP.BOW_LENGTH,
                       length=0.01, duration=0.2, mean_speed=0.05,
                       speed=sol0.speed, accel=sol0.accel, depth=0.0005,
                       volume_target=0.5, segments=[])
    executor.begin_episode(first)
    time.sleep(0.3)

    rows = []
    u = 0.5
    direction = "down"
    try:
        for rep in range(args.reps):
            for L, T, _v, _a, capped in grid:
                if capped:
                    continue
                # Keep the bow inside its travel: flip when there is no room.
                need = L / 1000.0 / PMP.BOW_LENGTH
                if direction == "down" and u + need > PMP.U_MAX - 0.02:
                    direction = "up"
                elif direction == "up" and u - need < PMP.U_MIN + 0.02:
                    direction = "down"
                r = measure_one(executor, PMP, u, direction, L / 1000.0, T)
                if r:
                    r["rep"] = rep
                    rows.append(r)
                    u = r["u_end"]
                    print(f"  rep{rep} L={L:5.1f}mm T={T:.2f}s -> "
                          f"motion {r['t_motion']:.3f}s  wall {r['t_wall']:.3f}s"
                          f"  ratio {r['t_motion'] / T:.3f}")
                direction = "up" if direction == "down" else "down"
                time.sleep(0.12)
    finally:
        try:
            executor.close()
        except Exception:
            pass

    good = [r for r in rows if np.isfinite(r["t_motion"]) and r["t_motion"] > 0]
    if len(good) < 4:
        sys.exit(f"only {len(good)} usable measurements — cannot fit")

    Tc = np.array([r["t_commanded"] for r in good])
    Ta = np.array([r["t_motion"] for r in good])
    Tw = np.array([r["t_wall"] for r in good])
    c1, c0 = np.polyfit(Tc, Ta, 1)
    pred = c0 + c1 * Tc
    resid = Ta - pred

    print(f"\n{'=' * 60}")
    print(f"  n = {len(good)}")
    print(f"  motion time  = {c0:+.4f} + {c1:.4f} * commanded")
    print(f"    fixed overhead  {1000 * c0:+7.1f} ms")
    print(f"    scale            {c1:7.4f}")
    print(f"    residual sd      {1000 * resid.std():7.1f} ms")
    print(f"  dispatch (wall - motion): med {1000 * np.median(Tw - Ta):.1f} ms")
    print(f"\n  per commanded duration:")
    print(f"{'T cmd':>8}{'n':>4}{'motion':>9}{'ratio':>8}")
    for T in sorted(set(Tc)):
        m = Tc == T
        print(f"{T:>8.2f}{m.sum():>4}{Ta[m].mean():>9.3f}"
              f"{Ta[m].mean() / T:>8.3f}")

    out = {
        "measured": date.today().isoformat(),
        "accel_max": float(PMP.ACCEL_MAX),
        "n": len(good),
        "model": "t_motion = c0 + c1 * t_commanded",
        "c0": float(c0), "c1": float(c1),
        "residual_sd_s": float(resid.std()),
        "dispatch_overhead_s": float(np.median(Tw - Ta)),
        "rows": good,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    print("To make a commanded duration come true, ask for "
          "(desired - c0) / c1 instead.")


if __name__ == "__main__":
    main()
