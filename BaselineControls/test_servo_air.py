#!/usr/bin/env python3
"""
test_servo_air.py — commission the servoL stroke loop IN THE AIR.

Runs the whole servo path — the 125 Hz loop, the clamping, the fault checks,
the mid-stroke setpoint change — with every pose raised LIFT_M clear of the
string. The arm moves exactly as it would while playing, but nothing touches
the instrument, so the loop can be tuned with no acoustic risk.

WHAT IT ANSWERS
  1. Does the loop hold 125 Hz in Python? (dt_max and jitter, per stroke)
  2. Does the bow travel the distance the speed setpoint asks for?
     Commanded vs achieved mean speed is the number that matters — it is what
     sets loudness once this is over the string.
  3. Does a mid-stroke setpoint change actually take effect, and how fast?
  4. Do the guards hold? A deliberately absurd speed must stop at U_MAX
     rather than running the bow off the end.

Nothing here presses into the string, and every pose is clamped to the taught
bow line regardless of what the setpoints say.

    python BaselineControls/test_servo_air.py
    python BaselineControls/test_servo_air.py --gain 500 --lookahead 0.05
"""

import argparse
import importlib.util
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PMP = _load("play_midi_pieces", REPO_ROOT / "BaselineControls" / "play_midi_pieces.py")
SP = _load("servo_player", REPO_ROOT / "BaselineControls" / "servo_player.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=int, default=SP.SERVO_GAIN)
    ap.add_argument("--lookahead", type=float, default=SP.SERVO_LOOKAHEAD)
    ap.add_argument("--dt", type=float, default=SP.SERVO_DT)
    ap.add_argument("--accel-limit", type=float, default=SP.SERVO_ACCEL_LIMIT,
                    help="servoL acceleration ceiling; short notes need 4L/T^2")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = ap.parse_args()

    print(f"Servo air test — dt {args.dt*1000:.1f} ms ({1/args.dt:.0f} Hz), "
          f"lookahead {args.lookahead:.3f}s, gain {args.gain}, "
          f"accel limit {args.accel_limit:.1f} m/s^2")
    print(f"Every pose lifted {PMP.LIFT_M*1000:.1f} mm clear of the string.\n")
    if not args.yes:
        if input("The arm will move. Continue? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return

    player = PMP.PiecePlayer(record_audio=False)
    servo = SP.ServoStrokePlayer(player.controller, PMP, dt=args.dt,
                                 lookahead=args.lookahead, gain=args.gain,
                                 lift_m=PMP.LIFT_M, accel_limit=args.accel_limit)
    rtde_c = player.rtde_c

    try:
        # ── 1. commanded vs COMMANDED-position vs MEASURED speed ──
        #
        # play_stroke's mean_speed is computed from its own integrated u, so it
        # only proves the setpoint -> servoL command path. Whether the ROBOT
        # tracked that command is a separate question, and the one that
        # matters: loudness follows the bow's real speed, and servoL
        # deliberately lags its target (that is what lookahead/gain do).
        #
        # StateLogger reads actual TCP at 100 Hz and projects onto the taught
        # frog->tip axis — the same instrument the baseline uses for its
        # performance summaries — so it can answer it directly.
        logger = player.logger
        logger.set_bow_reference(PMP.FROG, PMP.TIP)

        # Durations matter as much as speeds. The command jumps to full speed
        # at t=0, which nothing physical can follow, so every stroke loses some
        # distance to the startup transient. On a 1 s stroke that is a small
        # fraction; on the 0.25 s notes that make up 86% of challengepiece it
        # could dominate. Test the short ones, because those are the repertoire.
        print(f"{'cmd':>6} {'dur':>5} {'target':>7} {'MEASURED':>9} {'lag':>7} "
              f"{'len mm':>8} {'ticks':>6} {'dt_max':>8} {'jitter':>8}  stop")
        cases = [(0.09, 1.0), (0.15, 1.0), (0.22, 1.0),
                 (0.15, 0.50), (0.15, 0.25), (0.15, 0.12)]
        for speed, dur in cases:
            u0 = 0.35
            start = servo._pose(u0, 0.0)
            PMP.safe_moveL(player.controller, start, PMP.RESET_SPEED,
                           PMP.MOVE_ACCEL, what=f"go to u={u0} (lifted)")
            time.sleep(0.3)          # let it settle before logging

            sp = SP.Setpoint(speed=speed, depth=0.0)
            logger.start()
            t_start = time.time()
            r = servo.play_stroke(u0, "down", dur, sp, u_limit=PMP.U_MAX)
            t_end = time.time()
            time.sleep(0.05)         # let the 100 Hz logger catch the tail
            logger.stop()

            summary = logger.get_summary(t_start, t_end) or {}
            measured = summary.get("speed_mean", float("nan"))
            # Compare against what the note could ACHIEVE, not what was asked.
            # A short note is capped by accel_limit*T/4, and grading it against
            # an impossible target reports a tracking failure that is really
            # physics — the same distinction solve_stroke draws with CAP.
            target = min(speed, r["speed_ceiling"])
            lag = 100 * (measured - target) / target if measured == measured else float("nan")
            flag = " CAP" if r["speed_capped"] else ""
            print(f"{speed:>6.3f} {dur:>5.2f} {target:>7.3f} {measured:>9.3f} "
                  f"{lag:>+6.1f}% {r['length_m']*1000:>7.1f} {r['ticks']:>6} "
                  f"{r['dt_max']*1000:>7.1f}ms {r['dt_jitter']*1000:>7.2f}ms  "
                  f"{r['stopped_because']}{flag}")
        print("  cmd = asked for | servo = integrated command | MEASURED = actual TCP")

        # ── 2. mid-stroke change ──────────────────────────────────
        print("\nMid-stroke setpoint change (0.10 -> 0.22 m/s at t=0.5s):")
        u0, dur = 0.30, 1.2
        PMP.safe_moveL(player.controller, servo._pose(u0, 0.0),
                       PMP.RESET_SPEED, PMP.MOVE_ACCEL, what="go to u=0.30")
        time.sleep(0.2)

        sp = SP.Setpoint(speed=0.10, depth=0.0)
        marks = []

        def bump():
            time.sleep(0.5)
            sp.write(speed=0.22)
            marks.append(time.time())

        threading.Thread(target=bump, daemon=True).start()
        samples = []
        logger.start()
        t_start = time.time()
        r = servo.play_stroke(u0, "down", dur, sp, u_limit=PMP.U_MAX,
                              on_tick=lambda e, u, v, d: samples.append((e, u, v)))
        time.sleep(0.05)
        logger.stop()
        # Actual TCP either side of the change, from the 100 Hz log.
        log = [s for s in logger.log if t_start <= s["t"] <= t_start + r["duration"]]
        measured_split = {}
        for label, lo, hi in (("before", 0.10, 0.45), ("after", 0.65, 1.10)):
            seg = [(s["t"] - t_start, s["bow_position"]) for s in log
                   if lo <= s["t"] - t_start <= hi]
            if len(seg) > 5:
                a = np.asarray(seg)
                measured_split[label] = abs(
                    np.polyfit(a[:, 0], a[:, 1], 1)[0]) * PMP.BOW_LENGTH
        # Fit the bow's real velocity either side of the change. Reported
        # unconditionally — an earlier version printed nothing when the windows
        # came up short, which looked like the mechanism had failed when it had
        # not.
        arr = np.array([(e, u) for e, u, _ in samples])
        print(f"  ticks {len(arr)}   ran {r['duration']:.2f}s   "
              f"u {r['u_start']:.3f}->{r['u_end']:.3f}   "
              f"stopped_because={r['stopped_because']}")
        if len(arr) > 20:
            cut = 0.5
            first = arr[(arr[:, 0] > 0.1) & (arr[:, 0] < cut - 0.05)]
            last = arr[arr[:, 0] > cut + 0.15]
            for label, seg, asked in (("before", first, 0.10), ("after", last, 0.22)):
                meas = measured_split.get(label)
                meas_txt = (f"MEASURED {meas:.3f} ({100*(meas-asked)/asked:+.0f}%)"
                            if meas is not None else "MEASURED n/a")
                if len(seg) > 5:
                    v = abs(np.polyfit(seg[:, 0], seg[:, 1], 1)[0]) * PMP.BOW_LENGTH
                    print(f"  {label:>6} the change: servo {v:.3f}  {meas_txt}   "
                          f"(asked {asked:.3f})")
                else:
                    print(f"  {label:>6} the change: only {len(seg)} servo samples; "
                          f"{meas_txt}")
        print(f"  setpoint writes seen by the loop: {r['setpoint_updates']}")

        # ── 3. the guard ──────────────────────────────────────────
        print("\nGuard check — absurd 2.0 m/s setpoint must stop at U_MAX:")
        u0 = 0.60
        PMP.safe_moveL(player.controller, servo._pose(u0, 0.0),
                       PMP.RESET_SPEED, PMP.MOVE_ACCEL, what="go to u=0.60")
        time.sleep(0.2)
        sp = SP.Setpoint(speed=2.0, depth=0.0)
        r = servo.play_stroke(u0, "down", 2.0, sp)
        print(f"  u_end {r['u_end']:.4f} (U_MAX {PMP.U_MAX})   "
              f"stopped_because={r['stopped_because']}   "
              f"{'PASS' if r['u_end'] <= PMP.U_MAX + 1e-9 else 'FAIL'}")

    finally:
        try:
            rtde_c.servoStop()
        except Exception:
            pass
        pose = np.asarray(player.rtde_r.getActualTCPPose(), dtype=float)
        PMP.safe_moveL(player.controller, PMP.lifted(pose), PMP.RESET_SPEED,
                       PMP.MOVE_ACCEL, what="final lift clear")
        player.close()
        print("\nDone — bow left clear of the string.")


if __name__ == "__main__":
    main()
