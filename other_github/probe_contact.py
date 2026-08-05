"""
probe_contact.py — press IN from the TAUGHT point and record the force buildup.

Starts AT the taught contact pose (d=0, the point Zixian taught) and presses INTO the string in
1mm steps, settling briefly before each read. Goal: tell whether the frog-vs-mid force difference
is a TILTED line or just the bow's varying STIFFNESS:
  - force starts rising from ~d=0 at EVERY frac, only the SLOPE differs -> line parallel, diff =
    bow stiffness (frog stiff, tip soft) -> NO offset needed.
  - force only starts rising well PAST d=0 at some fracs (onset shifts) -> line is tilted.

Settles 0.3s before each read (springy bow), but transit/press are FAST (79% slider) — no time
wasted descending through free air (we start at the taught point).

Guarded: 1mm steps, abort >3.5N, retract on exit.

Run at a few fractions and compare:
  .venv-rtde/bin/python probe_contact.py 0.15
  .venv-rtde/bin/python probe_contact.py 0.50
  .venv-rtde/bin/python probe_contact.py 0.90
"""
import sys
import time

import numpy as np
import rtde_control
import rtde_receive
import rtde_io

sys.path.insert(0, ".")
import recording_grid as R

ROBOT_IP = "192.168.1.100"
TCP_OFFSET = [0.028210348281514253, -0.09610723587300697, -0.09969041498611403, 0.0, 0.0, 0.0]
SPEED_SLIDER = 0.79                 # Zixian 2026-06-22: stop wasting lab time on slow moves
VEL_T, ACC_T = 0.20, 0.6            # transit (free air)
VEL_P, ACC_P = 0.06, 0.3           # press-into-string moves (kept controlled)
TARE_HOVER = 0.012                  # 12mm above the taught point, for a free-air tare
STEP = 0.001                        # 1mm press-in steps
PRESS_CAP = 0.009                   # press up to 9mm past the taught point
FORCE_CAP = 3.5                     # abort
SETTLE = 0.3                        # let the springy bow damp before reading
E_PRESS = -R.RETRACT_DIR_BASE


def favg(r, n=15):
    return np.mean([np.asarray(r.getActualTCPForce(), float)[:3] for _ in range(n) if not time.sleep(0.012)], axis=0)


def safe(r):
    try:
        return (not r.isProtectiveStopped()) and int(r.getSafetyMode()) == 1
    except Exception:
        return False


def goL(c, r, pose, vel, acc):
    ok = c.moveL(R.clamp_reach(np.asarray(pose, float)).tolist(), vel, acc)
    return bool(ok) and safe(r)


def main():
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    target = R.bow_pose_full(frac)
    print(f"  frac {frac:.2f}: taught xyz {np.round(target[:3],4).tolist()}  reach {np.linalg.norm(target[:3]):.3f}m")
    c = rtde_control.RTDEControlInterface(ROBOT_IP)
    r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    io = rtde_io.RTDEIOInterface(ROBOT_IP)
    rows = []
    floor = None
    try:
        io.setSpeedSlider(SPEED_SLIDER)
        c.setTcp(TCP_OFFSET); time.sleep(0.2)
        try:
            c.setPayload(float(R.FT_PAYLOAD_KG), list(R.FT_PAYLOAD_COG))
        except Exception:
            pass
        if not safe(r):
            raise RuntimeError("robot not Normal/Running — clear pendant, set REMOTE, rerun.")
        # quick free-air tare just above the taught point, then drop to the taught point
        if not goL(c, r, R.apply_force_offset(target, -TARE_HOVER), VEL_T, ACC_T):
            raise RuntimeError("move to tare hover failed.")
        c.zeroFtSensor(); time.sleep(0.4)
        floor = float(np.linalg.norm(favg(r)))
        print(f"  tared; free-air floor |F| = {floor:.3f}N\n  {'d(mm)':>6} {'|F|(N)':>7} {'F_axis(N)':>9}  "
              f"(d=0 = YOUR taught point; pressing IN)")
        d = 0.0
        while d <= PRESS_CAP + 1e-9:
            if not goL(c, r, R.apply_force_offset(target, d), VEL_P, ACC_P):
                print("  move/safety fail."); break
            time.sleep(SETTLE)
            F = favg(r); fm = float(np.linalg.norm(F)); fax = float(np.dot(F, E_PRESS))
            print(f"  {d*1000:6.1f} {fm:7.2f} {fax:9.2f}")
            rows.append((d, fm, fax))
            if fm > FORCE_CAP:
                print("  reached force cap — stop."); break
            d += STEP
        goL(c, r, R.apply_force_offset(target, -TARE_HOVER), VEL_T, ACC_T)   # retract fast
    finally:
        try:
            c.servoStop()
        except Exception:
            pass
        c.stopScript(); c.disconnect(); r.disconnect()
        try:
            io.disconnect()
        except Exception:
            pass
        print("  session closed.")

    if len(rows) >= 2 and floor is not None:
        f0 = rows[0][1]
        last = rows[-1]
        slope = (last[1] - rows[0][1]) / max(last[0] - rows[0][0], 1e-6) / 1000.0  # N per mm
        thr = floor + 0.3
        onset = next((d for d, fm, fax in rows if fm > thr), None)
        print("\n" + "=" * 56)
        print(f"  |F| at YOUR taught point (d=0): {f0:.2f}N")
        print(f"  rises above floor+0.3 ({thr:.2f}N) at d = "
              + (f"{onset*1000:+.1f}mm" if onset is not None else "not within range"))
        print(f"  stiffness (force per mm pressed in): {slope:.2f} N/mm")
        print("  compare across 0.15/0.50/0.90: onset~0 everywhere + different N/mm = STIFFNESS (no offset);")
        print("  onset shifts with frac = real TILT (per-frac offset needed).")


if __name__ == "__main__":
    main()
