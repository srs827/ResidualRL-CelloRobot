"""
verify_contact_line.py — confirm the TAUGHT FROG/TIP line is good before depth-RL.

The waypoints are taught AT light contact, so we do NOT measure offsets (anchors
retired). We only CONFIRM two things, with light touches at a few bow fractions:
  1. the straight line FROG->TIP stays ON the string — contact onset ~at the taught
     pose at every fraction (offset ~0 = on string; negative = line digs in;
     positive = line lifts off / floats above the string);
  2. pressing along the press normal BUILDS force — i.e. the old RETRACT_DIR_BASE is
     still the right "into the string" direction (force grows, stays on-axis).

Guarded: tare in free air, descend in 0.5mm steps from 15mm above, ABORT+retract if
|F|>5N, retract on exit. Light touches only, plus one 1.5mm press at mid-bow.

Reading the result:
  - all three onsets within ~±2mm of the taught pose  -> line is on the string ✓
  - mid-bow press: |F| and F_axis both grow, F_axis ≈ |F| -> normal good ✓
  - large |onset| at some frac -> the straight line doesn't hold contact there (re-teach)
  - press builds |F| but F_axis << |F| (lots of lateral) -> normal wrong (re-derive)

Run (robot ON + REMOTE; HAND ON E-STOP):
  .venv-rtde/bin/python verify_contact_line.py
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
VEL, ACC = 0.03, 0.2
HOVER = 0.015          # start 15mm above the taught contact (free air), along +normal
STEP = 0.0005          # 0.5mm descent steps
MAX_PAST = 0.004       # search at most 4mm PAST the taught pose for contact onset
CONTACT_N = 0.70       # |F| onset threshold (well above the ~0.4N tared free-air floor)
ABORT_N = 5.0
NOMINAL = 0.0015       # mid-bow extra press to test that force builds
E_PRESS = -R.RETRACT_DIR_BASE
FRACS = [0.15, 0.5, 0.9]   # frog tip (frac<0.06) exceeds reach; stay in the safe band


def fvec(r):
    return np.asarray(r.getActualTCPForce(), float)[:3]


def safe(r):
    try:
        return (not r.isProtectiveStopped()) and int(r.getSafetyMode()) == 1
    except Exception:
        return False


def goto(c, r, pose):
    ok = c.moveL(R.clamp_reach(pose).tolist(), VEL, ACC)
    time.sleep(0.05)
    return bool(ok) and safe(r)


def main():
    c = rtde_control.RTDEControlInterface(ROBOT_IP)
    r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    io = rtde_io.RTDEIOInterface(ROBOT_IP)
    results = []
    try:
        io.setSpeedSlider(0.79)   # Zixian 2026-06-22: faster (VEL still caps the contact moves)
        c.setTcp(TCP_OFFSET); time.sleep(0.2)        # bow-tip frame (matches baseline)
        try:
            c.setPayload(float(R.FT_PAYLOAD_KG), list(R.FT_PAYLOAD_COG))
        except Exception:
            pass
        if not safe(r):
            raise RuntimeError("robot not Normal/Running — clear pendant, set REMOTE, rerun.")

        for fi, frac in enumerate(FRACS):
            contact = R.bow_pose_full(frac)
            hover = R.apply_force_offset(contact, -HOVER)
            print(f"\n--- frac {frac:.2f}: hover 15mm above taught contact ---")
            if not goto(c, r, hover):
                print("  move/safety fail — stopping."); break
            c.zeroFtSensor(); time.sleep(0.3)   # RE-TARE at EVERY frac's hover (free air) — a single
            fa = float(np.linalg.norm(fvec(r)))  # shared tare goes stale and false-triggers onset
            print(f"  tared; free-air |F| = {fa:.3f}N" + ("" if fa < CONTACT_N else "  ⚠️ hover NOT free air"))

            onset = None
            d = -HOVER
            while d <= MAX_PAST + 1e-9:
                if not goto(c, r, R.apply_force_offset(contact, d)):
                    print("  move/safety fail."); onset = "ERR"; break
                F = fvec(r); fm = float(np.linalg.norm(F)); fax = float(np.dot(F, E_PRESS))
                if fm > ABORT_N:
                    print(f"  !! |F|={fm:.1f}N > {ABORT_N}N — ABORT, retracting.")
                    goto(c, r, hover); onset = "ABORT"; break
                if fm > CONTACT_N:
                    onset = d
                    print(f"  contact onset at d={d*1000:+.1f}mm vs taught pose | "
                          f"|F|={fm:.2f}N  F_axis={fax:.2f}N  (lateral {fm-abs(fax):.2f}N)")
                    break
                d += STEP

            if onset is None:
                print(f"  NO contact within {MAX_PAST*1000:.0f}mm past the taught pose "
                      f"-> the line FLOATS above the string here.")
                results.append((frac, None, None, None))
            elif isinstance(onset, float):
                # normal-build test at mid-bow: press NOMINAL deeper, see force grow
                fb = None
                if abs(frac - 0.5) < 1e-6:
                    if goto(c, r, R.apply_force_offset(contact, onset + NOMINAL)):
                        F = fvec(r); fm = float(np.linalg.norm(F)); fax = float(np.dot(F, E_PRESS))
                        if fm <= ABORT_N:
                            fb = (fm, fax)
                            print(f"  +{NOMINAL*1000:.1f}mm press: |F|={fm:.2f}N F_axis={fax:.2f}N "
                                  f"(lateral {fm-abs(fax):.2f}N)")
                        else:
                            print(f"  !! press |F|={fm:.1f}N>{ABORT_N} ABORT"); goto(c, r, hover)
                results.append((frac, onset, True, fb))
            goto(c, r, hover)   # retract before next frac
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
        print("\n  session closed; robot stopped.")

    # ---- verdict ----
    print("\n" + "=" * 64 + "\nVERDICT\n" + "=" * 64)
    onsets = [o for (_, o, ok, _) in results if isinstance(o, float)]
    floats = [f for (f, o, ok, _) in results if o is None]
    if onsets:
        worst = max(abs(o) for o in onsets) * 1000
        on_str = ", ".join(f"{f:.2f}:{o*1000:+.1f}mm" for (f, o, ok, _) in results if isinstance(o, float))
        print(f"  contact onsets vs taught line: {on_str}")
        print(f"  -> line {'ON the string ✓ (all within ±2mm)' if worst < 2.0 else f'OFF by up to {worst:.1f}mm — re-teach'}")
    if floats:
        print(f"  ⚠️ floats above string at frac(s): {floats} — straight line does NOT hold contact")
    mid = next((fb for (f, o, ok, fb) in results if abs(f - 0.5) < 1e-6 and fb), None)
    if mid:
        fm, fax = mid
        print(f"  mid-bow press normal: |F|={fm:.2f}N F_axis={fax:.2f}N "
              + ("-> normal GOOD ✓ (force on-axis)" if abs(fax) > 0.7 * fm
                 else "-> LATERAL leak large; normal likely wrong -> re-derive RETRACT_DIR_BASE"))
    print("  if all ✓ -> ready for depth-RL.  Send this output to Claude.")


if __name__ == "__main__":
    main()
