"""
goto_frog.py — move the robot to EXACTLY the taught FROG pose and HOLD there so you can
eyeball the capture. Does nothing else (no deep press, no multi-point sweep).

Optional arg = bow fraction (default 0.0 = FROG; 1.0 = TIP; 0.5 = mid) so you can also park
on the tip / middle to inspect the whole taught line.

Safe approach: hover 15mm above the taught pose (free air) -> tare F/T -> descend in 0.5mm
steps to the taught pose, ABORTING if |F|>5N. Then it HOLDS the pose and waits for Enter; the
tared force it prints tells you whether that pose is true light contact (~0.3-0.6N) or pressed
in (>1.5N). Enter -> retract -> exit.

Run (robot ON + REMOTE; HAND ON E-STOP):
  .venv-rtde/bin/python goto_frog.py            # FROG
  .venv-rtde/bin/python goto_frog.py 1.0        # TIP
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
VEL, ACC = 0.10, 0.4        # Zixian 2026-06-22: faster, stop wasting lab time
HOVER = 0.015          # start 15mm above the taught pose, free air
STEP = 0.0005          # 0.5mm descent steps
ABORT_N = 5.0
E_PRESS = -R.RETRACT_DIR_BASE


def fmag(r):
    return float(np.linalg.norm(np.asarray(r.getActualTCPForce(), float)[:3]))


def faxis(r):
    return float(np.dot(np.asarray(r.getActualTCPForce(), float)[:3], E_PRESS))


def safe(r):
    try:
        return (not r.isProtectiveStopped()) and int(r.getSafetyMode()) == 1
    except Exception:
        return False


def main():
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    label = "FROG" if frac == 0.0 else ("TIP" if frac == 1.0 else f"frac {frac:.2f}")
    target = R.bow_pose_full(frac)
    rch = float(np.linalg.norm(target[:3]))
    print(f"  target = {label}: xyz {np.round(target[:3],4).tolist()}  reach {rch:.3f}m")
    if rch > R.MAX_SAFE_REACH:
        print(f"  ⚠️ REFUSED: reach {rch:.3f} > MAX_SAFE_REACH {R.MAX_SAFE_REACH} (arm near full extension "
              f"= singularity). The frog end exceeds reach; use a safe frac, e.g.  goto_frog.py 0.10")
        return

    c = rtde_control.RTDEControlInterface(ROBOT_IP)
    r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    io = rtde_io.RTDEIOInterface(ROBOT_IP)
    try:
        io.setSpeedSlider(0.79)
        c.setTcp(TCP_OFFSET); time.sleep(0.2)        # bow-tip frame (matches baseline)
        try:
            c.setPayload(float(R.FT_PAYLOAD_KG), list(R.FT_PAYLOAD_COG))
        except Exception:
            pass
        if not safe(r):
            raise RuntimeError("robot not Normal/Running — clear pendant, set REMOTE, rerun.")

        hover = R.clamp_reach(R.apply_force_offset(target, -HOVER))
        if not c.moveL(hover.tolist(), VEL, ACC) or not safe(r):
            raise RuntimeError("move to hover failed.")
        c.zeroFtSensor(); time.sleep(0.3)
        print(f"  hover 15mm above {label}; tared free-air |F| = {fmag(r):.3f}N. descending...")

        d = -HOVER
        reached = False
        while d <= 1e-9:                       # descend to the taught pose (d=0), not past it
            if not c.moveL(R.clamp_reach(R.apply_force_offset(target, d)).tolist(), VEL, ACC) or not safe(r):
                print("  move/safety fail — stopping."); break
            fm = fmag(r)
            if fm > ABORT_N:
                print(f"  !! |F|={fm:.1f}N > {ABORT_N}N at d={d*1000:+.1f}mm — stopped SHORT of the taught "
                      f"pose. The taught FROG is pressed in by at least {abs(d)*1000:.1f}mm.")
                break
            d += STEP
        else:
            reached = True

        if reached:
            print(f"\n  ===> AT the taught {label} pose. |F| = {fmag(r):.2f}N  F_axis = {faxis(r):.2f}N")
            print(f"       (light contact ≈ 0.3-0.6N; >1.5N = the taught pose is pressed into the string)")
        print("\n  HOLDING. Look at the bow: is it lightly on the A string at the mark, right angle?")
        input("  press Enter to retract and exit...")
    finally:
        try:
            c.moveL(R.clamp_reach(R.apply_force_offset(target, -HOVER)).tolist(), VEL, ACC)  # retract
        except Exception:
            pass
        try:
            c.servoStop()
        except Exception:
            pass
        c.stopScript(); c.disconnect(); r.disconnect()
        try:
            io.disconnect()
        except Exception:
            pass
        print("  retracted; session closed.")


if __name__ == "__main__":
    main()
