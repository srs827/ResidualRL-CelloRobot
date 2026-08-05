"""
goto_old_angle.py — rotate the bow IN PLACE to a known-good bow ANGLE, so you can then jog
translation-only to (re)place it on the string.

Restores the orientation from the most recent reteach_*.txt (the GOOD angle you last taught;
a skewed STROKE doesn't make the ANGLE bad). Falls back to recording_grid's OLD calibrated
angle only if no reteach file exists.

Commands ONE slow moveL that changes ONLY the orientation (position unchanged). Rotating in
place sweeps the bow tip through an arc -> the bow MUST be clear of the cello first.

Procedure:
  1. Freedrive the bow to a spot CLEARLY ABOVE/AWAY from the cello (position rough; just clear).
  2. Switch pendant to REMOTE. Run this -> it rotates to the good angle. It then exits.
  3. Switch to LOCAL, jog TRANSLATION (X/Y/Z, no rotation) to place the bow on the A string.

Run (robot ON + REMOTE; bow lifted clear; HAND ON E-STOP):
  .venv-rtde/bin/python goto_old_angle.py
"""
import glob
import os
import re
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
VEL, ACC = 0.12, 0.4        # Zixian 2026-06-22: faster transit


def target_orientation():
    files = sorted(glob.glob("reteach_*.txt"), key=os.path.getmtime)
    if files:
        m = re.search(r"FROG_TARGET_STR\s*=\s*np\.array\(\[([^\]]+)\]\)", open(files[-1]).read())
        if m:
            nums = [float(x) for x in m.group(1).split(",")]
            if len(nums) == 6:
                return np.array(nums[3:6]), f"latest taught angle ({os.path.basename(files[-1])})"
    return np.asarray(R.FROG_TARGET_STR, float)[3:], "recording_grid OLD calibrated angle (no reteach file)"


def main():
    ori_target, src = target_orientation()
    print(f"  target = {src}: rotvec {np.round(ori_target, 4).tolist()}")
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_io_c = rtde_io.RTDEIOInterface(ROBOT_IP)
    try:
        rtde_io_c.setSpeedSlider(0.79)
        rtde_c.setTcp(TCP_OFFSET); time.sleep(0.2)        # bow-tip frame (matches baseline)
        if rtde_r.isProtectiveStopped() or int(rtde_r.getSafetyMode()) != 1:
            raise RuntimeError("robot not Normal/Running — clear pendant, set REMOTE, rerun.")
        cur = np.asarray(rtde_r.getActualTCPPose(), float)
        rot_delta = float(np.linalg.norm(cur[3:] - ori_target))
        print(f"  current rotvec = {np.round(cur[3:],4).tolist()}")
        print(f"  rotation to apply ≈ {rot_delta:.3f} rad ({np.degrees(rot_delta):.0f}°), position UNCHANGED, "
              f"reach stays {float(np.linalg.norm(cur[:3])):.3f}m.")
        if input("\n  Bow CLEARLY clear of the cello? rotating sweeps the tip. "
                 "[y to rotate / anything else aborts]: ").strip().lower() != "y":
            print("  aborted."); return
        target = cur.copy(); target[3:] = ori_target
        ok = rtde_c.moveL(target.tolist(), VEL, ACC)
        time.sleep(0.2)
        if ok is False or rtde_r.isProtectiveStopped():
            raise RuntimeError("moveL failed / protective stop — check the robot.")
        final = np.asarray(rtde_r.getActualTCPPose(), float)
        print(f"  done. orientation now {np.round(final[3:],4).tolist()} "
              f"(residual {np.linalg.norm(final[3:]-ori_target):.4f} rad)")
        print("\n  NEXT: switch pendant to LOCAL, jog TRANSLATION (X/Y/Z, NOT rotation) to place the")
        print("        bow on the A string at the tape mark. Then: python teach_tip_on_axis.py")
    finally:
        try:
            rtde_c.servoStop()
        except Exception:
            pass
        rtde_c.stopScript(); rtde_c.disconnect(); rtde_r.disconnect()
        try:
            rtde_io_c.disconnect()
        except Exception:
            pass
        print("  session closed; robot stopped (control released for pendant jog).")


if __name__ == "__main__":
    main()
