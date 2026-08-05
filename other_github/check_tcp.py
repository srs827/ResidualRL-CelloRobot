"""
check_tcp.py — diagnose the TCP (tool center point) mismatch.

The waypoints + baseline use TCP_OFFSET (the bow tip, ~141mm from the flange); baseline_controller
and every Baseline-Runners script call setTcp(TCP_OFFSET). The new helper scripts did NOT, so a
pose READ in one TCP frame and COMMANDED in another point ~141mm apart -> "completely different,
way too deep". This proves it: it reads the currently-active TCP and shows how far the REPORTED
pose jumps when we set the correct bow TCP (no robot motion — only the TCP setting changes).

Run (robot ON + REMOTE):  .venv-rtde/bin/python check_tcp.py
"""
import sys
import time

import numpy as np
import rtde_control
import rtde_receive

ROBOT_IP = "192.168.1.100"
TCP_OFFSET = [0.028210348281514253, -0.09610723587300697, -0.09969041498611403, 0.0, 0.0, 0.0]


def main():
    c = rtde_control.RTDEControlInterface(ROBOT_IP)
    r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    try:
        try:
            cur = list(c.getTCPOffset())
            match = np.allclose(cur[:3], TCP_OFFSET[:3], atol=1e-4)
            print(f"  current active TCP offset: {[round(x,5) for x in cur]}")
            print(f"  canonical bow TCP_OFFSET : {[round(x,5) for x in TCP_OFFSET]}")
            print(f"  -> {'MATCH ✓ TCP already correct' if match else 'MISMATCH ✗ — THIS is the bug'}")
        except Exception as e:
            print(f"  (getTCPOffset unavailable: {e})")
        p0 = np.asarray(r.getActualTCPPose(), float)
        c.setTcp(TCP_OFFSET); time.sleep(0.3)
        p1 = np.asarray(r.getActualTCPPose(), float)
        jump = float(np.linalg.norm(p1[:3] - p0[:3]) * 1000)
        print(f"\n  reported pose BEFORE setTcp: xyz {np.round(p0[:3],4).tolist()}")
        print(f"  reported pose AFTER  setTcp: xyz {np.round(p1[:3],4).tolist()}")
        print(f"  -> SAME physical robot pose, reading jumped {jump:.0f}mm purely from the TCP setting.")
        print("     ~141mm jump => the new waypoints were captured under the WRONG TCP -> re-capture.")
        print("     ~0mm jump   => TCP was already correct; the problem is elsewhere.")
    finally:
        c.stopScript(); c.disconnect(); r.disconnect()
        print("  (TCP is now set to the bow tip on the controller.)")


if __name__ == "__main__":
    main()
