"""
lab_diag_readonly.py  —  READ-ONLY UR5e diagnostic for action-space feasibility.

Author: Claude (drafted for Zixian, 2026-05-28). Read-only by design.

WHAT IT DOES
------------
Connects to the UR5e via the RTDE *Receive* interface ONLY. It never sends a
control command and never moves the robot. It just prints, ~10x/sec:
  - the 6 joint angles (deg), with headroom to the +/-360 deg soft limit,
    highlighting the three WRIST joints (wrist1/2/3 = j4/j5/j6) that would
    carry any added bow tilt/yaw.
  - the TCP pose: x,y,z (m) and the rotation vector rx,ry,rz (axis-angle, rad).
  - the TCP force/torque (N, Nm).

HOW TO USE IT IN THE LAB (no motion commanded by this script)
-------------------------------------------------------------
1. IP defaults to 192.168.1.100 (confirmed 2026-06-01); pass --ip to override.
2. Run:  python lab_diag_readonly.py
3. Put the robot in FREEDRIVE using the teach-pendant button (hardware), so you
   can move the bow by hand. This script keeps reading the whole time.
4. Walk through the punch-list below, reading the live numbers off the console.

PUNCH-LIST THIS SCRIPT ANSWERS (read-only)
------------------------------------------
(d) WRIST HEADROOM: Freedrive to the A-string contact pose. Note wrist1/2/3.
    How many degrees of room before +/-360? Is wrist2 near 0 or +/-180 (wrist
    singularity risk)? This tells us how much tilt/yaw the wrist can absorb.
(b) BETA RANGE: With the bow on the A string, hand-slide the contact point
    toward the bridge, then toward the fingerboard. Record the x,y,z at the
    musically-usable extremes (where tone is still acceptable to your ear).
    That xyz span IS the beta actuation range.
(*) BAKED TILT/YAW: At the normal A-string playing pose, record rx,ry,rz.
    That is the bow's current fixed orientation; tilt/yaw actions perturb it.

NOT covered here (need motion -> separate script + explicit go-ahead):
(a) commanded bow speed (speedL vs servoL re-timing)
(c) tilt-under-force coupling (forceMode active while rotating the TCP)

Press Ctrl+C to stop.
"""

import argparse
import math
import sys
import time

try:
    import rtde_receive
except ImportError:
    sys.exit(
        "ur_rtde not importable. Activate the env that has it "
        "(see feedback_macos_ur_rtde_install: needs boost@1.85)."
    )

DEFAULT_IP = "192.168.1.100"  # robot static IP, confirmed 2026-06-01; override with --ip
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
WRIST_IDX = {3, 4, 5}
SOFT_LIMIT_DEG = 360.0  # UR e-series nominal per-joint range is +/-360 deg


def fmt_joint(name, rad, is_wrist):
    deg = math.degrees(rad)
    headroom = SOFT_LIMIT_DEG - abs(deg)
    mark = " <-- WRIST" if is_wrist else ""
    return f"  {name:>8}: {deg:+8.2f} deg  (headroom {headroom:6.1f}){mark}"


def main():
    ap = argparse.ArgumentParser(description="Read-only UR5e diagnostic.")
    ap.add_argument("--ip", default=DEFAULT_IP, help=f"robot IP (default {DEFAULT_IP})")
    ap.add_argument("--hz", type=float, default=10.0, help="print rate (default 10)")
    args = ap.parse_args()

    print(f"Connecting (READ-ONLY) to {args.ip} ...")
    rtde_r = rtde_receive.RTDEReceiveInterface(args.ip)
    print("Connected. Receive-only — this process sends NO control commands.\n")
    print("Use the pendant Freedrive button to move the bow by hand.\n")

    period = 1.0 / args.hz
    try:
        while True:
            q = rtde_r.getActualQ()              # 6 joint angles, rad
            pose = rtde_r.getActualTCPPose()     # x,y,z, rx,ry,rz
            force = rtde_r.getActualTCPForce()   # Fx,Fy,Fz, Tx,Ty,Tz

            print("\033[2J\033[H", end="")  # clear screen, cursor home
            print(f"=== UR5e READ-ONLY DIAG  ({time.strftime('%H:%M:%S')}) ===\n")

            print("JOINTS (deg, headroom to +/-360):")
            for i, (name, val) in enumerate(zip(JOINT_NAMES, q)):
                print(fmt_joint(name, val, i in WRIST_IDX))

            x, y, z, rx, ry, rz = pose
            print("\nTCP POSE:")
            print(f"  xyz : {x:+.4f}  {y:+.4f}  {z:+.4f}  (m)")
            print(f"  rot : {rx:+.4f}  {ry:+.4f}  {rz:+.4f}  (axis-angle, rad)")

            fx, fy, fz, tx, ty, tz = force
            print("\nTCP FORCE/TORQUE:")
            print(f"  F : {fx:+7.2f} {fy:+7.2f} {fz:+7.2f}  (N)")
            print(f"  T : {tx:+7.3f} {ty:+7.3f} {tz:+7.3f}  (Nm)")

            print("\n(b) slide contact pt toward/away from bridge -> watch xyz span")
            print("(d) wrist1/2/3 headroom = room for added tilt/yaw")
            print("Ctrl+C to stop.")
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped. (No motion was ever commanded.)")
    finally:
        rtde_r.disconnect()


if __name__ == "__main__":
    main()
