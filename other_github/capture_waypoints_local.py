"""
capture_waypoints_local.py — READ-ONLY FROG/TIP capture in LOCAL (pendant) mode.

Zixian's method (2026-06-27): drive the robot with the PENDANT in LOCAL mode. Set the bow to the
NEW desired angle (perpendicular to the string, hair flat, just touching) at the FROG; then use the
pendant's TRANSLATION jog (Move Tool / Move Base, orientation LOCKED) to slide to the TIP. The
translation jog mechanically GUARANTEES the angle stays constant, and it is the NEW angle (not a
reused old one). This script NEVER commands motion (rtde_receive only) -> safe in LOCAL mode.

⚠️ TCP: getActualTCPPose() returns the pose in the ROBOT's CURRENTLY-SET TCP frame. This script
reads it DIRECTLY and assumes the robot TCP is already the BOW TIP (our scripts — teach_tip_on_axis,
find_normal, baseline_controller — setTcp(TCP_OFFSET), which PERSISTS on the robot). DO NOT add a
flange->tip conversion: an earlier version did, and when the TCP was already the bow tip it
DOUBLE-counted 141mm and put the waypoints ~122mm too high (bow never touched the string).
SANITY: printed reach should be ~0.66-0.82m (bow-tip). If it reads ~0.10m LESS, the robot TCP is
the flange [0,0,0] — run any setTcp(TCP_OFFSET) script once first, or tell Claude.

Light touch = your EYES (we can't tare the F/T read-only): bow just kissing the string at each end.

Flow: place at FROG (set angle) -> Enter; TRANSLATION-jog to TIP -> Enter. Reports the orientation
drift FROG->TIP (≈0 if you translated) and prints paste-ready waypoints (bow-tip frame).

Run (robot in LOCAL/pendant mode, TCP already = bow tip):  .venv-rtde/bin/python capture_waypoints_local.py
"""
import time
from datetime import datetime

import numpy as np
from rtde_receive import RTDEReceiveInterface

ROBOT_IP = "192.168.1.100"
N = 25


def rotvec_R(rv):
    rv = np.asarray(rv, float); th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def grab(r):
    """Average the TCP pose (= bow tip, since the robot TCP is set to TCP_OFFSET). NO conversion."""
    return np.mean([np.asarray(r.getActualTCPPose(), float) for _ in range(N) if not time.sleep(0.03)], axis=0)


def ang(a, b):
    c = (np.trace(rotvec_R(a).T @ rotvec_R(b)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def capture(r, name, hint):
    input(f"\n>>> On the PENDANT, place the bow at the {name} end ({hint}). Enter when there...")
    p = grab(r)
    reach = float(np.linalg.norm(p[:3]))
    flag = "" if 0.60 < reach < 0.84 else "  ⚠ reach off — is the robot TCP the bow tip? (see header)"
    print(f"    {name}: bow-tip xyz={np.round(p[:3],4).tolist()}  reach={reach:.3f}m{flag}")
    return p


def main():
    r = RTDEReceiveInterface(ROBOT_IP)
    try:
        frog = capture(r, "FROG", "set the NEW angle: perpendicular to the string, hair flat, just touching")
        tip = capture(r, "TIP", "TRANSLATION-jog only (Move Tool/Base) so the angle stays locked, just touching")
    finally:
        r.disconnect()

    drift = ang(frog[3:], tip[3:])
    print(f"\n  orientation drift FROG->TIP = {drift:.2f}°  "
          + ("✓ angle held constant" if drift < 1.5 else "⚠ angle moved — redo TIP with TRANSLATION-only jog"))
    tip[3:] = frog[3:]                              # force identical (use the NEW frog angle)
    span = float(np.linalg.norm(tip[:3] - frog[:3]))

    def vec(p): return "np.array([" + ", ".join(f"{x:.9f}" for x in p) + "])"
    text = "\n".join([
        "# ---- recording_grid.py AND Baseline-Runners/baseline_controller.py (consistent pair) ----",
        "# READ-ONLY LOCAL-mode capture, NEW angle, translation-locked, bow-tip frame (no conversion). 2026-06-27.",
        f"FROG_TARGET_STR = {vec(frog)}",
        f"TIP_TARGET_STR  = {vec(tip)}",
        f"# fingerprint [FROG0,FROG2,TIP0,TIP2] = [{frog[0]:.9f}, {frog[2]:.9f}, {tip[0]:.9f}, {tip[2]:.9f}]",
    ])
    print(f"\n  span = {span*1000:.0f}mm")
    print("\n" + "=" * 70 + "\nPASTE-READY (send to Claude):\n" + "=" * 70 + "\n" + text)
    out = f"reteach_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    open(out, "w").write(text + f"\n# span={span*1000:.0f}mm drift={drift:.2f}deg\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
