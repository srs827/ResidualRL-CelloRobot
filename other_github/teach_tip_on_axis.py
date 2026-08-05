"""
teach_tip_on_axis.py — capture FROG + TIP in the CORRECT TCP frame, by hand-guide (teachMode).

Flow (Zixian 2026-06-27): for EACH point it (1) tells you to hand-move the bow there, (2) waits
for Enter, (3) grabs the pose + shows xyz/reach/|F|, (4) you Enter to accept or 'r' to re-move.
Symmetric for FROG then TIP. Capture at LIGHT touch (|F|≈0) so the waypoints = true-light line.

⚠️ TCP (2026-06-22 bug): getActualTCPPose() is only meaningful in the right TCP frame — the bow
tip is ~141mm from the flange. This script sets the TCP ITSELF (setTcp(TCP_OFFSET)) and uses RTDE
teachMode, so it never depends on the pendant's installation TCP.

TIP reuses FROG's orientation (consistent rotvec). After TIP it prints the off-axis (perpendicular
mm) of FROG->TIP vs the bow's long axis — keep it small so the sounding point doesn't drift.

GROUND TRUTH is your eyes: tape-mark the A string; the bow must touch THAT SAME mark at FROG & TIP.

Run (robot ON + REMOTE; HAND ON the bow; e-stop reachable):
  .venv-rtde/bin/python teach_tip_on_axis.py
"""
import sys
import time
from datetime import datetime

import numpy as np
import rtde_control
from rtde_receive import RTDEReceiveInterface

sys.path.insert(0, ".")
import recording_grid as R

ROBOT_IP = "192.168.1.100"
# MUST match baseline_controller.TCP_OFFSET (bow tip ~141mm from flange).
TCP_OFFSET = [0.028210348281514253, -0.09610723587300697, -0.09969041498611403, 0.0, 0.0, 0.0]
N = 25


def rotvec_R(rv):
    th = float(np.linalg.norm(rv)); k = rv / th if th > 1e-9 else rv * 0.0
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def grab(rtde_r):
    ps = np.array([np.asarray(rtde_r.getActualTCPPose(), float) for _ in range(N) if not time.sleep(0.03)])
    return ps.mean(0)


def fmag(rtde_r):
    return float(np.linalg.norm(np.asarray(rtde_r.getActualTCPForce())[:3]))


# Reuse the calibrated bow orientation for BOTH points — you do NOT hand-teach the angle.
# (The drift is ~translation; a translation doesn't rotate the string, so the bow attitude is
#  unchanged. You only place the tip POSITION; we apply this rotvec. Zixian 2026-06-27.)
OLD_ORI = np.asarray(R.FROG_TARGET_STR, float)[3:].copy()


def capture(rtde_r, name):
    """Prompt -> Enter when the tip is ON THE MARK (angle irrelevant) -> grab POSITION ->
    apply the calibrated OLD_ORI -> show -> Enter accept / 'r' re-move. Returns the pose."""
    while True:
        input(f"\n>>> Hand-place the BOW TIP on the tape mark at the {name} end, JUST touching "
              f"(|F|≈0). The ANGLE DOESN'T MATTER (we keep the calibrated orientation). Enter...")
        p = grab(rtde_r)
        p[3:] = OLD_ORI                       # position from your hand; orientation = calibrated
        f = fmag(rtde_r)
        light = "  (light ✓)" if f < 0.8 else "  ⚠ not light — lift the bow a hair, then 'r'"
        print(f"    {name}: xyz={np.round(p[:3],4).tolist()}  reach={np.linalg.norm(p[:3]):.3f}m  |F|={f:.2f}N{light}")
        if input(f"    Enter = accept {name}   |   'r' = re-place & re-capture: ").strip().lower() != "r":
            return p


def main():
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    frog = tip = None
    bow_axis = None
    try:
        rtde_c.setTcp(TCP_OFFSET); time.sleep(0.3)
        print(f">>> TCP set to bow tip {TCP_OFFSET[:3]}. Robot must be in REMOTE.")
        print(">>> Enabling hand-guide (teachMode) — you can now move the bow by hand.")
        rtde_c.teachMode()

        frog = capture(rtde_r, "FROG")
        tip = capture(rtde_r, "TIP")
        # Compare the NEW frog->tip direction to the OLD bow axis: ~parallel => translation only
        # (reused orientation valid); rotated => cello turned (reused rotvec suspect).
        old_axis = np.asarray(R.TIP_TARGET_STR, float)[:3] - np.asarray(R.FROG_TARGET_STR, float)[:3]
        old_axis = old_axis / np.linalg.norm(old_axis)
        d = tip[:3] - frog[:3]
        along = float(np.dot(d, old_axis))
        off_mm = float(np.linalg.norm(d - along * old_axis) * 1000)
        ang = float(np.degrees(np.arctan2(off_mm / 1000, abs(along) + 1e-9)))
        print(f"\n    new span={along*1000:.0f}mm   new line vs OLD bow axis: off={off_mm:.1f}mm ({ang:.1f}°)")
        print("    " + ("✓ ~parallel to old -> translation only -> reused orientation is valid"
                        if ang < 5 else
                        "⚠ new line ROTATED vs old -> cello likely turned -> reused rotvec may be wrong; tell Claude"))
    finally:
        try:
            rtde_c.endTeachMode()
        except Exception:
            pass
        try:
            rtde_c.stopScript(); rtde_c.disconnect()
        except Exception:
            pass
        rtde_r.disconnect()
        print("  teachMode off; robot stopped.")

    if frog is None or tip is None:
        print("  incomplete — nothing saved.")
        return
    span = float(np.linalg.norm(tip[:3] - frog[:3]))

    def vec(p): return "np.array([" + ", ".join(f"{x:.9f}" for x in p) + "])"
    text = "\n".join([
        "# ---- recording_grid.py  AND  Baseline-Runners/baseline_controller.py (one consistent pair) ----",
        "# captured in the CORRECT TCP frame (setTcp(TCP_OFFSET) + teachMode), 2026-06-27, LIGHT touch.",
        f"FROG_TARGET_STR = {vec(frog)}",
        f"TIP_TARGET_STR  = {vec(tip)}",
        "",
        "# ---- hardware_baseline.py fingerprint  [FROG[0], FROG[2], TIP[0], TIP[2]] ----",
        f"#   [{frog[0]:.9f}, {frog[2]:.9f}, {tip[0]:.9f}, {tip[2]:.9f}]",
    ])
    print("\n" + "=" * 70 + "\nPASTE-READY (send to Claude):\n" + "=" * 70 + "\n" + text)
    out = f"reteach_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    open(out, "w").write(text + f"\n\n# span={span*1000:.0f}mm TCP=bow_tip\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
