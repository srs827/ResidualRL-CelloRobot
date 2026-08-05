"""
restore_arm.py — moveJ the arm back to the REFERENCE joint configuration (elbow branch restore).

WHY: our frog/tip waypoints are CARTESIAN poses only. A Cartesian pose has multiple IK branches
(elbow up/down etc.); servoL/moveL stay in whatever branch the arm is currently in, so another
session's moveJ (e.g. Samantha's 5-config script) can leave the arm in a DIFFERENT branch and
every Cartesian waypoint then lands with shifted contact + loudness (the 2026-07-10 incident).
preflight.py DETECTS this (joints vs reference); this script FIXES it.

The reference joints are whatever posture the arm held at `preflight.py --accept` time — accept
at a sensible parked pose. Run this whenever preflight FAILs the arm-configuration check.

SAFETY: moveJ sweeps joint-space arcs — the bow can swing through space. Visually confirm the
arm is CLEAR of the cello (retract/park it first if not), keep a hand on the E-STOP.

Usage:  .venv-rtde/bin/python restore_arm.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(HERE, "config", "reference_state.json")
TOL_DONE = 0.02      # rad — already at the reference, nothing to do
SPEED, ACCEL = 0.35, 0.5   # conservative moveJ (defaults are 1.05/1.4)


def main():
    if not os.path.exists(REF_PATH):
        raise SystemExit(f"no {REF_PATH} — run preflight.py --accept first (nothing to restore to).")
    ref = json.load(open(REF_PATH))
    q_ref = ref.get("joints")
    if not q_ref:
        raise SystemExit("reference_state.json has no joints — re-run preflight.py --accept with the robot on.")

    import rtde_receive
    import rtde_control
    r = rtde_receive.RTDEReceiveInterface("192.168.1.100")
    q_now = [float(x) for x in r.getActualQ()]
    dq = [abs(a - b) for a, b in zip(q_now, q_ref)]
    print(f"  current joints:   {[round(x, 4) for x in q_now]}")
    print(f"  reference joints: {[round(x, 4) for x in q_ref]}  (accepted {ref.get('accepted', '?')})")
    print(f"  max |Δjoint| = {max(dq):.3f} rad" + ("  (>1 rad = elbow-branch flip)" if max(dq) > 1 else ""))
    if max(dq) < TOL_DONE:
        print("  already at the reference configuration — nothing to do.")
        return 0
    if r.isProtectiveStopped() or int(r.getSafetyMode()) != 1:
        raise SystemExit("robot is protective-stopped / not in normal safety mode — clear that first.")

    if input(">>> moveJ will SWEEP the arm (bow swings through space). Confirm the arm is CLEAR "
             "of the cello and your hand is on the E-STOP. Type 'go' to move: ").strip().lower() != "go":
        print("  aborted — nothing moved.")
        return 1
    c = rtde_control.RTDEControlInterface("192.168.1.100")
    try:
        ok = c.moveJ(q_ref, SPEED, ACCEL)
    finally:
        try:
            c.disconnect()
        except Exception:
            pass
    time.sleep(0.3)
    q_end = [float(x) for x in r.getActualQ()]
    dq_end = max(abs(a - b) for a, b in zip(q_end, q_ref))
    print(f"  moveJ returned {ok}; final max |Δjoint| = {dq_end:.4f} rad")
    if not ok or dq_end > 0.05:
        print("  !! RESTORE INCOMPLETE — do not trust Cartesian waypoints; investigate before bowing.")
        return 1
    print("  RESTORED — re-run preflight.py to confirm all checks PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
