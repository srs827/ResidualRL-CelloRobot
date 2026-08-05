"""
find_liftoff.py

Find a SAFE RESET for the bow: not just a direction that lifts off the string,
but a clearance at which the bow can retract, TRAVEL to another point on the
string, and descend — all without dragging on the string, and without exceeding
the robot's reach.

Why this shape: the string is a straight line along the bow axis. If we lift the
bow along a fixed direction d by a fixed clearance, the resulting "rail" is
parallel to the string and stays a constant distance from it. So ONE good
(direction, clearance) pair defines a safe reset for the whole string — but it
must be validated against the FULL cycle (retract -> travel -> descend), because
a direction that lifts a little can still re-contact during the lateral travel.

This tool does NOT rely on the F/T sensor (it appears untared/unreliable here).
You judge each test by WATCHING the bow.

Workflow:
    1. Pick a candidate lift direction.
    2. Enter a clearance distance (mm).
    3. The tool runs the full reset between two bow positions:
         contact@fracA -> retract -> travel at clearance -> descend@fracB
       You watch whether the bow stays off the string during retract+travel and
       only re-contacts at the final descent.
    4. When a (direction, clearance) passes, it prints both to paste into
       recording_script.py.

Run:  python find_liftoff.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import time
import rtde_control
import rtde_receive

ROBOT_IP = "192.168.1.100"

# A-string waypoints (from the working paganini URScript). Edit for another string.
FROG = np.array([.336637615375, .773335607743, .103937252349,
                 -1.369835518384, -2.336199267621, 1.326965437172])
TIP  = np.array([.525205911288, .350983193771, .214779688012,
                 -1.369835518377, -2.336199267606, 1.326965437192])

TCP_OFFSET  = [0.028210348281514253, -0.09610723587300697,
               -0.09969041498611403, 0.0, 0.0, 0.0]
PAYLOAD_KG  = 0.260000
PAYLOAD_COG = [0.050000, -0.008000, 0.024000]

MOVE_SPEED = 0.05
MOVE_ACCEL = 0.3
SAFE_REACH = 0.83      # clamp to avoid joint-limit faults

# The two bow positions the reset travels BETWEEN during a test.
TEST_FRAC_A = 0.65
TEST_FRAC_B = 0.45


def axisangle_to_R(rx, ry, rz):
    theta = np.sqrt(rx*rx + ry*ry + rz*rz)
    if theta < 1e-9:
        return np.eye(3)
    k = np.array([rx, ry, rz]) / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)


def clamp_reach(pose, max_reach=SAFE_REACH):
    out = np.array(pose, dtype=float).copy()
    r = np.linalg.norm(out[:3])
    if r > max_reach:
        out[:3] = out[:3] * (max_reach / r)
        print(f"      (reach-clamped {r:.3f}->{max_reach:.3f}m)")
    return out


def bow_pose(frac):
    """Pose on the string at a given bow fraction (0=frog, 1=tip)."""
    return FROG + frac * (TIP - FROG)


def candidates(pose):
    """Labelled candidate lift directions (base-frame unit vectors)."""
    R = axisangle_to_R(pose[3], pose[4], pose[5])
    bow = (TIP[:3] - FROG[:3]); bow /= np.linalg.norm(bow)

    cands = []
    cands.append(("world +Z (up)",  np.array([0, 0, 1.0])))
    out = np.array([0.89583677, 0.04158029, 0.44243367]); out /= np.linalg.norm(out)
    cands.append(("script 'out'",   out))
    for i, nm in enumerate(['toolX', 'toolY', 'toolZ']):
        cands.append((f"+{nm}",  R[:, i]))
        cands.append((f"-{nm}",  -R[:, i]))
    for i, nm in enumerate(['toolX', 'toolY', 'toolZ']):
        c = np.cross(bow, R[:, i])
        if np.linalg.norm(c) > 1e-6:
            c /= np.linalg.norm(c)
            cands.append((f"+bow×{nm}",  c))
            cands.append((f"-bow×{nm}",  -c))
    m = out + 0.5*np.array([0, 0, 1.0]); m /= np.linalg.norm(m)
    cands.append(("out+0.5*Zup",  m))
    return cands


def perp_fraction(d, bow):
    """Fraction of d perpendicular to the bow (how much it actually lifts)."""
    d = d/np.linalg.norm(d)
    perp = d - np.dot(d, bow)*bow
    return float(np.linalg.norm(perp))


def run_reset_cycle(rc, rr, lift_dir, clearance):
    """
    Full reset between TEST_FRAC_A and TEST_FRAC_B at the given lift dir/clearance.
    The human watches each step. Sequence:
      on string @A -> retract along lift_dir -> travel at clearance to above B
      -> descend onto B
    """
    d = np.asarray(lift_dir, dtype=float); d /= np.linalg.norm(d)

    A = clamp_reach(bow_pose(TEST_FRAC_A))
    B = bow_pose(TEST_FRAC_B)

    rc.moveL(A.tolist(), MOVE_SPEED, MOVE_ACCEL)
    time.sleep(0.4)
    print("      [1/4] on string at A")

    retr = A.copy(); retr[:3] = A[:3] + clearance*d
    retr = clamp_reach(retr)
    rc.moveL(retr.tolist(), MOVE_SPEED, MOVE_ACCEL)
    time.sleep(0.4)
    print("      [2/4] retracted (watch: did it LEAVE the string?)")

    aboveB = B.copy(); aboveB[:3] = B[:3] + clearance*d
    aboveB = clamp_reach(aboveB)
    rc.moveL(aboveB.tolist(), MOVE_SPEED, MOVE_ACCEL)
    time.sleep(0.4)
    print("      [3/4] traveled to above B (watch: any DRAG on the way?)")

    Bc = clamp_reach(B)
    rc.moveL(Bc.tolist(), MOVE_SPEED, MOVE_ACCEL)
    time.sleep(0.4)
    print("      [4/4] descended onto string at B")


def main():
    print("Connecting to", ROBOT_IP, "...")
    rc = rtde_control.RTDEControlInterface(ROBOT_IP)
    rr = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rc.setTcp(TCP_OFFSET)
    rc.setPayload(PAYLOAD_KG, PAYLOAD_COG)

    bow = (TIP[:3] - FROG[:3]); bow /= np.linalg.norm(bow)

    try:
        start = clamp_reach(bow_pose(TEST_FRAC_A))
        print("Moving to start contact point...")
        rc.moveL(start.tolist(), MOVE_SPEED, MOVE_ACCEL)
        time.sleep(0.5)

        cands = candidates(np.array(rr.getActualTCPPose()))

        while True:
            print("\n" + "="*60)
            print("Candidate lift directions (perp = fraction that actually lifts):")
            for i, (label, v) in enumerate(cands):
                print(f"  [{i:2d}] {label:14s} {np.round(v,3)}  perp={perp_fraction(v,bow):.2f}")
            print("  [ q] quit")
            choice = input("\nPick a direction number: ").strip()
            if choice == 'q':
                break
            if not choice.isdigit() or int(choice) >= len(cands):
                print("  invalid"); continue

            label, v = cands[int(choice)]
            cl_in = input("  clearance in mm (e.g. 30): ").strip()
            try:
                clearance = float(cl_in) / 1000.0
            except ValueError:
                print("  invalid"); continue

            print(f"\n  Running FULL reset cycle: dir={label}, "
                  f"clearance={clearance*1000:.0f}mm, between fracs "
                  f"{TEST_FRAC_A} and {TEST_FRAC_B}")
            run_reset_cycle(rc, rr, v, clearance)

            verdict = input("\n  Did the bow stay OFF the string during retract+travel,\n"
                            "  only touching again at the final descent? (y/n): ").strip()
            if verdict.lower().startswith('y'):
                vv = v/np.linalg.norm(v)
                print("\n" + "*"*60)
                print("SAFE RESET FOUND. Paste into recording_script.py:")
                print(f"  RETRACT_DIR_BASE = np.array({np.round(vv,6).tolist()})")
                print(f"  LIFT_DISTANCE    = {clearance:.3f}   # {clearance*1000:.0f}mm")
                print("*"*60)
                if not input("Keep searching? (y/ENTER): ").strip().lower().startswith('y'):
                    break

            rc.moveL(start.tolist(), MOVE_SPEED, MOVE_ACCEL)

    finally:
        try:
            rc.stopScript()
        except Exception:
            pass
        print("Done.")


if __name__ == '__main__':
    main()