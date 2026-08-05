"""
find_normal.py — re-derive the press normal (RETRACT_DIR_BASE) — MULTI-POINT version.

Per Zixian's workflow (2026-06-27): instead of one point, measure the outward normal at SEVERAL
points along the bow line and average, so a single noisy/friction-biased point can't skew it.
At each point it uses the LIFT-OFF GRADIENT method (robust): at a light contact, probe +-1mm on
each base axis; the direction where |F| DROPS fastest = the outward normal. This uses |F|
DIFFERENCES, so it is immune to friction direction and to a bad tare, and never trusts the old
normal. METHOD A (raw force vector) is reported per point only as a cross-check — it is
friction-contaminated (force = normal + tangential friction), so the AVERAGED GRADIENT is the
answer.

RUN ORDER: do this AFTER re-teaching FROG/TIP (teach_tip_on_axis.py) and updating recording_grid's
waypoints, so bow_pose_full() points at the CURRENT string.

Output: averaged RETRACT_DIR_BASE + per-point spread (consistency) + angle vs old. Paste to Claude
-> we update recording_grid.RETRACT_DIR_BASE.

Run (robot ON + REMOTE; HAND ON E-STOP):  .venv-rtde/bin/python find_normal.py
  optional: .venv-rtde/bin/python find_normal.py 0.3,0.5,0.7   (custom fracs)
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
VEL, ACC = 0.02, 0.15
FRACS = [0.30, 0.50, 0.70]   # reach-safe on the 2026-06-27 line (bow-tip reach 0.741/0.703/0.677)
LIFT = 0.050                 # 50mm straight up = clear free air
STEP = 0.0005               # descent step
PROBE = 0.0010              # +-1mm probe for the gradient (signal vs sensor noise)
CONTACT_N = 1.0             # press to ~this for a clean contact before probing
ABORT_N = 4.0


def fvec(r, n=20):
    return np.mean([np.asarray(r.getActualTCPForce(), float)[:3] for _ in range(n) if not time.sleep(0.012)], axis=0)


def favg_mag(r, n=10):
    return float(np.linalg.norm(fvec(r, n)))


def fmag(r):
    return float(np.linalg.norm(np.asarray(r.getActualTCPForce(), float)[:3]))


def safe(r):
    try:
        return (not r.isProtectiveStopped()) and int(r.getSafetyMode()) == 1
    except Exception:
        return False


def goL(c, r, pose):
    ok = c.moveL(R.clamp_reach(np.asarray(pose, float)).tolist(), VEL, ACC)
    time.sleep(0.05)
    return bool(ok) and safe(r)


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1, 1))))


def measure_at(c, r, frac):
    """lift -> tare in free air -> descend to light contact -> METHOD A + B. Returns dict or None."""
    target = R.bow_pose_full(frac)
    up = target.copy(); up[2] += LIFT
    if not goL(c, r, up):
        print(f"  frac {frac}: lift failed."); return None
    c.zeroFtSensor(); time.sleep(0.4)               # tare in clear air (Zixian: cancel gravity here)
    print(f"  frac {frac}: free-air |F|={fmag(r):.3f}N after tare; descending...")
    z = up[2]; contact = None
    while z > target[2] - 0.006:
        z -= STEP
        p = up.copy(); p[2] = z
        if not goL(c, r, p):
            print("    descent move/safety fail."); break
        m = fmag(r)
        if m > ABORT_N:
            print(f"    !! |F|={m:.1f}N abort"); break
        if m >= CONTACT_N:
            contact = p.copy()
            print(f"    light contact z={z:.4f} |F|={m:.2f}N")
            break
    if contact is None:
        print("    no contact in range."); goL(c, r, up); return None

    center = np.asarray(r.getActualTCPPose(), float)
    Fc = fvec(r, 25)                                  # METHOD A (cross-check only)
    n_force = Fc / (np.linalg.norm(Fc) + 1e-9)

    grad = np.zeros(3)                                # METHOD B — lift-off gradient (the answer)
    for axi in range(3):
        vals = {}
        for s in (+1, -1):
            p = center.copy(); p[axi] += s * PROBE
            if not goL(c, r, p):
                print("    probe move fail."); goL(c, r, up); return None
            m = favg_mag(r, 10)
            if m > ABORT_N:
                print(f"    !! |F|={m:.1f}N abort during probe"); goL(c, r, up); return None
            vals[s] = m
            goL(c, r, center)
        grad[axi] = (vals[+1] - vals[-1]) / (2 * PROBE)
    goL(c, r, up)                                     # retract to free air
    gn = float(np.linalg.norm(grad))
    n_grad = (-grad / gn) if gn > 1.0 else None
    print(f"    grad |.|={gn:.0f} N/m -> n_grad={None if n_grad is None else np.round(n_grad,3).tolist()}"
          f"  (n_force={np.round(n_force,3).tolist()})")
    return {"frac": frac, "n_force": n_force, "n_grad": n_grad}


def main():
    fracs = FRACS
    if len(sys.argv) > 1:
        fracs = [float(x) for x in sys.argv[1].split(",")]
    c = rtde_control.RTDEControlInterface(ROBOT_IP)
    r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    io = rtde_io.RTDEIOInterface(ROBOT_IP)
    results = []
    try:
        io.setSpeedSlider(0.79)
        c.setTcp(TCP_OFFSET); time.sleep(0.2)
        try:
            c.setPayload(float(R.FT_PAYLOAD_KG), list(R.FT_PAYLOAD_COG))
        except Exception:
            pass
        if not safe(r):
            raise RuntimeError("robot not Normal/Running — clear pendant, set REMOTE, rerun.")
        print(f"  measuring normal at fracs {fracs} (lift-off gradient, averaged)\n")
        for fr in fracs:
            m = measure_at(c, r, fr)
            if m is not None:
                results.append(m)
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
        print("  session closed; robot stopped.")

    grads = [m["n_grad"] for m in results if m["n_grad"] is not None]
    if not grads:
        print("\n  no usable gradient at any point — check bow/string contact, rerun.")
        return
    old = np.asarray(R.RETRACT_DIR_BASE, float); old /= np.linalg.norm(old)
    avg = np.mean(grads, axis=0); avg /= np.linalg.norm(avg)
    spread = max((ang(a, b) for a in grads for b in grads), default=0.0)
    print("\n" + "=" * 66)
    print(f"  per-point lift-off normals ({len(grads)} pts):")
    for m in results:
        if m["n_grad"] is not None:
            g = m["n_grad"]
            print(f"    frac {m['frac']:.2f}: [{g[0]:+.4f}, {g[1]:+.4f}, {g[2]:+.4f}]"
                  f"   (force-vec cross-check vs grad: {ang(m['n_force'], g):.1f}°)")
    print(f"\n  >>> AVERAGED RETRACT_DIR_BASE = [{avg[0]:.6f}, {avg[1]:.6f}, {avg[2]:.6f}]")
    print(f"  point-to-point spread = {spread:.1f}°  "
          + ("(GOOD — consistent normal)" if spread < 12 else "(HIGH — line may be curved / noisy; paste & we look)"))
    print(f"  angle vs OLD normal   = {ang(avg, old):.1f}°   (old = {np.round(old,4).tolist()})")
    print("  Paste ALL of the above to Claude -> we update recording_grid.RETRACT_DIR_BASE.")


if __name__ == "__main__":
    main()
