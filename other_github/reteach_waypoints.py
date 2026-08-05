"""
reteach_waypoints.py — FRESH re-calibration of the A-string FROG/TIP waypoints.

Context (2026-06-13): the cello was moved closer to the robot AND the bow was swapped +
re-rosined. The first re-teach attempt showed this is NOT a small pure translation (ends
shifted 160/83mm, bow angle 0.25-0.35 rad off old) — the bow swap alone invalidates the old
orientation. So we build a NEW waypoint pair from scratch.

READ-ONLY by design: it NEVER commands motion. You move the bow by hand with the pendant
Freedrive button; this only reads getActualTCPPose() and does the bookkeeping + checks.

THE ONE THING THAT MUST HOLD: FROG and TIP must be captured at the SAME bow angle
(bow_pose_full lerps the rotvec, and hardware_baseline asserts FROG[3:]==TIP[3:]).
TECHNIQUE: rest the bow ON the A string and SLIDE it from frog to tip WITHOUT lifting —
the string keeps the angle constant. (Or lock rotation in Freedrive if PolyScope supports it.)

What it does:
  1. capture the frog-end and tip-end contact poses (multi-sample average);
  2. check the two captures share one angle (frog-vs-tip spread must be small) — else RE-TEACH;
  3. output a clean pair with a common orientation (the mean), report reach (confirm under
     MAX_SAFE_REACH and that the move pulled the arm away from the singularity);
  4. print PASTE-READY blocks for recording_grid.py, baseline_controller.py (one consistent
     pair in both — Zixian's call; heads-up to Samantha), and the hardware_baseline fingerprint;
     save to reteach_<ts>.txt.

RETRACT_DIR_BASE (press normal): the string normal is cello-defined — unchanged by the bow
swap, and unchanged by a pure cello TRANSLATION; it only changes if the cello ROTATED. So KEEP
it provisionally and let acceptance_mount's F_axis vs |F| verify (clean axial = good; large
lateral = re-derive via find-liftoff). Run acceptance_mount AFTER updating the waypoints.

Run (robot ON; START with the pendant in REMOTE so the TCP guard can verify/set the bow-tip
frame — the script prompts when to switch to LOCAL + Freedrive; we never command motion):
  .venv-rtde/bin/python reteach_waypoints.py
"""
import sys
import time
from datetime import datetime

import numpy as np
from rtde_receive import RTDEReceiveInterface

sys.path.insert(0, ".")
import recording_grid as R          # OLD constants (for FYI comparison) + MAX_SAFE_REACH

ROBOT_IP = "192.168.1.100"
N_SAMPLES = 25
HOLD_STD_MM = 1.0
OLD_FROG = np.asarray(R.FROG_TARGET_STR, float)
OLD_TIP = np.asarray(R.TIP_TARGET_STR, float)


def reach(p):
    return float(np.linalg.norm(np.asarray(p)[:3]))


def capture(rtde_r, label):
    input(f"\n>>> {label} end: rest the bow ON the A string (NOT D) and position it at the {label} "
          f"end of the playable span. SLIDE along the string — do NOT lift between FROG and TIP — "
          f"so both ends keep the same angle. Hold still, then press Enter...")
    poses = np.array([np.asarray(rtde_r.getActualTCPPose(), float) for _ in range(N_SAMPLES)
                      if not time.sleep(0.04)])
    mean = poses.mean(0)
    spread_mm = float(np.linalg.norm(poses[:, :3].std(0)) * 1000)
    flag = "" if spread_mm <= HOLD_STD_MM else f"  !! moved {spread_mm:.1f}mm during capture — redo"
    # Frame backstop (2026-07-30): a flange-TCP capture reads ~0.10m SHORTER reach than a
    # bow-tip one, so a sane band catches the 141mm frame error even on an unverified run.
    if not (0.60 < reach(mean) < R.MAX_SAFE_REACH):
        flag += (f"  !! reach {reach(mean):.3f}m outside the bow-tip band "
                 f"(0.60-{R.MAX_SAFE_REACH:.2f}m) — TCP may be the FLANGE; run check_tcp.py")
    print(f"  {label}: xyz={np.round(mean[:3], 4).tolist()}  reach={reach(mean):.3f}m  "
          f"(spread {spread_mm:.2f}mm){flag}")
    return mean


def main():
    print(">>> FRESH WAYPOINT CALIBRATION (read-only; you drive via pendant Freedrive — robot will "
          "NOT move on its own).\n>>> TECHNIQUE: rest the bow on the A string and SLIDE frog->tip "
          "without lifting, so both ends share one angle.")
    rtde_r = RTDEReceiveInterface(ROBOT_IP)

    # TCP-frame guard (2026-07-30, hardened per review): capture() trusts getActualTCPPose,
    # which reports in whatever TCP persists on the controller — if Samantha's flange stack
    # ran last, every captured waypoint would be ~141mm off (the exact 06-21 incident).
    # This step needs the pendant in REMOTE; the script prompts when to switch back to
    # LOCAL + Freedrive for the actual capture. setTcp is a setting, not a motion.
    try:
        import rtde_control
        from tcp_guard import ensure_bowtip_tcp, JUMP_WARN_MM

        # SHARED-ARM GATE: constructing RTDEControlInterface uploads + starts a control
        # script, preempting any running/paused program (pendant .urp or Samantha's stack).
        # Only seize control when the controller is idle (runtime_state 1 == STOPPED; != 1
        # also catches PAUSED, which a script upload would convert to restart-from-top).
        _rt = rtde_r.getRuntimeState()
        if _rt != 1:
            raise RuntimeError(f"controller busy (runtime_state={_rt}, not STOPPED) — another "
                               "program is running/paused on the shared arm; not seizing control")

        _c = rtde_control.RTDEControlInterface(ROBOT_IP)
        try:
            ensure_bowtip_tcp(_c, rtde_r, label="reteach_waypoints")
            p_pre = np.asarray(rtde_r.getActualTCPPose(), float)  # robot static, bow-tip frame
        finally:
            try:
                _c.stopScript()
            except Exception:
                pass
            try:
                _c.disconnect()
            except Exception:
                pass
        # Falsify the persistence assumption: the TCP the guard set must SURVIVE
        # stopScript/disconnect, or the receive-only captures below are NOT in our frame.
        _jump = float(np.linalg.norm(
            np.asarray(rtde_r.getActualTCPPose(), float)[:3] - p_pre[:3]) * 1000.0)
        if _jump > JUMP_WARN_MM:
            raise RuntimeError(f"TCP did not survive stopScript (reported pose moved "
                               f"{_jump:.0f}mm after release) — captures would be mis-framed")
        input("\n>>> TCP verified/set. NOW switch the pendant to LOCAL, enable Freedrive, "
              "then press Enter...")
    except Exception as e:
        print(f"\n!! could not verify/set the TCP frame ({e}).")
        print("!! Most likely: pendant is in LOCAL mode — the guard needs REMOTE for setTcp.")
        print("!! Flip to REMOTE and rerun (the script prompts when to switch back to Local")
        print("!! for Freedrive), or run check_tcp.py first. If the flange-TCP stack ran")
        print("!! since our last bow-tip session, captures now would be ~141mm WRONG.")
        if input(">>> capture anyway with an UNVERIFIED TCP frame? [y/N] ").strip().lower() != "y":
            rtde_r.disconnect()
            return

    try:
        frog = capture(rtde_r, "FROG")
        tip = capture(rtde_r, "TIP")
    finally:
        rtde_r.disconnect()

    # The bow was swapped + cello moved -> build a NEW pair. MUST: same angle at both ends.
    frog_tip_spread = float(np.linalg.norm(frog[3:] - tip[3:]))
    ori = (frog[3:] + tip[3:]) / 2.0
    frog_out = np.concatenate([frog[:3], ori])
    tip_out = np.concatenate([tip[:3], ori])
    usable = frog_tip_spread < 0.05
    print(f"\n=== orientation consistency (FROG vs TIP — the one thing that MUST be small) ===")
    print(f"  frog-vs-tip angle spread = {frog_tip_spread:.4f} rad  "
          + ("(GOOD: both ends at one angle -> usable line)" if usable
             else "(TOO LARGE -> your two captures are at different angles. RE-TEACH by sliding the bow "
                  "ALONG the A string WITHOUT lifting. DO NOT use the values below.)"))

    span = float(np.linalg.norm(tip_out[:3] - frog_out[:3]))
    old_span = float(np.linalg.norm(OLD_TIP[:3] - OLD_FROG[:3]))
    rf, rt = reach(frog_out), reach(tip_out)
    print(f"\n=== reach + span ===")
    print(f"  reach FROG {rf:.3f}m, TIP {rt:.3f}m  "
          f"({'both < MAX_SAFE_REACH ' + str(R.MAX_SAFE_REACH) + ' ✓' if max(rf, rt) < R.MAX_SAFE_REACH else '⚠️ EXCEEDS MAX_SAFE_REACH!'})"
          + ("  (arm pulled IN vs old ✓ = away from the high-reach singularity)"
             if max(rf, rt) < max(reach(OLD_FROG), reach(OLD_TIP)) else ""))
    print(f"  playable span {span*1000:.0f}mm  (old {old_span*1000:.0f}mm — FYI; pick the same string "
          f"endpoints if you want to match the old span)")
    print(f"  FYI vs old waypoints: FROG moved {np.linalg.norm(frog_out[:3]-OLD_FROG[:3])*1000:.0f}mm, "
          f"TIP {np.linalg.norm(tip_out[:3]-OLD_TIP[:3])*1000:.0f}mm (bow swap + cello move; not pure translation)")

    # bow-axis alignment: FROG->TIP must run along the bow's long axis, else the robot draws the
    # bow diagonally and the string contact point (sounding point) drifts across the stroke.
    def rotvec_R(rv):
        th = float(np.linalg.norm(rv)); k = rv / th if th > 1e-9 else rv * 0.0
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    bow_tool = rotvec_R(OLD_FROG[3:]).T @ (OLD_TIP[:3] - OLD_FROG[:3])     # bow axis in tool frame (old cal)
    expected = rotvec_R(ori) @ (bow_tool / np.linalg.norm(bow_tool))      # expected stroke dir at new angle
    stroke = (tip_out[:3] - frog_out[:3]); stroke = stroke / (np.linalg.norm(stroke) + 1e-9)
    ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(expected, stroke)), -1, 1))))
    drift_mm = span * np.sin(np.radians(ang)) * 1000
    print(f"\n=== bow-axis alignment (stroke FROG->TIP vs the bow's long axis) ===")
    print(f"  off-axis angle = {ang:.1f}° -> sounding point would drift ~{drift_mm:.0f}mm across the stroke  "
          + ("(GOOD: drawn along the bow ✓)" if ang < 3
             else "(SKEWED -> use TOOL-frame jog along the bow axis (tool-X) to set TIP, re-capture)"))

    normal_verdict = ("KEEP old RETRACT_DIR_BASE provisionally — the string normal is cello-defined "
                      "(unchanged by the bow swap or a pure cello translation; changes only on cello "
                      "ROTATION). acceptance_mount F_axis vs |F| MUST verify: clean axial = good; large "
                      "lateral = cello rotated -> re-derive (find-liftoff).")

    def vec(p): return "np.array([" + ", ".join(f"{x:.9f}" for x in p) + "])"
    block = [
        "# ---- recording_grid.py  AND  Baseline-Runners/baseline_controller.py ----",
        "#      (Zixian's call: ONE consistent pair in BOTH files — also resolves the old 2mm z",
        "#       discrepancy. baseline_controller is Samantha's file -> heads-up to her.)",
        f"FROG_TARGET_STR = {vec(frog_out)}",
        f"TIP_TARGET_STR  = {vec(tip_out)}",
        "",
        "# ---- hardware_baseline.py (replace the fingerprint values in the assert) ----",
        "# [FROG[0], FROG[2], TIP[0], TIP[2]] =",
        f"#   [{frog_out[0]:.9f}, {frog_out[2]:.9f}, {tip_out[0]:.9f}, {tip_out[2]:.9f}]",
        "",
        f"# RETRACT_DIR_BASE: {normal_verdict}",
    ]
    text = "\n".join(block)
    print("\n" + "=" * 70 + "\nPASTE-READY:" + ("" if usable else "  ⚠️ ANGLE SPREAD TOO LARGE — RE-TEACH FIRST")
          + "\n" + "=" * 70 + "\n" + text)

    out = f"reteach_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(out, "w") as fh:
        fh.write(text + f"\n\n# frog_tip_spread={frog_tip_spread:.4f}rad usable={usable} "
                 f"span={span*1000:.0f}mm reach_frog={rf:.3f} reach_tip={rt:.3f}\n")
    print(f"\nsaved -> {out}")
    if usable:
        print("NEXT: send reteach_*.txt to Claude (updates 3 files) -> then "
              "acceptance_mount.py --quick --frac 0.30/0.50/0.65 to measure CONTACT_ANCHORS + verify the normal.")
    else:
        print("RE-TEACH: slide the bow along the A string without lifting, then run this again.")


if __name__ == "__main__":
    main()
