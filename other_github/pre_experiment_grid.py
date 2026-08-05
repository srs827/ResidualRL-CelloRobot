"""
pre_experiment_grid.py — run a SMALL depth x speed grid (+ deliberately-bad conditions)
to answer the two pre-experiments BEFORE investing in RL:
  A1: is there headroom over the baseline's nominal (depth/speed that sounds better)?
  A2: does the classifier separate good vs bad on OUR robot audio (reward valid in-domain)?

Reuses Samantha's recording_script machinery (record_stroke / calibrate / sessions) but
monkeypatches a SMALL grid and a SEPARATE output dir (pre_experiment/) so it does NOT
touch her dataset. The data we collect here is OURS (we ran it), no consent issue.

PREREQS:
  - recording_script.py present in the working tree (git pull origin first).
  - Run in the ur_rtde venv (.venv-rtde) at the robot, Focusrite connected, HAND ON E-STOP.
  - It will calibrate force<->depth (-> gives us LIGHT_CONTACT_Z for the RL too).

Grid: depth(5) x speed(3) x 1 zone x 2 repeats = 30 systematic + 6 bad x 2 = ~42 strokes.

Author: Claude (for Zixian, 2026-06-04).
"""

import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "Baseline-Runners"))  # for baseline_controller

# Use OUR copy of the recording pipeline. recording_grid.py = Samantha's recording_script.py
# + our fixes (import path, vertical-lift-first retract, no-arg get_summary). Her
# recording_script.py and baseline_controller.py stay byte-for-byte original.
import recording_grid as R   # noqa: E402

# --- small grid: keep her DEPTH_LEVELS (5) + BAD_CONDITIONS (6); trim the rest ---
R.SPEED_LEVELS = {"slow": 0.07, "medium": 0.10, "fast": 0.13}
R.BOW_ZONES = {"upper_middle": (0.45, 0.75)}
R.N_REPEATS = 2

# --- separate output so we never touch Samantha's dataset/ ---
R.OUTPUT_DIR = Path(HERE) / "pre_experiment"
R.AUDIO_DIR = R.OUTPUT_DIR / "audio"
R.STATE_DIR = R.OUTPUT_DIR / "states"
R.META_FILE = R.OUTPUT_DIR / "metadata.jsonl"
R.SESSION_LOG = R.OUTPUT_DIR / "sessions.jsonl"


def _rotvec_to_matrix(rv):
    rv = np.asarray(rv, float)
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def beta_axis(frog, tip):
    """Unit contact-point axis: along the string, perpendicular to the bow stroke and
    the into-string normal. The SIGN (which way = bridge) is unknown until heard."""
    frog = np.asarray(frog, float)
    tip = np.asarray(tip, float)
    e_bow = tip[:3] - frog[:3]
    e_bow /= np.linalg.norm(e_bow)
    e_normal = _rotvec_to_matrix(frog[3:])[:, 2]   # tool Z at frog = into-string
    e_beta = np.cross(e_bow, e_normal)
    e_beta /= np.linalg.norm(e_beta)
    return e_beta


def run_sweep(ctrl, conditions, session_id, n_reps=2, zone=None):
    """Run a list of conditions as DOWN strokes, recording audio. Each condition =
    {'depth','speed','beta_m','label'}; beta_m shifts the whole bow line along e_beta.
    Fixed bow_dir avoids the up/down confound. After EVERY stroke the bow retracts STRAIGHT
    UP first (recording_grid.retract) and returns to the fixed safe pose. zone defaults to
    upper_middle; pass a (start,end) tuple for a longer (frog-safe) stroke."""
    base_frog = np.asarray(R.FROG_TARGET_STR, float).copy()
    base_tip = np.asarray(R.TIP_TARGET_STR, float).copy()
    e_beta = beta_axis(base_frog, base_tip)
    zone = zone if zone is not None else R.BOW_ZONES["upper_middle"]
    try:
        for i, c in enumerate(conditions):
            b = float(c.get("beta_m", 0.0))
            R.FROG_TARGET_STR = base_frog.copy(); R.FROG_TARGET_STR[:3] += b * e_beta
            R.TIP_TARGET_STR = base_tip.copy();   R.TIP_TARGET_STR[:3] += b * e_beta
            print(f"\n[{i+1}/{len(conditions)}] {c['label']}")
            for rep in range(n_reps):
                R.record_stroke(ctrl=ctrl, depth=c["depth"], speed=c["speed"], bow_zone=zone,
                                condition_label=c["label"],
                                stroke_id=f"{session_id}_A_{c['label']}_{i:03d}_rep{rep}",
                                session_id=session_id, bow_dir="down", n_repeats=1,
                                tare_per_condition=(rep == 0))
                move_to_safe(ctrl)          # back to the FIXED safe pose after EVERY stroke
    finally:
        R.FROG_TARGET_STR = base_frog
        R.TIP_TARGET_STR = base_tip


def run_beta_show(ctrl, betas_m):
    """Move the bow to each beta contact point and HOLD (NO bowing) so you can SEE where it
    lands relative to the bridge. ENTER advances. depth/speed irrelevant (no stroke)."""
    base_frog = np.asarray(R.FROG_TARGET_STR, float).copy()
    base_tip = np.asarray(R.TIP_TARGET_STR, float).copy()
    e_beta = beta_axis(base_frog, base_tip)
    note = R.make_stroke_note(0.0015, 0.10, R.BOW_ZONES["upper_middle"], "down")
    try:
        for b in betas_m:
            R.FROG_TARGET_STR = base_frog.copy(); R.FROG_TARGET_STR[:3] += b * e_beta
            R.TIP_TARGET_STR = base_tip.copy();   R.TIP_TARGET_STR[:3] += b * e_beta
            R.place_bow_at_stroke_start(ctrl, note)   # descend onto the (shifted) string contact
            input(f"\n>>> beta {b*1000:+.0f} mm: bow is ON the string HERE. Look at the contact "
                  f"point vs the BRIDGE, then ENTER for next...")
    finally:
        R.FROG_TARGET_STR = base_frog
        R.TIP_TARGET_STR = base_tip
        move_to_safe(ctrl)


def move_to_safe(ctrl, lift=0.06):
    """Park the bow at a FIXED safe pose: the upper-middle zone-mid lifted straight UP
    off the string (clear of the string, in the safe zone, away from the frog singularity)."""
    frog = np.asarray(R.FROG_TARGET_STR, float)
    tip = np.asarray(R.TIP_TARGET_STR, float)
    safe = (frog + 0.6 * (tip - frog)).copy()
    safe[2] += lift                       # base +z is physically UP -> lifts off the string
    ctrl.rtde_c.moveL(safe.tolist(), 0.1, 0.3)
    print(f"parked at safe home (zone-mid + {lift*100:.0f}cm up)")


def _set_speed_slider_full(ip="192.168.1.100"):
    """Reset the global speed slider to 100%. verify_air's robot_setup leaves it at 20%,
    which silently bows everything at ~0.02 m/s and ruins the audio."""
    try:
        import rtde_io
        io = rtde_io.RTDEIOInterface(ip)
        io.setSpeedSlider(1.0)
        io.disconnect()
        print("speed slider -> 100%")
    except Exception as e:
        print(f"WARN: could not set speed slider ({e}). Set it to 100% on the pendant!")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="SAFE FIRST RUN: record ONE gentle condition (+ retract check) "
                         "instead of the full 42-stroke grid")
    ap.add_argument("--device", type=int, default=None,
                    help="audio input index (skips Focusrite auto-detect). Use for a "
                         "MOTION-ONLY test when the Focusrite isn't connected; the audio "
                         "is throwaway, NOT usable for the A1/A2 analysis.")
    ap.add_argument("--depth-only", action="store_true",
                    help="sweep DEPTH at ONE fixed speed/zone (isolate the depth axis)")
    ap.add_argument("--speed-only", action="store_true",
                    help="sweep SPEED at ONE fixed depth/zone (isolate the speed axis)")
    ap.add_argument("--beta-mm", type=str, default=None,
                    help="comma list of contact-point offsets in mm (e.g. '0,5,10,-5,-10'); "
                         "sweeps the sounding point along the string at fixed depth/speed/zone")
    ap.add_argument("--skip-cal", action="store_true",
                    help="skip force<->depth calibration (it REJECTS anyway; we use depth+audio)")
    ap.add_argument("--show", action="store_true",
                    help="with --beta-mm: MOVE to each contact point and HOLD (no bowing), so "
                         "you can SEE the position vs the bridge")
    ap.add_argument("--collect", action="store_true",
                    help="collect a good/bad dataset (systematic GOOD + deliberately-BAD "
                         "conditions) for training an in-domain classifier; long frog-safe zone")
    ap.add_argument("--reps", type=int, default=3, help="repeats per condition for --collect")
    args = ap.parse_args()

    print("=== PRE-EXPERIMENT " + ("TEST: 1 condition" if args.test else "small grid (~42)") +
          " — output: pre_experiment/ ===")
    R.list_audio_devices()
    if args.device is not None:
        device = args.device
        print(f"*** using audio device {device} (Focusrite auto-detect SKIPPED) ***")
        if not args.test:
            raise SystemExit("Refusing the full grid on a manual device — the analysis "
                             "needs Focusrite audio. Use --device only with --test.")
    else:
        device = R.find_focusrite_device()
    R.AUDIO_DEVICE = device   # record_stroke records from this device too

    ctrl = R.TargetStringBaselineController([])
    try:
        _set_speed_slider_full()           # verify_air may have left the slider at 20%
        ctrl.reset()                       # move to frog, verify contact
        R.validate_audio_chain(device)

        input("\nENTER to set payload + baseline F/T zero...")
        R.zero_ft_in_free_air(ctrl, set_payload=True)
        R.verify_ft_at_poses(ctrl)

        if not args.skip_cal and not getattr(R, "USE_FORCE_MODE", False):
            input("\nENTER to calibrate force<->depth "
                  "(bow presses in 0.25mm steps; auto-stops at 6N, then retracts)...")
            R.calibrate_force_depth(ctrl, at_frac=0.60)

        if args.collect:
            sid = R.datetime.now().strftime("%Y%m%d_%H%M%S")
            long_zone = (0.40, 0.90)   # 23.8 cm, ~2.4 s @ 0.10 m/s; reach-safe, frog-clear
            goods = [{"depth": d, "speed": s, "beta_m": 0.0,
                      "label": f"sys_d{d*1000:.1f}_s{s:.2f}"}
                     for d in (0.0008, 0.0015, 0.0025) for s in (0.07, 0.10)]
            bads = [{"depth": b["depth"], "speed": b["speed"], "beta_m": 0.0,
                     "label": f"bad_{b['label']}"} for b in R.BAD_CONDITIONS]
            conds = goods + bads
            n = len(conds) * args.reps
            input(f"\nENTER to COLLECT good/bad dataset: {len(goods)} GOOD + {len(bads)} BAD "
                  f"x {args.reps} reps = {n} strokes, long zone {long_zone}. HAND ON E-STOP...")
            run_sweep(ctrl, conds, sid, n_reps=args.reps, zone=long_zone)
        elif args.beta_mm is not None and args.show:
            betas = [float(x) / 1000.0 for x in args.beta_mm.split(",")]
            input(f"\nENTER: SHOW contact points {args.beta_mm} mm (move + HOLD, no bowing). "
                  f"HAND ON E-STOP...")
            run_beta_show(ctrl, betas)
        elif args.depth_only or args.speed_only or args.beta_mm is not None:
            sid = R.datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.beta_mm is not None:
                betas = [float(x) / 1000.0 for x in args.beta_mm.split(",")]
                conds = [{"depth": 0.0015, "speed": 0.10, "beta_m": b,
                          "label": f"beta_{b*1000:+.0f}mm"} for b in betas]
                what = f"contact-point beta {args.beta_mm} mm (depth 1.5mm, speed 0.10)"
            elif args.depth_only:
                conds = [{"depth": d, "speed": 0.10, "beta_m": 0.0,
                          "label": f"depth_{d*1000:.1f}mm"}
                         for d in sorted(R.DEPTH_LEVELS.values())]
                what = (f"DEPTH sweep {[round(d*1000,1) for d in sorted(R.DEPTH_LEVELS.values())]} "
                        f"mm @ speed 0.10")
            else:  # speed-only
                conds = [{"depth": 0.0015, "speed": s, "beta_m": 0.0,
                          "label": f"speed_{s:.2f}"} for s in sorted(R.SPEED_LEVELS.values())]
                what = f"SPEED sweep {sorted(R.SPEED_LEVELS.values())} m/s @ depth 1.5mm"
            input(f"\nENTER to run {what} — all DOWN strokes, upper-middle. HAND ON E-STOP...")
            run_sweep(ctrl, conds, sid)    # parks at safe home after each condition
        elif args.test:
            input("\nENTER to verify the retract clears the string...")
            R.probe_lift_axis(ctrl)
            input("\nENTER to record ONE test condition (HAND ON THE E-STOP)...")
            R.run_test_recording(ctrl)
            move_to_safe(ctrl)
        else:
            input("\nENTER to run the SMALL grid (HAND ON THE E-STOP)...")
            R.run_collection_session(ctrl, randomize=True)
            move_to_safe(ctrl)
    finally:
        try:
            ctrl.rtde_c.forceModeStop()
        except Exception:
            pass
        try:
            ctrl.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
