"""
preflight.py — session-start calibrate-or-refuse checks (2026-07-10 consolidation).

Catches, BEFORE any strokes are wasted, the four incident classes from the 07-05..07-10 sprint:
  1. arm left in a different CONFIGURATION (elbow flip -> contact line + loudness both shift);
  2. dead/mis-set mic chain (48V off, wrong device/channel, gain bumped);
  3. stale reward deployment (which joblib version is live, and its expected level domain);
  4. (with --probe) contact/level drift: one zero-residual stroke must land in the expected
     force band and score in the expected feature-rms band.

Reference state lives in config/reference_state.json. Update it deliberately with --accept
after a session you trust (it snapshots joints + mic + reward hash + probe bands).

Usage (robot ON + REMOTE for --probe AND --accept — both bow one stroke; bare run is read-only):
  .venv-rtde/bin/python preflight.py                 # read-only checks (mic, joints, reward)
  .venv-rtde/bin/python preflight.py --probe         # + one zero stroke at --nominal-mm
  .venv-rtde/bin/python preflight.py --accept        # snapshot state as reference (incl. probe;
                                                     #   answer q at the prompt to skip bowing)

Exit code 0 = all checks pass; 1 = at least one FAIL (do not start a session).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLF = os.path.join(os.path.dirname(HERE), "classifier_pilot")
for _p in (HERE, os.path.join(HERE, "Baseline-Runners"), os.path.join(HERE, "rl", "residual"), CLF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REF_PATH = os.path.join(HERE, "config", "reference_state.json")


class ProbeSkipped(RuntimeError):
    """User chose q at the probe prompt — a deliberate skip, NOT a failure."""
JOINT_TOL_RAD = 0.10          # per-joint tolerance; an elbow flip differs by >1 rad
AMBIENT_BAND = (0.0002, 0.02)  # quiet-room ambient rms at the locked gain


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def mic_check():
    from audio_recorder import AudioRecorder
    rec = AudioRecorder()
    rec.start(max_seconds=2.5)
    time.sleep(2.2)
    a, sr = rec.stop()
    rms = float(np.sqrt(np.mean(a ** 2)))
    ok_dev = rec.is_focusrite
    ok_lvl = AMBIENT_BAND[0] < rms < AMBIENT_BAND[1]
    r1 = check("mic device", ok_dev, f"'{rec.device_name}' ch{rec.mapping or [1]}")
    r2 = check("ambient level", ok_lvl,
               f"rms={rms:.5f} (band {AMBIENT_BAND}; ~0.000x = 48V/cable dead, >>0.02 = noisy/gain)")
    return (r1 and r2), {"device": rec.device_name, "channel": (rec.mapping or [1]), "ambient_rms": round(rms, 6)}


def reward_check(ref, accepting=False):
    from feature_tone_reward import DEFAULT_PATH, gate_params
    sha = hashlib.sha256(open(DEFAULT_PATH, "rb").read()).hexdigest()[:16]
    meta_p = os.path.splitext(DEFAULT_PATH)[0] + "_meta.json"
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    gp = gate_params(meta.get("feature_set", "v5"))   # the ACTIVE gate for the DEPLOYED set
    detail = f"sha={sha} ver={meta.get('versioned_file', '?')} gate={gp}"
    out = {"reward_sha": sha, "reward_gate": gp}
    if accepting:      # accepting BLESSES the current state — comparing it against the OLD
        return check("reward version", True, detail + "  (new reference)"), out   # reference is
    if ref and ref.get("reward_sha") and ref["reward_sha"] != sha:                # the trap chain
        return check("reward version", False,
                     detail + f" != reference {ref['reward_sha']} — deliberate retrain? re---accept"), out
    ref_gate = (ref or {}).get("reward_gate")
    if ref_gate is None and (ref or {}).get("rms_floor") is not None:
        ref_gate = {"feature_set": "v5", "rms_floor": ref["rms_floor"]}   # pre-v6 reference
    if ref_gate is not None and ref_gate != gp:
        return check("reward gate", False,
                     f"active gate {gp} != reference {ref_gate} — source edit / feature-set "
                     f"switch? re---accept"), out
    return check("reward version", True, detail), out


def joints_check(ref):
    import rtde_receive
    r = rtde_receive.RTDEReceiveInterface("192.168.1.100")
    try:
        q = [round(float(x), 4) for x in r.getActualQ()]
    finally:
        try:
            r.disconnect()
        except Exception:
            pass
    if not (ref and ref.get("joints")):
        return check("arm configuration", True, f"joints={q} (no reference yet — run --accept)"), {"joints": q}
    dq = [abs(a - b) for a, b in zip(q, ref["joints"])]
    ok = max(dq) < JOINT_TOL_RAD
    return check("arm configuration", ok,
                 f"max |Δjoint|={max(dq):.3f} rad vs reference (tol {JOINT_TOL_RAD}; "
                 f">1 rad = elbow flip — someone re-postured the arm)"), {"joints": q}


def probe_check(ref, nominal_mm, accepting=False):
    """One zero-residual stroke through the REAL RL path; assert landed force + feature-rms bands."""
    from feature_tone_reward import features, FeatureToneReward
    from hardware_baseline import HardwareBaseline
    from train_residual_robot import make_episode_notes, TIMESTEP_SEC
    from audio_recorder import AudioRecorder
    band_nom = (ref or {}).get("probe_nominal_mm")
    if band_nom is not None and not accepting and abs(band_nom - nominal_mm) > 0.1:
        raise RuntimeError(f"reference bands were accepted at nominal {band_nom:.1f}mm but this probe "
                           f"uses {nominal_mm:.1f}mm — comparing across depths is meaningless. "
                           f"Re-run with --nominal-mm {band_nom:g}, or --accept a new reference.")
    notes = make_episode_notes(1, 2.84)
    if input(f">>> PROBE will BOW ONE stroke at nominal {nominal_mm:.1f}mm. HAND ON THE E-STOP. "
             f"[Enter]=go, q=skip: ").strip().lower() == "q":
        raise ProbeSkipped("probe skipped by user")
    from train_residual_robot import set_speed_slider_full
    set_speed_slider_full()          # a leftover 20% pendant slider breaks servoL timing (06-08)
    hw = HardwareBaseline(notes, nominal_depth=nominal_mm / 1000.0, start_frac=0.20)
    rec = AudioRecorder()
    try:
        hw.reset()
        f_land = float(hw.get_full_state().get("contact_force_magnitude", 0.0))
        rec.start(max_seconds=8.0)
        while not hw.is_complete():
            a = hw.get_baseline_action()
            if a is None:
                break
            a = dict(a); a["depth_residual"] = 0.0
            hw.execute_timestep(a)
            time.sleep(TIMESTEP_SEC)
        audio, sr = rec.stop()
    finally:
        hw.close()
    ft = features(audio, sr)
    # depth/speed passed for samantha_cnn_v1 parity with the trainer; v5/v6 ignore them
    score = FeatureToneReward().score(audio, sr, depth_mm=nominal_mm, speed_ms=hw.bow_speed)
    out = {"probe_nominal_mm": nominal_mm, "probe_feat_rms": round(float(ft[0]), 5),
           "probe_flat": round(float(ft[1]), 4), "probe_score": round(float(score), 3),
           "probe_landed_n": round(f_land, 2)}
    band = (ref or {}).get("probe_rms_band")
    fband = (ref or {}).get("probe_force_band")
    if band and not accepting:
        ok = band[0] <= ft[0] <= band[1]
        if fband is not None:
            ok = ok and (fband[0] <= f_land <= fband[1])
        else:
            print("     (no force band in the reference — force not checked; re---accept to add one)")
        return check("probe stroke", ok,
                     f"feat_rms={ft[0]:.4f} (band {band}), landed={f_land:.2f}N (band {fband}), "
                     f"score={score:.3f} — outside band = level/contact drifted, RECALIBRATE"), out
    return check("probe stroke", True,
                 f"feat_rms={ft[0]:.4f} landed={f_land:.2f}N score={score:.3f} (no band yet — --accept)"), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="also bow ONE zero-residual stroke (robot must be ON+REMOTE)")
    ap.add_argument("--nominal-mm", type=float, default=2.0,
                    help="probe stroke nominal depth (default 2.0 = reward plateau; probing at the "
                         "level CLIFF (0) makes the accepted band meaninglessly volatile)")
    ap.add_argument("--accept", action="store_true", help="snapshot current state as the new reference")
    ap.add_argument("--force-accept", action="store_true", help="accept even with FAILed checks")
    args = ap.parse_args()

    try:
        ref = json.load(open(REF_PATH)) if os.path.exists(REF_PATH) else {}
    except Exception as e:
        print(f"  !! reference_state.json unreadable ({e}) — treating as no reference; "
              f"fix or re---accept")
        ref = {}
    print(f"PREFLIGHT  (reference: {REF_PATH if ref else 'NONE — first run, use --accept'})")
    if args.force_accept and not args.accept:
        print("  (note: --force-accept without --accept does nothing — did you mean both?)")
    results, snapshot = [], {"accepted": time.strftime("%Y-%m-%d %H:%M:%S")}

    for name, fn in (("mic device", mic_check), ("reward version", lambda: reward_check(ref, accepting=args.accept))):
        try:
            ok, info = fn(); results.append(ok); snapshot.update(info)
        except Exception as e:
            results.append(check(name, False, f"crashed: {e}"))
    try:
        ok, info = joints_check(ref); results.append(ok); snapshot.update(info)
    except Exception as e:
        results.append(check("arm configuration", False, f"robot unreachable: {e}"))
    if args.probe or args.accept:
        try:
            ok, info = probe_check(ref, args.nominal_mm, accepting=args.accept); results.append(ok); snapshot.update(info)
            if args.accept and "probe_feat_rms" in snapshot:
                r = snapshot["probe_feat_rms"]
                snapshot["probe_rms_band"] = [round(r * 0.5, 5), round(r * 2.0, 5)]
                fl = snapshot.get("probe_landed_n")
                if fl is not None:
                    snapshot["probe_force_band"] = [round(max(0.05, fl * 0.4), 2), round(fl * 2.5, 2)]
        except ProbeSkipped:
            print("  [SKIP] probe stroke: skipped by user — probe bands NOT updated "
                  "(--accept may proceed; the reference keeps no/old bands)")
        except Exception as e:
            results.append(check("probe stroke", False, f"{e}"))

    if args.accept:
        if not all(results) and not args.force_accept:
            print("REFUSING to --accept a FAILING state (a dead mic / wrong config would become "
                  "the reference). Fix the FAILs or pass --force-accept deliberately.")
            return 1
        if snapshot.get("joints") is None:
            print("  !! accepting WITHOUT a joints reference (robot unreachable) — the arm-"
                  "configuration check will be a no-op until you re---accept with the robot on.")
        os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
        with open(REF_PATH, "w") as fh:
            json.dump(snapshot, fh, indent=1)
        print(f"  reference ACCEPTED -> {REF_PATH}")
        return 0

    if all(results):
        print("PREFLIGHT PASS — session may start.")
        return 0
    print("PREFLIGHT FAIL — fix the failed checks (or deliberately --accept a new reference).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
