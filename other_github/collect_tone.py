"""
collect_tone.py — collect new-domain bow strokes spanning tone quality (surface -> clean ->
crushed) to train the interim tone classifier (the depth-RL reward), AFTER the cello move +
bow swap + 66deg normal change made the old probe OOD (2026-06-22).

WHY a new collector (not recording_grid/pre_experiment_grid): those bake in OLD-geometry
assumptions (up-clearance, verify_ft_at_poses, frog reset, depth grid centred on a not-2mm-in
line) that misfire on the new normal (e.g. 10N false free-air readings). This one is built on
the VERIFIED path (TargetStringBaselineController.execute_timestep servoL + recording_grid
geometry + the new RETRACT_DIR_BASE) and replicates hardware_baseline's safety.

KEY: depth is referenced to TRUE LIGHT CONTACT, not the taught line. The taught line sits
~2.2mm INTO the string (probe_contact: first touch at d≈-2 to -2.5mm), so depth=0 here = just
touching (surface end); depth grows = press in (firm/crushed end). One uniform BACKOFF, no
global recalibration — the RL's line is untouched until we set its nominal from the found band.

EXTENSIBLE: each CONDITION is {d_mm, speed, label}. Depth-only now (speed fixed = RL speed).
Adding speed/angle/beta later = more fields per condition + more rows; no rewrite.

Safety (per the verified RL path): setTcp(bow tip) via the controller; gentle reset = clear
(off-string) -> tare -> force-guarded descent to the start depth; per-tick |F|>5N -> servoStop +
retract; reach-clamped every command. HAND ON THE E-STOP.

Pass A (find the new good band): coarse sweep, LISTEN.
  .venv-rtde/bin/python collect_tone.py --depths 0,1,2,3,4 --reps 1 --out tone_passA
Pass B (fuller, around the band): finer sweep x reps -> train.
  .venv-rtde/bin/python collect_tone.py --depths 0,0.5,1,1.5,2,2.5,3,3.5,4 --reps 4 --out tone_newdomain
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
sys.path.insert(0, "Baseline-Runners")
sys.path.insert(0, "rl/residual")
import recording_grid as R                                  # noqa: E402
from baseline_controller import TargetStringBaselineController  # noqa: E402
from audio_recorder import AudioRecorder, find_focusrite_device  # noqa: E402

BACKOFF = 0.0             # depth 0 = the CAPTURED line (goto_frog showed it ~2.3N, NOT light). Sweep
                          # NEGATIVE depth = lift (tool -Z) toward surface; POSITIVE = press (tool +Z) toward crush.
E_PRESS = -R.RETRACT_DIR_BASE   # = bow TOOL +Z in base = press INTO the string (Zixian's LOCAL tool-jog 2026-06-27)
SPAN = float(np.linalg.norm(R.TIP_TARGET_STR[:3] - R.FROG_TARGET_STR[:3]))
FRAC_START, FRAC_END = 0.20, 0.80      # 2x longer stroke (Zixian 2026-06-27: 0.30-0.60 too short).
                                       # span 0.60 of the line ~294mm; hover reach max 0.791 (<0.83)
TICK = 0.05
FORCE_ABORT = 5.5         # lowered 7->5.5 (Zixian 2026-06-27): after the geometry fix the tare is clean
                          # (d=0 reads ~0.6N, no offset) so |F| is now ~accurate -> 5.5N reading ≈ real,
                          # under the bow's 3-6N structural limit. Watch the bow; e-stop if it bends hard.
CLEAR = 0.030             # m off the string ALONG THE NORMAL (= perpendicular to the string axis) for the
                          # hover/tare pose. The normal is mostly +Y (~horizontal) here — NOT world-up.
TARE_MAX_N = 1.5
STEP = 0.0005             # slow force-guarded descent step (near contact only)
SLOW_BAND = 0.003         # m above the line where the descent switches FAST(moveL)->SLOW(force-guarded)


def pose(frac, d_from_light):
    """TCP pose at bow fraction `frac`, pressed `d_from_light` (m) past TRUE LIGHT contact."""
    p = R.bow_pose_full(float(frac)).copy()
    p[:3] += (float(d_from_light) - BACKOFF) * E_PRESS
    return R.clamp_reach(p)


def fmag(ctrl):
    return float(np.linalg.norm(np.asarray(ctrl.rtde_r.getActualTCPForce())[:3]))


def safe(ctrl):
    try:
        return (not ctrl.rtde_r.isProtectiveStopped()) and int(ctrl.rtde_r.getSafetyMode()) == 1
    except Exception:
        return False


def retract(ctrl, lift=0.040):
    """Lift OFF along the +normal (= perpendicular to the string axis -> clean separation, no
    along-string drag), then travel in the air to the FRAC_START hover. The normal is mostly +Y
    (~horizontal) here, NOT world-up — that's what 'perpendicular to this cello's string' looks
    like (string axis is 47deg off vertical). Zixian caught -Z dragging along the string 2026-06-22."""
    try:
        ctrl.rtde_c.servoStop(); time.sleep(0.1)
    except Exception:
        pass
    try:
        cur = np.asarray(ctrl.rtde_r.getActualTCPPose(), float)
        off = cur.copy(); off[:3] += lift * np.asarray(R.RETRACT_DIR_BASE, float)   # lift along the normal (perp to string)
        ctrl.rtde_c.moveL(R.clamp_reach(off).tolist(), 0.06, 0.3)
        ctrl.rtde_c.moveL(pose(FRAC_START, -CLEAR).tolist(), 0.10, 0.4)             # travel in the AIR to the hover
    except Exception:
        pass


def reset_to(ctrl, d_start):
    """retract (perp-to-string lift-off + air-travel to the hover) -> tare -> descend ALONG THE
    NORMAL to depth d_start. The normal has ~0 component along the string axis, so the descent is
    perpendicular to the string and does NOT drag along it. It moves mostly -Y (~horizontal) — that
    is correct for this cello angle, not a bug. Zixian 2026-06-22."""
    retract(ctrl)
    if not safe(ctrl):
        raise RuntimeError("not safe after retract.")
    ctrl.rtde_c.zeroFtSensor(); time.sleep(0.3)
    f0 = fmag(ctrl)
    if f0 > TARE_MAX_N:
        raise RuntimeError(f"free-air |F|={f0:.2f}N > {TARE_MAX_N} after tare — not clear / payload off.")
    # FAST moveL through FREE AIR down to SLOW_BAND above the line (Zixian 2026-06-27: the full slow
    # descent wasted time; free air needs no force-guard). If d_start is higher/lifted, go straight there.
    fast_to = min(d_start, -SLOW_BAND)
    if not ctrl.rtde_c.moveL(pose(FRAC_START, fast_to).tolist(), 0.10, 0.4) or not safe(ctrl):
        raise RuntimeError("fast free-air approach failed / not safe.")
    f = fmag(ctrl)
    if f > FORCE_ABORT:
        retract(ctrl); raise RuntimeError(f"FORCE abort after fast approach |F|={f:.1f}N (contact higher than expected?)")
    # SLOW force-guarded ONLY for the last ~SLOW_BAND into contact
    d = fast_to
    while d <= d_start + 1e-9:
        ctrl.execute_timestep(pose(FRAC_START, d), ctrl.BASELINE_FORCE_A); time.sleep(0.12)
        f = fmag(ctrl)
        if f > FORCE_ABORT or not safe(ctrl):
            retract(ctrl); raise RuntimeError(f"FORCE/safety abort during descent |F|={f:.1f}N at d={d*1000:+.1f}mm")
        d += STEP
    return fmag(ctrl)


def bow_and_record(ctrl, rec, d, speed, max_rec):
    """Record audio while bowing FRAC_START->FRAC_END at fixed depth d. Returns (audio, sr, |F| samples)."""
    frac_per_tick = speed * TICK / SPAN
    rec.start(max_seconds=max_rec)
    forces, frac = [], FRAC_START
    while frac <= FRAC_END + 1e-9:
        ctrl.execute_timestep(pose(frac, d), ctrl.BASELINE_FORCE_A)
        f = fmag(ctrl); forces.append(f)
        if f > FORCE_ABORT or not safe(ctrl):
            audio, sr = rec.stop(); retract(ctrl)
            raise RuntimeError(f"FORCE/safety abort during stroke |F|={f:.1f}N at frac {frac:.2f}")
        frac += frac_per_tick
        time.sleep(TICK)
    audio, sr = rec.stop()
    return audio, sr, forces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="0,1,2,3,4", help="comma list, mm past TRUE LIGHT contact")
    ap.add_argument("--speed", type=float, default=0.11, help="bow speed m/s (RL ~0.109)")
    ap.add_argument("--speeds", default="", help="comma list of bow speeds m/s -> depth x speed GRID "
                                                 "(overrides --speed; e.g. 0.07,0.09,0.11,0.13,0.15)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default="tone_passA", help="output dir (kept SEPARATE from Samantha's dataset)")
    ap.add_argument("--max-rec", type=float, default=4.0)
    ap.add_argument("--label", action="store_true",
                    help="LIVE ear-labeling: after each stroke (bow lifted off first) prompt for a "
                         "1-4 score, written incrementally to <out>/ear_labels.json — the exact "
                         "file train_interim_reward.py --labels reads. In-room judgment beats "
                         "headphone playback (Zixian 2026-07-20).")
    args = ap.parse_args()
    depths_mm = [float(x) for x in args.depths.split(",")]
    speeds = [float(x) for x in args.speeds.split(",")] if args.speeds else [args.speed]
    conds = [{"d": dm / 1000.0, "speed": s, "label": f"D{dm:.1f}mm_S{s:.2f}"}
             for s in speeds for dm in depths_mm]
    assert len({c["label"] for c in conds}) == len(conds), \
        "condition labels collide (speeds/depths that round to the same label) — wavs would overwrite"

    outdir = os.path.abspath(args.out)
    os.makedirs(os.path.join(outdir, "audio"), exist_ok=True)
    meta_path = os.path.join(outdir, "metadata.jsonl")
    print(f"  output -> {outdir}  (depths {depths_mm} mm past true-light x speeds {speeds} m/s "
          f"x {args.reps} reps = {len(conds) * args.reps} strokes, frac {FRAC_START}-{FRAC_END})")
    print(f"  reach sanity: " + ", ".join(
        f"d{dm:.0f}:{np.linalg.norm(pose(FRAC_START, dm/1000)[:3]):.3f}/{np.linalg.norm(pose(FRAC_END, dm/1000)[:3]):.3f}"
        for dm in depths_mm) + "  (start/end reach, all must be < %.2f)" % R.MAX_SAFE_REACH)

    dev = find_focusrite_device()
    rec = AudioRecorder(device=dev)
    if not rec.is_focusrite:
        if input("  !! NOT the Focusrite. type 'yes' to use anyway: ").strip() != "yes":
            return
    ctrl = TargetStringBaselineController([])
    from train_residual_robot import set_speed_slider_full   # noqa: E402  (rl/residual on sys.path above)
    set_speed_slider_full()      # a leftover 20% pendant slider slows every servoL stroke -> the
                                 # collected tone domain silently differs from the RL's (2026-07-12 review)
    sid = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = 0
    n_skip = 0
    try:
        try:
            ctrl.rtde_c.setPayload(float(R.FT_PAYLOAD_KG), list(R.FT_PAYLOAD_COG))
        except Exception:
            pass
        input(f"\n>>> {len(conds)*args.reps} strokes. HAND ON THE E-STOP. ENTER to start...")
        order = [(c, r) for r in range(args.reps) for c in conds]
        for c, r in order:
            print(f"\n--- {c['label']} rep{r}: reset (force-guarded descent)...")
            try:                                  # one stroke's abort must NOT kill the session
                f_land = reset_to(ctrl, c["d"])
                print(f"    landed |F|={f_land:.2f}N; bowing + recording...")
                stroke_s = (FRAC_END - FRAC_START) * SPAN / c["speed"]     # slow strokes outlast --max-rec
                audio, sr, forces = bow_and_record(ctrl, rec, c["d"], c["speed"],
                                                   max(args.max_rec, stroke_s + 2.0))
            except RuntimeError as e:
                print(f"    SKIP {c['label']} rep{r} (force/safety abort, continuing): {e}")
                with open(meta_path, "a") as fh:   # machine-readable grid hole for train_2d_reward
                    fh.write(json.dumps({"label": c["label"], "rep": r, "skipped": str(e)}) + "\n")
                n_skip += 1
                continue
            fid = f"{sid}_{c['label']}_r{r:02d}.wav"
            sf.write(os.path.join(outdir, "audio", fid), audio, sr)
            rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))
            meta = {"audio_file": fid, "d_from_light_mm": c["d"] * 1000, "speed": c["speed"],
                    "frac_start": FRAC_START, "frac_end": FRAC_END, "rep": r,
                    "landed_F": round(f_land, 3), "stroke_F_mean": round(float(np.mean(forces)), 3),
                    "stroke_F_max": round(float(np.max(forces)), 3), "rms": round(rms, 5), "label": None}
            with open(meta_path, "a") as fh:
                fh.write(json.dumps(meta) + "\n")
            print(f"    saved {fid}  |F| mean {np.mean(forces):.2f} max {np.max(forces):.2f}N  rms {rms:.4f}")
            if args.label:
                retract(ctrl)          # lift OFF before pausing — no static pressed contact while typing
                lab_path = os.path.join(outdir, "ear_labels.json")
                labs = json.load(open(lab_path)) if os.path.exists(lab_path) else {}
                while True:
                    s = input("    >>> EAR SCORE 1-4 (Enter=skip): ").strip()
                    if s == "":
                        labs[fid] = None
                        break
                    try:
                        v = float(s)
                    except ValueError:
                        print("        1-4 or Enter."); continue
                    if 1.0 <= v <= 4.0:
                        labs[fid] = v
                        break
                    print("        1-4 or Enter.")
                with open(lab_path, "w") as fh:        # incremental, crash-safe
                    json.dump(labs, fh, indent=1)
            n += 1
            # next reset_to() lift-retracts before the next stroke; finally retracts after the last
    finally:
        retract(ctrl)
        try:
            ctrl.close()
        except Exception:
            pass
        print(f"\n  done: {n} strokes saved to {outdir} ({n_skip} skipped on abort). Listen, then label + train.")


if __name__ == "__main__":
    main()
