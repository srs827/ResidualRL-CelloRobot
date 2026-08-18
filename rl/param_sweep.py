"""
rl/param_sweep.py

Blinded, replicated, randomised parameter sweep on the real robot.

Why this exists: the 2026-08-18 ACCEL_MAX sweep compared one take per setting,
played in a fixed order, with the setting announced before listening. All three
of those independently invalidate the result:

  * ONE take per setting means no repeat sd, so a 0.03 period_corr difference
    could be the setting or could be nothing. It was never measured.
  * FIXED order means session drift is collinear with condition. The four takes
    scored 0.844 / 0.890 / 0.916 / 0.922 in the exact order they were played,
    and nobody can say how much of that was the instrument warming up.
  * ANNOUNCED settings mean the listener knows which take is which.

This runs conditions in shuffled order with replication, writes each take under
a blinded id, and refuses to reveal the mapping until you ask for it. So you can
rank the takes by ear first and decode afterwards.

It also separates the two clean single-variable axes, which the ad-hoc sweep
did not. Above ACCEL_MAX 5.0 the planned length and speed are IDENTICAL
(15.31 mm, 0.1380 m/s) and only commanded acceleration moves; tempo moves
length at fixed speed. So:

    acceleration axis   (5.0, 1.0) (5.5, 1.0) (6.0, 1.0)   accel 4.32/4.68/5.03
    length axis         (5.5, 1.0) (5.5, 1.1) (5.5, 1.2)   len 15.31/16.63/17.91

ACCEL_MAX 4.0 is deliberately NOT on either axis: it caps the plan, so it moves
length AND speed AND acceleration together and cannot isolate anything. Include
it with --with-4 only as a reference point, and read it as such.

Usage:
    # record (robot). ~45 s per take plus overhead; 5 conditions x 3 = 15 takes
    ~/venvs/cello311/bin/python rl/param_sweep.py --real --reps 3

    # verify the plumbing first, no robot: checks the env override actually
    # reaches the planner for every condition (a stale-__pycache__ failure
    # made several points of the 8/18 sweep measure the wrong value)
    ~/venvs/cello311/bin/python rl/param_sweep.py --mock

    # after you have ranked the takes by ear
    ~/venvs/cello311/bin/python rl/param_sweep.py --decode rl/sweeps/<dir>

Prefix the real run with `caffeinate -dims`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIECE = ("SoundClassifier/Data_Collection/public_annotation_packages_a_final/"
         "yunpiece.mxl")

# (accel_max, tempo_scale). The union of the two axes above; (5.5, 1.0) is the
# shared corner and therefore gets 2x the replication for free, which is what
# you want on the point both axes are measured against.
# A condition is (accel_max, tempo_scale, speed_residual). speed_residual None
# means the planner's nominal (zero residual); a number in [-1, 1] is played as
# a constant --fixed-action on the speed half of the envelope.
CONDITIONS = [(5.0, 1.00, None), (5.5, 1.00, None), (6.0, 1.00, None),
              (5.5, 1.10, None), (5.5, 1.20, None)]
REFERENCE = (4.0, 1.00, None)

# The SPEED axis. Measured 2026-08-18: the accel/length sweep above planned an
# identical 0.1380 m/s in every one of its five conditions, and a listener
# ranking of 15 blinded takes had NO relationship to condition (permutation
# p = 0.54) even though period_corr separated them at SNR 7.45. So acceleration
# (4.32-5.03) and length (15.31-17.91 mm) are both inaudible here, and the
# audible 4.0-vs-6.0 difference earlier that day -- which moved length, speed
# AND acceleration together -- most likely came from the one variable that
# sweep held fixed. This axis moves ONLY speed, via the same residual the
# policy itself controls, so a listener ranking here is directly the evidence
# the reward has to reproduce.
# +1.0 is omitted: at ACCEL_MAX 5.5 the upward side saturates at +0.5 (both
# give 16.92 mm / 0.1525 m/s), so it would spend robot time on a duplicate.
# -1.0 lands on 9.95 mm / 0.0897 m/s, which is exactly the plan the 8/17 13:42
# take played -- the one a listener called "decent" the next day. So this axis
# brackets that sound, and it puts writeup-aug17's Part 3 result (a listener
# ranked +35% speed FIRST of five takes and -35% LAST) directly under test,
# blinded and replicated this time rather than one take per condition.
CONDITIONS_SPEED = [(5.5, 1.00, r) for r in (-1.0, -0.5, 0.0, 0.5)]

F0_A = 220.5
BANDS = [(0.0, 0.15, "fast"), (0.15, 0.30, "mid"), (0.30, 9.9, "long")]


def plan_signature(accel_max: float, tempo: float,
                   speed_res: float | None = None) -> dict:
    """Commanded motion for a condition, in-process (no robot, no audio)."""
    env = dict(os.environ, CELLO_ACCEL_MAX=str(accel_max),
               PYTHONDONTWRITEBYTECODE="1")
    code = (
        "import sys,json,numpy as np; sys.path.insert(0,'.');"
        "from rl.piece_env import load_piece, PMP;"
        f"n,m,st=load_piece({PIECE!r},calibrated_dynamics=False,tempo_scale={tempo});"
        "L=[abs(s.u_end-s.u_start)*PMP.BOW_LENGTH for s in st];"
        "T=[s.duration for s in st];"
        f"m={1.0 if speed_res is None else 1.0 + speed_res * 0.35};"
        "so=[PMP.solve_stroke(l*m,t) for l,t in zip(L,T)];"
        "print(json.dumps({'accel_max':PMP.ACCEL_MAX,"
        "'len_mm':float(np.median([s.length for s in so])*1000),"
        "'speed':float(np.median([s.mean_speed for s in so])),"
        "'accel':float(np.mean([s.accel for s in so]))}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                         env=env, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"plan_signature failed for {accel_max}/{tempo}:\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def record_take(accel_max: float, tempo: float, speed_res: float | None,
                real: bool) -> Path | None:
    """One take. Returns the perform dir it created, or None in mock."""
    before = set(glob.glob(str(REPO_ROOT / "rl/checkpoints_piece/perform_*")))
    cmd = [sys.executable, "-u", "rl/perform.py",
           "--baseline", "--compile", "--render", "baseline",
           "--tempo-scale", str(tempo), PIECE,
           "--real" if real else "--mock"]
    if speed_res is not None:
        # speed half of the 6-dim envelope only; depth left at nominal so the
        # axis stays single-variable.
        cmd += ["--fixed-action", ",".join([str(speed_res)] * 3 + ["0"] * 3)]
    env = dict(os.environ, CELLO_ACCEL_MAX=str(accel_max))
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, input="\n",
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    TAKE FAILED (rc {r.returncode}): {r.stderr.strip()[-400:]}")
        return None
    after = set(glob.glob(str(REPO_ROOT / "rl/checkpoints_piece/perform_*")))
    new = after - before
    if not new:
        # perform_* dirs are named to the SECOND, so two takes finishing inside
        # one second land in the same directory and the later silently
        # overwrites the earlier. Real takes are ~45 s apart so this should
        # never fire; if it does, the take cannot be attributed to a condition
        # and a blinded sweep must not guess. Drop it instead.
        print("    NO NEW perform dir — take not attributable, dropping it. "
              "(Two takes in one second? Check for a fast-failing take.)")
        return None
    return Path(sorted(new)[-1])


def score_take(perform_dir: Path) -> dict:
    """period_corr by note-length band, from the take's own wav + timeline."""
    import soundfile as sf
    from rl.harmonicity import period_correlation
    wavs = list(perform_dir.glob("*.wav"))
    if not wavs:
        return {}
    base = str(wavs[0])[:-4]
    x, sr = sf.read(base + ".wav")
    if x.ndim > 1:
        x = x[:, 0]
    tl = json.loads(Path(base + "_performance.json").read_text())["timeline"]
    per = {b[2]: [] for b in BANDS}
    allv = []
    for s in tl:
        t0, d = s.get("actual_onset"), s.get("actual_duration")
        if t0 is None or d is None:
            continue
        seg = x[int(t0 * sr):min(int((t0 + d) * sr), len(x))]
        if len(seg) < int(0.009 * sr):
            continue
        v = period_correlation(seg, sr, F0_A)
        if v is None or not np.isfinite(v):
            continue
        allv.append(float(v))
        for lo, hi, nm in BANDS:
            if lo <= d < hi:
                per[nm].append(float(v))
    out = {f"pc_{nm}": (float(np.mean(per[nm])) if len(per[nm]) >= 3 else None)
           for _, _, nm in BANDS}
    out["pc_all"] = float(np.mean(allv)) if allv else None
    out["n_fast"] = len(per["fast"])
    out["rms_dbfs"] = float(20 * np.log10(np.sqrt(np.mean(x ** 2))))
    return out


def cmd_record(args) -> None:
    conds = (list(CONDITIONS_SPEED) if args.axis == "speed"
             else list(CONDITIONS) + ([REFERENCE] if args.with_4 else []))
    print("verifying the env override reaches the planner for each condition")
    sigs = {}
    for c in conds:
        s = plan_signature(*c)
        sigs[str(c)] = s
        ok = abs(s["accel_max"] - c[0]) < 1e-6
        print(f"  accel {c[0]:<4} tempo {c[1]:<5} spd_res {str(c[2]):<5} -> saw "
              f"{s['accel_max']:<4} len {s['len_mm']:6.2f}mm "
              f"speed {s['speed']:.4f} accel {s['accel']:5.3f}  "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit("the override did not take effect — refusing to record a "
                     "sweep whose conditions are not what they claim")
    distinct = {(round(s["len_mm"], 2), round(s["speed"], 4), round(s["accel"], 2))
                for s in sigs.values()}
    print(f"  {len(conds)} conditions -> {len(distinct)} distinct commanded "
          f"motions (identical ones would waste robot time)")

    out = REPO_ROOT / "rl" / "sweeps" / f"sweep_{datetime.now():%Y%m%d_%H%M%S}"
    (out / "takes").mkdir(parents=True, exist_ok=True)
    order = [c for c in conds for _ in range(args.reps)]
    random.Random(args.seed).shuffle(order)
    print(f"\n{len(order)} takes in shuffled order -> {out}")
    print("blinded ids only; run --decode when you have ranked them\n")

    key, results = {}, []
    for i, (am, ts, sr_) in enumerate(order, 1):
        tid = f"take_{i:02d}"
        print(f"[{i}/{len(order)}] {tid}", flush=True)
        d = record_take(am, ts, sr_, args.real)
        key[tid] = {"accel_max": am, "tempo_scale": ts, "speed_residual": sr_,
                    "order": i,
                    "perform_dir": str(d) if d else None}
        row = {"take": tid, "order": i}
        if d is not None and args.real:
            row.update(score_take(d))
            for w in d.glob("*.wav"):
                shutil.copy2(w, out / "takes" / f"{tid}.wav")
        results.append(row)
        # Written every take: a sweep interrupted at take 9 is still usable.
        (out / "key.json").write_text(json.dumps(
            {"key": key, "signatures": sigs, "seed": args.seed}, indent=2))
        (out / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nwrote {out}")
    print(f"  takes/       blinded wavs — listen and rank these")
    print(f"  results.json blinded scores")
    print(f"  key.json     the mapping. DO NOT open it before you have ranked.")
    print(f"\nthen: {sys.executable} rl/param_sweep.py --decode {out}")


def cmd_decode(path: Path) -> None:
    key = json.loads((path / "key.json").read_text())["key"]
    rows = json.loads((path / "results.json").read_text())
    by = {}
    for r in rows:
        k = key.get(r["take"])
        if not k or r.get("pc_fast") is None:
            continue
        by.setdefault((k["accel_max"], k["tempo_scale"],
                       k.get("speed_residual")), []).append(r)
    if not by:
        sys.exit("no scored takes — was this recorded with --real?")

    print(f"{'condition':>16}{'n':>4}{'fast':>17}{'all':>17}{'rms':>8}")
    for c in sorted(by):
        rs = by[c]
        f = np.array([r["pc_fast"] for r in rs])
        a = np.array([r["pc_all"] for r in rs])
        rms = np.mean([r["rms_dbfs"] for r in rs])
        print(f"  a{c[0]:<4} t{c[1]:<4} s{str(c[2]):<5}{len(rs):4d}"
              f"{f.mean():10.3f} ±{f.std(ddof=1) if len(f)>1 else 0:.3f}"
              f"{a.mean():10.3f} ±{a.std(ddof=1) if len(a)>1 else 0:.3f}"
              f"{rms:8.2f}")

    # The number that decides whether any of the above means anything.
    within = [np.array([r["pc_fast"] for r in rs]).std(ddof=1)
              for rs in by.values() if len(rs) > 1]
    spread = np.array([np.mean([r["pc_fast"] for r in rs]) for rs in by.values()])
    print()
    if within:
        rsd = float(np.mean(within))
        print(f"repeat sd (same condition, replays)  {rsd:.4f}")
        print(f"condition spread (sd of means)       {spread.std(ddof=1):.4f}")
        snr = spread.std(ddof=1) / rsd if rsd > 1e-9 else float("inf")
        print(f"SNR = spread / repeat                {snr:.2f}   "
              + ("conditions are indistinguishable from replay noise"
                 if snr < 1.0 else
                 "marginal" if snr < 2.0 else "conditions are real"))
    else:
        print("only one take per condition — no repeat sd, so no comparison "
              "here is interpretable. Re-run with --reps 3.")

    # Drift: randomised order makes this estimable instead of confounded.
    o = np.array([r["order"] for r in rows if r.get("pc_fast") is not None])
    v = np.array([r["pc_fast"] for r in rows if r.get("pc_fast") is not None])
    if len(o) > 3:
        sl = np.polyfit(o, v, 1)[0]
        print(f"\nsession drift  {sl*1000:+.2f} milli-pc per take "
              f"({sl*len(o):+.4f} across the sweep), "
              f"corr(order, score) {np.corrcoef(o, v)[0,1]:+.3f}")
        print("  Order was randomised, so this is an ESTIMATE of warm-up "
              "rather than a confound. Large values mean play the instrument "
              "in before recording.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--real", action="store_true", help="record on the robot")
    ap.add_argument("--mock", dest="real", action="store_false")
    ap.set_defaults(real=False)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--axis", choices=("accel", "speed"), default="accel",
                    help="'accel' sweeps the ceiling and tempo (planned speed "
                         "is identical across all of it); 'speed' sweeps the "
                         "policy's own speed residual at fixed ceiling/tempo")
    ap.add_argument("--with-4", action="store_true",
                    help="add the confounded ACCEL_MAX 4.0 reference point")
    ap.add_argument("--decode", metavar="SWEEP_DIR", default=None)
    args = ap.parse_args()
    if args.decode:
        cmd_decode(Path(args.decode))
    else:
        cmd_record(args)


if __name__ == "__main__":
    main()
