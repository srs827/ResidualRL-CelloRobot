"""
rl/ab_compare.py

Play the same piece twice on the robot, back to back, so the two can be
compared by ear: FIRST the pre-training baseline (the planner's nominal —
zero residual, what the robot played before any RL), THEN the trained policy.

Both passes go through rl/perform.py --compile, which precomputes the whole
rollout and then plays it as a few blended moveL paths. That is deliberate:
compiled mode holds tempo_ratio ~1.01 where stroke-by-stroke live playback
runs ~1.16, so the two passes differ by the POLICY rather than by dispatch
jitter. It also writes a compiled_full.wav per pass, so the comparison can be
re-listened to (and shown to someone else) without occupying the robot again.

The two numbers worth reading afterwards come from each pass's
compiled_report.json:

    tempo_ratio            wall time / written time — should be ~1.0 on both;
                           if they differ the comparison is not clean
    mean_quality_posthoc   the real classifier re-scoring the RECORDED audio,
                           which is the same judge the policy trained against

Usage:
    ~/venvs/cello311/bin/python rl/ab_compare.py RUN_DIR --piece PIECE
    ... --checkpoint PATH      override the auto-picked best checkpoint
    ... --gap 8                seconds of silence between passes (default 6)
    ... --order after-first    play the trained policy first
    ... --repeat 2             play the A/B pair more than once

Prefix with `caffeinate -dims` — Mac sleep has been traced to RTDE drops.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PERFORM = REPO_ROOT / "rl" / "perform.py"
OUT_ROOT = REPO_ROOT / "rl" / "checkpoints_piece"


def resolve_checkpoint(run_dir: Path, override: str | None) -> Path:
    """The winner select_best copied aside, unless told otherwise.

    Deliberately does NOT fall back to sac_piece_*_final.zip: the writeup's
    Saturday protocol is explicit that `final` is not the checkpoint to
    perform with (the last gradient step is not the best policy), and
    silently substituting it would make this script answer a different
    question than the one asked.
    """
    if override:
        p = Path(override)
        if not p.exists():
            sys.exit(f"--checkpoint {p} does not exist")
        return p
    best = run_dir / "sac_piece_best.zip"
    if not best.exists():
        sys.exit(f"{best} not found — run rl/select_best.py on {run_dir.name} "
                 f"first, or pass --checkpoint explicitly")
    return best


def preflight() -> None:
    """Fail before the first note rather than mid-performance."""
    try:
        import rtde_receive
    except ImportError:
        sys.exit("ur_rtde not importable — use ~/venvs/cello311/bin/python")
    modes = {0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING",
             3: "POWER_OFF", 4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE",
             7: "RUNNING", 8: "UPDATING_FIRMWARE"}
    safes = {1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY",
             5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP",
             7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT"}
    try:
        r = rtde_receive.RTDEReceiveInterface("192.168.1.100")
    except Exception as e:
        sys.exit(f"cannot reach the robot at 192.168.1.100 ({type(e).__name__}). "
                 f"Check the controller is powered and on the network.")
    rm, sm = r.getRobotMode(), r.getSafetyMode()
    print(f"robot_mode {rm} ({modes.get(rm, '?')})   "
          f"safety_mode {sm} ({safes.get(sm, '?')})")
    if (rm, sm) != (7, 1):
        sys.exit("robot is not RUNNING/NORMAL — release the E-stop, then "
                 "power on and brake-release from the pendant.")


def newest_perform_dir(after: float) -> Path | None:
    cands = [d for d in OUT_ROOT.glob("perform_*")
             if d.is_dir() and d.stat().st_mtime >= after]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def play(label: str, piece: str, checkpoint: Path | None,
         calibrated: bool, tempo_scale: float) -> dict:
    """One compiled pass. Returns {label, out_dir, wav, report}."""
    cmd = [sys.executable, str(PERFORM), "--real", "--compile",
           "--tempo-scale", str(tempo_scale)]
    if checkpoint is None:
        cmd.append("--baseline")
    else:
        cmd.append(str(checkpoint))
    cmd.append(piece)
    if not calibrated:
        cmd.append("--no-calibrated-dynamics")

    print(f"\n{'=' * 62}\n  {label}\n{'=' * 62}")
    print("  " + " ".join(cmd[1:]))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True,
                          capture_output=True)
    # perform.py prints: "compiled N strokes -> <out_dir>"
    m = re.search(r"compiled\s+\d+\s+strokes\s+->\s+(\S+)", proc.stdout or "")
    out_dir = Path(m.group(1)) if m else newest_perform_dir(t0)

    for line in (proc.stdout or "").splitlines():
        if not re.search(r"UserWarning|warnings\.warn", line):
            print("  " + line)
    if proc.returncode != 0:
        print(f"  !! perform.py exited {proc.returncode}")
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        for line in tail:
            print("  " + line)

    rep = {}
    if out_dir and (out_dir / "compiled_report.json").exists():
        rep = json.loads((out_dir / "compiled_report.json").read_text())
        rep = rep.get("report", rep)
    # The wav's name depends on the render path (render_compiled writes
    # compiled_full.wav, render_baseline lets PiecePlayer name it), so
    # discover it instead of hardcoding one dialect — the hardcoded name is
    # what printed "(no wav written)" for six takes that had recorded fine.
    wav = None
    if out_dir and out_dir.is_dir():
        wavs = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        wav = wavs[-1] if wavs else None
    return {"label": label, "ok": proc.returncode == 0, "out_dir": out_dir,
            "wav": wav, "report": rep}


def countdown(sec: int, nxt: str) -> None:
    if sec <= 0:
        return
    print(f"\n  --- {sec}s until: {nxt} ---")
    for s in range(sec, 0, -1):
        print(f"\r  {s:2d} ", end="", flush=True)
        time.sleep(1)
    print("\r      ")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Back-to-back baseline vs trained-policy playback")
    ap.add_argument("run_dir", help="training run dir containing "
                                    "sac_piece_best.zip (from select_best)")
    ap.add_argument("--piece", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="override the auto-picked best checkpoint")
    ap.add_argument("--gap", type=int, default=6,
                    help="seconds between passes (default 6)")
    ap.add_argument("--order", choices=("before-first", "after-first"),
                    default="before-first")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tempo-scale", type=float, default=1.0,
                    help="stretch note durations. Short notes cannot reach a "
                         "speed that makes the string speak (SPEED_MIN=0.09 "
                         "m/s); on yunpiece 70%% of strokes are under it at "
                         "1.0 and 3%% at 1.5. Applied to BOTH passes so the "
                         "comparison stays fair.")
    ap.add_argument("--calibrated-dynamics", action="store_true", default=True)
    ap.add_argument("--no-calibrated-dynamics", dest="calibrated_dynamics",
                    action="store_false",
                    help="must match how the policy was TRAINED")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"{run_dir} is not a directory")
    if not Path(args.piece).exists():
        sys.exit(f"piece not found: {args.piece}")
    ckpt = resolve_checkpoint(run_dir, args.checkpoint)

    print(f"piece      {args.piece}")
    print(f"before     baseline (zero residual — the planner's nominal)")
    print(f"after      {ckpt}")
    print(f"tempo      {args.tempo_scale}x")
    if (run_dir / "best.json").exists():
        b = json.loads((run_dir / "best.json").read_text())
        print(f"selected   {json.dumps(b.get('best', b))[:160]}")
    preflight()

    sides = [("BEFORE  (pre-training baseline)", None),
             ("AFTER   (trained policy)", ckpt)]
    if args.order == "after-first":
        sides.reverse()

    results = []
    for rep_i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n########## pass {rep_i + 1} of {args.repeat} ##########")
        for i, (label, ck) in enumerate(sides):
            results.append(play(label, args.piece, ck,
                                args.calibrated_dynamics,
                                args.tempo_scale))
            is_last = (rep_i == args.repeat - 1) and (i == len(sides) - 1)
            if not is_last:
                countdown(args.gap, sides[(i + 1) % len(sides)][0])

    # Fill the post-hoc column for baseline-render passes: rl/posthoc_score
    # re-scores the recorded take with the env-faithful judge inputs
    # (measured speed/torque from the state log, tone head mix). Commanded
    # phys said "no difference" on takes where measured phys said +0.062 at
    # p=4e-6 — see that file's docstring before "simplifying" this away.
    try:
        from rl.posthoc_score import score_dir
        for r in results:
            rp = r["report"] or {}
            if (rp.get("mode") == "baseline-render" and r["out_dir"]
                    and "mean_quality_posthoc" not in rp):
                s = score_dir(r["out_dir"])
                if s:
                    rp["mean_quality_posthoc"] = round(s["mean_tone"], 3)
                    rp["posthoc_overall"] = round(s["mean_overall"], 3)
                    r["report"] = rp
    except Exception as e:
        print(f"(post-hoc scoring unavailable: {type(e).__name__}: {e})")

    print(f"\n{'=' * 62}\n  SUMMARY\n{'=' * 62}")
    hdr = f"{'pass':<34}{'tempo':>7}{'drift ms':>10}{'post-hoc q':>12}"
    print(hdr)
    for r in results:
        rp = r["report"] or {}
        # Speak both report dialects: render_compiled writes tempo_ratio and
        # mean_quality_posthoc; render_baseline writes wall_s/written_s and
        # per-note drift stats instead (and no post-hoc score yet).
        tr = rp.get("tempo_ratio")
        if tr is None and rp.get("wall_s") and rp.get("written_s"):
            tr = rp["wall_s"] / max(rp["written_s"], 1e-6)
        dm = rp.get("drift_mean_ms")
        q = rp.get("mean_quality_posthoc")
        print(f"{r['label']:<34}"
              f"{(f'{tr:.3f}' if tr is not None else '—'):>7}"
              f"{(f'{dm:+.1f}' if dm is not None else '—'):>10}"
              f"{(f'{q:.3f}' if q is not None else '—'):>12}"
              + ("" if r["ok"] else "   [FAILED]"))
    if any((r["report"] or {}).get("mode") == "baseline-render"
           and "mean_quality_posthoc" not in (r["report"] or {})
           for r in results):
        print("\n(post-hoc scoring failed for a pass above — run "
              "rl/posthoc_score.py on its perform dir by hand)")
    print("\nrecordings:")
    for r in results:
        print(f"  {r['label']:<34}{r['wav'] or '(no wav written)'}")
    qs = [(r["label"], (r["report"] or {}).get("mean_quality_posthoc"))
          for r in results]
    qs = [(l, q) for l, q in qs if q is not None]
    if len(qs) >= 2:
        print("\nNote: post-hoc q is the judge's TONE head mix at measured "
              "phys (rl/posthoc_score.py),\nthe same scale as the stroke log "
              "and select_best. One number per take — treat a small\ngap as "
              "noise; for per-note paired stats run posthoc_score.py with "
              "both dirs.")


if __name__ == "__main__":
    main()
