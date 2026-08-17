"""
rl/reward_noise.py

Measure how repeatable the REWARD is, and compare that noise floor to the
effect size the action space can actually produce. This is the experiment that
decides whether a training run can succeed before you spend two hours on it.

The logic: SAC has to tell "action A is better than action B" apart from
measurement noise. If replaying the SAME action on the SAME stroke returns
rewards that vary by more than the spread between DIFFERENT actions, then the
gradient is noise and no amount of training, tuning or architecture work will
help. Three runs on yunpiece produced policies worse than the zero-residual
baseline; this tells you whether that was predictable.

Two numbers come out:

    sigma_repeat   sd of reward for one fixed action replayed N times
                   -- pure measurement noise (bow, mic, classifier, model)
    sigma_action   sd of reward across the action space, same stroke
                   -- the signal the policy is trying to climb

    SNR = sigma_action / sigma_repeat

A SNR near or below 1 means a stroke's reward says more about the measurement
than about the choice. It also reports per-TERM breakdowns, because the
headline can hide that (say) r_dynamic is pure noise while r_defect is clean --
which would tell you to reweight rather than abandon.

Usage:
    ~/venvs/cello311/bin/python rl/reward_noise.py --piece PIECE --real
    ... --repeats 8         replays per action (default 6)
    ... --probes 6          random actions sampled (default 6)
    ... --strokes 3         distinct strokes to test (default 3)
    ... --mock              plumbing check, no robot

Real cost is repeats*(probes+1)*strokes strokes; the default is ~126 strokes,
about a minute of playing plus overhead. Prefix with `caffeinate -dims`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TERMS = ("total", "quality", "r_dynamic", "r_defect", "r_onset",
         "r_speak", "r_envelope", "r_bow")


def build_env(args):
    from rl.piece_env import (PieceResidualEnv, MockExecutor, MockScorer,
                              RealScorer)
    if args.real:
        from rl.piece_hardware import HardwareExecutor
        executor, scorer = HardwareExecutor(), RealScorer()
    else:
        executor, scorer = MockExecutor(), MockScorer()
    return PieceResidualEnv(piece_path=args.piece, executor=executor,
                            scorer=scorer, tempo_scale=args.tempo_scale,
                            calibrated_dynamics=args.calibrated_dynamics)


def play_stroke_at(env, target_i: int, action: np.ndarray) -> dict:  # noqa: C901
    """Reset and step to `target_i`, playing `action` on that stroke.

    Everything before the target is played at zero residual, so the arm and
    the bow arrive in the same state each time -- the point is to vary ONE
    stroke's action, not the history that led to it.
    """
    try:
        obs, _ = env.reset(seed=0)
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        for i in range(target_i):
            obs, _, term, trunc, _ = env.step(zero)
            if term or trunc:
                print(f"      episode ended early at stroke {i} "
                      f"(wanted {target_i})", flush=True)
                return {}
        obs, reward, term, trunc, info = env.step(np.asarray(action, np.float32))
    except Exception as e:
        # A trial that dies must not take the whole measurement with it -- an
        # hour of hardware time is too expensive to lose to one bad stroke.
        print(f"      TRIAL FAILED: {type(e).__name__}: {e}", flush=True)
        return {}
    row = dict(info["stroke"])
    row["reward"] = float(reward)
    return row


def summarise(rows: list[dict]) -> dict:
    out = {}
    for t in TERMS:
        v = np.array([r[t] for r in rows if t in r], dtype=float)
        if len(v):
            out[t] = (float(v.mean()), float(v.std()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="reward repeatability vs action effect")
    ap.add_argument("--piece", required=True)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--mock", dest="real", action="store_false")
    ap.set_defaults(real=True)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--probes", type=int, default=6)
    ap.add_argument("--strokes", type=int, default=3)
    ap.add_argument("--tempo-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrated-dynamics", action="store_true", default=True)
    ap.add_argument("--no-calibrated-dynamics", dest="calibrated_dynamics",
                    action="store_false")
    ap.add_argument("--targets", default=None,
                    help="comma-separated stroke indices to probe. Default "
                         "spreads them 20-80%% through the piece, but each "
                         "trial REPLAYS from reset to reach the target, so a "
                         "late index costs a whole piece of robot time per "
                         "trial. Early indices measure the same thing far "
                         "cheaper.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = build_env(args)
    n_plan = len(env.plan)
    dim = env.action_space.shape[0]

    # Spread the probe strokes through the piece rather than clustering at the
    # start: bow position and remaining budget change what a residual can do.
    if args.targets:
        targets = [int(x) for x in args.targets.split(",")]
    else:
        targets = [int(round(f * (n_plan - 1)))
                   for f in np.linspace(0.2, 0.8, args.strokes)]
    print(f"piece {args.piece}  strokes {n_plan}  probing at {targets}")
    print(f"{'':4}repeats={args.repeats}  probes={args.probes}  "
          f"action_dim={dim}  mode={'REAL' if args.real else 'mock'}")

    report = {"piece": args.piece, "real": args.real, "targets": targets,
              "repeats": args.repeats, "probes": args.probes,
              "tempo_scale": args.tempo_scale,
              "t": datetime.now().isoformat(timespec="seconds"), "strokes": {}}

    for ti in targets:
        print(f"\n{'=' * 58}\n  stroke {ti}\n{'=' * 58}")
        # (a) one fixed action, replayed -> measurement noise
        fixed = np.zeros(dim, dtype=np.float32)
        rep_rows = []
        for k in range(args.repeats):
            print(f"    repeat {k + 1}/{args.repeats}", flush=True)
            r = play_stroke_at(env, ti, fixed)
            if r:
                rep_rows.append(r)
        # (b) different actions, once each -> effect of choosing
        probe_rows = []
        for k in range(args.probes):
            print(f"    probe {k + 1}/{args.probes}", flush=True)
            a = rng.uniform(-1, 1, size=dim).astype(np.float32)
            r = play_stroke_at(env, ti, a)
            if r:
                probe_rows.append(r)

        rep, prb = summarise(rep_rows), summarise(probe_rows)
        entry = {"repeat": rep, "probe": prb, "n_repeat": len(rep_rows),
                 "n_probe": len(probe_rows)}
        report["strokes"][str(ti)] = entry
        # Checkpoint after every probe stroke: partial data beats none.
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2))

        print(f"  {'term':<12}{'repeat sd':>11}{'action sd':>11}{'SNR':>8}   verdict")
        for t in TERMS:
            if t not in rep or t not in prb:
                continue
            sr, sa = rep[t][1], prb[t][1]
            snr = sa / sr if sr > 1e-9 else float("inf")
            verdict = ("NOISE — action invisible" if snr < 1.0 else
                       "weak" if snr < 2.0 else "usable")
            print(f"  {t:<12}{sr:>11.4f}{sa:>11.4f}{snr:>8.2f}   {verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")

    # Headline across strokes, on total reward.
    snrs = []
    for e in report["strokes"].values():
        if "total" in e["repeat"] and "total" in e["probe"]:
            sr = e["repeat"]["total"][1]
            if sr > 1e-9:
                snrs.append(e["probe"]["total"][1] / sr)
    if snrs:
        m = float(np.mean(snrs))
        print(f"\n{'=' * 58}")
        print(f"  mean SNR on total reward: {m:.2f}")
        if m < 1.0:
            print("  The action space moves the reward LESS than replay noise.")
            print("  Training cannot separate a good action from a bad one.")
            print("  Fix the measurement or widen the action space before "
                  "training again.")
        elif m < 2.0:
            print("  Marginal. Expect slow, unreliable learning and runs that")
            print("  sometimes end worse than baseline.")
        else:
            print("  The action space is visible above the noise; training is")
            print("  worth the hardware time.")
    env.close()


if __name__ == "__main__":
    main()
