"""
rl/driver_eval.py

Evaluation harness for the full driver (rl/driver_piece.py, obs 18->23).

Runs a trained checkpoint deterministically at fixed offsets on either
channel and under observation modes that form the attribution instrument:

    live       all five driver dims as measured
    masked     all five zeroed          (any sensing load-bearing?)
    masklevel  loudness dims zeroed     (isolate the loudness channel)
    maskmech   torque dims zeroed       (isolate the depth channel)
    sham       loudness err dims replaced by noise
    spoof      err dims clamped to --spoof-err-db, dBFS dim zeroed,
               TRUE offsets 0           (causal test: action follows obs?)

Sweeps: --offsets (gain, dB, depth held 0) or --depth-offsets (mm, gain 0).
All modes evaluate the SAME frozen checkpoint — only what it sees varies.

Usage (mock):
    .venv/bin/python rl/driver_eval.py CKPT.zip MIDI-Files/t1.mid --mock \
        --offsets=-1.5,-1,0,1,1.5 --episodes 20 --mode live
    .venv/bin/python rl/driver_eval.py CKPT.zip MIDI-Files/t1.mid --mock \
        --depth-offsets=-0.5,-0.25,0,0.25,0.5 --episodes 20 --mode live

Zixian Liu, 2026-08-10.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.driver_piece import (DriverPieceEnv, LOUD_DIMS, MECH_DIMS,
                             ERR_NORM_DB)
from rl.piece_env import OBS_DIM, MockExecutor, MockScorer

LO = OBS_DIM
MID = OBS_DIM + LOUD_DIMS
HI = OBS_DIM + LOUD_DIMS + MECH_DIMS


def split_action(a):
    """(speed_residual, depth_residual) means, for either action layout.

    v0.1.0: action = [speed, depth].  Envelope build (2026-08-12): action =
    [spd x3, depth x3] — report the segment means so the D2 slopes keep the
    same meaning ("how hard does the policy push each channel").
    """
    a = np.asarray(a, dtype=float).ravel()
    if a.size >= 4:
        h = a.size // 2
        return float(a[:h].mean()), float(a[h:].mean())
    return float(a[0]), float(a[1])


def transform_obs(obs, mode, rng, spoof_err_db):
    obs = obs.copy()
    if mode == "masked":
        obs[LO:HI] = 0.0
    elif mode == "masklevel":
        obs[LO:MID] = 0.0
    elif mode == "maskmech":
        obs[MID:HI] = 0.0
    elif mode == "sham":
        obs[LO] = rng.uniform(-1.5, 1.5) / ERR_NORM_DB
        obs[LO + 1] = rng.uniform(-1.5, 1.5) / ERR_NORM_DB
    elif mode == "spoof":
        obs[LO] = spoof_err_db / ERR_NORM_DB
        obs[LO + 1] = spoof_err_db / ERR_NORM_DB
        obs[LO + 2] = 0.0
    return obs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("checkpoint")
    ap.add_argument("piece")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--real", action="store_true",
                    help="explicit real-robot mode (the default when --mock "
                         "is absent; this flag just makes it visible)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--offsets", default=None,
                    help="comma list of GAIN offsets in dB (depth held 0)")
    ap.add_argument("--depth-offsets", default=None,
                    help="comma list of DEPTH offsets in mm (gain held 0)")
    ap.add_argument("--mode", default="live",
                    choices=["live", "masked", "masklevel", "maskmech",
                             "sham", "spoof"])
    ap.add_argument("--spoof-err-db", type=float, default=1.5)
    ap.add_argument("--calibrated-dynamics", action="store_true", default=False,
                    help="MUST match the training run (driver protocol: OFF)")
    ap.add_argument("--dbfs-calib-db", type=float, default=0.0,
                    help="sensor calibration constant; MUST match training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="jsonl dump path")
    args = ap.parse_args()

    from stable_baselines3 import SAC

    rng = np.random.default_rng(args.seed)
    if args.mode == "spoof":
        cells = [(0.0, 0.0)]
        sweep = "gain"
    elif args.depth_offsets:
        cells = [(0.0, float(x)) for x in args.depth_offsets.split(",")]
        sweep = "depth"
    else:
        offs = args.offsets or "0"
        cells = [(float(x), 0.0) for x in offs.split(",")]
        sweep = "gain"
    print(f"eval config: calibrated_dynamics={args.calibrated_dynamics} "
          f"(must match training), mode={args.mode}, sweep={sweep}, "
          f"seed={args.seed}")

    if not args.mock and len(cells) != 1:
        sys.exit("real-robot eval: one (offset, mode) cell per invocation — "
                 "the hardware connection is not rebuilt between cells")

    out = open(args.out, "a", buffering=1) if args.out else None
    rows = []
    for gain_off, depth_off in cells:
        if args.mock:
            executor = MockExecutor(rng=np.random.default_rng(args.seed + 1))
            scorer = MockScorer(rng=np.random.default_rng(args.seed + 2))
        else:
            from rl.piece_hardware import HardwareExecutor
            from rl.piece_env import RealScorer
            executor = HardwareExecutor()
            scorer = RealScorer()
        env = DriverPieceEnv(
            piece_path=args.piece,
            executor=executor,
            scorer=scorer,
            gain_fixed_db=gain_off,
            depth_fixed_mm=depth_off,
            dbfs_calib_db=args.dbfs_calib_db,
            calibrated_dynamics=args.calibrated_dynamics,
        )
        # MockExecutor emits MIC-frame dBFS natively (mic_offset_db), so mock
        # grading already matches real grading. The hot-mic patch that used to
        # live here would now add the json gain_offset_db a second time.
        model_off = getattr(env, "model_gain_offset_db", 0.0)
        if args.mock and model_off:
            print(f"mock simulates the hot mic: +{model_off:.2f} dB on "
                  f"generated dBFS")

        model = SAC.load(args.checkpoint, env=None, device="cpu")
        if model.observation_space.shape != env.observation_space.shape:
            sys.exit(f"checkpoint obs {model.observation_space.shape} != env "
                     f"{env.observation_space.shape} — wrong lineage")
        if model.action_space.shape != env.action_space.shape:
            # without this, a 6-dim checkpoint on the v0.1.0 worktree would
            # SILENTLY apply segment-1 speed as the depth residual
            sys.exit(f"checkpoint action {model.action_space.shape} != env "
                     f"{env.action_space.shape} — wrong lineage")

        errs, inzone, tone, a0_tail, a1_tail = [], [], [], [], []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + 100 * ep)
            done = False
            k = 0
            while not done:
                pobs = transform_obs(obs, args.mode, rng, args.spoof_err_db)
                action, _ = model.predict(pobs, deterministic=True)
                obs, r, done, _, info = env.step(action)
                s = info["stroke"]
                k += 1
                if s.get("zone") is not None:
                    # model-frame error when the json gain offset is active;
                    # raw err_db is in the mic frame and reads ~offset too hot
                    errs.append(abs(float(s.get("err_db_model", s["err_db"]))))
                    inzone.append(1.0 if s["r_dynamic"] >= 0.999 else 0.0)
                tone.append(float(s["quality"]))
                if k >= 2:
                    a_spd, a_dep = split_action(s["action"])
                    a0_tail.append(a_spd)
                    a1_tail.append(a_dep)
                if out:
                    out.write(json.dumps({
                        "gain_off": gain_off, "depth_off": depth_off,
                        "mode": args.mode, "episode": ep, "stroke": k,
                        **{kk: s[kk] for kk in
                           ("err_db", "err_db_model", "r_dynamic", "quality",
                            "action", "zone", "dbfs", "depth_mm")
                           if kk in s}}) + "\n")
        env.close()
        row = {"gain_off": gain_off, "depth_off": depth_off,
               "mode": args.mode,
               "in_zone": float(np.mean(inzone)) if inzone else None,
               "mean_abs_err_db": float(np.mean(errs)) if errs else None,
               "a0": float(np.mean(a0_tail)) if a0_tail else None,
               "a1": float(np.mean(a1_tail)) if a1_tail else None,
               "mean_tone": float(np.mean(tone)) if tone else None}
        rows.append(row)
        print(f"gain {gain_off:+.2f} dB  depth {depth_off:+.2f} mm  "
              f"mode {args.mode:9s}  in-zone {row['in_zone']:.2f}  "
              f"|err| {row['mean_abs_err_db']:.2f} dB  "
              f"a0(spd) {row['a0']:+.3f}  a1(dep) {row['a1']:+.3f}  "
              f"tone {row['mean_tone']:.3f}")

    if len(rows) > 1:
        if sweep == "gain":
            xs = [r["gain_off"] for r in rows]
            ys = [r["a0"] for r in rows]
            slope = float(np.polyfit(xs, ys, 1)[0])
            print(f"\nD2-level: d(a0)/d(gain) = {slope:+.3f} per dB "
                  f"(live: clearly negative; masked/masklevel: ~0)")
        else:
            xs = [r["depth_off"] for r in rows]
            ys = [r["a1"] for r in rows]
            slope = float(np.polyfit(xs, ys, 1)[0])
            print(f"\nD2-mech: d(a1)/d(depth) = {slope:+.3f} per mm "
                  f"(live: clearly negative; masked/maskmech: ~0)")


if __name__ == "__main__":
    main()
