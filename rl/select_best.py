"""
rl/select_best.py

Best-model retention, part 2: evaluate a run's numbered checkpoints
deterministically and keep the best one.

Why: the training curve is exploration-subsidised — on challengepiece the
train-time average was 0.468 while the deterministic mean action scored
0.015, and "keep the last checkpoint" shipped the 0.015. Numbered
checkpoints (rl/train_piece_logged.py --ckpt-every, default 10) plus this
sweep replace that with "keep the one that is actually best when played
straight".

Checkpoint lineage is auto-detected from the saved observation space:
18 dims -> stock PieceResidualEnv, 23 dims -> DriverPieceEnv (live obs,
zero curriculum offsets). All checkpoints in one sweep must share a lineage.

Cost on hardware: n_ckpts x episodes piece playthroughs. Default sweeps the
LAST 5 numbered checkpoints plus final (the interesting region — late
training); widen with --all.

Usage:
    .venv/bin/python rl/select_best.py RUN_DIR --piece MIDI-Files/t1.mid --mock
    .venv/bin/python rl/select_best.py RUN_DIR --piece MIDI-Files/t1.mid \
        --real --episodes 2            # one hardware connection, all ckpts

Writes RUN_DIR/best.json (per-checkpoint table) and copies the winner to
RUN_DIR/sac_piece_best.zip.

Zixian Liu, 2026-08-13.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_env(obs_dim, piece, mock, seed, calibrated_dynamics,
               blind_dyn_obs=False, blind_mech_obs=False):
    from rl.piece_env import PieceResidualEnv, OBS_DIM, MockExecutor, MockScorer
    if mock:
        executor = MockExecutor(rng=np.random.default_rng(seed + 1))
        scorer = MockScorer(rng=np.random.default_rng(seed + 2))
    else:
        from rl.piece_hardware import HardwareExecutor
        from rl.piece_env import RealScorer
        executor = HardwareExecutor()
        scorer = RealScorer()
    if obs_dim == OBS_DIM:
        env = PieceResidualEnv(piece_path=piece, executor=executor,
                               scorer=scorer,
                               calibrated_dynamics=calibrated_dynamics)
    else:
        from rl.driver_piece import DriverPieceEnv, DRIVER_EXTRA_DIMS
        if obs_dim != OBS_DIM + DRIVER_EXTRA_DIMS:
            sys.exit(f"checkpoint obs dim {obs_dim} matches neither stock "
                     f"({OBS_DIM}) nor driver ({OBS_DIM + DRIVER_EXTRA_DIMS})")
        env = DriverPieceEnv(piece_path=piece, executor=executor,
                             scorer=scorer, gain_fixed_db=0.0,
                             depth_fixed_mm=0.0,
                             blind_dyn_obs=blind_dyn_obs,
                             blind_mech_obs=blind_mech_obs,
                             calibrated_dynamics=calibrated_dynamics)
    # No mock hot-mic patch here: MockExecutor emits MIC-frame dBFS natively
    # (mic_offset_db), so adding gain_offset_db again would double-correct.
    return env


def main():
    ap = argparse.ArgumentParser(description="pick the best checkpoint")
    ap.add_argument("run_dir")
    ap.add_argument("--piece", required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--real", action="store_true",
                    help="explicit real-robot mode (the default when --mock "
                         "is absent; this flag just makes it visible)")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--last", type=int, default=5,
                    help="sweep the last N numbered ckpts (+final)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confirm-episodes", type=int, default=2,
                    help="extra episodes for the top-2 (fresh seeds) before "
                         "the final pick — winner's-curse insurance")
    ap.add_argument("--calibrated-dynamics", action="store_true", default=False,
                    help="MUST match how the run was trained (our protocol: "
                         "off, i.e. trained with --no-calibrated-dynamics)")
    ap.add_argument("--blind-dyn-obs", action="store_true", default=False,
                    help="MUST match training for blind-ablation checkpoints")
    ap.add_argument("--blind-mech-obs", action="store_true", default=False)
    args = ap.parse_args()

    from stable_baselines3 import SAC

    run_dir = Path(args.run_dir)
    numbered = sorted(run_dir.glob("ckpt_ep*.zip"))
    if not args.all:
        numbered = numbered[-args.last:]
    finals = sorted(run_dir.glob("sac_piece_*_final.zip"))
    ckpts = numbered + finals
    if not ckpts:
        sys.exit(f"no checkpoints in {run_dir} (need ckpt_ep*.zip — was the "
                 "run trained with --ckpt-every?)")

    first = SAC.load(ckpts[0], env=None, device="cpu")
    obs_dim = first.observation_space.shape[0]
    action_shape = first.action_space.shape
    del first

    env = None

    def build():
        e = _build_env(obs_dim, args.piece, args.mock, args.seed,
                       args.calibrated_dynamics,
                       args.blind_dyn_obs, args.blind_mech_obs)
        if e.action_space.shape != action_shape:
            sys.exit(f"checkpoint action space {action_shape} != env "
                     f"{e.action_space.shape} — wrong lineage (2-dim v0.1.0 "
                     "checkpoint against the envelope env, or vice versa)")
        return e

    def run_eval(model, episodes, seed_base):
        nonlocal env
        # mock: fresh env per eval -> identical noise sequence, fair ranking;
        # real: one env/connection for the whole sweep, hardware is hardware
        if env is None or args.mock:
            if env is not None:
                env.close()
            env = build()
        tones, rets = [], []
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed_base + 100 * ep)
            done, ret = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, _, info = env.step(action)
                ret += float(r)
                # "tone" = the length-aware head mix, NOT the overall head:
                # overall is documented to rank nearly opposite to the ear
                # on challengepiece (RL_METHOD / piece_env TONE_HEAD_WEIGHTS)
                s = info["stroke"]
                tones.append(float(s.get("tone", s["quality"])))
            rets.append(ret)
        return tones, rets

    rows = []
    models = {}
    for ck in ckpts:
        model = SAC.load(ck, env=None, device="cpu")
        if model.observation_space.shape[0] != obs_dim:
            sys.exit(f"{ck.name}: obs dim {model.observation_space.shape[0]} "
                     f"!= sweep's {obs_dim} — mixed lineages in one run dir")
        models[ck.name] = model
        tones, rets = run_eval(model, args.episodes, args.seed)
        rows.append({"ckpt": ck.name, "episodes": args.episodes,
                     "mean_tone": float(np.mean(tones)),
                     "mean_return": float(np.mean(rets))})
        print(f"{ck.name:24s} return {rows[-1]['mean_return']:7.2f}   "
              f"tone {rows[-1]['mean_tone']:.3f}")

    # Selection is by deterministic mean RETURN — the reward is the declared
    # objective; selecting on any single term (tone) can legally tank the
    # others. Tone is reported alongside for the human. Top-2 get a
    # confirmation pass on fresh seeds so one lucky pair of episodes cannot
    # crown a checkpoint (winner's curse).
    rows.sort(key=lambda r: -r["mean_return"])
    if args.confirm_episodes > 0 and len(rows) > 1:
        for r in rows[:2]:
            tones, rets = run_eval(models[r["ckpt"]], args.confirm_episodes,
                                   args.seed + 1000)
            n0, n1 = r["episodes"], args.confirm_episodes
            r["mean_return"] = (r["mean_return"] * n0 + float(np.sum(rets))) \
                / (n0 + n1)
            r["mean_tone"] = float((r["mean_tone"] * n0
                                    + np.mean(tones) * n1) / (n0 + n1))
            r["episodes"] = n0 + n1
            print(f"confirm {r['ckpt']:16s} return {r['mean_return']:7.2f}   "
                  f"tone {r['mean_tone']:.3f}   (n={r['episodes']})")
        rows.sort(key=lambda r: -r["mean_return"])
    if env is not None:
        env.close()

    best = rows[0]
    out = run_dir / "sac_piece_best.zip"
    shutil.copyfile(run_dir / best["ckpt"], out)
    (run_dir / "best.json").write_text(json.dumps({
        "criterion": "mean_return, deterministic, top-2 confirmed",
        "episodes_per_ckpt": args.episodes,
        "confirm_episodes": args.confirm_episodes, "mock": args.mock,
        "calibrated_dynamics": args.calibrated_dynamics,
        "obs_dim": obs_dim, "rows": rows, "best": best}, indent=2))
    print(f"\nbest: {best['ckpt']} (return {best['mean_return']:.2f}, "
          f"tone {best['mean_tone']:.3f}) -> {out.name}   [+ best.json]")


if __name__ == "__main__":
    main()
