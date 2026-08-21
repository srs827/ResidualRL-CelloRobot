"""
rl/play_piece.py

Roll a trained residual policy through a piece and report what it did —
per stroke and against the zero-residual baseline.

Usage:
    # mock: compare policy vs baseline over several episodes
    python rl/play_piece.py MIDI-Files/twinkle_twinkle-open.mid \
        --model rl/checkpoints_piece/sac_piece_twinkle_twinkle-open_final \
        --episodes 5

    # baseline only (no model): what does zero residual score?
    python rl/play_piece.py MIDI-Files/twinkle_twinkle-open.mid --episodes 5

    # perform on the robot with the learned shading
    python rl/play_piece.py piece.mxl --model ... --real --episodes 1

Run from the repository environment created by scripts/setup.sh or
scripts/setup.ps1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.piece_env import PieceResidualEnv, MockExecutor, MockScorer, RealScorer


def rollout(env: PieceResidualEnv, model=None, deterministic=True):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    log = None
    while not done:
        if model is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    log = info["episode_log"]
    return total_reward, info["mean_quality"], info.get("mean_dynamic"), log


def print_episode(log):
    print(f"{'#':>3} {'dir':>4} {'u range':>13} {'speed':>6} {'depth':>7} "
          f"{'a_spd':>6} {'a_dep':>6} {'qual':>5} {'dyn':>5} {'errdB':>6}")
    for s in log:
        print(f"{s['note_index']:>3} {s['direction']:>4} "
              f"{s['u_start']:.3f}->{s['u_end']:.3f} "
              f"{s['mean_speed']:6.3f} {s['depth_mm']:+6.2f}mm "
              f"{s['action'][0]:+6.2f} {s['action'][1]:+6.2f} "
              f"{s['quality']:5.2f} {s['r_dynamic']:5.2f} {s['err_db']:6.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("piece")
    parser.add_argument("--model", default=None,
                        help="SAC checkpoint (without .zip); omit for baseline")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--real-scorer", action="store_true")
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--tempo-scale", type=float, default=1.0)
    parser.add_argument("--bowing", default=None)
    parser.add_argument("--calibrated-dynamics", action="store_true", default=True,
                        help="aim each dynamic at its measured loudness zone (default on)")
    parser.add_argument("--no-calibrated-dynamics", action="store_false",
                        dest="calibrated_dynamics")
    parser.add_argument("--verbose", action="store_true",
                        help="print the per-stroke table for every episode")
    args = parser.parse_args(argv)

    if args.real:
        from rl.piece_hardware import HardwareExecutor
        executor = HardwareExecutor()
        scorer = RealScorer(checkpoint_path=args.classifier_checkpoint)
    else:
        executor = MockExecutor()
        scorer = (RealScorer(checkpoint_path=args.classifier_checkpoint)
                  if args.real_scorer else MockScorer())

    env = PieceResidualEnv(args.piece, executor=executor, scorer=scorer,
                           tempo_scale=args.tempo_scale, bowing_rule=args.bowing,
                           calibrated_dynamics=args.calibrated_dynamics)

    model = None
    if args.model:
        from stable_baselines3 import SAC
        model = SAC.load(args.model)

    label = f"policy {args.model}" if model else "baseline (zero residual)"
    print(f"Rolling out {label} on {args.piece} "
          f"({len(env.plan)} strokes/episode)")

    rewards, qualities, tones, dynamics, in_zone = [], [], [], [], []
    try:
        for ep in range(args.episodes):
            total, mean_q, mean_d, log = rollout(env, model)
            rewards.append(total)
            qualities.append(mean_q)
            # mean_q is the OVERALL head; the reward optimises the "tone"
            # head mix (same mislabel train_piece.py carried until 08-21).
            # Print both, labelled honestly.
            mean_tone = float(np.mean([s.get("tone", s["quality"])
                                       for s in log]))
            tones.append(mean_tone)
            zoned = sum(1 for s in log if s["r_dynamic"] >= 0.999)
            in_zone.append(zoned / len(log))
            extra = ""
            if mean_d is not None:
                dynamics.append(mean_d)
                extra = f"   dynamics {mean_d:.3f}  in-zone {zoned}/{len(log)}"
            print(f"episode {ep+1}: return {total:8.2f}   "
                  f"tone {mean_tone:.3f} (overall {mean_q:.3f}){extra}")
            if args.verbose or args.episodes == 1:
                print_episode(log)
    finally:
        env.close()

    print(f"\n{label}")
    print(f"  return        {np.mean(rewards):8.2f} ± {np.std(rewards):.2f}")
    print(f"  tone (reward) {np.mean(tones):8.3f} ± {np.std(tones):.3f}")
    print(f"  overall head  {np.mean(qualities):8.3f} ± {np.std(qualities):.3f}")
    if dynamics:
        print(f"  dynamics      {np.mean(dynamics):8.3f} ± {np.std(dynamics):.3f}")
        print(f"  in-zone       {100*np.mean(in_zone):7.1f}%")


if __name__ == "__main__":
    main()
