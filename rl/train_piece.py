"""
rl/train_piece.py

Train SAC to shade a whole piece for tone quality.

Mirrors other_github/rl/sac/train.py (same algorithm, same hyperparameter
philosophy) but over PieceResidualEnv: episode = one piece, step = one
stroke, action = (speed, depth) residual, reward = sound-classifier tone
score + dynamic accuracy + bow-budget discipline.

Usage:
    # hardware-free end-to-end check (mock executor + mock scorer)
    python rl/train_piece.py MIDI-Files/twinkle_twinkle-open.mid \
        --mock --timesteps 30000

    # mock robot but the REAL classifier as scorer (needs a mic-free score
    # path: classifier falls back to heuristic scoring of silence — only
    # useful for wiring checks, not learning)
    python rl/train_piece.py piece.mxl --mock --real-scorer --timesteps 1000

    # the real thing: robot + mic + your trained classifier checkpoint
    python rl/train_piece.py piece.mxl --real --timesteps 4000

    # resume a hardware run (model + replay buffer)
    python rl/train_piece.py piece.mxl --real --resume rl/checkpoints_piece/sac_piece_latest

Hardware runs are precious: the replay buffer is saved with every checkpoint
so a run can always be resumed without losing a single real stroke.

Run from the repository environment created by scripts/setup.sh or
scripts/setup.ps1.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.piece_env import (
    PieceResidualEnv, MockExecutor, MockScorer, RealScorer,
)


def build_env(args) -> PieceResidualEnv:
    if args.real:
        from rl.piece_hardware import HardwareExecutor
        executor = HardwareExecutor(
            audio_device=args.audio_device,
            audio_channel=args.audio_channel,
            save_dir=(Path(args.save_dir) / "stroke_audio"
                      if args.save_stroke_audio else None),
        )
        scorer = RealScorer(checkpoint_path=args.classifier_checkpoint)
    else:
        executor = MockExecutor()
        scorer = (RealScorer(checkpoint_path=args.classifier_checkpoint)
                  if args.real_scorer else MockScorer())

    return PieceResidualEnv(
        piece_path=args.piece,
        executor=executor,
        scorer=scorer,
        tempo_scale=args.tempo_scale,
        bowing_rule=args.bowing,
        calibrated_dynamics=args.calibrated_dynamics,
    )


class EpisodeQualityLogger:
    """
    Prints per-episode mean tone quality — the number that matters — and
    watches for a DEGENERATE reward.

    A classifier that returns the same score for every stroke gives the policy
    nothing to climb: training will appear to run fine while learning nothing.
    That is easy to miss and expensive to discover on hardware, so the per-
    stroke quality spread is checked once enough strokes have been seen.
    """

    FLAT_THRESHOLD = 1e-3      # std below this over a full episode is flat

    def __init__(self):
        self.history: list[float] = []
        self.dynamics: list[float] = []
        self._warned_flat = False

    def __call__(self, locals_, globals_):   # SB3 callback signature
        infos = locals_.get("infos", [])
        for info in infos:
            if "mean_quality" not in info:
                continue
            self.history.append(info["mean_quality"])
            n = len(self.history)
            recent = float(np.mean(self.history[-10:]))
            extra = ""
            if "mean_dynamic" in info:
                self.dynamics.append(info["mean_dynamic"])
                extra = (f"   dyn {info['mean_dynamic']:.3f} "
                         f"(last-10 {float(np.mean(self.dynamics[-10:])):.3f}, "
                         f"in-zone {info.get('in_zone', '?')})")
            print(f"[episode {n:4d}] mean tone quality "
                  f"{info['mean_quality']:.3f}   (last-10 avg {recent:.3f}){extra}")

            if not self._warned_flat and "episode_log" in info:
                scores = [s["quality"] for s in info["episode_log"]]
                spread = float(np.std(scores))
                if spread < self.FLAT_THRESHOLD:
                    self._warned_flat = True
                    print(f"##  WARNING: classifier returned an essentially "
                          f"CONSTANT score across all {len(scores)} strokes "
                          f"(std {spread:.2e}, value ~{np.mean(scores):.3f}). "
                          f"There is no tone gradient to learn from — the "
                          f"policy can only chase the dynamic/bow terms. "
                          f"Check the audio actually reaches the classifier "
                          f"and that the checkpoint discriminates on it.")
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("piece", help=".mid / .mxl / .musicxml to train on")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", default=True,
                      help="mock executor + mock scorer (default)")
    mode.add_argument("--real", action="store_true",
                      help="real robot + mic + real classifier")
    parser.add_argument("--real-scorer", action="store_true",
                        help="use the real classifier even with the mock executor")
    parser.add_argument("--classifier-checkpoint", default=None,
                        help="path to a SoundClassifier checkpoint "
                             "(default: SoundClassifier's own default)")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--tempo-scale", type=float, default=1.0,
                        help=">1 slows the piece down (more bow per note)")
    parser.add_argument("--bowing", default=None,
                        help="bowing rule when the score has none, "
                             "e.g. rule-of-downbow")
    parser.add_argument("--calibrated-dynamics", action="store_true", default=True,
                        help="aim each dynamic at its measured loudness zone by "
                             "inverting the fitted loudness model (default on)")
    parser.add_argument("--no-calibrated-dynamics", action="store_false",
                        dest="calibrated_dynamics",
                        help="use the planner's open-loop volume->speed rule instead")
    parser.add_argument("--resume", default=None,
                        help="checkpoint path (without .zip) to resume from")
    parser.add_argument("--save-dir", default=str(REPO_ROOT / "rl" / "checkpoints_piece"))
    parser.add_argument("--save-every", type=int, default=None,
                        help="steps between checkpoints "
                             "(default: 5000 mock, 250 real)")
    parser.add_argument("--audio-device", default=None)
    parser.add_argument("--audio-channel", type=int, default=0,
                        help="1-based input to record; 0 (default) probes the "
                             "device and picks the channel carrying signal")
    parser.add_argument("--save-stroke-audio", action="store_true",
                        help="keep every stroke's classifier window as a wav "
                             "(builds a dataset for retraining the classifier)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_every = args.save_every or (250 if args.real else 5000)

    env = Monitor(build_env(args))
    n_strokes = len(env.unwrapped.plan)
    print(f"Piece: {args.piece}  ({n_strokes} strokes/episode, "
          f"{'REAL ROBOT' if args.real else 'mock'})")

    try:
        import tensorboard  # noqa: F401
        tb_log = str(save_dir / "tb_logs")
    except ImportError:
        tb_log = None
        print("(tensorboard not installed — skipping tb logging; "
              "pip install tensorboard to enable)")

    if args.resume:
        model = SAC.load(args.resume, env=env)
        buffer_path = Path(str(args.resume) + "_buffer.pkl")
        if buffer_path.exists():
            model.load_replay_buffer(str(buffer_path))
            print(f"Resumed model + replay buffer "
                  f"({model.replay_buffer.size()} transitions) from {args.resume}")
        else:
            print(f"Resumed model from {args.resume} (no replay buffer found)")
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=100_000,
            # Real strokes are expensive — start learning almost immediately
            # and lean on the residual structure for safety.
            learning_starts=200 if args.real else 1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            # Squeeze more gradient work out of each real stroke.
            gradient_steps=4 if args.real else 1,
            ent_coef="auto",
            seed=args.seed,
            verbose=1,
            tensorboard_log=tb_log,
        )

    quality_logger = EpisodeQualityLogger()

    stem = Path(args.piece).stem
    latest = save_dir / f"sac_piece_{stem}_latest"

    steps_done = 0
    t0 = time.time()
    try:
        while steps_done < args.timesteps:
            chunk = min(save_every, args.timesteps - steps_done)
            model.learn(total_timesteps=chunk,
                        callback=quality_logger,
                        reset_num_timesteps=False)
            steps_done += chunk
            model.save(str(latest))
            model.save_replay_buffer(str(latest) + "_buffer.pkl")
            print(f"[checkpoint] {steps_done}/{args.timesteps} steps "
                  f"({time.time()-t0:.0f}s) -> {latest}.zip")
    except KeyboardInterrupt:
        print("\nInterrupted — saving before exit...")
        model.save(str(latest))
        model.save_replay_buffer(str(latest) + "_buffer.pkl")
        print(f"Saved {latest}.zip (+ replay buffer). "
              f"Resume with --resume {latest}")
    finally:
        env.close()

    final = save_dir / f"sac_piece_{stem}_final"
    model.save(str(final))
    print(f"Saved final model -> {final}.zip")
    if quality_logger.history:
        h = quality_logger.history
        print(f"Tone quality: first-10 avg {np.mean(h[:10]):.3f}  "
              f"->  last-10 avg {np.mean(h[-10:]):.3f}  "
              f"over {len(h)} episodes")


if __name__ == "__main__":
    main()
