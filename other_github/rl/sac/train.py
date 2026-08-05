"""
train.py

Trains a SAC policy on the mock CelloEnv.
Uses MockSoundClassifier as a stand-in for the real classifier.

Usage:
    python train.py
    python train.py --timesteps 500000 --note A
"""

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rl.sac.cello_env import CelloEnv, MockSoundClassifier


def make_env(note: str = "A", max_steps: int = 150) -> CelloEnv:
    classifier = MockSoundClassifier(noise=0.05)
    env = CelloEnv(classifier=classifier, max_steps=max_steps)
    return Monitor(env)


def main(args):
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    train_env = make_env(note=args.note)

    # Quick sanity check on the env before training
    check_env(train_env, warn=True)

    eval_env = make_env(note=args.note)

    model = SAC(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 3e-4,
        buffer_size     = 100_000,
        learning_starts = 500,        # steps of random exploration before updates
        batch_size      = 256,
        tau             = 0.005,      # soft-update coefficient
        gamma           = 0.99,       # discount — high because bow quality persists
        train_freq      = 1,
        gradient_steps  = 1,
        ent_coef        = "auto",     # auto-tune entropy for exploration
        verbose         = 1,
        tensorboard_log = "./tb_logs/",
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path = str(save_dir / "best"),
            log_path             = str(save_dir / "eval_logs"),
            eval_freq            = 2000,
            n_eval_episodes      = 10,
            deterministic        = True,
            verbose              = 1,
        ),
        CheckpointCallback(
            save_freq  = 10_000,
            save_path  = str(save_dir / "periodic"),
            name_prefix= f"sac_cello_{args.note}",
        ),
    ]

    print(f"Training SAC for {args.timesteps} steps on note {args.note} ...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    final_path = save_dir / f"sac_cello_{args.note}_final"
    model.save(str(final_path))
    print(f"Saved final model → {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--note",      type=str, default="A",
                        choices=["A", "D", "G", "C"],
                        help="Open string note being trained")
    main(parser.parse_args())
