"""
infer.py

Inference loop — mirrors the pseudocode from the spec:

    i = 0
    while note has not ended:
        obs          = get_observation()
        force_change = model.predict(obs)
        rtde.set_force(force_change)
        play(); i++

In practice, all of that (get_obs → predict → set_force → classifier)
is encapsulated in env.step(), so the loop stays clean.

Swap MockSoundClassifier for your real classifier and subclass CelloEnv
to wire in real RTDE + audio capture (_get_obs, _get_audio, _apply_force).

Usage:
    python infer.py --model checkpoints/best/best_model --note A --steps 150
"""

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from rl.sac.cello_env import CelloEnv, MockSoundClassifier


def play_note(
    model: SAC,
    env: CelloEnv,
    max_steps: int = 150,
    verbose: bool = True,
) -> dict:
    """
    Play one note stroke using the trained SAC policy.
    Returns the final info dict.
    """
    obs, _ = env.reset()
    i = 0

    while i < max_steps:
        # ── This is the loop from the spec ──────────────────────────────
        # obs already contains: tcp_pose, force_torque, speed, accel,
        #                       bow_position, current_force

        force_change, _ = model.predict(obs, deterministic=True)
        # force_change is a 1-dim array: normalised delta in [-1, 1]
        # env.step internally: scales → clips → applies via _apply_force
        #                      gets new obs → classifies audio → computes reward

        obs, reward, terminated, truncated, info = env.step(force_change)
        # ────────────────────────────────────────────────────────────────

        if verbose:
            print(
                f"  step {i:03d} | "
                f"force={info['force']:.3f} N | "
                f"{info['label']} {info['confidence']:.0%} | "
                f"reward={info['reward']:+.3f}"
            )

        i += 1
        if terminated or truncated:
            break

    return info


def main(args):
    model_path = Path(args.model)
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"No model found at {model_path}.zip")

    model = SAC.load(str(model_path))
    print(f"Loaded model: {model_path}")

    # Use MockSoundClassifier for offline testing.
    # Replace with your real classifier for hardware deployment.
    classifier = MockSoundClassifier()
    env = CelloEnv(classifier=classifier, max_steps=args.steps)

    print(f"\nPlaying note {args.note} for up to {args.steps} steps ...\n")
    info = play_note(model, env, max_steps=args.steps, verbose=args.verbose)

    print(f"\nDone. Final force: {info['force']:.3f} N | "
          f"Last classifier: {info['label']} {info['confidence']:.0%}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="checkpoints/best/best_model")
    parser.add_argument("--note",    type=str, default="A", choices=["A", "D", "G", "C"])
    parser.add_argument("--steps",   type=int, default=150)
    parser.add_argument("--verbose", action="store_true", default=True)
    main(parser.parse_args())
