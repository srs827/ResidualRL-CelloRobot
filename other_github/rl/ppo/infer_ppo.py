"""Inference for custom PPO — mirrors rl/sac/infer.py.

Usage:
    python -m rl.ppo.infer_ppo --model checkpoints/ppo_cello_A_final.pt --note A --steps 150
"""

import argparse
from pathlib import Path

import numpy as np

from rl.ppo.ppo import PPOAgent
from rl.sac.cello_env import CelloEnv, MockSoundClassifier


def play_note_ppo(agent: PPOAgent, env: CelloEnv, max_steps: int = 150, verbose: bool = True):
    obs, _ = env.reset()
    forces, rewards = [], []
    info = {}

    for i in range(max_steps):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        forces.append(info["force"])
        rewards.append(reward)

        if verbose:
            print(
                f"  step {i:03d} | "
                f"force={info['force']:.3f} N | "
                f"{info['label']} {info['confidence']:.0%} | "
                f"reward={info['reward']:+.3f}"
            )

        if terminated or truncated:
            break

    return {
        "mean_force": float(np.mean(forces)) if forces else 0.0,
        "std_force": float(np.std(forces)) if forces else 0.0,
        "total_reward": float(np.sum(rewards)),
        "final_info": info,
    }


def main(args):
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")

    classifier = MockSoundClassifier()
    env = CelloEnv(classifier=classifier, max_steps=args.steps)

    agent = PPOAgent(env=env, device="cpu")
    agent.load(model_path)
    print(f"Loaded model: {model_path}")

    print(f"\nPlaying note {args.note} for up to {args.steps} steps ...\n")
    summary = play_note_ppo(agent, env, max_steps=args.steps, verbose=args.verbose)

    print(
        f"\nDone. mean_force={summary['mean_force']:.3f} N | "
        f"std_force={summary['std_force']:.3f} N | "
        f"total_reward={summary['total_reward']:+.3f}"
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/ppo_cello_A_final.pt")
    parser.add_argument("--note", type=str, default="A", choices=["A", "D", "G", "C"])
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--verbose", action="store_true", default=True)
    main(parser.parse_args())
