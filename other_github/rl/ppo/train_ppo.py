"""Train custom PPO on CelloEnv with MockSoundClassifier. Pure PyTorch, no sb3.

Note: the mock classifier responds in ~50ms. The real classifier needs ~1s of
audio — that's an env-level latency concern, not handled here.

Usage:
    python -m rl.ppo.train_ppo
    python -m rl.ppo.train_ppo --timesteps 500000 --note A
"""

import argparse
from pathlib import Path

import numpy as np

from rl.ppo.ppo import PPOAgent
from rl.sac.cello_env import CelloEnv, MockSoundClassifier


def make_env(max_steps: int = 150) -> CelloEnv:
    classifier = MockSoundClassifier(noise=0.05)
    return CelloEnv(classifier=classifier, max_steps=max_steps)


def evaluate(agent: PPOAgent, env: CelloEnv, n_episodes: int = 10):
    total_rewards, mean_forces = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        forces = []
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            forces.append(info["force"])
            done = terminated or truncated
        total_rewards.append(ep_reward)
        mean_forces.append(float(np.mean(forces)))
    return float(np.mean(total_rewards)), float(np.mean(mean_forces))


def main(args):
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    train_env = make_env()
    eval_env = make_env()

    agent = PPOAgent(
        env=train_env,
        lr=args.lr,
        n_steps=256,
        n_epochs=4,
        batch_size=64,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        seed=args.seed,
    )

    n_updates = args.timesteps // agent.n_steps
    best_eval_reward = -float("inf")
    print(
        f"Training custom PPO for {args.timesteps} steps "
        f"({n_updates} updates of {agent.n_steps}) on note {args.note} ..."
    )

    for update in range(1, n_updates + 1):
        rollout_info = agent.collect_rollout()
        train_info = agent.learn_from_buffer()

        ep_rews = rollout_info["ep_rewards"]
        mean_ep_reward = float(np.mean(ep_rews)) if ep_rews else float("nan")
        print(
            f"update {update:4d} | steps {update * agent.n_steps:7d} "
            f"| ep_rew {mean_ep_reward:+.3f} "
            f"| pi {train_info['policy_loss']:+.4f} "
            f"| v {train_info['value_loss']:.4f} "
            f"| H {train_info['entropy']:.3f} "
            f"| kl {train_info['approx_kl']:+.4f} "
            f"| clip {train_info['clip_frac']:.2f}"
        )

        if update % args.eval_interval == 0:
            eval_reward, eval_mean_force = evaluate(agent, eval_env, n_episodes=10)
            print(
                f"  eval | reward {eval_reward:+.3f} "
                f"| mean_force {eval_mean_force:.3f} N"
            )
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(save_dir / "best" / f"ppo_cello_{args.note}_best.pt")

    final_path = save_dir / f"ppo_cello_{args.note}_final.pt"
    agent.save(final_path)
    print(f"Saved final model -> {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--note", type=str, default="A", choices=["A", "D", "G", "C"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=20,
        help="Run eval every N updates (each update = n_steps env steps).",
    )
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient (higher = more exploration).")
    parser.add_argument("--lr", type=float, default=3e-4)
    main(parser.parse_args())
