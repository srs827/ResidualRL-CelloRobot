"""
rl/train.py

Residual PPO training loop for single-string cello bowing.

Swap the two import lines at the top when real components are ready.
Everything else stays identical.

Usage:
    python -m rl.train --string A --timesteps 50000
    python -m rl.train --string A --timesteps 200000 --real-robot
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# ── Swap these two lines when real components are ready ───────────
from mocks.mock_baseline   import MockBaselineController as BaselineController
from mocks.mock_classifier import MockSoundClassifier    as SoundClassifier
# from baseline_controller import BaselineController
# from classifier          import BowingQualityClassifier as SoundClassifier
# ─────────────────────────────────────────────────────────────────

from mocks.mock_midi  import make_single_string_sequence
from reward.reward_fn import CompleteCelloReward
from rl.env           import SingleStringResidualEnv
from rl.human_feedback import HumanFeedbackLogger, apply_human_flags_to_buffer
from constants        import OBS_DIM, ACTION_DIM

# PPO from existing codebase
from rl.ppo.ppo           import PPOAgent
from rl.ppo.rollout_buffer import RolloutBuffer

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, BaseCallback
    )
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("stable_baselines3 not installed — using custom PPO from ppo.py")


# ── Reward component logger callback (SB3) ────────────────────────

class RewardComponentLogger(BaseCallback if SB3_AVAILABLE else object):
    """Logs all reward components separately to tensorboard."""

    def __init__(self, verbose=0):
        if SB3_AVAILABLE:
            super().__init__(verbose)
        self._component_buffers = {
            'quality': [], 'trend': [], 'dynamic': [],
            'bow': [], 'timing': [], 'smooth': [], 'human': []
        }

    def _on_step(self):
        if not SB3_AVAILABLE:
            return True
        for info in self.locals.get('infos', []):
            for key in self._component_buffers:
                if key in info:
                    self._component_buffers[key].append(info[key])
        return True

    def _on_rollout_end(self):
        if not SB3_AVAILABLE:
            return
        for key, values in self._component_buffers.items():
            if values:
                self.logger.record(f'reward/{key}', np.mean(values))
        self._component_buffers = {k: [] for k in self._component_buffers}


# ── Main training function ─────────────────────────────────────────

def train(
    string          = 'A',
    total_timesteps = 50_000,
    save_dir        = 'checkpoints',
    use_real_robot  = False,
    midi_path       = None,
    log_interval    = 10,
    seed            = 0,
):
    save_dir = Path(save_dir) / f'string_{string}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Build note sequence ────────────────────────────────────────
    if midi_path is not None:
        # Use real MIDI file
        sys.path.append(str(Path(__file__).parent.parent))
        from robot_runner import parse_midi
        note_sequence = parse_midi(midi_path)
        print(f"Loaded MIDI: {midi_path} ({len(note_sequence)} events)")
    else:
        # Use mock sequence for testing
        note_sequence = make_single_string_sequence(string=string, n_notes=8)
        print(f"Using mock note sequence: {len(note_sequence)} notes on string {string}")

    # ── Initialize components ──────────────────────────────────────
    classifier  = SoundClassifier()
    reward_fn   = CompleteCelloReward(classifier_model=classifier)
    baseline    = BaselineController(note_sequence=note_sequence)
    human_fb    = HumanFeedbackLogger(flag_key='f', log_path=str(save_dir/'flags.jsonl'))

    def make_env():
        env = SingleStringResidualEnv(
            baseline  = BaselineController(note_sequence=note_sequence),
            reward_fn = CompleteCelloReward(classifier_model=SoundClassifier()),
            string    = string,
            mock_mic  = not use_real_robot,
        )
        if SB3_AVAILABLE:
            return Monitor(env)
        return env

    train_env = make_env()
    eval_env  = make_env()

    # ── Validate reward signal before training ─────────────────────
    print("\nValidating reward signal...")
    _validate_reward(baseline, reward_fn, string, note_sequence)

    # ── Train ──────────────────────────────────────────────────────
    if SB3_AVAILABLE:
        _train_sb3(
            train_env, eval_env, string, total_timesteps,
            save_dir, log_interval, seed, human_fb,
        )
    else:
        _train_custom_ppo(
            train_env, string, total_timesteps,
            save_dir, log_interval, seed, human_fb,
        )

    train_env.close()
    eval_env.close()
    human_fb.stop()
    print(f"\nTraining complete. Checkpoints saved to {save_dir}")


def _train_sb3(train_env, eval_env, string, total_timesteps,
               save_dir, log_interval, seed, human_fb):
    """Train using Stable Baselines 3 PPO."""

    model = PPO(
        policy          = 'MlpPolicy',
        env             = train_env,
        learning_rate   = 3e-4,
        n_steps         = 256,
        batch_size      = 64,
        n_epochs        = 4,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = dict(net_arch=[64, 64]),
        verbose         = 1,
        seed            = seed,
        tensorboard_log = str(save_dir / 'tb_logs'),
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path = str(save_dir / 'best'),
            log_path             = str(save_dir / 'eval_logs'),
            eval_freq            = 2000,
            n_eval_episodes      = 5,
            deterministic        = True,
            verbose              = 1,
        ),
        CheckpointCallback(
            save_freq   = 10_000,
            save_path   = str(save_dir / 'periodic'),
            name_prefix = f'residual_{string}',
        ),
        RewardComponentLogger(),
    ]

    human_fb.start()
    print(f"\nTraining SB3 PPO on string {string} for {total_timesteps} steps...")

    model.learn(
        total_timesteps = total_timesteps,
        callback        = callbacks,
        progress_bar    = True,
    )

    model.save(str(save_dir / f'residual_{string}_final'))
    print(f"Saved final model → {save_dir}/residual_{string}_final.zip")


def _train_custom_ppo(train_env, string, total_timesteps,
                      save_dir, log_interval, seed, human_fb):
    """Train using custom PPO from existing ppo.py."""

    agent    = PPOAgent(env=train_env, seed=seed)
    n_updates = total_timesteps // agent.n_steps
    step_ref  = [0]

    human_fb.start()
    print(f"\nTraining custom PPO on string {string} "
          f"for {total_timesteps} steps ({n_updates} updates)...")

    history = []

    for update in range(1, n_updates + 1):

        # Set episode context for human feedback
        human_fb.set_episode(episode_id=update, step_ref=step_ref)

        # Collect rollout
        rollout_info = agent.collect_rollout()

        # Apply human flags before gradient update
        episode_flags = human_fb.get_episode_flags()
        if episode_flags:
            apply_human_flags_to_buffer(agent.buffer, episode_flags)
        human_fb.clear_episode()

        # PPO update
        train_info = agent.learn_from_buffer()

        ep_rews    = rollout_info.get('ep_rewards', [])
        mean_rew   = float(np.mean(ep_rews)) if ep_rews else float('nan')

        entry = {
            'update':          update,
            'timesteps':       update * agent.n_steps,
            'mean_ep_reward':  mean_rew,
            'human_flags':     len(episode_flags),
            **train_info,
        }
        history.append(entry)

        if update % log_interval == 0:
            print(
                f"update {update:4d} | "
                f"steps {entry['timesteps']:7d} | "
                f"ep_rew {mean_rew:+.3f} | "
                f"pi {entry['policy_loss']:+.4f} | "
                f"v {entry['value_loss']:.4f} | "
                f"H {entry['entropy']:.3f} | "
                f"flags {len(episode_flags)}"
            )

    agent.save(str(save_dir / f'residual_{string}_final.pt'))
    return history


# ── Pre-training validation ────────────────────────────────────────

def _validate_reward(baseline, reward_fn, string, note_sequence):
    """
    Verify reward signal correctly separates good from bad.
    Must pass before training — if it fails, fix classifier first.
    """
    from mocks.mock_classifier import MockSoundClassifier

    clf     = MockSoundClassifier()
    results = {}

    for condition, force_mult in [('good', 1.0), ('bad', 2.2)]:
        reward_fn.reset()
        baseline.reset(string=string)
        scores = []

        for _ in range(20):
            ba = baseline.get_baseline_action()
            if ba is None:
                break

            # Simulate good/bad force
            rs = baseline.get_simulated_state()
            rs['measured_force'] = ba['force_nominal'] * force_mult

            physical = np.array([
                rs['measured_force'], 0.0,
                rs.get('bow_speed', 0.0),
                rs.get('bow_position', 0.5),
                0.0, 0.0,
            ])

            sc       = baseline.get_score_context()
            r, comps = reward_fn.compute(None, physical, rs, sc)
            if comps['quality'] != 0.5:   # not neutral (buffer filling)
                scores.append(comps['quality'])

        results[condition] = float(np.mean(scores)) if scores else 0.5

    separation = results['good'] - results['bad']
    print(f"  Good condition quality: {results['good']:.3f}")
    print(f"  Bad  condition quality: {results['bad']:.3f}")
    print(f"  Separation:             {separation:.3f}")

    if separation < 0.2:
        print("  ⚠ WARNING: reward signal cannot separate good from bad.")
        print("    Do not train until this is resolved.")
    elif separation < 0.4:
        print("  ⚠ Marginal separation — RL will learn slowly.")
    else:
        print("  ✓ Good separation — reward signal ready for training.")


# ── Evaluation vs baseline ─────────────────────────────────────────

def evaluate_vs_baseline(model, string, note_sequence, n_episodes=10):
    """
    Compare policy against zero-residual baseline.
    This is your primary experimental result.
    """
    from mocks.mock_classifier import MockSoundClassifier

    results = {'baseline': [], 'policy': []}

    for condition in ['baseline', 'policy']:
        for ep in range(n_episodes):
            clf      = MockSoundClassifier()
            reward_fn = CompleteCelloReward(classifier_model=clf)
            baseline = BaselineController(note_sequence=note_sequence)

            env = SingleStringResidualEnv(
                baseline  = baseline,
                reward_fn = reward_fn,
                string    = string,
                mock_mic  = True,
            )

            obs, _     = env.reset()
            ep_rewards = []
            done       = False

            while not done:
                if condition == 'baseline':
                    action = np.zeros(ACTION_DIM)   # zero residual
                else:
                    if SB3_AVAILABLE:
                        action, _ = model.predict(obs, deterministic=True)
                    else:
                        action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_rewards.append(reward)
                done = terminated or truncated

            results[condition].append({
                'total_reward': float(np.sum(ep_rewards)),
                'mean_quality': float(np.mean([
                    r for r in ep_rewards
                    if r != 0.0
                ])) if ep_rewards else 0.0,
            })

    print("\n── Evaluation Results ──────────────────────────────────")
    print(f"{'Metric':<25} {'Baseline':>12} {'Policy':>12} {'Δ':>8}")
    print("-" * 57)

    for metric in ['total_reward', 'mean_quality']:
        base = np.mean([r[metric] for r in results['baseline']])
        pol  = np.mean([r[metric] for r in results['policy']])
        diff = pol - base
        marker = '↑' if diff > 0.02 else ('↓' if diff < -0.02 else '~')
        print(f"  {metric:<23} {base:>12.3f} {pol:>12.3f} {diff:>+7.3f} {marker}")

    return results


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--string',     default='A',
                        choices=['A', 'D', 'G', 'C'])
    parser.add_argument('--timesteps',  type=int, default=50_000)
    parser.add_argument('--save-dir',   default='checkpoints')
    parser.add_argument('--real-robot', action='store_true')
    parser.add_argument('--midi',       default=None,
                        help='Path to MIDI file (optional)')
    parser.add_argument('--seed',       type=int, default=0)
    args = parser.parse_args()

    train(
        string          = args.string,
        total_timesteps = args.timesteps,
        save_dir        = args.save_dir,
        use_real_robot  = args.real_robot,
        midi_path       = args.midi,
        seed            = args.seed,
    )