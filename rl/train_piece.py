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
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rl.piece_env as pe
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
        # Episodes already completed before this process started. Stays 0 for
        # a fresh run; a resuming wrapper sets it so the printed episode
        # number matches the one written to the stroke log. Without it a
        # correct --resume-run prints "[episode 1]" while logging episode
        # N+1, which reads as a stale checkpoint and invites the restart that
        # flag exists to prevent.
        self.episode_offset = 0

    def __call__(self, locals_, globals_):   # SB3 callback signature
        infos = locals_.get("infos", [])
        for info in infos:
            if "mean_quality" not in info:
                continue
            # The reward optimises the length-aware head MIX ("tone" in the
            # stroke log, what select_best reports), not the overall head.
            # From 08-13 to 08-20 this line printed the overall head labelled
            # "mean tone quality"; the ~+0.18 level gap between the heads
            # read as a train-vs-eval collapse that never existed. Print the
            # reward's own number; keep overall alongside for continuity
            # with old run transcripts.
            ep_log = info.get("episode_log")
            mean_tone = (float(np.mean([s.get("tone", s["quality"])
                                        for s in ep_log]))
                         if ep_log else info["mean_quality"])
            self.history.append(mean_tone)
            n = len(self.history) + self.episode_offset
            recent = float(np.mean(self.history[-10:]))
            extra = ""
            if "mean_dynamic" in info:
                self.dynamics.append(info["mean_dynamic"])
                extra = (f"   dyn {info['mean_dynamic']:.3f} "
                         f"(last-10 {float(np.mean(self.dynamics[-10:])):.3f}, "
                         f"in-zone {info.get('in_zone', '?')})")
            # Return is what SAC maximises and what select_best ranks by;
            # leaving it off the console is how "watched tone, optimised
            # total" confusions start.
            ep_ret = (sum(s["total"] for s in ep_log) if ep_log else None)
            ret_txt = f"  ret {ep_ret:6.2f}" if ep_ret is not None else ""
            print(f"[episode {n:4d}] tone {mean_tone:.3f}{ret_txt} "
                  f"(overall {info['mean_quality']:.3f})   "
                  f"(last-10 tone {recent:.3f}){extra}")

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

        # (2) START THE POLICY AT THE BASELINE.
        #
        # The action is a RESIDUAL, so zero IS the planner's nominal -- the one
        # known-good point in the space. Stock SAC does not know that: it fills
        # the buffer with `learning_starts` steps of UNIFORM RANDOM actions and
        # begins with ent_coef near 0.85, so it samples at |a| ~ 0.57, wider
        # than uniform. The policy therefore optimises away from random
        # flailing, never away from the planner, and nothing in training ever
        # asks whether it beats doing nothing.
        #
        # That predicts what we measured across three pieces: climbing away
        # from random lands ABOVE a weak baseline (yunpiece, ~0 within noise)
        # and BELOW a strong one (vocalise -0.058, twinkle negative). The
        # outcome tracked the baseline's quality, which is the signature of the
        # baseline not being the starting point.
        #
        # Zero-initialising the actor's output layer makes the untrained
        # deterministic policy the baseline exactly, so any departure is a
        # deliberate choice the critic has to justify. Standard practice in
        # residual RL; omitted here until now.
        if pe.ALT_ZERO_INIT:
            import torch
            mu = model.policy.actor.mu
            with torch.no_grad():
                mu.weight.zero_()
                if mu.bias is not None:
                    mu.bias.zero_()
            model.ent_coef = "auto_0.1"          # start far less exploratory
            model.learning_starts = min(model.learning_starts, 20)
            print("  ALT zero-init: actor mu zeroed (policy starts AT the "
                  "baseline), ent_coef auto_0.1, learning_starts "
                  f"{model.learning_starts}")

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

            # (3) ZERO-RESIDUAL REFERENCE EPISODE.
            #
            # Nothing else in the loop asks whether the policy beats doing
            # nothing. The residual's zero action IS the planner's nominal,
            # but it is one sample among thousands and select_best is the
            # first thing to test it -- after the whole run. On vocalise the
            # training curve rose cleanly (tone_eff 0.363 -> 0.444) while the
            # deterministic policy finished BELOW its own baseline (-0.058).
            #
            # Run here, at the chunk boundary, and NOT from the callback: the
            # callback fires inside collect_rollouts, so resetting the env
            # there would leave SB3's _last_obs stale and corrupt the rollout.
            # Between chunks nothing is in flight and a reset is safe.
            if pe.ALT_ZERO_INIT or os.environ.get("CELLO_REFERENCE", ""):
                try:
                    renv = env.envs[0].unwrapped if hasattr(env, "envs") else env
                    obs0, _ = renv.reset()
                    zero = np.zeros(renv.action_space.shape, dtype=np.float32)
                    tot, tones, done = 0.0, [], False
                    while not done:
                        obs0, r0, te, tr, i0 = renv.step(zero)
                        tot += float(r0); done = te or tr
                        st = i0.get("stroke") or {}
                        if "quality_eff" in st:
                            tones.append(st["quality_eff"])
                    tn = float(np.mean(tones)) if tones else float("nan")
                    print(f"[reference] zero-residual baseline at "
                          f"{steps_done}/{args.timesteps}: return {tot:.2f}  "
                          f"tone_eff {tn:.3f}   <- the line the policy has to "
                          f"cross")
                except Exception as e:
                    print(f"(reference episode failed: {e} — continuing)")

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
