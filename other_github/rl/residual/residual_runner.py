"""
residual_runner.py — residual-RL orchestration loop (OFFLINE dry-run).

Validates the whole plumbing without a robot or gym:
    obs -> PPO.sample -> residual on baseline -> execute -> reward -> PPO.update

Two reward modes:
  --reward classifier : reward = RobotToneReward.batch_score(a recorded wav).
        Tests the real reward path, but offline the audio is FIXED (decoupled from
        the action), so reward is constant -> validates PLUMBING, not learning.
  --reward synthetic  : reward = f(action) (high when the depth residual is near a
        target). The reward RESPONDS to the action, so PPO should LEARN: mean reward
        climbs and the mean action converges to the target -> validates LEARNING.

1D action (delta_depth). OBS_DIM = 10 mechanical (v1) or 15 with prev-stroke audio (v2, AUDIO_OBS);
networks take obs_dim from the caller.

Run (from the classifier_pilot venv that has torch + audiobox):
  .../python rl/residual/residual_runner.py --reward synthetic --episodes 30
  .../python rl/residual/residual_runner.py --reward classifier --audio <wav>

Author: Claude (for Zixian, 2026-06-04).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # rl/residual
REPO = os.path.dirname(os.path.dirname(HERE))              # repo root
CLF = os.path.join(os.path.dirname(REPO), "classifier_pilot")
for _p in (HERE, REPO, CLF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from zero_baseline import ZeroBaseline           # noqa: E402
from obs_norm import normalize                    # noqa: E402
# NOTE: rl.ppo.ppo (torch) is imported lazily in main() so that importing this
# module for build_obs/apply_residual works in the torch-free robot venv (R41).

# v1 observation (agreed 2026-06-04): 10 mechanical features, in order, no audio.
OBS_FEATURES = ["depth", "contact_force_magnitude", "bow_speed", "bow_position",
                "bow_direction", "dev_depth", "bow_accel", "j4", "j5", "j6"]

# v2 (2026-07-05, closed-loop upgrade): + the PREVIOUS stroke's audio — the policy "hears" the
# last stroke and decides this stroke's depth from it. This is what makes the deployed per-stroke
# policy a genuine feedback controller (the 06-29 review: constant obs => RL-tuned open-loop).
# Flip AUDIO_OBS=False to recover the v1 10-dim obs exactly.
AUDIO_OBS = True
AUDIO_OBS_FEATURES = ["prev_rms", "prev_flatness", "prev_hf2k", "prev_centroid", "prev_reward"]
AUDIO_OBS_SCALES = np.array([0.10,    # rms — CURRENT domain (2026-07-10 restored config: good stroke
                                      # feat_rms ~0.09-0.11). Was 0.03 (old chain, d=3 ~0.028): under
                                      # the new level that fed the policy prev_rms ~3.3 instead of ~1
                                      # (2026-07-12 review). Ckpts store their scales in meta — eval
                                      # warns on mismatch; retrain rather than mix scales.
                             0.15,    # spectral flatness (silence ~0.17, clean ~0.06)
                             0.4,     # hf2k ratio (0.2-0.5)
                             2000.0,  # spectral centroid Hz (~1600-2200)
                             1.0],    # prev reward already 0..1
                            dtype=np.float32)

OBS_DIM = len(OBS_FEATURES) + (len(AUDIO_OBS_FEATURES) if AUDIO_OBS else 0)   # 15 (v2) / 10 (v1)
ACT_DIM = 1                     # delta_depth (1D start)
DEPTH_RESIDUAL_CLAMP = 0.005    # +-5mm (Zixian 2026-06-27: nominal d=2 far from the good band, ±5mm = room
                                # to climb; swept-safe). KEEP == hardware_baseline.DEPTH_RESIDUAL_CLAMP.
SYNTH_TARGET = 0.5              # normalised residual the policy should learn (synthetic mode)
SYNTH_SIGMA = 0.4


def build_obs(state: dict, prev_audio=None) -> np.ndarray:
    """v2 obs: 10 mechanical features (normalised) + the previous stroke's audio
    [rms, flatness, hf2k, centroid, reward] (normalised by AUDIO_OBS_SCALES).
    prev_audio=None (first stroke / no history) -> zeros. AUDIO_OBS=False -> v1 10-dim."""
    raw = np.array([state.get(k, 0.0) for k in OBS_FEATURES], dtype=np.float32)
    obs = normalize(raw)
    if AUDIO_OBS:
        pa = (np.zeros(len(AUDIO_OBS_FEATURES), dtype=np.float32) if prev_audio is None
              else np.asarray(prev_audio, dtype=np.float32))
        assert pa.shape == (len(AUDIO_OBS_FEATURES),), f"prev_audio shape {pa.shape}"
        obs = np.concatenate([obs, pa / AUDIO_OBS_SCALES]).astype(np.float32)
    return obs


def apply_residual(a_base: dict, delta: np.ndarray) -> dict:
    """Attach the (clamped) depth residual in METERS. Baseline-agnostic:
    ZeroBaseline ignores it; HardwareBaseline applies it to the TCP z."""
    d = float(np.clip(delta[0], -1.0, 1.0)) * DEPTH_RESIDUAL_CLAMP
    a = dict(a_base)
    a["depth_residual"] = d
    return a


def synth_reward(action: np.ndarray) -> float:
    """Action-dependent reward: peaks (=1) when the residual equals SYNTH_TARGET."""
    a = float(np.clip(action[0], -1.0, 1.0))
    return float(np.exp(-((a - SYNTH_TARGET) / SYNTH_SIGMA) ** 2))


def collect_episode(baseline, agent, reward_mode, reward_fn=None, audio=None, sr=None):
    baseline.reset()
    transitions, actions = [], []
    while not baseline.is_complete():
        obs = build_obs(baseline.get_full_state())
        out = agent.sample(obs)
        baseline.execute_timestep(apply_residual(baseline.get_baseline_action(),
                                                 out["clipped_action"]))
        r = synth_reward(out["action"]) if reward_mode == "synthetic" else None
        transitions.append({"obs": obs, "action": out["action"], "value": out["value"],
                            "log_prob": out["log_prob"], "reward": r, "done": False})
        actions.append(float(out["action"][0]))
    transitions[-1]["done"] = True

    if reward_mode == "classifier":
        scores = reward_fn.batch_score(audio, sr)
        n = len(transitions)
        for i, tr in enumerate(transitions):
            if len(scores):       # average the tick's score SPAN (parity with run_rl)
                j0 = int(i / n * len(scores))
                j1 = max(j0 + 1, int((i + 1) / n * len(scores)))
                tr["reward"] = float(np.mean(scores[j0:j1]))
            else:
                tr["reward"] = 0.0

    last_obs = build_obs(baseline.get_full_state())
    return transitions, last_obs, float(np.mean(actions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["classifier", "synthetic"], default="classifier")
    ap.add_argument("--audio", default=None, help="wav (required for --reward classifier)")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=30)
    args = ap.parse_args()

    reward_fn, audio, sr = None, None, None
    if args.reward == "classifier":
        if not args.audio:
            ap.error("--reward classifier needs --audio")
        import soundfile as sf
        from robot_tone_reward import RobotToneReward
        audio, sr = sf.read(args.audio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        reward_fn = RobotToneReward()      # in-domain WavLM probe (Phase-1 reward)

    from rl.ppo.ppo import PPOAgent   # lazy: torch only needed when actually training
    baseline = ZeroBaseline(n_ticks=args.ticks)
    agent = PPOAgent(env=None, obs_dim=OBS_DIM, act_dim=ACT_DIM,
                     n_steps=args.ticks, n_epochs=4, batch_size=16,
                     device="cpu", seed=0)

    hdr = f"OBS_DIM={OBS_DIM} ACT_DIM={ACT_DIM} | reward={args.reward}"
    if args.reward == "synthetic":
        hdr += f" (target residual={SYNTH_TARGET})"
    print(hdr)
    print("--- offline residual-RL dry-run ---")
    for ep in range(1, args.episodes + 1):
        tr, last_obs, mean_a = collect_episode(baseline, agent, args.reward,
                                               reward_fn, audio, sr)
        mean_r = float(np.mean([t["reward"] for t in tr]))
        info = agent.update(tr, last_obs)
        extra = f"  mean_action={mean_a:+.3f}" if args.reward == "synthetic" else ""
        print(f"ep {ep:2d}: mean_reward={mean_r:.3f}{extra}  "
              f"| pi={info['policy_loss']:+.4f} v={info['value_loss']:.3f} "
              f"H={info['entropy']:.3f} kl={info['approx_kl']:+.4f}")

    if args.reward == "synthetic":
        print(f"\nIf learning works: mean_reward should climb toward 1.0 and "
              f"mean_action toward {SYNTH_TARGET}.")
    print("DRY-RUN OK: obs -> sample -> residual -> reward -> update ran end-to-end.")


if __name__ == "__main__":
    main()
