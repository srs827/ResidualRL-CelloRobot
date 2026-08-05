"""
demo.py

Audio file → next force command on terminal.

Pipeline:
    1. Load audio WAV
    2. Extract 7 features (librosa-based replacement for Essentia)
    3. Classify with the team-provided DeepMLP (Good/Bad + confidence)
    4. Build a 16-dim RL observation
    5. Run the PPO policy → next force command

Usage:
    python demo.py --audio path/to/sound.wav
    python demo.py --audio path/to/sound.wav --model checkpoints/ppo_cello_A_final.pt
    python demo.py --audio path/to/sound.wav --current-force 4.5

If --model is omitted, a fresh PPOAgent is used (random weights). The demo
still works — it shows the pipeline end-to-end — but the predicted force will
be near the initial force since the network hasn't learnt anything.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import librosa
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# Project imports
from rl.sac.cello_env import CelloEnv
from rl.ppo.ppo import PPOAgent
from reward.real_classifier import RealSoundClassifier


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

WIDTH = 72
RULE  = "=" * WIDTH
SUB   = "-" * WIDTH


def section(title: str):
    print(f"\n{title}")
    print(SUB)


def banner(title: str):
    print(RULE)
    print(f"  {title}")
    print(RULE)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_demo(
    audio_path: str,
    model_path: str | None,
    current_force: float,
    pitch_midi: float | None,
):
    banner("Robot Cello RL Demo  --  Audio to Next Force")

    # ---------- [1/5] Audio Input ----------
    section("[1/5] Audio Input")
    audio_path = Path(audio_path)
    y, sr = librosa.load(audio_path, sr=44100, mono=True)
    duration = len(y) / sr
    rms      = float(np.sqrt(np.mean(y ** 2)))
    print(f"  File          : {audio_path.name}")
    print(f"  Duration      : {duration:.2f} s  @ {sr} Hz")
    print(f"  Samples       : {len(y)}")
    print(f"  RMS amplitude : {rms:.4f}")

    # ---------- [2/5] Feature Extraction ----------
    section("[2/5] Feature Extraction (librosa-based, mimics Essentia)")
    # Decide pitch: explicit flag wins, otherwise try filename, otherwise estimate
    if pitch_midi is None:
        # Try filename, e.g. "note_007_A3.wav" -> A3 -> 57
        stem = audio_path.stem
        try:
            note_name = stem.split("_")[-1]
            pitch_midi = float(librosa.note_to_midi(note_name))
            pitch_source = f"parsed from filename ({note_name})"
        except (ValueError, IndexError):
            pitch_midi = None
            pitch_source = "estimated from audio (librosa.pyin)"
    else:
        pitch_source = "user-specified"

    classifier = RealSoundClassifier(pitch_midi=pitch_midi)
    label, conf = classifier.predict(y, sr=sr)
    details = classifier.last_details

    pitch_used = details["pitch_midi"]
    note_name_used = librosa.midi_to_note(int(round(pitch_used)))
    print(f"  Pitch source  : {pitch_source}  ->  MIDI {pitch_used:.1f} ({note_name_used})")
    print()
    for name, value in details["features_raw"].items():
        if name == "pitch":
            continue
        short = name.split(".", 1)[-1] if "." in name else name
        print(f"    {short:<32} {value:>10.4f}")

    # ---------- [3/5] Sound Classifier ----------
    section("[3/5] Sound Classifier (DeepMLP, 7 -> 512 -> 256 -> 128 -> 2)")
    scaled = details["features_scaled"]
    logits = details["logits"]
    probs  = details["probs"]

    print(f"  Scaled input  : [{', '.join(f'{v:+.2f}' for v in scaled)}]")
    print(f"  Raw logits    : [Bad={logits[0]:+.3f},  Good={logits[1]:+.3f}]")
    print(f"  Softmax probs : [Bad={probs[0]:.3f},  Good={probs[1]:.3f}]")
    print(f"  -> Prediction : {label}  (confidence {conf:.1%})")

    # ---------- [4/5] RL State ----------
    section("[4/5] RL State (16-dim observation)")
    # We don't have a real robot. Build a plausible observation:
    #   - TCP pose / F-T sensor : zeros (stubs)
    #   - bow position          : 0.5 (mid-bow)
    #   - speed/accel           : 0
    #   - current force         : argument
    env = CelloEnv(classifier=classifier)  # only needed to share constants
    env._force = current_force
    hw_obs = np.zeros(15, dtype=np.float32)
    hw_obs[14] = 0.5     # bow position mid-stroke
    obs = np.append(hw_obs, current_force).astype(np.float32)

    print(f"  tcp_pose      : {hw_obs[0:6].tolist()}  (stub)")
    print(f"  force_torque  : {hw_obs[6:12].tolist()}  (stub)")
    print(f"  bow speed     : {hw_obs[12]:.2f} m/s   bow accel: {hw_obs[13]:.2f} m/s^2")
    print(f"  bow position  : {hw_obs[14]:.2f}  (0 = frog, 1 = tip)")
    print(f"  current force : {current_force:.2f} N")

    # ---------- [5/5] PPO Policy ----------
    section("[5/5] PPO Policy (Actor: 16 -> 64 -> 64 -> 1)")
    agent = PPOAgent(env=env)
    if model_path is not None and Path(model_path).exists():
        agent.load(model_path)
        model_info = f"loaded from {model_path}"
    else:
        model_info = "fresh random-weight network (no training)"
    print(f"  Model state   : {model_info}")

    # Get the action (deterministic)
    action, _value = agent.predict(obs, deterministic=True)
    delta_norm = float(np.asarray(action).flatten()[0])
    delta_N    = delta_norm * env.MAX_DELTA_N
    next_force = float(np.clip(current_force + delta_N, env.FORCE_MIN, env.FORCE_MAX))

    print(f"  Action (norm.): {delta_norm:+.3f}   (in [-1, +1])")
    print(f"  Scaled dForce : {delta_N:+.3f} N   (action * MAX_DELTA_N={env.MAX_DELTA_N})")
    print(f"  Force command : {current_force:.2f}  ->  {next_force:.2f} N "
          f"(clamped to [{env.FORCE_MIN}, {env.FORCE_MAX}])")

    # ---------- Final result ----------
    print()
    banner(f"RESULT: next force command = {next_force:.2f} N")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Robot Cello RL — audio to force demo")
    p.add_argument("--audio", required=True, help="path to a WAV file")
    p.add_argument("--model", default=None, help="path to trained PPO .pt (optional)")
    p.add_argument("--current-force", type=float, default=3.0,
                   help="assumed current applied force in Newtons (default 3.0)")
    p.add_argument("--pitch", type=float, default=None,
                   help="MIDI pitch override (e.g. 57 = A3). If omitted, parsed "
                        "from filename or estimated from audio.")
    args = p.parse_args()
    run_demo(args.audio, args.model, args.current_force, args.pitch)


if __name__ == "__main__":
    main()
