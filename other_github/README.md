# Robot Cello RL

Closed-loop reinforcement learning for a UR5e robot arm bowing a cello. The robot
reads a musical score (MIDI), bows the string on real hardware, and a learned
audio-quality reward lets a policy adjust its bowing in real time — extending prior
open-loop score-to-sound work toward expressive, self-correcting performance.

Purdue AIM Research Group (ai4musicians.org). Target: ICRA 2027.

## Overview

- **Baseline controller** — parses a MIDI file, maps notes to a string, and bows them
  on the UR5e via RTDE. Open-loop position control; contact "force" is approximated by
  a small depth offset along the press direction.
- **Recording** — drives systematic bow strokes (depth × speed × bow position) and
  records synchronized audio + robot state for analysis and reward training.
- **Reinforcement learning** — a custom PPO agent learns a small *residual* on top of
  the baseline (currently a 1-D depth correction), rewarded by an audio tone-quality
  classifier scored from the recorded stroke.

## Repository layout

```
Baseline-Runners/         Robot control + RTDE scripts
  baseline_controller.py    MIDI -> bowing on the UR5e (the baseline)
  Physical-Data/            Cello geometry / bow-pose calibration
recording_script.py       Synchronized stroke recording (audio + robot state)
mic_recorder.py           Audio capture
rl/
  ppo/                      Custom PPO (pure PyTorch)
  residual/                 Residual RL on top of the baseline (the closed loop)
  sac/                      Earlier SAC pipeline (legacy)
reward/                    Audio tone-quality classifier + reward signal
models/                    GP safety gate / reward surrogate (experimental)
MIDI-Files/                Input scores
```

## Setup

```bash
pip install -r requirements.txt        # learning + audio
pip install -r requirements-rtde.txt   # robot control (ur_rtde; needs boost@1.85 on macOS)
```

## Usage

```bash
# Bow a MIDI file on the robot with the baseline controller
python Baseline-Runners/baseline_controller.py MIDI-Files/twinkle_twinkle-open.mid

# PPO smoke test (simulation)
python -m rl.ppo.train_ppo --timesteps 3000 --seed 0
```

## Status

Active development toward an ICRA 2027 submission: characterizing the bowing action
space on the real robot, building an in-domain audio reward, and training residual RL
starting from the depth dimension.
