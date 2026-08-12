# ResidualRL Cello Robot

Residual reinforcement learning for expressive cello bowing with a UR robot,
a sound-quality classifier, and hardware-free mock environments. The mock path
lets contributors parse a score, plan bow strokes, and exercise the RL
environment without access to the robot.

## Quick setup

Install 64-bit Python 3.11, clone the repository, and run the setup script from
the repository root. Python 3.11 is used on both platforms to keep the PyTorch
environment consistent.

### macOS

```bash
./scripts/setup.sh
source .venv/bin/activate
```

If `python3.11` is not on your path, select an interpreter explicitly:

```bash
PYTHON_BIN=/path/to/python3.11 ./scripts/setup.sh
```

### Windows (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

Both scripts create `.venv`, install the project in editable mode, and run the
smoke tests. To set up manually:

```bash
python -m venv .venv
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Try the hardware-free path

```bash
python rl/play_piece.py MIDI-Files/t1.mid --episodes 1
```

Train a residual SAC policy against the mock executor and scorer:

```bash
python rl/train_piece.py MIDI-Files/twinkle_twinkle-open.mid --mock --timesteps 30000
```

Generated checkpoints, recordings, NumPy state captures, local datasets, and
annotation exports are ignored by Git. The small classifier checkpoints in
`SoundClassifier/checkpoints/` are intentionally versioned so inference works
after cloning.

## Optional profiles

Real-robot support adds `ur-rtde`:

```bash
# macOS
./scripts/setup.sh --hardware

# Windows PowerShell
.\scripts\setup.ps1 -Hardware
```

Annotation tooling adds Label Studio:

```bash
# macOS
./scripts/setup.sh --labeling

# Windows PowerShell
.\scripts\setup.ps1 -Labeling
```

Use both flags when both profiles are needed. On macOS, `ur-rtde` may need
CMake and Boost from Homebrew if a prebuilt wheel is unavailable. Real-robot
commands require the lab network and a correctly configured UR controller;
run hardware-free validation first and review the taught poses and safety
limits before enabling motion.

## Documentation

- [RL_METHOD.md](RL_METHOD.md): full technical description of the residual RL
  loop — MDP formulation, state and action spaces, the reward function term by
  term, the classifier used as reward model, SAC hyperparameters, measured
  results, and limitations. Written as a methods reference.
- [PIPELINE.md](PIPELINE.md): system architecture, the three RL environments,
  and the score → planner → robot → audio → reward data path.
- [RESULTS.md](RESULTS.md): measured outcomes and evaluation protocol.

## Repository map

- `rl/`: Gymnasium environments, whole-piece SAC training, and playback.
- `reward/`: classifier-window and residual reward logic.
- `SoundClassifier/`: audio features, model training, inference, and data tools.
- `BaselineControls/`: score planning and verified robot motion primitives.
- `MIDI-Files/`: small example scores used by mock runs and tests.
- `Robot-Programs/`: UR programs and installation artifacts.
- `other_github/`: retained reference implementation from earlier development.

The repository currently has no declared open-source license. Ask the project
owner before redistributing the code, model weights, or robot calibration data.
