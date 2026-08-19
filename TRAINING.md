# Running a training run

```bash
PIECE=SoundClassifier/Data_Collection/public_annotation_packages_a_final/yunpiece.mxl
```

Use `.mxl` and not `.mid`.


### 1. Check the robot setup

```bash
ping -c3 192.168.1.100
~/venvs/cello311/bin/python -c "import sounddevice as sd; \
  print([d['name'] for d in sd.query_devices() if 'carlett' in d['name']])"
```

### 2. Calibrate on the piece you will train


Finds the offset between what the model predicts and what the microphone hears (audio recording calibration).

```bash
caffeinate -dims ~/venvs/cello311/bin/python rl/calibrate_gain.py \
  --measure --real --piece "$PIECE"          # add --write once the zones agree
```

### 3. Gate (optional, ~25 min)

```bash
caffeinate -dims ~/venvs/cello311/bin/python rl/reward_noise.py \
  --piece "$PIECE" --real --no-calibrated-dynamics --out rl/gate.json
```

### 4. Train

```bash
caffeinate -dims ~/venvs/cello311/bin/python -u rl/train_piece_logged.py \
  "$PIECE" --real --timesteps 18200 --ckpt-every 10 --no-calibrated-dynamics
```

`--timesteps` is strokes (182/episode × 100 = 18200, ~1.8 h).

Check the header says the aiming mode and gain offset you expect. Then:
`dyn 0.000` with in-zone 0 means the calibration is wrong — **stop**. Ignore
tone in episodes 1–2 (random warm-up).

### 5. When the link drops

Checkpoints save every 250 steps and on Ctrl-C, so little is lost. If the log
stops advancing for a minute the process is hung in RTDE reconnect:

```bash
pkill -f train_piece_logged
D=$(ls -dt rl/checkpoints_piece/run_* | head -1)     # newest — resumes make new dirs
caffeinate -dims ~/venvs/cello311/bin/python -u rl/train_piece_logged.py \
  "$PIECE" --real --timesteps REMAINING --ckpt-every 10 \
  --no-calibrated-dynamics --resume-run "$D"
```

`REMAINING` = budget minus every segment done (the per-segment counter
restarts). Never hand-type `--resume`/`--episode-offset`.

### 6. Select and listen

```bash
D=$(ls -dt rl/checkpoints_piece/run_* | head -1)
caffeinate -dims ~/venvs/cello311/bin/python rl/select_best.py "$D" \
  --piece "$PIECE" --real --episodes 2
caffeinate -dims ~/venvs/cello311/bin/python rl/ab_compare.py "$D" \
  --piece "$PIECE" --gap 8
```

`--tempo-scale` above 1 slows the piece down.

## Environment knobs

| | |
|---|---|
| `CELLO_ACCEL_MAX` | acceleration ceiling (default 5.5) |
| `CELLO_BOW_CONFIG` | standard, board, bridge, topangle, botangle |
| `CELLO_DYNAMIC_MAP` | rewrite dynamics, e.g. `p=f` |
| `CELLO_ARTICULATION_REF` | staccato crispness; lower = bigger gaps |
| `CELLO_SPEED_TRIM` / `CELLO_DEPTH_TRIM_MM` | scale plan speed / add depth |


