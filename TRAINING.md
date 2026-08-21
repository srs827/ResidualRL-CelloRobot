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
  --measure --real --write --piece "$PIECE"
```

### 3. Gate 

Checks whether the reward is consistent per-stroke, and differs between different strokes. 
Important to run this before training because if the tests fail there is some calibration issue.

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

Ctrl-C once saves the model and replay buffer — resume with `--resume-run`
(see 5), setting `--timesteps` to what is left and re-setting any `CELLO_*`
you started with. Stopped after 6959 of 18200 strokes:

```bash
CELLO_DYNAMIC_MAP="p=f" caffeinate -dims ~/venvs/cello311/bin/python -u \
  rl/train_piece_logged.py "$PIECE" --real --timesteps 11241 --ckpt-every 10 \
  --no-calibrated-dynamics --resume-run rl/checkpoints_piece/run_20260819_153618
```

The interrupt message names the run dir; otherwise it is the newest. Strokes
done is that file's line count minus its header line:

```bash
D=$(ls -dt rl/checkpoints_piece/run_* | head -1)
echo $(( $(wc -l < "$D/stroke_log.jsonl") - 1 ))
```

Count strokes, not episodes — the `[checkpoint] N/M` counter restarts at zero
on every resume.

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
`--no-calibrated-dynamics` turns off calibrate_dynamics() so the planner converts each written dynamic to a bow speed
                           with a plain linear rule instead of inverting the loudness model to re-aim the note at its target dB level.

## Environment knobs

| | |
|---|---|
| `CELLO_ACCEL_MAX` | acceleration ceiling (default 5.5) |
| `CELLO_BOW_CONFIG` | standard, board, bridge, topangle, botangle |
| `CELLO_DYNAMIC_MAP` | rewrite dynamics, e.g. `p=f` |
| `CELLO_ARTICULATION_REF` | staccato crispness; lower = bigger gaps |
| `CELLO_SPEED_TRIM` / `CELLO_DEPTH_TRIM_MM` | scale plan speed / add depth |



## Alternate loop (`CELLO_ALT_LOOP=1`)

Four changes, one switch. Each targets a measured cause of the residual not
beating the planner:

| | |
|---|---|
| timing penalty | charges a stroke for the dispatch cost of shaping it |
| zero-init | actor starts AT the baseline — `mu` **and** `log_std`, ent_coef 0.1 |
| reference episodes | zero-residual take every 10 episodes, logged as `{"reference": true}` |
| phrase scoring | judge sees a full window instead of a fill-shifted fragment |

It also raises `ENVELOPE_MIN_DURATION` 0.40 → 0.50 s so the policy only
shapes notes the judge can grade (the CNN agrees with human ratings +0.879 on
sustained notes and inverts on 0.111 s ones).

Run the stock command with the prefix; drop it for the before/after pair.

```bash
CELLO_ALT_LOOP=1 caffeinate -dims ~/venvs/cello311/bin/python -u \
  rl/train_piece_logged.py "$PIECE" --real --timesteps 18200 --ckpt-every 10 \
  --no-calibrated-dynamics
```

The header must print `ALT loop: timing_penalty=True zero_init=True
phrase_scoring=True`. If it does not, the variable did not take and this is a
stock run. Early episodes look *quiet* rather than random — that is the
zero-init working, not a stall.

Individual toggles: `CELLO_TIMING_PENALTY`, `CELLO_ZERO_INIT`,
`CELLO_PHRASE_SCORING`, `CELLO_REFERENCE_EVERY`, `CELLO_ENVELOPE_MIN_S`.

**`ab_compare` needs `--no-calibrated-dynamics` explicitly.** Its
`--calibrated-dynamics` default is True and, until 2026-08-21, appended
nothing — so every A/B silently ran uncalibrated. Now that the flag works,
the default would run *calibrated* and no longer match how the run trained.
