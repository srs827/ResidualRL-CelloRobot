# Running a training run

```bash
PIECE=SoundClassifier/Data_Collection/public_annotation_packages_a_final/yunpiece.mxl
```

Prefer `.mxl` over `.mid` — MIDI carries no bowings, dynamics or articulation.

**Pass `--no-calibrated-dynamics` to every command below.** `reward_noise` and
`train_piece` default it on, `select_best` and `perform` default it off; the
defaults gate, train, select and perform in different regimes.

### 1. Check the rig

```bash
ping -c3 192.168.1.100
~/venvs/cello311/bin/python -c "import sounddevice as sd; \
  print([d['name'] for d in sd.query_devices() if 'carlett' in d['name']])"
```

Both must succeed; the USB bus carries the robot link and the audio interface
and both drop. Reseat before starting, not during.

### 2. Calibrate — on the piece you will train

```bash
caffeinate -dims ~/venvs/cello311/bin/python rl/calibrate_gain.py \
  --measure --real --piece "$PIECE"          # add --write once the zones agree
```

The offset is measured through the same window the reward grades through, so
an offset from another piece is wrong in proportion to the note-length
difference. `--write` is refused if zones disagree by >1.5 dB — understand it,
don't bypass it. The guard needs two zones at n>=8 to fire at all, so a
single-dynamic piece can't trip it and its clean result is untrustworthy.

### 3. Gate (optional, ~25 min)

```bash
caffeinate -dims ~/venvs/cello311/bin/python rl/reward_noise.py \
  --piece "$PIECE" --real --no-calibrated-dynamics --out rl/gate.json
```

Read the **per-stroke** SNR. The headline is a mean of three ratios and one
good stroke carries it.

### 4. Train

```bash
caffeinate -dims ~/venvs/cello311/bin/python -u rl/train_piece_logged.py \
  "$PIECE" --real --timesteps 18200 --ckpt-every 10 --no-calibrated-dynamics
```

`--timesteps` is **strokes** (182/episode × 100 = 18200, ~1.8 h).

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

`select_best` only sees one directory, so interrupted runs leave earlier
checkpoints behind. `--tempo-scale` above 1 slows the piece down.

## Environment knobs

All default to no-op, print when set, and shift the **planner's nominal** — so
baseline and policy both get them, and a policy trained with a trim must be
performed with it.

| | |
|---|---|
| `CELLO_ACCEL_MAX` | acceleration ceiling (default 5.5) |
| `CELLO_BOW_CONFIG` | standard, board, bridge, topangle, botangle |
| `CELLO_DYNAMIC_MAP` | rewrite dynamics, e.g. `p=f` |
| `CELLO_ARTICULATION_REF` | staccato crispness; lower = bigger gaps |
| `CELLO_SPEED_TRIM` / `CELLO_DEPTH_TRIM_MM` | scale plan speed / add depth |

## Two limits

**Short notes.** The tone judge scores +0.879 against human ratings on the
sustained notes it was trained on and *inverts* below ~0.15 s. On a piece
mostly shorter than that, expect training to stop the policy doing the wrong
thing rather than teach it a better one.

**Contact force.** `torque_contact_est_N` should sit inside 3.69–5.66 N. If
commanding more speed and depth makes the recording *quieter*, the bow isn't
gripping — check hair tension and rosin, don't compensate in software.
