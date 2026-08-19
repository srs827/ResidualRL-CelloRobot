# Running a training run

How to train a residual policy on a score, end to end, on the real robot.
Written 2026-08-19 from the yunpiece and twinkle sessions; the troubleshooting
section is every failure those runs actually hit.

Replace `PIECE` throughout with your score, e.g.:

```bash
PIECE=SoundClassifier/Data_Collection/public_annotation_packages_a_final/yunpiece.mxl
```

`.mxl` is strongly preferred over `.mid`. MIDI has no bowings, slurs,
articulations or written dynamics, so the planner has to guess loudness from
velocity alone — measured on yunpiece that was 3.8 mm of bow per note against
11.3 mm from the `.mxl` of the same piece.

---

## 0. Before you touch the robot

```bash
ping -c3 192.168.1.100                      # the UR
~/venvs/cello311/bin/python -c "import sounddevice as sd; \
  print([d['name'] for d in sd.query_devices() if 'carlett' in d['name']])"
```

Both must succeed. The USB bus carries the ethernet dongle **and** the audio
interface, and both drop — eight link drops in one session on 2026-08-19.
If either is missing, reseat before starting rather than during.

Then check the piece is sane, offline and free:

```bash
~/venvs/cello311/bin/python -c "
import sys; sys.path.insert(0,'.')
from rl.piece_env import load_piece, PMP
import numpy as np, collections
n,m,st = load_piece('$PIECE', calibrated_dynamics=False)
d=np.array([s.duration for s in st]); u=np.array([[s.u_start,s.u_end] for s in st])
print(f'{len(st)} strokes  dur med {np.median(d):.3f}s  under 0.15s {np.mean(d<0.15):.0%}')
print(f'u {u.min():.3f}-{u.max():.3f} (limits {PMP.U_MIN}-{PMP.U_MAX})')
print('dynamics', collections.Counter(s.dynamic for s in st))"
```

**The "under 0.15 s" number decides how much to trust the result.** The tone
judge was trained on 0.5 s windows of sustained bowing. A piece that is mostly
shorter than that is outside its domain — see *Known limits* below.

---

## 1. Calibrate on the piece you are about to train

The gain offset is measured through the same analysis window the reward grades
through, so it must be measured **on the piece you will train**. An offset
from another piece is wrong in proportion to how much the note lengths differ.

```bash
caffeinate -dims ~/venvs/cello311/bin/python rl/calibrate_gain.py \
  --measure --real --piece "$PIECE"
```

Read the per-zone lines before writing anything. Then re-run with `--write`.

`--write` is refused if the per-zone medians disagree by more than 1.5 dB.
That guard is doing its job — a single scalar cannot represent the piece — and
the fix is to understand why, not to bypass it. Note the guard needs **two
zones with n >= 8** to fire at all, so a single-dynamic piece cannot trip it
and its clean-looking result should not be trusted.

Expect the offset to be stable across days: +5.22 (t1, 8/17) and +5.06
(yunpiece, 8/18) agreed to 0.16 dB once piece-matched.

---

## 2. The gate (optional, ~25 min, worth it)

```bash
caffeinate -dims ~/venvs/cello311/bin/python rl/reward_noise.py \
  --piece "$PIECE" --real --no-calibrated-dynamics \
  --out rl/reward_noise_$(date +%Y%m%d).json
```

Asks whether the reward can tell a good action from a bad one at all: it
replays one action to get the measurement noise, then varies the action to get
the signal. **Read the per-stroke SNR, not the headline** — the headline is
`np.mean` of three ratios and one good stroke can carry it. Yunpiece read
8.12 / 1.28 / 0.69 and printed "worth the hardware time" while two of three
strokes were below the tool's own action-invisible line.

`--no-calibrated-dynamics` is **not** optional: the script defaults it to
*true* while training defaults it to *false*, so without it the gate measures
a different environment than the run it is gating.

---

## 3. Train

```bash
caffeinate -dims ~/venvs/cello311/bin/python -u rl/train_piece_logged.py \
  "$PIECE" --real --timesteps 18200 --ckpt-every 10 --no-calibrated-dynamics
```

`--timesteps` is **strokes**, not episodes: strokes-per-episode × episodes.
182 strokes × 100 passes = 18200, about 1.8 h.

Always pass `--no-calibrated-dynamics`. `train_piece.py` defaults it on while
`select_best.py` and `perform.py` default it off, so the default path trains,
selects and performs in three different regimes.

### Watch the first two episodes

The header must read what you expect:

```
  aiming: planner open-loop
  gain_offset: +5.06 dB (Measured 2026-08-18)
```

Then, per episode: `dyn` and in-zone should start near your calibration's
in-zone rate and hold or climb. **`dyn 0.000` with in-zone 0 means the
calibration is wrong — stop.** That is not training going badly.

The first ~200 strokes are uniform random (`learning_starts`), so ignore tone
in episodes 1–2 entirely.

---

## 4. When the link drops

It will. Nothing is lost beyond the episodes since the last checkpoint — the
model and replay buffer are saved every 250 real steps, and again on Ctrl-C.

```bash
ping -c2 192.168.1.100                      # is it back?
ps aux | grep '[t]rain_piece_logged'        # is it hung?
```

If the log has not advanced for a minute, the process is spinning in RTDE's
reconnect loop and will not recover. Kill it, then resume from the **newest**
run directory — each resume creates a new one:

```bash
pkill -f train_piece_logged
D=$(ls -dt rl/checkpoints_piece/run_* | head -1)
caffeinate -dims ~/venvs/cello311/bin/python -u rl/train_piece_logged.py \
  "$PIECE" --real --timesteps REMAINING --ckpt-every 10 \
  --no-calibrated-dynamics --resume-run "$D"
```

`REMAINING` = your original budget minus every segment completed so far. The
per-segment counter restarts, so add up the `[checkpoint] N/M` lines.

Never hand-type `--resume` and `--episode-offset`; `--resume-run` derives both
and refuses to run alongside them. A stale hand-typed resume silently rewound
three runs to the same early checkpoint.

Verify the banner: it prints the episode number the first resumed episode
should show, and the scores should be continuous with where you stopped.

---

## 5. Pick a checkpoint

```bash
D=$(ls -dt rl/checkpoints_piece/run_* | head -1)
caffeinate -dims ~/venvs/cello311/bin/python rl/select_best.py "$D" \
  --piece "$PIECE" --real --episodes 2
```

Writes `sac_piece_best.zip` and `best.json`. Note it only sees checkpoints in
**one** directory — if the run was interrupted, the earlier segments' numbered
checkpoints are elsewhere and will be skipped.

Return and tone routinely disagree, and at `n=2` the difference is usually not
resolvable. If robot time is short you can rank checkpoints from the training
logs instead, but those averages are exploration-subsidised and can hide a bad
deterministic policy.

---

## 6. Listen

```bash
caffeinate -dims ~/venvs/cello311/bin/python rl/ab_compare.py "$D" \
  --piece "$PIECE" --gap 8
```

Baseline, a gap, then the policy — both through the per-note player, which
tracks about 7 ms. Each pass writes a wav so you can re-listen without the
robot.

`--tempo-scale` accepts a value **above** 1 to slow the piece down.

---

## Environment knobs

All default to no-op and print themselves when set.

| variable | effect |
|---|---|
| `CELLO_ACCEL_MAX` | planner acceleration ceiling (default 5.5) |
| `CELLO_BOW_CONFIG` | pendant-taught bow setup: standard, board, bridge, topangle, botangle |
| `CELLO_DYNAMIC_MAP` | rewrite written dynamics, e.g. `p=f` |
| `CELLO_ARTICULATION_REF` | staccato crispness; lower = shorter notes, bigger gaps |
| `CELLO_SPEED_TRIM` | multiply the plan's bow speed |
| `CELLO_DEPTH_TRIM_MM` | add to the plan's depth |

They shift the **planner's nominal**, so the baseline and the policy both get
them. A policy trained with a trim must be performed with the same trim.

They exist because editing constants between runs silently fails: a same-length
edit inside one second does not invalidate `__pycache__`, and several points of
a parameter sweep measured a stale value that way.

---

## Known limits

**The tone judge is out of distribution on short notes.** It reads +0.879
against human ratings on the sustained recordings it was trained on, and
inverts on 0.111 s notes — it scored a take bowing at the speaking floor 0.674
against 0.392 for a good baseline. `period_corr` inverts the same way. On a
piece that is mostly short notes, most of the reward cannot see the music, and
`r_speed_target` exists only to stop the resulting exploit.

If a piece is more than about half under 0.15 s, expect the run to stop the
policy doing the wrong thing rather than to teach it a better thing.

**Check contact force before trusting tone numbers.** `torque_contact_est_N`
should sit inside `dataset_a_final`'s 3.69–5.66 N. It fell to ~3.3 N on
2026-08-19 and every tone measurement that day was on a moving baseline.
Symptom: commanding more speed and depth makes the recording *quieter*, which
means the bow is not gripping — check hair tension and rosin, and do not try
to compensate in software.
