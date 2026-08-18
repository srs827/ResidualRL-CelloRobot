# writeup-aug18

Progress since August 17. One training run completed (103 episodes), and the
reason it failed the same way the previous three did was found and fixed.

---

## Part 1: The headline

**The trained policy learned to bow onto the speaking floor, exactly like the
three runs before it. The cause was one line, and it was not the reward.**

`ENVELOPE_MIN_DURATION` is 0.40 s. 86% of yunpiece is under 0.15 s. So
`_build_exec_stroke` used the mean of the three segment residuals on notes
long enough to shape, and **segment 0 alone** on every other note:

```python
speed_scale = 1.0 + (spd_res.mean() if use_envelope else spd_res[0]) * SPEED_RESIDUAL_FRAC
```

Dim 0 was therefore doing two incompatible jobs: the gentle START of a swell
on 14% of notes, and the WHOLE SPEED of the other 86%.

The policy learned `seg0 = -0.95` (54% of strokes pinned at the bound). As an
attack that is musically right — it built a genuine messa di voce, slow/light
→ faster/heavier → light again, which is the "character" a listener heard and
liked. As a whole-note speed it is ruinous: `1 - 0.95*0.35 = 0.67`, taking
0.1380 m/s to **0.092**, onto the 0.09 speaking floor, across most of the
piece. Measured on the take: median 0.0917 m/s.

Depth had the identical bug and drove short notes to **-1.19 mm** against the
planner's -0.45, closing on the -1.5 limit. Slow AND light is the corner
`expand_dynamic_range()` documents as "thin and unsteady rather than soft".

Both now use the mean. Verified on the existing checkpoint, no retraining:

| | speed (med) | depth (med) |
|---|---|---|
| baseline | 0.1380 | -0.450 mm |
| policy, seg0 coupling | 0.0917 | -1.193 mm |
| policy, decoupled | **0.1254** | **-0.768 mm** |

0.1254 sits inside the human-rated sweet band; 0.0917 was below its worst
point. The swell on long notes is untouched.

**Effect on the recorded takes** (same weights, one line changed):

| | drift mean | period_corr (fast notes) |
|---|---|---|
| policy before, tempo 1.00 | 66.9 ms | 0.732 |
| policy after, tempo 1.00 | **25.0 ms** | **0.882** |
| policy before, tempo 0.90 | 185.0 ms | — |
| policy after, tempo 0.90 | **30.8 ms** | 0.819 |

Baseline at tempo 1.00 is 7.2 ms / 0.886. The policy went from clearly worse
than baseline to matching it on tone while keeping its dynamic shaping.

---

## Part 2: The deeper problem — the reward cannot see this repertoire

The coupling bug explains the magnitude. It does not explain why the policy
wanted to go slow at all. That is the reward, and it is a regime problem.

### 2.1 Both tone terms are validated on sustained strokes and invert on short ones

Measured against `dataset_a_final`'s 500 human-annotated recordings, and
against blinded listener rankings collected today:

| term | sustained 3.2 s strokes | yunpiece 0.111 s notes |
|---|---|---|
| CNN (`W_QUALITY` 0.50) | **+0.879** vs human tone_quality | scored a 0.0917 m/s take **0.674** vs **0.392** for the 0.1380 baseline |
| `period_corr` (in `W_DEFECT`) | **+0.779** vs human tone_quality | **-0.600** vs the listener's ranking |

The CNN is *excellent* where it was trained. On the annotated set it prefers
faster bow, monotonically, in agreement with the labels:

| commanded speed | human tone_quality | CNN |
|---|---|---|
| 0.09 | 2.405 | 0.323 |
| 0.12 | 2.795 | 0.411 |
| 0.15 | **2.810** | 0.482 |
| 0.20 | 2.795 | 0.561 |
| 0.25 | 2.645 | 0.564 |

`corr(CNN, human tone_quality) = +0.879` in the rig's own standard config,
where `corr(human tone_quality, speed) = +0.756`.

But on 0.111 s notes the same judge pays for bowing to the floor. With
`W_QUALITY` 0.50 and `W_DEFECT` 0.25, **three quarters of the reward is out of
distribution on 86% of this piece**, and reweighting between two terms that
are both wrong there cannot fix it. Most of a day was spent doing exactly
that before the regime split was visible.

### 2.2 Fix: a physical prior where the judge is blind

New term `r_speed_target` rewards the 0.12–0.20 m/s band the annotations show
as a plateau, falling off over 0.04 m/s outside — a deadband like the loudness
zones, for the same reason. It is weighted by `(1 - fill)`: full authority on
the shortest notes, **zero** on notes long enough that the CNN was validated
on them and prefers a faster bow unaided.

`W_SPEED_TARGET = 0.20`, calibrated against the exploit it closes: the CNN
paid +0.28 for the floor, worth +0.14 at `W_QUALITY` 0.50; at 0.0917 m/s the
term reads 0.29, so leaving the band costs 0.71*W = 0.14.

### 2.3 Earlier rebalance, superseded in part

Before the regime split was understood, three weights moved on the evidence
then available: `period_corr` good bound 0.95 → 0.88 (saturating it across the
fast-passage speaking range so it acts as a floor guard), `W_DEFECT` 0.35 →
0.25, `WINDOW_FILL_FLOOR` 0.35 → 0.55. Those still stand, but §2.2 is the
change that addresses the cause rather than the symptom.

---

## Part 3: Calibration

### 3.1 The gain offset was measured on a piece that cannot calibrate it

`+8.04 dB` had been measured on `string_crossings.mid`: **14 notes, all
exactly 0.999 s, one written dynamic**, planned speed 0.1680–0.1831 m/s,
depth constant at +0.536 mm. That is 0.7 dB of the model's speed axis, one
operating point sampled 14 times. Its tight sd (0.26) was homogeneity, not
precision. It also puts every note in a single zone, so `calibrate_gain`'s
cross-zone `--write` guard **could not fire**.

Re-measured on yunpiece — the only one of the three candidate pieces where
that guard is live (p:166, f:16):

```
gain offset = +5.06 dB   (robust sd 2.63, n 182)
  zone f  : +5.41 dB  (n 16)
  zone p  : +4.88 dB  (n 166)          spread 0.53 dB
```

`+5.22` measured on t1.mid three days earlier agrees to **0.16 dB**.

### 3.2 Note duration is the largest factor missing from the loudness model

Regressing (measured − model) on log duration, speed and depth across three
pieces and 402 strokes (random actions, so speed and depth are decorrelated
from the score): **+7.73 dB per decade of note length**.

| duration | n | mean residual |
|---|---|---|
| 0.05–0.15 s | 188 | **-2.25 dB** |
| 0.15–0.30 s | 83 | -0.44 dB |
| 0.30–0.60 s | 33 | +1.64 dB |
| 0.60–1.10 s | 91 | +2.88 dB |
| 1.10–2.00 s | 7 | +2.89 dB |

Per-piece slopes agree (yunpiece +6.99, t1 +8.30; string_crossings is
single-duration and pins the intercept at +2.93).

Correcting for it collapses the cross-piece gain offset spread from **1.87 dB
to 0.71 dB**, with yunpiece and string_crossings — whose median durations
differ 9× — landing within 0.01 dB of each other.

**Not shipped.** A knee at `_window_bounds`' 0.6 s branch switch scores *worse*
than pure log-linear (R² 0.364 vs 0.382), t1 does not fully collapse (-0.95 dB
residual), and overall scatter only falls 3.30 → 2.64 dB against the model's
own 1.14. The form is unvalidated in the 0.15–0.6 s band, where only 33 of 402
strokes land. Documented in `rl/loudness.py`.

Why it matters: zones are 2.5 dB wide, 70% of yunpiece is under 0.15 s, and
those notes are graded ~2 dB too quiet. It is a **bias, not noise**, so it
never appears in a repeat-sd measurement — `reward_noise` would keep passing
while this stays broken.

---

## Part 4: ACCEL_MAX, and a method that actually works

### 4.1 The sweep

`ACCEL_MAX` had been raised 4.0 → 6.0 on 8/17. A listener reported the
baseline sounding "bouncy" and worse than the previous day. Initial A/B takes
suggested 4.0 was better — but that comparison moved **three variables at
once** (length 12.31→15.31 mm, speed 0.1109→0.1380, accel 3.60→5.03) and was
run in a fixed order with settings announced.

`rl/param_sweep.py` was written to do it properly: shuffled order, `--reps`
replication, blinded take ids, and a key it refuses to print until `--decode`.
It separates the two clean single-variable axes an offline grid revealed —
above `ACCEL_MAX` 5.0 the planned length and speed are *identical* and only
commanded acceleration moves.

**15 blinded takes, 5 conditions, 3 reps, listener-ranked:**

| condition | listener ranks | period_corr |
|---|---|---|
| a5.0 t1.0 | 4, 12, 13 | 0.911 |
| a5.5 t1.0 | 2, 7, 9 | 0.892 |
| a5.5 t1.1 | 1, 8, 10 | 0.890 |
| a5.5 t1.2 | 3, 6, 11 | 0.894 |
| a6.0 t1.0 | 5, 14, 15 | 0.834 |

```
repeat sd 0.0040   condition spread 0.0295   SNR 7.45
permutation test on condition mean ranks: p = 0.54
```

**Acceleration and length are inaudible on this piece.** period_corr separates
the conditions at 19 repeat sds and predicts the listener not at all. Session
drift came out slightly *negative* (-1.68 milli-pc/take), so the monotone rise
across the earlier unrandomised takes was the conditions, not warm-up —
randomising is what made that answerable.

Settled at **5.5**: the smallest ceiling that gives the residual any upward
speed authority at all. Below 5.0 the plan and the residual are capped at the
same place, so `v(a=+1) == v(a=0)` and the policy's speed lever only points
*down* — the direction writeup-aug17's listener test ranked worst of five.

### 4.2 A speed axis, and the finding that led to Part 2

All five conditions above planned an identical 0.1380 m/s. A `--axis speed`
sweep via `--fixed-action` found the listener preferring **faster** (mean rank
2.67 vs 4.33, p = 0.20, n = 6) while period_corr preferred **slower** by
0.042 — ten repeat sds. That inversion is what sent the investigation to the
annotated dataset and Part 2.

---

## Part 5: The reward_noise gate, run to completion for the first time

```
stroke  36:  total SNR 8.12   usable
stroke  90:  total SNR 1.28   weak
stroke 145:  total SNR 0.69   NOISE — action invisible
mean 3.37 → "training is worth the hardware time"
```

**The headline is `np.mean` of three ratios and is carried by the first.**
Median 1.28, one stroke below the tool's own action-invisible line. Same shape
as the partial reading that morning (5.07 / 0.67 / 0.95 → 2.23).

Underneath, repeat sd grows monotonically through an episode while the action
signal stays flat:

| repeat sd | stroke 36 | stroke 90 | stroke 145 |
|---|---|---|---|
| total | 0.0115 | 0.0725 | 0.1278 |
| r_dynamic | 0.0866 | 0.2757 | 0.4497 |
| r_defect | 0.0074 | 0.0721 | 0.1249 |
| **action sd (signal)** | 0.0932 | 0.0931 | 0.0888 |

Cause unknown. It is a between-replay effect and cannot be diagnosed from a
single episode; within one episode the loudness residual does not grow with
stroke index (corr 0.008), and bow position does not account for it either.

Saved as `rl/reward_noise_20260818.json`.

---

## Part 6: Things measured wrong today, and why

Recorded because the traps are re-derivable and cost hours.

**The analysis window does NOT drift.** Two separate readings said otherwise —
"a median 40 ms early", "-0.50 ms/stroke drift, ~72 ms by stroke 145". Both
were artifacts of **audio-energy onset detection**, which writeup-aug17
already records inventing 80–107 false onsets per take. A fix built on them
measured neutral and was reverted; it was solving a non-problem. Matching
strokes to bow **direction reversals** instead: median **-0.9 ms**, drift
-0.02 ms/stroke, corr(offset, stroke index) -0.062.

**A second detector trap.** First-threshold-crossing of `|bow_speed|` also
fails: through a continuous passage the bow never stops between notes, so the
first sample above threshold belongs to the *previous* stroke and reports a
spurious ~-94 ms that looks like a step change partway through the piece (it
is just where the piece stops having rests). `bow_speed` is **signed** — use
sign changes.

**`perform.py --render baseline` without `--compile` does nothing.**
`--render` is documented as "how `--compile` dispatches". Omitting `--compile`
falls through to the stroke-by-stroke live path, which leaves ~64 ms of dead
air with the bow motionless on the string between commands — writeup-aug17
§2.5's documented cause of scratchy note-by-note playback. Two A/B takes were
run this way and produced a spurious "611 ms mean slip, 157/182 notes late"
alongside audible crunchiness, both attributed to the instrument before the
command was found.

**Stale bytecode silently defeats source-edited sweeps.** A same-length edit
to a constant inside one second does not invalidate `__pycache__`, so several
points of a parameter sweep measured a stale value. `CELLO_ACCEL_MAX` now
overrides `ACCEL_MAX` per process for this reason, and `param_sweep --mock`
verifies the override reached the planner for every condition before recording.

---

## Part 7: Unresolved

1. **The 3.9 dB level jump is unexplained.** Today's takes record ~-18.0 dBFS
   against 2026-08-17's -21.9. period_corr improves with SNR, so *any*
   cross-day tone comparison is untrustworthy until this is understood. Check
   whether the Scarlett's gain was touched.
2. **`reward_noise` repeat-sd growth** (11× through an episode, Part 5).
   Needs repeated trials at several stroke indices.
3. **The duration correction is documented, not shipped** (§3.2). Needs points
   in the 0.15–0.6 s band, where only 33 of 402 strokes land.
4. **`attack_overshoot` and `f0_stability_cents` are unvalidated** against
   human ratings. The labels are already collected; this is free.
5. **`select_best` chose `sac_piece_yunpiece_final.zip`** over every numbered
   checkpoint (return 49.45 / tone 0.498 vs ckpt_ep0100's 45.28 / 0.433, n=4).
   `ab_compare`'s docstring says `final` should not be the performance
   checkpoint. The measurement disagreed with the heuristic; worth knowing
   which to trust.
6. **The policy still adds 14–20 ms of drift** at every tempo — the
   `moveL(path)` cost on the 14% of notes it shapes. That is the price of the
   swell.

**Bugs found, not fixed:**
- `ab_compare.py`'s summary table reports "(no wav written)" and blank scores
  for both passes even though both recorded correctly. Its stdout parsing is
  broken; all six documentation takes had to be scored by hand.
- `train_piece_logged.py` ignores `--save-dir` and always writes its own run
  directory under `rl/checkpoints_piece/`, so mock runs land next to real ones.
- `driver_eval.py:138` constructs a `HardwareExecutor` inside its grid loop.
  Since the 8/18 audio fix made `close()` stop closing the stream, an N-cell
  sweep now leaks N open input streams.

---

## Part 8: What to do next

1. **Retrain with the Part 1 and 2 changes.** Every checkpoint before today is
   mismatched — `ACCEL_MAX`, the gain offset, the action semantics and the
   reward have all moved.
2. **Consider proving the loop on `twinkle_twinkle-open` first.** It measured
   0.796 baseline tone with all-0.5 s notes, inside the judge's distribution.
   yunpiece is 86% notes shorter than anything the judge was trained on, and
   §2 is a statement that the reward cannot see most of it. A run that works
   there would separate "the pipeline is sound" from "this piece is hard".
3. **Annotate short-note recordings and retrain the judge.** This is the only
   thing that makes the tone reward real on this repertoire. The pipeline
   exists (`label_studio_bridge`).
4. **Explain the 3.9 dB before trusting any cross-day comparison.**

---

## Files changed

**Reward and environment** (`rl/piece_env.py`) — short notes decoupled from
segment 0 in both speed and depth; new `r_speed_target` physical prior
weighted by `(1 - fill)`; `period_corr` good bound 0.95 → 0.88; `W_DEFECT`
0.35 → 0.25; `WINDOW_FILL_FLOOR` 0.35 → 0.55; aiming mode and gain offset
printed at startup.

**Loudness** (`rl/loudness.py`) — `gain_offset_note` loaded so provenance can
be printed; duration-dependent bias documented.

**Calibration** (`SoundClassifier/checkpoints/loudness_model.json`) — gain
offset +8.04 → +5.06, measured on the training piece.

**Planner** (`BaselineControls/play_midi_pieces.py`) — `ACCEL_MAX` 6.0 → 5.5
with the sweep recorded; `CELLO_ACCEL_MAX` env override.

**Training** (`rl/train_piece.py`, `rl/train_piece_logged.py`) —
`--resume-run` prints the episode number it actually logs; 100 Hz bow motion
saved per episode as `ep####_state.npy`.

**Playback** (`rl/perform.py`) — `--only depth|speed` ablates half the
envelope action with no retraining.

**New tools** — `rl/param_sweep.py` (blinded, replicated, randomised
parameter sweeps with repeat-sd and drift estimation).

**Data** — `rl/reward_noise_20260818.json`; `rl/sweeps/` gitignored.
