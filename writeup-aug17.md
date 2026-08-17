# writeup-aug17

Progress since August 14.

---

Upon training yunpiece.midi multiple times, each run 
produced policies that sounded worse than the baseline, both rhymically and tonally. 

---

## Failures 

### 1.1 The policy was based on delayed notes

The policy's main tool is shaping the bow within a note: start the stroke gently, fill out through the middle. To allow that, the system splits any note longer than 0.4 s into three consecutive segments, each with its own speed and
pressure.

Splitting a note changes how the robot is commanded. A single-segment note is sent as one simple `moveL`. A three-segment note has to be sent as a path that the controller blends through.

We measured, on the same piece:

| how the note was sent | how late it finished |
|---|---|
| single `moveL` (unshaped note) | **5.6 ms** |
| `moveL` path (shaped note) | **119.9 ms** |

**So every note the policy shaped ran ~120 ms late.** On a half-second note
that's 24%.

Two consequences:

- The rhythm breaks, audibly.
- The reward looks at the wrong audio. It grabs a window of sound for the note's
  expected duration, but the note is still sounding 120 ms later. So the loudness
  reading, the neural network's input, and the acoustic measurements were all
  taken from a note that had overrun its slot.

The policy therefore was being penalised for acting in each run.

Why 120 ms:

1. **~125 ms** — each segment inherited the note's overall acceleration, which
   for a long quiet note is the conservative default of 1.2 m/s². Changing bow
   speed at a junction at 1.2 m/s² takes ~42 ms, and there are three junctions.
2. **~4 ms** — the controller "blends" corners between waypoints so the bow
   doesn't stop. The blend was capped at 30% of a segment's length, giving
   ~5 mm on our segments. The code's own comment records 5 mm as too small to
   carry speed through a junction.
3. **~51 ms** — sending a *path* costs more than sending a single pose,
   regardless of what's in it. Confirmed independently by a standalone
   calibration.

Fixed all three. Shaped notes now finish **26 ms** late instead of 120 ms.

### 1.2 The reward doesn't properly reflect human opinion

I recorded the same piece twice in two configurations. The second configuration sounded much better by ear, especially for fast notes. However, the scores do not reflect this. 

| | take 1 | take 2 |
|---|---|---|
| **neural network score** | **0.474** | **0.473** |
| network on mid-piece excerpts | 0.524 | **0.406** ← backwards |
| harmonics-to-noise | 6.94 | 8.94 |

The network, half of the reward, scored two audibly different takes
identically, and on excerpts ranked them backwards.

The network was trained on 0.5-second windows of sustained bowing. Most
notes in this piece are 60–110 ms. When it's handed a short note it pads the
window out with whatever comes next, so it's partly judging the neighbours. It
was validated by slicing *clean sustained* recordings to different lengths and
checking it still ranked them correctly which it did. That's a fair test of
ranking clean material. It is not a test of hearing a real short note inside a
real passage.

So a policy maximising this reward was, on net, being paid to make the sound
worse.

### 1.3 Issue with r_speak

A bowed string only "speaks" above a minimum bow speed, measured on this instrument at
0.09 m/s (maybe verify this is the accurate minimum speed, not so sure...)

`r_speak` measures this parameter. It fired on **0%** of this
piece's notes because:

1. It only looked at notes that had been split into segments — and 72% of this
   piece's notes are too short to split, so it skipped them entirely.
2. It checked the speed we asked for, not the speed the note actually got. The
   planner shortens a stroke when a note is too short to cover the distance in
   time, and that shortening is what drives speed below the floor. 

---

## Part 2: Modifications

### 2.1 Feed the planner the .mxl score, not the .mid file 

We had been training from `yunpiece.mid`. There is also a `yunpiece.mxl`
(MusicXML) of the same piece sitting in the same folder.

**MIDI stores note pitches, start times, and a "velocity" number per note. It
has no way to express bowings, slurs, articulation marks, or written dynamics**

From MIDI the planner has to guess how loud each note should be from
velocity alone. It guessed too quiet, and quieter target means a slower bow and **shorter
stroke**, because stroke length is speed x duration.

| | from `.mid` | from `.mxl` |
|---|---|---|
| bow travel per note | **3.8 mm** | **11.3 mm** |
| median bow speed | 0.061 m/s | 0.102 m/s |

.mxl is much better in terms of result/sound quality. 

### 2.2 Make the reward agree with a listener

Four changes:

**a. Trust the neural network less on short notes.** Its opinion is now scaled
down in proportion to how much of its 0.5 s window the note actually fills. A
full-length note counts fully; the shortest notes now keep only 35% of the
network's vote, where before they kept 85%. (In code: `WINDOW_FILL_FLOOR`,
0.85 → 0.35.)

**b. Weight the acoustic measurements more.** They're what got the ordering right in testing. Raised from 15% to 35% of the reward. (`W_DEFECT`.)

**c. Drop a dead measurement.** One of the acoustic measures ("what fraction of
the note is voiced") read exactly 1.000 on every take. It was contributing a
constant one-fifth of that term and diluting the measures that do discriminate.
Weight set to zero.

**d. Add a measurement that works on short notes.** New file
`rl/harmonicity.py`. Three measures, none needing human labels:

- **`period_corr`** — does each cycle of the waveform look like the next one? 
- **`harmonic_ratio`** — what fraction of the note's energy sits on the harmonic
  series of the fundamental?
- **`onset_periods`** — how many cycles before the waveform settles into
  repeating? 

The existing acoustic measures use an FFT with a 23.2 ms window, and this piece's fast notes are 8–23 ms — the window was mostly zero-padding, so those measures were literally analysing
silence. The new ones sidestep that because we already know the frequency: the
robot only plays the open A, measured at 220.5 Hz. When you know the frequency
in advance you can test for it directly instead of computing a whole spectrum
and looking for peaks, and that has no window-size floor.

Only `period_corr` is wired into the reward. `harmonic_ratio` went the wrong
way on long quiet notes, which looks like it penalising soft rather than bad,
so it's recorded but not scored.

Maybe explore further short-note metrics??

The existing acoustic measures now shrink their FFT window to fit short notes, so they analyse signal instead of silence. (The `n_fft=1024 is too large for input signal` warnings occurred before and are now avoided). 

### 2.3 Make the speaking-threshold term actually work

Both bugs in Section 1.3 fixed: it now checks every note, and it checks the speed the
note actually got.

It's now split in two, because the two halves differ in whether the policy can
do anything about them:

- **Priced (weight 0.35):** a note where one *segment* is too slow. The policy
  controls the segment shape, so it can fix this — and the old policy was doing
  this on every long note and getting away with it.
- **Recorded but not priced:** a note whose overall speed is too slow. This
  comes from the planner shortening the stroke, which no policy action can undo.
  Charging for it would just be a constant penalty the policy can't reduce.

### 2.4 Give the policy back its full range

The planner has a ceiling on how hard the arm may accelerate, set conservatively
at 4.0 m/s². The policy can ask for up to 35% more bow speed — but asking for
more speed means covering more distance in the same time, which needs more
acceleration, and the planner was quietly clipping the request at the ceiling.

| ceiling | slowest the policy can ask for | fastest | usable range |
|---|---|---|---|
| 4.0 | 0.0665 | **0.1109** | 0.0445 |
| **6.0** | 0.0665 | **0.1381** | 0.0716 |

**At 4.0 the policy was losing 46% of its upward speed range** — and speed is
the main lever for both loudness and tone. All three training runs ran with half
the range silently unavailable. Raised to 6.0, where it saturates.

We also decoupled the reward from this ceiling. The penalty for a harsh attack
was being scaled by the acceleration limit, so raising the limit silently made
harsh attacks ~4× cheaper. 

### 2.5 Fix the playback path

Separately from training, the code used to perform a finished piece had its
own rhythm problem.

The original player (`play_midi_pieces.py`) sends **one simple `moveL` per
note** and waits until each note's written start time before sending it. It
tracks to 3 ms per note.

The newer playback code (`perform.py --compile`, from the takeover branch)
batches many notes into a few blended paths. That was done for tone — it
removes the ~64 ms of dead air where the bow sits motionless on the string
between commands, which is what makes note-by-note playback sound scratchy. But
it gave up per-note timing, and nothing held the beat between batches.

`perform.py --render baseline` now routes through the original player's loop.

Along the way, two more bugs in the batched path:

- **Staccato was being deleted.** The score marks 153 notes staccato, with a
  14 ms silence after each. The batching only started a new command for gaps
  over 20 ms, so all of these fell through and the notes were run together. Each
  note landed 11% early, and after ~16 notes the accumulated rush was dumped as
  one audible hole — seven holes, matching seven batch boundaries exactly.
- **Unshaped notes collapsing.** With no policy corrections, a note
  split into three segments has three *identical* segments — the same motion as
  one segment — but it still got sent as a path and still paid the 120 ms.
  Collapsing those took drift from 1.67 s to 0.20 s.

---

## Part 3: Results

**Timing** (how late each note lands, average across the piece):

| | before | after |
|---|---|---|
| unshaped playback | 7 ms | 7 ms |
| policy playback | **318 ms** | **36 ms** |
| shaped notes specifically | **120 ms** | **26 ms** |

**Does the reward now agree with a listener?** I recorded five takes of the
same piece, identical except for one constant nudge applied to every note, and
had them ranked by ear:

| take | what was changed | listener rank | reward score |
|---|---|---|---|
| D | +35% bow speed | **1st** (clear winner) | 0.1084 |
| B | maximum bow pressure | 2nd | 0.1066 |
| A | nothing — the planner's plan | 3rd | 0.0872 |
| C | minimum bow pressure | 4th | 0.0797 |
| E | −35% bow speed | **5th** (worst by far) | **0.0071** |

**The reward reproduces the ranking exactly** (Spearman +1.00).

Two keys:

- **Doing nothing is not optimal.** The planner's own plan ranks third of five.
  More speed *and* more pressure both improve the tone, by ear and by reward.
  There is a real gradient for a policy to climb.
- **It explains the old failures.** The previous policy had learned to *reduce*
  bow speed. 

---

## Part 4: Unresolved Issues

1. **The reward was validated using one constant nudge applied to every note.**
   A real policy varies its decision note by note. We've shown the reward points
   in the right *direction*; we have not shown it behaves sensibly under
   fine-grained variation.
2. **We never measured whether the reward's signal exceeds its own noise.** The
   tool exists (`rl/reward_noise.py`) but has never produced a full reading. If
   replaying the *same* action gives rewards that vary more than *different*
   actions do, the gradient is noise and no architecture change helps.

**Bugs:**
- Retakes (repositioning the bow) can't be started early in the new playback
  path, because the stroke object passed to the player doesn't carry how long
  the retake needs. Probably part of the residual timing error.
- `play_midi_pieces.py --yes` is accepted but never checked, so the script
  always stops for a keypress.

**Position-Adjustment gap:**
- There is no control over where between the bridge and the fingerboard the
  bow contacts the string. The robot has no such axis, which is why "play fast quiet notes with
  good tone" has no clean solution here. Adding it means a mechanical change, a
  recalibration, and a new action dimension.

**Discard the existing policies.** All three runs learned through the broken
reward and the 120 ms penalty. They should be thrown away.

---

## Part 5: Training Note

Running `rl/reward_noise.py` to completion first would cost ~20 minutes of robot
time and would tell us whether the loop is learnable. Do this before training for 2 hours.

### Commands

```bash
# 1. gain calibration — the mic chain drifted 2.97-4.39 dB over one day
caffeinate -dims ~/venvs/cello311/bin/python rl/calibrate_gain.py \
  --measure --real --write --piece MIDI-Files/t1.mid

# 2. train — .mxl, NOT .mid.  182 notes x 100 passes, ~1.8 h
caffeinate -dims ~/venvs/cello311/bin/python -u rl/train_piece_logged.py \
  SoundClassifier/Data_Collection/public_annotation_packages_a_final/yunpiece.mxl \
  --real --timesteps 18200 --ckpt-every 10

# 3. pick the best checkpoint, then play it
RUN=$(ls -td rl/checkpoints_piece/run_* | head -1)
caffeinate -dims ~/venvs/cello311/bin/python rl/select_best.py "$RUN" \
  --piece SoundClassifier/Data_Collection/public_annotation_packages_a_final/yunpiece.mxl \
  --real --episodes 2
printf '\n' | caffeinate -dims ~/venvs/cello311/bin/python -u rl/perform.py \
  --real --compile --render baseline "$RUN/sac_piece_best.zip" \
  SoundClassifier/Data_Collection/public_annotation_packages_a_final/yunpiece.mxl
```

**Watch the first few episodes.** The dynamics score and in-zone count should
start near 0.70 and 60%. If they read 0.000 and 0, stop — that means the
loudness calibration is wrong, not that training is going badly.

**If it dies partway** (an emergency stop and an Ethernet dropout both happened
in one afternoon), compute the resume arguments from the run rather than copying
them from anywhere. A hardcoded resume command silently rewound three separate
runs back to the same early checkpoint and discarded ~45 minutes of robot time.
The episode-offset argument only relabels the log, so it looks like it's
continuing even when the weights have been reset.

---

## Files changed

**Reward and environment** (`rl/piece_env.py`) — network downweighted on short
notes, acoustic measures upweighted, dead measure zeroed, `period_corr` added,
speaking-threshold term fixed and split, FFT window fitted to short notes,
per-segment acceleration raised, path-command cost compensated, attack penalty
decoupled from the acceleration ceiling.

**Playback** (`rl/perform.py`, `BaselineControls/play_midi_pieces.py`) —
`--render baseline` routes through the original per-note player; staccato gaps
honoured; flat envelopes collapsed; blend widened; audio recording wired up
(it was silently writing to an empty channel).

**New tools** — `rl/harmonicity.py` (short-note tone measures),
`rl/calibrate_timing.py` (measures how long the robot actually takes versus what
was asked), `rl/reward_noise.py` (reward signal-to-noise),
`rl/ab_compare.py` (back-to-back playback comparison).

**New diagnostics** — every take now saves the bow's own motion at 100 Hz.
Reading rhythm from the bow's direction reversals is exact; reading it from
audio invented 80–107 false note onsets per take. Note that the old
whole-piece `tempo_ratio` cannot detect a rhythm problem at all — it read 1.002
across four takes whose fast passages differed by 5%, because a rush and the
pause after it cancel out.
