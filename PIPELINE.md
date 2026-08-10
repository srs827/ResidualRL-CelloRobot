# Classifier and RL loop structures

Reference for how the sound classifier and the RL loops actually fit together,
written to settle the recurring "which window size are we using?" confusion.

Last verified against the code on 2026-08-10.

---

## 0. The one-paragraph version

A frozen CNN listens to the cello and predicts what a human annotator would
rate the tone, 0–1. A reinforcement-learning policy plays a piece on the robot
and is rewarded by that score plus how accurately it hits the written dynamics.
The policy never drives the robot directly — a rule-based planner decides the
bowings, stroke lengths and timing, and the policy only nudges two numbers per
stroke: bow speed and bow depth.

---

## 1. Three different rates, constantly confused

Most of the disagreement in this project comes from collapsing three distinct
numbers into one. They are independent:

| Rate | What it means | Current value |
|---|---|---|
| **Analysis window** | How much audio the classifier scores at once | 500 ms |
| **Policy / setpoint rate** | How often the commanded (speed, depth) changes | once per stroke |
| **Motion control rate** | How often the robot controller gets a command | one `moveL` per stroke |

"Use 500 ms instead of 50 ms" is a statement about the **analysis window**.
"Don't re-command the robot every 50 ms" is a statement about the **setpoint
rate**. They are not the same argument and can be answered separately.

---

## 2. The classifier

`SoundClassifier/` — trained offline, then **frozen**. The RL never updates it;
no gradients flow through it. It is a reward function, not a learner.

### 2.1 Inputs

Three parallel inputs per scored window:

| Input | Shape | Source |
|---|---|---|
| Log-mel spectrogram | `(1, 64 mels, 87 frames)` | 500 ms of audio @ 44.1 kHz |
| Scalar audio features | `(40,)` | `audio_features.extract_scalar_features()` |
| Physical features | `(6,)` | robot state during the stroke |
| `window_pos` | scalar `[0,1]` | where the window sits in the stroke (0 = onset, 1 = release) |

The 40 scalar features are:

```
hnr_db_mean, hnr_db_std                      harmonic-to-noise (scratchiness)
flatness_mean, flatness_std                  tonal vs noisy
f0_mean_hz, f0_stability_cents, voiced_fraction
rms_mean, rms_std, envelope_cv,
envelope_trend, envelope_outlier_rate        loudness shape over the window
attack_time_s, attack_overshoot              onset behaviour
mfcc0..12_mean, mfcc0..12_std                timbre (13 MFCCs x mean/std = 26)
```

The 6 physical features, in `dataset.PHYSICAL_FEATURE_NAMES` order:

```
depth_or_force, force_deviation_or_zero, bow_speed,
bow_position, torque_or_lateral, torque_max_or_torque
```

Audio is **peak-normalised** before the model sees it, so absolute loudness is
deliberately invisible to the tone judgement. (Loudness is handled separately —
see §5.)

### 2.2 Architecture

`SoundClassifier/quality_classifier.py::MelQualityCNN`

```
mel (B,1,64,87) ──> conv stack ──> AdaptiveAvgPool2d((1, None))   # pool freq, KEEP time
                                        │
                                        └─> concat[ mean(t), std(t) ]  ->  (B,128)
                                                    │
scalar (B,40) ──> normalise ────────────────────────┤
physical (B,6) ─> normalise ────────────────────────┤──> fusion ──> heads ──> sigmoid
window_pos ─────────────────────────────────────────┘
```

**The trunk is length-agnostic.** It pools the time axis with mean+std, so it
produces a fixed 128-dim vector for *any* input length. `TARGET_FRAMES = 87` is
only used by `audio_to_mel_tensor()` to resize inputs for batching — it is a
convenience, not an architectural requirement. Variable-length windows are
therefore cheap on the model side (the expensive part is labels, see §2.5).

### 2.3 Outputs

- `predict(...)` -> single `overall` score in `[0, 1]`
- `predict_detailed(...)` -> all Tier-1 dimensions:
  `overall, tone_quality, attack_quality, release_quality, bow_control, dynamic_accuracy`

**What the number means.** The model regresses human ratings mapped by
`annotation_score_01 = (mean(overall) - 1) / 3`, where `overall` is a 1–4 scale.
So the score is a *predicted mean human rating*:

| score | ≈ rating |
|---|---|
| 0.20 | 1.6 / 4 |
| 0.50 | 2.5 / 4 |
| 0.80 | 3.4 / 4 |

### 2.4 Checkpoints

| file | labels | n | recording ρ | val MSE |
|---|---|---|---|---|
| `quality_cnn.pt` | **pseudo** (`pseudo_heuristic_v1`) | 500 | 0.99 | 0.0003 |
| `quality_cnn_human_current.pt` | human (2 annotators) | 349 | 0.702 | 0.0355 |
| **`quality_cnn_human_A1_A5.pt`** | **human (5 annotators)** | **500** | **0.798** | **0.0242** |

`quality_cnn.pt` is `classifier.py`'s `DEFAULT_CHECKPOINT` but is trained on
**pseudo-labels** — its near-perfect metrics are it re-learning a deterministic
heuristic, not human judgement. `rl/piece_env.py::RealScorer` therefore prefers
`quality_cnn_human_A1_A5.pt` and prints a warning if a pseudo-label checkpoint
is loaded.

Per bow config (A1–A5 model), recording-level ρ:

```
standard 0.847   <- the only config the RL plays in
bridge   0.804
board    0.641
botangle 0.614
topangle 0.614
```

### 2.5 Training

`train_classifier.py --meta <metadata.jsonl> --audio-dir <audio/>`

- Source: `dataset_a_final` — **500 sustained full-bow strokes**
- Each recording is sliced into 500 ms windows hopping 250 ms
  (`dataset.py: WINDOW_SEC = 0.5, TRAIN_HOP_SEC = 0.25`) → 7,280 training windows
- Every window inherits its parent recording's human rating
- Annotations are imported with `label_studio_bridge.py import`, which **mutates
  `--meta` in place** — always copy the pristine `metadata.jsonl` first

**Known distribution limit.** Every training window comes from a sustained
full-bow stroke. The model has never seen a short note in musical context, which
is why short notes score unreliably (§4.4).

---

## 3. The three RL loops

There are three, they differ, and this is the source of most cross-talk.

| Loop | Analysis window | Action rate | Status |
|---|---|---|---|
| `rl/piece_env.py` + `rl/piece_hardware.py` | **500 ms** | **one per stroke** | current, runs on the robot |
| `rl/env.py` + `constants.py` + `reward/` | 500 ms (buffered) | 50 ms tick | earlier single-string design |
| `other_github/rl/sac/cello_env.py` | **50 ms** | 50 ms tick, 150/stroke | partner's one-stroke prototype |

### 3.1 Why the "50 ms" confusion happens

`constants.py` says, in as many words:

```python
TIMESTEP_SEC     = 0.05    # 50ms per RL step
STEPS_PER_STROKE = 150
```

and `rl/env.py::_record_audio()` records exactly `TIMESTEP_SEC * 44100` = 2205
samples. Read on its own, that looks like a 50 ms analysis window.

It is not. `reward/sliding_window.py` buffers **ten** 50 ms chunks and only then
scores:

```python
SlidingWindowClassifier(window_sec=0.5, hop_sec=0.05)
# "Receives 50ms chunks, scores on 500ms windows"
```

So in that loop 50 ms is the *action* rate; the analysis window was already
500 ms.

The genuine 50 ms case is `other_github/rl/sac/cello_env.py`:

```python
audio_samples: int = 2205,   # ~50 ms @ 44100 Hz
...
audio = self._get_audio()
label, confidence = self.classifier.predict(audio, current_force=self._force)
```

— passed straight to `predict()` with no buffering. Note it also uses a
*different* classifier (`other_github/reward/real_classifier.py`, a DeepMLP over
7 scalar features), which degrades more gracefully at 50 ms than an 87-frame
spectrogram would.

`constants.py`'s `TIMESTEP_SEC` is unused by the current full-song pipeline and
is worth quarantining, because it keeps causing this misreading.

---

## 4. The current full-song RL loop

### 4.1 The three layers — only the middle one learns

| Layer | Role | Trained? |
|---|---|---|
| `BowPlanner` (`play_midi_pieces.py`) | bowings, stroke lengths, timing, bow budget | **No** — rule-based |
| **SAC policy** | shades speed & depth per stroke | **Yes** |
| Classifier | judges the resulting sound | **No** — frozen |

### 4.2 The loop, one iteration = one bow stroke

```
observe (18-dim)
   note duration, written dynamic, bow direction, current u,
   bow remaining, next note's dynamic/duration/direction,
   last stroke's quality, EMA quality, progress, previous action
        │
        ▼
actor samples a ~ pi(.|s)          a = (speed residual, depth residual), tanh -> [-1,1]
        │
        ▼
apply residual to the PLANNED stroke
   mean_speed x (1 + a0 * 0.20)        +/-20% ~ +/-1.6 dB
   depth      + (a1 * 0.5 mm)          clamped to [-1.5, +2.0] mm
   length clamped to remaining bow in [U_MIN=0.04, U_MAX=0.96]
        │
        ▼
robot plays it            one moveL (or blended path for a swell)
        │
        ├──> StateLogger @100 Hz  -> measured speed, torque, bow position
        └──> microphone           -> 500 ms window, peak-normalised
        │
        ▼
reward  (see 4.3)
        │
        ▼
store (s, a, r, s') in replay buffer;  4 gradient updates (real-robot setting)
```

Episode = one pass through the piece. `reset()` re-places the bow and clears
`u`, the quality EMA and the previous action.

### 4.3 Reward

```
total = 0.50 * quality_eff     tone
      + 0.25 * r_dynamic       dynamics
      + 0.15 * r_bow           bow budget (<= 0, penalty)
      + 0.10 * r_smooth        action jerk (<= 0, penalty)
```

with

```
quality_eff = 0.5 + fill * (quality - 0.5),   fill = min(1, duration / 0.5 s)

r_dynamic   = zone reward on MEASURED dBFS      (see 5)
              or, without a loudness model, clip(1 - |err_dB| / 3, 0, 1)
              where the target speed is capped at accel_max * T / 4

r_bow       = -clip(3 * shortfall, 0, 1) - edge penalty near U_MIN/U_MAX
r_smooth    = -0.5 * mean|action - previous_action|
```

Two corrections worth understanding, both found by reading a real baseline:

- **`fill`** — the classifier scores a fixed 500 ms window, but most notes are
  shorter, so most of the window is neighbouring notes. The tone term is shrunk
  toward neutral in proportion to how much of the window the note occupies. The
  raw `quality` is still logged; only the gradient is discounted.
- **achievable dynamic target** — covering length L in time T needs
  `accel >= 4L/T^2`, so the fastest possible mean speed is `accel_max * T / 4`
  (~0.08 m/s for an 0.08 s note). Grading against the written dynamic anyway
  penalised the policy up to **8 dB** for the planner's acceleration cap, which
  no residual could fix.

### 4.4 Why short notes are a problem

Not because short windows are inherently bad — because the window is **fixed at
500 ms while notes vary 18-fold** (0.08–1.5 s). For a note shorter than the
window, most of what is scored is not that note.

Share of the tone signal coming from notes long enough to fill the window:

| piece | full-weight notes | share of tone signal |
|---|---|---|
| t1 | 5 / 25 | **31%** |
| minuet_no_2v2 | 120 / 334 | 53% |
| batman | 218 / 278 | 86% |
| twinkle | 42 / 42 | 100% |

This also explains the baseline gap: twinkle's baseline tone is 0.796 and t1's
is 0.193, largely because twinkle is 100% in-distribution and t1 is 31%.

The principled fix is a **note-matched window**, `min(note_duration, 500 ms)`.
The model can already take variable lengths (§2.2); what is missing is human
labels for short notes and refitted feature normalisers.

### 4.5 SAC configuration

| | mock | real robot |
|---|---|---|
| `learning_starts` | 1000 | **200** |
| `gradient_steps` | 1 | **4** |
| checkpoint every | 5000 steps | **250 steps** |

Shared: `lr 3e-4`, `gamma 0.99`, `buffer 100k`, `batch 256`, `tau 0.005`,
`train_freq 1`, `ent_coef "auto"`.

Off-policy replay is what makes hardware RL feasible: 4,000 real strokes x 4
updates x 256 samples ≈ **4.1 M transition-samples from 4,000 real strokes** —
each stroke replayed roughly a thousand times. The replay buffer is saved with
every checkpoint, because it *is* the collected robot data.

---

## 5. Closed-loop dynamics

`rl/loudness.py` + `SoundClassifier/checkpoints/loudness_model.json`

Originally the dynamics reward compared measured **bow speed** to a target
speed — speed as a proxy for loudness. Now it measures loudness directly.

Fitted on the 100 standard-config recordings:

```
dBFS = 1.131 * 20log10(bow_speed) + 0.421 * depth_mm - 5.29
R^2 = 0.892,  residual sd = 1.14 dB
```

`a ≈ 1.13` confirms amplitude is proportional to bow speed; `b ≈ 0.42 dB/mm`
makes depth worth ~1 dB across the whole envelope, i.e. a timbral trim, not a
dynamic lever.

**Bow position is deliberately not a predictor** — every dataset recording is a
full-bow stroke, so its midpoint is 0.535 ± 0.0002. Including it gives a
degenerate fit (coefficient ~9800, predictions near -372 dBFS). This is a real
limitation: the RL plays partial strokes anywhere in u = 0.08–0.8, where leverage
genuinely differs, and that error is *not* in the 1.14 dB residual.

### Dynamic zones

Four equal 2.5 dB slices of the reachable ~10 dB span:

| zone | dBFS | speed at centre |
|---|---|---|
| p | -28.7 … -26.2 | 0.102 m/s |
| mp | -26.2 … -23.7 | 0.132 m/s |
| mf | -23.7 … -21.2 | 0.170 m/s |
| f | -21.2 … -18.7 | 0.220 m/s |

A stroke inside its zone scores 1.0; outside, reward falls linearly to 0 over
3 dB. Flat inside rather than peaked, because the model's own residual is
1.14 dB — rewarding sub-decibel precision would be rewarding noise.

`measure_dbfs()` must be given the **raw** capture: the classifier window is
peak-normalised, and normalising destroys exactly the level being measured.

**Caveat: dBFS is absolute**, so the zones are tied to this microphone, gain and
placement. Change any of them and the zones must be refitted or they are
silently wrong.

### Feed-forward calibration — implemented but NOT usable yet

`--calibrated-dynamics` inverts the model to re-aim each written dynamic at its
zone centre, widening p..f contrast from 3.5 dB to 6.7 dB. It also keeps depth on
the *original* written volume, so quiet notes are not dragged into the
slow-and-light corner where the string stops speaking.

It is off by default because it **wrecks the bow trajectory**: planned bow range
goes from u 0.221–0.730 to **0.083–0.710**, parking the bow near `U_MIN` where
tone is worst. Measured baseline tone fell 0.203 → 0.159. Fix the drift (a
partial, e.g. 50%, calibration is the obvious next thing to try) before enabling.

---

## 6. Results to date (t1, real robot)

| | tone quality | notes |
|---|---|---|
| Baseline (zero residual) | **0.203 ± 0.006** | 3 episodes |
| Tone-only policy | **0.479 ± 0.006** | 3 episodes, +136% |
| Dynamics-aware policy | 0.222 → **0.539** | *training curve only*, not yet evaluated |

Baseline re-measured after 40 min of continuous bowing was 0.203 vs 0.193
before, so instrument drift is ruled out.

**What the tone policy learned:** bow **faster and lighter** — speed residual
≈ +0.25, depth ≈ -0.23. That is a real cello principle (more speed, less
pressure gives a clearer tone), rediscovered from the classifier alone.

The gain is largest where the classifier is most trustworthy, which is the
pattern you want if it is genuine rather than the policy exploiting an
out-of-distribution region:

| note length | baseline | policy | gain |
|---|---|---|---|
| ≥0.5 s (trusted) | 0.297 | 0.665 | **+0.368** |
| 0.2–0.5 s | 0.186 | 0.462 | +0.275 |
| <0.2 s | 0.159 | 0.389 | +0.230 |

**Caveat:** the tone-only policy bought part of its gain by playing everything
louder, showing 7–11 dB errors once graded by the closed-loop zones. That is what
the dynamics-aware run was meant to prevent.

---

## 7. Open questions

1. **Note-matched windows.** Architecture supports it; labels do not. Cheapest
   next test: slice progressively shorter windows (500 → 250 → 100 → 50 ms) out
   of the *existing* 500 recordings and see whether scores stay consistent with
   the known label. If they hold, refit normalisers; if they collapse, the
   ~8,000 stroke recordings collected during training need annotating.
2. **Intra-stroke control.** Needs `servoL` streaming (absolute poses, so a
   stalled loop stops the bow rather than letting it run), with the policy and
   classifier running **asynchronously** — inference inside the control loop
   would freeze the bow every tick. Note the payoff is repertoire-dependent: at a
   500 ms setpoint rate a note needs ≥1 s for two decisions, and t1 has 2 of 25.
   `batman` (109/278) and `string_crossings` (13/14) are the pieces where it
   matters.
3. **`servoL` bypasses `safe_moveL`'s fault checking**, so equivalent guards are
   needed in the loop: clamp `u` every tick, fault check, defined stall
   behaviour.
4. **Repertoire.** `t1.mid` is the only piece in `MIDI-Files/` with written
   dynamics (range 0.43); every other file is 0.00. It is also the worst piece
   for classifier reliability (31% in-distribution). Those two facts pull in
   opposite directions when choosing what to train on.
