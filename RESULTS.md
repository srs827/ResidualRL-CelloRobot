# Results

Measured results from the residual-RL cello pipeline. Architecture is in
[PIPELINE.md](PIPELINE.md).

All robot numbers are on the **A string, `standard` bow config**, scored by
`quality_cnn_human_A1_A5.pt`. "Tone quality" is the classifier's predicted mean
human rating on a 0–1 scale, where `rating = 1 + 3 x score` on the annotators'
1–4 scale.

---

## 1. Classifier

Retrained on all five annotators (2026-08-10).

| | current (A1–A2) | **new (A1–A5)** |
|---|---|---|
| Recordings | 349 | **500** |
| Ratings | 399 | **999** |
| Annotators | 2 | **5** |
| Recordings with ≥2 raters | 50 | **499** |
| Recording-level Spearman ρ | 0.702 | **0.798** |
| Window-level ρ | 0.694 | **0.777** |
| Validation MSE | 0.0355 | **0.0242** |

Per bow configuration (recording-level ρ):

| config | old | new |
|---|---|---|
| **standard** (used by the RL) | 0.834 | **0.847** |
| bridge | 0.776 | **0.804** |
| botangle | 0.580 | **0.614** |
| board | 0.664 | 0.641 |
| topangle | 0.636 | 0.614 |

Built from the pristine `dataset_a_final/metadata.jsonl` into a new
`metadata_human_A1_A5.jsonl`; the previous metadata and checkpoint are
untouched.

**Caveat.** The two models were trained on different datasets (349 vs 500
recordings), so the validation splits differ. The gain is large and consistent
across window- and recording-level, but this is not a controlled A/B.

---

## 2. Baselines (zero residual)

| piece | tone quality | episodes | note |
|---|---|---|---|
| `twinkle_twinkle-open` | **0.796** | 1 | all notes ≥0.5 s |
| `t1` | **0.203 ± 0.006** | 3 | 84% of notes shorter than the classifier window |

The 4x gap is largely a distribution effect, not a musical one: twinkle is 100%
in-distribution for the classifier, t1 is 31% (see PIPELINE.md §4.4).

Baseline re-measured **after** 40 minutes of continuous bowing: 0.203, versus
0.193 before. Instrument drift is therefore ruled out as an explanation for the
policy gains below.

---

## 3. Run 1 — tone-only policy (t1)

4,000 strokes / 160 episodes / ~40 min on the robot. Reward: tone + open-loop
(bow-speed proxy) dynamics.

**Training curve:** first-10 avg 0.232 → last-10 avg 0.457

**Evaluation, deterministic, 3 episodes each:**

| | tone quality | ≈ human rating |
|---|---|---|
| Baseline | 0.203 ± 0.006 | 1.61 / 4 |
| **Policy** | **0.479 ± 0.006** | **2.44 / 4** |

**+136%, a 2.4x improvement.**

### What it learned

Averaged over the piece: **speed residual ≈ +0.25, depth residual ≈ −0.23** —
i.e. **bow faster and lighter**. That is a real cello principle (more bow speed
with less pressure gives a clearer, more resonant tone; slow-and-heavy crushes
the string), recovered from the classifier alone.

### Is the gain real, or reward hacking?

RL readily exploits out-of-distribution regions of a learned reward, and 69% of
t1's tone-reward mass comes from notes too short to fill the classifier's
window. Splitting the gain by note length:

| note length | baseline | policy | gain |
|---|---|---|---|
| **≥0.5 s (classifier trusted)** | 0.297 | 0.665 | **+0.368** |
| 0.2–0.5 s | 0.186 | 0.462 | +0.275 |
| <0.2 s | 0.159 | 0.389 | +0.230 |

The gain is **largest where the classifier is most trustworthy** and smallest on
the shortest notes. Reward hacking would show the opposite pattern.

### Known limitation

Graded afterwards by the closed-loop dynamic zones (§5), this policy shows
**7–11 dB dynamic errors**. It bought part of its tone gain by playing
everything louder — which the open-loop reward barely penalised.

---

## 4. Run 2 — dynamics-aware policy (t1)

Identical setup, except the dynamics term grades **measured dBFS against the
written dynamic's zone** instead of using bow speed as a proxy.

**Training curve:** first-10 avg 0.222 → last-10 avg **0.539**

Head-to-head against Run 1 at the same episode (both training curves):

| episode | run 1 (tone-only) | run 2 (dynamics-aware) |
|---|---|---|
| 30 | 0.293 | 0.293 |
| 60 | 0.237 | 0.243 |
| 90 | 0.332 | **0.356** |
| 120 | 0.350 | **0.427** |
| 150 | 0.426 | **0.527** |
| 160 (final) | 0.457 | **0.539** |

The two ran in lockstep for ~90 episodes, then Run 2 pulled ahead at every
subsequent checkpoint — consistent with the dynamics penalty blocking the "just
play louder" shortcut and pushing the policy toward genuinely better bowing.

> **NOT YET EVALUATED.** 0.539 is a *training* figure measured with exploration
> noise still active, and is not directly comparable to Run 1's evaluated 0.479.
> For reference, Run 1's training curve ended at 0.457 while its clean
> deterministic evaluation came out at 0.479. A robot evaluation of Run 2 is
> still outstanding.

---

## 5. Loudness model and dynamic zones

Fitted on the 100 `standard`-config recordings:

```
dBFS = 1.131 * 20log10(bow_speed) + 0.421 * depth_mm - 5.29
R^2 = 0.892,  residual sd = 1.14 dB
```

The speed exponent ≈1.13 confirms amplitude is proportional to bow speed;
depth at 0.42 dB/mm is worth ~1 dB across the whole envelope — a timbral trim,
not a dynamic lever.

Four zones, equal 2.5 dB slices of the reachable ~10 dB:

| zone | dBFS | speed at centre |
|---|---|---|
| p | −28.7 … −26.2 | 0.102 m/s |
| mp | −26.2 … −23.7 | 0.132 m/s |
| mf | −23.7 … −21.2 | 0.170 m/s |
| f | −21.2 … −18.7 | 0.220 m/s |

**Bow position is deliberately excluded** — every dataset recording is a
full-bow stroke, so its midpoint is 0.535 ± 0.0002. Including it produced a
degenerate fit (coefficient ~9800, predictions near −372 dBFS). The RL plays
partial strokes across u = 0.08–0.8, and that error is *not* captured in the
1.14 dB residual.

**dBFS is absolute**, so the zones are tied to this microphone, gain and
placement. Change any of them and the zones must be refitted.

---

## 6. Negative result — feed-forward calibration

Inverting the loudness model to re-aim each dynamic at its zone centre widens
p..f contrast from **3.5 dB to 6.7 dB**. It also measurably **hurts**:

| | bow range | baseline tone |
|---|---|---|
| uncalibrated | u 0.221 – 0.730 | **0.203** |
| calibrated | u **0.083** – 0.710 | **0.159** |

Cause: stroke length = speed x duration, so re-aiming speeds changes how much
bow each note consumes — asymmetrically.

| | down-bow mean | up-bow mean | net drift per pair |
|---|---|---|---|
| uncalibrated | 49.1 mm | 54.6 mm | −5.5 mm |
| **calibrated** | **37.6 mm** | 50.4 mm | **−12.8 mm** |

The bow walks toward the frog 2.3x faster and bottoms out near `U_MIN = 0.04`,
in the region where measured tone is worst (0.64–0.73 lower bow vs 0.87–0.93
upper). The planner's `BOW_SLACK = ±20%` cannot correct a 25% asymmetry, and
t1's gaps are too short for a retake to re-anchor.

`--calibrated-dynamics` is therefore **off by default**. Partial (e.g. 50%)
calibration, keeping the asymmetry inside the planner's slack, is the obvious
next thing to try.

---

## 7. Reproducing

Environment: `~/venvs/cello311` (outside iCloud sync; see the ur_rtde note
below). Robot at `192.168.1.100`, Focusrite Scarlett input — the channel is
**probed and proven live** at startup, because an empty socket returns a valid
stream of its own noise floor.

```bash
# baseline
python rl/play_piece.py MIDI-Files/t1.mid --real --episodes 3

# train
python rl/train_piece.py MIDI-Files/t1.mid --real --timesteps 4000 --save-stroke-audio

# evaluate a policy
python rl/play_piece.py MIDI-Files/t1.mid --real --episodes 3 \
    --model rl/checkpoints_piece/sac_piece_t1_final
```

**ur_rtde build note.** `pip install ur_rtde` fails against Homebrew's Boost
1.90 (`boost_system` config was dropped). Build against `boost@1.85`:

```bash
brew install boost@1.85
CMAKE_PREFIX_PATH=/opt/homebrew/opt/boost@1.85 pip install --no-cache-dir ur_rtde
```

---

## 8. Data not in this repo

Training runs produced **8,082 stroke recordings** (~307 MB) — 500 ms
peak-normalised windows, named `ep<episode>_s<stroke>.wav` so each maps to a
known note in the piece. They are gitignored and distributed separately.

They are the raw material for the open problem in PIPELINE.md §7: the classifier
needs a fixed 500 ms window, but most notes in real repertoire are shorter, and
no human has ever rated a short note. These recordings are short notes *in
musical context*, which is exactly what an annotation round would need.
