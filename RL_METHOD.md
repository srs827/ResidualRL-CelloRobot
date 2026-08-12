# Residual RL for expressive cello bowing — technical description

Written as a methods reference: every quantity below is the value in the code,
not an illustrative one. Companion documents are [PIPELINE.md](PIPELINE.md)
(system architecture, the three RL loops in the repository, the classifier) and
[RESULTS.md](RESULTS.md) (measured outcomes).

Primary sources: `rl/piece_env.py`, `rl/piece_hardware.py`, `rl/loudness.py`,
`rl/train_piece.py`, `BaselineControls/play_midi_pieces.py`.

---

## 1. Problem formulation

A musical score is executed on a UR5e bowing the open A string of a cello. A
rule-based planner converts the score into a sequence of bow strokes; a learned
policy applies bounded corrections to each stroke; a frozen neural sound-quality
model supplies the reward.

The task is a finite-horizon MDP, one **episode per piece** and one **timestep
per bow stroke**:

- **State** `s_t ∈ ℝ^18` — score context, bow state, recent quality (§3)
- **Action** `a_t ∈ [-1,1]^6` — a per-segment bowing envelope (§4)
- **Transition** — the robot physically executes the stroke; the next state
  follows from the score and the achieved bow position (§5)
- **Reward** `r_t ∈ ℝ` — weighted sum of tone, dynamic accuracy, bow-budget
  and smoothness terms (§6)
- **Horizon** — the number of strokes the planner produces (14 for
  `string_crossings`, 25 for `t1`, 182 for `challengepiece`)

The formulation is **residual**: the policy never generates a trajectory. The
planner fixes bow directions, note onsets and note durations; the policy shades
speed and depth within bounded limits. A uniformly random policy therefore still
plays the piece recognisably, which is what makes on-hardware training feasible
at all.

### 1.1 Three layers, one learned

| Layer | Responsibility | Learned? |
|---|---|---|
| `BowPlanner` | bowings, stroke lengths, bow budget, timing | No — rule-based |
| **SAC policy** | per-segment speed and depth residuals | **Yes** |
| `BowingQualityClassifier` | tone judgement (reward) | No — frozen |

No gradient flows through the classifier; SAC treats it as a black-box reward.

---

## 2. Physical parameterisation

Bow position is a scalar `u ∈ [0,1]`, the fraction along the taught frog→tip
line. Poses are

```
pose(u, d) = apply_depth( lerp(FROG, TIP, u), CFG, d )
```

with translation a straight lerp and rotation a lerp **after canonicalising the
tip's axis-angle vector into the frog's hemisphere**. The taught frog and tip
rotation vectors sit in opposite hemispheres of the axis-angle double cover
(|r| ≈ 3.136 and 3.097, both just under π); a naive lerp passes through r ≈ 0
and produces a 178.5° wrist flip at the stroke midpoint. Canonicalised, the
lerp deviates < 0.0021° from the geodesic.

Constants (from `dataset_a_final` calibration and the taught waypoints):

| Symbol | Value | Meaning |
|---|---|---|
| `BOW_LENGTH` | 0.5249 m | frog→tip distance |
| `U_MIN, U_MAX` | 0.04, 0.96 | usable bow, hard margins |
| `SPEED_MIN, SPEED_MAX` | 0.09, 0.25 m/s | calibrated bow-speed grid |
| `MOVE_ACCEL` | 1.2 m/s² | pendant-verified nominal |
| `ACCEL_MAX` | 4.0 m/s² | planner ceiling |
| depth envelope | −1.5 … +2.0 mm | signed, − is lighter |

**Loudness is bow speed.** Over the calibrated grid, radiated amplitude is
proportional to bow speed: 20·log₁₀(0.25/0.09) = 8.87 dB predicted against
8.84 dB measured. Depth spans < 1 dB over the entire ±2 mm envelope and is
therefore a timbral parameter, not a dynamic one.

### 2.1 Feasibility constraint

A stroke starting and ending at rest follows at best a triangular velocity
profile, so covering length `L` in time `T` requires

```
a_required = 4L / T²
```

and the fastest achievable **mean** speed for a note of duration `T` is

```
v_max(T) = a_max · T / 4
```

For `a_max = 4.0`, a 0.08 s note tops out near 0.08 m/s regardless of the
dynamic marked. `solve_stroke()` shortens the stroke rather than missing the
beat, and §6.2 grades against `v_max(T)` rather than the written dynamic — a
distinction worth ~8 dB of spurious penalty on short notes.

---

## 3. State space (18 dimensions)

Normalised to approximately [−1, 1]:

| idx | quantity | normalisation |
|---|---|---|
| 0 | planned note duration | ÷ 2 s |
| 1 | planned mean speed | ÷ `SPEED_MAX` |
| 2 | planned depth | ÷ `DEPTH_HI` |
| 3 | bow direction | +1 down, −1 up |
| 4 | current bow position `u` | already [0,1] |
| 5 | available bow in this direction | fraction of usable range |
| 6 | written dynamic (volume target) | [0,1] |
| 7 | volume at note end (swell) | [0,1] |
| 8 | stroke begins with a retake | binary |
| 9 | stroke is part of a split note | binary |
| 10 | next stroke's volume | [0,1] |
| 11 | next stroke's duration | ÷ 2 s |
| 12 | next stroke's direction | +1/−1/0 |
| 13 | previous stroke's tone score | [0,1] |
| 14 | EMA of tone this episode | α = 0.7 |
| 15 | progress through the piece | [0,1] |
| 16–17 | previous action (mean speed, mean depth) | [−1,1] |

Indices 10–12 provide one-step lookahead; 13–15 provide history. Both are
needed because bow budget and phrasing are sequential: bow consumed now
constrains what is playable several notes later.

---

## 4. Action space — a per-note bowing envelope

`a_t ∈ [-1,1]^6`, interpreted as three segments × (speed, depth):

```
a = [ σ₁, σ₂, σ₃ , δ₁, δ₂, δ₃ ]
```

### 4.1 Rendering

For a note of duration `T ≥ 0.40 s` (`ENVELOPE_MIN_DURATION`):

1. **Stroke length** is set by the *mean* speed residual, because bow budget is
   a property of the whole note:
   ```
   scale = 1 + mean(σ) · 0.35            (SPEED_RESIDUAL_FRAC)
   L_desired = L_planned · scale
   L = clip(L_desired, 1e-4, bow available in this direction)
   ```
2. **Time is split equally** across the three segments, `T/3` each.
3. **Bow is allocated in proportion to segment speed weights**:
   ```
   w_i = clip(1 + σ_i · 0.35, 0.25, 4.0)
   span_i = (u_end − u_start) · w_i / Σw
   ```
   Equal time with unequal distance is what makes `σ_i` read as "speed during
   this part of the note".
4. **Depth is set per segment outright**:
   ```
   d_i = clip( d_base + δ_i · 0.001 m , −0.0015 , +0.002 )
   ```
5. Each segment is solved with `solve_stroke(span_i, T/3)` for its commanded
   speed and acceleration, and the segments are emitted as one blended
   `moveL` path.

> **Known defect (2026-08-11).** `solve_stroke` solves a **rest-to-rest**
> trapezoid (§2.1), so each segment accelerates from zero and decelerates back
> toward zero. The blend radius is capped at 5 mm against a ~56 mm segment, so
> it rounds the junction without removing the deceleration ramp: the result is
> one *command* but three velocity humps, with the bow slowing to near zero
> twice inside every note. At near-zero bow speed the Helmholtz motion breaks
> down and the string re-articulates, which is **clearly audible as three
> separate segments** — reported by a listener, and *not* penalised by the
> reward, which scored the segmented policy slightly above the flat baseline.
> A correct envelope requires a continuous velocity profile with non-zero
> interior boundary velocities; only the first and last boundaries are at rest.
> Until that is fixed, §9.2's envelope result should be read as measuring a
> defective renderer, not the value of intra-note shaping.

**Total length and total duration are preserved exactly.** The envelope decides
only how travel is distributed in time; it cannot acquire extra bow or alter
the rhythm, which remain planner decisions.

For `T < 0.40 s` the note is executed as a single segment using `σ₁, δ₁`. Three
segments of a 0.25 s note are ~12 mm of bow each, and the `moveL` blend radius
(≤ 5 mm) would smear them together. Dimension 0 therefore means "the attack"
consistently: for a short note the attack *is* the note.

### 4.2 Why feed-forward rather than closed loop

`moveL` executes a pre-specified path, so the envelope is feed-forward: the
shape is chosen before the note sounds. Reacting *during* a note requires
streaming (`servoL`), which was implemented (`BaselineControls/servo_player.py`)
and measured against `moveL` on 14 identical 1.0 s strokes:

| | moveL | servoL (lookahead 0.03 / 0.10 / 0.20) |
|---|---|---|
| tone_quality | **0.840** | 0.617 / 0.641 / 0.717 |
| f0_stability (cents, lower better) | **0.464** | 1.286 / 0.545 / 7.223 |

No lookahead setting matched `moveL`. Low lookahead tracks the streamed target
tightly and transmits micro-jitter to the string (pitch instability); high
lookahead smooths it but lags (−26.6% speed at 0.20). `moveL` avoids the
dilemma because the controller *generates* a smooth trajectory internally.

Closed-loop control is additionally limited by latency: a 0.5 s analysis window
plus ~0.1 s inference means no correction can arrive before ~0.6 s, so a note
must exceed ~1 s to be corrected on the basis of its own sound — 3 of 182 notes
in `challengepiece`, 13 of 14 in `string_crossings`.

---

## 5. Transition — execution and measurement

Per stroke (`rl/piece_hardware.py`):

1. Sleep the notated gap, or perform a planned retake (lift → traverse → set
   down), keeping the bowing intact.
2. Execute the blended `moveL` path through `safe_moveL`, which raises on a
   failed move or a stopped control script.
3. `StateLogger` samples RTDE at 100 Hz; `get_summary(t_start, t_end)` returns
   the achieved mean bow speed and torque statistics over exactly that window.
4. Slice the analysis window from a continuously running microphone stream
   (§5.1), peak-normalise it, and measure its raw RMS level in dBFS **before**
   normalisation.
5. Score with the frozen classifier and compute the reward.

### 5.1 Note-matched analysis windows

The classifier was trained on 0.5 s windows, but most notes in real repertoire
are shorter — 86% of `challengepiece`, 84% of `t1`. A fixed 0.5 s window centred
on an 80 ms note contains 80 ms of that note and **420 ms of its neighbours**,
so the score is largely about other notes.

The window is therefore cut to the note:

```
if T > WINDOW + 2·PRE_ROLL:      # long note
    centre on the steady middle, past the onset transient; width = 0.5 s
else:                            # short note
    window = [t_start, t_start + max(T, 0.08 s)]
```

`MIN_ANALYSIS_SEC = 0.08` is a signal-processing floor: at 50 ms an open A
yields only ~11 pitch periods, insufficient for F0 stability or HNR. A note
below the floor is extended into its own release, never into the next note.

This is viable because the classifier's ranking degrades only slightly with
window length. Scoring identical `standard`-config recordings at four lengths:

| window | 0.10 s | 0.20 s | 0.35 s | 0.50 s |
|---|---|---|---|---|
| recording-level ρ | 0.824 | 0.834 | 0.828 | 0.835 |

The trunk pools time with mean+std, so padded frames contribute a constant
while the real frames drive the statistics.

---

## 6. Reward function

```
r =  0.50 · quality_eff        tone
   + 0.25 · r_dynamic          dynamic accuracy
   + 0.15 · r_bow              bow budget          (≤ 0)
   + 0.10 · r_smooth           inter-stroke jerk   (≤ 0)
   − 0.15 · r_defect           objective defects   (≥ 0, subtracted)
   + 0.05 · r_onset            onset acceleration  (≤ 0)
   + 0.05 · r_envelope         intra-stroke jerk   (≤ 0)
```

### 6.1 Tone

The classifier exposes six human-rated heads. The reward is a length-dependent
weighted mean, because **a short note is nearly all attack and release** — there
is little sustain to judge:

```
base weights:  tone_quality 0.45, attack 0.25, release 0.15, overall 0.15
fill = min(1, T / 0.5)
if fill < 1:  shift 0.30·(1−fill) from tone_quality to attack
              shift 0.10·(1−fill) from overall     to release
```

Effective weights:

| note | tone_quality | attack | release | overall |
|---|---|---|---|---|
| ≥ 0.50 s | 0.45 | 0.25 | 0.15 | 0.15 |
| 0.25 s | 0.30 | 0.40 | 0.20 | 0.10 |
| 0.07 s | 0.19 | **0.51** | 0.24 | 0.06 |

`attack_quality` and `release_quality` are human judgements about events short
enough to fit inside a brief note, which makes them the only sound estimator
available at that length — no additional annotation is required.

A light length discount remains:

```
quality_eff = 0.5 + (0.85 + 0.15·fill) · (tone − 0.5)
```

The floor is 0.85 rather than 0.5 because the measured degradation with window
length is small (table in §5.1); the residual accounts for feature reliability
at short lengths, not for window contamination, which note-matched windowing
already removed.

### 6.2 Dynamic accuracy

Closed loop on **measured acoustic level**, not on commanded bow speed. The
level model, fitted on the 100 `standard`-config recordings:

```
dBFS = 1.131 · 20log₁₀(v) + 0.421 · depth_mm − 5.295
R² = 0.892,  residual sd = 1.14 dB
```

The speed exponent ≈ 1.13 confirms amplitude ∝ bow speed; depth at
0.42 dB/mm ≈ 1 dB across the envelope. **Bow position is deliberately excluded**:
every dataset recording is a full-bow stroke, so its midpoint is
0.535 ± 0.0002 and including it yields a degenerate fit (coefficient ~9800,
predictions near −372 dBFS). This is a genuine limitation — the policy plays
partial strokes across u = 0.08–0.8, and that error is not in the 1.14 dB
residual.

Written dynamics are mapped to four **zones**, equal 2.5 dB slices of the
~10 dB reachable span:

| zone | dBFS (model frame) | speed at centre |
|---|---|---|
| p | −28.7 … −26.2 | 0.102 m/s |
| mp | −26.2 … −23.7 | 0.132 m/s |
| mf | −23.7 … −21.2 | 0.170 m/s |
| f | −21.2 … −18.7 | 0.220 m/s |

```
r_dynamic = 1                                  inside the zone
          = max(0, 1 − distance / 3 dB)        outside
```

Flat inside rather than peaked at the centre, because the model's own residual
is 1.14 dB — rewarding sub-decibel precision would reward measurement noise.

**Gain calibration.** dBFS is absolute, so the zones are tied to the microphone
and gain that recorded `dataset_a_final`. Measured 2026-08-11, the current rig
reads **+4.20 dB hotter**: strokes at −17.6 dBFS against a predicted −21.8,
above even the `f` ceiling, making `r_dynamic` a constant 0 with 0/14 strokes in
zone. `gain_offset_db` corrects the measurement into the model frame and must be
re-measured whenever the audio path changes.

Where no level model or microphone is available the term falls back to a
speed proxy, `r_dynamic = clip(1 − |err_dB| / 3, 0, 1)` with the target capped
at `v_max(T)` from §2.1.

### 6.3 Bow budget, smoothness, defects, onset

```
r_bow      = −clip(3 · shortfall, 0, 1) − 0.5·(1 − edge/0.03) if edge < 0.03
r_smooth   = −0.5 · mean|a_t − a_{t−1}|
r_envelope = −0.5 · ( mean|Δσ| + mean|Δδ| )        envelope only
r_onset    = −clip( (accel − 1.2) / (4.0 − 1.2), 0, 1 )
```

`shortfall` is the fraction of the desired stroke length that had to be cut to
stay inside `[U_MIN, U_MAX]`; `edge` is the distance from the final bow position
to the nearer hard limit. `r_onset` prices attack harshness explicitly, since
`solve_stroke` otherwise raises acceleration freely to fit short notes and
higher acceleration *is* a harder attack.

**Objective defect penalty.** Four hand-engineered features, monotonic and
computable at any window length, scored linearly between a good and a bad value
and averaged:

| feature | direction | good | bad | defect |
|---|---|---|---|---|
| `hnr_db_mean` | higher better | 12.0 | 3.0 | scratchy |
| `voiced_fraction` | higher better | 0.98 | 0.70 | string not speaking |
| `attack_overshoot` | lower better | 0.85 | 1.40 | harsh attack |
| `f0_stability_cents` | lower better | 1.0 | 20.0 | pitch wobble |

Spectral flatness and `envelope_cv` are **deliberately excluded** although the
codebase's heuristic scorer uses them: some noise is intrinsic to cello timbre
(driving flatness to zero yields a sterile tone), and a swell has high envelope
variation by design. Because these features need no labels and no minimum
window, they remain valid exactly where the CNN is weakest.

---

## 7. Reward model (frozen classifier)

`SoundClassifier/quality_classifier.py`, `MelQualityCNN`.

**Inputs** — log-mel spectrogram (64 mels × T frames, 0.5 s → 87 frames);
40 hand-engineered scalar features (HNR, spectral flatness, F0 stability,
voiced fraction, RMS envelope statistics, attack time and overshoot, 13 MFCC
means and standard deviations); a 6-dim physical vector (commanded depth, force
deviation, measured bow speed, bow position, mean and peak torque); and
`window_pos ∈ [0,1]`, the window's position within the stroke.

**Trunk** — convolutional stack → `AdaptiveAvgPool2d((1, None))` (pools
frequency fully, retains time) → concatenation of mean and standard deviation
over time → 128-dim. **This is length-agnostic**: any frame count produces the
same vector, verified from 18 to 87 frames, which is what permits note-matched
windows.

**Outputs** — six sigmoid heads over `TIER1_FIELDS`: `overall`, `tone_quality`,
`attack_quality`, `release_quality`, `bow_control`, `dynamic_accuracy`.

**Targets** — mean annotator rating on a 1–4 scale mapped to [0,1] by
`(mean − 1)/3`, so a score is a *predicted mean human rating*.

**Training data** — 500 sustained full-bow strokes, 999 ratings from 5
annotators (499 recordings double-rated), cut into 0.5 s windows hopping 0.25 s.
`attack_quality` is supervised only on early windows and `release_quality` only
on late ones (`position_weight`). Loss is MSE on `overall` + 0.5 × in-batch
pairwise ranking hinge + 0.3 × position-weighted auxiliary MSE.

**Performance** — recording-level Spearman ρ = 0.798, window-level 0.777,
validation MSE 0.0242. Per bow configuration: `standard` 0.847 (the only
configuration the policy plays), `bridge` 0.804, `board` 0.641, `botangle`
0.614, `topangle` 0.614.

**Checkpoint provenance matters.** `classifier.py`'s `DEFAULT_CHECKPOINT`
(`quality_cnn.pt`) is trained on `pseudo_heuristic_v1` **pseudo-labels** — its
near-perfect metrics (ρ 0.99, MSE 0.0003) reflect re-learning a deterministic
heuristic, not human judgement. Optimising a policy against it would maximise a
hand-written formula. `RealScorer` therefore prefers human-labelled checkpoints
and warns on pseudo-label ones.

---

## 8. Algorithm and hyperparameters

Soft Actor-Critic (Stable-Baselines3), `MlpPolicy` (two 256-unit hidden layers),
twin Q-critics, automatic entropy tuning.

| | mock | **real robot** |
|---|---|---|
| learning rate | 3e-4 | 3e-4 |
| discount γ | 0.99 | 0.99 |
| replay buffer | 100,000 | 100,000 |
| batch size | 256 | 256 |
| τ (soft update) | 0.005 | 0.005 |
| `train_freq` | 1 | 1 |
| `learning_starts` | 1000 | **200** |
| `gradient_steps` | 1 | **4** |
| checkpoint interval | 5000 steps | **250 steps** |

**Off-policy replay is what makes hardware training feasible.** Each stroke
costs ~0.6–1.5 s of robot time. A 4,000-stroke run performs 4,000 × 4 gradient
steps × 256 batch ≈ 4.1 M transition-samples — each stroke replayed ~1,000
times. An on-policy method reusing each sample ~10 times would need roughly an
order of magnitude more robot time for equivalent learning (~70 h versus 40 min).

γ = 0.99 over a 14–182 step episode gives a discount horizon spanning the whole
piece, which the critic needs because bow consumed now constrains later strokes.

The replay buffer is checkpointed with the model: it *is* the collected robot
data, and a crash otherwise discards it. Note that a changed reward invalidates
stored transitions, so reward modifications require a fresh run rather than
`--resume`.

---

## 9. Measured results

### 9.1 Tone-only policy (t1, 2-dim action, pre-envelope reward)

4,000 strokes / 160 episodes / ~40 min. Deterministic evaluation, 3 episodes:

| | tone quality | ≈ human rating |
|---|---|---|
| baseline (zero residual) | 0.203 ± 0.006 | 1.61 / 4 |
| **policy** | **0.479 ± 0.006** | **2.44 / 4** |

**+136%.** Baseline re-measured after 40 min of continuous bowing was 0.203
against 0.193 before, ruling out instrument drift.

**Learned behaviour:** speed residual ≈ +0.25, depth ≈ −0.23 — *bow faster and
lighter*, a standard cello principle recovered from the classifier alone.

**Gain concentrated where the reward is most trustworthy**, which is the
opposite of what reward hacking on out-of-distribution short notes would
produce:

| note length | baseline | policy | gain |
|---|---|---|---|
| ≥ 0.5 s | 0.297 | 0.665 | **+0.368** |
| 0.2–0.5 s | 0.186 | 0.462 | +0.275 |
| < 0.2 s | 0.159 | 0.389 | +0.230 |

**Caveat:** this policy bought part of its gain by playing louder, showing
7–11 dB errors when graded by the closed-loop dynamic zones — the failure that
motivated §6.2.

### 9.2 Baseline under the current reward (string_crossings, 6-dim action)

| | value |
|---|---|
| tone quality | 0.644 ± 0.027 |
| dynamic accuracy | 1.000 (14/14 in zone) |
| return | 4.89 ± 0.06 |

Dynamics is saturated at the baseline action on this piece (single `mf`
marking, correctly hit) but not during training, where exploration moves the
level out of zone — so the term functions as an anchor against buying tone with
loudness rather than as an objective with headroom.

### 9.3 Envelope policy — a negative result

1,500 strokes / 107 episodes / 30 min. Deterministic evaluation, **8 episodes**
each (a 3-episode baseline gave 0.644 ± 0.027 and was too noisy to compare
against; the 8-episode figure is the reliable one):

| | tone quality | return |
|---|---|---|
| baseline (flat) | **0.679 ± 0.007** | 5.00 ± 0.03 |
| envelope policy | 0.671 ± 0.009 | **5.08 ± 0.03** |

**The policy raised return by 0.08 while lowering tone by 0.008.** It learned a
consistent shape — segment-1 speed residual −0.13 on all 14 strokes, i.e. a
soft onset — and depth shaping went unused (spread 0.024).

This is a reward-specification failure, not a training failure. `r_onset`,
`r_smooth` and `r_envelope` all pay for a gentle first segment, while **no term
prices the bow decelerating to near zero at the interior segment boundaries**
(§4.1 defect). SAC correctly maximised the objective it was given. A listener
identified the segmentation immediately; the composite reward rated it an
improvement.

### 9.4 Same policy, continuous renderer

`_build_envelope` was changed to give each segment a **cruise** speed — scaled
by the whole-stroke peak/mean ratio so the two real end ramps are still paid
for — instead of an independent rest-to-rest `solve_stroke` result, and the
blend cap was raised from 5 mm to 25 mm. Interior boundaries now carry
velocity. For the learned action the three segments run 0.1917 / 0.1971 /
0.2039 m/s, a smooth 6% rise, where previously each returned to zero.

| | tone quality | return |
|---|---|---|
| baseline (flat) | 0.679 ± 0.007 | 5.00 ± 0.03 |
| policy, rest-to-rest segments | 0.671 ± 0.009 | 5.08 ± 0.03 |
| **policy, continuous segments** | **0.750 ± 0.015** | 5.52 ± 0.11 |

Rendering alone is worth **+0.079** tone; against the flat baseline the
continuous envelope is **+10.5%**. Intra-note shaping does help — §9.3
measured the renderer defect cancelling it out.

Two caveats. The policy was *trained* against the rest-to-rest renderer, so
this is a transfer result and a retrain should do better. And the reward still
contains no term penalising mid-note velocity collapse: the defect that
produced §9.3 would go unpunished again if it recurred, so this number rests on
the renderer being correct rather than on the objective detecting it.

---

## 10. Limitations

1. **Single string, single configuration.** All results are open A in the
   `standard` bow configuration. The classifier is markedly weaker on other
   configurations (ρ 0.61–0.64 for the tilt extremes).
2. **The reward model is also the evaluation metric.** Improvements are
   measured by the same network being optimised. Two cases have already been
   found where it disagreed with human hearing: `overall` ranked runs opposite
   to a listener on `challengepiece`, and `servoL` scored acceptably while
   sounding clearly worse. A third and worse case is the segmented envelope
   above — audibly broken into three pieces, yet scored *above* the flat
   baseline. Note that `envelope_cv` is exactly the feature that would have
   caught it, and §6.3 deliberately excludes it on the grounds that "a swell
   has high envelope variation by design." That exclusion removed the only
   defect detector sensitive to mid-note amplitude dips. Human listening
   remains the ground truth.
3. **Short notes lack direct supervision.** No annotator has rated an 80 ms
   note. The reward substitutes `attack_quality`, a human-labelled judgement
   about a short event, but this is a proxy.
4. **Context, not just length.** Note-matched windows eliminate neighbour
   contamination, but validation used clean slices of isolated sustained
   strokes; short notes embedded in fast passages remain untested.
5. **The loudness model omits bow position** (§6.2) despite the policy playing
   across u = 0.08–0.8.
6. **Absolute dBFS zones** depend on the microphone and gain; a +4.20 dB
   discrepancy silently zeroed the dynamics term until measured.
7. **Dynamic range is hardware-limited.** The instrument spans 8.9 dB in total;
   a written *p*–*f* uses ~3.5 dB of it. Against a human cellist's 30+ dB, the
   achievable contrast is small regardless of policy.
8. **Envelope control is unvalidated on hardware.** Mock training confirmed the
   6-dim space is learnable, but the mock's shape preference was authored rather
   than measured, making that a learnability test, not a value test.

---

## 11. Reproducing

```bash
# baseline (zero residual)
python rl/play_piece.py MIDI-Files/string_crossings.mid --real --episodes 3

# train
python rl/train_piece.py MIDI-Files/string_crossings.mid --real \
    --timesteps 1500 --save-stroke-audio

# evaluate a policy
python rl/play_piece.py MIDI-Files/string_crossings.mid --real --episodes 3 \
    --model rl/checkpoints_envelope/sac_piece_string_crossings_final
```

Robot at `192.168.1.100`; the audio channel is probed and proven live at
startup, because an empty input returns a valid stream of its own noise floor
(a full session was once recorded at −90 dBFS this way).

`ur_rtde` must be built against `boost@1.85`; Homebrew's Boost 1.90 no longer
ships the `boost_system` CMake component the 1.6.3 sources require.
