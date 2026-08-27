# Computational Evaluation Results & Metrics Reference

This document defines the core metrics used in the computational evaluation of robotic cello performances, and presents a review of results across three pieces ("Yunpiece", "Vocalise", and "Twinkle-short") after applying a Residual Reinforcement Learning (RL) policy. The claims below have been re-checked against the raw stroke-level CSV data.

\---

## 1\. Evaluation Metrics Guide

### 1.1. Acoustic & Overall Quality Metrics

* **`quality_eff` (Effective Quality)**: The comprehensive acoustic quality score of a bowing stroke, computed as a weighted sum of four neural-network classifier predictions (tone, attack, release, and overall score). The weighting is dynamic with respect to note duration: longer notes place higher weight on tone, while shorter notes emphasize the beginning and end of the sound (attack/release).
* **`bow_control`**: A neural-network-based predictive score evaluating how stably and accurately the robot physically controlled bow angle, position, and pressure.
* **`defect_score`**: A penalty score in \[0,1] quantifying the degree of acoustic defects or noise in the sound, computed from Harmonic-to-Noise Ratio (HNR), voiced fraction, attack overshoot, and F0 stability. Higher values indicate more acoustic defects.

### 1.2. Dynamics & Timing Metrics

* **`dynamic_accuracy`**: A classifier-predicted, comprehensive accuracy score for dynamic (loudness) expression.
* **`dynamic_zone_hit`**: A binary indicator (1 = success, 0 = failure) of whether the robot's actual loudness fell within the target dB zone associated with the score's dynamic marking (f/mf/mp/p).
* **`dynamic_distance_db`**: The physical error margin (in dB) between the actual sound and the target dB zone. Zero when the sound falls inside the target zone.
* **`drift` / `drift_s`**: A timing metric independent of loudness, measuring how late the robot executed a stroke relative to the score's designated timing. Reported in seconds (`drift_s`) in internal logic and CSV records, and in milliseconds (`drift`) in terminal output and reporting.



> **Note**: `dynamic_accuracy` (classifier-predicted) and `dynamic_zone_hit` / `dynamic_distance_db` (measured from raw dBFS) moved in different — and in some cases opposite — directions across all three pieces. The two are likely capturing different things, so we recommend treating the measured metrics (`dynamic_zone_hit` / `dynamic_distance_db`) as the primary indicator of dynamic compliance, and `dynamic_accuracy` as a secondary, supporting metric.

\---

## 2\. Residual RL Performance Evaluation by Piece

Under the current multi-objective reward function, the RL policy's outcome relative to the rule-based baseline varied substantially in direction depending on the piece. Below, each piece's result has been re-verified against the raw stroke-level CSV data.

### 2.1. "Yunpiece" (182 strokes, mean note duration 0.21 s / median 0.13 s — very short)

**Summary**: Tone and release improved substantially; dynamics worsened.

* `quality_eff` +0.0731 (SD 0.0089, n=3) — `tone_quality` rose from \~0.34 to \~0.54, and `defect_score` improved by −0.0955.
* `dynamic_zone_hit` −23.6 pp, `dynamic_distance_db` +0.76 dB — worsened. Per-zone analysis shows this is driven by **RL playing louder than the target in the soft (p) dynamic zone** (mean centre error: baseline +0.75 dB → RL +2.18 dB; 91% of the piece's 546 strokes fall in the p zone).
* `drift_s` +8.5 ms — a very minor increase.
* `attack_quality` fell slightly (−0.054) — the only sub-score that regressed for this piece.
* **Verification**: To check whether the `defect_score` improvement was simply a side effect of playing louder, we compared defect scores within matched loudness quartiles. RL's defect score was lower than baseline's in every quartile, indicating the tone/defect improvement reflects a genuine gain in audio
quality rather than being purely an artifact of increased loudness.

### 2.2. "Vocalise" (17 strokes, mean note duration 1.58 s — longest of the three pieces)

**Summary**: Release quality, bow control, and drift improved substantially; tone quality was essentially flat; attack quality showed inconsistent, seed-dependent results; dynamics worsened.

* `quality_eff` +0.0502 (SD 0.0559, n=3) — this gain is driven primarily not by tone but by **`release_quality`** (Base 0.325 → RL 0.518, **+0.194**, the single largest sub-score change observed across all three pieces). `tone_quality` actually decreased slightly (Base 0.670 → RL
0.661, **−0.009**).
* `defect_score` −0.0976 and `bow_control` +0.1585 — both improved.
* `attack_quality` rose by +0.030 on a pooled-stroke basis — the only piece with a positive average — but this is **not consistent across seeds**: seed 1 (+0.053) and seed 2 (+0.062) improved, while seed 3 (**−0.027**) regressed. This contrasts with Yunpiece (all three seeds negative: −0.088, −0.036, −0.038) and Twinkle-short (all three seeds negative: −0.083, −0.066, −0.083), where the direction was consistent across seeds. Given this, Vocalise's attack-quality result is better
described as "the only piece with a mixed/inconsistent direction" rather than as a clear improvement.
* `drift_s` −27 ms — timing was actually slightly more accurate than baseline.
* `dynamic_zone_hit` −21.6 pp, `dynamic_distance_db` +0.74 dB — worsened.
* **Verification (revised)**: Per-zone dBFS analysis shows that the baseline itself already overshoots the target loudness in the soft zones (mp: +2.7 dB, p: +5.0 dB over target), and RL fails to correct this — it makes it worse (mp: +3.7 dB, p: +6.7 dB over target). Only the f zone improved.
In other words, this is better described as **RL inheriting and amplifying a pre-existing baseline deficiency in the soft dynamic zones**, rather than a deliberate trade-off of dynamics for tone.
* **Statistical limitation (newly identified)**: `dynamic_zone_hit` was identical across all three RL runs — exactly 1/17 (=0.059) in each of RL1, RL2, and RL3. This is better interpreted as a sign that the sample size (17 strokes) yields low effective information for this metric, rather than as
evidence of consistent, independent replication across seeds. This piece's sample size is substantially smaller than the other two (126 and 182 strokes), and this should be stated explicitly.
* The correlation between `duration_error_s` and `defect_score` is only +0.31, much weaker than in
Twinkle-short (see below, +0.81). This suggests Vocalise's tone improvement operates through a
different mechanism than Twinkle-short's "delay → defect" chain.

### 2.3. "Twinkle-short" (42 strokes, mean note duration 0.55 s — intermediate)

**Summary**: Dynamics improved substantially; nearly everything else worsened, with severe cumulative timing drift.

* `dynamic_zone_hit` **+50.0 pp**, `dynamic_distance_db` **−0.41 dB** — the only piece of the three where dynamics clearly improved. This piece consists of a single `mf` dynamic marking, and RL played closer to the target loudness than baseline (mean centre error: +1.35 dB → +0.67 dB).
* `quality_eff` −0.0273, `defect_score` +0.1169 (worse), `smoothness_jerk` +32% (worse).
* `drift_s` **+645 ms** — drift accumulates roughly linearly across strokes, reaching approximately 1.37 s of delay by the 42nd note. Within the RL data, `duration_error_s` correlates strongly with `defect_score` (**r = +0.81**) — strokes that run longer than planned tend to also show worse
acoustic defects.
* **Interpretation**: The residual policy's tendency to slow down note execution accumulates into substantial absolute timing error on this piece's relatively long notes, producing simultaneous schedule collapse and quality degradation.

\---

## 3\. Cross-Piece Comparison

||Yunpiece|Vocalise|Twinkle-short|
|-|-|-|-|
|Strokes|182|17|42|
|Mean note duration|0.21 s|1.58 s|0.55 s|
|Baseline quality_eff|0.348|0.620|0.676|
|quality_eff Δ|+0.073|+0.050|−0.027|
|tone_quality Δ|Large improvement (+0.204)|\~No change (−0.009)|\~No change (+0.012)|
|release_quality Δ|+0.169|**+0.194 (main driver of quality_eff gain)**|−0.025|
|attack_quality Δ|−0.054 (all 3 seeds negative)|+0.030 pooled (inconsistent across seeds)|−0.077 (all 3 seeds negative)|
|defect_score Δ|Improved|Improved|Worsened|
|dynamic_zone_hit Δ|−23.6 pp|−21.6 pp|**+50.0 pp**|
|drift_s Δ|+8.5 ms|**−27 ms**|**+645 ms (up to \~1.37 s cumulative)**|

Viewing all three pieces together, two previously proposed hypotheses do not hold up against the full dataset.

1. **"RL hurts performance when baseline quality is already high" (based on Twinkle-short alone)**:
Vocalise's baseline `quality_eff` (0.620) is not meaningfully different from Twinkle-short's (0.676), yet Vocalise improved under RL. This causal explanation does not generalize across all three pieces.
2. **"Slower tempo / longer strokes are ideal for RL" (based on Vocalise alone)**:
Looking at note duration alone, both the shortest piece (Yunpiece) and the longest piece (Vocalise) show relatively favorable outcomes, while the intermediate-length piece (Twinkle-short) shows the worst outcome. This is not a linear "longer is better" relationship but rather closer to a **U-shaped, non-monotonic
pattern**. Note duration alone is unlikely to fully explain the direction of the outcome; other variables — such as the piece's dynamic-zone composition (Twinkle-short uses a single `mf` zone, while the other two pieces mix f/mf/mp/p) or the frequency of string crossings — may play an important role as well.

Additionally, the "excess loudness in soft dynamic zones" bias observed in common between Yunpiece and Vocalise actually worked in the opposite (improving) direction in Twinkle-short (a single-zone, `mf`-only piece). This is more consistent with an interpretation in which **the dynamic-zone loudness calibration itself is unstable, and its effect manifests differently depending on the piece's dynamic composition**, rather than the policy strategically choosing between tone and dynamics on a per-piece basis.

\---

## 4\. Discussion & Future Works

The evaluation across three pieces shows that the current multi-objective reward function applied to the Residual RL model produces markedly different — and at times opposite — outcomes depending on the piece's structural characteristics (note duration, dynamic-zone composition, etc.). Rather than framing
this as "the RL model deliberately choosing a different trade-off for each piece," it is more useful for guiding future improvements to separate the observations into **at least two distinct underlying causes** that dominate to different degrees in different pieces.

**Cause A — Timing drift (dominant in Twinkle-short)**: The residual policy's speed adjustments are biased toward extending note duration beyond the plan, and on pieces with relatively long and/or many notes, this error accumulates into severe schedule collapse (\~645 ms, up to 1.37 s at worst) and correlated quality degradation (duration_error–defect correlation of +0.81). → The reward function should weight the drift penalty substantially more heavily relative to the reward gained from dynamic accuracy.

**Cause B — Dynamic-zone calibration (dominant in Yunpiece and Vocalise)**: In the soft dynamic zones (mp/p), a pre-existing baseline tendency to overshoot the target loudness is either left uncorrected or amplified by RL. This is a separate issue from Cause A and will not be resolved simply by increasing the drift penalty. → Recalibration of the per-zone loudness reward is needed, with an explicit, stronger penalty for soft-dynamic deviations. It should also be noted that the classifier-based `dynamic_accuracy` metric and the measured `dynamic_zone_hit` / `dynamic_distance_db` metrics were inconsistent — at times moving in opposite directions — across all three pieces. This suggests the classifier's `dynamic_accuracy` head may be capturing something other than literal target-loudness compliance (e.g., relative dynamic contrast or articulation patterns). We recommend prioritizing the measured, dBFS-based metrics over the classifier's
`dynamic_accuracy` score when designing future rewards or selecting evaluation criteria.

**Limitation regarding sample size**: Vocalise has only 17 strokes, substantially fewer than the other two pieces (126 and 182 strokes), and its `dynamic_zone_hit` value was identical across all three seeds. Results for this piece — particularly the dynamics-related findings — should therefore be treated as preliminary. Across all three pieces, the small number of seeds (n=3) means these results should be reported descriptively, as an indication of direction, rather than as statistically significant findings.

