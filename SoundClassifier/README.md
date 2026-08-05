# Bowing Quality Classifier

Scores a cello bow stroke from 0 (poor) to 1 (great) using the audio it
produced plus the robot's physical state while producing it. The primary
`overall` score is the signal the RL reward function
(`reward/reward_funct.py`) uses to judge stroke quality, via
`reward/sliding_window.py`. The model is multi-task: alongside `overall` it
also predicts five auxiliary technical-quality dimensions (tone, bow control,
attack, release, dynamic accuracy), available via `predict_detailed()` for
anything that wants a finer-grained signal later (e.g. an attack-specific
reward term).

It replaces `example_sound_classifier.py`'s `ExSoundClassifier`, which faked
a quality score from force alone, purely so RL training could be developed
before any real classifier existed. `BowingQualityClassifier` in
`classifier.py` is the real thing, trained on actual recordings and human
ratings from `Data_Collection/`.

## Inputs and outputs

**Inputs** (per 500ms window):

| Input | Shape | Source |
|---|---|---|
| Log-mel spectrogram | `(1, 64, 87)` | `audio_features.compute_log_mel_spectrogram` |
| Hand-engineered scalar features | `(40,)` | `audio_features.extract_scalar_features` (HNR, flatness, F0 stability, envelope stats, attack shape, 13 MFCCs mean+std) |
| Physical state vector | `(6,)` | `dataset.build_physical_features` / `rl/env.py`'s `_get_physical_params()` at inference |
| Window position | scalar in `[0,1]` | `dataset.py`'s `window_center_fraction` (0 = stroke onset, 1 = release); defaults to `0.5` if not supplied |

**Outputs:**

| Call | Returns |
|---|---|
| `predict()` | a single `overall` float in `[0,1]` -- the RL reward signal, unchanged contract |
| `predict_detailed()` | `{'overall', 'tone_quality', 'attack_quality', 'release_quality', 'bow_control', 'dynamic_accuracy'}`, each a float in `[0,1]` |

## How it works, end to end

```
 robot stroke
   |
   |-- microphone --------> audio_features.py --> mel-spectrogram ----\
   |                                          \--> 40 hand-engineered   \
   |                                               scalar features       >--> quality_classifier.py (CNN, multi-task) --> overall [0,1]
   |                                                                    /                                              \-> tone_quality, attack_quality,
   \-- robot state -------> 6-dim physical feature vector ------------/                                                    release_quality, bow_control,
   |                                                                                                                        dynamic_accuracy  [0,1] each
   \-- window position (0=onset..1=release) ----------------------------------------------------------------------------/
```

Three inputs feed the model:

1. **Audio.** A 500ms window of audio is converted into a log-mel
   spectrogram (what the CNN's conv layers actually look at) *and* a small
   set of hand-engineered features that target specific failure modes a
   spectrogram alone doesn't cleanly expose:

   | Feature | What it catches |
   |---|---|
   | HNR (harmonic-to-noise ratio) | Low HNR = scratching/noise instead of a clean tone |
   | Spectral flatness | Near 0 = pure tone; near 1 = noise-like |
   | F0 stability (cents around median) | A held note that wavers/breaks pitch |
   | Voiced fraction | How much of the window has a detectable pitch at all |
   | RMS envelope mean/std/CV | Overall loudness and how unstable it is |
   | Envelope trend | Is the stroke ramping up/down in level, or steady |
   | Envelope outlier rate | Spikes/dips/dropouts relative to the local median |
   | Attack time + overshoot | A clean stroke onset is quick and doesn't overshoot into a scratchy transient |
   | MFCCs (13 coefficients, mean+std) | Timbre -- "what the string actually sounds like" beyond just pitch/loudness |

2. **Robot physical state.** A 6-dim vector built from whatever's measured
   for that stroke (bow speed, bow position, press depth/force, joint
   torque) -- see [Physical features](#physical-features-and-their-caveat)
   below for exactly what's in each slot and why.

3. **Window position.** Where this 500ms window sits within the stroke's
   active region. Needed because attack quality can only be judged from
   early windows and release quality from late ones -- see
   [The CNN](#the-cnn) and [Training](#training) for how this actually gets
   used (it's a fusion input *and* it reweights the aux losses).

The CNN fuses all three: a conv stack + temporal pooling reduces the
mel-spectrogram to a 128-dim embedding, concatenates it with the (normalized)
audio + physical feature vectors and window position, then splits into one
head per Tier-1 dimension -- see [The CNN](#the-cnn) for the full diagram.

## Files

| File | Role |
|---|---|
| `audio_features.py` | All signal-processing feature extraction (mel-spectrogram + the 40 hand-engineered scalar features above). No torch dependency -- usable standalone. |
| `dataset.py` | Reads `metadata.jsonl` + `annotations`, cuts each annotated recording into 500ms training windows, computes each window's position, builds the physical feature vector, maps Tier-1 annotations to `[0,1]`, does group-aware train/val splitting. |
| `quality_classifier.py` | The `MelQualityCNN` model definition (temporal pooling + multi-task heads) + `FeatureNormalizer` (z-score normalization fit on training data, reused at inference). |
| `train_classifier.py` | Training CLI: multi-task loss, Spearman-rho metrics, LOGO-CV, optional Ridge baseline. Saves a checkpoint `classifier.py` loads. |
| `classifier.py` | `BowingQualityClassifier` -- the real-time inference class the RL reward function actually imports (`predict()` / `predict_detailed()`). |
| `inspect_features.py` | Standalone diagnostic: feature distributions + correlation with human ratings, independent of the CNN. |
| `example_sound_classifier.py` | The old force-only placeholder, kept for reference/comparison. |
| `Data_Collection/annotate.py` | Quick terminal annotator -- single 1-4 `overall` score per recording, no browser needed. |
| `Data_Collection/label_studio_bridge.py` | Browser-based annotation via Label Studio: exports/imports the full two-layer schema (see [Annotation schema](#annotation-schema)). |
| `Data_Collection/` (rest) | Recording (`recording_a_only.py`) and dataset validation (`validate_dataset.py`) tooling, and the datasets themselves (`dataset_a_configs/`, etc.) |

## Annotation schema

Each recording can be rated on two layers (`label_studio_bridge.py`'s
`LABEL_CONFIG`); the terminal tool `annotate.py` only ever collects the
first field of Layer 1.

**Layer 1 -- technical quality, 1-4 each, required** (`dataset.TIER1_FIELDS`):
`overall`, `tone_quality`, `bow_control`, `attack_quality`,
`release_quality`, `dynamic_accuracy`. `overall` is a holistic judgment, not
the mean of the other five, and is the only field the RL reward reads.

**Layer 2 -- tonal character, 1-5 bipolar, optional, not trained on yet**:
`dark_bright`, `cold_warm`, `harsh_sweet`, `dry_resonant`, `grainy_smooth`.
Collected for future CLAP-style timbre work; kept in metadata but ignored by
`dataset.py`/`quality_classifier.py` today.

Plus a `skip_flag` (`audio_problem` / `robot_problem`) and free-text `notes`.
`label_studio_bridge.py import` records the flag but excludes that
annotation's quality scores when a problem is flagged.

An annotator only needs to answer Layer 1 every time; Layer 2 is meant to be
rated roughly every third recording, when Layer 1 was unambiguous.

## Data flow: from a recording to a training example

Each row in `Data_Collection/dataset_a_configs/metadata.jsonl` is one full
bow stroke: an audio file, the robot's measured state during the stroke
(`measured`), what was commanded (`commanded`), and (once annotated) a list
of human `annotations`, each `{annotator, timestamp, ...Tier-1/2 fields}`
(see [Annotation schema](#annotation-schema)).

`dataset.build_training_examples()`:

1. Skips any recording with no annotations -- there's no label to train on.
2. Averages all annotators' `overall` scores and maps `1..4 -> 0..1`
   (`dataset.annotation_score_01`), and does the same per-field for every
   `TIER1_FIELDS` entry (`dataset.annotation_multidim`) -- a field missing
   from every annotation (older recordings rated with only `overall`) falls
   back to the recording's overall score, so old and new annotations mix
   without special-casing downstream.
3. Loads the audio and cuts it into 500ms windows, hopping by 250ms, across
   the stroke's active region (`audio_timing.stroke_start_s/stroke_end_s`
   from the recording script, padded a little on each side to keep onset
   and release in view). A 5-6 second stroke becomes ~20 overlapping
   windows -- this is also how the pipeline gets usable training volume out
   of a small number of annotated *recordings*. Each window also gets a
   `window_pos` -- its center position as a fraction of the active region
   (0 = onset, 1 = release) -- computed from the same cut bounds.
4. Every window from one recording gets the recording's per-field labels (a
   strong assumption for most fields -- see [Limitations](#limitations),
   though `window_pos`-based loss weighting specifically corrects for it on
   attack/release) and is grouped by `stroke_id` + `repeat` so a later
   train/val split never puts windows from the same recording on both
   sides.

This windowing is intentional, not incidental: training windows are the
exact same 500ms/44.1kHz shape that `reward/sliding_window.py` scores at
runtime, so the model never sees a different kind of input at inference
than it was trained on.

## Physical features and their caveat

`dataset.PHYSICAL_FEATURE_NAMES` (6 slots), filled from
`metadata.jsonl`'s `measured`/`commanded`/`force_contact` fields:

| Slot | Offline (training) source | Online (RL) source |
|---|---|---|
| 0 | `force_contact.force_mean` if present, else `commanded.depth_m` | `physical_params[0]` (measured force) |
| 1 | `force_contact.force_std`, else 0 | `physical_params[1]` (abs force deviation) |
| 2 | `measured.speed_mean` | `physical_params[2]` (bow speed) |
| 3 | midpoint of `measured.bow_pos_start/end` | `physical_params[3]` (bow position) |
| 4 | `measured.torque_mag_mean` | `physical_params[4]` (FT lateral force) |
| 5 | `measured.torque_mag_max` | `physical_params[5]` (FT torque) |

`physical_params` is the same 6-dim vector already defined by
`rl/env.py`'s `_get_physical_params()` and threaded through
`reward/sliding_window.py` -- this schema was deliberately kept in that
order so `classifier.py` can pass it straight through with no remapping.

**The catch:** `dataset_a_configs/metadata.jsonl` has no working force
sensor yet (`force_contact` is `null` on every record), so offline slot 0 is
actually always the *commanded press depth in meters* (0-0.004 range), while
online slot 0 is a *measured force in Newtons* (0-8 range). These aren't the
same unit, but both represent "how hard is the stroke being pressed," and
z-score normalization (fit per-slot on the training set, reapplied at
inference) cares about relative spread, not literal units -- so this is a
workable stand-in, not a correct one. Revisit this mapping once the robot
has working force/depth telemetry that matches what `rl/env.py` expects.

## The CNN

`quality_classifier.MelQualityCNN` (`multitask=True`, the default):

```
mel-spectrogram (1, 64, 87)      scalar features (40,)  physical features (6,)  window_pos (1,)
        |                                |                       |                    |
  3x [Conv2d -> BN -> ReLU -> (Pool)]     |                       |                    |
        |  (B, 64, 1, T')                |                       |                    |
  AdaptiveAvgPool2d((1,None))             |                       |                    |
   keeps time, pools freq                 |                       |                    |
        |                                 |                       |                    |
  [mean(dim=T), std(dim=T)] -> 128-dim    |                       |                    |
        \________________________________|_______________________|____________________/
                                          |
                                  concat -> 175-dim
                        ___________________|____________________
                       |              |             |           |     ...
                  head_overall   tone_quality   attack_quality  ...  (one small head per aux field)
                Linear->32->ReLU    Linear->16->ReLU (x5, in heads_aux ModuleDict)
                ->Dropout->1            ->1
                       |              |             |           |
                   sigmoid        sigmoid       sigmoid      sigmoid
                       |              |             |           |
                   'overall'    'tone_quality' 'attack_quality' ...   -- forward_multitask()
                                                                       (forward() returns only 'overall')
```

87 is the frame count a 500ms window produces at the chosen FFT settings
(`audio_features.mel_spectrogram_frames_for_duration`) -- fixed so every
window, train or inference, produces an identically-shaped tensor.

Mean *and* std pooling over time (128-dim, not the single 64-dim
global-average-pooled vector the first version used) is deliberate: mean
captures average timbre, std captures within-window instability
(graininess), and plain average pooling throws away exactly the temporal
shape ("clean attack, rough finish") that attack/release/consistency
judgments need.

`window_pos` defaults to `0.5` when not supplied (`forward()`'s and
`predict()`'s backward-compatible path), so old call sites that don't know
about it still get sensible mid-stroke behavior.

**Backward compatibility:** checkpoints saved before this multi-task head
existed (`schema_version` absent, implicitly 1) used a different trunk --
conv straight into `AdaptiveAvgPool2d(1)` (64-dim, no time-keeping), one
single `head`, no `window_pos` input. `MelQualityCNN(multitask=False)`
reproduces that exact architecture key-for-key so `classifier.py` can load
either vintage of checkpoint without remapping.

## Training

```bash
python train_classifier.py \
    --meta Data_Collection/dataset_a_configs/metadata.jsonl \
    --audio-dir Data_Collection/dataset_a_configs/audio \
    --out checkpoints/quality_cnn.pt \
    --baseline   # optional: also fits a Ridge baseline for comparison
```

Temporary no-human-annotation bootstrap for RL:

```bash
python train_classifier.py \
    --meta Data_Collection/dataset_a_final/metadata.jsonl \
    --audio-dir Data_Collection/dataset_a_final/audio \
    --out checkpoints/quality_cnn.pt \
    --pseudo-labels \
    --baseline
```

`--pseudo-labels` derives provisional Tier-1 targets from deterministic
audio/metadata heuristics (`pseudo_heuristic_v1`) and ignores human
annotations. This is useful for getting the MelQualityCNN contract into RL
before annotator results come back, but its validation metrics are measured
against the heuristic teacher, not human ground truth. Pseudo-trained
checkpoints are marked with `label_source='pseudo_heuristic_v1'` and
`pseudo_labels=True`.

**Loss** (`train_classifier.compute_loss`) is three terms:

```
L = MSE(pred_overall, y_overall)
  + 0.5 * pairwise_ranking(pred_overall, y_overall)      # in-batch, pairs with |Δlabel| > 0.15, margin 0.1
  + 0.3 * mean_over_aux_fields( w_f(window_pos) * MSE(pred_f, y_f) )
```

The ranking term pushes the model to get the *relative order* of strokes in
a batch right, not just the absolute score -- useful when labels are noisy
but relative comparisons are more reliable. The aux term is weighted by
`window_pos` so `attack_quality` is only supervised on early windows and
`release_quality` only on late ones (`w = clamp(1-2p, 0, 1)` and
`clamp(2p-1, 0, 1)` respectively; every other aux field gets `w = 1`) --
this stops the model being penalized for not hearing the attack in a
mid-stroke window.

**Metrics:** every epoch reports MSE plus Spearman ρ on `overall`, at both
window level and recording level (window predictions averaged per
`group_id` first). Model selection uses recording-level ρ (falls back to
`-val_mse` when ρ isn't computable, e.g. a validation split with only one
recording). After training it also prints:
- **bad-condition separation** -- mean predicted `overall` for `bad_*`
  vs. systematic recordings, warns if the gap is under `0.15`;
- **per-config breakdown** -- recording-level ρ for each of
  `standard/bridge/board/topangle/botangle`, once a config has 5+ annotated
  recordings;
- **Ridge baseline** (`--baseline`) -- `sklearn.linear_model.Ridge(alpha=1.0)`
  on `scalar_features ++ physical_features` alone, same splits. If the CNN
  doesn't beat it, that's a loud warning, not a footnote -- it means the
  audio model isn't adding value over the cheap features yet.

`validate_dataset.py`'s own readiness checklist wants 50+ annotated
recordings before training means anything. Below 20 annotated recordings
(grouped by source recording, not by window), `train_classifier.py` refuses
to save a checkpoint and instead runs **leave-one-group-out cross-validation**
-- one fold per annotated recording -- and prints the mean held-out MSE plus
the recording-level Spearman ρ pooled across all folds' held-out predictions
(a single fold only has one recording, so within-fold ρ is undefined). Pass
`--force-single-split` to override and save a checkpoint anyway (useful for
smoke-testing the pipeline, not for trusting the result).

Checkpoints (`torch.save`) contain the model weights, the fitted
`FeatureNormalizer` stats for both feature vectors, the target frame count,
`aux_fields` (`TIER1_FIELDS`, so `classifier.py` knows what the aux heads
mean), a `schema_version` (`2` for the multi-task architecture; absent/`1`
means the old single-head checkpoint format), and how many
recordings/what validation MSE and ρ it was trained with --
`classifier.py` prints that provenance when it loads one.

## Annotating more data

Two tools collect annotations into the same `metadata.jsonl` format; either
can be used, and their output mixes freely (`dataset.annotation_multidim`
falls back to `overall` for any field an annotator didn't rate).

**Quick terminal pass** -- single `overall` score only, no browser:

```bash
python Data_Collection/annotate.py \
    --meta Data_Collection/dataset_a_configs/metadata.jsonl \
    --audio-dir Data_Collection/dataset_a_configs/audio \
    --annotator <your initials>
```

Plays each unrated recording and asks for a 1-4 score (poor/fair/good/
great). Saves after every single rating, so it's safe to stop anytime and
resume later. `annotate.py --check-agreement` reports inter-annotator
Cohen's kappa once two or more people have rated overlapping recordings.

**Full two-layer pass** -- via Label Studio, for the complete Tier-1 +
Tier-2 schema (see [Annotation schema](#annotation-schema)):

```bash
# generate task files + the label_config.xml for each annotator
python Data_Collection/label_studio_bridge.py export \
    --meta Data_Collection/dataset_a_configs/metadata.jsonl \
    --audio-dir Data_Collection/dataset_a_configs/audio \
    --out-dir Data_Collection/label_studio_tasks_a \
    --annotators SK A1 A2 --annotations-per-sample 3 --hide-condition

# after annotating in the Label Studio UI and exporting each project's JSON:
python Data_Collection/label_studio_bridge.py import \
    --meta Data_Collection/dataset_a_configs/metadata.jsonl \
    --exports SK=exports/SK.json A1=exports/A1.json A2=exports/A2.json
```

See `Data_Collection/LABEL_STUDIO_ANNOTATION.md` for the full UI walkthrough
(starting the server, project setup, data import).

## Inference / RL integration

```python
from classifier import BowingQualityClassifier

classifier = BowingQualityClassifier()   # loads checkpoints/quality_cnn.pt if present
score = classifier.predict(audio_chunk, physical_params, string='A')

# optional finer-grained read, e.g. for future attack-specific reward shaping:
detail = classifier.predict_detailed(audio_chunk, physical_params, string='A', window_pos=0.1)
# -> {'overall': ..., 'tone_quality': ..., 'attack_quality': ..., 'release_quality': ...,
#     'bow_control': ..., 'dynamic_accuracy': ...}
```

`predict()` is a drop-in replacement for `ExSoundClassifier` /
`MockSoundClassifier` -- same `predict(audio_chunk, physical_params,
string)` signature (plus a trailing optional `window_pos=None`, so existing
callers are unaffected), used identically by `reward/sliding_window.py`'s
`SlidingWindowClassifier` and `reward/reward_funct.py`'s
`CompleteCelloReward`. `audio_chunk` should be the ~500ms, already
peak-normalized window `sliding_window.py` builds; shorter/longer/`None`
chunks are padded, cropped, or zero-filled rather than raising.

`BowingQualityClassifier` infers the checkpoint's `schema_version` on load
and instantiates the matching model architecture, so it can load either an
old single-head checkpoint or a new multi-task one. `predict_detailed()` on
an old (`schema_version` 1) checkpoint falls back to `{'overall': ...}` --
there are no trained aux heads to report from.

If no checkpoint exists yet at `checkpoints/quality_cnn.pt`,
`BowingQualityClassifier` falls back to `heuristic_quality_score()` -- a
hand-weighted blend of HNR, flatness, envelope stability, and attack
overshoot (for `predict()`) or `{'overall': heuristic}` (for
`predict_detailed()`). It's there so the reward pipeline always returns
*something* usable, not so you should rely on it; train a real checkpoint
once enough annotations exist.

## Diagnostics

```bash
python inspect_features.py \
    --meta Data_Collection/dataset_a_configs/metadata.jsonl \
    --audio-dir Data_Collection/dataset_a_configs/audio
```

Runs the hand-engineered features over every recording and prints each
feature's mean/std across the dataset, plus its Pearson correlation with
human ratings once 3+ recordings are annotated. Use this to sanity-check the
*features themselves* (e.g. "does low HNR actually track with strokes
people rated poorly?") independent of whether the CNN has enough data to
learn anything yet.

## Limitations

- **Sample size.** As of writing, 7 of 635 recordings in
  `dataset_a_configs` are annotated. A model trained on that few examples
  will not generalize -- `train_classifier.py`'s LOGO-CV gate exists
  specifically to stop a misleadingly "trained" checkpoint from being saved
  and used in RL before there's enough signal to train on.
- **Whole-stroke labeling, partially mitigated.** Every 500ms window from a
  recording still inherits that recording's per-field scores (there's no
  within-stroke ground truth finer than one rating per dimension per
  recording). `window_pos`-weighted aux losses correct for the one case
  where this matters most -- attack/release quality being judged from
  windows that can't actually hear the attack/release -- but `overall`,
  `tone_quality`, `bow_control`, and `dynamic_accuracy` are still applied
  uniformly across all of a recording's windows.
- **Layer 2 (tonal character) isn't trained on yet.** It's collected and
  stored in `metadata.jsonl` (`dark_bright`, `cold_warm`, etc.) but
  `dataset.py`/`quality_classifier.py` don't use it -- reserved for future
  CLAP-style timbre work.
- **Physical feature unit mismatch.** See
  [Physical features](#physical-features-and-their-caveat) above --
  the offline/online press-depth-vs-force slot is an approximation, not an
  exact correspondence.
- **No multi-string conditioning yet.** `predict()` accepts a `string`
  argument for interface compatibility with `reward/sliding_window.py`, but
  the model doesn't currently use it -- `dataset_a_configs` is A-string
  only. Revisit if/when D/G/C string data is collected.
