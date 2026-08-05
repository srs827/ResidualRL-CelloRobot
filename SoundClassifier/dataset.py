"""
dataset.py

Turns dataset_a_configs/{metadata.jsonl, audio/} into training examples for
the CNN quality classifier: each annotated recording is cut into one or more
500ms windows (the same window size/sample-rate the real-time classifier
scores at, via audio_features.SR / sliding_window.py's window_sec=0.5), so
training and inference see identically-shaped inputs.

Robot-data supplementation: each window is paired with a 6-dim physical
feature vector built from the recording's measured/commanded fields
(PHYSICAL_FEATURE_NAMES below). The same 6 slots are filled at live RL time
from the `physical_params` array already defined in rl/env.py's
_get_physical_params() -- see classifier.py for that mapping. Units don't
match exactly between the rich offline metadata and the slim online vector
(e.g. press depth in meters vs. measured force in Newtons both occupy slot
0); z-score normalization (fit on training data, applied at inference)
absorbs the scale difference so what matters is relative spread, not literal
units. Revisit once the live system has working force/depth telemetry.

A label only exists where a human annotated the recording (annotate.py).
Unannotated recordings are loaded but skipped for training -- still useful
via inspect_features.py for feature-distribution sanity checks.
"""

import json
from pathlib import Path

import numpy as np

import audio_features as af

WINDOW_SEC      = 0.5     # must match reward/sliding_window.py's window_sec
TRAIN_HOP_SEC   = 0.25    # overlap between augmented training windows
PRE_PAD_S       = 0.05    # include a little pre-onset audio for attack features
POST_PAD_S      = 0.10

PHYSICAL_FEATURE_NAMES = [
    'depth_or_force', 'force_deviation_or_zero', 'bow_speed',
    'bow_position', 'torque_or_lateral', 'torque_max_or_torque',
]
N_PHYSICAL_FEATURES = len(PHYSICAL_FEATURE_NAMES)

# Layer 1 (technical quality) annotation fields. 'overall' is the primary RL
# reward target; the rest are auxiliary multi-task targets (see
# quality_classifier.py's heads_aux / train_classifier.py's loss).
TIER1_FIELDS = ['overall', 'tone_quality', 'attack_quality',
                'release_quality', 'bow_control', 'dynamic_accuracy']

PSEUDO_LABEL_SOURCE = 'pseudo_heuristic_v1'


def load_records(meta_file):
    records = []
    with open(meta_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get('record_type', 'stroke') != 'stroke':
                continue
            records.append(record)
    return records


def annotation_score_01(record):
    """Mean of all annotators' 1-4 'overall' scores, mapped to [0,1]. None if unannotated."""
    scores = [a['overall'] for a in record.get('annotations', []) if 'overall' in a]
    if not scores:
        return None
    return float((np.mean(scores) - 1.0) / 3.0)


def annotation_multidim(record):
    """
    Per-TIER1_FIELDS mean annotator score mapped to [0,1]. A field missing
    from every annotation (e.g. older recordings annotated with only
    'overall') falls back to the record's overall score. Returns all-None
    if the record has no annotations at all.
    """
    annotations = record.get('annotations', [])
    overall = annotation_score_01(record)
    out = {}
    for field in TIER1_FIELDS:
        values = [a[field] for a in annotations if field in a]
        if values:
            out[field] = float((np.mean(values) - 1.0) / 3.0)
        else:
            out[field] = overall
    return out


def _clip01(x):
    return float(np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))


def _feature_dict(scalar_features):
    clean = np.nan_to_num(np.asarray(scalar_features, dtype=np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
    return dict(zip(af.SCALAR_FEATURE_NAMES, clean))


def pseudo_labels_from_features(record, scalar_features, window_pos=0.5):
    """
    Provisional non-human labels for bootstrapping RL before annotations return.

    These labels intentionally come from interpretable audio/metadata heuristics,
    not human ratings. They let the CNN learn the same input/output contract and
    deployment shape as the eventual annotation-trained model, while the saved
    checkpoint is marked with PSEUDO_LABEL_SOURCE so it is not mistaken for a
    human-supervised model.
    """
    f = _feature_dict(scalar_features)

    hnr = _clip01((f['hnr_db_mean'] + 3.0) / 23.0)
    low_hnr_var = 1.0 - _clip01(f['hnr_db_std'] / 10.0)
    tonal = 1.0 - _clip01(f['flatness_mean'] * 5.0)
    low_flat_var = 1.0 - _clip01(f['flatness_std'] * 20.0)
    voiced = _clip01(f['voiced_fraction'])
    f0_stable = 1.0 - _clip01(f['f0_stability_cents'] / 120.0)
    envelope_stable = 1.0 - _clip01(f['envelope_cv'] * 2.5)
    envelope_outlier_clean = 1.0 - _clip01(f['envelope_outlier_rate'] * 5.0)
    trend_stable = 1.0 - _clip01(abs(f['envelope_trend']) / 1.5)
    attack_fast = 1.0 - _clip01(f['attack_time_s'] / 0.18)
    attack_clean = 1.0 - _clip01((f['attack_overshoot'] - 1.0) / 1.2)

    tone_quality = _clip01(
        0.36 * hnr + 0.24 * tonal + 0.18 * voiced + 0.14 * f0_stable + 0.08 * low_flat_var
    )
    bow_control = _clip01(
        0.38 * envelope_stable + 0.24 * f0_stable + 0.20 * envelope_outlier_clean
        + 0.18 * trend_stable
    )
    attack_quality = _clip01(
        0.42 * attack_clean + 0.28 * attack_fast + 0.18 * hnr + 0.12 * tonal
    )
    release_quality = _clip01(
        0.36 * envelope_stable + 0.24 * envelope_outlier_clean + 0.20 * hnr
        + 0.20 * tonal
    )

    speed_accuracy = record.get('speed_accuracy', {}) or {}
    max_ratio = speed_accuracy.get('max_ratio')
    if max_ratio is None:
        speed_score = 0.65
    else:
        speed_score = 1.0 - _clip01(abs(float(max_ratio) - 1.0) / 0.75)
    peak = record.get('audio_peak')
    peak_score = 0.65 if peak is None else _clip01(float(peak) / 0.12)
    dynamic_accuracy = _clip01(0.65 * speed_score + 0.35 * peak_score)

    # Position-aware provisional overall: early windows lean a little more on
    # attack; late windows lean a little more on release.
    pos = _clip01(window_pos)
    attack_w = max(0.0, 1.0 - 2.0 * pos)
    release_w = max(0.0, 2.0 * pos - 1.0)
    mid_w = max(0.0, 1.0 - attack_w - release_w)
    transient_quality = (
        attack_w * attack_quality + release_w * release_quality
        + mid_w * 0.5 * (attack_quality + release_quality)
    )
    overall = _clip01(
        0.34 * tone_quality + 0.26 * bow_control + 0.20 * transient_quality
        + 0.12 * dynamic_accuracy + 0.08 * voiced
    )

    return {
        'overall': np.float32(overall),
        'tone_quality': np.float32(tone_quality),
        'attack_quality': np.float32(attack_quality),
        'release_quality': np.float32(release_quality),
        'bow_control': np.float32(bow_control),
        'dynamic_accuracy': np.float32(dynamic_accuracy),
    }


def build_physical_features(record) -> np.ndarray:
    """6-dim physical feature vector from a metadata.jsonl record. See module docstring."""
    measured = record.get('measured', {}) or {}
    commanded = record.get('commanded', {}) or {}
    force_contact = record.get('force_contact', {}) or {}

    depth_or_force = force_contact.get('force_mean')
    if depth_or_force is None:
        depth_or_force = commanded.get('depth_m', 0.0)

    force_dev = force_contact.get('force_std') or 0.0

    bow_speed = measured.get('speed_mean', commanded.get('speed', 0.0))

    pos_start = measured.get('bow_pos_start')
    pos_end = measured.get('bow_pos_end')
    if pos_start is not None and pos_end is not None:
        bow_position = (pos_start + pos_end) / 2.0
    else:
        bow_position = 0.5

    torque_a = measured.get('torque_mag_mean', 0.0)
    torque_b = measured.get('torque_mag_max', 0.0)

    return np.array([depth_or_force, force_dev, bow_speed, bow_position,
                      torque_a, torque_b], dtype=np.float32)


def _active_region(record, audio_len_samples, sr):
    timing = record.get('audio_timing', {}) or {}
    start_s = timing.get('stroke_start_s')
    end_s = timing.get('stroke_end_s')
    if start_s is None or end_s is None or end_s <= start_s:
        return 0.0, audio_len_samples / sr

    start_s = max(0.0, start_s - PRE_PAD_S)
    end_s = min(audio_len_samples / sr, end_s + POST_PAD_S)
    return start_s, end_s


def cut_windows(audio, sr, record, window_sec=WINDOW_SEC, hop_sec=TRAIN_HOP_SEC):
    """
    Slice `audio` into `window_sec`-long chunks spanning the recording's
    active region (with attack/release padding), hopping by `hop_sec`.
    Always returns at least one window.

    Returns a list of (chunk, start_n, end_n) tuples; start_n/end_n are the
    window's sample bounds within `audio`, letting callers compute the
    window's position within the active region (see build_training_examples).
    """
    window_n = int(round(window_sec * sr))
    hop_n = int(round(hop_sec * sr))
    start_s, end_s = _active_region(record, len(audio), sr)
    start_n, end_n = int(start_s * sr), int(end_s * sr)

    windows = []
    pos = start_n
    if end_n - start_n <= window_n:
        windows.append((start_n, end_n))
    else:
        while pos + window_n <= end_n:
            windows.append((pos, pos + window_n))
            pos += hop_n
        if not windows or windows[-1][1] < end_n:
            windows.append((end_n - window_n, end_n))

    out = []
    for a, b in windows:
        a = max(0, a)
        chunk = audio[a:b]
        if len(chunk) < window_n:
            chunk = np.pad(chunk, (0, window_n - len(chunk)), mode='reflect') if len(chunk) > 1 \
                else np.zeros(window_n, dtype=np.float32)
        out.append((chunk.astype(np.float32), a, b))
    return out


def build_training_examples(meta_file, audio_dir, sr=af.SR, verbose=True,
                            pseudo_labels=False):
    """
    Returns a list of dicts, one per training window:
        {audio, scalar_features, physical_features, label, labels_multidim,
         window_pos, group_id, condition_label, config}
    `group_id` identifies the source recording (stroke_id + repeat) so
    callers can split train/val without leaking windows from the same
    recording across both sides. `condition_label` / `config` are carried
    through for train_classifier.py's per-condition evaluation breakdowns.
    """
    audio_dir = Path(audio_dir)
    records = load_records(meta_file)
    examples = []
    n_labeled = 0
    n_human = 0
    n_pseudo = 0
    n_missing_audio = 0

    for record in records:
        human_label = annotation_score_01(record)
        if human_label is None and not pseudo_labels:
            continue
        n_labeled += 1
        if human_label is None:
            n_pseudo += 1
        else:
            n_human += 1

        fpath = audio_dir / record['audio_file']
        if not fpath.exists():
            n_missing_audio += 1
            continue

        audio = af.load_audio(fpath, sr=sr)
        physical = build_physical_features(record)
        group_id = f"{record['stroke_id']}_r{record['repeat']}"
        human_multidim = annotation_multidim(record) if human_label is not None else None

        active_start_s, active_end_s = _active_region(record, len(audio), sr)
        active_start_n, active_end_n = int(active_start_s * sr), int(active_end_s * sr)
        active_span_n = max(1, active_end_n - active_start_n)

        for window, a, b in cut_windows(audio, sr, record):
            window_center_fraction = float(((a + b) / 2 - active_start_n) / active_span_n)
            window = af.normalize_peak(window)
            scalar = np.nan_to_num(af.extract_scalar_features(window, sr=sr),
                                   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            if human_label is None:
                multidim = pseudo_labels_from_features(record, scalar, window_center_fraction)
                label = float(multidim['overall'])
                label_source = PSEUDO_LABEL_SOURCE
            elif pseudo_labels:
                multidim = pseudo_labels_from_features(record, scalar, window_center_fraction)
                label = float(multidim['overall'])
                label_source = PSEUDO_LABEL_SOURCE
            else:
                multidim = human_multidim
                label = float(human_label)
                label_source = 'human_annotation'
            examples.append({
                'audio': window,
                'scalar_features': scalar,
                'physical_features': np.nan_to_num(physical, nan=0.0, posinf=0.0, neginf=0.0),
                'label': np.float32(label),
                'labels_multidim': {k: np.float32(v if v is not None else label)
                                    for k, v in multidim.items()},
                'window_pos': np.float32(window_center_fraction),
                'group_id': group_id,
                'condition_label': record.get('condition_label'),
                'config': record.get('config') or (record.get('commanded') or {}).get('config'),
                'label_source': label_source,
            })

    if verbose:
        source_msg = (f"{n_human} human-labeled, {n_pseudo} pseudo-labeled"
                      if pseudo_labels else f"{n_human} annotated")
        print(f"Records: {len(records)} total, {source_msg}, "
              f"{n_missing_audio} missing audio.")
        print(f"Training windows: {len(examples)} from {n_labeled - n_missing_audio} recordings.")
        if pseudo_labels:
            print(f"Pseudo-label source: {PSEUDO_LABEL_SOURCE} (temporary, not human supervision).")

    return examples


def group_train_val_split(examples, val_frac=0.2, seed=0):
    """Split by group_id (source recording) so no recording's windows appear on both sides."""
    rng = np.random.RandomState(seed)
    groups = sorted(set(e['group_id'] for e in examples))
    rng.shuffle(groups)

    n_val_groups = max(1, int(round(len(groups) * val_frac))) if len(groups) > 1 else 0
    val_groups = set(groups[:n_val_groups])

    train_idx = [i for i, e in enumerate(examples) if e['group_id'] not in val_groups]
    val_idx = [i for i, e in enumerate(examples) if e['group_id'] in val_groups]
    return train_idx, val_idx


def compute_normalization_stats(examples, indices):
    """Per-dim mean/std for scalar and physical features, fit on the training split only."""
    scalar = np.stack([examples[i]['scalar_features'] for i in indices])
    physical = np.stack([examples[i]['physical_features'] for i in indices])
    return {
        'scalar_mean': scalar.mean(axis=0), 'scalar_std': scalar.std(axis=0) + 1e-6,
        'physical_mean': physical.mean(axis=0), 'physical_std': physical.std(axis=0) + 1e-6,
    }
