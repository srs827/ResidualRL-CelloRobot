#!/usr/bin/env python3
"""
analyze_dataset_features.py

Runs audio_features.py over Data_Collection/dataset (721 systematically-swept
bow strokes: commanded force x speed x bow-zone, plus a handful of
intentionally "bad" strokes) and produces summary figures showing how the
hand-engineered features respond to the commanded physical parameters.

No human quality annotations exist yet (annotate.py hasn't been run on this
set), so these figures are about feature *behavior*, not feature/quality
correlation -- that's what inspect_features.py is for once annotations exist.

Usage:
    python analyze_dataset_features.py \\
        --meta Data_Collection/dataset/metadata.jsonl \\
        --audio-dir Data_Collection/dataset/audio \\
        --out-dir figures
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import audio_features as af
from dataset import load_records

# Two recording sessions used different commanded-parameter naming/control
# modes (force-controlled vs. depth-controlled, label prefixes "F" vs "D"),
# so condition_label text isn't a reliable grouping key across the whole
# dataset. measured.force_mean / measured.speed_mean (actual sensed values)
# are present for ~all records regardless of session, so we bin on those
# instead and quantile-bin them into 5 ordered levels per sweep variable.
LEVEL_NAMES = ['very_low', 'low', 'medium', 'high', 'very_high']

# Zone is well-defined by commanded bow_start/bow_end for every record.
ZONE_RANGES = {
    (0.05, 0.35): 'frog',
    (0.25, 0.55): 'lower_middle',
    (0.45, 0.75): 'upper_middle',
    (0.65, 0.95): 'tip',
}
ZONE_ORDER = ['frog', 'lower_middle', 'upper_middle', 'tip']


def zone_from_commanded(record):
    c = record.get('commanded', {}) or {}
    start, end = c.get('bow_start'), c.get('bow_end')
    if start is None or end is None:
        return None
    key = (round(start, 2), round(end, 2))
    return ZONE_RANGES.get(key)


def quantile_levels(values, n_bins=5, names=LEVEL_NAMES):
    """Map each value to one of `names` by quantile bin, ignoring NaNs."""
    values = np.asarray(values, dtype=np.float64)
    levels = np.full(len(values), None, dtype=object)
    valid = ~np.isnan(values)
    if valid.sum() < n_bins:
        return levels
    edges = np.quantile(values[valid], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_idx = np.digitize(values[valid], edges[1:-1], right=False)
    levels[valid] = np.array(names)[bin_idx]
    return levels


def extract_table(records, audio_dir, max_records=None, cache_path=None):
    if cache_path and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    if max_records:
        records = records[:max_records]

    rows, zone, cond_type, force_mean, speed_mean, stroke_id = [], [], [], [], [], []
    n_missing = 0

    for i, record in enumerate(records):
        fpath = audio_dir / record['audio_file']
        if not fpath.exists():
            n_missing += 1
            continue
        audio = af.normalize_peak(af.load_audio(fpath))
        rows.append(af.extract_scalar_features(audio))

        measured = record.get('measured', {}) or {}
        zone.append(zone_from_commanded(record))
        cond_type.append(record.get('condition_type'))
        force_mean.append(measured.get('force_mean', np.nan))
        speed_mean.append(measured.get('speed_mean', np.nan))
        stroke_id.append(record.get('stroke_id'))

        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(records)} processed")

    print(f"Extracted features for {len(rows)} recordings ({n_missing} missing audio).")

    table = {
        'X': np.stack(rows),
        'feature_names': np.array(af.SCALAR_FEATURE_NAMES),
        'zone': np.array(zone, dtype=object),
        'condition_type': np.array(cond_type, dtype=object),
        'force_mean': np.array(force_mean, dtype=np.float64),
        'speed_mean': np.array(speed_mean, dtype=np.float64),
        'stroke_id': np.array(stroke_id, dtype=object),
    }
    if cache_path:
        np.savez(cache_path, **table)
    return table


def _box_by_category(ax, X, names, feat_name, categories, order, title):
    j = list(names).index(feat_name)
    col = X[:, j]
    data, labels = [], []
    for cat in order:
        mask = categories == cat
        if mask.sum() == 0:
            continue
        data.append(col[mask])
        labels.append(cat)
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', rotation=40, labelsize=8)
    ax.set_ylabel(feat_name, fontsize=8)


def fig_feature_distributions(table, out_dir):
    X, names = table['X'], table['feature_names']
    keys = ['hnr_db_mean', 'flatness_mean', 'f0_stability_cents', 'voiced_fraction',
            'attack_time_s', 'attack_overshoot', 'envelope_cv', 'envelope_outlier_rate']
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, key in zip(axes.flat, keys):
        j = list(names).index(key)
        ax.hist(X[:, j], bins=40, color='steelblue', edgecolor='white')
        ax.set_title(key, fontsize=10)
    fig.suptitle('Hand-engineered feature distributions, all systematic-sweep strokes (n=%d)' % len(X))
    fig.tight_layout()
    fig.savefig(out_dir / '01_feature_distributions.png', dpi=150)
    plt.close(fig)


def fig_force_sweep(table, out_dir):
    X, names = table['X'], table['feature_names']
    mask = table['condition_type'] == 'systematic'
    X = X[mask]
    force_level = quantile_levels(table['force_mean'][mask])
    keys = ['hnr_db_mean', 'flatness_mean', 'attack_overshoot', 'envelope_cv']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key in zip(axes, keys):
        _box_by_category(ax, X, names, key, force_level, LEVEL_NAMES, key)
    fig.suptitle('Feature vs. measured bow force, quintile-binned (systematic sweep)')
    fig.tight_layout()
    fig.savefig(out_dir / '02_feature_vs_force.png', dpi=150)
    plt.close(fig)


def fig_speed_sweep(table, out_dir):
    X, names = table['X'], table['feature_names']
    mask = table['condition_type'] == 'systematic'
    X = X[mask]
    speed_level = quantile_levels(table['speed_mean'][mask])
    keys = ['hnr_db_mean', 'flatness_mean', 'attack_time_s', 'f0_stability_cents']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key in zip(axes, keys):
        _box_by_category(ax, X, names, key, speed_level, LEVEL_NAMES, key)
    fig.suptitle('Feature vs. measured bow speed, quintile-binned (systematic sweep)')
    fig.tight_layout()
    fig.savefig(out_dir / '03_feature_vs_speed.png', dpi=150)
    plt.close(fig)


def fig_zone_sweep(table, out_dir):
    X, names = table['X'], table['feature_names']
    mask = table['condition_type'] == 'systematic'
    X, zone = X[mask], table['zone'][mask]
    keys = ['hnr_db_mean', 'flatness_mean', 'rms_mean', 'envelope_cv']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key in zip(axes, keys):
        _box_by_category(ax, X, names, key, zone, ZONE_ORDER, key)
    fig.suptitle('Feature vs. bow zone (systematic sweep)')
    fig.tight_layout()
    fig.savefig(out_dir / '04_feature_vs_zone.png', dpi=150)
    plt.close(fig)


def fig_good_vs_bad(table, out_dir):
    X, names = table['X'], table['feature_names']
    cond = table['condition_type']
    keys = ['hnr_db_mean', 'flatness_mean', 'attack_overshoot', 'envelope_cv',
            'voiced_fraction', 'rms_std']
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, key in zip(axes.flat, keys):
        j = names.tolist().index(key)
        good = X[cond == 'systematic', j]
        bad = X[cond == 'bad', j]
        ax.boxplot([good, bad], tick_labels=['systematic', 'bad (intentional)'], showfliers=False)
        ax.set_title(key, fontsize=10)
    fig.suptitle('Systematic sweep vs. intentionally bad strokes')
    fig.tight_layout()
    fig.savefig(out_dir / '05_good_vs_bad.png', dpi=150)
    plt.close(fig)


def fig_correlation_heatmap(table, out_dir):
    X, names = table['X'], table['feature_names']
    keep = [n for n in names if not n.startswith('mfcc')]
    idx = [names.tolist().index(n) for n in keep]
    Xc = X[:, idx]
    corr = np.corrcoef(Xc.T)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels(keep, rotation=90, fontsize=7)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(keep, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Correlation between non-MFCC scalar features')
    fig.tight_layout()
    fig.savefig(out_dir / '06_feature_correlation.png', dpi=150)
    plt.close(fig)


def _most_typical_systematic_record(records):
    """The systematic-sweep record whose measured force/speed are closest to
    the sweep's median, as a stand-in 'normal' stroke (label text isn't a
    reliable pick across sessions -- see note on LEVEL_NAMES above)."""
    systematic = [r for r in records if r.get('condition_type') == 'systematic']
    force = np.array([(r.get('measured') or {}).get('force_mean', np.nan) for r in systematic])
    speed = np.array([(r.get('measured') or {}).get('speed_mean', np.nan) for r in systematic])
    valid = ~(np.isnan(force) | np.isnan(speed))
    force_z = (force[valid] - np.nanmedian(force)) / (np.nanstd(force) + 1e-9)
    speed_z = (speed[valid] - np.nanmedian(speed)) / (np.nanstd(speed) + 1e-9)
    dist = force_z ** 2 + speed_z ** 2
    valid_records = [r for r, v in zip(systematic, valid) if v]
    return valid_records[int(np.argmin(dist))]


def fig_example_spectrograms(records, audio_dir, out_dir):
    by_label = {}
    for r in records:
        by_label.setdefault(r.get('condition_label'), r)

    chosen = [('typical (median force/speed)', _most_typical_systematic_record(records))]
    for lbl in ['bad_too_fast', 'bad_too_slow', 'bad_barely_touching', 'bad_heavy_press_slow']:
        if lbl in by_label:
            chosen.append((lbl, by_label[lbl]))

    if len(chosen) < 2:
        print("Skipping spectrogram figure: expected example labels not found.")
        return

    fig, axes = plt.subplots(1, len(chosen), figsize=(4.2 * len(chosen), 4))
    if len(chosen) == 1:
        axes = [axes]
    for ax, (lbl, record) in zip(axes, chosen):
        fpath = audio_dir / record['audio_file']
        audio = af.normalize_peak(af.load_audio(fpath))
        mel = af.compute_log_mel_spectrogram(audio)
        ax.imshow(mel, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel('frame')
        ax.set_ylabel('mel bin')
    fig.suptitle('Example log-mel spectrograms')
    fig.tight_layout()
    fig.savefig(out_dir / '07_example_spectrograms.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta', default='Data_Collection/dataset/metadata.jsonl')
    parser.add_argument('--audio-dir', default='Data_Collection/dataset/audio')
    parser.add_argument('--out-dir', default='figures')
    parser.add_argument('--max-records', type=int, default=None)
    parser.add_argument('--cache', default='Data_Collection/dataset/_feature_cache.npz',
                         help="Path to cache extracted features so re-runs skip recomputation.")
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.meta)
    print(f"Loaded {len(records)} records from {args.meta}")

    cache_path = None if args.no_cache else args.cache
    table = extract_table(records, audio_dir, max_records=args.max_records, cache_path=cache_path)

    fig_feature_distributions(table, out_dir)
    fig_force_sweep(table, out_dir)
    fig_speed_sweep(table, out_dir)
    fig_zone_sweep(table, out_dir)
    fig_good_vs_bad(table, out_dir)
    fig_correlation_heatmap(table, out_dir)
    fig_example_spectrograms(records, audio_dir, out_dir)

    print(f"\nFigures written to {out_dir}/")


if __name__ == '__main__':
    main()
