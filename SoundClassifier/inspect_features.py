#!/usr/bin/env python3
"""
inspect_features.py

Standalone diagnostic: runs the hand-engineered audio features
(audio_features.py) over every recording in a metadata.jsonl, and reports
which features actually correlate with human quality ratings (annotate.py).
Useful before/while training the CNN to sanity-check the feature
extraction itself (e.g. "does low HNR really track with the recordings
people rated as poor?") independent of any model.

Usage:
    python inspect_features.py \\
        --meta Data_Collection/dataset_a_configs/metadata.jsonl \\
        --audio-dir Data_Collection/dataset_a_configs/audio
"""

import argparse
from pathlib import Path

import numpy as np

import audio_features as af
from dataset import load_records, annotation_score_01


def main():
    parser = argparse.ArgumentParser(description="Inspect hand-engineered audio features")
    parser.add_argument('--meta', required=True)
    parser.add_argument('--audio-dir', required=True)
    parser.add_argument('--max-records', type=int, default=None,
                         help="Cap how many recordings to process (for a quick look).")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    records = load_records(args.meta)
    if args.max_records:
        records = records[:args.max_records]

    feature_rows = []
    labels = []
    n_missing = 0

    for record in records:
        fpath = audio_dir / record['audio_file']
        if not fpath.exists():
            n_missing += 1
            continue
        audio = af.normalize_peak(af.load_audio(fpath))
        feature_rows.append(af.extract_scalar_features(audio))
        labels.append(annotation_score_01(record))

    if not feature_rows:
        print("No audio found.")
        return

    X = np.stack(feature_rows)
    y = np.array([l if l is not None else np.nan for l in labels])
    annotated_mask = ~np.isnan(y)

    print(f"\nProcessed {len(feature_rows)} recordings ({n_missing} missing audio).")
    print(f"Annotated: {annotated_mask.sum()} / {len(feature_rows)}\n")

    header = f"{'feature':24s} {'mean':>10s} {'std':>10s}"
    if annotated_mask.sum() >= 3:
        header += f" {'corr_w_score':>14s}"
    print(header)
    print("-" * len(header))

    for j, name in enumerate(af.SCALAR_FEATURE_NAMES):
        col = X[:, j]
        line = f"{name:24s} {np.mean(col):10.4f} {np.std(col):10.4f}"
        if annotated_mask.sum() >= 3:
            col_ann = col[annotated_mask]
            y_ann = y[annotated_mask]
            if np.std(col_ann) > 1e-9:
                corr = float(np.corrcoef(col_ann, y_ann)[0, 1])
            else:
                corr = float('nan')
            line += f" {corr:14.3f}"
        print(line)

    if annotated_mask.sum() < 3:
        print("\nFewer than 3 annotated recordings -- correlations not shown. "
              "Run annotate.py to add more ratings.")


if __name__ == '__main__':
    main()
