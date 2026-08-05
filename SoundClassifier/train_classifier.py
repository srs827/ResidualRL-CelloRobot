#!/usr/bin/env python3
"""
train_classifier.py

Trains MelQualityCNN on annotated recordings from dataset_a_configs (or any
metadata.jsonl + audio/ dir in the same format) and saves a checkpoint that
classifier.py's BowingQualityClassifier loads for real-time inference.

validate_dataset.py's own readiness checklist wants 50+ annotated
recordings before training is meaningful; as of writing dataset_a_configs
has 7. Below ~20 annotated recordings (groups), this script runs
leave-one-group-out cross-validation instead of a single train/val split,
since a single held-out group would be too noisy to mean anything -- still
not a substitute for more annotations, just the least-bad estimate
available at this sample size.

Usage:
    python train_classifier.py \\
        --meta Data_Collection/dataset_a_configs/metadata.jsonl \\
        --audio-dir Data_Collection/dataset_a_configs/audio \\
        --out checkpoints/quality_cnn.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import dataset as ds
import quality_classifier as qc

MIN_ANNOTATED_FOR_SINGLE_SPLIT = 20
MIN_RECORDINGS_PER_CONFIG = 5
BAD_COND_GAP_WARN = 0.15

RANK_WEIGHT = 0.5
RANK_MARGIN = 0.1
RANK_MIN_GAP = 0.15
AUX_WEIGHT = 0.3


# ---------------------------------------------------------------- metrics --

def rankdata(a):
    """Average ranks (1-based), ties broken by mean rank -- avoids a scipy dependency."""
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[sorter] = np.arange(1, len(a) + 1)

    sorted_a = a[sorter]
    tie_starts = np.flatnonzero(np.r_[True, sorted_a[1:] != sorted_a[:-1], True])
    for lo, hi in zip(tie_starts[:-1], tie_starts[1:]):
        if hi - lo > 1:
            ranks[sorter[lo:hi]] = ranks[sorter[lo:hi]].mean()
    return ranks


def spearman_corr(x, y):
    """Spearman rank correlation. NaN if fewer than 2 points or either side is constant."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float('nan')
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def aggregate_by_group(preds, labels, groups):
    """Mean prediction/label per group_id -> (sorted_group_ids, group_preds, group_labels)."""
    pred_map, label_map = {}, {}
    for p, l, g in zip(preds, labels, groups):
        pred_map.setdefault(g, []).append(p)
        label_map.setdefault(g, []).append(l)
    group_ids = sorted(pred_map)
    group_preds = [float(np.mean(pred_map[g])) for g in group_ids]
    group_labels = [float(np.mean(label_map[g])) for g in group_ids]
    return group_ids, group_preds, group_labels


# ------------------------------------------------------------------ loss ---

def position_weight(field, window_pos):
    """Only supervise attack_quality on early windows / release_quality on late windows."""
    if field == 'attack_quality':
        return torch.clamp(1.0 - 2.0 * window_pos, 0.0, 1.0)
    if field == 'release_quality':
        return torch.clamp(2.0 * window_pos - 1.0, 0.0, 1.0)
    return torch.ones_like(window_pos)


def weighted_mse(pred, target, weight):
    return (weight * (pred - target) ** 2).mean()


def pairwise_ranking_loss(pred, target, margin=RANK_MARGIN, min_gap=RANK_MIN_GAP):
    """In-batch hinge ranking loss: for pairs with |label gap| > min_gap, penalize
    predictions that don't preserve the label ordering by at least `margin`."""
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)
    upper = torch.triu(torch.ones_like(diff_target, dtype=torch.bool), diagonal=1)
    mask = upper & (diff_target.abs() > min_gap)
    if not mask.any():
        return pred.new_zeros(())
    sign = torch.sign(diff_target)
    hinge = torch.clamp(margin - sign * diff_pred, min=0.0)
    return hinge[mask].mean()


def compute_loss(pred, targets, window_pos):
    """
    pred, targets: {field: (B,) tensor} for every ds.TIER1_FIELDS entry.
    Returns (total_loss, overall_mse_value) -- overall_mse_value is a plain
    float for logging/model-selection, decoupled from the ranking/aux terms.
    """
    mse_fn = nn.MSELoss()
    overall_mse = mse_fn(pred['overall'], targets['overall'])
    rank_loss = pairwise_ranking_loss(pred['overall'], targets['overall'])

    aux_fields = qc.AUX_FIELDS
    if aux_fields:
        aux_weight_each = AUX_WEIGHT / len(aux_fields)
        aux_loss = sum(
            aux_weight_each * weighted_mse(pred[f], targets[f], position_weight(f, window_pos))
            for f in aux_fields
        )
    else:
        aux_loss = pred['overall'].new_zeros(())

    total = overall_mse + RANK_WEIGHT * rank_loss + aux_loss
    return total, overall_mse.item()


# --------------------------------------------------------------- dataset --

class WindowDataset(Dataset):
    def __init__(self, examples, indices, scalar_norm, physical_norm):
        self.examples = examples
        self.indices = indices
        self.scalar_norm = scalar_norm
        self.physical_norm = physical_norm

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ex = self.examples[self.indices[i]]
        mel = qc.audio_to_mel_tensor(ex['audio'])
        scalar = self.scalar_norm(ex['scalar_features'])
        physical = self.physical_norm(ex['physical_features'])
        multidim = np.array([ex['labels_multidim'][f] for f in ds.TIER1_FIELDS], dtype=np.float32)
        return (
            torch.from_numpy(mel.astype(np.float32)),
            torch.from_numpy(scalar.astype(np.float32)),
            torch.from_numpy(physical.astype(np.float32)),
            torch.tensor(ex['window_pos'], dtype=torch.float32),
            torch.tensor(ex['label'], dtype=torch.float32),
            torch.from_numpy(multidim),
            ex['group_id'],
        )


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _targets_from_multidim(multidim):
    """(B, len(TIER1_FIELDS)) tensor -> {field: (B,) tensor}."""
    return {field: multidim[:, i] for i, field in enumerate(ds.TIER1_FIELDS)}


def train_one_split(examples, train_idx, val_idx, device, epochs, batch_size, lr, verbose=True):
    stats = ds.compute_normalization_stats(examples, train_idx)
    scalar_norm = qc.FeatureNormalizer(stats['scalar_mean'], stats['scalar_std'])
    physical_norm = qc.FeatureNormalizer(stats['physical_mean'], stats['physical_std'])

    train_loader = DataLoader(WindowDataset(examples, train_idx, scalar_norm, physical_norm),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(examples, val_idx, scalar_norm, physical_norm),
                             batch_size=batch_size, shuffle=False) if val_idx else None

    model = qc.MelQualityCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = float('-inf')     # selection metric, higher is better
    best_val_mse = float('inf')
    best_window_rho = float('nan')
    best_record_rho = float('nan')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_mse_sum, n_train = 0.0, 0
        for mel, scalar, physical, window_pos, label, multidim, _group_id in train_loader:
            mel, scalar, physical, window_pos, label, multidim = (
                t.to(device) for t in (mel, scalar, physical, window_pos, label, multidim))
            targets = _targets_from_multidim(multidim)
            optimizer.zero_grad()
            pred = model.forward_multitask(mel, scalar, physical, window_pos)
            loss, overall_mse = compute_loss(pred, targets, window_pos)
            loss.backward()
            optimizer.step()
            b = len(label)
            train_mse_sum += overall_mse * b
            n_train += b
        train_mse = train_mse_sum / n_train

        if val_loader is not None:
            model.eval()
            val_mse_sum, n_val = 0.0, 0
            window_preds, window_labels, window_groups = [], [], []
            with torch.no_grad():
                for mel, scalar, physical, window_pos, label, multidim, group_id in val_loader:
                    mel, scalar, physical, window_pos, label, multidim = (
                        t.to(device) for t in (mel, scalar, physical, window_pos, label, multidim))
                    targets = _targets_from_multidim(multidim)
                    pred = model.forward_multitask(mel, scalar, physical, window_pos)
                    _, overall_mse = compute_loss(pred, targets, window_pos)
                    b = len(label)
                    val_mse_sum += overall_mse * b
                    n_val += b
                    window_preds.extend(pred['overall'].cpu().numpy().tolist())
                    window_labels.extend(targets['overall'].cpu().numpy().tolist())
                    window_groups.extend(group_id)
            val_mse = val_mse_sum / max(1, n_val)
            window_rho = spearman_corr(window_preds, window_labels)
            _, rec_preds, rec_labels = aggregate_by_group(window_preds, window_labels, window_groups)
            record_rho = spearman_corr(rec_preds, rec_labels) if len(rec_preds) >= 2 else float('nan')
        else:
            val_mse, window_rho, record_rho = train_mse, float('nan'), float('nan')

        # Model selection: recording-level Spearman rho when it's computable
        # (needs >=2 distinct held-out recordings with varying labels);
        # otherwise fall back to minimizing val MSE.
        selection_score = record_rho if not np.isnan(record_rho) else -val_mse
        if selection_score > best_score:
            best_score = selection_score
            best_val_mse = val_mse
            best_window_rho = window_rho
            best_record_rho = record_rho
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch+1:3d}/{epochs}  train_mse={train_mse:.4f}  val_mse={val_mse:.4f}  "
                  f"window_rho={window_rho:.3f}  record_rho={record_rho:.3f}")

    model.load_state_dict(best_state)
    return model, scalar_norm, physical_norm, best_val_mse, best_window_rho, best_record_rho


def run_loego_cv(examples, device, epochs, batch_size, lr):
    """Leave-one-group-out CV: one fold per source recording. Reports mean held-out MSE
    plus the recording-level Spearman rho pooled across all folds' held-out predictions
    (a single fold only has one recording, so within-fold rho is undefined)."""
    groups = sorted(set(e['group_id'] for e in examples))
    print(f"\nLeave-one-group-out CV across {len(groups)} recordings "
          f"(too few annotations for a stable single split):")

    fold_mses = []
    group_preds, group_labels = [], []
    for g in groups:
        train_idx = [i for i, e in enumerate(examples) if e['group_id'] != g]
        val_idx = [i for i, e in enumerate(examples) if e['group_id'] == g]
        model, scalar_norm, physical_norm, val_mse, _, _ = train_one_split(
            examples, train_idx, val_idx, device, epochs, batch_size, lr, verbose=False)
        fold_mses.append(val_mse)

        val_loader = DataLoader(WindowDataset(examples, val_idx, scalar_norm, physical_norm),
                                 batch_size=batch_size, shuffle=False)
        model.eval()
        preds = []
        with torch.no_grad():
            for mel, scalar, physical, window_pos, _label, _multidim, _group_id in val_loader:
                mel, scalar, physical, window_pos = (
                    t.to(device) for t in (mel, scalar, physical, window_pos))
                preds.extend(model(mel, scalar, physical, window_pos).cpu().numpy().tolist())
        group_preds.append(float(np.mean(preds)))
        group_labels.append(float(examples[val_idx[0]]['label']))

        print(f"  held out {g:50s}  val_mse={val_mse:.4f}")

    pooled_rho = spearman_corr(group_preds, group_labels)
    print(f"\nLOGO-CV mean val MSE: {np.mean(fold_mses):.4f}  (std {np.std(fold_mses):.4f})")
    print(f"LOGO-CV recording-level Spearman rho (pooled across folds): {pooled_rho:.3f}")
    print("This is a sample-size sanity check, not a model you should deploy from directly.")


# ------------------------------------------------------------- evaluation --

def evaluate_bad_condition_separation(examples, model, scalar_norm, physical_norm, device, batch_size):
    """Mean predicted overall for records whose condition_label starts with bad_ vs the rest."""
    loader = DataLoader(WindowDataset(examples, list(range(len(examples))), scalar_norm, physical_norm),
                         batch_size=batch_size, shuffle=False)
    model.eval()
    preds, groups = [], []
    with torch.no_grad():
        for mel, scalar, physical, window_pos, _label, _multidim, group_id in loader:
            mel, scalar, physical, window_pos = (
                t.to(device) for t in (mel, scalar, physical, window_pos))
            preds.extend(model(mel, scalar, physical, window_pos).cpu().numpy().tolist())
            groups.extend(group_id)

    pred_by_group = {}
    for p, g in zip(preds, groups):
        pred_by_group.setdefault(g, []).append(p)

    condition_by_group = {e['group_id']: e.get('condition_label') for e in examples}
    bad_scores, ok_scores = [], []
    for g, group_preds in pred_by_group.items():
        cond = condition_by_group.get(g) or ''
        mean_pred = float(np.mean(group_preds))
        (bad_scores if cond.startswith('bad_') else ok_scores).append(mean_pred)

    if not bad_scores or not ok_scores:
        print("\nBad-condition separation: skipped (need both bad_* and systematic recordings).")
        return

    gap = float(np.mean(ok_scores) - np.mean(bad_scores))
    print(f"\nBad-condition separation: systematic mean={np.mean(ok_scores):.3f}  "
          f"bad_* mean={np.mean(bad_scores):.3f}  gap={gap:.3f}")
    if gap < BAD_COND_GAP_WARN:
        print(f"  WARNING: gap below {BAD_COND_GAP_WARN} -- model may not be separating known-bad strokes.")


def evaluate_per_config_breakdown(examples, model, scalar_norm, physical_norm, device, batch_size):
    """Recording-level Spearman rho broken down by config, for configs with >=5 annotated recordings."""
    loader = DataLoader(WindowDataset(examples, list(range(len(examples))), scalar_norm, physical_norm),
                         batch_size=batch_size, shuffle=False)
    model.eval()
    preds, labels, groups = [], [], []
    with torch.no_grad():
        for mel, scalar, physical, window_pos, label, _multidim, group_id in loader:
            mel, scalar, physical, window_pos = (
                t.to(device) for t in (mel, scalar, physical, window_pos))
            preds.extend(model(mel, scalar, physical, window_pos).cpu().numpy().tolist())
            labels.extend(label.numpy().tolist())
            groups.extend(group_id)

    config_by_group = {e['group_id']: e.get('config') for e in examples}
    print("\nPer-config breakdown (recording-level Spearman rho, configs with "
          f">= {MIN_RECORDINGS_PER_CONFIG} annotated recordings):")
    by_config = {}
    for p, l, g in zip(preds, labels, groups):
        cfg = config_by_group.get(g) or 'unknown'
        by_config.setdefault(cfg, {'preds': [], 'labels': [], 'groups': []})
        by_config[cfg]['preds'].append(p)
        by_config[cfg]['labels'].append(l)
        by_config[cfg]['groups'].append(g)

    any_reported = False
    for cfg in sorted(by_config):
        d = by_config[cfg]
        group_ids, group_preds, group_labels = aggregate_by_group(d['preds'], d['labels'], d['groups'])
        if len(group_ids) < MIN_RECORDINGS_PER_CONFIG:
            continue
        any_reported = True
        rho = spearman_corr(group_preds, group_labels)
        print(f"  {cfg:12s}  n_recordings={len(group_ids):3d}  rho={rho:.3f}")
    if not any_reported:
        print("  (no config has enough annotated recordings yet)")


def run_ridge_baseline(examples, train_idx, val_idx):
    """Ridge(alpha=1.0) on scalar_features ++ physical_features, same splits as the CNN."""
    from sklearn.linear_model import Ridge

    stats = ds.compute_normalization_stats(examples, train_idx)
    scalar_norm = qc.FeatureNormalizer(stats['scalar_mean'], stats['scalar_std'])
    physical_norm = qc.FeatureNormalizer(stats['physical_mean'], stats['physical_std'])

    def features(idx):
        X = np.stack([
            np.concatenate([scalar_norm(examples[i]['scalar_features']),
                             physical_norm(examples[i]['physical_features'])])
            for i in idx
        ])
        y = np.array([examples[i]['label'] for i in idx], dtype=np.float64)
        groups = [examples[i]['group_id'] for i in idx]
        return X, y, groups

    X_train, y_train, _ = features(train_idx)
    X_val, y_val, groups_val = features(val_idx)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    _, rec_preds, rec_labels = aggregate_by_group(preds, y_val, groups_val)
    rho = spearman_corr(rec_preds, rec_labels) if len(rec_preds) >= 2 else float('nan')
    print(f"\nRidge baseline (scalar+physical features only): recording-level Spearman rho={rho:.3f}")
    return rho


# ---------------------------------------------------------------- main ----

def main():
    parser = argparse.ArgumentParser(description="Train the bowing-quality CNN classifier")
    parser.add_argument('--meta', required=True)
    parser.add_argument('--audio-dir', required=True)
    parser.add_argument('--out', default='checkpoints/quality_cnn.pt')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--force-single-split', action='store_true',
                         help="Skip LOGO-CV even with few annotations.")
    parser.add_argument('--baseline', action='store_true',
                         help="Also train a Ridge baseline on scalar+physical features "
                              "and compare its recording-level Spearman rho to the CNN's.")
    parser.add_argument('--pseudo-labels', action='store_true',
                         help="Train from deterministic audio/metadata pseudo-labels instead of "
                              "requiring human annotations. Temporary bootstrap mode for RL.")
    args = parser.parse_args()

    examples = ds.build_training_examples(args.meta, args.audio_dir,
                                          pseudo_labels=args.pseudo_labels)
    if not examples:
        print("No labeled recordings with audio found -- nothing to train on. "
              "Run annotate.py first, or pass --pseudo-labels for temporary heuristic labels.")
        return

    n_groups = len(set(e['group_id'] for e in examples))
    device = pick_device(args.device)
    print(f"Device: {device}")

    if n_groups < MIN_ANNOTATED_FOR_SINGLE_SPLIT and not args.force_single_split:
        run_loego_cv(examples, device, args.epochs, args.batch_size, args.lr)
        print(f"\nAnnotate {MIN_ANNOTATED_FOR_SINGLE_SPLIT - n_groups} more recordings "
              f"(or pass --force-single-split) to save a deployable checkpoint.")
        return

    train_idx, val_idx = ds.group_train_val_split(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"Train windows: {len(train_idx)}  Val windows: {len(val_idx)}  "
          f"(from {n_groups} annotated recordings)")

    model, scalar_norm, physical_norm, best_val_mse, window_rho, record_rho = train_one_split(
        examples, train_idx, val_idx, device, args.epochs, args.batch_size, args.lr)
    print(f"\nBest val MSE: {best_val_mse:.4f}  window_rho={window_rho:.3f}  record_rho={record_rho:.3f}")

    evaluate_bad_condition_separation(examples, model, scalar_norm, physical_norm, device, args.batch_size)
    evaluate_per_config_breakdown(examples, model, scalar_norm, physical_norm, device, args.batch_size)

    if args.baseline:
        ridge_rho = run_ridge_baseline(examples, train_idx, val_idx)
        if not np.isnan(ridge_rho) and not np.isnan(record_rho) and record_rho <= ridge_rho:
            print(f"  WARNING: CNN record_rho={record_rho:.3f} does not beat the Ridge "
                  f"baseline (rho={ridge_rho:.3f}) -- the audio model isn't adding value yet.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scalar_norm': scalar_norm.state_dict(),
        'physical_norm': physical_norm.state_dict(),
        'target_frames': qc.TARGET_FRAMES,
        'n_annotated_recordings': n_groups,
        'best_val_mse': best_val_mse,
        'best_val_record_rho': record_rho,
        'aux_fields': ds.TIER1_FIELDS,
        'schema_version': 2,
        'label_source': ds.PSEUDO_LABEL_SOURCE if args.pseudo_labels else 'human_annotation',
        'pseudo_labels': bool(args.pseudo_labels),
    }, out_path)
    print(f"Saved checkpoint to {out_path}")


if __name__ == '__main__':
    main()
