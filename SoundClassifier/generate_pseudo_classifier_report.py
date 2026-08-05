#!/usr/bin/env python3
"""
Generate presentation figures for the temporary pseudo-labeled sound classifier.

The checkpoint is trained against deterministic heuristic labels, not human
ratings. These figures therefore describe how well the CNN distills that
temporary teacher and how the provisional scores distribute across the current
recording set.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import dataset as ds
import quality_classifier as qc
from train_classifier import aggregate_by_group, spearman_corr

_mpl_cache = Path(__file__).resolve().parent / "figures" / ".matplotlib_cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ReportDataset(Dataset):
    def __init__(self, examples, scalar_norm, physical_norm):
        self.examples = examples
        self.scalar_norm = scalar_norm
        self.physical_norm = physical_norm

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        mel = qc.audio_to_mel_tensor(ex["audio"])
        scalar = self.scalar_norm(ex["scalar_features"])
        physical = self.physical_norm(ex["physical_features"])
        multidim = np.array([ex["labels_multidim"][f] for f in ds.TIER1_FIELDS], dtype=np.float32)
        return (
            torch.from_numpy(mel.astype(np.float32)),
            torch.from_numpy(scalar.astype(np.float32)),
            torch.from_numpy(physical.astype(np.float32)),
            torch.tensor(ex["window_pos"], dtype=torch.float32),
            torch.from_numpy(multidim),
            i,
        )


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = qc.MelQualityCNN(multitask=ckpt.get("schema_version", 1) >= 2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scalar_norm = qc.FeatureNormalizer.from_state_dict(ckpt["scalar_norm"])
    physical_norm = qc.FeatureNormalizer.from_state_dict(ckpt["physical_norm"])
    return ckpt, model, scalar_norm, physical_norm


def collect_predictions(examples, model, scalar_norm, physical_norm, device, batch_size):
    loader = DataLoader(ReportDataset(examples, scalar_norm, physical_norm),
                        batch_size=batch_size, shuffle=False)
    preds = {field: [] for field in ds.TIER1_FIELDS}
    labels = {field: [] for field in ds.TIER1_FIELDS}
    indices = []

    with torch.no_grad():
        for mel, scalar, physical, window_pos, multidim, idx in loader:
            mel = mel.to(device)
            scalar = scalar.to(device)
            physical = physical.to(device)
            window_pos = window_pos.to(device)
            out = model.forward_multitask(mel, scalar, physical, window_pos)
            for field_i, field in enumerate(ds.TIER1_FIELDS):
                preds[field].extend(out[field].cpu().numpy().tolist())
                labels[field].extend(multidim[:, field_i].numpy().tolist())
            indices.extend(idx.numpy().tolist())

    return preds, labels, indices


def aggregate_records(examples, preds, labels):
    rows = {}
    for i, ex in enumerate(examples):
        group = ex["group_id"]
        row = rows.setdefault(group, {
            "group_id": group,
            "config": ex.get("config") or "unknown",
            "condition_label": ex.get("condition_label") or "",
            "n_windows": 0,
        })
        row["n_windows"] += 1
        for field in ds.TIER1_FIELDS:
            row.setdefault(f"pred_{field}", []).append(preds[field][i])
            row.setdefault(f"label_{field}", []).append(labels[field][i])

    out = []
    for row in rows.values():
        for field in ds.TIER1_FIELDS:
            row[f"pred_{field}"] = float(np.mean(row[f"pred_{field}"]))
            row[f"label_{field}"] = float(np.mean(row[f"label_{field}"]))
        out.append(row)
    return sorted(out, key=lambda r: r["group_id"])


def save_record_csv(rows, path):
    fields = ["group_id", "config", "condition_label", "n_windows"]
    for field in ds.TIER1_FIELDS:
        fields.extend([f"label_{field}", f"pred_{field}"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def compute_metrics(rows):
    metrics = {}
    for field in ds.TIER1_FIELDS:
        y = np.array([r[f"label_{field}"] for r in rows], dtype=np.float64)
        p = np.array([r[f"pred_{field}"] for r in rows], dtype=np.float64)
        metrics[field] = {
            "record_mse": float(np.mean((p - y) ** 2)),
            "record_spearman": spearman_corr(p, y),
            "label_mean": float(np.mean(y)),
            "pred_mean": float(np.mean(p)),
        }

    by_config = {}
    for config in sorted(set(r["config"] for r in rows)):
        subset = [r for r in rows if r["config"] == config]
        y = np.array([r["label_overall"] for r in subset], dtype=np.float64)
        p = np.array([r["pred_overall"] for r in subset], dtype=np.float64)
        by_config[config] = {
            "n_recordings": len(subset),
            "overall_spearman": spearman_corr(p, y),
            "overall_label_mean": float(np.mean(y)),
            "overall_pred_mean": float(np.mean(p)),
        }
    metrics["by_config"] = by_config
    return metrics


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)


def plot_pred_vs_label(rows, metrics, out_path):
    y = np.array([r["label_overall"] for r in rows])
    p = np.array([r["pred_overall"] for r in rows])
    configs = sorted(set(r["config"] for r in rows))
    cmap = plt.get_cmap("tab10")
    color = {cfg: cmap(i) for i, cfg in enumerate(configs)}

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for cfg in configs:
        idx = [i for i, r in enumerate(rows) if r["config"] == cfg]
        ax.scatter(y[idx], p[idx], s=32, alpha=0.78, label=cfg, color=color[cfg])
    lo = min(float(y.min()), float(p.min())) - 0.02
    hi = max(float(y.max()), float(p.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.2, linestyle="--", label="ideal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Pseudo-label overall score")
    ax.set_ylabel("CNN predicted overall score")
    ax.set_title("CNN distills pseudo-label teacher at recording level")
    rho = metrics["overall"]["record_spearman"]
    mse = metrics["overall"]["record_mse"]
    ax.text(0.02, 0.98, f"Spearman rho = {rho:.3f}\nMSE = {mse:.4f}\nN = {len(rows)}",
            transform=ax.transAxes, va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.85"})
    ax.legend(frameon=False, ncol=2, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_distribution_by_config(rows, out_path):
    configs = sorted(set(r["config"] for r in rows))
    data = [[r["pred_overall"] for r in rows if r["config"] == cfg] for cfg in configs]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_edgecolor("#2F4B6C")
        body.set_alpha(0.55)
    parts["cmeans"].set_color("#1F2933")
    ax.boxplot(data, widths=0.18, patch_artist=True,
               boxprops={"facecolor": "white", "edgecolor": "#1F2933", "alpha": 0.75},
               medianprops={"color": "#D97706", "linewidth": 1.6},
               whiskerprops={"color": "#1F2933"},
               capprops={"color": "#1F2933"})
    ax.set_xticks(range(1, len(configs) + 1))
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel("Predicted overall score")
    ax.set_title("Temporary classifier score distribution by bowing configuration")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_aux_means(rows, out_path):
    fields = ds.TIER1_FIELDS
    means = np.array([np.mean([r[f"pred_{f}"] for r in rows]) for f in fields])
    stds = np.array([np.std([r[f"pred_{f}"] for r in rows]) for f in fields])
    labels = [f.replace("_", "\n") for f in fields]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.bar(range(len(fields)), means, yerr=stds, capsize=4, color="#59A14F", alpha=0.8)
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Predicted score")
    ax.set_title("Multi-task head outputs: mean +/- SD across recordings")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_config_heatmap(rows, out_path):
    configs = sorted(set(r["config"] for r in rows))
    fields = ds.TIER1_FIELDS
    mat = np.array([
        [np.mean([r[f"pred_{field}"] for r in rows if r["config"] == cfg]) for field in fields]
        for cfg in configs
    ])
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels([f.replace("_", "\n") for f in fields], fontsize=9)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    ax.set_title("Mean predicted Tier-1 scores by configuration")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.62 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Predicted score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-classifier presentation figures")
    parser.add_argument("--meta", default="SoundClassifier/Data_Collection/dataset_a_final/metadata.jsonl")
    parser.add_argument("--audio-dir", default="SoundClassifier/Data_Collection/dataset_a_final/audio")
    parser.add_argument("--checkpoint", default="SoundClassifier/checkpoints/quality_cnn.pt")
    parser.add_argument("--out-dir", default="SoundClassifier/figures/pseudo_classifier_report")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("Building pseudo-labeled examples...")
    examples = ds.build_training_examples(args.meta, args.audio_dir, pseudo_labels=True, verbose=True)
    print("Loading checkpoint...")
    ckpt, model, scalar_norm, physical_norm = load_checkpoint(args.checkpoint, device)
    print(f"Checkpoint label_source={ckpt.get('label_source')} pseudo_labels={ckpt.get('pseudo_labels')}")

    print("Running model...")
    preds, labels, _ = collect_predictions(examples, model, scalar_norm, physical_norm,
                                           device, args.batch_size)
    rows = aggregate_records(examples, preds, labels)
    metrics = compute_metrics(rows)

    save_record_csv(rows, out_dir / "record_level_predictions.csv")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_pred_vs_label(rows, metrics, out_dir / "01_pred_vs_pseudolabel.png")
    plot_distribution_by_config(rows, out_dir / "02_score_distribution_by_config.png")
    plot_aux_means(rows, out_dir / "03_multitask_head_summary.png")
    plot_config_heatmap(rows, out_dir / "04_config_tier1_heatmap.png")

    print(f"Wrote figures and tables to {out_dir}")
    print(json.dumps({
        "overall_record_spearman": metrics["overall"]["record_spearman"],
        "overall_record_mse": metrics["overall"]["record_mse"],
        "n_recordings": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
