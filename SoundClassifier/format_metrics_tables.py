#!/usr/bin/env python3
"""Format pseudo-classifier metric averages as markdown tables."""

import argparse
import json
from pathlib import Path


QUALITY_ORDER = [
    "overall",
    "tone_quality",
    "attack_quality",
    "release_quality",
    "bow_control",
    "dynamic_accuracy",
]


def fmt(x):
    return f"{float(x):.3f}"


def quality_table(metrics):
    lines = [
        "## By Quality Category",
        "",
        "| Quality category | Avg pseudo-label score | Avg predicted score |",
        "|---|---:|---:|",
    ]
    for name in QUALITY_ORDER:
        row = metrics[name]
        lines.append(f"| {name} | {fmt(row['label_mean'])} | {fmt(row['pred_mean'])} |")
    return "\n".join(lines)


def config_table(metrics):
    lines = [
        "## By Bowing Angle / Configuration",
        "",
        "| Configuration | Avg pseudo-label overall | Avg predicted overall |",
        "|---|---:|---:|",
    ]
    for name in sorted(metrics["by_config"]):
        row = metrics["by_config"][name]
        lines.append(
            f"| {name} | {fmt(row['overall_label_mean'])} | {fmt(row['overall_pred_mean'])} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate markdown average-score tables from metrics.json")
    parser.add_argument(
        "--metrics",
        default="SoundClassifier/figures/pseudo_classifier_report/metrics.json",
        help="Path to metrics.json",
    )
    parser.add_argument(
        "--out",
        default="SoundClassifier/figures/pseudo_classifier_report/score_average_tables.md",
        help="Output markdown path",
    )
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    text = config_table(metrics) + "\n\n" + quality_table(metrics) + "\n"
    Path(args.out).write_text(text)
    print(text)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
