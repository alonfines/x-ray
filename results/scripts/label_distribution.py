#!/usr/bin/env python3
"""
Generate label distribution plots for MIMIC-CXR-JPG split CSVs.

Reads each split CSV, counts positive labels (1.0), checks for sample
overlap between splits, and saves per-split and combined distribution plots.

Plot design: readable by a 50-year-old without glasses on a small laptop via Zoom.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = PROJECT_ROOT / "csv_files" / "mimic"
IMAGE_DIR = PROJECT_ROOT / "results" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ── labels (same order used throughout the project) ────────────────────
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]

SPLIT_FILES = {
    "Train": "train_split.csv",
    "Validation": "valid_split.csv",
    "Conformal": "conformal_split.csv",
    "Test": "test_split.csv",
}

# ── plotting defaults (Zoom-call readable) ─────────────────────────────
PALETTE = "mako"
FIGSIZE = (18, 11)
TITLE_SIZE = 28
AXIS_LABEL_SIZE = 22
TICK_SIZE = 18
ANNOTATION_SIZE = 17
SUPTITLE_SIZE = 30


def load_splits():
    """Load all split CSVs and return dict of DataFrames."""
    splits = {}
    for name, fname in SPLIT_FILES.items():
        path = CSV_DIR / fname
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        splits[name] = pd.read_csv(path)
    return splits


def check_overlap(splits):
    """Verify no dicom_id overlap between splits. Exit on overlap."""
    names = list(splits.keys())
    overlap_found = False
    for i, n1 in enumerate(names):
        ids1 = set(splits[n1]["dicom_id"])
        for n2 in names[i + 1:]:
            ids2 = set(splits[n2]["dicom_id"])
            common = ids1 & ids2
            if common:
                print(f"OVERLAP: {n1} & {n2} share {len(common)} dicom_ids")
                overlap_found = True
    if overlap_found:
        sys.exit(1)
    print("No overlap between splits.")


def count_positives(df):
    """Count positive (1.0) samples per label."""
    counts = {}
    for label in LABELS:
        if label in df.columns:
            counts[label] = int((df[label] == 1.0).sum())
        else:
            counts[label] = 0
    return counts


def plot_distribution(labels, percentages, counts, total, title, save_path):
    """Plot a single vertical bar chart with percentage annotations."""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    sns.barplot(x=labels, y=percentages, hue=labels, palette=PALETTE,
                legend=False, ax=ax)

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    ax.set_xlabel("Pathology / Observation", fontsize=AXIS_LABEL_SIZE,
                  fontweight="bold", labelpad=15)
    ax.set_ylabel("Positive Samples (%)", fontsize=AXIS_LABEL_SIZE,
                  fontweight="bold", labelpad=15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_SIZE,
                       fontweight="bold")
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylim(0, max(percentages) + 5)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for idx, (pct, cnt) in enumerate(zip(percentages, counts)):
        ax.text(idx, pct + 0.5, f"{pct:.1f}%", ha="center",
                fontsize=ANNOTATION_SIZE, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    splits = load_splits()
    if not splits:
        print("No split CSVs found.")
        sys.exit(1)

    check_overlap(splits)

    # ── per-split plots ────────────────────────────────────────────────
    all_counts = {}
    for split_name, df in splits.items():
        counts = count_positives(df)
        all_counts[split_name] = counts
        total = len(df)

        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, _ in sorted_items]
        vals = [v for _, v in sorted_items]
        pcts = [(v / total) * 100 for v in vals]

        title = (f"MIMIC CXR-JPG — {split_name} Set Distribution\n"
                 f"Total Samples: {total:,}")
        save_name = f"mimic_label_dist_{split_name.lower()}.png"
        plot_distribution(labels, pcts, vals, total, title,
                          IMAGE_DIR / save_name)

    # ── combined (all splits) plot ─────────────────────────────────────
    combined = pd.concat(list(splits.values()), ignore_index=True)
    total_all = len(combined)
    counts_all = count_positives(combined)
    sorted_items = sorted(counts_all.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    pcts = [(v / total_all) * 100 for v in vals]

    title = (f"MIMIC CXR-JPG — Combined Distribution\n"
             f"Total Samples Across Splits: {total_all:,}")
    plot_distribution(labels, pcts, vals, total_all, title,
                      IMAGE_DIR / "mimic_label_dist_combined.png")

    # ── summary table ──────────────────────────────────────────────────
    print(f"\n{'Label':<30} ", end="")
    for name in splits:
        print(f"{name:>12}", end="")
    print(f"{'Combined':>12}")
    print("-" * (30 + 12 * (len(splits) + 1)))
    for label in LABELS:
        print(f"{label:<30} ", end="")
        for name in splits:
            print(f"{all_counts[name][label]:>12,}", end="")
        print(f"{counts_all[label]:>12,}")
    print(f"\n{'Total samples':<30} ", end="")
    for name, df in splits.items():
        print(f"{len(df):>12,}", end="")
    print(f"{total_all:>12,}")


if __name__ == "__main__":
    main()
