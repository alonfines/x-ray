#!/usr/bin/env python3
"""
Plot per-label AUC comparison: unified model vs per-label binary models.

Reads the unified model CSV (from test.py) and per-label test_results.csv
files (from test_per_label.py), computes AUC for each, and produces a
grouped bar chart showing both side by side with the AUC delta annotated.

Plot design: readable by a 50-year-old without glasses on a small laptop via Zoom.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils import safe_name

UNIFIED_CSV = (
    PROJECT_ROOT / "results" / "outputs" / "mimic"
    / "densenet-alllabels-bce-hier-epoch=50-val_auroc_mean=0.785.csv"
)

PER_LABEL_DIR = PROJECT_ROOT / "conformal_calibration" / "mimic" / "per_label_hier"

IMAGE_DIR = PROJECT_ROOT / "results" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]

# ── plotting defaults (Zoom-call readable) ───────────────────────────
FIGSIZE = (22, 12)
TITLE_SIZE = 28
AXIS_LABEL_SIZE = 22
TICK_SIZE = 18
ANNOTATION_SIZE = 15


def compute_unified_aucs(df):
    """Compute per-label AUC from the unified model CSV."""
    results = {}
    for label in LABELS:
        prob_col = f"{label}_prob"
        true_col = f"{label}_true_label"
        if prob_col not in df.columns or true_col not in df.columns:
            continue
        mask = df[true_col] >= 0
        y_true = df.loc[mask, true_col].values.astype(float)
        y_prob = df.loc[mask, prob_col].values.astype(float)
        if len(np.unique(y_true)) < 2:
            continue
        results[label] = roc_auc_score(y_true, y_prob)
    return results


def compute_per_label_aucs():
    """Compute AUC from each per-label model's test_results.csv."""
    results = {}
    for label in LABELS:
        csv_path = PER_LABEL_DIR / safe_name(label) / "test_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        mask = df["true_label"] >= 0
        y_true = df.loc[mask, "true_label"].values.astype(float)
        y_prob = df.loc[mask, "prob"].values.astype(float)
        if len(np.unique(y_true)) < 2:
            continue
        results[label] = roc_auc_score(y_true, y_prob)
    return results


def main():
    if not UNIFIED_CSV.exists():
        print(f"Unified CSV not found: {UNIFIED_CSV}")
        return

    unified_aucs = compute_unified_aucs(pd.read_csv(UNIFIED_CSV))
    per_label_aucs = compute_per_label_aucs()

    if not per_label_aucs:
        print("No valid per-label AUC scores computed.")
        return

    # All labels that have a per-label model, sorted by per-label AUC descending
    all_labels = [l for l in LABELS if l in per_label_aucs]
    all_labels.sort(key=lambda l: per_label_aucs[l], reverse=True)

    # Mean AUCs: unified over its 13 labels, per-label over all 14
    common_labels = [l for l in all_labels if l in unified_aucs]
    mean_unified = np.mean([unified_aucs[l] for l in common_labels])
    mean_per_label = np.mean([per_label_aucs[l] for l in all_labels])

    # ── plot ──────────────────────────────────────────────────────────
    UNIFIED_COLOR = "#2b6cb0"
    PER_LABEL_COLOR = "#e8873d"

    x = np.arange(len(all_labels))
    bar_width = 0.55

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")

    unified_heights = [unified_aucs.get(l, 0) for l in all_labels]
    per_label_heights = [per_label_aucs[l] for l in all_labels]

    # Draw taller bar first (behind), shorter bar second (in front)
    for idx, label in enumerate(all_labels):
        pl = per_label_heights[idx]
        u = unified_heights[idx]

        if label not in unified_aucs:
            # Per-label only (No Finding)
            ax.bar(x[idx], pl, bar_width, color=PER_LABEL_COLOR,
                   edgecolor="white", linewidth=0.5)
        elif pl >= u:
            # Per-label taller → draw it behind, unified in front
            ax.bar(x[idx], pl, bar_width, color=PER_LABEL_COLOR,
                   edgecolor="white", linewidth=0.5)
            ax.bar(x[idx], u, bar_width, color=UNIFIED_COLOR,
                   edgecolor="white", linewidth=0.5)
        else:
            # Unified taller → draw it behind, per-label in front
            ax.bar(x[idx], u, bar_width, color=UNIFIED_COLOR,
                   edgecolor="white", linewidth=0.5)
            ax.bar(x[idx], pl, bar_width, color=PER_LABEL_COLOR,
                   edgecolor="white", linewidth=0.5)

    # Legend patches with correct colors
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=UNIFIED_COLOR, label="Unified Model"),
        Patch(facecolor=PER_LABEL_COLOR, label="Per-Label Model"),
    ]

    # Annotate only the taller (top) bar's score
    for idx, label in enumerate(all_labels):
        pl = per_label_heights[idx]
        u = unified_heights[idx]

        if label not in unified_aucs:
            top, color = pl, PER_LABEL_COLOR
        elif pl >= u:
            top, color = pl, PER_LABEL_COLOR
        else:
            top, color = u, UNIFIED_COLOR

        ax.text(x[idx], top + 0.008, f"{top:.3f}",
                ha="center", fontsize=ANNOTATION_SIZE, fontweight="bold",
                color=color)

    title = (
        f"MIMIC CXR-JPG — Per-Label AUC Comparison\n"
        f"Unified Mean (13): {mean_unified:.4f}  |  "
        f"Per-Label Mean (14): {mean_per_label:.4f}"
    )
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    ax.set_xlabel("Pathology / Observation", fontsize=AXIS_LABEL_SIZE,
                  fontweight="bold", labelpad=15)
    ax.set_ylabel("AUC Score", fontsize=AXIS_LABEL_SIZE,
                  fontweight="bold", labelpad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right",
                       fontsize=TICK_SIZE, fontweight="bold")
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(handles=legend_handles, fontsize=TICK_SIZE, loc="lower right",
              framealpha=0.9)

    fig.tight_layout()
    save_path = IMAGE_DIR / "auc_unified_vs_per_label.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")
    print(f"Unified Mean AUC (13):   {mean_unified:.4f}")
    print(f"Per-Label Mean AUC (14): {mean_per_label:.4f}")


if __name__ == "__main__":
    main()
