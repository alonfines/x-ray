#!/usr/bin/env python3
"""
Plot per-label AUC scores from a test inference CSV.

Reads the CSV produced by test.py (columns: {label}_prob, {label}_true_label),
computes per-label AUC, and saves a bar chart to results/images/.

Plot design: readable by a 50-year-old without glasses on a small laptop via Zoom.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

# ── CHANGE THIS PATH to point at the CSV you want to plot ─────────────
CSV_PATH = (
    Path(__file__).resolve().parents[2]
    / "results" / "outputs" / "mimic"
    / "densenet-alllabels-bce-epoch=60-val_auroc_mean=0.784.csv"
)
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "results" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]

# ── plotting defaults (Zoom-call readable) ─────────────────────────────
PALETTE = "mako"
FIGSIZE = (18, 11)
TITLE_SIZE = 28
AXIS_LABEL_SIZE = 22
TICK_SIZE = 18
ANNOTATION_SIZE = 17


def compute_aucs(df):
    """Compute per-label AUC, skipping uncertain (-1) labels."""
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


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    aucs = compute_aucs(df)
    if not aucs:
        print("No valid AUC scores computed.")
        return

    # Sort by AUC descending
    sorted_items = sorted(aucs.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in sorted_items]
    scores = [v for _, v in sorted_items]
    mean_auc = sum(scores) / len(scores)

    # ── plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    sns.barplot(x=labels, y=scores, hue=labels, palette=PALETTE,
                legend=False, ax=ax)

    title = (f"MIMIC CXR-JPG — Per-Label AUC\n"
             f"Mean AUC: {mean_auc:.4f} | Unified Model")
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    ax.set_xlabel("Pathology / Observation", fontsize=AXIS_LABEL_SIZE,
                  fontweight="bold", labelpad=15)
    ax.set_ylabel("AUC Score", fontsize=AXIS_LABEL_SIZE,
                  fontweight="bold", labelpad=15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_SIZE,
                       fontweight="bold")
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for idx, score in enumerate(scores):
        ax.text(idx, score + 0.01, f"{score:.3f}", ha="center",
                fontsize=ANNOTATION_SIZE, fontweight="bold")

    fig.tight_layout()
    save_path = IMAGE_DIR / f"auc_{CSV_PATH.stem}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")
    print(f"Mean AUC: {mean_auc:.4f}")


if __name__ == "__main__":
    main()
