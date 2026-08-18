#!/usr/bin/env python3
"""Co-occurrence dependency analysis for CheXpert labels in MIMIC-CXR.

Computes:
1. Marginal prevalence per label
2. Conditional probability matrix P(j=1 | i=1)
3. PMI matrix (pointwise mutual information)
4. Generates heatmaps and dependency graph visualization
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_CSV = PROJECT_ROOT / "csv_files" / "mimic" / "train_split.csv"
IMAGE_DIR = PROJECT_ROOT / "results" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]

# Short names for plots
SHORT = {
    "No Finding": "NoFind",
    "Enlarged Cardiomediastinum": "EnlCard",
    "Cardiomegaly": "Cardio",
    "Lung Opacity": "LungOp",
    "Lung Lesion": "LungLes",
    "Edema": "Edema",
    "Consolidation": "Consol",
    "Pneumonia": "Pneum",
    "Atelectasis": "Atelec",
    "Pneumothorax": "PneuTx",
    "Pleural Effusion": "PlEff",
    "Pleural Other": "PlOth",
    "Fracture": "Frac",
    "Support Devices": "SupDev",
}

# Plot defaults (Zoom-call readable)
FIGSIZE = (18, 14)
TITLE_SIZE = 24
LABEL_SIZE = 16
ANNOT_SIZE = 12


def main():
    print(f"Reading training data from {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    print(f"  {len(df)} samples")

    # Binarize: treat NaN and -1 as 0, only 1.0 counts as positive
    binary = pd.DataFrame()
    for label in LABELS:
        if label in df.columns:
            binary[label] = (df[label] == 1.0).astype(float)
        else:
            print(f"  WARNING: {label} not in CSV columns")

    n = len(binary)
    n_labels = len(LABELS)

    # ── 1. Marginal prevalence ──────────────────────────────────────
    print("\n=== MARGINAL PREVALENCE ===")
    prevalence = binary.mean()
    for label in LABELS:
        count = int(binary[label].sum())
        print(f"  {label:<35} {prevalence[label]:.4f}  (n={count})")

    # ── 2. Conditional probability P(j=1 | i=1) ────────────────────
    print("\n=== CONDITIONAL PROBABILITY P(col=1 | row=1) ===")
    cond_prob = np.zeros((n_labels, n_labels))
    for i, li in enumerate(LABELS):
        mask_i = binary[li] == 1.0
        n_i = mask_i.sum()
        if n_i == 0:
            continue
        for j, lj in enumerate(LABELS):
            cond_prob[i, j] = binary.loc[mask_i, lj].mean()

    # Print key relationships
    print("\n  Key hierarchy relationships (P(parent=1 | child=1)):")
    pairs = [
        ("Consolidation", "Lung Opacity", "child→parent"),
        ("Edema", "Lung Opacity", "child→parent"),
        ("Atelectasis", "Lung Opacity", "child→parent"),
        ("Lung Lesion", "Lung Opacity", "child→parent"),
        ("Cardiomegaly", "Enlarged Cardiomediastinum", "child→parent"),
        ("Pneumonia", "Consolidation", "child→parent (questionable)"),
        ("Pneumonia", "Lung Opacity", "child→grandparent"),
    ]
    for child, parent, note in pairs:
        ci = LABELS.index(child)
        pi = LABELS.index(parent)
        print(f"    P({SHORT[parent]:>7}=1 | {SHORT[child]:<7}=1) = {cond_prob[ci, pi]:.3f}  ({note})")

    print("\n  Clinically expected co-occurrences:")
    extra_pairs = [
        ("Edema", "Pleural Effusion", "heart failure"),
        ("Pleural Effusion", "Edema", "heart failure reverse"),
        ("Atelectasis", "Pleural Effusion", "compression"),
        ("Pleural Effusion", "Atelectasis", "compression reverse"),
        ("Edema", "Cardiomegaly", "cardiac"),
        ("Support Devices", "Edema", "ICU patients"),
        ("Edema", "Support Devices", "ICU patients reverse"),
    ]
    for a, b, note in extra_pairs:
        ai = LABELS.index(a)
        bi = LABELS.index(b)
        print(f"    P({SHORT[b]:>7}=1 | {SHORT[a]:<7}=1) = {cond_prob[ai, bi]:.3f}  ({note})")

    # ── 3. PMI matrix ───────────────────────────────────────────────
    print("\n=== PMI MATRIX (top dependencies & exclusions) ===")
    pmi = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                pmi[i, j] = 0
                continue
            p_i = prevalence[LABELS[i]]
            p_j = prevalence[LABELS[j]]
            p_ij = ((binary[LABELS[i]] == 1) & (binary[LABELS[j]] == 1)).mean()
            if p_ij > 0 and p_i > 0 and p_j > 0:
                pmi[i, j] = np.log2(p_ij / (p_i * p_j))
            elif p_ij == 0:
                pmi[i, j] = -10  # strong exclusion (cap)
            else:
                pmi[i, j] = 0

    # Print strongest positive and negative PMIs
    pmi_pairs = []
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            pmi_pairs.append((LABELS[i], LABELS[j], pmi[i, j]))

    pmi_pairs.sort(key=lambda x: x[2], reverse=True)
    print("\n  Top 10 POSITIVE dependencies (co-occurrence beyond chance):")
    for a, b, val in pmi_pairs[:10]:
        print(f"    PMI({SHORT[a]:>7}, {SHORT[b]:<7}) = {val:+.3f}")

    print("\n  Top 10 NEGATIVE dependencies (exclusion/anti-correlation):")
    for a, b, val in pmi_pairs[-10:]:
        print(f"    PMI({SHORT[a]:>7}, {SHORT[b]:<7}) = {val:+.3f}")

    # ── 4. Heatmaps ────────────────────────────────────────────────
    short_labels = [SHORT[l] for l in LABELS]

    # Conditional probability heatmap
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    mask_diag = np.eye(n_labels, dtype=bool)
    cond_display = cond_prob.copy()
    cond_display[mask_diag] = np.nan

    sns.heatmap(cond_display, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax, vmin=0, vmax=1, annot_kws={"size": ANNOT_SIZE},
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "P(column=1 | row=1)", "shrink": 0.8})
    ax.set_title("Conditional Probability Matrix\nP(column label = 1 | row label = 1)",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    ax.set_xlabel("Target Label", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Given Label = 1", fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(labelsize=LABEL_SIZE)
    fig.tight_layout()
    save_path = IMAGE_DIR / "conditional_probability_matrix.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {save_path}")

    # PMI heatmap
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    pmi_display = pmi.copy()
    pmi_display[mask_diag] = np.nan
    # Cap for display
    pmi_display = np.clip(pmi_display, -5, 5)

    sns.heatmap(pmi_display, annot=True, fmt=".2f", cmap="RdBu_r",
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax, center=0, vmin=-5, vmax=5,
                annot_kws={"size": ANNOT_SIZE},
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "PMI (bits)", "shrink": 0.8})
    ax.set_title("Pointwise Mutual Information (PMI) Matrix\n"
                 "Red = co-occur more than expected | Blue = co-occur less than expected",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    ax.set_xlabel("Label", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Label", fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(labelsize=LABEL_SIZE)
    fig.tight_layout()
    save_path = IMAGE_DIR / "pmi_matrix.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")

    # ── 5. Dependency graph (significant edges only) ────────────────
    fig, ax = plt.subplots(figsize=(20, 16), facecolor="white")

    # Position labels in a circle
    n = n_labels
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 4
    positions = {LABELS[i]: (radius * np.cos(angles[i] - np.pi/2),
                              radius * np.sin(angles[i] - np.pi/2))
                 for i in range(n)}

    # Draw edges for |PMI| > 0.5
    pmi_threshold = 0.5
    drawn_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            val = pmi[i, j]
            if abs(val) < pmi_threshold:
                continue
            x1, y1 = positions[LABELS[i]]
            x2, y2 = positions[LABELS[j]]
            if val > 0:
                color = plt.cm.Reds(min(val / 4, 1.0))
                style = "-"
            else:
                color = plt.cm.Blues(min(abs(val) / 4, 1.0))
                style = "--"
            width = min(abs(val) * 1.5, 5)
            ax.plot([x1, x2], [y1, y2], style, color=color,
                    linewidth=width, alpha=0.7, zorder=1)
            # PMI annotation at midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, f"{val:+.1f}", fontsize=9, ha="center",
                    va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="gray", alpha=0.85), zorder=3)
            drawn_edges.append((LABELS[i], LABELS[j], val))

    # Draw nodes
    for label in LABELS:
        x, y = positions[label]
        prev = prevalence[label]
        size = 800 + prev * 4000
        ax.scatter(x, y, s=size, c="#2b6cb0", edgecolors="white",
                   linewidths=2, zorder=5, alpha=0.9)
        ax.text(x, y - 0.55, SHORT[label], fontsize=13, ha="center",
                va="top", fontweight="bold", zorder=6)
        ax.text(x, y + 0.1, f"{prev:.1%}", fontsize=10, ha="center",
                va="bottom", color="white", fontweight="bold", zorder=6)

    ax.set_title(f"Label Dependency Graph (|PMI| > {pmi_threshold})\n"
                 f"Red solid = positive dependency | Blue dashed = exclusion\n"
                 f"Node size = prevalence",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    save_path = IMAGE_DIR / "dependency_graph.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")

    print(f"\nDependency graph edges (|PMI| > {pmi_threshold}): {len(drawn_edges)}")
    for a, b, val in sorted(drawn_edges, key=lambda x: abs(x[2]), reverse=True):
        sign = "CO-OCCUR" if val > 0 else "EXCLUDE"
        print(f"  {SHORT[a]:>7} -- {SHORT[b]:<7}  PMI={val:+.2f}  ({sign})")


if __name__ == "__main__":
    main()
