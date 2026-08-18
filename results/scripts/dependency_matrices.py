#!/usr/bin/env python3
"""Directional dependency matrices combining medical hierarchy and PMI.

Generates:
1. Directional dependency matrix: parent→child = 1.0 (medical), others = PMI
2. Per-label confusion matrices from test results with dependency annotations
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils import safe_name

TRAIN_CSV = PROJECT_ROOT / "csv_files" / "mimic" / "train_split.csv"
PER_LABEL_DIR = PROJECT_ROOT / "conformal_calibration" / "mimic" / "per_label"
PER_LABEL_HIER_DIR = PROJECT_ROOT / "conformal_calibration" / "mimic" / "per_label_hier"
IMAGE_DIR = PROJECT_ROOT / "results" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]

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

# Medical parent-child relationships (definitional, directional: child → parent)
# These are hard constraints: if child is positive, parent MUST be positive
MEDICAL_HIERARCHY = {
    # (child, parent): description
    ("Cardiomegaly", "Enlarged Cardiomediastinum"): "enlarged heart → enlarged mediastinum",
    ("Lung Lesion", "Lung Opacity"): "lesion is a type of opacity",
    ("Edema", "Lung Opacity"): "edema manifests as opacity",
    ("Consolidation", "Lung Opacity"): "consolidation is dense opacity",
    ("Atelectasis", "Lung Opacity"): "atelectasis is volume-loss opacity",
    # NOTE: Pneumonia → Consolidation deliberately EXCLUDED (not definitional)
}

# Plot defaults
TITLE_SIZE = 24
LABEL_SIZE = 16
ANNOT_SIZE = 11
TICK_SIZE = 14


def compute_pmi(binary, labels):
    """Compute PMI matrix from binarized label DataFrame."""
    n_labels = len(labels)
    prevalence = binary.mean()
    pmi = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                continue
            p_i = prevalence[labels[i]]
            p_j = prevalence[labels[j]]
            p_ij = ((binary[labels[i]] == 1) & (binary[labels[j]] == 1)).mean()
            if p_ij > 0 and p_i > 0 and p_j > 0:
                pmi[i, j] = np.log2(p_ij / (p_i * p_j))
            elif p_ij == 0:
                pmi[i, j] = -10
    return pmi


def compute_conditional_prob(binary, labels):
    """Compute P(col=1 | row=1) matrix."""
    n_labels = len(labels)
    cond = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        mask = binary[labels[i]] == 1
        n_i = mask.sum()
        if n_i == 0:
            continue
        for j in range(n_labels):
            cond[i, j] = binary.loc[mask, labels[j]].mean()
    return cond


def plot_directional_dependency_matrix(pmi, cond_prob, labels):
    """Create the directional dependency matrix.

    - Parent-child medical cells: show 1.0 (medical truth, overrides data)
    - All other cells: show PMI from data
    - Annotate medical cells with a special marker
    """
    n = len(labels)
    short_labels = [SHORT[l] for l in labels]

    # Build the display matrix: start with PMI, overlay medical hierarchy
    display = pmi.copy()
    is_medical = np.zeros((n, n), dtype=bool)

    for (child, parent), desc in MEDICAL_HIERARCHY.items():
        ci = labels.index(child)
        pi = labels.index(parent)
        # Direction: row=child, col=parent → "given child=1, parent should be 1"
        display[ci, pi] = 1.0  # Medical truth (definitional)
        is_medical[ci, pi] = True

    # Also mark No Finding exclusions as medical (definitional: No Finding = no pathology)
    nf_idx = labels.index("No Finding")
    for i in range(n):
        if i != nf_idx and labels[i] != "Support Devices":
            display[nf_idx, i] = -5.0  # Strong medical exclusion
            is_medical[nf_idx, i] = True
            display[i, nf_idx] = -5.0
            is_medical[i, nf_idx] = True

    # Set diagonal to NaN
    np.fill_diagonal(display, np.nan)

    # Clip for display
    display_clipped = np.clip(display, -5, 3)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(20, 16), facecolor="white")

    # Custom colormap: blue (exclusion) → white (independent) → red (co-occur)
    # Medical hierarchy cells (1.0) will appear as strong red
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=3)

    sns.heatmap(display_clipped, annot=False, cmap="RdBu_r", norm=norm,
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax, linewidths=0.8, linecolor="white",
                cbar_kws={"label": "Dependency strength", "shrink": 0.8},
                mask=np.eye(n, dtype=bool))

    # Custom annotations: medical cells get bold + star, PMI cells get plain value
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = display_clipped[i, j]
            if is_medical[i, j]:
                # Medical hierarchy: show value with star, bold, black background
                if val > 0:
                    txt = f"M\n{val:+.1f}"
                    color = "white"
                    bbox = dict(boxstyle="round,pad=0.15", facecolor="#2d5016",
                                edgecolor="white", alpha=0.9)
                else:
                    txt = f"M\n{val:.1f}"
                    color = "white"
                    bbox = dict(boxstyle="round,pad=0.15", facecolor="#1a3a5c",
                                edgecolor="white", alpha=0.9)
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color, bbox=bbox)
            else:
                # PMI value
                if abs(val) < 0.3:
                    color = "gray"
                elif val > 0:
                    color = "darkred" if val > 1.5 else "black"
                else:
                    color = "darkblue" if val < -1.5 else "black"
                ax.text(j + 0.5, i + 0.5, f"{val:+.1f}", ha="center", va="center",
                        fontsize=ANNOT_SIZE, fontweight="bold" if abs(val) > 1.0 else "normal",
                        color=color)

    ax.set_title("Directional Dependency Matrix\n"
                 "M = Medical hierarchy (definitional) | Values = PMI (data-driven)\n"
                 "Row = given label | Column = target label",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    ax.set_xlabel("Target Label", fontsize=LABEL_SIZE, fontweight="bold", labelpad=10)
    ax.set_ylabel("Given Label = Positive", fontsize=LABEL_SIZE, fontweight="bold", labelpad=10)
    ax.tick_params(labelsize=TICK_SIZE)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="#2d5016", edgecolor="white",
                       label="Medical hierarchy (child→parent = 1.0)"),
        mpatches.Patch(facecolor="#1a3a5c", edgecolor="white",
                       label="Medical exclusion (No Finding ↔ pathology)"),
        mpatches.Patch(facecolor="#c0392b", edgecolor="white",
                       label="PMI > 0: co-occur more than expected"),
        mpatches.Patch(facecolor="#2980b9", edgecolor="white",
                       label="PMI < 0: co-occur less than expected"),
    ]
    ax.legend(handles=legend_handles, fontsize=13, loc="upper left",
              bbox_to_anchor=(0.0, -0.08), ncol=2, framealpha=0.9)

    fig.tight_layout()
    save_path = IMAGE_DIR / "directional_dependency_matrix.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_conditional_with_hierarchy(cond_prob, labels):
    """Conditional probability matrix with medical hierarchy cells highlighted."""
    n = len(labels)
    short_labels = [SHORT[l] for l in labels]

    # Build medical overlay mask
    is_medical = np.zeros((n, n), dtype=bool)
    for (child, parent), _ in MEDICAL_HIERARCHY.items():
        ci = labels.index(child)
        pi = labels.index(parent)
        is_medical[ci, pi] = True

    # Display matrix
    display = cond_prob.copy()
    np.fill_diagonal(display, np.nan)

    fig, ax = plt.subplots(figsize=(20, 16), facecolor="white")

    sns.heatmap(display, annot=False, fmt=".2f", cmap="YlOrRd",
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax, vmin=0, vmax=1, linewidths=0.8, linecolor="white",
                cbar_kws={"label": "P(column=1 | row=1)", "shrink": 0.8},
                mask=np.eye(n, dtype=bool))

    # Annotate with values; highlight medical cells
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = display[i, j]
            if np.isnan(val):
                continue
            if is_medical[i, j]:
                # Medical: show actual P alongside the medical truth (1.0)
                txt = f"{val:.2f}\n(M:1.0)"
                bbox = dict(boxstyle="round,pad=0.15", facecolor="#2d5016",
                            edgecolor="white", alpha=0.85)
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white", bbox=bbox)
            else:
                color = "black" if val < 0.6 else "white"
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center",
                        fontsize=ANNOT_SIZE, color=color)

    ax.set_title("Conditional Probability with Medical Hierarchy\n"
                 "P(column=1 | row=1) — Green boxes show labeling gap vs medical truth (1.0)",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    ax.set_xlabel("Target Label", fontsize=LABEL_SIZE, fontweight="bold", labelpad=10)
    ax.set_ylabel("Given Label = Positive", fontsize=LABEL_SIZE, fontweight="bold", labelpad=10)
    ax.tick_params(labelsize=TICK_SIZE)

    fig.tight_layout()
    save_path = IMAGE_DIR / "conditional_prob_with_hierarchy.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_classifier_confusion_with_dependencies(labels):
    """Per-label 3x3 confusion matrices from test results.

    For each label, show:
    - The 3x3 confusion matrix (true Pos/Unc/Neg vs predicted Pos/Unc/Neg)
    - Annotate with related labels (medical parents/children and top PMI partners)
    """
    # Collect available test results
    results = {}
    for label in labels:
        sn = safe_name(label)
        # Try per_label first, then per_label_hier
        for d, tag in [(PER_LABEL_DIR, ""), (PER_LABEL_HIER_DIR, " (hier)")]:
            csv_path = d / sn / "test_results.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                results[label + tag] = df

    if not results:
        print("No test results found for confusion matrices.")
        return

    # Find medical relationships for annotations
    med_parents = {}
    med_children = {}
    for (child, parent), desc in MEDICAL_HIERARCHY.items():
        med_parents.setdefault(child, []).append(parent)
        med_children.setdefault(parent, []).append(child)

    # Use per_label results (non-hier) for confusion matrices
    plot_labels = [l for l in labels if safe_name(l) in
                   [d.name for d in PER_LABEL_DIR.iterdir() if d.is_dir()]]

    if not plot_labels:
        print("No per-label directories found.")
        return

    n_labels = len(plot_labels)
    cols = 4
    rows = (n_labels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(28, 7 * rows), facecolor="white")
    axes = axes.flatten()

    for idx, label in enumerate(plot_labels):
        ax = axes[idx]
        sn = safe_name(label)
        csv_path = PER_LABEL_DIR / sn / "test_results.csv"
        if not csv_path.exists():
            ax.set_visible(False)
            continue

        df = pd.read_csv(csv_path)

        # Build 3x3: true (Pos=1, Unc=-1, Neg=0) vs predicted (Pos, Unc, Neg)
        # Predicted from conformal_set or prediction column
        pred_col = None
        for col_name in ["bcops_prediction", "conformal_set", "prediction"]:
            if col_name in df.columns:
                pred_col = col_name
                break
        if pred_col is None:
            ax.set_visible(False)
            continue

        true_col = "true_label"

        # Map true labels to categories
        def true_cat(v):
            if v == 1.0:
                return "Positive"
            elif v == 0.0:
                return "Negative"
            else:
                return "Uncertain"

        # Map predictions to categories
        def pred_cat(v):
            v_str = str(v).lower().strip()
            if v_str in ("positive", "present", "{present}", "1", "1.0"):
                return "Positive"
            elif v_str in ("negative", "absent", "{absent}", "0", "0.0"):
                return "Negative"
            else:
                return "Uncertain"

        df["true_cat"] = df[true_col].apply(true_cat)
        df["pred_cat"] = df[pred_col].apply(pred_cat)

        cats = ["Positive", "Uncertain", "Negative"]
        cm = np.zeros((3, 3), dtype=int)
        for ti, tc in enumerate(cats):
            for pi, pc in enumerate(cats):
                cm[ti, pi] = ((df["true_cat"] == tc) & (df["pred_cat"] == pc)).sum()

        # Normalize by row (true label)
        cm_pct = np.zeros((3, 3))
        for ti in range(3):
            row_sum = cm[ti].sum()
            if row_sum > 0:
                cm_pct[ti] = cm[ti] / row_sum * 100

        # Plot
        sns.heatmap(cm_pct, annot=False, cmap="Blues", ax=ax,
                    xticklabels=["Pos", "Unc", "Neg"],
                    yticklabels=["Pos", "Unc", "Neg"],
                    vmin=0, vmax=100, linewidths=1, linecolor="white",
                    cbar=False)

        # Annotate with count and percentage
        for ti in range(3):
            for pi in range(3):
                count = cm[ti, pi]
                pct = cm_pct[ti, pi]
                color = "white" if pct > 50 else "black"
                ax.text(pi + 0.5, ti + 0.5,
                        f"{pct:.0f}%\n({count})",
                        ha="center", va="center", fontsize=11,
                        fontweight="bold", color=color)

        # Title with dependency info
        dep_info = []
        if label in med_parents:
            dep_info.append(f"Parent: {', '.join(SHORT[p] for p in med_parents[label])}")
        if label in med_children:
            dep_info.append(f"Children: {', '.join(SHORT[c] for c in med_children[label])}")

        title = f"{SHORT[label]}"
        if dep_info:
            title += f"\n({'; '.join(dep_info)})"

        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)

    # Hide unused axes
    for idx in range(n_labels, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Per-Label Confusion Matrices (True vs Conformal Prediction)\n"
                 "Medical dependencies shown in subtitle of each label",
                 fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_path = IMAGE_DIR / "confusion_matrices_with_dependencies.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_dependency_violation_analysis(labels):
    """Analyze how often classifier predictions violate medical dependencies.

    For each parent-child pair, count:
    - Child=Positive, Parent=Positive (correct)
    - Child=Positive, Parent=Negative (VIOLATION)
    - Child=Positive, Parent=Uncertain (partial violation)
    """
    violations = []

    for (child, parent), desc in MEDICAL_HIERARCHY.items():
        child_sn = safe_name(child)
        parent_sn = safe_name(parent)

        child_csv = PER_LABEL_DIR / child_sn / "test_results.csv"
        parent_csv = PER_LABEL_DIR / parent_sn / "test_results.csv"

        if not child_csv.exists() or not parent_csv.exists():
            continue

        child_df = pd.read_csv(child_csv)
        parent_df = pd.read_csv(parent_csv)

        # Get predictions
        for col_name in ["bcops_prediction", "conformal_set", "prediction"]:
            if col_name in child_df.columns:
                pred_col = col_name
                break
        else:
            continue

        def is_positive(v):
            return str(v).lower().strip() in ("positive", "present", "{present}", "1", "1.0")

        def is_negative(v):
            return str(v).lower().strip() in ("negative", "absent", "{absent}", "0", "0.0")

        child_pos = child_df[pred_col].apply(is_positive)
        parent_pos = parent_df[pred_col].apply(is_positive)
        parent_neg = parent_df[pred_col].apply(is_negative)

        n_child_pos = child_pos.sum()
        if n_child_pos == 0:
            continue

        n_correct = (child_pos & parent_pos).sum()
        n_violation = (child_pos & parent_neg).sum()
        n_partial = n_child_pos - n_correct - n_violation

        violations.append({
            "child": SHORT[child],
            "parent": SHORT[parent],
            "desc": desc,
            "child_pos": int(n_child_pos),
            "parent_also_pos": int(n_correct),
            "parent_neg_VIOLATION": int(n_violation),
            "parent_uncertain": int(n_partial),
            "violation_rate": n_violation / n_child_pos if n_child_pos > 0 else 0,
        })

    if not violations:
        print("No violation data available.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(18, 8), facecolor="white")

    pair_labels = [f"{v['child']}→{v['parent']}" for v in violations]
    x = np.arange(len(pair_labels))
    width = 0.6

    correct = [v["parent_also_pos"] / v["child_pos"] * 100 for v in violations]
    uncertain = [v["parent_uncertain"] / v["child_pos"] * 100 for v in violations]
    violation = [v["violation_rate"] * 100 for v in violations]

    bars1 = ax.bar(x, correct, width, label="Parent also Positive (correct)", color="#27ae60")
    bars2 = ax.bar(x, uncertain, width, bottom=correct, label="Parent Uncertain (partial)", color="#f39c12")
    bars3 = ax.bar(x, violation, width,
                   bottom=[c + u for c, u in zip(correct, uncertain)],
                   label="Parent Negative (VIOLATION)", color="#c0392b")

    # Annotate violation rate
    for idx, v in enumerate(violations):
        rate = v["violation_rate"] * 100
        total = v["child_pos"]
        y_pos = correct[idx] + uncertain[idx] + violation[idx] + 1
        ax.text(x[idx], y_pos, f"{rate:.0f}%\n(n={total})",
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                color="#c0392b" if rate > 10 else "black")

    ax.set_title("Medical Hierarchy Violation Analysis\n"
                 "When child is predicted Positive, is parent also Positive?",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    ax.set_xlabel("Child → Parent Relationship", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("% of child-positive predictions", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=TICK_SIZE, fontweight="bold")
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=14, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    save_path = IMAGE_DIR / "hierarchy_violation_analysis.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Print summary
    print("\nHierarchy Violation Summary:")
    print(f"  {'Pair':<25} {'Child+':<8} {'Parent+':<9} {'Violation':<10} {'Rate':<8}")
    print("  " + "-" * 60)
    for v in violations:
        print(f"  {v['child']+'→'+v['parent']:<25} {v['child_pos']:<8} "
              f"{v['parent_also_pos']:<9} {v['parent_neg_VIOLATION']:<10} "
              f"{v['violation_rate']:.1%}")


def main():
    print("Reading training data...")
    df = pd.read_csv(TRAIN_CSV)
    binary = pd.DataFrame()
    for label in LABELS:
        if label in df.columns:
            binary[label] = (df[label] == 1.0).astype(float)

    print("Computing PMI matrix...")
    pmi = compute_pmi(binary, LABELS)

    print("Computing conditional probability matrix...")
    cond_prob = compute_conditional_prob(binary, LABELS)

    print("\n=== Plot 1: Directional Dependency Matrix ===")
    plot_directional_dependency_matrix(pmi, cond_prob, LABELS)

    print("\n=== Plot 2: Conditional Probability with Medical Hierarchy ===")
    plot_conditional_with_hierarchy(cond_prob, LABELS)

    print("\n=== Plot 3: Per-Label Confusion Matrices ===")
    plot_classifier_confusion_with_dependencies(LABELS)

    print("\n=== Plot 4: Hierarchy Violation Analysis ===")
    plot_dependency_violation_analysis(LABELS)

    print("\nDone!")


if __name__ == "__main__":
    main()
