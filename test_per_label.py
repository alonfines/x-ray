#!/usr/bin/env python3
"""
Test per-label binary models with BCOPS conformal predictions.

For each per-label checkpoint, runs test inference, applies calibrated
BCOPS thresholds, saves CSV results and confusion matrix PNGs to:
    conformal_calibration/mimic/per_label/{label}/

Requires calibration to be run first (calibrate_per_label.py).

Usage:
    # All labels
    python test_per_label.py

    # Single label
    python test_per_label.py --label 'Atelectasis'
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data import get_data_module, parse_labels_config
from train import CXRClassifier
from train_per_label import generate_config, ALL_LABELS

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_CONFIG = PROJECT_ROOT / "config.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "mimic" / "per_label"
OUTPUT_BASE = PROJECT_ROOT / "conformal_calibration" / "mimic" / "per_label"


def find_checkpoint(safe_name: str) -> Path | None:
    """Find the checkpoint file for a given label."""
    matches = list(CHECKPOINT_DIR.glob(f"densenet-{safe_name}-*-epoch=*.ckpt"))
    if not matches:
        return None
    return max(matches, key=os.path.getctime)


def apply_bcops(probs: np.ndarray, presence_thresh: float, absence_thresh: float) -> np.ndarray:
    """
    Classify samples using BCOPS dual thresholds.

    Returns array of strings: 'Positive', 'Negative', or 'Uncertain'.
    """
    predictions = np.full(len(probs), 'Uncertain', dtype=object)
    predictions[probs >= (1.0 - presence_thresh)] = 'Positive'
    predictions[probs <= absence_thresh] = 'Negative'
    return predictions


def plot_confusion_matrix(true_labels: np.ndarray, bcops_preds: np.ndarray,
                          label_name: str, save_path: Path,
                          presence_thresh: float, absence_thresh: float,
                          alpha: float, auc: float = None):
    """
    Generate a 3x3 confusion matrix: true {Positive, Uncertain, Negative} vs
    predicted {Positive, Uncertain, Negative}.
    """
    true_categories = ['Positive', 'Uncertain', 'Negative']
    pred_categories = ['Positive', 'Uncertain', 'Negative']

    # Map true label values: 1 -> Positive, -1 -> Uncertain, 0 -> Negative
    true_cat_map = {1: 'Positive', -1: 'Uncertain', 0: 'Negative'}

    # Build the matrix
    matrix = np.zeros((3, 3), dtype=int)
    for i, true_cat in enumerate(true_categories):
        true_val = {v: k for k, v in true_cat_map.items()}[true_cat]
        mask = true_labels == true_val
        for j, pred_cat in enumerate(pred_categories):
            matrix[i, j] = np.sum(mask & (bcops_preds == pred_cat))

    # Percentages per row
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1  # avoid div by zero
    pct_matrix = matrix / row_totals * 100

    # Annotation: count + percentage
    annot = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            annot[i, j] = f"{matrix[i, j]}\n({pct_matrix[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(18, 11), facecolor='white')
    sns.heatmap(
        matrix, annot=annot, fmt='', cmap='mako',
        xticklabels=pred_categories, yticklabels=true_categories,
        ax=ax, linewidths=2, linecolor='white',
        annot_kws={'fontsize': 22, 'fontweight': 'bold'},
        cbar_kws={'label': 'Count'}
    )

    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    title = f"Label: {label_name}, AUC: {auc_str}"
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted (BCOPS)', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel('True Label', fontsize=22, fontweight='bold', labelpad=15)
    ax.tick_params(axis='both', labelsize=18)

    # Make tick labels bold
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved confusion matrix: {save_path}")


def test_single_label(label: str, base_config: dict):
    """Test a single per-label model with BCOPS conformal predictions."""
    safe_name = label.replace(" ", "_").lower()

    # Find checkpoint
    checkpoint = find_checkpoint(safe_name)
    if checkpoint is None:
        print(f"  [SKIP] No checkpoint found for '{label}'")
        return False

    # Check that thresholds exist
    label_dir = OUTPUT_BASE / safe_name
    thresholds_path = label_dir / 'bcops_thresholds.pt'
    if not thresholds_path.exists():
        print(f"  [ERROR] No thresholds found for '{label}'. "
              f"Run calibrate_per_label.py first.")
        return False

    print(f"\n{'=' * 70}")
    print(f"TESTING: {label}")
    print(f"  Checkpoint: {checkpoint.name}")
    print(f"{'=' * 70}")

    # Generate per-label config
    config_path = generate_config(label, base_config)

    # Override evaluation checkpoint
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["evaluation"]["checkpoint_path"] = str(checkpoint)
    cfg["conformal"]["calibration_dir"] = str(label_dir)
    # Disable two-stage loading — we're loading a finished checkpoint, not training
    cfg["loss"]["pretrained_checkpoint"] = None
    cfg["loss"]["reinit_classifier"] = False
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Load thresholds
    thresholds = torch.load(thresholds_path, weights_only=True)
    presence_thresh = thresholds['presence_thresholds'][0].item()
    absence_thresh = thresholds['absence_thresholds'][0].item()
    alpha = thresholds.get('alpha', 0.1)
    print(f"  Presence threshold: {1.0 - presence_thresh:.3f} (nonconformity: {presence_thresh:.3f})")
    print(f"  Absence threshold:  {absence_thresh:.3f}")
    print(f"  Alpha: {alpha}")

    # Load model
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  Device: {device}")

    model = CXRClassifier.load_from_checkpoint(
        checkpoint, strict=False, config_path=str(config_path)
    )
    model = model.to(device)
    model.eval()

    # Load test data
    data_module = get_data_module(config_path=str(config_path), num_workers=4)
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    # Run inference
    print(f"  Running inference on test set...")
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels_batch in tqdm(test_loader, desc=f"  {label}", unit="batch"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(labels_batch.cpu())

    all_probs = torch.cat(all_probs, dim=0).squeeze(1).numpy()  # (N,)
    all_labels = torch.cat(all_labels, dim=0).squeeze(1).numpy()  # (N,)
    print(f"  Inference complete: {len(all_probs)} samples")

    # Apply BCOPS predictions
    bcops_preds = apply_bcops(all_probs, presence_thresh, absence_thresh)

    # Filter out uncertain true labels (-1) for metrics
    definite_mask = all_labels >= 0
    probs_definite = all_probs[definite_mask]
    labels_definite = all_labels[definite_mask]
    preds_definite = bcops_preds[definite_mask]

    # Compute AUC
    try:
        auc = roc_auc_score(labels_definite, probs_definite)
        print(f"  AUC: {auc:.4f} (n={len(labels_definite)})")
    except ValueError:
        auc = None
        print(f"  AUC: N/A (single class)")

    # BCOPS prediction distribution (all samples)
    for cat in ['Positive', 'Uncertain', 'Negative']:
        count = np.sum(bcops_preds == cat)
        pct = count / len(bcops_preds) * 100 if len(bcops_preds) > 0 else 0
        print(f"  BCOPS {cat:<12}: {count:>6} ({pct:.1f}%)")

    # Save CSV (all samples, including uncertain true labels)
    label_dir.mkdir(parents=True, exist_ok=True)
    csv_path = label_dir / 'test_results.csv'
    df = pd.DataFrame({
        'sample_id': range(len(all_probs)),
        'prob': [f"{p:.6f}" for p in all_probs],
        'true_label': all_labels.astype(int),
        'bcops_prediction': bcops_preds,
    })
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    # Generate 3x3 confusion matrix (including uncertain true labels)
    cm_path = label_dir / 'confusion_matrix.png'
    plot_confusion_matrix(
        all_labels, bcops_preds, label,
        cm_path, presence_thresh, absence_thresh, alpha, auc
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test per-label binary models with BCOPS conformal predictions."
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Test a single label (e.g. 'Cardiomegaly'). "
             "If omitted, tests all labels with checkpoints and thresholds."
    )
    args = parser.parse_args()

    with open(BASE_CONFIG) as f:
        base_config = yaml.safe_load(f)

    if args.label:
        if args.label not in ALL_LABELS:
            print(f"Unknown label: {args.label}")
            print(f"Available labels: {ALL_LABELS}")
            return
        labels = [args.label]
    else:
        labels = ALL_LABELS

    print(f"\n{'=' * 80}")
    print("BCOPS CONFORMAL TEST: PER-LABEL BINARY MODELS")
    print(f"{'=' * 80}")
    print(f"\nLabels to process: {len(labels)}")

    results = {}
    auc_results = {}
    for i, label in enumerate(labels, 1):
        print(f"\n[{i}/{len(labels)}] {label}")
        success = test_single_label(label, base_config)
        results[label] = success

    # Summary
    tested = [l for l, s in results.items() if s]
    skipped = [l for l, s in results.items() if not s]

    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Tested: {len(tested)}/{len(labels)}")
    for label in tested:
        print(f"    [OK] {label}")
    if skipped:
        print(f"  Skipped: {len(skipped)}")
        for label in skipped:
            print(f"    [--] {label}")
    print(f"\nResults saved to: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
