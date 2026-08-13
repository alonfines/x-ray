#!/usr/bin/env python3
"""
Test per-label binary models with BCOPS conformal predictions.

For each per-label checkpoint, runs test inference, applies calibrated
BCOPS thresholds, saves CSV results and confusion matrix PNGs to:
    conformal_calibration/mimic/per_label/{label}/

Requires calibration to be run first (calibrate_per_label.py).

Usage:
    python test_per_label.py
    python test_per_label.py --label 'Atelectasis'
    python test_per_label.py --include-val   # use thresholds from conformal+val calibration
    python test_per_label.py --hierarchy     # use hierarchy-corrected checkpoints
"""
import argparse

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

from data import get_data_module
from train import CXRClassifier
from utils import (
    BASE_CONFIG, OUTPUT_BASE, OUTPUT_BASE_WITH_VAL,
    OUTPUT_BASE_HIER, OUTPUT_BASE_WITH_VAL_HIER,
    find_checkpoint, generate_config, parse_labels, safe_name,
)


def apply_bcops(probs: np.ndarray, presence_thresh: float, absence_thresh: float) -> np.ndarray:
    """Classify samples using BCOPS dual thresholds.

    Each sample is included in the positive set if prob >= (1 - presence_thresh)
    and in the negative set if prob <= absence_thresh.  When thresholds overlap,
    a sample can belong to both sets (uncertain) or neither (also uncertain).
    """
    in_positive = probs >= (1.0 - presence_thresh)
    in_negative = probs <= absence_thresh

    predictions = np.full(len(probs), 'Uncertain', dtype=object)
    predictions[in_positive & ~in_negative] = 'Positive'
    predictions[in_negative & ~in_positive] = 'Negative'
    # Both or neither → stays 'Uncertain'
    return predictions


def plot_confusion_matrix(true_labels: np.ndarray, bcops_preds: np.ndarray,
                          label_name: str, save_path: Path, auc: float = None):
    """Generate a 3x3 confusion matrix: true {Positive, Uncertain, Negative} vs
    predicted {Positive, Uncertain, Negative}."""
    categories = ['Positive', 'Uncertain', 'Negative']
    true_val_map = {'Positive': 1, 'Uncertain': -1, 'Negative': 0}

    matrix = np.zeros((3, 3), dtype=int)
    for i, true_cat in enumerate(categories):
        mask = true_labels == true_val_map[true_cat]
        for j, pred_cat in enumerate(categories):
            matrix[i, j] = np.sum(mask & (bcops_preds == pred_cat))

    row_totals = matrix.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    pct_matrix = matrix / row_totals * 100

    annot = np.empty_like(matrix, dtype=object)
    for i in range(3):
        for j in range(3):
            annot[i, j] = f"{pct_matrix[i, j]:.1f}%\n({matrix[i, j]})"

    fig, ax = plt.subplots(figsize=(18, 11), facecolor='white')
    sns.heatmap(
        pct_matrix, annot=annot, fmt='', cmap='YlOrRd',
        xticklabels=categories, yticklabels=categories,
        ax=ax, linewidths=2, linecolor='white',
        annot_kws={'fontsize': 22, 'fontweight': 'bold'},
        cbar_kws={'label': 'Row %'}, vmin=0, vmax=100
    )

    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    ax.set_title(f"Label: {label_name}, AUC: {auc_str}",
                 fontsize=28, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted (BCOPS)', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel('True Label', fontsize=22, fontweight='bold', labelpad=15)
    ax.tick_params(axis='both', labelsize=18)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved confusion matrix: {save_path}")


def test_single_label(label: str, base_config: dict,
                      include_val: bool = False, hierarchy: bool = False):
    """Test a single per-label model with BCOPS conformal predictions."""
    name = safe_name(label)

    checkpoint = find_checkpoint(label, hierarchy=hierarchy)
    if checkpoint is None:
        print(f"  [SKIP] No checkpoint found for '{label}'")
        return False

    if hierarchy:
        output_base = OUTPUT_BASE_WITH_VAL_HIER if include_val else OUTPUT_BASE_HIER
    else:
        output_base = OUTPUT_BASE_WITH_VAL if include_val else OUTPUT_BASE
    label_dir = output_base / name
    thresholds_path = label_dir / 'bcops_thresholds.pt'
    if not thresholds_path.exists():
        flag = " --include-val" if include_val else ""
        hier_flag = " --hierarchy" if hierarchy else ""
        print(f"  [ERROR] No thresholds found for '{label}'. "
              f"Run calibrate_per_label.py{flag}{hier_flag} first.")
        return False

    print(f"\n{'=' * 70}")
    print(f"TESTING: {label}")
    print(f"  Checkpoint: {checkpoint.name}")
    if hierarchy:
        print(f"  Hierarchy correction: enabled")
    if include_val:
        print(f"  Thresholds from: conformal + validation sets")
    print(f"{'=' * 70}")

    config_path = generate_config(label, base_config, hierarchy=hierarchy)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["evaluation"]["checkpoint_path"] = str(checkpoint)
    cfg["conformal"]["calibration_dir"] = str(label_dir)
    cfg["loss"]["pretrained_checkpoint"] = None
    cfg["loss"]["reinit_classifier"] = False
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    thresholds = torch.load(thresholds_path, weights_only=True)
    presence_thresh = thresholds['presence_thresholds'][0].item()
    absence_thresh = thresholds['absence_thresholds'][0].item()
    alpha = thresholds.get('alpha', 0.1)
    print(f"  Presence threshold: {1.0 - presence_thresh:.3f} (nonconformity: {presence_thresh:.3f})")
    print(f"  Absence threshold:  {absence_thresh:.3f}")
    print(f"  Alpha: {alpha}")

    # Test predictions can be shared across both modes (same checkpoint, same test set)
    # Check both directories for cached predictions
    predictions_file = label_dir / f'{checkpoint.stem}_test.pt'
    other_dir = OUTPUT_BASE / name if include_val else OUTPUT_BASE_WITH_VAL / name
    if hierarchy:
        other_dir = OUTPUT_BASE_HIER / name if include_val else OUTPUT_BASE_WITH_VAL_HIER / name
    other_predictions = other_dir / f'{checkpoint.stem}_test.pt'

    if predictions_file.exists():
        print(f"  Loading cached predictions: {predictions_file}")
        cached = torch.load(predictions_file, weights_only=True)
        all_probs = cached['probs'].numpy()
        all_labels = cached['labels'].numpy()
        print(f"  Loaded {len(all_probs)} samples from cache")
    elif other_predictions.exists():
        print(f"  Loading cached predictions: {other_predictions}")
        cached = torch.load(other_predictions, weights_only=True)
        all_probs = cached['probs'].numpy()
        all_labels = cached['labels'].numpy()
        print(f"  Loaded {len(all_probs)} samples from cache")
    else:
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

        data_module = get_data_module(config_path=str(config_path), num_workers=4)
        data_module.setup(stage='test')

        print(f"  Running inference on test set...")
        all_probs_list = []
        all_labels_list = []
        with torch.no_grad():
            for images, labels_batch in tqdm(data_module.test_dataloader(), desc=f"  {label}", unit="batch"):
                images = images.to(device)
                logits = model(images)
                all_probs_list.append(torch.sigmoid(logits).cpu())
                all_labels_list.append(labels_batch.cpu())

        all_probs_t = torch.cat(all_probs_list, dim=0).squeeze(1)
        all_labels_t = torch.cat(all_labels_list, dim=0).squeeze(1)

        torch.save({'probs': all_probs_t, 'labels': all_labels_t}, predictions_file)
        print(f"  Cached predictions to: {predictions_file}")

        all_probs = all_probs_t.numpy()
        all_labels = all_labels_t.numpy()

    print(f"  Inference complete: {len(all_probs)} samples")

    # Use labels directly from the DataModule — no hierarchy correction.
    # The model was trained without hierarchy propagation, so test labels
    # must match what the model learned. The test DataModule preserves
    # uncertain labels (-1) for the 3x3 confusion matrix.

    bcops_preds = apply_bcops(all_probs, presence_thresh, absence_thresh)

    # AUC on definite labels only (requires binary ground truth)
    definite_mask = all_labels >= 0
    try:
        auc = roc_auc_score(all_labels[definite_mask], all_probs[definite_mask])
        print(f"  AUC: {auc:.4f} (n={definite_mask.sum()})")
    except ValueError:
        auc = None
        print(f"  AUC: N/A (single class)")

    for cat in ['Positive', 'Uncertain', 'Negative']:
        count = np.sum(bcops_preds == cat)
        pct = count / len(bcops_preds) * 100 if len(bcops_preds) > 0 else 0
        print(f"  BCOPS {cat:<12}: {count:>6} ({pct:.1f}%)")

    label_dir.mkdir(parents=True, exist_ok=True)
    csv_path = label_dir / 'test_results.csv'
    pd.DataFrame({
        'sample_id': range(len(all_probs)),
        'prob': [f"{p:.6f}" for p in all_probs],
        'true_label': all_labels.astype(int),
        'bcops_prediction': bcops_preds,
    }).to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    plot_confusion_matrix(
        all_labels, bcops_preds, label,
        label_dir / f'confusion_matrix_{name}.png', auc
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test per-label binary models with BCOPS conformal predictions."
    )
    parser.add_argument("--label", type=str, default=None,
                        help="Test a single label (e.g. 'Cardiomegaly').")
    parser.add_argument("--include-val", action="store_true",
                        help="Use thresholds from conformal+validation calibration.")
    parser.add_argument("--hierarchy", action="store_true",
                        help="Use hierarchy-corrected checkpoints and thresholds.")
    args = parser.parse_args()

    with open(BASE_CONFIG) as f:
        base_config = yaml.safe_load(f)

    labels = parse_labels(args.label)
    if args.hierarchy:
        output_base = OUTPUT_BASE_WITH_VAL_HIER if args.include_val else OUTPUT_BASE_HIER
    else:
        output_base = OUTPUT_BASE_WITH_VAL if args.include_val else OUTPUT_BASE

    print(f"\n{'=' * 80}")
    print("BCOPS CONFORMAL TEST: PER-LABEL BINARY MODELS")
    if args.hierarchy:
        print("  (hierarchy label correction enabled)")
    if args.include_val:
        print("  (using thresholds from conformal + validation calibration)")
    print(f"{'=' * 80}")
    print(f"\nLabels to process: {len(labels)}")

    results = {}
    for i, label in enumerate(labels, 1):
        print(f"\n[{i}/{len(labels)}] {label}")
        results[label] = test_single_label(
            label, base_config, args.include_val, args.hierarchy)

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
    print(f"\nResults saved to: {output_base}")


if __name__ == "__main__":
    main()
