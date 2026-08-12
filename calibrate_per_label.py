#!/usr/bin/env python3
"""
Calibrate BCOPS conformal thresholds for per-label binary models.

For each per-label checkpoint, loads the model, runs calibration on the
conformal split, and saves thresholds to:
    conformal_calibration/mimic/per_label/{label}/bcops_thresholds.pt

Usage:
    # All labels (skips labels without checkpoints)
    python calibrate_per_label.py

    # Single label
    python calibrate_per_label.py --label 'Atelectasis'
"""
import argparse
import os

import torch
import yaml
from pathlib import Path

from data import get_data_module, parse_labels_config
from train import CXRClassifier
from train_per_label import generate_config, ALL_LABELS
from calculate_conformal_pred import _calibrate_marginal

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_CONFIG = PROJECT_ROOT / "config.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "mimic" / "per_label"
OUTPUT_BASE = PROJECT_ROOT / "conformal_calibration" / "mimic" / "per_label"


def find_checkpoint(safe_name: str) -> Path | None:
    """Find the checkpoint file for a given label."""
    matches = list(CHECKPOINT_DIR.glob(f"densenet-{safe_name}-*-epoch=*.ckpt"))
    if not matches:
        return None
    # Return the most recent if multiple exist
    return max(matches, key=os.path.getctime)


def calibrate_single_label(label: str, base_config: dict):
    """Calibrate BCOPS thresholds for a single per-label model."""
    safe_name = label.replace(" ", "_").lower()

    # Find checkpoint
    checkpoint = find_checkpoint(safe_name)
    if checkpoint is None:
        print(f"  [SKIP] No checkpoint found for '{label}'")
        return False

    print(f"\n{'=' * 70}")
    print(f"CALIBRATING: {label}")
    print(f"  Checkpoint: {checkpoint.name}")
    print(f"{'=' * 70}")

    # Generate per-label config with correct overrides
    config_path = generate_config(label, base_config)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    label_output_dir = OUTPUT_BASE / safe_name
    label_output_dir.mkdir(parents=True, exist_ok=True)
    cfg["evaluation"]["checkpoint_path"] = str(checkpoint)
    cfg["conformal"]["calibration_dir"] = str(label_output_dir)
    # Disable two-stage loading — we're loading a finished checkpoint, not training
    cfg["loss"]["pretrained_checkpoint"] = None
    cfg["loss"]["reinit_classifier"] = False

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    predictions_file = label_output_dir / 'conformal_predictions.pt'

    if predictions_file.exists():
        # Reuse cached predictions — skip model inference
        print(f"  Loading cached predictions: {predictions_file}")
        cached = torch.load(predictions_file, weights_only=True)
        all_probs = cached['probs']
        all_labels = cached['labels']
        print(f"  Loaded {len(all_probs)} samples from cache")
    else:
        # Run model inference on conformal set
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
        print(f"  Model loaded ({len(model.state_dict())} weights)")

        data_module = get_data_module(config_path=str(config_path), num_workers=4)
        data_module.setup(stage='fit')

        print("  Collecting conformal predictions...")
        all_probs_list = []
        all_labels_list = []
        with torch.no_grad():
            for images, labels_batch in data_module.conformal_dataloader():
                images = images.to(device)
                logits = model(images)
                all_probs_list.append(torch.sigmoid(logits).cpu())
                all_labels_list.append(labels_batch.cpu())

        all_probs = torch.cat(all_probs_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        # Cache predictions for future re-runs
        torch.save({'probs': all_probs, 'labels': all_labels}, predictions_file)
        print(f"  Cached predictions to: {predictions_file}")

    # Compute thresholds from predictions (no definite-sample filtering —
    # NaN/missing labels are already 0 from the data module, which is correct:
    # "not mentioned" by the radiologist means condition absent)
    pathologies, _ = parse_labels_config(cfg)
    num_classes = len(pathologies)
    labels_clean = all_labels.clone()
    labels_clean[labels_clean == -1] = 0

    alpha = cfg.get('conformal', {}).get('alpha', 0.1)
    presence_thresholds, absence_thresholds = _calibrate_marginal(
        all_probs, labels_clean, alpha, num_classes, pathologies
    )

    # Save thresholds
    presence_thresholds = torch.tensor(presence_thresholds)
    absence_thresholds = torch.tensor(absence_thresholds)
    calib_file = label_output_dir / 'bcops_thresholds.pt'
    torch.save({
        'presence_thresholds': presence_thresholds,
        'absence_thresholds': absence_thresholds,
        'alpha': alpha,
        'pathologies': pathologies
    }, calib_file)
    print(f"  Thresholds saved to: {calib_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate BCOPS conformal thresholds for per-label binary models."
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Calibrate a single label (e.g. 'Cardiomegaly'). "
             "If omitted, calibrates all labels with checkpoints."
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
    print("BCOPS CONFORMAL CALIBRATION: PER-LABEL BINARY MODELS")
    print(f"{'=' * 80}")
    print(f"\nLabels to process: {len(labels)}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Output base: {OUTPUT_BASE}")

    results = {}
    for i, label in enumerate(labels, 1):
        print(f"\n[{i}/{len(labels)}] {label}")
        success = calibrate_single_label(label, base_config)
        results[label] = success

    # Summary
    calibrated = [l for l, s in results.items() if s]
    skipped = [l for l, s in results.items() if not s]

    print(f"\n{'=' * 80}")
    print("CALIBRATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Calibrated: {len(calibrated)}/{len(labels)}")
    for label in calibrated:
        print(f"    [OK] {label}")
    if skipped:
        print(f"  Skipped (no checkpoint): {len(skipped)}")
        for label in skipped:
            print(f"    [--] {label}")


if __name__ == "__main__":
    main()
