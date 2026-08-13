#!/usr/bin/env python3
"""
Calibrate BCOPS conformal thresholds for per-label binary models.

For each per-label checkpoint, loads the model, runs calibration on the
conformal split, and saves thresholds to:
    conformal_calibration/mimic/per_label/{label}/bcops_thresholds.pt

Usage:
    python calibrate_per_label.py
    python calibrate_per_label.py --label 'Atelectasis'
    python calibrate_per_label.py --include-val   # use conformal + validation sets
    python calibrate_per_label.py --hierarchy     # use hierarchy-corrected checkpoints
"""
import argparse

import torch
import yaml

from data import get_data_module, parse_labels_config
from train import CXRClassifier
from calculate_conformal_pred import _calibrate_marginal
from utils import (
    BASE_CONFIG, OUTPUT_BASE, OUTPUT_BASE_WITH_VAL,
    OUTPUT_BASE_HIER, OUTPUT_BASE_WITH_VAL_HIER, PROJECT_ROOT,
    find_checkpoint, generate_config, parse_labels, safe_name,
)


def calibrate_single_label(label: str, base_config: dict,
                           include_val: bool = False, hierarchy: bool = False):
    """Calibrate BCOPS thresholds for a single per-label model."""
    name = safe_name(label)

    checkpoint = find_checkpoint(label, hierarchy=hierarchy)
    if checkpoint is None:
        print(f"  [SKIP] No checkpoint found for '{label}'")
        return False

    print(f"\n{'=' * 70}")
    print(f"CALIBRATING: {label}")
    print(f"  Checkpoint: {checkpoint.name}")
    if hierarchy:
        print(f"  Hierarchy correction: enabled")
    if include_val:
        print(f"  Using: conformal + validation sets")
    print(f"{'=' * 70}")

    config_path = generate_config(label, base_config, hierarchy=hierarchy)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if hierarchy:
        output_base = OUTPUT_BASE_WITH_VAL_HIER if include_val else OUTPUT_BASE_HIER
    else:
        output_base = OUTPUT_BASE_WITH_VAL if include_val else OUTPUT_BASE
    label_output_dir = output_base / name
    label_output_dir.mkdir(parents=True, exist_ok=True)
    cfg["evaluation"]["checkpoint_path"] = str(checkpoint)
    cfg["conformal"]["calibration_dir"] = str(label_output_dir)
    cfg["loss"]["pretrained_checkpoint"] = None
    cfg["loss"]["reinit_classifier"] = False

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    suffix = '_conformal_val' if include_val else '_conformal'
    predictions_file = label_output_dir / f'{checkpoint.stem}{suffix}.pt'

    if predictions_file.exists():
        print(f"  Loading cached predictions: {predictions_file}")
        cached = torch.load(predictions_file, weights_only=True)
        all_probs = cached['probs']
        all_labels = cached['labels']
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
        print(f"  Model loaded ({len(model.state_dict())} weights)")

        data_module = get_data_module(config_path=str(config_path), num_workers=4)
        data_module.setup(stage='fit')

        dataloaders = [data_module.conformal_dataloader()]
        if include_val:
            dataloaders.append(data_module.val_dataloader())

        print("  Collecting predictions...")
        all_probs_list = []
        all_labels_list = []
        with torch.no_grad():
            for dl in dataloaders:
                for images, labels_batch in dl:
                    images = images.to(device)
                    logits = model(images)
                    all_probs_list.append(torch.sigmoid(logits).cpu())
                    all_labels_list.append(labels_batch.cpu())

        all_probs = torch.cat(all_probs_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        torch.save({'probs': all_probs, 'labels': all_labels}, predictions_file)
        print(f"  Cached predictions to: {predictions_file}")

    # Use labels directly from the DataModule.
    # When hierarchy correction is enabled, the DataModule already applied
    # the correction to conformal/val splits, so labels match what the model learned.
    pathologies, _ = parse_labels_config(cfg)
    num_classes = len(pathologies)
    labels_clean = all_labels  # already (N, 1) from DataModule

    alpha = cfg.get('conformal', {}).get('alpha', 0.1)
    presence_thresholds, absence_thresholds = _calibrate_marginal(
        all_probs, labels_clean, alpha, num_classes, pathologies
    )

    calib_file = label_output_dir / 'bcops_thresholds.pt'
    torch.save({
        'presence_thresholds': torch.tensor(presence_thresholds),
        'absence_thresholds': torch.tensor(absence_thresholds),
        'alpha': alpha,
        'pathologies': pathologies
    }, calib_file)
    print(f"  Thresholds saved to: {calib_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate BCOPS conformal thresholds for per-label binary models."
    )
    parser.add_argument("--label", type=str, default=None,
                        help="Calibrate a single label (e.g. 'Cardiomegaly').")
    parser.add_argument("--include-val", action="store_true",
                        help="Use conformal + validation sets for calibration.")
    parser.add_argument("--hierarchy", action="store_true",
                        help="Use hierarchy-corrected checkpoints and labels.")
    args = parser.parse_args()

    with open(BASE_CONFIG) as f:
        base_config = yaml.safe_load(f)

    labels = parse_labels(args.label)

    print(f"\n{'=' * 80}")
    print("BCOPS CONFORMAL CALIBRATION: PER-LABEL BINARY MODELS")
    if args.hierarchy:
        print("  (hierarchy label correction enabled)")
    if args.include_val:
        print("  (using conformal + validation sets)")
    print(f"{'=' * 80}")
    print(f"\nLabels to process: {len(labels)}")

    results = {}
    for i, label in enumerate(labels, 1):
        print(f"\n[{i}/{len(labels)}] {label}")
        results[label] = calibrate_single_label(
            label, base_config, args.include_val, args.hierarchy)

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
