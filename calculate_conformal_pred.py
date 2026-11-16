import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data import get_data_module, parse_labels_config
from inference import load_model, predict

MIN_CALIBRATION_SAMPLES = 10
FALLBACK_THRESHOLD = 0.5


def definite_label_masks(csv_path: str, pathologies: list[str]) -> dict[str, np.ndarray]:
    """For each pathology, a boolean mask of rows whose original label is definite (1.0 or 0.0)."""
    df = pd.read_csv(csv_path)
    return {p: df[p].isin([0.0, 1.0]).to_numpy() for p in pathologies}


def compute_class_weights(labels: torch.Tensor, weighting_scheme: str = "inverse_prevalence") -> torch.Tensor:
    """Per-class weights from labels (N, C) in {0, 1}: inverse prevalence (mean-normalised) or uniform."""
    if weighting_scheme == "uniform":
        return torch.ones(labels.shape[1])
    weights = 1.0 / (labels.sum(dim=0) / len(labels) + 1e-6)
    return weights / weights.mean()


def _conformal_threshold(scores: torch.Tensor, target_coverage: float, weight: float | None = None):
    """Split-conformal quantile threshold of nonconformity `scores`.

    Returns (threshold, empirical coverage, n). Falls back to 0.5 with zero coverage when
    there are too few samples.
    """
    n = len(scores)
    if n < MIN_CALIBRATION_SAMPLES:
        return FALLBACK_THRESHOLD, 0.0, n
    q_level = np.ceil((n + 1) * target_coverage / (weight or 1.0)) / n
    threshold = torch.quantile(scores, min(max(q_level, 0.0), 1.0)).item()
    return threshold, (scores <= threshold).float().mean().item(), n


def _calibrate_marginal(probs: torch.Tensor, labels: torch.Tensor, alpha: float, pathologies: list[str],
                        use_weights: bool = False, weighting_scheme: str = "uniform",
                        definite: dict[str, np.ndarray] | None = None) -> tuple[list[float], list[float]]:
    """Calibrate per-class presence/absence thresholds independently (marginal coverage, BCOPS).

    Presence: nonconformity 1 - p over positives. Absence: nonconformity p over negatives.
    If `definite` is given, only samples with definite original labels are used.

    Returns:
        (presence_thresholds, absence_thresholds)
    """
    print("\nCalculating per-class thresholds (BCOPS: dual presence/absence)...")
    if use_weights:
        print(f"  Using weighted quantiles (weighting_scheme: {weighting_scheme})")
    if definite:
        print("  Filtering to definite-label samples only per class")

    target_coverage = 1.0 - alpha
    class_weights = compute_class_weights(labels, weighting_scheme) if use_weights else None
    presence_thresholds, absence_thresholds = [], []

    for i, pathology in enumerate(pathologies):
        keep = torch.from_numpy(definite[pathology]) if definite else torch.ones(len(labels), dtype=torch.bool)
        weight = class_weights[i].item() if use_weights else None
        weight_str = f" (weight={weight:.2f})" if use_weights else ""

        print(f"  Class {i:2d}:")
        for kind, label_value, scores, thresholds in [
            ("Presence", 1, 1.0 - probs[:, i], presence_thresholds),
            ("Absence", 0, probs[:, i], absence_thresholds),
        ]:
            scores = scores[(labels[:, i] == label_value) & keep]
            threshold, coverage, n = _conformal_threshold(scores, target_coverage, weight)
            if n < MIN_CALIBRATION_SAMPLES:
                print(f"  Warning: Class {i:2d} has too few {kind.lower()} samples ({n}). Fallback to 0.5.")
            thresholds.append(threshold)
            print(f"    {kind} threshold = {threshold:.3f}, coverage = {coverage:.1%} ({n} samples){weight_str}")

    return presence_thresholds, absence_thresholds


def calibrate_per_class_thresholds(data_module, model, device, config_path: str = "config.yaml",
                                   conformal_csv_path: str | None = None):
    """CALIBRATION: find per-class BCOPS thresholds (presence/absence) on the conformal split and save them."""
    print("\n" + "=" * 70)
    print("CALIBRATION STEP: Finding BCOPS thresholds (dual presence/absence)")
    print("=" * 70)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    conformal_config = config.get('conformal', {})
    alpha = conformal_config.get('alpha', 0.1)
    calibration_dir = conformal_config.get('calibration_dir', './conformal_calibration')
    pathologies, _ = parse_labels_config(config)

    Path(calibration_dir).mkdir(parents=True, exist_ok=True)

    definite = None
    if conformal_csv_path:
        print(f"\nLoading label certainty from {conformal_csv_path}...")
        definite = definite_label_masks(conformal_csv_path, pathologies)
        for pathology, mask in definite.items():
            print(f"  {pathology:<35} {mask.sum()} definite samples")

    print(f"\nTarget coverage: {1.0 - alpha:.1%}")
    print(f"Calibration set size: {len(data_module.conformal_dataset)} samples")
    print(f"Number of classes: {len(pathologies)}")

    print("\nCollecting calibration predictions...")
    probs, labels = predict(model, data_module.conformal_dataloader(), device)
    labels[labels == -1] = 0

    presence_thresholds, absence_thresholds = _calibrate_marginal(
        probs, labels, alpha, pathologies,
        conformal_config.get('use_weights', False),
        conformal_config.get('weighting_scheme', 'uniform'),
        definite)

    calib_file = os.path.join(calibration_dir, 'bcops_thresholds.pt')
    torch.save({
        'presence_thresholds': torch.tensor(presence_thresholds),
        'absence_thresholds': torch.tensor(absence_thresholds),
        'alpha': alpha,
        'pathologies': pathologies,
    }, calib_file)
    print(f"\n✓ Saved BCOPS thresholds to: {calib_file}")


def main(config_path: str = "config.yaml"):
    print("\n" + "=" * 80)
    print("BCOPS CONFORMAL PREDICTION: CALIBRATION ONLY")
    print("=" * 80)

    print("\n[1] Loading trained model...")
    model, _, device = load_model(config_path)

    print("\n[2] Loading data module...")
    data_module = get_data_module(config_path=config_path, num_workers=4)
    data_module.setup(stage='fit')

    print("\n[3] Running calibration...")
    calibrate_per_class_thresholds(data_module, model, device, config_path, data_module.conformal_csv)

    print("\n✅ Calibration complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/gpfs0/tamyr/users/alonfi/XRay/config.yaml')
    args = parser.parse_args()
    main(config_path=args.config)
