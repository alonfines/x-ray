import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from train import CXRClassifier
from data import get_data_module, parse_labels_config


def load_label_certainty_from_csv(csv_path: str, pathologies: List[str]) -> Dict[int, Dict[str, str]]:
    """
    Load original label certainty from CSV.

    Returns:
        {sample_idx: {pathology: certainty_str}} where certainty_str is '1.0', '0.0', '-1.0', or 'missing'
    """
    certainty = {}
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        certainty[idx] = {}
        for pathology in pathologies:
            val = row[pathology]
            if pd.isna(val) or val == '':
                certainty[idx][pathology] = 'missing'
            else:
                certainty[idx][pathology] = str(float(val))

    return certainty


def identify_definite_samples_per_class(certainty_dict: Dict[int, Dict[str, str]], pathologies: List[str]) -> Dict[str, set]:
    """
    For each class, identify which sample indices have definite labels (1.0 or 0.0).

    Returns:
        {pathology: set_of_definite_sample_indices}
    """
    definite_by_class = {p: set() for p in pathologies}

    for sample_idx, labels in certainty_dict.items():
        for pathology in pathologies:
            if labels[pathology] in ('1.0', '0.0'):
                definite_by_class[pathology].add(sample_idx)

    return definite_by_class


def compute_class_weights(labels_clean: torch.Tensor, weighting_scheme: str = "inverse_prevalence") -> torch.Tensor:
    """
    Compute weights for each class based on prevalence.

    Args:
        labels_clean: Labels (N, num_classes) with values {0, 1}
        weighting_scheme: "inverse_prevalence" (rare classes get higher weight) or "uniform"

    Returns:
        weights: (num_classes,) tensor of class weights
    """
    num_classes = labels_clean.shape[1]

    if weighting_scheme == "uniform":
        return torch.ones(num_classes)

    # Inverse prevalence: weight = 1 / prevalence
    prevalence = labels_clean.sum(dim=0) / len(labels_clean)
    weights = 1.0 / (prevalence + 1e-6)  # Add small epsilon to avoid division by zero
    weights = weights / weights.mean()  # Normalize so mean weight = 1

    return weights


def _calibrate_marginal(all_probs: torch.Tensor, labels_clean: torch.Tensor, alpha: float, num_classes: int, pathologies: List[str], use_weights: bool = False, weighting_scheme: str = "uniform", definite_by_class: Dict[str, set] = None) -> tuple:
    """Calibrate per-class thresholds independently (marginal coverage) using BCOPS framework.

    If definite_by_class is provided, only uses samples with definite labels for calibration.

    Returns:
        (presence_thresholds, absence_thresholds): Two lists of thresholds per class
    """
    print("\nCalculating per-class thresholds (BCOPS: dual presence/absence)...")
    if use_weights:
        print(f"  Using weighted quantiles (weighting_scheme: {weighting_scheme})")
    if definite_by_class:
        print(f"  Filtering to definite-label samples only per class")

    presence_thresholds = []
    absence_thresholds = []
    class_weights = compute_class_weights(labels_clean, weighting_scheme) if use_weights else torch.ones(num_classes)

    # Calculate target coverage (1 - alpha)
    target_coverage = 1.0 - alpha

    # Pre-compute definite masks if needed (avoid repeated tensor creation)
    definite_masks = {}
    if definite_by_class:
        for class_idx in range(num_classes):
            definite_set = definite_by_class[pathologies[class_idx]]
            indices = torch.arange(len(labels_clean))
            definite_set_tensor = torch.tensor(list(definite_set), dtype=torch.long)
            definite_masks[class_idx] = torch.isin(indices, definite_set_tensor)

    for class_idx in range(num_classes):
        class_probs = all_probs[:, class_idx]
        class_labels = labels_clean[:, class_idx]

        # Nonconformity scores
        s_pre = 1.0 - class_probs  # Nonconformity for presence
        s_abs = class_probs        # Nonconformity for absence

        # ===== PRESENCE THRESHOLD =====
        positive_mask = class_labels == 1

        # Filter to definite samples if provided
        if definite_by_class:
            definite_mask = definite_masks[class_idx]
            positive_mask = positive_mask & definite_mask

        positive_s_pre = s_pre[positive_mask]
        n_positives = len(positive_s_pre)

        if n_positives < 10:
            print(f"  Warning: Class {class_idx:2d} has too few positive samples ({n_positives}). Fallback to 0.5.")
            presence_threshold = 0.5
            presence_coverage = 0.0
        else:
            if use_weights:
                weight = class_weights[class_idx].item()
                q_level = np.ceil((n_positives + 1) * target_coverage / weight) / n_positives
            else:
                q_level = np.ceil((n_positives + 1) * target_coverage) / n_positives

            q_level = min(max(q_level, 0.0), 1.0)
            presence_threshold = torch.quantile(positive_s_pre, q_level).item()

            # Coverage: fraction of positives where s_pre <= threshold
            presence_coverage = (positive_s_pre <= presence_threshold).float().mean().item()

        # ===== ABSENCE THRESHOLD =====
        negative_mask = class_labels == 0

        # Filter to definite samples if provided
        if definite_by_class:
            definite_mask = definite_masks[class_idx]
            negative_mask = negative_mask & definite_mask

        negative_s_abs = s_abs[negative_mask]
        n_negatives = len(negative_s_abs)

        if n_negatives < 10:
            print(f"  Warning: Class {class_idx:2d} has too few negative samples ({n_negatives}). Fallback to 0.5.")
            absence_threshold = 0.5
            absence_coverage = 0.0
        else:
            if use_weights:
                weight = class_weights[class_idx].item()
                q_level = np.ceil((n_negatives + 1) * target_coverage / weight) / n_negatives
            else:
                q_level = np.ceil((n_negatives + 1) * target_coverage) / n_negatives

            q_level = min(max(q_level, 0.0), 1.0)
            absence_threshold = torch.quantile(negative_s_abs, q_level).item()

            # Coverage: fraction of negatives where s_abs <= threshold
            absence_coverage = (negative_s_abs <= absence_threshold).float().mean().item()

        presence_thresholds.append(presence_threshold)
        absence_thresholds.append(absence_threshold)

        weight_str = f" (weight={class_weights[class_idx]:.2f})" if use_weights else ""
        print(f"  Class {class_idx:2d}:")
        print(f"    Presence threshold = {presence_threshold:.3f}, coverage = {presence_coverage:.1%} ({n_positives} positives){weight_str}")
        print(f"    Absence threshold  = {absence_threshold:.3f}, coverage = {absence_coverage:.1%} ({n_negatives} negatives){weight_str}")

    return presence_thresholds, absence_thresholds


def calibrate_per_class_thresholds(data_module, model, device, config_path: str = "config.yaml", conformal_csv_path: str = None) -> tuple:
    """
    CALIBRATION: Find optimal thresholds for BCOPS conformal prediction.

    Returns:
        (presence_thresholds, absence_thresholds): Two sets of per-class thresholds
    """
    print("\n" + "=" * 70)
    print("CALIBRATION STEP: Finding BCOPS thresholds (dual presence/absence)")
    print("=" * 70)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    conformal_config = config.get('conformal', {})
    alpha = conformal_config.get('alpha', 0.1)
    use_weights = conformal_config.get('use_weights', False)
    weighting_scheme = conformal_config.get('weighting_scheme', 'uniform')
    calibration_dir = conformal_config.get('calibration_dir', './conformal_calibration')

    pathologies, _ = parse_labels_config(config)
    num_classes = len(pathologies)

    Path(calibration_dir).mkdir(parents=True, exist_ok=True)

    target_coverage = 1.0 - alpha

    # Load label certainty if CSV provided
    definite_by_class = None
    if conformal_csv_path:
        print(f"\nLoading label certainty from {conformal_csv_path}...")
        certainty = load_label_certainty_from_csv(conformal_csv_path, pathologies)
        definite_by_class = identify_definite_samples_per_class(certainty, pathologies)
        for pathology, definite_set in definite_by_class.items():
            print(f"  {pathology:<35} {len(definite_set)} definite samples")

    print(f"\nTarget coverage: {target_coverage:.1%}")
    print(f"Calibration set size: {len(data_module.conformal_dataset)} samples")
    print(f"Number of classes: {num_classes}")

    print("\nCollecting calibration predictions...")
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_module.conformal_dataloader():
            images = images.to(device)
            logits = model(images)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Clean labels
    labels_clean = all_labels.clone()
    labels_clean[labels_clean == -1] = 0

    presence_thresholds, absence_thresholds = _calibrate_marginal(all_probs, labels_clean, alpha, num_classes, pathologies, use_weights, weighting_scheme, definite_by_class)

    presence_thresholds = torch.tensor(presence_thresholds)
    absence_thresholds = torch.tensor(absence_thresholds)

    # Save thresholds
    calib_file = os.path.join(calibration_dir, 'bcops_thresholds.pt')
    torch.save({
        'presence_thresholds': presence_thresholds,
        'absence_thresholds': absence_thresholds,
        'alpha': alpha,
        'pathologies': pathologies
    }, calib_file)
    print(f"\n✓ Saved BCOPS thresholds to: {calib_file}")

    return presence_thresholds, absence_thresholds


def main(config_path: str = "config.yaml"):
    """Main calibration function."""
    print("\n" + "=" * 80)
    print("BCOPS CONFORMAL PREDICTION: CALIBRATION ONLY")
    print("=" * 80)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    conformal_config = config.get('conformal', {})
    calibration_dir = conformal_config.get('calibration_dir', './conformal_calibration')

    # Load model
    print("\n[1] Loading trained model...")
    eval_config = config.get('evaluation', {})
    checkpoint_path = eval_config.get('checkpoint_path')

    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            print(f"  ✗ Checkpoint not found at: {checkpoint_path}")
            return
    else:
        checkpoint_dir = eval_config.get('checkpoint_dir', './checkpoints')
        checkpoint_files = list(Path(checkpoint_dir).glob('densenet-*.ckpt'))
        if not checkpoint_files:
            print(f"  ✗ No checkpoints found in {checkpoint_dir}")
            return
        checkpoint_file = max(checkpoint_files, key=os.path.getctime)

    print(f"  ✓ Found checkpoint: {checkpoint_file.name}")

    # Hardware Acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Using device: {device}")

    model = CXRClassifier.load_from_checkpoint(checkpoint_file, strict=False, config_path=config_path)
    model = model.to(device)
    model.eval()
    print(f"  ✓ Model loaded ({len(model.state_dict())} weights)")

    # Load data module
    print("\n[2] Loading data module...")
    data_module = get_data_module(config_path=config_path, num_workers=4)
    data_module.setup(stage='fit')
    print(f"  ✓ Data module loaded (fit stage for conformal)")

    # Calibration
    print("\n[3] Running calibration...")
    data_config = config.get('data', {})
    conformal_csv = os.path.join(data_config.get('working_dir', os.getcwd()), 'conformal_split.csv')
    calibrate_per_class_thresholds(data_module, model, device, config_path, conformal_csv)

    print("\n✅ Calibration complete!")


if __name__ == "__main__":
    main(config_path="/gpfs0/tamyr/users/alonfi/XRay/config.yaml")
