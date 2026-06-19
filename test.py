import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from train import CXRClassifier
from data import CXRDataModule

def compute_per_class_nonconformity(probs: torch.Tensor, labels: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class nonconformity scores.

    For each class separately, nonconformity = 1 if true label is 1 but predicted as 0
    (i.e., we miss a positive case).

    Args:
        probs: Predictions (N, num_classes) in [0, 1]
        labels: True labels (N, num_classes) with values {0, 1, -1}
        thresholds: Per-class thresholds (num_classes,)

    Returns:
        nonconformity: Per-sample, per-class indicator (N, num_classes)
                      1 = missed this class, 0 = caught it
    """
    # Clean labels: replace -1 with 0
    labels_clean = labels.clone()
    labels_clean[labels_clean == -1] = 0

    # Predictions: class in set if prob > threshold
    predictions = (probs >= thresholds.unsqueeze(0)).float()

    # Nonconformity: true=1 but predicted=0 (false negative)
    nonconformity = labels_clean * (1 - predictions)

    return nonconformity


def calibrate_per_class_thresholds(data_module, model, device, config_path: str = "config.yaml") -> torch.Tensor:
    """
    CALIBRATION: Find optimal threshold for EACH of 14 pathologies using exact quantiles.

    For each class independently:
    - Isolates true positive samples (handles extreme class imbalance).
    - Uses torch.quantile to find the exact probability threshold that bounds false negatives to alpha.
    """
    print("\n" + "=" * 70)
    print("CALIBRATION STEP: Finding per-class exact quantiles")
    print("=" * 70)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    conformal_config = config.get('conformal', {})
    alpha = conformal_config.get('alpha', 0.1)
    calibration_dir = conformal_config.get('calibration_dir', './conformal_calibration')

    pathologies = config.get('use_labels', [])
    num_classes = len(pathologies) if pathologies else 14

    Path(calibration_dir).mkdir(parents=True, exist_ok=True)

    target_coverage = 1.0 - alpha

    print(f"\nTarget conditional coverage per class: {target_coverage:.1%}")
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

    print("\nCalculating empirical quantiles for per-class thresholds...")
    thresholds = []

    for class_idx in range(num_classes):
        class_probs = all_probs[:, class_idx]
        class_labels = labels_clean[:, class_idx]

        # ONLY look at samples where the disease is actually present
        positive_mask = class_labels == 1
        positive_probs = class_probs[positive_mask]
        
        n_positives = len(positive_probs)

        if n_positives < 10:
            print(f"  Warning: Class {class_idx:2d} has too few positive samples ({n_positives}). Fallback to 0.5.")
            best_threshold = 0.5
            coverage = 0.0
        else:
            # Finite sample correction for strict Conformal Prediction guarantees
            # We want the threshold at the alpha quantile of the TRUE POSITIVE probability distribution
            q_level = np.ceil((n_positives + 1) * alpha) / n_positives
            q_level = min(max(q_level, 0.0), 1.0)
            
            # Exact quantile calculation
            best_threshold = torch.quantile(positive_probs, q_level).item()

            # Verify actual conditional coverage
            predictions = (class_probs >= best_threshold).float()
            missed = class_labels * (1 - predictions)
            coverage = 1.0 - (missed[positive_mask].sum().item() / n_positives)

        thresholds.append(best_threshold)
        print(f"  Class {class_idx:2d}: threshold = {best_threshold:.3f}, positive cases = {n_positives}, conditional coverage = {coverage:.1%}")

    thresholds = torch.tensor(thresholds)

    # Save thresholds
    calib_file = os.path.join(calibration_dir, 'per_class_thresholds.pt')
    torch.save({
        'thresholds': thresholds,
        'alpha': alpha
    }, calib_file)
    print(f"\n✓ Saved per-class thresholds to: {calib_file}")

    return thresholds


def create_prediction_sets_per_class(probs: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """
    Create prediction SETS using per-class thresholds.
    """
    return (probs >= thresholds.unsqueeze(0)).float()


def test_with_per_class_conformal(config_path: str = "config.yaml"):
    """
    Main test function with per-class conformal prediction.
    """
    print("\n" + "=" * 70)
    print("CONFORMAL PREDICTION: PER-CLASS QUANTILES")
    print("=" * 70)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_config = config.get('output', {})

    print("\n[1] Loading trained model...")
    checkpoint_dir = output_config.get('checkpoint_dir', './checkpoints')
    checkpoint_files = list(Path(checkpoint_dir).glob('densenet-*.ckpt'))

    if not checkpoint_files:
        print(f"  ✗ No checkpoints found in {checkpoint_dir}")
        return

    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"  ✓ Found checkpoint: {latest_checkpoint.name}")

    # Hardware Acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Let PyTorch Lightning handle the state_dict mapping perfectly
    model = CXRClassifier.load_from_checkpoint(
        latest_checkpoint, 
        strict=False, 
        config_path=config_path
    )    
    model = model.to(device)
    model.eval()

    print(f"  ✓ Model loaded ({len(model.state_dict())} weights)")

    print("\n[2] Loading data module...")
    # Increased workers to speed up I/O processing
    data_module = CXRDataModule(config_path=config_path, num_workers=0)
    data_module.setup(stage='test')
    print(f"  ✓ Data module loaded")

    print("\n[3] Calibration step...")
    # Pass device to the calibration function
    thresholds = calibrate_per_class_thresholds(data_module, model, device, config_path)

    print("\n" + "=" * 70)
    print("PREDICTION STEP: Creating per-class prediction sets")
    print("=" * 70)

    pathologies = config.get('use_labels', [])
    if not pathologies:
        pathologies = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                      'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                      'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    print(f"\nGenerating per-class prediction sets on TEST data...")
    
    # Use the test_split_csv to evaluate the conformal thresholds
    test_loader = data_module.test_dataloader()

    results = {
        'predictions': [],
        'prediction_sets': [],
        'true_labels': [],
        'nonconformity': [],
        'thresholds': thresholds
    }

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Send tensors to GPU/MPS
            images = images.to(device)
            
            logits = model(images)
            probs = torch.sigmoid(logits).cpu() # Bring back to CPU for math and storage
            labels = labels.cpu()

            sets = create_prediction_sets_per_class(probs, thresholds)
            nonconf = compute_per_class_nonconformity(probs, labels, thresholds)

            results['predictions'].append(probs)
            results['prediction_sets'].append(sets)
            results['true_labels'].append(labels)
            results['nonconformity'].append(nonconf)

    for key in ['predictions', 'prediction_sets', 'true_labels', 'nonconformity']:
        results[key] = torch.cat(results[key], dim=0)

    if len(results['predictions']) > 0:
        print("\n" + "-" * 70)
        print("Example: Sample 0 - Per-Class Prediction Set")
        print("-" * 70)

        sample_idx = 0
        print(f"\nSample {sample_idx}:")
        print(f"{'Pathology':<35} {'Prob':<8} {'Thresh':<8} {'In Set?':<8} {'True':<8}")
        print("-" * 70)

        for i, pathology in enumerate(pathologies):
            prob = results['predictions'][sample_idx, i].item()
            thresh = results['thresholds'][i].item()
            in_set = int(results['prediction_sets'][sample_idx, i].item())
            true_val = int(results['true_labels'][sample_idx, i].item())

            set_marker = "✓ YES" if in_set else "  no"
            true_marker = "✓" if true_val == 1 else " "

            print(f"{pathology:<35} {prob:.3f}    {thresh:.3f}    {set_marker:<8} {true_marker}{true_val}")

        nonconf = results['nonconformity'][sample_idx]
        missed_classes = [pathologies[i] for i in range(len(pathologies)) if nonconf[i] == 1]

        print(f"\nMissed classes (nonconformity): {missed_classes if missed_classes else 'None'}")

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    calibration_dir = config.get('conformal', {}).get('calibration_dir', './conformal_calibration')
    Path(calibration_dir).mkdir(parents=True, exist_ok=True)

    results_file = os.path.join(calibration_dir, 'conformal_predictions_per_class.pt')
    torch.save({
        'predictions': results['predictions'],
        'prediction_sets': results['prediction_sets'],
        'true_labels': results['true_labels'],
        'nonconformity': results['nonconformity'],
        'thresholds': results['thresholds'],
        'pathologies': pathologies
    }, results_file)

    print(f"\n✓ Saved predictions to: {results_file}")

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    labels_clean = results['true_labels'].clone()
    labels_clean[labels_clean == -1] = 0

    print("\nFinal Conditional Per-class coverage (% TRUE positive samples caught):")
    for i, pathology in enumerate(pathologies):
        class_labels = labels_clean[:, i]
        class_nonconf = results['nonconformity'][:, i]
        
        positive_mask = class_labels == 1
        n_positives = positive_mask.sum().item()
        
        if n_positives > 0:
            caught_positives = (class_nonconf[positive_mask] == 0).float().sum().item()
            coverage = caught_positives / n_positives
        else:
            coverage = 0.0
            
        print(f"  {pathology:<35} {coverage:.1%} (from {int(n_positives)} cases)")

    all_correct = (results['nonconformity'] == 0).all(dim=1).float()
    overall_coverage = all_correct.mean().item()
    print(f"\nOverall strict coverage (ALL positive classes caught for a patient): {overall_coverage:.1%}")

    avg_set_size = results['prediction_sets'].sum(dim=1).mean().item()
    print(f"Average prediction set size: {avg_set_size:.1f} / {len(pathologies)} classes")

    print("\n✅ Per-class conformal prediction complete!")


if __name__ ==  "__main__":
    print('='*80)
    print('Running Per-Class Conformal Prediction')
    print('='*80)
    test_with_per_class_conformal(config_path="/gpfs0/tamyr/users/alonfi/XRay/config.yaml")