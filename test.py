import os
import yaml
import torch
import pandas as pd
from pathlib import Path
from train import CXRClassifier
from data import get_data_module, ALL_CHEXPERT_LABELS, parse_labels_config
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def save_results_to_csv(predictions: torch.Tensor, true_labels: torch.Tensor, pathologies: list, csv_path: str) -> None:
    """Save model predictions and true labels to CSV."""
    num_samples = predictions.shape[0]
    rows = []

    # Vectorized computation
    predictions_np = predictions.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()

    for sample_idx in range(num_samples):
        row = {'sample_id': sample_idx}

        for class_idx, pathology in enumerate(pathologies):
            prob = predictions_np[sample_idx, class_idx]
            true_label = int(true_labels_np[sample_idx, class_idx])

            row[f'{pathology}_prob'] = f"{prob:.6f}"
            row[f'{pathology}_true_label'] = true_label

        rows.append(row)

    # Use pandas for batch writing
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print(f"\n✓ Saved test results to: {csv_path}")


def test_inference(config_path: str = "config.yaml"):
    """
    Test inference: Run model on test set and save predictions + true labels.
    """
    print("\n" + "=" * 70)
    print("TEST INFERENCE: Model prediction on test set")
    print("=" * 70)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    eval_config = config.get('evaluation', {})
    output_config = config.get('output', {})
    data_config = config.get('data', {})
    working_dir = data_config.get('working_dir', os.getcwd())

    print("\n[1] Loading trained model...")
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

    print("\n[2] Loading test data...")
    data_module = get_data_module(config_path=config_path, num_workers=4)
    data_module.setup(stage='test')
    print(f"  ✓ Test dataloader ready")

    pathologies, _ = parse_labels_config(config)

    print("\n[3] Running inference on test set...")
    test_loader = data_module.test_dataloader()

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        # Wrap test_loader with tqdm for the progress bar
        for batch_idx, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Calculating", unit="batch"):
            # Send tensors to GPU/MPS
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            labels = labels.cpu()

            all_predictions.append(probs)
            all_true_labels.append(labels)

    # Concatenate results
    if len(all_predictions) > 0:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_true_labels = torch.cat(all_true_labels, dim=0)
        print(f"  ✓ Inference complete: {len(all_predictions)} samples")
    else:
        print("  ⚠ Warning: Test dataloader is empty")
        return

    # Save to CSV
    print("\n[4] Saving results...")
    results_csv = output_config.get('results_csv', 'results.csv')
    csv_path = os.path.join(working_dir, results_csv)
    save_results_to_csv(all_predictions, all_true_labels, pathologies, csv_path)

    # Compute and print AUC per label
    print("\n[5] AUC Results:")
    print("-" * 45)
    predictions_np = all_predictions.numpy()
    true_labels_np = all_true_labels.numpy()
    aucs = []
    for i, pathology in enumerate(pathologies):
        # Filter out uncertain (-1.0) labels for AUC computation
        mask = true_labels_np[:, i] >= 0
        y_true = true_labels_np[mask, i]
        y_pred = predictions_np[mask, i]
        try:
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            print(f"  {pathology:<35} {auc:.4f}  (n={mask.sum()})")
        except ValueError:
            print(f"  {pathology:<35} N/A (single class)")
    if aucs:
        mean_auc = sum(aucs) / len(aucs)
        print("-" * 45)
        print(f"  {'Mean AUC':<35} {mean_auc:.4f}")

    print("\n✅ Test inference complete!")


if __name__ == "__main__":
    print('=' * 80)
    print('TEST INFERENCE')
    print('=' * 80)
    test_inference(config_path="/gpfs0/tamyr/users/alonfi/XRay/config.yaml")
