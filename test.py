import os
from pathlib import Path

import yaml
from sklearn.metrics import roc_auc_score

from data import get_data_module, parse_labels_config
from inference import build_results_df, load_model, predict


def print_aucs(predictions, true_labels, pathologies: list[str]) -> None:
    """Print per-label AUC, excluding uncertain (-1.0) labels."""
    print("-" * 45)
    aucs = []
    for i, pathology in enumerate(pathologies):
        mask = true_labels[:, i] >= 0
        try:
            auc = roc_auc_score(true_labels[mask, i], predictions[mask, i])
        except ValueError:
            print(f"  {pathology:<35} N/A (single class)")
            continue
        aucs.append(auc)
        print(f"  {pathology:<35} {auc:.4f}  (n={mask.sum()})")
    if aucs:
        print("-" * 45)
        print(f"  {'Mean AUC':<35} {sum(aucs) / len(aucs):.4f}")


def test_inference(config_path: str = "config.yaml"):
    """Run the model on the test split; save predictions + true labels and print per-label AUC."""
    print("\n" + "=" * 70)
    print("TEST INFERENCE: Model prediction on test set")
    print("=" * 70)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    working_dir = config.get('data', {}).get('working_dir', os.getcwd())
    pathologies, _ = parse_labels_config(config)

    print("\n[1] Loading trained model...")
    model, checkpoint_file, device = load_model(config_path)

    print("\n[2] Loading test data...")
    data_module = get_data_module(config_path=config_path, num_workers=4)
    data_module.setup(stage='test')

    # Row order matches the DataLoader (shuffle=False)
    test_df = data_module.test_dataset.df
    id_cols = [c for c in ['subject_id', 'study_id', 'dicom_id', 'Path'] if c in test_df.columns]
    sample_ids = test_df[id_cols] if id_cols else None

    print("\n[3] Running inference on test set...")
    predictions, true_labels = predict(model, data_module.test_dataloader(), device)
    print(f"  ✓ Inference complete: {len(predictions)} samples")

    print("\n[4] Saving results...")
    output_dir = Path(working_dir) / "results" / "outputs" / config.get('dataset', 'mimic')
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{checkpoint_file.stem}.csv"
    build_results_df(predictions, true_labels, pathologies, sample_ids).to_csv(csv_path, index=False)
    print(f"  ✓ Saved test results to: {csv_path}")

    print("\n[5] AUC Results:")
    print_aucs(predictions.numpy(), true_labels.numpy(), pathologies)

    print("\n✅ Test inference complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/gpfs0/tamyr/users/alonfi/XRay/config.yaml')
    args = parser.parse_args()
    test_inference(config_path=args.config)
