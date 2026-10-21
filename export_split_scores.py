"""Export model predictions on validation and conformal splits to CSV."""
import os
import yaml
import torch
import pandas as pd
from pathlib import Path
from train import CXRClassifier
from data import get_data_module, parse_labels_config
from tqdm import tqdm


def run_inference(model, dataloader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Inference", unit="batch"):
            logits = model(images.to(device))
            all_preds.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_preds), torch.cat(all_labels)


def build_csv(predictions, true_labels, pathologies, sample_ids):
    num_samples = predictions.shape[0]
    pred_np = predictions.numpy()
    label_np = true_labels.numpy()

    data = {}
    for i, p in enumerate(pathologies):
        data[f'{p}_prob'] = [f"{pred_np[j, i]:.6f}" for j in range(num_samples)]
        data[f'{p}_true_label'] = label_np[:, i].astype(int)

    pred_df = pd.DataFrame(data)
    if sample_ids is not None and len(sample_ids) == num_samples:
        return pd.concat([sample_ids.reset_index(drop=True), pred_df], axis=1)
    return pd.concat([pd.DataFrame({'sample_id': range(num_samples)}), pred_df], axis=1)


def main(config_path: str = "config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    eval_config = config.get('evaluation', {})
    data_config = config.get('data', {})
    working_dir = data_config.get('working_dir', os.getcwd())

    # Load checkpoint
    checkpoint_path = eval_config.get('checkpoint_path')
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
    else:
        checkpoint_dir = eval_config.get('checkpoint_dir', './checkpoints')
        checkpoint_files = list(Path(checkpoint_dir).glob('densenet-*.ckpt'))
        if not checkpoint_files:
            print(f"No checkpoints found in {checkpoint_dir}")
            return
        checkpoint_file = max(checkpoint_files, key=os.path.getctime)

    print(f"Checkpoint: {checkpoint_file.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = CXRClassifier.load_from_checkpoint(checkpoint_file, strict=False, config_path=config_path, _skip_pretrained=True)
    model = model.to(device)
    model.eval()

    data_module = get_data_module(config_path=config_path, num_workers=4)
    data_module.setup()

    pathologies, _ = parse_labels_config(config)
    dataset_name = config.get('dataset', 'mimic')
    output_dir = Path(working_dir) / "results" / "outputs" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        'validation': (data_module.val_dataloader(), data_module.val_dataset),
        'conformal': (data_module.conformal_dataloader(), data_module.conformal_dataset),
    }

    for split_name, (dataloader, dataset) in splits.items():
        print(f"\n--- {split_name} split ({len(dataset)} samples) ---")
        preds, labels = run_inference(model, dataloader, device)

        id_cols = [c for c in ['subject_id', 'study_id', 'dicom_id'] if c in dataset.df.columns]
        sample_ids = dataset.df[id_cols].reset_index(drop=True) if id_cols else None

        df = build_csv(preds, labels, pathologies, sample_ids)
        csv_path = output_dir / f"{checkpoint_file.stem}_{split_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    print("\nDone!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/gpfs0/tamyr/users/alonfi/XRay/config_hier_unified.yaml')
    args = parser.parse_args()
    main(config_path=args.config)
