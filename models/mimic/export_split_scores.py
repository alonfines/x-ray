"""Export model predictions on train / validation / conformal splits to CSV."""
import os
from pathlib import Path

import torch
import yaml

from models.mimic.data import get_data_module, parse_labels_config
from models.mimic.inference import build_results_df, load_model, predict

AVAILABLE_SPLITS = ['train', 'validation', 'conformal']


def main(config_path: str = "models/mimic/configs/config.yaml", splits_to_export: list[str] | None = None):
    splits_to_export = splits_to_export or AVAILABLE_SPLITS

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    working_dir = config.get('data', {}).get('working_dir', os.getcwd())
    pathologies, _ = parse_labels_config(config)

    model, checkpoint_file, device = load_model(config_path)

    data_module = get_data_module(config_path=config_path, num_workers=4)
    data_module.setup()

    output_dir = Path(working_dir) / "data" / config.get('dataset', 'mimic') / "probs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train loader with shuffle=False so rows align with the dataset's ids
    train_loader = torch.utils.data.DataLoader(
        data_module.train_dataset, batch_size=data_module.batch_size,
        num_workers=4, shuffle=False, pin_memory=True)

    all_splits = {
        'train': (train_loader, data_module.train_dataset),
        'validation': (data_module.val_dataloader(), data_module.val_dataset),
        'conformal': (data_module.conformal_dataloader(), data_module.conformal_dataset),
    }

    for split_name in splits_to_export:
        dataloader, dataset = all_splits[split_name]
        print(f"\n--- {split_name} split ({len(dataset)} samples) ---")
        preds, labels = predict(model, dataloader, device)

        id_cols = [c for c in ['subject_id', 'study_id', 'dicom_id'] if c in dataset.df.columns]
        sample_ids = dataset.df[id_cols] if id_cols else None

        csv_path = output_dir / f"{checkpoint_file.stem}_{split_name}.csv"
        build_results_df(preds, labels, pathologies, sample_ids).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    print("\nDone!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/gpfs0/tamyr/users/alonfi/XRay/models/mimic/configs/config_hier_unified.yaml')
    parser.add_argument('--splits', nargs='+', choices=AVAILABLE_SPLITS, default=None,
                        help='Splits to export (default: all). E.g. --splits train validation')
    args = parser.parse_args()
    main(config_path=args.config, splits_to_export=args.splits)
