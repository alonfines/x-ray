"""
Stratified split of CheXpert dataset into train/validation/conformal/test sets.

This script:
1. Reads raw train.csv and valid.csv
2. Splits train.csv into: train (80%), validation (20%), and a conformal set
3. Uses the raw valid.csv purely as the test set
4. Saves four new CSV files
"""

import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def get_stratification_target(df, pathologies):
    """
    Create a stratification target based on dominant pathologies.
    """
    labels = df[pathologies].fillna(0).astype(int)

    # Create a simple hash-based stratification group
    # This groups samples with similar pathology patterns together
    strat_target = (labels.sum(axis=1) % 3).astype(str)

    # Check if we have enough samples per group
    value_counts = strat_target.value_counts()
    min_samples = value_counts.min()

    # If any group has only 1 sample, use a coarser stratification
    if min_samples < 2 and len(value_counts) > 1:
        strat_target = (labels.iloc[:, 0].fillna(0) > 0).astype(str)

    # If still problematic, return None (no stratification)
    if strat_target.value_counts().min() < 2:
        return None

    return strat_target


def split_dataset():
    """Main splitting function."""
    config_path = "/gpfs0/tamyr/users/alonfi/XRay/config.yaml"

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config.get('data', {})
    working_dir = data_config.get('working_dir')
    chexpert_dir = data_config.get('chexpert_dir')
    conformal_size = data_config.get('conformal_split_size', 1)

    # Read raw CSVs
    train_csv_path = os.path.join(chexpert_dir, data_config.get('raw_train_csv', 'train.csv'))
    valid_csv_path = os.path.join(chexpert_dir, data_config.get('raw_valid_csv', 'valid.csv'))

    print(f"Reading train.csv from {train_csv_path}")
    train_df = pd.read_csv(train_csv_path)
    print(f"  -> {len(train_df)} samples")

    print(f"Reading valid.csv from {valid_csv_path}")
    test_df = pd.read_csv(valid_csv_path)
    print(f"  -> {len(test_df)} samples (This will be used purely as the TEST set)")

    # ==========================================
    # Get pathologies dynamically from config
    # ==========================================
    pathologies = config.get('use_labels', [])
    if not pathologies:
        print("Warning: 'use_labels' not found in config. Defaulting to all 14 CheXpert labels.")
        pathologies = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
        ]
    else:
        print(f"Using {len(pathologies)} labels specified in config.yaml.")

    # Get stratification target based ONLY on the chosen pathologies from train_df
    strat_target = get_stratification_target(train_df, pathologies)

    # Step 1: Extract conformal set (conformal_size samples) FROM TRAIN_DF
    if len(train_df) > conformal_size:
        print(f"\nStep 1: Extracting conformal set ({conformal_size} samples) from train.csv")
        train_temp_df, conformal_df = train_test_split(
            train_df,
            test_size=conformal_size,
            stratify=strat_target if strat_target is not None else None,
            random_state=42
        )
        print(f"  -> Conformal: {len(conformal_df)} samples")
        print(f"  -> Remaining Train Data: {len(train_temp_df)} samples")
    else:
        print(f"\nWarning: Total train samples ({len(train_df)}) <= conformal_size ({conformal_size})")
        print(f"  -> Conformal: {conformal_size} samples")
        conformal_df = train_df.iloc[:conformal_size]
        train_temp_df = train_df.iloc[conformal_size:]

    # Step 2: Split remaining into train (80%) / validation (20%)
    if len(train_temp_df) > 1:
        print(f"\nStep 2: Splitting remaining data into train (80%) / validation (20%)")
        strat_target_temp = get_stratification_target(train_temp_df, pathologies)
        train_df_final, valid_df_final = train_test_split(
            train_temp_df,
            test_size=0.2,
            stratify=strat_target_temp if strat_target_temp is not None else None,
            random_state=42
        )
        print(f"  -> Train: {len(train_df_final)} samples")
        print(f"  -> Validation: {len(valid_df_final)} samples")
    else:
        print(f"\nWarning: Not enough samples for train/val split")
        train_df_final = train_temp_df
        valid_df_final = pd.DataFrame(columns=train_temp_df.columns)

    # Save split CSVs
    output_dir = working_dir
    os.makedirs(output_dir, exist_ok=True)
    
    train_split_path = os.path.join(output_dir, data_config.get('train_split_csv', 'train_split.csv'))
    valid_split_path = os.path.join(output_dir, data_config.get('valid_split_csv', 'valid_split.csv'))
    conformal_split_path = os.path.join(output_dir, data_config.get('conformal_split_csv', 'conformal_split.csv'))
    test_split_path = os.path.join(output_dir, data_config.get('test_split_csv', 'test_split.csv'))

    print(f"\nSaving split CSVs:")
    train_df_final.to_csv(train_split_path, index=False)
    print(f"  ✓ {train_split_path}")

    valid_df_final.to_csv(valid_split_path, index=False)
    print(f"  ✓ {valid_split_path}")

    conformal_df.to_csv(conformal_split_path, index=False)
    print(f"  ✓ {conformal_split_path}")

    test_df.to_csv(test_split_path, index=False)
    print(f"  ✓ {test_split_path}")

    # Print label distribution
    print(f"\nLabel distribution across splits:")
    splits_to_print = [
        ("Train", train_df_final), 
        ("Validation", valid_df_final), 
        ("Conformal", conformal_df),
        ("Test", test_df)
    ]
    
    for name, df in splits_to_print:
        if len(df) > 0:
            print(f"\n{name}:")
            for pathology in pathologies:
                if pathology in df.columns:
                    positive = (df[pathology] == 1.0).sum()
                    uncertain = (df[pathology] == -1.0).sum()
                    negative = (df[pathology] == 0.0).sum()
                    print(f"  {pathology:<30} {positive:>5} positive, {uncertain:>5} uncertain, {negative:>5} negative")


if __name__ == '__main__':
    split_dataset()