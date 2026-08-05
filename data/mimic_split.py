"""Stratified split of MIMIC-CXR-JPG dataset into train/validation/conformal/test sets.

This script:
1. Reads metadata CSV and CheXpert-labeler CSV
2. Filters to frontal views (AP/PA) only
3. Filters to images that actually exist on disk
4. Joins metadata with labels on (subject_id, study_id)
5. Splits into train/validation/conformal/test with stratification
6. Filters uncertain (-1.0) labels from train/validation/conformal (test keeps them)
7. Saves four new CSV files
"""

import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from data import parse_labels_config
from data.split_utils import get_stratification_target, print_split_statistics


def find_existing_images(files_dir):
    """Scan the files directory and return a set of existing dicom_ids."""
    existing = set()
    for p_group in os.listdir(files_dir):
        p_group_path = os.path.join(files_dir, p_group)
        if not os.path.isdir(p_group_path) or not p_group.startswith('p'):
            continue
        for patient in os.listdir(p_group_path):
            patient_path = os.path.join(p_group_path, patient)
            if not os.path.isdir(patient_path):
                continue
            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue
                for fname in os.listdir(study_path):
                    if fname.endswith('.jpg'):
                        existing.add(fname[:-4])  # strip .jpg
    return existing


def split_dataset():
    """Main splitting function for MIMIC-CXR-JPG."""
    config_path = "/gpfs0/tamyr/users/alonfi/XRay/config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    mimic_config = config.get('mimic', {})
    data_config = config.get('data', {})
    working_dir = data_config.get('working_dir')

    files_dir = mimic_config.get('files_dir')
    metadata_csv = mimic_config.get('metadata_csv')
    chexpert_csv = mimic_config.get('chexpert_csv')
    views = mimic_config.get('views', ['AP', 'PA'])

    split_config = mimic_config.get('split_ratios', {})
    train_ratio = split_config.get('train', 0.70)
    valid_ratio = split_config.get('validation', 0.10)
    conformal_ratio = split_config.get('conformal', 0.10)
    test_ratio = split_config.get('test', 0.10)

    # Step 1: Read metadata and filter to frontal views
    print(f"Reading metadata from {metadata_csv}")
    meta_df = pd.read_csv(metadata_csv)
    print(f"  -> {len(meta_df)} total images in metadata")

    print(f"\nFiltering to views: {views}")
    meta_df = meta_df[meta_df['ViewPosition'].isin(views)].copy()
    print(f"  -> {len(meta_df)} frontal images")

    # Step 2: Scan for existing files on disk
    print(f"\nScanning {files_dir} for downloaded images...")
    existing_dicoms = find_existing_images(files_dir)
    print(f"  -> {len(existing_dicoms)} images found on disk")

    meta_df = meta_df[meta_df['dicom_id'].isin(existing_dicoms)].copy()
    print(f"  -> {len(meta_df)} frontal images available on disk")

    # Step 3: Read CheXpert labels and join
    print(f"\nReading CheXpert labels from {chexpert_csv}")
    labels_df = pd.read_csv(chexpert_csv)
    print(f"  -> {len(labels_df)} study-level label records")

    # Join metadata (image-level) with labels (study-level) on (subject_id, study_id)
    merged_df = meta_df.merge(labels_df, on=['subject_id', 'study_id'], how='inner')
    print(f"  -> {len(merged_df)} images after joining with labels")

    pathologies, _ = parse_labels_config(config)
    print(f"\nUsing {len(pathologies)} labels: {pathologies}")

    # Step 4: Stratified splitting
    strat_target = get_stratification_target(merged_df, pathologies)

    if strat_target is not None:
        print(f"\nStratification using top pathologies (Mondrian-style):")
        print(f"  Stratification groups: {strat_target.nunique()}")
        print(f"  Samples per group: min={strat_target.value_counts().min()}, max={strat_target.value_counts().max()}")
    else:
        print("\nNo stratification applied (insufficient unique groups)")

    # Split: train vs (val+conformal+test)
    print(f"\nStep 1: train {train_ratio:.0%} vs rest {(valid_ratio+conformal_ratio+test_ratio):.0%}")
    train_temp, remaining_df = train_test_split(
        merged_df,
        test_size=(valid_ratio + conformal_ratio + test_ratio),
        stratify=strat_target if strat_target is not None else None,
        random_state=42
    )
    print(f"  -> Train (temp): {len(train_temp)}, Remaining: {len(remaining_df)}")

    # Split remaining: (val+conformal) vs test
    strat_remaining = get_stratification_target(remaining_df, pathologies)
    val_conf_df, test_df_final = train_test_split(
        remaining_df,
        test_size=test_ratio / (valid_ratio + conformal_ratio + test_ratio),
        stratify=strat_remaining if strat_remaining is not None else None,
        random_state=42
    )
    print(f"  -> Val+Conformal: {len(val_conf_df)}, Test: {len(test_df_final)}")

    # Split val+conformal: validation vs conformal
    strat_val_conf = get_stratification_target(val_conf_df, pathologies)
    valid_temp, conformal_temp = train_test_split(
        val_conf_df,
        test_size=conformal_ratio / (valid_ratio + conformal_ratio),
        stratify=strat_val_conf if strat_val_conf is not None else None,
        random_state=42
    )
    print(f"  -> Validation: {len(valid_temp)}, Conformal: {len(conformal_temp)}")

    # Step 5: Filter uncertain labels from train/validation/conformal
    print(f"\nFiltering uncertain (-1.0) labels from train/validation/conformal sets...")
    train_df_final = train_temp[~(train_temp[pathologies] == -1).any(axis=1)].copy()
    valid_df_final = valid_temp[~(valid_temp[pathologies] == -1).any(axis=1)].copy()
    conformal_df = conformal_temp[~(conformal_temp[pathologies] == -1).any(axis=1)].copy()

    print(f"  -> Train: {len(train_temp)} -> {len(train_df_final)} (removed {len(train_temp) - len(train_df_final)})")
    print(f"  -> Validation: {len(valid_temp)} -> {len(valid_df_final)} (removed {len(valid_temp) - len(valid_df_final)})")
    print(f"  -> Conformal: {len(conformal_temp)} -> {len(conformal_df)} (removed {len(conformal_temp) - len(conformal_df)})")
    print(f"  -> Test: {len(test_df_final)} (uncertain labels preserved)")

    # Step 6: Save split CSVs
    output_dir = working_dir
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, mimic_config.get('train_split_csv', 'csv_files/mimic_train_split.csv'))
    valid_path = os.path.join(output_dir, mimic_config.get('valid_split_csv', 'csv_files/mimic_valid_split.csv'))
    conformal_path = os.path.join(output_dir, mimic_config.get('conformal_split_csv', 'csv_files/mimic_conformal_split.csv'))
    test_path = os.path.join(output_dir, mimic_config.get('test_split_csv', 'csv_files/mimic_test_split.csv'))

    # Ensure output directory exists
    for path in [train_path, valid_path, conformal_path, test_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f"\nSaving split CSVs:")
    train_df_final.to_csv(train_path, index=False)
    print(f"  -> {train_path}")

    valid_df_final.to_csv(valid_path, index=False)
    print(f"  -> {valid_path}")

    conformal_df.to_csv(conformal_path, index=False)
    print(f"  -> {conformal_path}")

    test_df_final.to_csv(test_path, index=False)
    print(f"  -> {test_path}")

    splits = [
        ("Train", train_df_final),
        ("Validation", valid_df_final),
        ("Conformal", conformal_df),
        ("Test", test_df_final)
    ]
    print_split_statistics(splits, pathologies)


if __name__ == '__main__':
    split_dataset()
