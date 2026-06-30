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
    Create a Mondrian-style stratification target for conformal prediction.
    Stratifies by presence/absence of each pathology to ensure balanced class distribution.
    """
    labels = df[pathologies].fillna(0).astype(int)

    # Create stratification based on presence of top 3 most prevalent pathologies
    # This ensures each individual class is reasonably balanced across splits
    top_n = min(3, len(pathologies))

    # Calculate prevalence of each pathology
    prevalence = labels.sum(axis=0)
    top_pathologies_idx = prevalence.nlargest(top_n).index.tolist()

    # Create a binary string representing which top pathologies are positive
    # Example: "101" means 1st and 3rd top pathologies are positive, 2nd is not
    strat_groups = []
    for idx in range(len(df)):
        group_str = ""
        for path_idx in top_pathologies_idx:
            group_str += "1" if labels.iloc[idx][path_idx] > 0 else "0"
        strat_groups.append(group_str)

    strat_target = pd.Series(strat_groups, index=df.index)

    # Check if we have enough samples per group
    value_counts = strat_target.value_counts()
    min_samples = value_counts.min()

    # If any group has only 1 sample, fall back to coarser stratification
    if min_samples < 2 and len(value_counts) > 1:
        # Stratify by presence of single most prevalent pathology
        most_prevalent_idx = prevalence.idxmax()
        strat_target = (labels[most_prevalent_idx] > 0).astype(str)

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

    # Get split ratios
    split_config = data_config.get('split_ratios', {})
    train_ratio = split_config.get('train', 0.70)
    valid_ratio = split_config.get('validation', 0.10)
    conformal_ratio = split_config.get('conformal', 0.10)
    test_ratio = split_config.get('test', 0.10)

    # Read raw CSVs
    train_csv_path = os.path.join(chexpert_dir, data_config.get('raw_train_csv', 'train.csv'))
    valid_csv_path = os.path.join(chexpert_dir, data_config.get('raw_valid_csv', 'valid.csv'))

    print(f"Reading train.csv from {train_csv_path}")
    train_df = pd.read_csv(train_csv_path)
    print(f"  -> {len(train_df)} samples")

    print(f"Reading valid.csv from {valid_csv_path}")
    valid_df = pd.read_csv(valid_csv_path)
    print(f"  -> {len(valid_df)} samples")

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

    # Combine both CSVs for unified splitting
    print(f"\nCombining train.csv and valid.csv for unified split...")
    combined_df = pd.concat([train_df, valid_df], ignore_index=True)
    print(f"  -> Combined total: {len(combined_df)} samples")

    # Get stratification target from combined data
    strat_target_main = get_stratification_target(combined_df, pathologies)

    if strat_target_main is not None:
        print(f"\nStratification using top pathologies (Mondrian-style):")
        print(f"  Stratification groups: {strat_target_main.nunique()}")
        print(f"  Samples per group: min={strat_target_main.value_counts().min()}, max={strat_target_main.value_counts().max()}")
    else:
        print("\nNo stratification applied (insufficient unique groups)")

    # Step 1: Split combined data into train vs (val+conformal+test)
    print(f"\nStep 1: Splitting combined data into train {train_ratio:.0%} and (val+conformal+test) {(valid_ratio+conformal_ratio+test_ratio):.0%}")
    train_temp, remaining_df = train_test_split(
        combined_df,
        test_size=(valid_ratio + conformal_ratio + test_ratio),
        stratify=strat_target_main if strat_target_main is not None else None,
        random_state=42
    )
    print(f"  -> Train (temp): {len(train_temp)} samples")
    print(f"  -> (Val+Conformal+Test): {len(remaining_df)} samples")

    # Step 2: Split remaining into (val+conformal) and test
    print(f"\nStep 2: Splitting (val+conformal+test) into (val+conformal) {(valid_ratio+conformal_ratio)/(valid_ratio+conformal_ratio+test_ratio):.0%} and test {test_ratio/(valid_ratio+conformal_ratio+test_ratio):.0%}")
    strat_target_remaining = get_stratification_target(remaining_df, pathologies)
    val_conf_df, test_df_final = train_test_split(
        remaining_df,
        test_size=test_ratio / (valid_ratio + conformal_ratio + test_ratio),
        stratify=strat_target_remaining if strat_target_remaining is not None else None,
        random_state=42
    )
    print(f"  -> (Val+Conformal): {len(val_conf_df)} samples")
    print(f"  -> Test: {len(test_df_final)} samples")

    # Step 3: Split val+conformal into validation and conformal
    print(f"\nStep 3: Splitting (val+conformal) into validation {valid_ratio/(valid_ratio+conformal_ratio):.0%} and conformal {conformal_ratio/(valid_ratio+conformal_ratio):.0%}")
    strat_target_val_conf = get_stratification_target(val_conf_df, pathologies)
    valid_temp, conformal_temp = train_test_split(
        val_conf_df,
        test_size=conformal_ratio / (valid_ratio + conformal_ratio),
        stratify=strat_target_val_conf if strat_target_val_conf is not None else None,
        random_state=42
    )
    print(f"  -> Validation (temp): {len(valid_temp)} samples")
    print(f"  -> Conformal (temp): {len(conformal_temp)} samples")

    # Step 4: Filter out uncertain labels from train/validation/conformal (keep them only in test)
    print(f"\nStep 4: Filtering uncertain (-1.0) labels from train/validation/conformal sets...")
    train_df_final = train_temp[~(train_temp[pathologies] == -1).any(axis=1)].copy()
    valid_df_final = valid_temp[~(valid_temp[pathologies] == -1).any(axis=1)].copy()
    conformal_df = conformal_temp[~(conformal_temp[pathologies] == -1).any(axis=1)].copy()

    print(f"  -> Train: {len(train_temp)} -> {len(train_df_final)} samples (removed {len(train_temp) - len(train_df_final)})")
    print(f"  -> Validation: {len(valid_temp)} -> {len(valid_df_final)} samples (removed {len(valid_temp) - len(valid_df_final)})")
    print(f"  -> Conformal: {len(conformal_temp)} -> {len(conformal_df)} samples (removed {len(conformal_temp) - len(conformal_df)})")
    print(f"  -> Test: {len(test_df_final)} samples (uncertain labels preserved)")

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

    test_df_final.to_csv(test_split_path, index=False)
    print(f"  ✓ {test_split_path}")

    # Print label distribution
    print(f"\nLabel distribution across splits:")
    splits_to_print = [
        ("Train", train_df_final),
        ("Validation", valid_df_final),
        ("Conformal", conformal_df),
        ("Test", test_df_final)
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

    # Verify conformal set has sufficient samples per class for per-class quantile estimation
    print(f"\n" + "=" * 70)
    print("CONFORMAL CALIBRATION READINESS CHECK")
    print("=" * 70)
    print(f"\nFor per-class conformal prediction, recommend ~100+ positive samples per pathology.")
    print(f"Your conformal set has {len(conformal_df)} total samples.\n")

    min_positives = float('inf')
    for pathology in pathologies:
        if pathology in conformal_df.columns:
            positives = (conformal_df[pathology] == 1.0).sum()
            min_positives = min(min_positives, positives)
            status = "✓ OK" if positives >= 100 else "⚠ LOW" if positives >= 50 else "✗ TOO FEW"
            print(f"  {pathology:<30} {positives:>4} positives {status}")

    if min_positives < 50:
        print(f"\n⚠ WARNING: Conformal set has insufficient samples for stable per-class quantiles.")
        print(f"           Consider increasing conformal_split_size or using Mondrian conformal.")
    elif min_positives < 100:
        print(f"\n⚠ NOTE: Some classes are below 100 positives. Results may be borderline.")
    else:
        print(f"\n✓ Conformal set appears adequate for per-class calibration.")

    # Final statistics summary
    print(f"\n" + "=" * 70)
    print("FINAL SPLIT STATISTICS")
    print("=" * 70)
    print(f"Train set:       {len(train_df_final):>6} samples")
    print(f"Validation set:  {len(valid_df_final):>6} samples")
    print(f"Conformal set:   {len(conformal_df):>6} samples")
    print(f"Test set:        {len(test_df_final):>6} samples")
    print(f"{'-'*70}")
    print(f"Total:           {len(train_df_final) + len(valid_df_final) + len(conformal_df) + len(test_df_final):>6} samples")
    print("=" * 70)

    # Calculate pos_weights from training set for balanced loss
    print(f"\n" + "=" * 70)
    print("CALCULATED POS_WEIGHTS FOR config.yaml")
    print("=" * 70)
    print(f"\npos_weight = num_negatives / num_positives per class\n")

    pos_weights = []
    for pathology in pathologies:
        if pathology in train_df_final.columns:
            num_positives = (train_df_final[pathology] == 1.0).sum()
            num_negatives = (train_df_final[pathology] == 0.0).sum()
            pos_weight = num_negatives / (num_positives + 1e-8)
            prevalence = 100 * num_positives / len(train_df_final)
            pos_weights.append(pos_weight)
            print(f"  {pathology:<30} pos_weight: {pos_weight:.3f}  ({num_positives:>5} pos, {prevalence:>5.1f}% prevalence)")

    # Normalize to rarest class (max pos_weight = 1.0)
    if pos_weights:
        max_weight = max(pos_weights)
        normalized_weights = [w / max_weight for w in pos_weights]
        print(f"\nNormalized pos_weights (rarest=1.0):")
        print(f"task_weights:")
        for pathology, weight in zip(pathologies, normalized_weights):
            print(f"  - {weight:.3f} # {pathology}")
        print("=" * 70)


if __name__ == '__main__':
    split_dataset()