#!/usr/bin/env python3
"""
Create filtered MIMIC-CXR CSV with only records that have matching images.
Apply No Finding logic: if No Finding=1.0, set all other labels to 0.0 (except Support Devices).
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

def main():
    """Create filtered CSV with existing images and corrected labels."""
    csv_path = Path("/gpfs0/tamyr/projects/data/MIMIC-CXR/csv/mimic-cxr-2.0.0-chexpert.csv")
    images_dir = Path("/gpfs0/tamyr/projects/data/MIMIC-CXR/files")
    output_csv = Path("/gpfs0/tamyr/users/alonfi/XRay/mimic_cxr_filtered.csv")

    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total records in CSV: {len(df)}\n")

    # Extract unique subject_id/study_id pairs with their records
    print("Checking image coverage...")
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering", unit="record", ncols=100):
        subject_id = int(row['subject_id'])
        study_id = int(row['study_id'])

        # Build expected path: files/p{subset}/p{subject_id}/s{study_id}/
        subset = f"p{str(subject_id)[:2]}"
        study_path = images_dir / subset / f"p{subject_id}" / f"s{study_id}"

        # Check if this study has images
        if study_path.exists():
            images = list(study_path.glob("*.jpg"))
            if images:
                valid_indices.append(idx)

    print(f"\nFiltered to {len(valid_indices)} records with matching images\n")

    # Keep only records with images
    df_filtered = df.iloc[valid_indices].copy().reset_index(drop=True)

    # Apply No Finding logic
    print("Applying No Finding label correction...")
    pathology_cols = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
        'Pneumonia', 'Pleural Effusion', 'Pleural Other', 'Pneumothorax'
    ]
    # Note: 'No Finding' and 'Support Devices' are excluded from the correction

    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Correcting", unit="record", ncols=100):
        # Check if No Finding = 1.0
        if pd.notna(row['No Finding']) and row['No Finding'] == 1.0:
            # Set all other pathology labels to 0.0, except Support Devices
            for col in pathology_cols:
                if col in df_filtered.columns:
                    df_filtered.at[idx, col] = 0.0

    print(f"\nSaved filtered CSV to {output_csv}")
    df_filtered.to_csv(output_csv, index=False)

    # Print summary stats
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"Total records with matching images: {len(df_filtered)}")
    print(f"\nLabel distribution (non-null counts):")
    for col in ['No Finding'] + pathology_cols + ['Support Devices']:
        if col in df_filtered.columns:
            count = df_filtered[col].notna().sum()
            pos_count = (df_filtered[col] == 1.0).sum()
            neg_count = (df_filtered[col] == 0.0).sum()
            print(f"  {col:30s}: {count:6d} (1.0: {pos_count:5d}, 0.0: {neg_count:5d})")

if __name__ == "__main__":
    main()
