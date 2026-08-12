#!/usr/bin/env python3
"""
Filter MIMIC-CXR CSV to only include PA and AP view records based on metadata.
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def main():
    """Filter to PA and AP view records."""
    filtered_csv = Path("/gpfs0/tamyr/users/alonfi/XRay/mimic_cxr_filtered.csv")
    metadata_gz = Path("/gpfs0/tamyr/projects/data/MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv.gz")
    output_csv = Path("/gpfs0/tamyr/users/alonfi/XRay/mimic_cxr_filtered.csv")

    print("Loading filtered CSV...")
    df = pd.read_csv(filtered_csv)
    print(f"Starting with {len(df)} records\n")

    print("Loading metadata to identify PA and AP views...")
    metadata = pd.read_csv(metadata_gz, compression='gzip')
    print(f"Loaded metadata with {len(metadata)} images\n")

    # Get PA and AP view records
    pa_ap_metadata = metadata[metadata['ViewPosition'].isin(['PA', 'AP'])]
    print(f"Found {len(pa_ap_metadata)} PA/AP view images in metadata\n")
    print(f"  PA views: {len(metadata[metadata['ViewPosition'] == 'PA'])}")
    print(f"  AP views: {len(metadata[metadata['ViewPosition'] == 'AP'])}\n")

    # Get unique (subject_id, study_id) pairs for PA/AP views
    pa_ap_pairs = set(zip(pa_ap_metadata['subject_id'], pa_ap_metadata['study_id']))
    print(f"Unique (subject_id, study_id) pairs for PA/AP views: {len(pa_ap_pairs)}\n")

    # Filter main CSV to only PA/AP records
    print("Filtering CSV to PA/AP records...")
    df_pa_ap = df[
        df.apply(
            lambda row: (row['subject_id'], row['study_id']) in pa_ap_pairs,
            axis=1
        )
    ].copy().reset_index(drop=True)

    print(f"Filtered to {len(df_pa_ap)} PA/AP records")
    print(f"Removed {len(df) - len(df_pa_ap)} non-PA/AP records\n")

    # Save
    df_pa_ap.to_csv(output_csv, index=False)
    print(f"✓ Saved to {output_csv}\n")

    # Print summary
    print(f"{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"Original records: {len(df)}")
    print(f"PA/AP records: {len(df_pa_ap)}")
    print(f"Retention rate: {100*len(df_pa_ap)/len(df):.1f}%")
    print(f"\nLabel distribution (PA/AP views):")

    pathology_cols = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
        'Pneumonia', 'Pleural Effusion', 'Pleural Other', 'Pneumothorax', 'Support Devices'
    ]

    for col in pathology_cols:
        if col in df_pa_ap.columns:
            count = df_pa_ap[col].notna().sum()
            pos_count = (df_pa_ap[col] == 1.0).sum()
            neg_count = (df_pa_ap[col] == 0.0).sum()
            print(f"  {col:30s}: {count:6d} (1.0: {pos_count:5d}, 0.0: {neg_count:5d})")

if __name__ == "__main__":
    main()
