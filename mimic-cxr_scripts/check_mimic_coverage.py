#!/usr/bin/env python3
"""
Check how many images in MIMIC-CXR match the chexpert CSV.
Maps subject_id/study_id to file paths and verifies they exist.
Also analyzes view distribution (PA, Lateral, AP, etc.).
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter

def main():
    """Check image coverage for CSV records."""
    csv_path = Path("/gpfs0/tamyr/projects/data/MIMIC-CXR/csv/mimic-cxr-2.0.0-chexpert.csv")
    images_dir = Path("/gpfs0/tamyr/projects/data/MIMIC-CXR/files")

    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total records in CSV: {len(df)}")

    # Extract unique subject_id/study_id pairs
    unique_records = df[['subject_id', 'study_id']].drop_duplicates()
    print(f"Unique subject_id/study_id pairs: {len(unique_records)}\n")

    # Check for image matches
    print("Checking image coverage...")
    found = 0
    not_found = 0
    image_counts = []

    for idx, row in tqdm(unique_records.iterrows(), total=len(unique_records), desc="Matching", unit="record", ncols=100):
        subject_id = int(row['subject_id'])
        study_id = int(row['study_id'])

        # Build expected path: files/p{subset}/p{subject_id}/s{study_id}/
        # Subject ID determines the p{10,11,12,etc} subset
        subset = f"p{str(subject_id)[:2]}"
        study_path = images_dir / subset / f"p{subject_id}" / f"s{study_id}"

        # Count images in this study
        if study_path.exists():
            images = list(study_path.glob("*.jpg"))
            if images:
                found += 1
                image_counts.append(len(images))
            else:
                not_found += 1
        else:
            not_found += 1

    total_images = sum(image_counts) if image_counts else 0

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"✓ Records with matching images: {found}")
    print(f"✗ Records without matching images: {not_found}")
    print(f"Coverage: {found}/{len(unique_records)} ({100*found/len(unique_records):.1f}%)")
    print(f"\nTotal images found: {total_images}")
    if image_counts:
        print(f"Avg images per study: {total_images/len(image_counts):.1f}")
        print(f"Min images per study: {min(image_counts)}")
        print(f"Max images per study: {max(image_counts)}")

    # Analyze view distribution
    print(f"\n{'='*60}")
    print(f"View Distribution:")
    print(f"{'='*60}")

    # Load metadata file which contains view information
    metadata_path = Path("/gpfs0/tamyr/projects/data/MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv.gz")
    if metadata_path.exists():
        print(f"Loading metadata from {metadata_path}...")
        metadata_df = pd.read_csv(metadata_path, compression='gzip')

        # Merge on subject_id and study_id
        merged = df.merge(metadata_df[['subject_id', 'study_id', 'ViewPosition']],
                         on=['subject_id', 'study_id'],
                         how='left')

        # Count view distribution
        view_counts = Counter(merged['ViewPosition'].dropna())
        total_with_view = len(merged['ViewPosition'].dropna())

        print(f"Total records in dataset: {len(merged)}")
        print(f"Records with view info: {total_with_view}")
        print()

        # Sort by count descending
        for view, count in sorted(view_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / total_with_view
            print(f"  {view:20s}: {count:7d} ({percentage:5.1f}%)")
    else:
        print(f"Metadata file not found at {metadata_path}")

if __name__ == "__main__":
    main()
