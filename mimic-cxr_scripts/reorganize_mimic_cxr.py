#!/usr/bin/env python3
"""
Reorganize flat MIMIC-CXR downloads into proper directory structure.
Maps images based on the filenames list to match: files/p10/p10000001/s50414282/imagename.jpg
"""
import os
import sys
from pathlib import Path
from tqdm import tqdm

def main():
    """Reorganize MIMIC-CXR images from flat to directory structure."""
    output_dir = Path("/gpfs0/tamyr/projects/data/MIMIC-CXR")
    cache_dir = Path(__file__).parent / ".download_cache"
    file_list_cache = cache_dir / "filelist.txt"

    # Create a mapping of filename -> full path from the file list
    if not file_list_cache.exists():
        print(f"Error: File list cache not found at {file_list_cache}")
        print("Run download_mimic_cxr.py first to generate the file list cache")
        sys.exit(1)

    # Build mapping with progress bar
    print("Loading file mappings...")
    filename_to_path = {}
    with open(file_list_cache) as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Building index", unit="file", ncols=100):
        line = line.strip()
        if line:
            # Path is like: files/p10/p10000001/s50414282/imagename.jpg
            filename = line.split("/")[-1]
            if filename in filename_to_path:
                tqdm.write(f"Warning: Duplicate filename {filename} found in file list")
            filename_to_path[filename] = line

    print(f"Loaded {len(filename_to_path)} file mappings\n")

    # Find all flat image files in output directory
    print("Scanning for flat images...")
    flat_images = []
    for item in tqdm(output_dir.iterdir(), desc="Scanning", unit="file", ncols=100):
        if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg']:
            flat_images.append(item)

    print(f"Found {len(flat_images)} flat image files\n")

    if not flat_images:
        print("No images to reorganize!")
        return

    # Reorganize images
    moved = 0
    failed = 0

    for flat_file in tqdm(flat_images, desc="Reorganizing", unit="file", ncols=100):
        filename = flat_file.name

        if filename not in filename_to_path:
            tqdm.write(f"Warning: {filename} not found in file list, skipping")
            failed += 1
            continue

        # Get target path
        target_path_str = filename_to_path[filename]
        target_path = output_dir / target_path_str

        # Create parent directories
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Move file
        try:
            flat_file.rename(target_path)
            moved += 1
        except Exception as e:
            tqdm.write(f"Error moving {filename}: {e}")
            failed += 1

    print(f"\n✓ Successfully moved: {moved} files")
    print(f"✗ Failed/Skipped: {failed} files\n")

    # Update manifest to reflect new structure
    print("Updating manifest...")
    manifest_file = output_dir / ".downloaded_manifest"
    try:
        new_files = set()
        for file_path in tqdm(output_dir.rglob("*.jpg"), desc="Indexing", unit="file", ncols=100):
            if file_path.is_file():
                rel_path_full = file_path.relative_to(output_dir).as_posix()
                # Normalize to match filelist format (should already start with files/)
                if "/files/" in rel_path_full:
                    rel_path = rel_path_full[rel_path_full.index("/files/") + 1:]
                else:
                    rel_path = rel_path_full
                new_files.add(rel_path)

        manifest_file.write_text('\n'.join(sorted(new_files)))
        print(f"✓ Updated manifest with {len(new_files)} total files")
    except Exception as e:
        print(f"Warning: Failed to update manifest ({e})")

if __name__ == "__main__":
    main()
