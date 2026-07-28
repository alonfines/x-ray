#!/usr/bin/env python3
"""
Download MIMIC-CXR dataset using wget with parallel downloads.
Credentials loaded from .env file or environment variables.
Includes real-time progress tracking with speed and ETA.
Supports downloading specific patient subsets (e.g., p10, p11, etc.)
"""
import subprocess
import os
import sys
import tempfile
import time
import threading
import argparse
from pathlib import Path


def _monitor_progress(output_dir, total_files):
    """Monitor download directory and show progress in real-time."""
    start_time = time.time()
    prev_size = 0
    prev_time = start_time
    prev_count = 0

    while True:
        time.sleep(5)  # Update every 5 seconds

        try:
            # Count files and total size
            file_count = 0
            total_size = 0
            for root, _, files in os.walk(output_dir):
                for f in files:
                    file_path = os.path.join(root, f)
                    if os.path.exists(file_path):
                        file_count += 1
                        total_size += os.path.getsize(file_path)

            if file_count == 0:
                continue

            elapsed = time.time() - start_time
            current_time = time.time()
            time_delta = current_time - prev_time

            # Calculate speed and ETA
            size_delta = total_size - prev_size
            if time_delta > 0:
                speed_mbps = (size_delta / (1024 * 1024)) / time_delta
            else:
                speed_mbps = 0

            count_delta = file_count - prev_count
            if count_delta > 0:
                remaining_files = total_files - file_count
                files_per_sec = count_delta / time_delta if time_delta > 0 else 0
                eta_secs = remaining_files / files_per_sec if files_per_sec > 0 else 0
                eta_str = _format_time(eta_secs)
            else:
                eta_str = "calculating..."

            # Format output
            percent = (file_count / total_files * 100) if total_files > 0 else 0
            size_gb = total_size / (1024**3)
            elapsed_str = _format_time(elapsed)

            progress_bar = "█" * int(percent / 2) + "░" * (50 - int(percent / 2))
            print(
                f"\r[{progress_bar}] {percent:5.1f}% | "
                f"Files: {file_count}/{total_files} | "
                f"Size: {size_gb:.2f}GB | "
                f"Speed: {speed_mbps:.1f}MB/s | "
                f"Elapsed: {elapsed_str} | "
                f"ETA: {eta_str}     ",
                end="",
                flush=True,
            )

            prev_size = total_size
            prev_time = current_time
            prev_count = file_count

            if file_count >= total_files:
                break

        except Exception:
            # Silently continue on errors (file may be in use, etc.)
            pass


def _format_time(seconds):
    """Format seconds to HH:MM:SS."""
    if seconds < 0 or seconds == float("inf"):
        return "unknown"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    """Main download function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Download MIMIC-CXR dataset with optional subset filtering"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Comma-separated list of patient subsets to download (e.g., p10,p11,p12). "
             "If not specified, downloads all files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/gpfs0/tamyr/projects/data/MIMIC-CXR",
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=1,
        help="Number of parallel download jobs",
    )
    parser.add_argument(
        "--skip-existing",
        type=bool,
        default=True,
        help="Skip files that already exist",
    )
    args = parser.parse_args()

    # Load .env file if it exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    if key:
                        os.environ[key] = value

    # Configuration
    PHYSIONET_USERNAME = os.getenv("PHYSIONET_USERNAME", "").strip()
    PHYSIONET_PASSWORD = os.getenv("PHYSIONET_PASSWORD", "").strip()
    BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
    OUTPUT_DIR = args.output_dir
    PARALLEL_JOBS = args.parallel_jobs
    CACHE_DIR = Path(__file__).parent / ".download_cache"
    FILE_LIST_CACHE = CACHE_DIR / "filelist.txt"
    SKIP_EXISTING = args.skip_existing
    SUBSETS = set(args.subset.split(",")) if args.subset else None

    # Validate credentials
    if not PHYSIONET_USERNAME or not PHYSIONET_PASSWORD:
        print("Error: PHYSIONET_USERNAME and PHYSIONET_PASSWORD environment variables required")
        sys.exit(1)

    # Create output directory and cache directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    print(f"Starting MIMIC-CXR dataset download to {OUTPUT_DIR}...")
    print(f"Username: {PHYSIONET_USERNAME}")
    print(f"Skip existing files: {SKIP_EXISTING}")
    if SUBSETS:
        print(f"Download mode: Subset (subsets: {', '.join(sorted(SUBSETS))})")
    else:
        print("Download mode: Full dataset")

    list_file = None
    try:
        # Step 1: Download or use cached IMAGE_FILENAMES index
        print("\nFetching file list...")

        # Check if file list is cached
        if FILE_LIST_CACHE.exists():
            print(f"Using cached file list from {FILE_LIST_CACHE}")
            list_file = str(FILE_LIST_CACHE)
        else:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                list_file = f.name

            # Download the file list
            cmd_list = f"""
            wget --quiet --user {PHYSIONET_USERNAME} --password {PHYSIONET_PASSWORD} \
              -O {list_file} {BASE_URL}IMAGE_FILENAMES
            """
            result = subprocess.run(cmd_list, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"✗ Failed to download file list: {result.stderr}")
                sys.exit(1)

            # Cache the file list
            import shutil
            shutil.copy(list_file, FILE_LIST_CACHE)
            print(f"Cached file list to {FILE_LIST_CACHE}")

        # Count total files and filter out already-downloaded ones
        all_files = []
        with open(list_file) as f:
            all_files = [line.strip() for line in f if line.strip()]

        # Extract just filenames for fast matching against flat-downloaded files
        filenames_only = {}  # maps "filename.jpg" -> full_path
        for file_path in all_files:
            filename = file_path.split("/")[-1]  # Extract just the filename
            filenames_only[filename] = file_path

        # Filter by subset if specified
        if SUBSETS:
            filtered_files = []
            for file_path in all_files:
                # Extract subset from path (e.g., "files/p10/..." -> "p10")
                parts = file_path.split("/")
                if len(parts) > 1 and parts[1] in SUBSETS:
                    filtered_files.append(file_path)
            all_files = filtered_files
            print(f"Filtered to subsets: {', '.join(sorted(SUBSETS))}")
            print(f"Total files in selected subsets: {len(all_files)}")
            if all_files:
                print(f"Sample from filelist: {all_files[0]}\n")

        if SKIP_EXISTING:
            # Build set of existing files (faster than checking each file individually)
            print("Scanning existing files...")
            existing_files = set()
            manifest_file = Path(OUTPUT_DIR) / ".downloaded_manifest"

            # Try to load from manifest first (faster on restarts)
            if manifest_file.exists():
                print(f"Loading existing files from manifest...")
                try:
                    existing_files = set(manifest_file.read_text().splitlines())
                    print(f"Loaded {len(existing_files)} files from manifest")
                except Exception as e:
                    print(f"Warning: Failed to load manifest ({e}), rescanning...")
                    existing_files = set()

            # If manifest didn't work or doesn't exist, scan directory
            if not existing_files and os.path.exists(OUTPUT_DIR):
                print(f"Scanning {OUTPUT_DIR} for existing files...")
                output_path = Path(OUTPUT_DIR)
                # Use rglob for faster traversal (recursive glob is faster than os.walk)
                scan_count = 0
                for file_path in output_path.rglob("*"):
                    if file_path.is_file():
                        scan_count += 1
                        # Store relative path matching the format in file list
                        # File list has paths like: "files/p10/..."
                        # But wget creates: "mimic-cxr-jpg/2.1.0/files/p10/..."
                        # So we extract everything from "files/" onward
                        rel_path_full = file_path.relative_to(output_path).as_posix()

                        # Find "files/" in the path and use everything from there
                        if "/files/" in rel_path_full:
                            rel_path = rel_path_full[rel_path_full.index("/files/") + 1:]
                        else:
                            rel_path = rel_path_full

                        existing_files.add(rel_path)
                        if scan_count <= 3:  # Show first 3 files for debugging
                            print(f"  Sample full: {rel_path_full}")
                            print(f"  Sample normalized: {rel_path}")

                print(f"Found {len(existing_files)} existing files")

                # Save manifest for faster restarts
                if existing_files:
                    try:
                        manifest_file.write_text('\n'.join(sorted(existing_files)))
                    except Exception as e:
                        print(f"Warning: Failed to save manifest ({e})")

            # Filter to only pending files
            # Check both full path match and filename-only match (for old flat downloads)
            pending_files = []
            for f in all_files:
                filename = f.split("/")[-1]  # Extract just the filename
                # File is downloaded if either full path or just filename exists
                if f not in existing_files and filename not in existing_files:
                    pending_files.append(f)

            if pending_files:
                print(f"Total files: {len(all_files)}, Already downloaded: {len(all_files) - len(pending_files)}, Pending: {len(pending_files)}")
                # Write pending files to temp file for download
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                    list_file = f.name
                    for file_path in pending_files:
                        f.write(file_path + '\n')
            else:
                print(f"✓ All {len(all_files)} files already downloaded!")
                return

            total_files = len(pending_files)
        else:
            total_files = len(all_files)

        print(f"Total files to download: {total_files}\n")

        # Track download progress in background
        progress_thread = threading.Thread(
            target=_monitor_progress,
            args=(OUTPUT_DIR, total_files),
            daemon=True,
        )
        progress_thread.start()

        # Step 2: Download all files in parallel using wget + xargs
        print(f"Downloading files (parallel with {PARALLEL_JOBS} jobs)...\n")
        cmd_download = f"""
        cat {list_file} | xargs -P {PARALLEL_JOBS} -I {{}} wget \
          --progress=dot:giga \
          --user {PHYSIONET_USERNAME} \
          --password {PHYSIONET_PASSWORD} \
          --continue \
          -P {OUTPUT_DIR} \
          {BASE_URL}{{}} 2>&1
        """

        result = subprocess.run(cmd_download, shell=True)

        # Wait a bit for final progress update
        time.sleep(2)

        if result.returncode == 0:
            print("\n✓ Download completed successfully!")
            # Update manifest for next run
            if SKIP_EXISTING:
                manifest_file = Path(OUTPUT_DIR) / ".downloaded_manifest"
                try:
                    output_path = Path(OUTPUT_DIR)
                    new_files = set()
                    for file_path in output_path.rglob("*"):
                        if file_path.is_file():
                            rel_path_full = file_path.relative_to(output_path).as_posix()
                            # Normalize paths to match filelist format
                            if "/files/" in rel_path_full:
                                rel_path = rel_path_full[rel_path_full.index("/files/") + 1:]
                            else:
                                rel_path = rel_path_full
                            new_files.add(rel_path)
                    manifest_file.write_text('\n'.join(sorted(new_files)))
                    print(f"Updated manifest with {len(new_files)} total files")
                except Exception as e:
                    print(f"Warning: Failed to update manifest ({e})")
        else:
            print(f"\n✗ Download failed with return code: {result.returncode}")
            sys.exit(1)

    finally:
        # Clean up temp file (but keep cache)
        if list_file and os.path.exists(list_file) and list_file != str(FILE_LIST_CACHE):
            os.remove(list_file)


if __name__ == "__main__":
    main()
