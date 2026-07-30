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
from tqdm import tqdm


def _monitor_progress(manifest_file, total_files, initial_manifest_lines):
    """Monitor manifest file line count and show progress in real-time with tqdm."""
    start_time = time.time()
    prev_count = 0

    pbar = tqdm(total=total_files, desc="Downloading", unit="file", ncols=100,
                bar_format='{desc}: {n_fmt}/{total_fmt} [{bar}] {percentage:3.0f}% | {postfix}')

    try:
        while True:
            time.sleep(3)

            try:
                # Count lines in manifest (cheap I/O vs walking entire directory)
                if os.path.exists(manifest_file):
                    with open(manifest_file) as f:
                        current_lines = sum(1 for _ in f)
                else:
                    current_lines = 0

                newly_downloaded = max(0, current_lines - initial_manifest_lines)

                elapsed = time.time() - start_time
                elapsed_str = _format_time(elapsed)

                files_to_add = newly_downloaded - prev_count
                if files_to_add > 0:
                    pbar.update(files_to_add)

                if prev_count > 0 and elapsed > 0:
                    files_per_sec = newly_downloaded / elapsed
                    remaining = total_files - newly_downloaded
                    eta_secs = remaining / files_per_sec if files_per_sec > 0 else 0
                    eta_str = _format_time(eta_secs)
                else:
                    eta_str = "calculating..."

                postfix = f"Elapsed: {elapsed_str} | ETA: {eta_str}"
                pbar.set_postfix_str(postfix)

                prev_count = newly_downloaded

                if newly_downloaded >= total_files:
                    pbar.close()
                    break

            except Exception:
                pass
    finally:
        pbar.close()

    print()


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
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Verify manifest against disk (rescan directory to rebuild manifest)",
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
            print(f"Total files in selected subsets: {len(all_files)}\n")

        manifest_file = Path(OUTPUT_DIR) / ".downloaded_manifest"

        if args.verify:
            # Rebuild manifest from disk scan
            print("Verifying: scanning disk to rebuild manifest...")
            existing_files = set()
            output_path = Path(OUTPUT_DIR)
            if output_path.exists():
                for fp in output_path.rglob("*"):
                    if fp.is_file():
                        rel_path_full = fp.relative_to(output_path).as_posix()
                        if "/files/" in rel_path_full:
                            rel_path = rel_path_full[rel_path_full.index("/files/") + 1:]
                        else:
                            rel_path = rel_path_full
                        existing_files.add(rel_path)
            manifest_file.write_text('\n'.join(sorted(existing_files)))
            print(f"Rebuilt manifest with {len(existing_files)} files")

        if SKIP_EXISTING:
            # Load manifest (deduplicated)
            existing_files = set()
            if manifest_file.exists():
                print("Loading manifest...")
                existing_files = set(line for line in manifest_file.read_text().splitlines() if line)
                print(f"Manifest: {len(existing_files)} files already downloaded")

            # Filter to only pending files
            pending_files = [f for f in all_files if f not in existing_files]

            if pending_files:
                print(f"Total files: {len(all_files)}, Already downloaded: {len(all_files) - len(pending_files)}, Pending: {len(pending_files)}")
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

        # Count current manifest lines for progress tracking
        initial_manifest_lines = 0
        if manifest_file.exists():
            with open(manifest_file) as f:
                initial_manifest_lines = sum(1 for _ in f)

        # Track download progress in background
        progress_thread = threading.Thread(
            target=_monitor_progress,
            args=(str(manifest_file), total_files, initial_manifest_lines),
            daemon=True,
        )
        progress_thread.start()

        # Step 2: Download files in parallel
        # Each file appends to manifest ONLY on successful wget (exit code 0)
        print(f"Downloading files (parallel with {PARALLEL_JOBS} jobs)...\n")
        cmd_download = (
            f"cat {list_file} | xargs -P {PARALLEL_JOBS} -I {{}} sh -c '"
            f"wget --quiet "
            f"--user {PHYSIONET_USERNAME} "
            f"--password {PHYSIONET_PASSWORD} "
            f"--continue "
            f"--cut-dirs=3 "
            f"-P {OUTPUT_DIR} "
            f"{BASE_URL}{{}} 2>/dev/null "
            f"&& echo {{}} >> {manifest_file}"
            f"'"
        )

        result = subprocess.run(cmd_download, shell=True)

        # Wait for final progress update
        time.sleep(3)

        if result.returncode == 0:
            print("\n✓ Download completed successfully!")
        else:
            print(f"\n⚠ Some downloads may have failed (return code: {result.returncode})")
            print("Re-run the script to retry failed files.")

    finally:
        # Clean up temp file (but keep cache)
        if list_file and os.path.exists(list_file) and list_file != str(FILE_LIST_CACHE):
            os.remove(list_file)


if __name__ == "__main__":
    main()
