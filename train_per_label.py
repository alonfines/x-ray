#!/usr/bin/env python3
"""
Train 14 single-label binary models sequentially, skipping labels
that already have a checkpoint.

Usage:
    python train_per_label.py
    python train_per_label.py --label 'Atelectasis'
    python train_per_label.py --hierarchy   # train with hierarchy label correction
"""
import argparse
import subprocess

import yaml

from utils import (
    BASE_CONFIG, PROJECT_ROOT,
    find_checkpoint, generate_config, parse_labels,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default=None,
                        help="Train a single label (e.g. 'Cardiomegaly'). "
                             "If omitted, trains all labels sequentially.")
    parser.add_argument("--hierarchy", action="store_true",
                        help="Enable hierarchy label correction during training.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing per-label checkpoints instead of "
                             "the all-labels backbone. Retrains all labels.")
    args = parser.parse_args()

    with open(BASE_CONFIG) as f:
        base_config = yaml.safe_load(f)

    labels = parse_labels(args.label)

    if args.hierarchy:
        print("\n[HIERARCHY] Training with hierarchy label correction enabled")
    if args.resume:
        print("[RESUME] Fine-tuning from existing per-label checkpoints")

    config_paths = []
    for label in labels:
        config_path = generate_config(label, base_config, hierarchy=args.hierarchy,
                                      resume_from_existing=args.resume)
        config_paths.append((label, config_path))
        print(f"  {label:<30} -> {config_path.relative_to(PROJECT_ROOT)}")

    if args.resume:
        # When resuming, retrain all labels (don't skip any)
        remaining = config_paths
        done = []
    else:
        done = [label for label, _ in config_paths
                if find_checkpoint(label, hierarchy=args.hierarchy)]
        remaining = [(label, cfg) for label, cfg in config_paths if label not in done]

        if len(done) == len(labels):
            print("\nAll labels already have checkpoints. Starting a fresh training round.")
            remaining = config_paths
        elif done:
            print(f"\nSkipping {len(done)} already-trained labels:")
            for label in done:
                print(f"  [DONE] {label}")

    print(f"\n{'=' * 70}")
    print(f"Training {len(remaining)}/{len(labels)} binary models...")
    print(f"{'=' * 70}\n")

    for i, (label, config_path) in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] Training: {label}")
        print("-" * 50)
        subprocess.run(
            ["python", str(PROJECT_ROOT / "train.py"), "--config", str(config_path)],
            cwd=str(PROJECT_ROOT),
        )


if __name__ == "__main__":
    main()
