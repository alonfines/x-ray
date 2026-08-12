#!/usr/bin/env python3
"""
Train 14 single-label binary models sequentially, skipping labels
that already have a checkpoint.

Submit as a single runai-bgu job:
    runai-bgu submit python -n per-label --cpu 4 --memory 16Gi --gpu-memory 16Gi \
        -- "cd /gpfs0/tamyr/users/alonfi/XRay && python train_per_label.py"
"""
import argparse
import copy
import subprocess
from pathlib import Path

import yaml

from data import ALL_CHEXPERT_LABELS

ALL_LABELS = ["No Finding"] + list(ALL_CHEXPERT_LABELS)

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_CONFIG = PROJECT_ROOT / "config.yaml"
CONFIGS_DIR = PROJECT_ROOT / "configs" / "per_label"


def label_has_checkpoint(label: str, chkpt_dir: Path) -> bool:
    """Check if a checkpoint file already exists for this label."""
    safe_name = label.replace(" ", "_").lower()
    # Checkpoint pattern: densenet-{label}-bce-epoch=XX-val_auroc_mean=X.XXX.ckpt
    return any(chkpt_dir.glob(f"densenet-{safe_name}-*-epoch=*.ckpt"))


PRETRAINED_CKPT = PROJECT_ROOT / "checkpoints" / "mimic" / "densenet-alllabels-bce-epoch=60-val_auroc_mean=0.784.ckpt"


def generate_config(label: str, base_config: dict) -> Path:
    """Create a per-label config file. Returns the config path."""
    cfg = copy.deepcopy(base_config)
    safe_name = label.replace(" ", "_").lower()

    cfg["labels"] = [label]
    cfg["training"]["uncertain_strategy"] = "u_zeros"
    cfg["training"].pop("task_weights", None)

    # Weighted BCE with pretrained all-labels backbone
    cfg["loss"] = {
        "type": "bce",
        "pretrained_checkpoint": str(PRETRAINED_CKPT),
        "reinit_classifier": True,
    }

    # Faster training: higher LR, fewer epochs, lower patience
    cfg["optimizer"]["lr"] = 0.001
    cfg["trainer"]["max_epochs"] = 60
    for cb in cfg.get("trainer", {}).get("callbacks", []):
        if "EarlyStopping" in cb.get("class_path", ""):
            cb["init_args"]["patience"] = 15

    # Wandb
    logger_args = cfg.setdefault("trainer", {}).setdefault("logger", {}).setdefault("init_args", {})
    logger_args["name"] = f"binary_{safe_name}_auc"
    logger_args["tags"] = ["binary", safe_name, cfg["loss"]["type"]]

    # Checkpoint dir
    cfg["chkpt_callback"]["dirpath"] = str(PROJECT_ROOT / "checkpoints" / "mimic" / "per_label")

    # Disable hierarchical/conditional (not applicable for single-label)
    cfg["conditional_training"] = {"enabled": False}
    cfg["hierarchical_inference"] = {"enabled": False}

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS_DIR / f"{safe_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return config_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default=None,
                        help="Train a single label (e.g. 'Cardiomegaly'). "
                             "If omitted, trains all labels sequentially.")
    args = parser.parse_args()

    with open(BASE_CONFIG) as f:
        base_config = yaml.safe_load(f)

    chkpt_dir = PROJECT_ROOT / "checkpoints" / "mimic" / "per_label"

    if args.label:
        if args.label not in ALL_LABELS:
            print(f"Unknown label: {args.label}")
            print(f"Available labels: {ALL_LABELS}")
            return
        labels = [args.label]
    else:
        labels = ALL_LABELS

    config_paths = []
    for label in labels:
        config_path = generate_config(label, base_config)
        config_paths.append((label, config_path))
        print(f"  {label:<30} -> {config_path.relative_to(PROJECT_ROOT)}")

    # Check which labels already have checkpoints
    done = [label for label, _ in config_paths if label_has_checkpoint(label, chkpt_dir)]
    remaining = [(label, cfg) for label, cfg in config_paths if label not in done]

    # All labels done — start a fresh round from the beginning
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
