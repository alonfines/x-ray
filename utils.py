"""Shared utilities for per-label binary model scripts."""
import copy
import os
from pathlib import Path

import yaml

from data import ALL_CHEXPERT_LABELS

ALL_LABELS = ["No Finding"] + list(ALL_CHEXPERT_LABELS)

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_CONFIG = PROJECT_ROOT / "config.yaml"
CONFIGS_DIR = PROJECT_ROOT / "configs" / "per_label"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "mimic" / "per_label"
OUTPUT_BASE = PROJECT_ROOT / "conformal_calibration" / "mimic" / "per_label"
OUTPUT_BASE_WITH_VAL = PROJECT_ROOT / "conformal_calibration" / "mimic" / "per_label_with_val"
PRETRAINED_CKPT = PROJECT_ROOT / "checkpoints" / "mimic" / "densenet-alllabels-bce-epoch=60-val_auroc_mean=0.784.ckpt"

# Hierarchy-corrected pathway (separate directories to avoid overwriting)
CONFIGS_DIR_HIER = PROJECT_ROOT / "configs" / "per_label_hier"
CHECKPOINT_DIR_HIER = PROJECT_ROOT / "checkpoints" / "mimic" / "per_label_hier"
OUTPUT_BASE_HIER = PROJECT_ROOT / "conformal_calibration" / "mimic" / "per_label_hier"
OUTPUT_BASE_WITH_VAL_HIER = PROJECT_ROOT / "conformal_calibration" / "mimic" / "per_label_hier_with_val"


def safe_name(label: str) -> str:
    return label.replace(" ", "_").lower()


def find_checkpoint(label: str, hierarchy: bool = False) -> Path | None:
    """Find the most recent checkpoint file for a given label."""
    checkpoint_dir = CHECKPOINT_DIR_HIER if hierarchy else CHECKPOINT_DIR
    matches = list(checkpoint_dir.glob(f"densenet-{safe_name(label)}-*-epoch=*.ckpt"))
    if not matches:
        return None
    return max(matches, key=os.path.getctime)


def generate_config(label: str, base_config: dict, hierarchy: bool = False) -> Path:
    """Create a per-label config file. Returns the config path."""
    cfg = copy.deepcopy(base_config)
    name = safe_name(label)

    cfg["labels"] = [label]
    cfg["training"]["uncertain_strategy"] = "u_zeros"
    cfg["training"].pop("task_weights", None)

    cfg["loss"] = {
        "type": "bce",
        "pretrained_checkpoint": str(PRETRAINED_CKPT),
        "reinit_classifier": True,
    }

    cfg["optimizer"]["lr"] = 0.001
    cfg["trainer"]["max_epochs"] = 60
    for cb in cfg.get("trainer", {}).get("callbacks", []):
        if "EarlyStopping" in cb.get("class_path", ""):
            cb["init_args"]["patience"] = 15

    logger_args = cfg.setdefault("trainer", {}).setdefault("logger", {}).setdefault("init_args", {})
    hier_tag = "_hier" if hierarchy else ""
    logger_args["name"] = f"binary_{name}{hier_tag}_auc"
    tags = ["binary", name, cfg["loss"]["type"]]
    if hierarchy:
        tags.append("hierarchy")
    logger_args["tags"] = tags

    checkpoint_dir = CHECKPOINT_DIR_HIER if hierarchy else CHECKPOINT_DIR
    cfg["chkpt_callback"]["dirpath"] = str(checkpoint_dir)
    cfg["hierarchy_correction"] = {"enabled": hierarchy}

    configs_dir = CONFIGS_DIR_HIER if hierarchy else CONFIGS_DIR
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"{name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return config_path


def parse_labels(args_label: str | None) -> list[str]:
    """Validate --label arg and return list of labels to process."""
    if args_label:
        if args_label not in ALL_LABELS:
            raise ValueError(f"Unknown label: {args_label}\nAvailable: {ALL_LABELS}")
        return [args_label]
    return ALL_LABELS
