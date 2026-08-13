"""Shared utilities for per-label binary model scripts."""
import copy
import os
from pathlib import Path

import torch
import yaml

from data import ALL_CHEXPERT_LABELS
from data.hierarchy import CHEXPERT_HIERARCHY

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
    cfg["conditional_training"] = {"enabled": False}
    cfg["hierarchical_inference"] = {"enabled": False}
    cfg["hierarchy_correction"] = {"enabled": hierarchy}

    configs_dir = CONFIGS_DIR_HIER if hierarchy else CONFIGS_DIR
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"{name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return config_path


def propagate_hierarchy_labels(labels: torch.Tensor, pathologies: list[str]) -> torch.Tensor:
    """Propagate positive labels upward through the CheXpert hierarchy.

    If a child label is positive (1), all its ancestors are set to positive.
    E.g., Edema=1 implies Lung Opacity=1; Pneumonia=1 implies Consolidation=1
    and Lung Opacity=1.

    This corrects NaN-as-0 misattributions where a parent label was "not
    mentioned" (NaN → 0) despite a positive child logically guaranteeing it.

    Args:
        labels: Tensor of shape (N, num_classes) with values {0, 1}.
        pathologies: List of label names matching the columns of labels.

    Returns:
        Corrected labels tensor (same shape, modified in-place on a clone).
    """
    label_to_idx = {name: i for i, name in enumerate(pathologies)}
    corrected = labels.clone()

    for child, parents in CHEXPERT_HIERARCHY.items():
        if child not in label_to_idx or not parents:
            continue
        child_idx = label_to_idx[child]
        child_positive = corrected[:, child_idx] == 1
        for parent in parents:
            if parent in label_to_idx:
                parent_idx = label_to_idx[parent]
                corrected[child_positive, parent_idx] = 1

    return corrected


def load_hierarchy_corrected_labels(
    csv_paths: list[str | Path], target_label: str,
    keep_uncertain: bool = False,
) -> torch.Tensor:
    """Load labels from CSV(s) with hierarchy propagation applied.

    Reads the full label columns from the CSV to propagate child→parent
    positives, then returns the corrected column for ``target_label``.

    Hierarchy propagation uses only definite labels (NaN → 0, -1 → 0) to
    determine which children are positive.  After propagation, if
    ``keep_uncertain`` is True the original -1 values for the *target*
    label are restored (useful for test evaluation where uncertain is a
    distinct category) — unless the target was corrected from 0 to 1 by
    propagation.

    Args:
        csv_paths: One or more split CSV paths (e.g. conformal, validation).
        target_label: The single label column to return after correction.
        keep_uncertain: If True, preserve -1 in the target column for
            samples that were not corrected by propagation.

    Returns:
        1-D tensor of corrected labels (length = total rows across CSVs).
    """
    import pandas as pd

    all_label_cols = list(CHEXPERT_HIERARCHY.keys())
    # Ensure target_label is included even if not in hierarchy (e.g. No Finding)
    if target_label not in all_label_cols:
        all_label_cols.append(target_label)

    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    # Remember original uncertain positions for the target label
    target_raw = df[target_label].values.astype(float)
    uncertain_mask = target_raw == -1.0

    # Build full label tensor: NaN → 0, -1 → 0
    label_matrix = []
    for col in all_label_cols:
        vals = df[col].fillna(0.0).values.astype(float)
        vals[vals == -1.0] = 0.0
        label_matrix.append(vals)

    labels_tensor = torch.tensor(label_matrix, dtype=torch.float32).T  # (N, C)
    corrected = propagate_hierarchy_labels(labels_tensor, all_label_cols)

    target_idx = all_label_cols.index(target_label)
    result = corrected[:, target_idx]

    if keep_uncertain:
        # Restore -1 for samples that were originally uncertain AND were not
        # corrected to 1 by hierarchy propagation
        still_zero = result == 0
        restore_mask = torch.tensor(uncertain_mask) & still_zero
        result[restore_mask] = -1.0

    return result


def parse_labels(args_label: str | None) -> list[str]:
    """Validate --label arg and return list of labels to process."""
    if args_label:
        if args_label not in ALL_LABELS:
            raise ValueError(f"Unknown label: {args_label}\nAvailable: {ALL_LABELS}")
        return [args_label]
    return ALL_LABELS
