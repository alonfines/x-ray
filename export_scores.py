#!/usr/bin/env python
"""
Export model prediction scores (probabilities + true labels) for conformal
and validation sets. Supports both the unified hier model and per-label models.

Outputs CSVs to results/outputs/mimic/scores/.
"""
import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from train import CXRClassifier
from data import get_data_module, parse_labels_config, ALL_CHEXPERT_LABELS
from utils import ALL_LABELS, find_checkpoint, safe_name, generate_config, BASE_CONFIG


SCORES_DIR = Path(__file__).resolve().parent / "results" / "outputs" / "mimic" / "scores"


def run_inference(model, dataloader, device):
    """Run inference, return (probs, labels) tensors."""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, unit="batch"):
            images = images.to(device)
            logits = model(images)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


def save_scores_csv(probs, labels, pathologies, csv_path):
    """Save probs + true labels to CSV."""
    rows = []
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    for i in range(len(probs_np)):
        row = {"sample_id": i}
        for j, p in enumerate(pathologies):
            row[f"{p}_prob"] = f"{probs_np[i, j]:.6f}"
            row[f"{p}_true_label"] = labels_np[i, j]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(df)} samples)")


def export_unified_hier():
    """Export conformal and validation scores for the unified hier model."""
    print("\n" + "=" * 70)
    print("UNIFIED HIER MODEL: Exporting conformal & validation scores")
    print("=" * 70)

    config_path = str(Path(__file__).resolve().parent / "config_hier_unified.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pathologies, _ = parse_labels_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    eval_config = config.get("evaluation", {})
    checkpoint_path = eval_config["checkpoint_path"]
    print(f"  Checkpoint: {Path(checkpoint_path).name}")
    print(f"  Device: {device}")

    model = CXRClassifier.load_from_checkpoint(checkpoint_path, strict=False, config_path=config_path)
    model = model.to(device)
    model.eval()

    # Load data
    data_module = get_data_module(config_path=config_path, num_workers=4)
    data_module.setup(stage="fit")  # fit stage gives us train, val, conformal

    output_dir = SCORES_DIR / "unified_hier"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Conformal set
    print("\n  Running inference on conformal set...")
    probs, labels = run_inference(model, data_module.conformal_dataloader(), device)
    save_scores_csv(probs, labels, pathologies, output_dir / "conformal_scores.csv")

    # Validation set
    print("  Running inference on validation set...")
    probs, labels = run_inference(model, data_module.val_dataloader(), device)
    save_scores_csv(probs, labels, pathologies, output_dir / "validation_scores.csv")


def export_per_label():
    """Export conformal and validation scores for all per-label models."""
    print("\n" + "=" * 70)
    print("PER-LABEL MODELS: Exporting conformal & validation scores")
    print("=" * 70)

    with open(BASE_CONFIG) as f:
        base_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # We'll collect all per-label predictions into combined DataFrames
    conformal_dfs = []
    validation_dfs = []

    for label in ALL_LABELS:
        checkpoint = find_checkpoint(label, hierarchy=False)
        if checkpoint is None:
            print(f"\n  [SKIP] No checkpoint for '{label}'")
            continue

        print(f"\n  Processing: {label} ({checkpoint.name})")
        config_path = generate_config(label, base_config, hierarchy=False)

        # Disable pretrained loading for inference
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["evaluation"]["checkpoint_path"] = str(checkpoint)
        cfg["loss"]["pretrained_checkpoint"] = None
        cfg["loss"]["reinit_classifier"] = False
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        model = CXRClassifier.load_from_checkpoint(
            checkpoint, strict=False, config_path=str(config_path)
        )
        model = model.to(device)
        model.eval()

        data_module = get_data_module(config_path=str(config_path), num_workers=4)
        data_module.setup(stage="fit")

        name = safe_name(label)

        # Conformal
        probs, labels_t = run_inference(model, data_module.conformal_dataloader(), device)
        probs_np = probs.squeeze(1).numpy()
        labels_np = labels_t.squeeze(1).numpy()
        conformal_dfs.append(pd.DataFrame({
            f"{label}_prob": [f"{p:.6f}" for p in probs_np],
            f"{label}_true_label": labels_np,
        }))

        # Validation
        probs, labels_t = run_inference(model, data_module.val_dataloader(), device)
        probs_np = probs.squeeze(1).numpy()
        labels_np = labels_t.squeeze(1).numpy()
        validation_dfs.append(pd.DataFrame({
            f"{label}_prob": [f"{p:.6f}" for p in probs_np],
            f"{label}_true_label": labels_np,
        }))

    output_dir = SCORES_DIR / "per_label"
    output_dir.mkdir(parents=True, exist_ok=True)

    if conformal_dfs:
        combined = pd.concat(conformal_dfs, axis=1)
        combined.insert(0, "sample_id", range(len(combined)))
        csv_path = output_dir / "conformal_scores.csv"
        combined.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path} ({len(combined)} samples)")

    if validation_dfs:
        combined = pd.concat(validation_dfs, axis=1)
        combined.insert(0, "sample_id", range(len(combined)))
        csv_path = output_dir / "validation_scores.csv"
        combined.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path} ({len(combined)} samples)")


if __name__ == "__main__":
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    export_unified_hier()
    export_per_label()
    print("\n" + "=" * 70)
    print("Done! All scores exported.")
    print("=" * 70)
