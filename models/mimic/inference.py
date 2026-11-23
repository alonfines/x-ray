"""Shared inference helpers: checkpoint lookup, model loading, prediction, result tables."""
import os
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from models.mimic.train import CXRClassifier


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_checkpoint(config: dict) -> Path:
    """Return `evaluation.checkpoint_path`, else the newest densenet-*.ckpt in `evaluation.checkpoint_dir`."""
    eval_config = config.get('evaluation', {})
    checkpoint_path = eval_config.get('checkpoint_path')
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        return checkpoint_file

    checkpoint_dir = eval_config.get('checkpoint_dir', './checkpoints')
    checkpoint_files = list(Path(checkpoint_dir).glob('densenet-*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(checkpoint_files, key=os.path.getctime)


def load_model(config_path: str) -> tuple[CXRClassifier, Path, torch.device]:
    """Load the configured checkpoint in eval mode. Returns (model, checkpoint_file, device)."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    checkpoint_file = find_checkpoint(config)
    device = get_device()
    print(f"  ✓ Checkpoint: {checkpoint_file.name}  (device: {device})")

    model = CXRClassifier.load_from_checkpoint(
        checkpoint_file, strict=False, config_path=config_path, _skip_pretrained=True)
    model = model.to(device).eval()
    print(f"  ✓ Model loaded ({len(model.state_dict())} weights)")
    return model, checkpoint_file, device


@torch.no_grad()
def predict(model, dataloader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a dataloader; returns (sigmoid probabilities, labels)."""
    probs, labels = [], []
    for images, batch_labels in tqdm(dataloader, desc="Inference", unit="batch"):
        probs.append(torch.sigmoid(model(images.to(device))).cpu())
        labels.append(batch_labels.cpu())
    return torch.cat(probs), torch.cat(labels)


def build_results_df(predictions: torch.Tensor, true_labels: torch.Tensor, pathologies: list[str],
                     sample_ids: pd.DataFrame | None = None) -> pd.DataFrame:
    """One `{label}_prob` / `{label}_true_label` column pair per pathology, prefixed by the id columns."""
    pred_np = predictions.numpy()
    label_np = true_labels.numpy()

    data = {}
    for i, pathology in enumerate(pathologies):
        data[f'{pathology}_prob'] = [f"{p:.6f}" for p in pred_np[:, i]]
        data[f'{pathology}_true_label'] = label_np[:, i].astype(int)
    pred_df = pd.DataFrame(data)

    if sample_ids is None or len(sample_ids) != len(pred_df):
        sample_ids = pd.DataFrame({'sample_id': range(len(pred_df))})
    return pd.concat([sample_ids.reset_index(drop=True), pred_df], axis=1)
