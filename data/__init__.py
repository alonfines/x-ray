"""Data package for chest X-ray classification.

Provides dataset and DataModule implementations for CheXpert and MIMIC-CXR-JPG.
"""

from typing import List, Tuple
from data.hierarchy import CHEXPERT_HIERARCHY, apply_hierarchical_inference

# All 13 CheXpert pathologies (excluding No Finding)
ALL_CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]
NO_FINDING_COL = "No Finding"
SUPPORT_DEVICES_COL = "Support Devices"


def parse_labels_config(config: dict) -> Tuple[List[str], bool]:
    """Parse the 'labels' field from config.

    Returns:
        (pathologies, use_all_labels): list of label names, and whether all labels mode is active.
    """
    labels = config.get('labels', 'all')
    if labels == 'all':
        return list(ALL_CHEXPERT_LABELS), True
    return list(labels), False


def get_data_module(config_path: str = "config.yaml", **kwargs):
    """Factory that returns the appropriate DataModule based on config['dataset']."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset = config.get('dataset', 'chexpert')
    if dataset == 'chexpert':
        from data.chexpert import CheXpertDataModule
        return CheXpertDataModule(config_path=config_path, **kwargs)
    elif dataset == 'mimic':
        from data.mimic import MIMICDataModule
        return MIMICDataModule(config_path=config_path, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'chexpert' or 'mimic'.")
