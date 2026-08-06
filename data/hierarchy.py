"""Disease hierarchy for CheXpert/MIMIC-CXR pathology labels.

Based on Pham et al. (2020) "Interpreting chest X-rays via CNNs that exploit
hierarchical disease dependencies and uncertainty labels."

The hierarchy encodes parent-child relationships among the 14 CheXpert labels.
Root-level labels have no parents. Child labels are conditioned on their parents
being positive (e.g., Pneumonia requires Consolidation and Lung Opacity to be positive).
"""

import torch
import pandas as pd
from typing import List

# Maps each label to its ordered list of parents (root → child direction).
# Labels with no parents have an empty list.
CHEXPERT_HIERARCHY = {
    "Enlarged Cardiomediastinum": [],
    "Cardiomegaly": ["Enlarged Cardiomediastinum"],
    "Lung Opacity": [],
    "Lung Lesion": ["Lung Opacity"],
    "Edema": ["Lung Opacity"],
    "Consolidation": ["Lung Opacity"],
    "Pneumonia": ["Lung Opacity", "Consolidation"],
    "Atelectasis": ["Lung Opacity"],
    "Pneumothorax": [],
    "Pleural Effusion": [],
    "Pleural Other": [],
    "Fracture": [],
    "Support Devices": [],
}


def get_conditional_mask(df: pd.DataFrame, pathologies: List[str]) -> pd.Series:
    """Return a boolean mask over rows where all parent labels are positive (1.0).

    For Phase 1 of conditional training: keeps only samples where every parent
    label (for every active pathology that has parents) equals 1.0.

    Args:
        df: DataFrame with pathology columns.
        pathologies: Active pathology labels being trained on.

    Returns:
        Boolean Series — True for rows to include in conditional training.
    """
    mask = pd.Series(True, index=df.index)

    for pathology in pathologies:
        parents = CHEXPERT_HIERARCHY.get(pathology, [])
        for parent in parents:
            if parent in df.columns:
                # Parent must be exactly 1.0 (positive)
                mask = mask & (df[parent] == 1.0)

    return mask


def apply_hierarchical_inference(
    probs: torch.Tensor,
    pathologies: List[str],
) -> torch.Tensor:
    """Multiply conditional probabilities along the hierarchy to get unconditional probs.

    At inference time, the model outputs P(child | parents positive). To get
    P(child), we multiply along the parent chain:
        P(Pneumonia) = P(LungOpacity) * P(Consolidation|LungOpacity) * P(Pneumonia|Consolidation,LungOpacity)

    If a parent label is not in the active pathologies list, it is treated as 1.0
    (i.e., no multiplication), which gracefully handles subset label configurations.

    Args:
        probs: Tensor of shape (B, K) — conditional probabilities from sigmoid(logits).
        pathologies: List of K pathology names matching the columns of probs.

    Returns:
        Tensor of shape (B, K) — unconditional probabilities.
    """
    # Build index lookup: label name -> column index
    label_to_idx = {label: i for i, label in enumerate(pathologies)}

    unconditional = probs.clone()

    for label in pathologies:
        parents = CHEXPERT_HIERARCHY.get(label, [])
        idx = label_to_idx[label]
        for parent in parents:
            if parent in label_to_idx:
                parent_idx = label_to_idx[parent]
                unconditional[:, idx] = unconditional[:, idx] * probs[:, parent_idx]

    return unconditional
