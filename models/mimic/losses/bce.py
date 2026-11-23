import torch
import torch.nn as nn


def get_bce_loss(weighted: bool = False, pos_weight: torch.Tensor = None) -> nn.Module:
    if weighted and pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    return nn.BCEWithLogitsLoss(reduction='mean')


def calculate_pos_weights(train_labels: torch.Tensor) -> torch.Tensor:
    labels = train_labels.clone()

    # 1. Replace NaN (missing values) with 0
    labels = torch.nan_to_num(labels, nan=0.0)

    # 2. Replace uncertain labels (-1) with 0
    labels[labels == -1.0] = 0.0

    num_samples = labels.shape[0]
    num_positives = labels.sum(dim=0)
    num_negatives = num_samples - num_positives

    # Avoid division by zero
    pos_weight = num_negatives / (num_positives + 1e-8)

    return pos_weight
