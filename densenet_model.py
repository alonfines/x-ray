import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from torchxrayvision.models import DenseNet


class CXRDenseNet(nn.Module):
    """DenseNet classifier for multi-label chest X-ray classification."""

    def __init__(self, config_path: str = "config.yaml", num_classes: int = None):
        """
        Initialize DenseNet with torchxrayvision X-ray pretrained weights.

        Args:
            config_path: Path to config.yaml file
            num_classes: Number of output classes (if None, derived from use_labels in config)
        """
        super().__init__()

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Determine number of classes
        if num_classes is None:
            use_labels = config.get('use_labels', [])
            self.num_classes = len(use_labels) if use_labels else 14
        else:
            self.num_classes = num_classes

        # Load torchxrayvision pretrained DenseNet
        self.model = DenseNet(num_classes=self.num_classes, apply_sigmoid=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input batch of chest X-ray images (B, 1, H, W)

        Returns:
            logits: Raw model outputs (B, num_classes) for use with BCEWithLogitsLoss
        """
        return self.model(x)


def get_loss_function(weighted: bool = False, pos_weight: torch.Tensor = None) -> nn.Module:
    """
    Get loss function for multi-label classification.

    Args:
        weighted: If True, use weighted BCEWithLogitsLoss
        pos_weight: Class weights for imbalanced datasets (shape: [num_classes])
                   Higher weight for rare positive examples

    Returns:
        Loss function (nn.BCEWithLogitsLoss)

    Notes:
        BCEWithLogitsLoss is ideal for multi-label classification because:
        - It combines sigmoid and binary cross-entropy for numerical stability
        - Each of the 14 pathologies is independent (multi-label, not multi-class)
        - pos_weight helps handle class imbalance in CheXpert dataset
    """
    if weighted and pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    else:
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
