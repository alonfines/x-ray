import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from torchxrayvision.models import DenseNet

# Import label constants for consistency with data.py
ALL_CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]


class CXRDenseNet(nn.Module):
    """DenseNet classifier for multi-label chest X-ray classification."""

    def __init__(self, config_path: str = "config.yaml", num_classes: int = None):
        """
        Initialize DenseNet with torchxrayvision X-ray pretrained weights.

        Args:
            config_path: Path to config.yaml file
            num_classes: Number of output classes (if None, derived from config)
        """
        super().__init__()

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Determine number of classes
        if num_classes is None:
            use_all_labels = config.get('use_all_labels', False)
            if use_all_labels:
                self.num_classes = len(ALL_CHEXPERT_LABELS)  # 13
            else:
                use_labels = config.get('use_labels', [])
                self.num_classes = len(use_labels) if use_labels else 13
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


class AUCMarginLoss(nn.Module):
    """
    AUC Margin Loss from Yuan et al. (2021).
    "Large-scale Robust Deep AUC Maximization" (arXiv:2012.03173)

    Implements the min-max surrogate loss (equation 8) for multi-label classification.
    For each label k, maintains learnable parameters (a_k, b_k, alpha_k) where:
      - a_k: tracks mean prediction on positive samples
      - b_k: tracks mean prediction on negative samples
      - alpha_k: dual variable (maximized, constrained to >= 0)

    The model parameters are minimized while alpha is maximized (primal-dual).
    """

    def __init__(self, num_classes: int, margin: float = 1.0, imratio: list = None):
        super().__init__()
        self.margin = margin
        self.num_classes = num_classes

        # Learnable primal-dual parameters (per class)
        self.a = nn.Parameter(torch.zeros(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))
        self.alpha = nn.Parameter(torch.zeros(num_classes))

        # Prior probability p_k = Pr(y_k = 1) per class
        if imratio is not None:
            self.register_buffer('p', torch.tensor(imratio, dtype=torch.float32))
        else:
            self.register_buffer('p', torch.full((num_classes,), 0.5))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(logits)
        p = self.p

        # Equation (8): F_M(w, a, b, α; z)
        # Term 1: (1-p)(pred - a)^2 * y
        loss = (1 - p) * ((preds - self.a) ** 2) * targets
        # Term 2: p * (pred - b)^2 * (1-y)
        loss = loss + p * ((preds - self.b) ** 2) * (1 - targets)
        # Term 3: -p(1-p) * alpha^2
        loss = loss - p * (1 - p) * (self.alpha ** 2)
        # Term 4: 2*alpha * (p(1-p)*m + p*pred*(1-y) - (1-p)*pred*y)
        loss = loss + 2 * self.alpha * (
            p * (1 - p) * self.margin
            + p * preds * (1 - targets)
            - (1 - p) * preds * targets
        )

        return loss.mean()


def get_loss_function(loss_type: str = 'bce', weighted: bool = False,
                      pos_weight: torch.Tensor = None, num_classes: int = None,
                      margin: float = 1.0, imratio: list = None) -> nn.Module:
    """
    Get loss function for multi-label classification.

    Args:
        loss_type: 'bce' for BCEWithLogitsLoss, 'auc_margin' for AUC Margin Loss
        weighted: If True, use weighted BCEWithLogitsLoss (only for 'bce')
        pos_weight: Class weights for imbalanced datasets (only for 'bce')
        num_classes: Number of output classes (required for 'auc_margin')
        margin: Margin parameter m for AUC Margin Loss (only for 'auc_margin')
        imratio: Prior probability of positive class per label (only for 'auc_margin')

    Returns:
        Loss function module
    """
    if loss_type == 'auc_margin':
        return AUCMarginLoss(num_classes=num_classes, margin=margin, imratio=imratio)
    elif weighted and pos_weight is not None:
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
