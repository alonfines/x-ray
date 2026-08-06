import os
import yaml
import torch
import torch.nn as nn
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
            labels = config.get('labels', 'all')
            if labels == 'all':
                self.num_classes = len(ALL_CHEXPERT_LABELS)  # 13
            else:
                self.num_classes = len(labels) if labels else 13
        else:
            self.num_classes = num_classes

        # Load torchxrayvision pretrained DenseNet
        self.model = DenseNet(num_classes=self.num_classes, apply_sigmoid=False)

    def freeze_backbone(self):
        """Freeze all layers except the final classifier head (for CT Phase 2)."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier.weight.requires_grad = True
        self.model.classifier.bias.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
