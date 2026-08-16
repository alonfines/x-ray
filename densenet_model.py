import yaml
import torch
import torch.nn as nn
from torchxrayvision.models import DenseNet

from data import ALL_CHEXPERT_LABELS


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
