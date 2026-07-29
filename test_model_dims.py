#!/usr/bin/env python3
"""
Local smoke test for model output dimensions with random weights.
Tests both 5-label and 13-label modes without requiring compute resources.
"""

import torch
from densenet_model import CXRDenseNet

def test_model_dims():
    """Test model output shapes for different num_classes."""
    test_cases = [
        (5, "5-label mode (use_all_labels=false)"),
        (13, "13-label mode (use_all_labels=true)")
    ]

    for num_classes, description in test_cases:
        print(f"\nTesting {description}...")

        model = CXRDenseNet(config_path="config.yaml", num_classes=num_classes)
        model.eval()

        # Create batch of 2 random images (B, 1, 224, 224)
        x = torch.randn(2, 1, 224, 224)

        with torch.no_grad():
            out = model(x)

        # Verify output shape
        expected_shape = (2, num_classes)
        actual_shape = tuple(out.shape)

        assert actual_shape == expected_shape, \
            f"Shape mismatch: expected {expected_shape}, got {actual_shape}"

        print(f"  ✓ Output shape: {actual_shape} (correct)")
        print(f"  ✓ Output range: [{out.min():.3f}, {out.max():.3f}]")

if __name__ == "__main__":
    print("="*60)
    print("Local Model Dimension Smoke Test")
    print("="*60)

    try:
        test_model_dims()
        print("\n" + "="*60)
        print("All dimension checks PASSED ✓")
        print("="*60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
