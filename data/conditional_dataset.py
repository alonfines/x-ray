"""Conditional dataset wrapper for hierarchical training Phase 1.

Filters a base dataset to only include samples where all parent labels
in the disease hierarchy are positive (1.0). Used for the conditional
training phase from Pham et al. (2020).
"""

import pandas as pd
from torch.utils.data import Dataset
from typing import List

from data.hierarchy import get_conditional_mask, CHEXPERT_HIERARCHY


class ConditionalSubset(Dataset):
    """Wraps a CheXpert/MIMIC dataset, filtering to rows with all parents positive.

    The filtered indices are computed once at construction from the base dataset's
    DataFrame. __getitem__ delegates to the base dataset using the filtered indices.
    """

    def __init__(self, base_dataset: Dataset, pathologies: List[str]):
        self.base_dataset = base_dataset
        self.pathologies = pathologies

        mask = get_conditional_mask(base_dataset.df, pathologies)
        self.indices = mask[mask].index.tolist()

        total = len(base_dataset)
        filtered = len(self.indices)
        print(f"[ConditionalSubset] Filtered {total} → {filtered} samples "
              f"({filtered / total * 100:.1f}%) where all parent labels are positive")

        if filtered < 1000:
            print(f"  WARNING: Only {filtered} samples remain after conditional filtering. "
                  f"Phase 1 training may underperform.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

    @property
    def df(self) -> pd.DataFrame:
        """Expose filtered DataFrame for compatibility with calculate_pos_weights()."""
        return self.base_dataset.df.iloc[self.indices]
