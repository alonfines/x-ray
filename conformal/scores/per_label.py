"""True-class score for a single label: s(x, 0) = p, s(x, 1) = 1 - p.

Defined for one-label clusters only. With every label its own cluster this gives classical
per-label conformal; the quantile rule (none / bonferroni / sidak / max_rank) does the rest.
"""

from __future__ import annotations

import numpy as np

from ..data import Data


class PerLabel:
    def fit(self, train: Data, labels: list[str], seed: int = 0):
        if len(labels) != 1:
            raise ValueError(f"per_label needs one-label clusters, got {labels}")
        j = train.labels.index(labels[0])
        return lambda d: np.stack([d.probs[:, j], 1.0 - d.probs[:, j]], axis=1)
