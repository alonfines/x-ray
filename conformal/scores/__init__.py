"""Nonconformity scores over a cluster's joint states.

A score object has one method:

    fit(train: Data, labels: list[str], seed: int) -> callable
        the callable maps a Data split to an (n, 2^m) array: the score of every joint state of
        `labels` (data.states order) for every row. Lower = more conforming.

`train` is the proper-training split (validation); scores that learn nothing ignore it.
"""

from .lp_scp import LPSCP
from .mlp import MLP
from .per_label import PerLabel

SCORES = {"per_label": PerLabel, "lp_scp": LPSCP, "mlp": MLP}
