"""One call from a partition to per-resplit metrics.

    from conformal import run, partition
    df = run.run(partition.singletons(), score="per_label", rules=["bonferroni", "max_rank"])
    run.evaluate.summarize(df)

Scorers are fit once on the validation split and score the whole pool once; each draw only
recomputes thresholds on its calibration half and metrics on its test half. Every threshold uses
k = ceil((n+1)(1-alpha')).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate, quantile
from .data import draws, load, pool, state_index
from .partition import check
from .regions import cluster_regions, joint
from .scores import SCORES

FLOAT32_ABOVE = 1024     # score matrices with more states are stored as float32 to bound memory


def run(partition: list[list[str]], score="per_label", rules=("max_rank",), alpha: float = 0.1,
        guard: bool = False, n_resplits: int = 100, seed: int = 0,
        data: dict | None = None, dataset: str = "mimic") -> pd.DataFrame:
    """Rows = (rule, resplit); resplit -1 is the frozen conformal/test draw.

    score: a name in scores.SCORES or a score object (e.g. scores.LPSCP(mu=0, nu=0)).
    dataset: a name in data.DATASETS; ignored when `data` is given.
    """
    data = data or load(dataset=dataset)
    labels_all = data["validation"].labels
    check(partition, labels_all)
    scorer = SCORES[score]() if isinstance(score, str) else score
    pooled = pool(data)

    S, true = [], []                     # per cluster: (N, 2^m) state scores, (N,) true-state score
    for labels in partition:
        s = scorer.fit(data["validation"], labels, seed)(pooled)
        s = s.astype(np.float32) if s.shape[1] > FLOAT32_ABOVE else s
        S.append(s)
        true.append(s[np.arange(pooled.n), state_index(pooled.y[:, [labels_all.index(l) for l in labels]])])
    true = np.stack(true, axis=1)
    y_cols = [[labels_all.index(l) for l in labels] for labels in partition]
    n_labels = sum(map(len, partition))

    rows = []
    for r, cal, test in draws(data, n_resplits):
        for rule in rules:
            t = quantile.calibrate(true[cal], alpha, rule)
            regs = cluster_regions([s[test] for s in S], t, guard)
            j = joint(regs, [pooled.y[test][:, c] for c in y_cols], partition, pooled.hierarchy)
            rows.append({"rule": rule, "resplit": r, "n_cal": cal.size, **evaluate.metrics(j, n_labels),
                         **{f"threshold_c{c}": float(v) for c, v in enumerate(t)}})
    return pd.DataFrame(rows)
