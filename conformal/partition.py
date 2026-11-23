"""Which labels are calibrated together. A partition is a list of clusters (lists of label names).

    singletons()        every label alone (classical per-label conformal)
    random_partition()  random clusters with given sizes (the baseline a designed partition must beat)
    search()            exact minimum-cross-entropy partition per m_max: the MLP scorer's holdout CE
                        for every candidate cluster, then a DP over all label subsets. Validation only.
"""

from __future__ import annotations

import itertools
import os
from multiprocessing import Pool

import numpy as np

from .data import LABELS, Data

Partition = list[list[str]]


def singletons(labels: list[str] = LABELS) -> Partition:
    return [[l] for l in labels]


def random_partition(sizes: list[int], rng: np.random.Generator, labels: list[str] = LABELS) -> Partition:
    if sum(sizes) != len(labels):
        raise ValueError(f"cluster sizes {sizes} do not sum to {len(labels)} labels")
    perm, cuts = rng.permutation(labels).tolist(), np.cumsum([0] + list(sizes))
    return [perm[a:b] for a, b in zip(cuts[:-1], cuts[1:])]


def check(partition: Partition, labels: list[str] = LABELS) -> list[str]:
    """The partition's labels, in model order; raises on unknown or repeated labels."""
    flat = [l for c in partition for l in c]
    if len(set(flat)) != len(flat) or not set(flat) <= set(labels):
        raise ValueError(f"not a partition of model labels: {partition}")
    return [l for l in labels if l in flat]


# ---------------------------------------------------------------------------
# DP partition search
# ---------------------------------------------------------------------------

def _holdout_ce(args):
    from .scores.mlp import MLP
    import torch
    torch.set_num_threads(1)
    train, cluster, seed = args
    scorer = MLP()
    scorer.fit(train, list(cluster), seed)
    return cluster, scorer.info["holdout_ce"]


def best_partition(cost: dict[frozenset, float], labels: list[str], m_max: int) -> tuple[float, Partition]:
    """Exact DP over label subsets: f[A] = min over clusters c containing A's first label of cost[c] + f[A - c]."""
    bit = {l: 1 << i for i, l in enumerate(labels)}
    cost_bits = {sum(bit[l] for l in c): v for c, v in cost.items() if len(c) <= m_max}
    f, arg = {0: 0.0}, {}
    for mask in range(1, 1 << len(labels)):
        low, sub = mask & -mask, mask
        f[mask] = np.inf
        while sub:
            if sub & low and sub in cost_bits and cost_bits[sub] + f[mask ^ sub] < f[mask]:
                f[mask], arg[mask] = cost_bits[sub] + f[mask ^ sub], sub
            sub = (sub - 1) & mask
    parts, mask = [], (1 << len(labels)) - 1
    while mask:
        parts.append([l for l in labels if arg[mask] & bit[l]])
        mask ^= arg[mask]
    return f[(1 << len(labels)) - 1], sorted(parts, key=len, reverse=True)


def search(train: Data, m_max_grid=(3, 4, 5, 6), labels: list[str] = LABELS, seed: int = 0,
           processes: int | None = None) -> dict:
    """Minimum-cost partition for each m_max. Cost of a cluster = holdout CE of its MLP scorer;
    a partition's cost = the sum over its clusters."""
    cands = [c for k in range(1, max(m_max_grid) + 1) for c in itertools.combinations(labels, k)]
    with Pool(processes or os.cpu_count()) as pool:
        cost = {frozenset(c): ce for c, ce in pool.imap_unordered(_holdout_ce, [(train, c, seed) for c in cands], 4)}
    singles = sum(cost[frozenset([l])] for l in labels)
    out = {"singletons_cost": singles, "cluster_costs": {" | ".join(sorted(c)): v for c, v in cost.items()},
           "partitions": {}}
    for mm in m_max_grid:
        total, parts = best_partition(cost, labels, mm)
        out["partitions"][mm] = {"cost": total, "gain_vs_singletons": singles - total, "clusters": parts}
    return out
