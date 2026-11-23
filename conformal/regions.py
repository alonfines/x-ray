"""Prediction regions. Cluster c keeps the states whose score is <= its threshold t_c; the joint set
is the product of the cluster regions, so it is never materialised: coverage is "every cluster
covers its true state" and size is the product of the per-cluster sizes.

No hierarchy information enters the sets. The hierarchy is only measured: admissible_size counts
the set's hierarchy-admissible vectors. Clusters linked by a hierarchy edge are enumerated together
for that count, so it stays exact; a hierarchy-closed partition needs no merging.

guard=True adds a cluster's best-scoring state to rows where its region is empty.
"""

from __future__ import annotations

import numpy as np

from .data import admissible as is_admissible, state_index, states
from models.mimic.data.hierarchy import CHEXPERT_HIERARCHY


def components(partition: list[list[str]], hierarchy: dict = CHEXPERT_HIERARCHY) -> list[list[int]]:
    """Groups of cluster indices linked by hierarchy edges (enumerated together to count admissible vectors)."""
    groups = [[c] for c in range(len(partition))]
    owner = {l: c for c, cl in enumerate(partition) for l in cl}
    for child, parents in hierarchy.items():
        for parent in parents:
            if child in owner and parent in owner:
                a = next(g for g in groups if owner[child] in g)
                b = next(g for g in groups if owner[parent] in g)
                if a is not b:
                    a.extend(b)
                    groups.remove(b)
    return [sorted(g) for g in groups]


def cluster_regions(scores: list[np.ndarray], thresholds: np.ndarray, guard: bool) -> list[np.ndarray]:
    """Per cluster, the boolean (n, 2^m) region at its threshold."""
    out = []
    for s, t in zip(scores, thresholds):
        reg = s <= t
        if guard:
            empty = ~reg.any(axis=1)
            reg[empty, s[empty].argmin(axis=1)] = True
        out.append(reg)
    return out


def _admissible_count(regions, comp, partition, n, hierarchy) -> np.ndarray:
    """Number of hierarchy-admissible vectors in one component's (product) region."""
    labels = [l for c in comp for l in partition[c]]
    S, reg, a = states(len(labels)), np.ones((n, 2 ** len(labels)), bool), 0
    for c in comp:
        m = len(partition[c])
        reg &= regions[c][:, state_index(S[:, a:a + m])]
        a += m
    return (reg & is_admissible(labels, S, hierarchy)).sum(axis=1)


def joint(regions: list[np.ndarray], y_true: list[np.ndarray], partition, hierarchy: dict = CHEXPERT_HIERARCHY) -> dict:
    """Per-row joint quantities for the product set: covered, size, admissible_size, decisive,
    correct, violating."""
    n = len(regions[0])
    true_idx = [state_index(y) for y in y_true]
    cluster_cov = np.stack([r[np.arange(n), t] for r, t in zip(regions, true_idx)], axis=1)
    size = np.prod([r.sum(axis=1) for r in regions], axis=0).astype(float)
    adm_size = np.prod([_admissible_count(regions, comp, partition, n, hierarchy) for comp in components(partition, hierarchy)],
                       axis=0).astype(float)
    vec = np.concatenate([states(len(cl))[r.argmax(axis=1)] for r, cl in zip(regions, partition)], axis=1)
    run_labels = [l for c in partition for l in c]
    truth = np.concatenate(y_true, axis=1)
    decisive = size == 1
    return {"covered": cluster_cov.all(axis=1), "cluster_covered": cluster_cov, "size": size,
            "admissible_size": adm_size, "decisive": decisive,
            "correct": decisive & (vec == truth).all(axis=1),
            "violating": decisive & ~is_admissible(run_labels, vec, hierarchy)}
