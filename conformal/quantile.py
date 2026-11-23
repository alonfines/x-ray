"""Split-conformal quantiles, and the rules that split alpha across K clusters.

Every method calibrates the same object: an (n, K) matrix of calibration scores, column c being
cluster c's score of its own true joint state. A rule turns it into K thresholds, and the
prediction set keeps every state scoring <= its cluster's threshold.

    none        each column at alpha                 per-cluster coverage only (NOT jointly valid)
    bonferroni  each column at alpha / K             union bound
    sidak       each column at 1 - (1-alpha)^(1/K)   union bound, sharpened under independence
    max_rank    Timans et al. 2025, Alg. 2           jointly valid; uses the dependence between columns

One convention everywhere: k = ceil((n+1)(1-alpha)), the k-th smallest score, no interpolation.
k > n means no finite threshold exists (threshold = +inf, every state kept).
"""

from __future__ import annotations

import math

import numpy as np

RULES = ("none", "bonferroni", "sidak", "max_rank")


def conformal_rank(n: int, alpha: float) -> int:
    return int(math.ceil((n + 1) * (1.0 - alpha)))


def order_statistic(values: np.ndarray, k: int) -> float:
    """k-th smallest (1-indexed); +inf if k > len(values)."""
    return float("inf") if k > values.size else float(np.partition(values, k - 1)[k - 1])


def quantile(scores: np.ndarray, alpha: float) -> float:
    """The split-conformal threshold of a 1-D score vector."""
    return order_statistic(scores, conformal_rank(scores.size, alpha))


def column_ranks(scores: np.ndarray) -> np.ndarray:
    """Ascending ranks 1..n within each column; ties take the largest rank of their block (conservative)."""
    ranks = np.empty(scores.shape, dtype=np.int64)
    for j in range(scores.shape[1]):
        col = scores[:, j]
        ranks[:, j] = np.searchsorted(np.sort(col), col, side="right")
    return ranks


def calibrate(scores: np.ndarray, alpha: float, rule: str) -> np.ndarray:
    """(n, K) true-state calibration scores -> (K,) thresholds."""
    n, K = scores.shape
    if rule == "max_rank":
        k = conformal_rank(n, alpha)
        if k > n:
            return np.full(K, np.inf)
        k_star = int(np.partition(column_ranks(scores).max(axis=1), k - 1)[k - 1])
        return np.array([order_statistic(scores[:, j], k_star) for j in range(K)])
    level = {"none": alpha, "bonferroni": alpha / K, "sidak": 1 - (1 - alpha) ** (1 / K)}[rule]
    return np.array([quantile(scores[:, j], level) for j in range(K)])
