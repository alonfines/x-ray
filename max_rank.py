"""max-rank joint conformal calibration (Timans et al. 2025, Algorithm 2).

Calibrates a group of labels as one joint object so the guarantee is over the label
*vector*: P(y_l in C_l(x) for all l in group) >= 1 - alpha.

How it works, on the locked true-class nonconformity score:
  1. Rank each calibration row within every column (ascending score -> ascending rank).
  2. Reduce each row to its l-inf over ranks: r_i = max_l rank(s_il).
  3. Take the split-conformal order statistic of r: k = ceil((n+1)(1-alpha))-th smallest.
  4. Per column, the threshold is the k-th smallest score in that column.

Step 3 is where the multiplicity is paid for: one quantile on the row-wise max rank
rather than a per-column quantile, so the correction is learned from the observed
dependence between columns instead of assumed from a union bound.

Usage
-----
    python max_rank.py --clusters configs/clusters/example.yaml --experiment experiments/046

Importable
----------
    from max_rank import calibrate_max_rank, run
"""

from __future__ import annotations

import argparse

import numpy as np

import conformal_common as cc

METHOD = "max_rank"


def column_ranks(scores: np.ndarray) -> np.ndarray:
    """Ascending ranks 1..n within each column, ties broken by max rank.

    Max-rank tie handling is the conservative choice: tied calibration scores land on
    the same (higher) rank, which can only widen the resulting threshold.
    """
    n, m = scores.shape
    ranks = np.empty((n, m), dtype=np.int64)
    for j in range(m):
        order = np.argsort(scores[:, j], kind="mergesort")
        r = np.empty(n, dtype=np.int64)
        r[order] = np.arange(1, n + 1)
        # promote ties to the largest rank in their tie block
        s_sorted = scores[order, j]
        uniq, inv = np.unique(s_sorted, return_inverse=True)
        last = np.zeros(uniq.size, dtype=np.int64)
        np.maximum.at(last, inv, np.arange(1, n + 1))
        r[order] = last[inv]
        ranks[:, j] = r
    return ranks


def calibrate_max_rank(group: cc.GroupData, alpha: float) -> tuple[Dict[str, float], Dict]:
    """Algorithm 2: row-wise l-inf of ranks -> rank quantile -> column order statistic."""
    n, m = group.scores.shape
    if n == 0:
        raise RuntimeError("no usable calibration rows for this group")

    ranks = column_ranks(group.scores)
    row_max_rank = ranks.max(axis=1)

    k_rank = cc.conformal_rank(n, alpha)
    if k_rank > n:
        # n too small for this alpha: no finite threshold, every class is retained.
        thresholds = {lab: float("inf") for lab in group.labels}
        return thresholds, {"n_calibration": n, "k_rank": k_rank, "degenerate": True}

    # the (1-alpha) order statistic of the row-wise max rank, as an integer rank index
    k_star = int(np.partition(row_max_rank, k_rank - 1)[k_rank - 1])
    k_star = int(min(max(k_star, 1), n))

    thresholds = {lab: cc.order_statistic(group.scores[:, j], k_star)
                  for j, lab in enumerate(group.labels)}
    diagnostics = {
        "n_calibration": n,
        "m": m,
        "k_rank": k_rank,
        "k_star": k_star,
        "k_star_fraction": k_star / n,
        "bonferroni_k_for_reference": cc.conformal_rank(n, alpha / m),
        "positives_per_label": group.positives_per_label(),
        "degenerate": False,
    }
    return thresholds, diagnostics


def run(
    clusters_path: str,
    experiment_dir: str,
    alpha: float | None = None,
    stem: str | None = None,
    force: bool = False,
    run_gate0: bool = True,
    splits: cc.Splits | None = None,
) -> "cc.pd.DataFrame":
    """Calibrate + evaluate every pre-registered cluster. Returns the results frame.

    `splits` can be passed in to reuse CSVs already loaded by a caller (e.g. a driver
    that runs both max-rank and the Bonferroni family in one process).
    """
    clusters, spec = cc.load_clusters(clusters_path)
    alpha = alpha if alpha is not None else float(spec.get("alpha", 0.1))
    splits = splits or cc.load_splits(stem=stem, run_gate0=run_gate0)
    cache = cc.ThresholdCache(experiment_dir, splits, force=force)

    records: List[Dict] = []
    for cluster in clusters:
        cal = cc.extract_group(splits.calibration, cluster.labels)
        ev = cc.extract_group(splits.evaluation, cluster.labels)
        warnings = cc.warn_small_positive_support(cal, METHOD)

        thresholds, diagnostics, cached = cache.get_or_compute(
            METHOD, cluster, alpha, lambda c=cal: calibrate_max_rank(c, alpha)
        )
        rec = {
            "method": METHOD,
            "variant": "",
            "cluster": cluster.name,
            "m": cluster.m,
            "alpha": alpha,
            "alpha_effective": alpha,
            "labels": "|".join(cluster.labels),
            "n_calibration": cal.n,
            "k_star": diagnostics.get("k_star"),
            "thresholds_cached": cached,
            "low_support_warning": bool(warnings),
        }
        rec.update(cc.evaluate(ev, thresholds))
        rec.update({f"threshold[{lab}]": thresholds[lab] for lab in cluster.labels})
        records.append(rec)

    df = cc.results_to_frame(records)
    cc.print_coverage_table(df, alpha)
    cc.save_results(df, experiment_dir, f"{METHOD}_results")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cc.add_common_args(p)
    a = p.parse_args()
    run(a.clusters, a.experiment, alpha=a.alpha, stem=a.stem, force=a.force,
        run_gate0=not a.skip_gate0)


if __name__ == "__main__":
    main()
