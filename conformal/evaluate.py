"""Metrics of a joint set on the test rows, and summaries over resplits.

Headline pair: joint coverage (validity) and set size (efficiency). share_impossible is the mean,
over non-empty sets, of the fraction of the set's label vectors that violate the hierarchy. Size is reported as the mean
and median number of label vectors kept, and as mean log2(size) / M (M = number of labels), which
compares partitions with different M.

CIs are 2.5/97.5 percentiles over resplits, conditional on the fixed pool and the fixed trained
scorers, so they understate total uncertainty. Say so with every CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(j: dict, n_labels: int) -> dict:
    size, dec = j["size"], j["decisive"]
    ne = size >= 1
    return {
        "n_test": int(size.size),
        "joint_coverage": float(j["covered"].mean()),
        "mean_size": float(size.mean()),
        "median_size": float(np.median(size)),
        "log2_size_over_m": float(np.log2(size[ne]).mean() / n_labels) if ne.any() else np.nan,
        "empty_rate": float((~ne).mean()),
        "mean_admissible_size": float(j["admissible_size"].mean()),
        "share_impossible": float((1 - j["admissible_size"][ne] / size[ne]).mean()) if ne.any() else np.nan,
        "decisive_rate": float(dec.mean()),
        "correct_decisive_rate": float(j["correct"].mean()),
        "decisive_accuracy": float(j["correct"][dec].mean()) if dec.any() else np.nan,
        "violation_rate_decisive": float(j["violating"][dec].mean()) if dec.any() else np.nan,
        **{f"coverage_c{c}": float(v) for c, v in enumerate(j["cluster_covered"].mean(axis=0))},
    }


def summarize(df: pd.DataFrame, by=("rule",), metric_cols=None) -> pd.DataFrame:
    """Per group: resplit mean, sd, 2.5/97.5 percentiles, and the frozen-draw value (resplit == -1)."""
    metric_cols = metric_cols or ["joint_coverage", "mean_size", "median_size", "log2_size_over_m",
                                  "decisive_rate", "correct_decisive_rate"]
    rows = []
    for key, g in df.groupby(list(by), sort=False):
        rs, frozen = g[g.resplit >= 0], g[g.resplit == -1]
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)), n_resplits=len(rs))
        for c in metric_cols:
            v = rs[c].to_numpy(float)
            rec.update({f"{c}_mean": np.nanmean(v), f"{c}_sd": np.nanstd(v, ddof=1),
                        f"{c}_lo": np.nanpercentile(v, 2.5), f"{c}_hi": np.nanpercentile(v, 97.5),
                        f"{c}_frozen": float(frozen[c].iloc[0]) if len(frozen) else np.nan})
        rows.append(rec)
    return pd.DataFrame(rows)


def paired(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict:
    """Mean and 95% percentile CI of a - b, paired within resplit (resplit >= 0 rows of each)."""
    x = a[a.resplit >= 0].set_index("resplit")[metric]
    y = b[b.resplit >= 0].set_index("resplit")[metric]
    d = (x - y.reindex(x.index)).to_numpy(float)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"metric": metric, "mean_diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_resplits": int(d.size)}
