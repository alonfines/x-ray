"""Lung / cardio / rest: independence score + Bonferroni / max-rank vs MLP + additive sum.

Baselines: per cluster, the independence score s(x, y) = -sum_j log p_j(y_j) on the raw model
probabilities (no training), combined across clusters by Bonferroni or max-rank
(conformal.quantile / conformal.regions unchanged; product set).
Additive: one split-conformal threshold on S(x, y) = sum_c -log p_c(y_c | x); the set is every
13-label vector with S <= q (not a product), counted by enumerating all 4 * 64 * 32 = 8192 states.
Per-cluster coverage of the additive set = the true cluster state appears in the set's projection,
i.e. s_c(true) + sum_{c' != c} min s_c' <= q.

alpha = 0.10, MLP seed 0, frozen draw + 100 subject-level resplits, all-13-clean population.
Run: .venv/bin/python experiments/060_lung_cardio_rest_additive/run.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from conformal import evaluate, quantile
from conformal.data import LABELS, draws, load, pool, state_index, states
from conformal.regions import cluster_regions, joint
from conformal.scores.mlp import MLP
from conformal.scores.lp_scp import LPSCP

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA, SEED, N_RESPLITS = 0.10, 0, 100
PARTITION = {
    "lung": ["Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis"],
    "cardio": ["Enlarged Cardiomediastinum", "Cardiomegaly"],
    "rest": ["Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"],
}
NAMES, CLUSTERS = list(PARTITION), list(PARTITION.values())
assert sorted(l for c in CLUSTERS for l in c) == sorted(LABELS)


def additive_sizes(S: list[np.ndarray], q: float, batch: int = 1024) -> np.ndarray:
    """Number of 13-label vectors with sum_c S_c[state_c] <= q, per row (3 clusters)."""
    a, b, c = S
    out = np.empty(len(a), dtype=np.int64)
    for i in range(0, len(a), batch):
        tot = (a[i:i + batch, :, None, None] + b[i:i + batch, None, :, None]
               + c[i:i + batch, None, None, :])
        out[i:i + batch] = (tot <= q).reshape(len(tot), -1).sum(axis=1)
    return out


def independence_scores(probs: np.ndarray, labels: list[str]) -> np.ndarray:
    """(N, 2^m) -sum_j log p_j(y_j) over the cluster's labels, states in states(m) order."""
    p = np.clip(probs[:, [LABELS.index(l) for l in labels]], 1e-6, 1 - 1e-6)
    st = states(len(labels))
    return -(np.log(p) @ st.T + np.log1p(-p) @ (1 - st).T)


def main():
    data = load()
    pooled = pool(data)
    rows_idx = np.arange(pooled.n)
    S, true, info = [], [], {}
    for name, labels in PARTITION.items():
        scorer = MLP()
        s = scorer.fit(data["validation"], labels, SEED)(pooled)
        info[name] = scorer.info
        S.append(s)
        true.append(s[rows_idx, state_index(pooled.y[:, [LABELS.index(l) for l in labels]])])
    true = np.stack(true, axis=1)                        # (N, 3) true-state scores
    mins = np.stack([s.min(axis=1) for s in S], axis=1)  # (N, 3) best state score per cluster
    y_cols = [[LABELS.index(l) for l in labels] for labels in CLUSTERS]
    print("MLP fits:", info)
    S_ind = [independence_scores(pooled.probs, labels) for labels in CLUSTERS]
    S_lp = [LPSCP(mu=1.0, nu=1.0).fit(data["validation"], labels, SEED)(pooled) for labels in CLUSTERS]
    true_lp = np.stack([s[rows_idx, state_index(pooled.y[:, c])] for s, c in zip(S_lp, y_cols)], axis=1)
    true_ind = np.stack([s[rows_idx, state_index(pooled.y[:, c])] for s, c in zip(S_ind, y_cols)], axis=1)

    rows = []
    for r, cal, test in draws(data, N_RESPLITS):
        for rule in ("bonferroni", "max_rank"):
            t = quantile.calibrate(true_ind[cal], ALPHA, rule)
            regs = cluster_regions([s[test] for s in S_ind], t, guard=False)
            j = joint(regs, [pooled.y[test][:, c] for c in y_cols], CLUSTERS)
            m = evaluate.metrics(j, 13)
            rows.append({"rule": "indep_" + rule, "resplit": r, "joint_coverage": m["joint_coverage"],
                         "mean_size": m["mean_size"], "median_size": m["median_size"],
                         "empty_rate": m["empty_rate"],
                         **{f"coverage_{n}": m[f"coverage_c{c}"] for c, n in enumerate(NAMES)}})
        t = quantile.calibrate(true_lp[cal], ALPHA, "sidak")        # LP-SCP (mu = nu = 1), each cluster at 96.5%, product set
        regs = cluster_regions([s[test] for s in S_lp], t, guard=False)
        m = evaluate.metrics(joint(regs, [pooled.y[test][:, c] for c in y_cols], CLUSTERS), 13)
        rows.append({"rule": "lpscp_sidak", "resplit": r, "joint_coverage": m["joint_coverage"],
                     "mean_size": m["mean_size"], "median_size": m["median_size"], "empty_rate": m["empty_rate"],
                     **{f"coverage_{n}": m[f"coverage_c{c}"] for c, n in enumerate(NAMES)}})
        t = quantile.calibrate(true[cal], ALPHA, "sidak")        # MLP, each cluster at 0.9^(1/3) = 96.5%
        regs = cluster_regions([s[test] for s in S], t, guard=False)
        m = evaluate.metrics(joint(regs, [pooled.y[test][:, c] for c in y_cols], CLUSTERS), 13)
        rows.append({"rule": "mlp_sidak", "resplit": r, "joint_coverage": m["joint_coverage"],
                     "mean_size": m["mean_size"], "median_size": m["median_size"], "empty_rate": m["empty_rate"],
                     **{f"coverage_{n}": m[f"coverage_c{c}"] for c, n in enumerate(NAMES)}})
        q = quantile.quantile(true[cal].sum(axis=1), ALPHA)
        tot_true = true[test].sum(axis=1)
        size = additive_sizes([s[test] for s in S], q).astype(float)
        proj = true[test] + (mins[test].sum(axis=1, keepdims=True) - mins[test])
        rows.append({"rule": "additive", "resplit": r, "joint_coverage": float((tot_true <= q).mean()),
                     "mean_size": float(size.mean()), "median_size": float(np.median(size)),
                     "empty_rate": float((size == 0).mean()),
                     **{f"coverage_{n}": float((proj[:, c] <= q).mean()) for c, n in enumerate(NAMES)}})
        if r % 20 == 0:
            print("resplit", r, "done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "resplits.csv"), index=False)
    cols = ["joint_coverage", "mean_size", "median_size", "empty_rate"] + [f"coverage_{n}" for n in NAMES]
    summ = evaluate.summarize(df, by=("rule",), metric_cols=cols)
    summ.to_csv(os.path.join(HERE, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summ[["rule"] + [f"{c}_mean" for c in cols] + ["joint_coverage_lo", "joint_coverage_hi",
                                                             "mean_size_lo", "mean_size_hi"]].round(4))


if __name__ == "__main__":
    main()
