"""Exp058: MLP joint-state scorer with vs without the hierarchy mask (renormalize_admissible).

Partition: DP m_max=6 (classifier/reports/partition_search/partitions.json), hierarchy-closed.
Rules: max_rank, bonferroni. alpha 0.1. Sets are NOT filtered to admissible vectors (admissible=False),
so share_impossible measures how much of each set is hierarchy-violating. MLP seed 0 in both arms;
frozen draw + 100 subject-level resplits.

    .venv/bin/python experiments/058_mlp_mask/run.py

NOTE: after this experiment the mask option (MLP(renormalize_admissible=...)) and the admissible
set filter were removed from conformal/, so this script no longer runs as-is; it is the record of
how the numbers in REPORT.md were produced.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from conformal import evaluate, run                                   # noqa: E402
from conformal.data import LABELS, admissible, load, pool, state_index, states  # noqa: E402
from conformal.scores.mlp import MLP                                 # noqa: E402

PARTITION = json.load(open(os.path.join(HERE, "..", "..", "classifier", "reports", "partition_search",
                                        "partitions.json")))["partitions"]["6"]["clusters"]
RULES, ALPHA, N_RESPLITS, SEED = ("max_rank", "bonferroni"), 0.1, 100, 0
ARMS = {"no_mask": False, "mask": True}


def diagnostics(data):
    """Per arm and cluster: training info, mean probability mass on impossible states, and how often
    the most likely joint state is impossible (on the pooled conformal+test rows)."""
    pooled, rows = pool(data), []
    for arm, renorm in ARMS.items():
        for c, labels in enumerate(PARTITION):
            scorer = MLP(renormalize_admissible=renorm)
            s = scorer.fit(data["validation"], labels, SEED)(pooled)
            bad = ~admissible(labels, states(len(labels)))
            p = np.exp(-s)
            true = state_index(pooled.y[:, [LABELS.index(l) for l in labels]])
            rows.append({"arm": arm, "cluster": c, "labels": " | ".join(labels), "n_impossible_states": int(bad.sum()),
                         **scorer.info, "test_ce": float(s[np.arange(pooled.n), true].mean()),
                         "mass_impossible": float(p[:, bad].sum(axis=1).mean()) if bad.any() else 0.0,
                         "top1_impossible": float(bad[s.argmin(axis=1)].mean()) if bad.any() else 0.0})
    return pd.DataFrame(rows)


def main():
    data = load()
    diag = diagnostics(data)
    diag.to_csv(os.path.join(HERE, "diagnostics.csv"), index=False)

    dfs = {arm: run.run(PARTITION, MLP(renormalize_admissible=renorm), rules=RULES, alpha=ALPHA,
                        admissible=False, n_resplits=N_RESPLITS, seed=SEED, data=data).assign(arm=arm)
           for arm, renorm in ARMS.items()}
    df = pd.concat(dfs.values(), ignore_index=True)
    df.to_csv(os.path.join(HERE, "resplits.csv"), index=False)

    cols = ["joint_coverage", "mean_size", "median_size", "log2_size_over_m", "mean_admissible_size",
            "share_impossible", "decisive_rate", "correct_decisive_rate"]
    summ = evaluate.summarize(df, by=("arm", "rule"), metric_cols=cols)
    summ.to_csv(os.path.join(HERE, "summary.csv"), index=False)

    paired = [{"rule": r, **evaluate.paired(dfs["no_mask"][dfs["no_mask"].rule == r],
                                            dfs["mask"][dfs["mask"].rule == r], m)}
              for r in RULES for m in cols]
    pd.DataFrame(paired).to_csv(os.path.join(HERE, "paired_no_mask_minus_mask.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(diag.drop(columns=["labels"]).round(4).to_string(index=False))
    show = ["arm", "rule"] + [f"{c}_{s}" for c in cols for s in ("mean", "lo", "hi")]
    print(summ[show].round(4).T.to_string())
    print(pd.DataFrame(paired).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
