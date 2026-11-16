"""Exp051 — cluster-granularity curve under max-rank, Rungs A-E (13 -> 1).

Pre-registered in experiments/051_cluster_granularity_curve/README.md before this file was run.
Reuses conformal_common (loading, Gate 0, score, metrics helpers), max_rank.calibrate_max_rank,
and bonferrony.calibrate_per_label(correction="none") as the independent baseline. No pipeline file
is modified.

Run:
    cd /Users/alonfines/Desktop/x-ray && .venv_exp048/bin/python experiments/051_cluster_granularity_curve/run.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import conformal_common as cc                      # noqa: E402
import bonferrony                                  # noqa: E402
import max_rank                                    # noqa: E402
from data.hierarchy import CHEXPERT_HIERARCHY      # noqa: E402

SEED = 0
ALPHA = 0.10
np.random.seed(SEED)  # no stochastic step; recorded for reproducibility only

ALL13 = cc.ALL_LABELS
assert len(ALL13) == 13

CARDIO = ["Enlarged Cardiomediastinum", "Cardiomegaly"]
LUNG = ["Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis"]
PLEURAL = ["Pleural Effusion", "Pleural Other"]

# Pre-registered ladder. Lists are the exact partitions from the README.
RUNGS = {
    "A": [[l] for l in ALL13],
    "B": [CARDIO] + [[l] for l in ALL13 if l not in CARDIO],
    "C": [CARDIO, LUNG] + [[l] for l in ["Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]],
    "D": [CARDIO, LUNG, PLEURAL] + [[l] for l in ["Pneumothorax", "Fracture", "Support Devices"]],
    "E": [ALL13],
}

# Every label appears exactly once per rung (partition check).
for r, clusters in RUNGS.items():
    flat = [l for c in clusters for l in c]
    assert sorted(flat) == sorted(ALL13), f"rung {r} is not a partition: {flat}"

# Nested, strict coarsening: each rung's clusters are unions of the previous rung's clusters.
def _is_coarsening(fine, coarse):
    coarse_sets = [set(c) for c in coarse]
    return all(any(set(f) <= cs for cs in coarse_sets) for f in fine)

assert _is_coarsening(RUNGS["A"], RUNGS["B"]) and _is_coarsening(RUNGS["B"], RUNGS["C"])
assert _is_coarsening(RUNGS["C"], RUNGS["D"]) and _is_coarsening(RUNGS["D"], RUNGS["E"])

FLAG_LABELS_MIN = cc.MIN_POSITIVE_CALIBRATION_ROWS


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main() -> None:
    out = {"seed": SEED, "alpha": ALPHA, "stem": None, "gates": {}, "sanity": {}, "rungs": {}, "triggers": {}}

    # ------------------------------------------------------------------ Gate 0
    splits = cc.load_splits(with_validation=True, run_gate0=True)   # raises on any overlap
    out["stem"] = splits.stem
    overlap = cc.gate0_check(splits, raise_on_overlap=True)
    counts = {
        "conformal_rows": len(splits.calibration), "test_rows": len(splits.evaluation),
        "validation_rows": len(splits.validation),
        "conformal_subjects": splits.calibration.subject_id.nunique(),
        "test_subjects": splits.evaluation.subject_id.nunique(),
        "validation_subjects": splits.validation.subject_id.nunique(),
    }
    assert counts["conformal_rows"] == 18754 and counts["test_rows"] == 24394 and counts["validation_rows"] == 18570
    assert counts["conformal_subjects"] == 5926 and counts["test_subjects"] == 6395 and counts["validation_subjects"] == 5898
    out["gates"]["gate0_overlap"] = overlap
    out["gates"]["gate0_counts"] = counts
    assert all(v == 0 for v in overlap.values()), overlap
    print("GATE 0 pass:", overlap, counts)

    # ---------------------------------------------------------------- Gate 0.5
    cal = splits.calibration
    g05 = []
    for lab in ALL13:
        pos = int((cal[f"{lab}_true_label"] == 1).sum())
        neg = int((cal[f"{lab}_true_label"] == 0).sum())
        unc = int((cal[f"{lab}_true_label"] == -1).sum())
        g05.append({"label": lab, "conformal_positive_rows": pos, "conformal_negative_rows": neg,
                    "conformal_uncertain_rows": unc, "flag_under_150": pos < FLAG_LABELS_MIN})
    g05_df = pd.DataFrame(g05)
    g05_df.to_csv(os.path.join(HERE, "gate05_positive_counts.csv"), index=False)
    out["gates"]["gate05_positive_counts"] = g05
    out["gates"]["gate05_flagged"] = [r["label"] for r in g05 if r["flag_under_150"]]
    print("GATE 0.5:")
    print(g05_df.to_string(index=False))

    # ------------------------------------------------- Rung A sanity check
    rows = []
    for lab in ALL13:
        grp = cc.extract_group(cal, [lab])
        t_mr, _ = max_rank.calibrate_max_rank(grp, ALPHA)
        t_ind, _ = bonferrony.calibrate_per_label(grp, ALPHA, "none")
        # Under the locked shared score both candidate classes use one threshold per label,
        # so the (positive, negative) pair is (t, t) on each side.
        for side in ("positive", "negative"):
            rows.append({"label": lab, "side": side,
                         "maxrank_A": t_mr[lab], "independent_per_label": t_ind[lab],
                         "abs_diff": abs(t_mr[lab] - t_ind[lab])})
    san = pd.DataFrame(rows)
    san.to_csv(os.path.join(HERE, "sanity_rungA_vs_independent.csv"), index=False)
    max_diff = float(san["abs_diff"].max())
    out["sanity"] = {"n_thresholds_compared": int(len(san)), "max_abs_diff": max_diff,
                     "tolerance": 1e-12, "pass": bool(max_diff <= 1e-12),
                     "note": "Pair (positive, negative) is (t_j, t_j) under the locked shared score; 26 entries = 13 values duplicated."}
    print(f"\nSANITY Rung A vs independent per-label: n={len(san)} max|diff|={max_diff:.3e} pass={max_diff <= 1e-12}")

    # ------------------------------------------------ Rung-by-rung calibration + test
    test_group = cc.extract_group(splits.evaluation, ALL13)  # 13-label exclusion, locked for all rungs
    n_test = test_group.n
    subj_test = int(len(np.unique(test_group.subject_id)))
    out["gates"]["test_N_rows_after_13label_exclusion"] = n_test
    out["gates"]["test_N_subjects_after_13label_exclusion"] = subj_test
    out["gates"]["test_rows_dropped_13label"] = int(test_group.n_rows_before_filter - n_test)
    print(f"\nTest N after 13-label exclusion: rows={n_test} subjects={subj_test}")

    rung_rows = []
    for r, clusters in RUNGS.items():
        thr_all: dict[str, float] = {}
        cluster_info = []
        for cl in clusters:
            cgroup = cc.extract_group(cal, cl)
            thr, diag = max_rank.calibrate_max_rank(cgroup, ALPHA)
            thr_all.update(thr)
            cluster_info.append({
                "cluster": "|".join(cl), "m": len(cl), "n_calibration": cgroup.n,
                "k_star": diag.get("k_star"), "k_rank": diag.get("k_rank"),
                "degenerate": diag.get("degenerate"),
                "positives_per_label_in_cluster_calibration": cgroup.positives_per_label(),
                "thresholds": {l: thr[l] for l in cl},
            })
        assert set(thr_all) == set(ALL13)

        sets = cc.prediction_sets(test_group, thr_all)          # (n, 13, 2)
        sz = sets.sum(axis=2)                                    # per-label set size
        y = test_group.y
        true_kept = np.take_along_axis(sets, y[:, :, None], axis=2)[:, :, 0]
        covered = true_kept.all(axis=1)
        joint_size = sz.prod(axis=1).astype(float)              # product over all 13 == product over clusters
        dec_pl = sz == 1
        fully = dec_pl.all(axis=1)
        vectors = sets[:, :, 1].astype(int)

        # per-cluster joint sizes (info only)
        cluster_joint_mean = {}
        for cl in clusters:
            idx = [ALL13.index(l) for l in cl]
            cluster_joint_mean["|".join(cl)] = float(sz[:, idx].prod(axis=1).mean())

        if fully.any():
            hv_rate = float((~cc.hierarchy_admissible(ALL13, vectors[fully])).mean())
            dec_acc = float((vectors[fully] == y[fully]).all(axis=1).mean())
        else:
            hv_rate, dec_acc = float("nan"), float("nan")

        rec = {
            "rung": r, "K": len(clusters), "m_per_cluster": [len(c) for c in clusters],
            "N_rows": n_test, "N_subjects": subj_test,
            "joint_coverage": float(covered.mean()),                       # metric 1
            "mean_joint_set_size": float(joint_size.mean()),               # metric 2
            "median_joint_set_size": float(np.median(joint_size)),
            "hier_violation_rate_fully_decisive": hv_rate,                 # metric 3
            "n_fully_decisive_rows": int(fully.sum()),
            "per_label_decisive_rate": float(dec_pl.mean()),               # metric 4 (13 x N pairs)
            "fully_decisive_rate": float(fully.mean()),                    # metric 5 (per row)
            "empty_label_set_rate": float((sz == 0).mean()),
            "ambiguous_label_set_rate": float((sz == 2).mean()),
            "decisive_accuracy_fully_decisive": dec_acc,
            "rows_with_empty_joint_set": int((joint_size == 0).sum()),
            "cluster_joint_size_mean_info": cluster_joint_mean,
            "clusters": cluster_info,
        }
        out["rungs"][r] = rec
        rung_rows.append({k: rec[k] for k in ("rung", "K", "N_rows", "N_subjects", "joint_coverage",
                                              "mean_joint_set_size", "hier_violation_rate_fully_decisive",
                                              "per_label_decisive_rate", "fully_decisive_rate",
                                              "empty_label_set_rate", "ambiguous_label_set_rate",
                                              "decisive_accuracy_fully_decisive", "n_fully_decisive_rows",
                                              "rows_with_empty_joint_set")})

        if r == "E":
            qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
            out["rung_E_retained_set_size_distribution"] = {
                "joint_size_quantiles": {str(q): float(np.quantile(joint_size, q)) for q in qs},
                "count_joint_size_0": int((joint_size == 0).sum()),
                "count_joint_size_1": int((joint_size == 1).sum()),
                "count_joint_size_eq_2pow13": int((joint_size == 2 ** 13).sum()),
                "per_label_set_size_counts": {
                    lab: {"size0": int((sz[:, j] == 0).sum()), "size1": int((sz[:, j] == 1).sum()),
                          "size2": int((sz[:, j] == 2).sum())}
                    for j, lab in enumerate(ALL13)},
            }

    rung_df = pd.DataFrame(rung_rows)
    rung_df.to_csv(os.path.join(HERE, "rungs_summary.csv"), index=False)
    print("\n" + rung_df.to_string(index=False))

    # -------------------------------------------------------------- triggers
    tr = {}
    tr["T1_sanity_fail"] = not out["sanity"]["pass"]
    tr["T2_BCD_joint_cov_ge_0.89"] = [r for r in "BCD" if out["rungs"][r]["joint_coverage"] >= 0.89]
    jc = [out["rungs"][r]["joint_coverage"] for r in "BCD"]
    dr = [out["rungs"][r]["per_label_decisive_rate"] for r in "BCD"]
    fd = [out["rungs"][r]["fully_decisive_rate"] for r in "BCD"]
    tr["T3_BCD_joint_cov_direction"] = [jc[1] - jc[0], jc[2] - jc[1]]
    tr["T3_BCD_decisive_direction"] = [dr[1] - dr[0], dr[2] - dr[1]]
    tr["T3_BCD_fully_decisive_direction"] = [fd[1] - fd[0], fd[2] - fd[1]]
    e = out["rungs"]["E"]
    tr["T4_E_empty_rows"] = e["rows_with_empty_joint_set"]
    out["triggers"] = tr
    print("\nTRIGGERS:", json.dumps(jsonable(tr), indent=1))

    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(jsonable(out), f, indent=2)
    print("\nwrote metrics.json")


if __name__ == "__main__":
    main()
