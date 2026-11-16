"""Exp052 - cluster-granularity curve, 3-method comparison, Rungs A-E (13 -> 1).

Pre-registered in experiments/052_cluster_granularity_curve_3method/README.md before this file was run.
Methods:
  - max-rank           CITED from experiments/051_cluster_granularity_curve (not rerun).
  - bonferroni, sidak  bonferrony.calibrate_per_label(..., correction=...) per cluster.
  - LP-SCP (penalised, mu=nu=1, no hierarchy enforcement), validation -> proper-training,
    conformal -> calibration, test -> evaluation. Both non-empty-guard variants reported.
No pipeline file is edited. The LP-SCP scoring is the same formula as lp_scp.state_scores,
computed in row chunks (see README, Compute). Equivalence to lp_scp.py is asserted below.

Seed 0 (same convention as Exp051). No stochastic step.

Run:
    cd /Users/alonfines/Desktop/x-ray && .venv_exp048/bin/python experiments/052_cluster_granularity_curve_3method/run.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import conformal_common as cc            # noqa: E402
import bonferrony                        # noqa: E402
import lp_scp                            # noqa: E402

SEED = 0
ALPHA = 0.10
np.random.seed(SEED)  # no stochastic step; recorded for reproducibility only

ALL13 = cc.ALL_LABELS
assert len(ALL13) == 13

CARDIO = ["Enlarged Cardiomediastinum", "Cardiomegaly"]
LUNG = ["Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis"]
PLEURAL = ["Pleural Effusion", "Pleural Other"]

# Pre-registered ladder, verbatim from Exp051 README / run.py.
RUNGS = {
    "A": [[l] for l in ALL13],
    "B": [CARDIO] + [[l] for l in ALL13 if l not in CARDIO],
    "C": [CARDIO, LUNG] + [[l] for l in ["Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]],
    "D": [CARDIO, LUNG, PLEURAL] + [[l] for l in ["Pneumothorax", "Fracture", "Support Devices"]],
    "E": [ALL13],
}
for r, clusters in RUNGS.items():
    flat = [l for c in clusters for l in c]
    assert sorted(flat) == sorted(ALL13), f"rung {r} is not a partition"


def _is_coarsening(fine, coarse):
    coarse_sets = [set(c) for c in coarse]
    return all(any(set(f) <= cs for cs in coarse_sets) for f in fine)


assert _is_coarsening(RUNGS["A"], RUNGS["B"]) and _is_coarsening(RUNGS["B"], RUNGS["C"])
assert _is_coarsening(RUNGS["C"], RUNGS["D"]) and _is_coarsening(RUNGS["D"], RUNGS["E"])

EXP051 = os.path.join(REPO, "experiments", "051_cluster_granularity_curve")
N_TEST_EXPECTED = 18877          # 051 gates
SUBJ_TEST_EXPECTED = 5932        # 051 gates
N_FIT_EXPECTED = 18570           # validation, zero uncertain rows (checked 2026-11-01)
N_CAL_EXPECTED = 18754
CHUNK = 512
MU = lp_scp.MU_PRIMARY           # 1.0, fixed a priori
NU = lp_scp.NU_PRIMARY           # 1.0
COND_LIMIT = lp_scp.COND_LIMIT   # 1e10

TOL_THR = 1e-12                  # Rung A threshold / metric tolerance (spec T-B)
TOL_EQUIV = 1e-10                # chunked-vs-lp_scp score tolerance (T-E)


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


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------- LP-SCP helpers
def true_state_scores_direct(probs, y, struct, mu, nu):
    """Score of each row's own (true) joint state. Same formula as lp_scp.true_state_scores."""
    r = np.abs(y - probs)
    quad = np.einsum("ni,ij,nj->n", r, struct.sigma_inv, r)
    idx = cc.state_indices(y)
    return np.sqrt(np.maximum(quad, 0.0)) + mu * struct.hp[idx] + nu * struct.cp[idx]


def regions_chunked(probs, struct, thr, mu, nu):
    """(n, 2^m) bool region masks: guard-off (score <= thr) and guard-on (plus argmin state).

    Same formula as lp_scp.state_scores + lp_scp.build_region, computed in row chunks so
    the (n, 2^m, m) intermediate never materialises for the full split.
    """
    states = cc.enumerate_states(probs.shape[1])
    n, S = probs.shape[0], states.shape[0]
    off = np.empty((n, S), dtype=bool)
    on = np.empty((n, S), dtype=bool)
    for a in range(0, n, CHUNK):
        p = probs[a:a + CHUNK]
        r = np.abs(states[None, :, :] - p[:, None, :])
        quad = np.einsum("nsi,ij,nsj->ns", r, struct.sigma_inv, r)
        sc = np.sqrt(np.maximum(quad, 0.0)) + mu * struct.hp[None, :] + nu * struct.cp[None, :]
        reg = sc <= thr
        g = reg.copy()
        g[np.arange(sc.shape[0]), sc.argmin(axis=1)] = True
        off[a:a + CHUNK] = reg
        on[a:a + CHUNK] = g
    return off, on


def lp_cluster(cl, val, cal, test_group):
    """Fit, calibrate and test LP-SCP (penalised arm) for one cluster. Returns a dict."""
    cl = list(cl)
    m = len(cl)
    fit = cc.extract_group(val, cl)                 # validation -> proper-training
    cg = cc.extract_group(cal, cl)                  # conformal -> calibration
    struct = lp_scp.fit_structure(fit)

    # Sigma diagnostics: pre-ridge cond (recomputed exactly as fit_structure does), post-ridge cond.
    r = np.abs(fit.y - fit.probs)
    cov = np.atleast_2d(np.cov(r, rowvar=False))
    cond_pre = float(np.linalg.cond(cov))
    ridge = float(struct.diagnostics["sigma_ridge"])
    cond_post = float(np.linalg.cond(cov + ridge * np.eye(m))) if ridge > 0 else cond_pre
    sigma_trigger = (not np.isfinite(cond_pre)) or (ridge > 0 and cond_post > COND_LIMIT)

    # Calibration scores: direct formula, verified against lp_scp.true_state_scores.
    s_cal = true_state_scores_direct(cg.probs, cg.y, struct, MU, NU)
    if m <= 6:
        ref = lp_scp.true_state_scores(cg, struct, MU, NU)
        cal_diff = float(np.max(np.abs(ref - s_cal)))
        cal_check = "full_split"
    else:
        sub = cg.probs[:300], cg.y[:300]
        sub_g = cc.GroupData(labels=cl, probs=sub[0], y=sub[1], scores=cg.scores[:300],
                             n_rows_before_filter=300)
        ref = lp_scp.true_state_scores(sub_g, struct, MU, NU)
        cal_diff = float(np.max(np.abs(ref - s_cal[:300])))
        cal_check = "first_300_rows"
    assert cal_diff <= TOL_EQUIV, f"cal score equivalence failed for {cl}: {cal_diff}"

    k = cc.conformal_rank(cg.n, ALPHA)
    thr = cc.order_statistic(s_cal, k)

    # Test regions over the cluster's states, 13-clean test rows.
    pos = [ALL13.index(l) for l in cl]
    tp = test_group.probs[:, pos]
    ty = test_group.y[:, pos]
    off, on = regions_chunked(tp, struct, thr, MU, NU)

    # Region equivalence vs lp_scp.build_region on a 300-row test subset (exact mask equality).
    sub_g = cc.GroupData(labels=cl, probs=tp[:300], y=ty[:300],
                         scores=test_group.scores[:300][:, pos], n_rows_before_filter=300)
    ref_off = lp_scp.build_region(sub_g, struct, thr, MU, NU, False, False)
    ref_on = lp_scp.build_region(sub_g, struct, thr, MU, NU, False, True)
    assert np.array_equal(ref_off, off[:300]), f"region (guard off) mismatch for {cl}"
    assert np.array_equal(ref_on, on[:300]), f"region (guard on) mismatch for {cl}"

    states = cc.enumerate_states(m)
    tidx = cc.state_indices(ty)
    within_cov_off = float(off[np.arange(ty.shape[0]), tidx].mean())
    within_cov_on = float(on[np.arange(ty.shape[0]), tidx].mean())
    return {
        "labels": cl, "m": m, "n_fit": fit.n, "n_cal": cg.n, "k": k, "threshold": float(thr),
        "sigma_cond_pre_ridge": cond_pre, "sigma_ridge": ridge, "sigma_cond_post_ridge": cond_post,
        "sigma_trigger": bool(sigma_trigger),
        "n_observed_label_sets_fit": int(struct.diagnostics["n_observed_label_sets"]),
        "n_states": int(2 ** m), "unobserved_states_hp_gt0": int(struct.diagnostics["unobserved_states"]),
        "cal_score_check": cal_check, "cal_score_max_abs_diff": cal_diff,
        "within_cluster_joint_coverage_guard_off": within_cov_off,
        "within_cluster_joint_coverage_guard_on": within_cov_on,
        "mean_region_size_guard_off": float(off.sum(1).mean()),
        "mean_region_size_guard_on": float(on.sum(1).mean()),
        "_off": off, "_on": on, "_states": states,
    }


def lp_rung_eval(clusters, cache, which, N, y_all):
    """Project per-cluster LP-SCP regions onto labels; joint metrics on the 13-label population."""
    sz = np.zeros((N, 13), dtype=int)
    vec = np.zeros((N, 13), dtype=int)
    covered = np.ones(N, dtype=bool)
    joint = np.ones(N, dtype=float)
    for cl in clusters:
        R = cache[tuple(cl)]
        reg = R[f"_{which}"]
        states = R["_states"]
        pos = [ALL13.index(l) for l in cl]
        ty = y_all[:, pos]
        tidx = cc.state_indices(ty)
        covered &= reg[np.arange(N), tidx]
        joint *= reg.sum(1).astype(float)
        for jpos, p in enumerate(pos):
            b1 = states[:, jpos] == 1
            has1 = reg[:, b1].any(axis=1) if b1.any() else np.zeros(N, bool)
            has0 = reg[:, ~b1].any(axis=1) if (~b1).any() else np.zeros(N, bool)
            sz[:, p] = has0.astype(int) + has1.astype(int)
            vec[:, p] = has1.astype(int)
    return sz, vec, covered, joint


def per_label_eval(test_group, thr_all):
    sets = cc.prediction_sets(test_group, thr_all)           # (N, 13, 2)
    sz = sets.sum(axis=2)
    y = test_group.y
    true_kept = np.take_along_axis(sets, y[:, :, None], axis=2)[:, :, 0]
    covered = true_kept.all(axis=1)
    joint = sz.prod(axis=1).astype(float)
    vec = sets[:, :, 1].astype(int)
    return sz, vec, covered, joint


def joint_metrics(y, sz, vec, covered, joint):
    """Metric block identical in definition to Exp051 run.py, applied to a shared projection."""
    dec_pl = sz == 1
    fully = dec_pl.all(axis=1)
    if fully.any():
        hv = float((~cc.hierarchy_admissible(ALL13, vec[fully])).mean())
        dec_acc = float((vec[fully] == y[fully]).all(axis=1).mean())
    else:
        hv, dec_acc = float("nan"), float("nan")
    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    per_label_sizes = {lab: {"size0": int((sz[:, j] == 0).sum()), "size1": int((sz[:, j] == 1).sum()),
                             "size2": int((sz[:, j] == 2).sum())} for j, lab in enumerate(ALL13)}
    return {
        "N_rows": int(sz.shape[0]),
        "joint_coverage": float(covered.mean()),                                  # metric 1
        "mean_joint_set_size": float(joint.mean()),                               # metric 2
        "median_joint_set_size": float(np.median(joint)),
        "hier_violation_rate_fully_decisive": hv,                                 # metric 3
        "n_fully_decisive_rows": int(fully.sum()),
        "per_label_decisive_rate": float(dec_pl.mean()),                          # metric 4
        "fully_decisive_rate": float(fully.mean()),                               # metric 5
        "empty_label_set_rate": float((sz == 0).mean()),
        "ambiguous_label_set_rate": float((sz == 2).mean()),
        "decisive_accuracy_fully_decisive": dec_acc,
        "rows_with_empty_joint_set": int((joint == 0).sum()),
        "joint_size_quantiles": {str(q): float(np.quantile(joint, q)) for q in qs},
        "joint_size_count_eq0": int((joint == 0).sum()),
        "joint_size_count_eq1": int((joint == 1).sum()),
        "joint_size_count_eq_2pow13": int((joint == 2 ** 13).sum()),
        "per_label_set_size_counts": per_label_sizes,
    }


def main() -> None:
    t_start = time.time()
    out = {"experiment": "052_cluster_granularity_curve_3method", "seed": SEED, "alpha": ALPHA,
           "mu": MU, "nu": NU, "cond_limit": COND_LIMIT, "gates": {}, "rung_A_checks": {},
           "rungs": {}, "gap_sidak_minus_bonferroni": {}, "lp_cluster_diagnostics": {},
           "triggers": {}, "timing_sec": {}}

    # ------------------------------------------------------------- Gate 0
    splits = cc.load_splits(with_validation=True, run_gate0=True)     # raises on any overlap
    overlap = cc.gate0_check(splits, raise_on_overlap=True)
    counts = {
        "conformal_rows": len(splits.calibration), "test_rows": len(splits.evaluation),
        "validation_rows": len(splits.validation),
        "conformal_subjects": int(splits.calibration.subject_id.nunique()),
        "test_subjects": int(splits.evaluation.subject_id.nunique()),
        "validation_subjects": int(splits.validation.subject_id.nunique()),
    }
    assert counts["conformal_rows"] == 18754 and counts["test_rows"] == 24394 and counts["validation_rows"] == 18570
    assert counts["conformal_subjects"] == 5926 and counts["test_subjects"] == 6395 and counts["validation_subjects"] == 5898
    assert all(v == 0 for v in overlap.values()), overlap
    out["gates"]["gate0_overlap"] = overlap
    out["gates"]["gate0_counts"] = counts
    print("GATE 0 pass:", overlap, counts)

    # ------------------------------------------------- Gate 0.5 (reused from 051, not recomputed)
    g05 = pd.read_csv(os.path.join(EXP051, "gate05_positive_counts.csv"))
    assert len(g05) == 13
    out["gates"]["gate05_reused_from_051"] = {
        "file": "experiments/051_cluster_granularity_curve/gate05_positive_counts.csv",
        "sha256": sha256(os.path.join(EXP051, "gate05_positive_counts.csv")),
        "flagged_labels": [r for r, f in zip(g05["label"], g05["flag_under_150"]) if f],
        "min_positive_label": str(g05.loc[g05["conformal_positive_rows"].idxmin(), "label"]),
        "min_positive_rows": int(g05["conformal_positive_rows"].min()),
    }
    assert not out["gates"]["gate05_reused_from_051"]["flagged_labels"]

    # ----------------------------------------------- cited Exp051 max-rank numbers
    c051 = json.load(open(os.path.join(EXP051, "metrics.json")))
    out["gates"]["test_N_rows_051"] = c051["gates"]["test_N_rows_after_13label_exclusion"]
    out["gates"]["test_N_subjects_051"] = c051["gates"]["test_N_subjects_after_13label_exclusion"]
    out["maxrank_cited_from_051"] = {
        "file": "experiments/051_cluster_granularity_curve/metrics.json",
        "sha256": sha256(os.path.join(EXP051, "metrics.json")),
        "rungs": {r: {k: v for k, v in c051["rungs"][r].items() if k != "clusters"} for r in RUNGS},
        "rung_A_thresholds": {c["cluster"]: c["thresholds"] for c in c051["rungs"]["A"]["clusters"]},
    }

    # ------------------------------------------------------------- populations
    test_group = cc.extract_group(splits.evaluation, ALL13)           # 13-label exclusion, all methods
    N = test_group.n
    subj_test = int(len(np.unique(test_group.subject_id)))
    assert N == N_TEST_EXPECTED and subj_test == SUBJ_TEST_EXPECTED, (N, subj_test)
    assert N == out["gates"]["test_N_rows_051"]
    y_all = test_group.y
    out["gates"]["test_N_rows_052"] = N
    out["gates"]["test_N_subjects_052"] = subj_test
    out["gates"]["test_rows_dropped_13label"] = int(test_group.n_rows_before_filter - N)
    print(f"Test population (shared): rows={N} subjects={subj_test}")

    # ---------------------------------------------- Bonferroni / Sidak (Rung A checks + rung calibration)
    bonf_cache = {}      # (correction, cluster tuple) -> (thresholds dict, diagnostics)

    def bonf_thr(corr, cl):
        key = (corr, tuple(cl))
        if key not in bonf_cache:
            grp = cc.extract_group(splits.calibration, cl)
            bonf_cache[key] = bonferrony.calibrate_per_label(grp, ALPHA, corr)
        return bonf_cache[key]

    cited_A_thr = out["maxrank_cited_from_051"]["rung_A_thresholds"]
    for corr in ("bonferroni", "sidak"):
        diffs = []
        for lab in ALL13:
            thr, _ = bonf_thr(corr, [lab])
            diffs.append(abs(thr[lab] - cited_A_thr[lab][lab]))
        max_abs = float(max(diffs))
        out["rung_A_checks"][corr] = {"n_thresholds_compared": len(diffs), "max_abs_diff_vs_051_maxrank": max_abs,
                                      "tolerance": TOL_THR, "pass": bool(max_abs <= TOL_THR)}
        print(f"RUNG A threshold check {corr} vs 051 max-rank: max|d|={max_abs:.3e}")

    # ----------------------------------------------- per-rung evaluation: bonferroni, sidak
    for corr, name in (("bonferroni", "bonferroni"), ("sidak", "sidak")):
        out["rungs"].setdefault(name, {})
        for r, clusters in RUNGS.items():
            thr_all = {}
            info = []
            for cl in clusters:
                thr, diag = bonf_thr(corr, cl)
                thr_all.update(thr)
                info.append({"cluster": "|".join(cl), "m": len(cl), "alpha_effective": diag["alpha_effective"],
                             "k": diag["k"], "n_calibration": diag["n_calibration"],
                             "degenerate": diag["degenerate"], "thresholds": {l: thr[l] for l in cl}})
            assert set(thr_all) == set(ALL13)
            sz, vec, cov, jt = per_label_eval(test_group, thr_all)
            met = joint_metrics(y_all, sz, vec, cov, jt)
            met["N_fit_per_cluster"] = None
            met["clusters"] = info
            out["rungs"][name][r] = met

    # Rung A: Bonferroni/Sidak metrics must equal 051 cited Rung A metrics (identical thresholds).
    for name in ("bonferroni", "sidak"):
        mA = out["rungs"][name]["A"]
        cA = out["maxrank_cited_from_051"]["rungs"]["A"]
        keys = ["joint_coverage", "mean_joint_set_size", "hier_violation_rate_fully_decisive",
                "per_label_decisive_rate", "fully_decisive_rate", "empty_label_set_rate",
                "ambiguous_label_set_rate", "decisive_accuracy_fully_decisive"]
        dmax = 0.0
        for k_ in keys:
            a, b = mA[k_], cA[k_]
            if isinstance(a, float) and math.isnan(a) and math.isnan(b):
                continue
            dmax = max(dmax, abs(a - b))
        cnt_ok = mA["n_fully_decisive_rows"] == cA["n_fully_decisive_rows"]
        out["rung_A_checks"][name]["metric_max_abs_diff_vs_051"] = float(dmax)
        out["rung_A_checks"][name]["fully_decisive_count_matches_051"] = bool(cnt_ok)
        out["rung_A_checks"][name]["metric_pass"] = bool(dmax <= TOL_THR and cnt_ok)
        print(f"RUNG A metric check {name} vs 051: max|d|={dmax:.3e} count_match={cnt_ok}")

    # Šidák - Bonferroni gap, per rung
    for r in RUNGS:
        b = out["rungs"]["bonferroni"][r]
        s = out["rungs"]["sidak"][r]
        out["gap_sidak_minus_bonferroni"][r] = {
            "joint_coverage_gap": s["joint_coverage"] - b["joint_coverage"],
            "mean_joint_set_size_gap": s["mean_joint_set_size"] - b["mean_joint_set_size"],
            "per_label_decisive_rate_gap": s["per_label_decisive_rate"] - b["per_label_decisive_rate"],
            "fully_decisive_rate_gap": s["fully_decisive_rate"] - b["fully_decisive_rate"],
        }

    # ------------------------------------------------------------- LP-SCP
    t_lp = time.time()
    lp_cache = {}
    val = splits.validation
    cal = splits.calibration
    try:
        unique_clusters = []
        for r, clusters in RUNGS.items():
            for cl in clusters:
                if tuple(cl) not in unique_clusters:
                    unique_clusters.append(tuple(cl))
        for cl in unique_clusters:
            lp_cache[cl] = lp_cluster(cl, val, cal, test_group)
            d = lp_cache[cl]
            out["lp_cluster_diagnostics"]["|".join(cl)] = {k_: v for k_, v in d.items() if not k_.startswith("_")}
            print(f"LP-SCP {'|'.join(cl)} m={len(cl)}: n_fit={d['n_fit']} n_cal={d['n_cal']} k={d['k']} "
                  f"thr={d['threshold']:.6g} cond_pre={d['sigma_cond_pre_ridge']:.3g} ridge={d['sigma_ridge']:.3g} "
                  f"within_cov_off={d['within_cluster_joint_coverage_guard_off']:.4f}")
    except Exception as e:  # T-A: crash / memory at any cluster
        out["triggers"]["T-A_LP_execution"] = {"fired": True, "error": repr(e)}
        out["timing_sec"]["lp_scp_stage"] = time.time() - t_lp
        with open(os.path.join(HERE, "metrics.json"), "w") as f:
            json.dump(jsonable(out), f, indent=2)
        raise
    out["timing_sec"]["lp_scp_stage"] = time.time() - t_lp

    for which, name in (("off", "lp_scp_guard_off"), ("on", "lp_scp_guard_on")):
        out["rungs"][name] = {}
        for r, clusters in RUNGS.items():
            sz, vec, cov, jt = lp_rung_eval(clusters, lp_cache, which, N, y_all)
            met = joint_metrics(y_all, sz, vec, cov, jt)
            fits = [lp_cache[tuple(c)]["n_fit"] for c in clusters]
            cals = [lp_cache[tuple(c)]["n_cal"] for c in clusters]
            met["N_fit_per_cluster_min"] = int(min(fits))
            met["N_fit_per_cluster_max"] = int(max(fits))
            met["N_cal_per_cluster_min"] = int(min(cals))
            met["N_cal_per_cluster_max"] = int(max(cals))
            met["clusters"] = [{"cluster": "|".join(c), "m": len(c), "n_fit": lp_cache[tuple(c)]["n_fit"],
                                "n_cal": lp_cache[tuple(c)]["n_cal"], "k": lp_cache[tuple(c)]["k"],
                                "threshold": lp_cache[tuple(c)]["threshold"],
                                "within_cluster_joint_coverage": lp_cache[tuple(c)][f"within_cluster_joint_coverage_{'guard_off' if which == 'off' else 'guard_on'}"],
                                "mean_region_size": lp_cache[tuple(c)][f"mean_region_size_{'guard_off' if which == 'off' else 'guard_on'}"]}
                               for c in clusters]
            out["rungs"][name][r] = met
            print(f"LP-SCP {name} rung {r}: cov={met['joint_coverage']:.4f} size={met['mean_joint_set_size']:.3f} "
                  f"hv={met['hier_violation_rate_fully_decisive']} dec={met['per_label_decisive_rate']:.4f} "
                  f"full={met['fully_decisive_rate']:.4f} empty_rows={met['rows_with_empty_joint_set']}")

    # ------------------------------------------------------------- triggers
    tr = {}
    tr["T-B_rungA_match"] = {"fired": not (out["rung_A_checks"]["bonferroni"]["pass"] and out["rung_A_checks"]["sidak"]["pass"]
                                           and out["rung_A_checks"]["bonferroni"]["metric_pass"] and out["rung_A_checks"]["sidak"]["metric_pass"]),
                             "detail": {c: out["rung_A_checks"][c] for c in ("bonferroni", "sidak")}}
    sig_fired = [k_ for k_, d in out["lp_cluster_diagnostics"].items() if d["sigma_trigger"]]
    tr["T-C_sigma_conditioning"] = {"fired": bool(sig_fired), "clusters": sig_fired,
                                    "ridge_applied_clusters": [k_ for k_, d in out["lp_cluster_diagnostics"].items() if d["sigma_ridge"] > 0]}
    fit_drops = {k_: d["n_fit"] for k_, d in out["lp_cluster_diagnostics"].items() if abs(d["n_fit"] - N_FIT_EXPECTED) > 0.005 * N_FIT_EXPECTED}
    tr["T-D_N"] = {"fired": bool(fit_drops) or N != N_TEST_EXPECTED, "fit_clusters_outside_0.5pct": fit_drops,
                   "test_N": N}
    tr["T-A_LP_execution"] = {"fired": False, "lp_stage_sec": out["timing_sec"]["lp_scp_stage"],
                              "wall_clock_limit_sec": 7200}
    tr["T-A_LP_execution"]["fired"] = out["timing_sec"]["lp_scp_stage"] > 7200
    tr["T-E_equivalence"] = {"fired": False, "note": "asserts passed; a failing assert would have stopped the run"}
    out["triggers"] = tr

    out["timing_sec"]["total"] = time.time() - t_start
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(jsonable(out), f, indent=2)

    # ---------------------------------------------------------- summary CSV
    rows = []
    for name in ["maxrank_cited_from_051", "bonferroni", "sidak", "lp_scp_guard_off", "lp_scp_guard_on"]:
        for r in RUNGS:
            if name == "maxrank_cited_from_051":
                m_ = out["maxrank_cited_from_051"]["rungs"][r]
            else:
                m_ = out["rungs"][name][r]
            rows.append({"method": name, "rung": r, "K": len(RUNGS[r]), "N_rows_test": m_["N_rows"] if "N_rows" in m_ else N_TEST_EXPECTED,
                         "joint_coverage": m_["joint_coverage"], "mean_joint_set_size": m_["mean_joint_set_size"],
                         "hier_violation_rate_fully_decisive": m_["hier_violation_rate_fully_decisive"],
                         "per_label_decisive_rate": m_["per_label_decisive_rate"],
                         "fully_decisive_rate": m_["fully_decisive_rate"],
                         "empty_label_set_rate": m_["empty_label_set_rate"],
                         "decisive_accuracy_fully_decisive": m_["decisive_accuracy_fully_decisive"],
                         "n_fully_decisive_rows": m_["n_fully_decisive_rows"],
                         "rows_with_empty_joint_set": m_["rows_with_empty_joint_set"]})
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "rungs_summary_3method.csv"), index=False)

    print("\nTRIGGERS:")
    print(json.dumps(jsonable(tr), indent=1)[:3000])
    print(f"\nwrote {os.path.join(HERE, 'metrics.json')}  total {out['timing_sec']['total']:.1f}s")


if __name__ == "__main__":
    main()
