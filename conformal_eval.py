"""Joint-region evaluation helpers extracted from Exp055 (`experiments/055_learned_score_full/run.py`).

Exp055's run.py is left unmodified; these are verbatim-logic copies with hard-coded constants
turned into arguments (Exp055's values are the defaults). Regions are boolean arrays over
joint states in `conformal_common.enumerate_states` order, shape (n, 2^m).

Population / Gate 0 : all13_clean, hier_violations, n_edges_of
Region builders     : subset, product_region, threshold_eval, refs_eval
Metrics             : metric_row
Resplit CIs         : summarize, paired
Provenance          : sha256, package_versions
"""

from __future__ import annotations

import hashlib
import platform

import numpy as np
import pandas as pd

import conformal_common as cc
from conformal_methods import bonferrony as bf
from conformal_methods import max_rank as mr
from data.hierarchy import CHEXPERT_HIERARCHY

ALPHA = 0.10
COV_LO, COV_HI = 0.895, 0.905
METRICS = ["joint_coverage", "cal_side_coverage", "mean_joint_set_size", "median_joint_set_size",
           "log2_over_m_mean", "empty_set_rate", "fully_decisive_rate", "correct_decisive_rate",
           "decisive_accuracy", "hierarchy_violation_rate_decisive", "threshold"]
PAIR_METRICS = ["mean_joint_set_size", "log2_over_m_mean", "correct_decisive_rate", "joint_coverage",
                "median_joint_set_size"]


def sha256(path):
    """Hex sha256 of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions():
    """Versions of python/numpy/pandas/torch/pyyaml, for run_info.json."""
    import torch
    import yaml
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "pyyaml": yaml.__version__}


def all13_clean(df):
    """Rows with every one of the 13 true labels in {0,1} and every prob finite."""
    y = df[[f"{lab}_true_label" for lab in cc.ALL_LABELS]].to_numpy(dtype=float)
    p = df[[f"{lab}_prob" for lab in cc.ALL_LABELS]].to_numpy(dtype=float)
    keep = np.all(np.isin(y, [0.0, 1.0]), axis=1) & np.all(np.isfinite(p), axis=1)
    return df.loc[keep].reset_index(drop=True)


def hier_violations(df, labels):
    """(violating vectors, rows checked) for rows whose cluster members are all in {0,1}."""
    y = df[[f"{lab}_true_label" for lab in labels]].to_numpy(dtype=float)
    ok = np.all(np.isin(y, [0.0, 1.0]), axis=1)
    v = y[ok].astype(int)
    return int((~cc.hierarchy_admissible(labels, v)).sum()), int(ok.sum())


def n_edges_of(labels):
    """Directed parent->child pairs (CHEXPERT_HIERARCHY) with both endpoints in labels.

    This is the number of separate constraint terms hierarchy_admissible applies.
    """
    L = set(labels)
    return int(sum(1 for ch, ps in CHEXPERT_HIERARCHY.items() if ch in L for p in ps if p in L))


def subset(g, idx):
    """GroupData restricted to row indices idx."""
    return cc.GroupData(labels=g.labels, probs=g.probs[idx], y=g.y[idx], scores=g.scores[idx],
                        n_rows_before_filter=int(len(idx)),
                        subject_id=None if g.subject_id is None else g.subject_id[idx])


def product_region(group, thr):
    """Product of per-label prediction sets, as a region over cc-ordered joint states (n, 2^m)."""
    sets = cc.prediction_sets(group, thr)              # (n, m, 2)
    states = cc.enumerate_states(group.m)              # (2^m, m)
    reg = np.ones((group.n, states.shape[0]), dtype=bool)
    for j in range(group.m):
        reg &= sets[:, j, :][:, states[:, j]]
    return reg


def threshold_eval(raw, smask, adm, y_cc, idxA, idxB, alpha=ALPHA):
    """Split-conformal threshold over joint-state scores; guarded and unguarded regions.

    raw   (N, S) raw score, cc order;  smask = raw with +inf on inadmissible states.
    Threshold: k = ceil((n+1)(1-alpha))-th smallest calibration score of the TRUE state
    (conformal_rank / order_statistic, no interpolation). Admissibility is masked explicitly
    so an infinite threshold cannot re-admit inadmissible states.
    Returns {guard: (regA, regB, thr, k, n_bad)} for guard in (True, False).
    """
    s_true = smask[idxA, y_cc[idxA]]
    n_bad = int(np.count_nonzero(~adm[y_cc[idxA]]))
    assert n_bad == 0, "inadmissible true state in calibration half: Gate 0 violated"
    k = cc.conformal_rank(idxA.size, alpha)
    thr = cc.order_statistic(s_true, k)
    out = {}
    for guard in (True, False):
        regA = adm[None, :] & (raw[idxA] <= thr)
        regB = adm[None, :] & (raw[idxB] <= thr)
        if guard:
            regA[np.arange(idxA.size), smask[idxA].argmin(axis=1)] = True
            regB[np.arange(idxB.size), smask[idxB].argmin(axis=1)] = True
        out[guard] = (regA, regB, float(thr), int(k), n_bad)
    return out


def refs_eval(gA, gB, adm, alpha=ALPHA):
    """Classical references, each as a region over cc states on A (calibration) and B (test).

    indep_per_label : alpha per label, product
    bonferroni      : alpha/m per label, product            (bonferrony.calibrate_per_label)
    sidak           : 1-(1-alpha)^(1/m) per label, product
    maxrank         : max_rank.calibrate_max_rank (Timans Alg. 2), product
    maxrank_adm     : maxrank product intersected with hierarchy-admissible states
    bonferroni_adm  : bonferroni product intersected with hierarchy-admissible states
    All thresholds use k = ceil((n+1)(1-alpha')) via conformal_rank (one convention).
    Returns {name: (regA, regB, k)}.
    """
    thr_i, di = bf.calibrate_per_label(gA, alpha, "none")
    thr_b, db = bf.calibrate_per_label(gA, alpha, "bonferroni")
    thr_s, ds = bf.calibrate_per_label(gA, alpha, "sidak")
    thr_m, dm = mr.calibrate_max_rank(gA, alpha)
    for d in (di, db, ds, dm):
        assert not d.get("degenerate", False), "degenerate reference threshold"
    out = {}
    out["indep_per_label"] = (product_region(gA, thr_i), product_region(gB, thr_i), int(di["k"]))
    out["bonferroni"] = (product_region(gA, thr_b), product_region(gB, thr_b), int(db["k"]))
    out["sidak"] = (product_region(gA, thr_s), product_region(gB, thr_s), int(ds["k"]))
    out["maxrank"] = (product_region(gA, thr_m), product_region(gB, thr_m), int(dm["k_star"]))
    rA, rB, k = out["maxrank"]
    out["maxrank_adm"] = (rA & adm[None, :], rB & adm[None, :], k)
    rA, rB, k = out["bonferroni"]
    out["bonferroni_adm"] = (rA & adm[None, :], rB & adm[None, :], k)
    return out


def metric_row(base, gA, gB, regA, regB):
    """Locked metric set on test half B, plus calibration-side coverage on A. `base` = dict of id columns."""
    m = gB.m
    met = cc.evaluate_joint(gB, regB)
    sizes = regB.sum(axis=1)
    ne = sizes >= 1
    log2m = float(np.mean(np.log2(sizes[ne])) / m) if ne.any() else float("nan")
    true_b = cc.state_indices(gB.y)
    decisive = sizes == 1
    chosen = regB.argmax(axis=1)
    cdr = float((decisive & (chosen == true_b)).mean())
    if decisive.any():
        assert abs(cdr - met["fully_decisive_rate"] * met["decisive_accuracy"]) < 1e-12
    cal_cov = float(regA[np.arange(gA.n), cc.state_indices(gA.y)].mean())
    return dict(base,
                n_cal=int(gA.n), n_test=int(gB.n),
                joint_coverage=met["joint_coverage"],
                cal_side_coverage=cal_cov,
                mean_joint_set_size=met["mean_joint_set_size"],
                median_joint_set_size=met["median_joint_set_size"],
                log2_over_m_mean=log2m,
                empty_set_rate=met["empty_set_rate"],
                fully_decisive_rate=met["fully_decisive_rate"],
                correct_decisive_rate=cdr,
                decisive_accuracy=met["decisive_accuracy"],
                hierarchy_violation_rate_decisive=met["hierarchy_violation_rate_decisive"])


def summarize(df, metrics=METRICS, cov_lo=COV_LO, cov_hi=COV_HI):
    """Per-arm resplit summary (mean, sd, 2.5/97.5 percentiles, frozen-draw value and percentile rank).

    df: long frame with columns family, unit, parent, draw_id, m, n_edges, n_admissible, labels,
    variant, seed, resplit (>=0 resplits, -1 frozen draw) and the metric columns.
    Adds a coverage gate on the resplit-mean joint coverage.
    """
    key = ["family", "unit", "parent", "draw_id", "m", "n_edges", "n_admissible", "labels", "variant", "seed"]
    out = []
    for keyvals, g in df.groupby(key, sort=False):
        rs = g[g.resplit >= 0].sort_values("resplit")
        fr = g[g.resplit == -1]
        rec = dict(zip(key, keyvals))
        rec["n_resplits"] = int(rs.shape[0])
        for mt in metrics:
            v = rs[mt].to_numpy(dtype=float)
            f = float(fr[mt].iloc[0]) if len(fr) else np.nan
            rec[f"{mt}_mean"] = float(np.nanmean(v))
            rec[f"{mt}_sd"] = float(np.nanstd(v, ddof=1))
            rec[f"{mt}_p2.5"] = float(np.nanpercentile(v, 2.5))
            rec[f"{mt}_p97.5"] = float(np.nanpercentile(v, 97.5))
            rec[f"{mt}_frozen"] = f
            rec[f"{mt}_frozen_pct_le"] = float(np.mean(v <= f)) if np.isfinite(f) else np.nan
        cov = rec["joint_coverage_mean"]
        rec["coverage_gate"] = "pass" if (cov_lo <= cov <= cov_hi) else "coverage-unequal"
        out.append(rec)
    return pd.DataFrame(out)


def paired(rs, pairs, pair_metrics=PAIR_METRICS, n_resplits=100):
    """Paired-within-resplit differences with percentile CIs.

    pairs: list of (unit, family, (varA, seedA), (varB, seedB)); rs: resplit rows (resplit >= 0).
    """
    out = []
    for mt in pair_metrics:
        wide = rs.pivot_table(index="resplit", columns=["unit", "variant", "seed"], values=mt, aggfunc="first")
        for unit, fam, (va, sa), (vb, sb) in pairs:
            a = wide[(unit, va, sa)].to_numpy(dtype=float)
            b = wide[(unit, vb, sb)].to_numpy(dtype=float)
            assert len(a) == n_resplits and len(b) == n_resplits
            d = a - b
            lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            out.append(dict(unit=unit, family=fam, arm=va, seed_arm=sa, comparator=vb, seed_comparator=sb,
                            metric=mt, n_resplits=len(d), mean_diff=float(d.mean()),
                            ci_lo_2p5=float(lo), ci_hi_97p5=float(hi),
                            ci_excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(out)
