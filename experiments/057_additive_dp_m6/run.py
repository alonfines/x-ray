#!/usr/bin/env python3
"""Exp057 - additive factorized-likelihood score on the P6 DP partition (SPEC.md §1-§8).

Steps: Gate 0 -> R1 (19 distinct fits via fit_cluster(return_net=True); restored-net CE check; cap
check) -> joint tables -> frozen draw + 100 resplits (Exp056 seeds) -> arms -> summaries, paired
deltas, ratios, readings. R2 is NOT re-run: r2_resplit_identity.csv (earlier run) stands. This script
cross-checks the references it recomputes against the Exp056 vector CSV (1e-9).

Reused unedited: conformal_common.py, conformal_methods/{max_rank,bonferrony}.py,
reports/partition_search/find_partitions.py (load, init, fit_cluster), experiments/056 run.py
(STEM, P_H, all13_clean, edges_across, product_region, subset, vector_summary, cond_cov).

Run: /usr/bin/python3 experiments/057_additive_dp_m6/run.py
"""
from __future__ import annotations

import importlib.util as ilu
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
sys.path.insert(0, REPO)

import conformal_common as cc  # noqa: E402
from conformal_methods import max_rank as mr  # noqa: E402
from conformal_methods import bonferrony as bf  # noqa: E402


def _load(name, path):
    spec = ilu.spec_from_file_location(name, path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fp = _load("find_partitions", os.path.join(REPO, "reports", "partition_search", "find_partitions.py"))
e56 = _load("exp056_run", os.path.join(REPO, "experiments", "056_three_cluster_union", "run.py"))

ALPHA = e56.ALPHA
LABELS = list(cc.ALL_LABELS)
N13 = 13
N_RESPLITS = 100
TOL_R1 = 1e-6
TOL_X = 1e-9
COV_LO, COV_HI = 0.895, 0.905
SEED = 0
RARE = ["Pleural Other", "Fracture", "Lung Lesion"]
PARTS = os.path.join(REPO, "reports", "partition_search", "partitions.json")
COSTS = os.path.join(REPO, "reports", "partition_search", "cluster_costs.csv")
EXP056_VEC = os.path.join(REPO, "experiments", "056_three_cluster_union", "resplit_metrics_vector.csv")


class Stop(Exception):
    pass


def canon(labs):
    return sorted(LABELS.index(l) for l in labs)


# ---------------------------------------------------------------- Gate 0
def gate0():
    splits = cc.load_splits(stem=e56.STEM, with_validation=True, with_train=False, run_gate0=True)
    overlap = cc.gate0_check(splits, raise_on_overlap=True)
    if not all(v == 0 for v in overlap.values()):
        raise Stop("gate0_overlap")
    val13 = e56.all13_clean(splits.validation)
    conf13 = e56.all13_clean(splits.calibration)
    test13 = e56.all13_clean(splits.evaluation)
    rows = dict(validation=len(val13), conformal=len(conf13), test=len(test13))
    if rows != dict(validation=18570, conformal=18754, test=18877):
        raise Stop(f"gate0_population {rows}")
    pool = pd.concat([conf13, test13], ignore_index=True)
    if (len(pool), int(pool.subject_id.nunique())) != (37631, 11858):
        raise Stop("gate0_pool")
    return dict(overlap=overlap, rows=rows, pool_rows=len(pool), pool_subjects=int(pool.subject_id.nunique()),
                n_conf=len(conf13), passed=True), pool


# ---------------------------------------------------------------- partitions
def partitions():
    p6 = json.load(open(PARTS))["partitions"]["6"]["clusters"]
    P6 = [(f"P6C{i+1}", canon(c)) for i, c in enumerate(p6)]
    PH = [(f"PHC{i+1}", canon(c)) for i, (_, c) in enumerate(e56.P_H)]
    SING = [(f"S_{LABELS[c]}", [c]) for c in range(N13)]
    return P6, PH, SING


def distinct_fits(parts):
    seen, out = set(), []
    for _, c in parts:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            out.append(list(c))
    return out


def joint_table(clusters, enforce=True):
    """Admissible joint states: cluster-state indices (J x C), 13-label vectors (J x 13), per-cluster admissible lists."""
    per = []
    for _, cols in clusters:
        st = cc.enumerate_states(len(cols))
        per.append(np.flatnonzero(cc.hierarchy_admissible([LABELS[c] for c in cols], st)))
    combos = np.array(list(itertools.product(*per)), dtype=np.int64)
    Y = np.zeros((len(combos), N13), dtype=int)
    for ci, (_, cols) in enumerate(clusters):
        Y[:, cols] = cc.enumerate_states(len(cols))[combos[:, ci]]
    if enforce and not cc.hierarchy_admissible(LABELS, Y).all():
        raise Stop("joint_table_inadmissible")
    return dict(combos=combos, Y=Y, per=per, dims=tuple(len(p) for p in per))


# ---------------------------------------------------------------- R1: retrain with return_net
def train_and_check(cost_ref):
    labels, x, y, is_ho = fp.load()
    fp.init((labels, x, y, is_ho))
    torch.set_num_threads(1)
    fits = distinct_fits(partitions()[0] + partitions()[1] + partitions()[2])
    rows, nets = [], {}
    for cols in fits:
        _, ce, best_ep, stop_ep, net = fp.fit_cluster(tuple(cols), return_net=True)
        m = len(cols)
        adm = torch.tensor(cc.hierarchy_admissible([labels[c] for c in cols], cc.enumerate_states(m)))
        neg = torch.where(adm, 0.0, float("-inf"))
        Xho = torch.tensor(x[is_ho][:, cols], dtype=torch.float32)
        yho = torch.tensor(cc.state_indices(y[is_ho][:, cols]))
        with torch.no_grad():
            ce_restored = float(torch.nn.functional.cross_entropy(net(Xho) + neg, yho).item())
        ref_ce, ref_ep = cost_ref[frozenset(cols)]
        rows.append(dict(cluster=" | ".join(labels[c] for c in cols), m=m, ce_fit=ce, ce_ref=ref_ce,
                         ce_restored_net=ce_restored, abs_fit_vs_ref=abs(ce - ref_ce),
                         abs_restored_vs_fit=abs(ce_restored - ce), epoch_fit=best_ep, epoch_ref=ref_ep,
                         stop_epoch=stop_ep, cap_hit_200=bool(stop_ep >= 200)))
        nets[tuple(cols)] = net
    r1 = pd.DataFrame(rows)
    r1.to_csv(os.path.join(EXP_DIR, "r1_returnnet_holdout_ce.csv"), index=False)
    ok = bool((r1.abs_fit_vs_ref <= TOL_R1).all() and (r1.abs_restored_vs_fit <= TOL_R1).all()
              and (r1.epoch_fit == r1.epoch_ref).all())
    return r1, nets, ok


def cost_reference():
    df = pd.read_csv(COSTS)
    return {frozenset(canon(s.strip('"').split(" | "))): (float(ce), int(ep))
            for s, ce, ep in zip(df.cluster, df.holdout_ce, df.best_epoch)}


def nll_tables(nets, pool):
    """Per cluster (keyed by tuple(cols)): NLL over all 2^m states (N x S, +inf if inadmissible), true-state index, true NLL."""
    p13 = np.clip(pool[[f"{l}_prob" for l in LABELS]].to_numpy(float), fp.CLIP, 1 - fp.CLIP)
    x13 = np.log(p13 / (1 - p13))
    Y = pool[[f"{l}_true_label" for l in LABELS]].to_numpy(int)
    out = {}
    for cols_t, net in nets.items():
        cols = list(cols_t)
        adm = torch.tensor(cc.hierarchy_admissible([LABELS[c] for c in cols], cc.enumerate_states(len(cols))))
        neg = torch.where(adm, 0.0, float("-inf")).double()
        with torch.no_grad():
            X = torch.tensor(x13[:, cols], dtype=torch.float32)
            nll = -torch.log_softmax(net(X).double() + neg, dim=1).numpy()
        T = cc.state_indices(Y[:, cols])
        true = nll[np.arange(len(T)), T]
        if not np.isfinite(true).all():
            raise Stop("inadmissible_or_infinite_true_state")
        out[cols_t] = dict(nll=nll, T=T, true=true)
    return out, Y


# ---------------------------------------------------------------- set arms (additive and max-rank product)
def set_arm(kind, cl, tabs, jt, idxA, idxB, Y_pool):
    """kind 'add': S = sum_c NLL_c, q = k-th smallest S on calibration true states, set = {j : S_j <= q}.
    kind 'max': per-cluster max-rank thresholds (Timans Alg. 2), set = product of per-cluster regions."""
    keys = [tuple(c) for _, c in cl]
    dims, combos, Y, per = jt["dims"], jt["combos"], jt["Y"], jt["per"]
    nA = len(idxA)
    rec = dict(n_cal=nA)
    if kind == "add":
        S_A = sum(tabs[k]["true"][idxA] for k in keys)
        k_rank = cc.conformal_rank(nA, ALPHA)
        if k_rank > nA:
            raise Stop("k_gt_n")
        q = cc.order_statistic(S_A, k_rank)
        if not np.isfinite(q):
            raise Stop("infinite_q")
        cal_cov = float((S_A <= q).mean())
        rec.update(k=k_rank, thr=float(q), cal_cov=cal_cov,
                   cal_S_true=S_A)
    else:
        M = np.column_stack([tabs[k]["true"][idxA] for k in keys])
        gA = cc.GroupData(labels=[str(i) for i in range(len(keys))], probs=np.zeros(M.shape),
                          y=np.zeros(M.shape, int), scores=M, n_rows_before_filter=nA, subject_id=None)
        thr_d, diag = mr.calibrate_max_rank(gA, ALPHA)
        if diag.get("degenerate"):
            raise Stop("degenerate_maxrank")
        thr = [thr_d[str(i)] for i in range(len(keys))]
        if not all(np.isfinite(t) for t in thr):
            raise Stop("infinite_maxrank_thr")
        regs = [tabs[k]["nll"] <= t for k, t in zip(keys, thr)]
        cal_cov = float(np.logical_and.reduce([tabs[k]["true"][idxA] <= t for k, t in zip(keys, thr)]).mean())
        rec.update(k=int(diag["k_star"]), thr={str(i): float(t) for i, t in enumerate(thr)}, cal_cov=cal_cov)

    # positions for the true joint state
    posarr = []
    for ci in range(len(keys)):
        a = np.full(tabs[keys[ci]]["nll"].shape[1], -1, dtype=np.int64)
        a[per[ci]] = np.arange(len(per[ci]))
        posarr.append(a)

    J = len(combos)
    n = len(idxB)
    sizes, covered, viol, sz_adm = [], [], [], []
    admY = cc.hierarchy_admissible(LABELS, Y)
    proj = {i: [] for i in range(len(keys))}
    for s0 in range(0, n, 1000):
        rb = idxB[s0:s0 + 1000]
        if kind == "add":
            Sb = sum(tabs[k]["nll"][rb][:, combos[:, ci]] for ci, k in enumerate(keys))
            inset = Sb <= q
            cov = sum(tabs[k]["true"][rb] for k in keys) <= q
        else:
            inset = np.ones((len(rb), J), dtype=bool)
            for ci, k in enumerate(keys):
                inset &= regs[ci][rb][:, combos[:, ci]]
            cov = np.logical_and.reduce([regs[ci][rb, tabs[k]["T"][rb]] for ci, k in enumerate(keys)])
        # consistency: true joint state is in the set iff covered
        jt_idx = np.ravel_multi_index(tuple(posarr[ci][tabs[k]["T"][rb]] for ci, k in enumerate(keys)), dims)
        if (jt_idx < 0).any():
            raise Stop("true_state_not_in_joint_table")
        if not np.array_equal(inset[np.arange(len(rb)), jt_idx], cov):
            raise Stop("covered_inconsistent_with_set")
        sz = inset.sum(axis=1)
        sz_adm.append((inset & admY[None, :]).sum(axis=1))
        dec = sz == 1
        v = np.zeros(len(rb), dtype=bool)
        if dec.any():
            v[dec] = ~cc.hierarchy_admissible(LABELS, Y[inset[dec].argmax(axis=1)])
        sizes.append(sz); covered.append(cov); viol.append(v)
        for ci, k in enumerate(keys):
            if kind == "add":
                oh = np.zeros((J, tabs[k]["nll"].shape[1]), dtype=np.float32)
                oh[np.arange(J), combos[:, ci]] = 1.0
                pc = (inset.astype(np.float32) @ oh) > 0
            else:
                pc = regs[ci][rb]
            proj[ci].append(pc[np.arange(len(rb)), tabs[k]["T"][rb]])
    rec["sizes"] = np.concatenate(sizes).astype(np.int64)
    rec["covered"] = np.concatenate(covered).astype(bool)
    rec["viol"] = np.concatenate(viol)
    rec["sizes_adm"] = np.concatenate(sz_adm).astype(np.int64)
    rec["proj"] = {i: np.concatenate(v) for i, v in proj.items()}
    rec["cluster_names"] = [" | ".join(LABELS[c] for c in c_) for _, c_ in cl]
    return rec


# ---------------------------------------------------------------- 13-label product references
def ref13(name, gA, gB, adm13, states13, yB):
    if name == "maxrank_adm":
        thr_d, diag = mr.calibrate_max_rank(gA, ALPHA)
        if diag.get("degenerate"):
            raise Stop("degenerate_maxrank13")
        thr = thr_d
        regB = e56.product_region(gB, thr) & adm13[None, :]
    else:
        corr = {"bonferroni": "bonferroni", "sidak": "sidak"}[name]
        thr, diag = bf.calibrate_per_label(gA, ALPHA, corr)
        if diag.get("degenerate"):
            raise Stop(f"degenerate_{name}")
        regB = e56.product_region(gB, thr)
    sizes = regB.sum(axis=1).astype(np.int64)
    covered = regB[np.arange(regB.shape[0]), yB]
    v = np.zeros(regB.shape[0], dtype=bool)
    dec = sizes == 1
    if dec.any():
        v[dec] = ~cc.hierarchy_admissible(LABELS, states13[regB[dec].argmax(axis=1)])
    return dict(sizes=sizes, covered=covered, viol=v, cal_cov=float("nan"),
                k=int(diag.get("k", diag.get("k_star", -1))))


# ---------------------------------------------------------------- summaries
def row_summary(arm, r, n_cal, n_test, k, cal_cov, sizes, covered, viol, cond_masks, proj_cov, extra=None):
    s = e56.vector_summary(sizes, covered, viol, N13)
    row = dict(arm=arm, resplit=r, n_cal=n_cal, n_test=n_test, k=k, cal_cov=cal_cov, **s)
    for mk, mm in cond_masks.items():
        row[f"cond_cov_{mk}"], row[f"n_cond_{mk}"] = e56.cond_cov(covered, mm)
    for nm, v in (proj_cov or {}).items():
        row[f"cov_cluster_{nm}"] = float(v.mean())
    if extra:
        row.update(extra)
    return row


def pct(v):
    v = np.asarray(v, float)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ---------------------------------------------------------------- summary (SPEC §6-§8)
METRICS = ["joint_coverage", "mean_joint_set_size", "median_joint_set_size", "empty_set_rate",
           "log2_over_m_mean", "fully_decisive_rate", "decisive_accuracy", "hierarchy_violation_rate_decisive"]
GATED = ["ADD-P6", "ADD-PH", "ADD-1", "MR-P6", "maxrank_adm"]
COMPARATORS = ["MR-P6", "ADD-PH", "ADD-1", "maxrank_adm", "EXP056_C", "EXP056_D"]


def lookups_and_crosscheck(res):
    exp = pd.read_csv(EXP056_VEC)
    C = exp[(exp.pid == "P_H") & (exp.arm == "C") & (exp.guard == "unguarded") & (exp.scorer_seed == 0)]
    D = exp[(exp.pid == "none") & (exp.arm == "D") & (exp.guard == "unguarded") & (exp.scorer_seed == 0)]
    extra = []
    for name, E in (("EXP056_C", C), ("EXP056_D", D)):
        for _, e in E.iterrows():
            d = {m: float(e[m]) for m in METRICS}
            extra.append(dict(arm=name, resplit=int(e.resplit), n_cal=np.nan, n_test=np.nan, k=np.nan, cal_cov=np.nan, **d))
    cross = {}
    for nm in ("maxrank_adm", "bonferroni", "sidak"):
        E = exp[(exp.pid == "none") & (exp.arm == nm)].set_index("resplit")
        mine = res[res.arm == nm].set_index("resplit")
        diffs = {m: float(np.nanmax(np.abs(mine.loc[E.index, m].to_numpy(float) - E[m].to_numpy(float))))
                 for m in ("joint_coverage", "mean_joint_set_size", "log2_over_m_mean")}
        cross[nm] = dict(max_abs_diff=diffs, passed=bool(max(diffs.values()) <= TOL_X), n_resplits=int(len(E)))
    return pd.DataFrame(extra), cross


def arm_table(allres):
    out = {}
    for arm, g in allres.groupby("arm"):
        rs = g[g.resplit >= 0].sort_values("resplit")
        fr = g[g.resplit == -1]
        d = dict(n_resplits=int(len(rs)))
        for m in METRICS:
            lo, hi = pct(rs[m])
            d[m] = dict(resplit_mean=float(rs[m].mean()), p2_5=lo, p97_5=hi,
                        frozen=float(fr[m].iloc[0]) if len(fr) else None)
        out[arm] = d
    return out


def paired(allres, a, b, metric):
    x = allres[(allres.arm == a) & (allres.resplit >= 0)].sort_values("resplit")[metric].to_numpy(float)
    y = allres[(allres.arm == b) & (allres.resplit >= 0)].sort_values("resplit")[metric].to_numpy(float)
    d = x - y
    lo, hi = pct(d)
    return dict(mean=float(d.mean()), p2_5=lo, p97_5=hi)


def summarize(res, share_df, r1_info, cross, allres):
    out = dict(cross_check=cross, r1=r1_info)
    out["arms"] = arm_table(allres)
    gates = {}
    for a in GATED:
        cm = out["arms"][a]["joint_coverage"]["resplit_mean"]
        gates[a] = dict(resplit_mean_cov=cm, in_band=bool(COV_LO <= cm <= COV_HI))
    out["gates"] = gates
    out["gates"]["Bonferroni_Sidak_note"] = "valid if >= 0.90; not compared"
    deltas = {}
    for c in COMPARATORS:
        dz = paired(allres, "ADD-P6", c, "mean_joint_set_size")
        dl = paired(allres, "ADD-P6", c, "log2_over_m_mean")
        de = paired(allres, "ADD-P6", c, "empty_set_rate")
        valid = bool(gates["ADD-P6"]["in_band"] and gates.get(c, {"in_band": True})["in_band"])
        deltas[c] = dict(delta_mean_size=dz, delta_log2_over_13=dl, delta_empty_rate=de, comparison_valid=valid)
    out["paired_deltas_ADD_P6_minus"] = deltas
    ratios = {}
    for c in ("bonferroni", "maxrank_adm"):
        a = allres[(allres.arm == "ADD-P6") & (allres.resplit >= 0)].sort_values("resplit")
        b = allres[(allres.arm == c) & (allres.resplit >= 0)].sort_values("resplit")
        rr = a.mean_joint_set_size.to_numpy() / b.mean_joint_set_size.to_numpy()
        geo = 2.0 ** (13.0 * (a.log2_over_m_mean.to_numpy() - b.log2_over_m_mean.to_numpy()))
        lo1, hi1 = pct(rr); lo2, hi2 = pct(geo)
        ratios[c] = dict(ratio_mean_size=float(a.mean_joint_set_size.mean() / b.mean_joint_set_size.mean()),
                         ratio_p2_5=lo1, ratio_p97_5=hi1,
                         geo_ratio_2pow13dlog=float(geo.mean()), geo_p2_5=lo2, geo_p97_5=hi2)
    out["efficiency_ratios_ADD_P6_vs"] = ratios
    cond = {}
    for a in ("ADD-P6", "MR-P6", "maxrank_adm"):
        g = allres[allres.arm == a]
        rs = g[g.resplit >= 0]
        cond[a] = {}
        for mk in ("pos_Pleural_Other", "pos_Fracture", "pos_Lung_Lesion", "any_pos"):
            v = rs[f"cond_cov_{mk}"].to_numpy(float)
            lo, hi = pct(v)
            cond[a][mk] = dict(resplit_mean=float(np.nanmean(v)), p2_5=lo, p97_5=hi,
                               n_mean=float(rs[f"n_cond_{mk}"].mean()),
                               frozen=float(g[g.resplit == -1][f"cond_cov_{mk}"].iloc[0]),
                               n_frozen=int(g[g.resplit == -1][f"n_cond_{mk}"].iloc[0]))
    out["conditional_coverage_test"] = cond
    out["p6_share_of_S_frozen"] = share_df.to_dict(orient="records")
    # pre-registered flags (SPEC §9)
    flags = []
    for a in ("ADD-P6", "ADD-PH", "ADD-1"):
        er = out["arms"][a]["empty_set_rate"]["resplit_mean"]
        if er > 0.02:
            flags.append(f"{a} empty-set rate {er:.4f} > 2%")
        g = allres[(allres.arm == a)]
        lo_ok = (g.cal_cov >= 1 - ALPHA - 1e-12) & (g.cal_cov <= 1 - ALPHA + 1.0 / (g.n_cal + 1) + 1e-12)
        out.setdefault("cal_side_bounds_ok", {})[a] = bool(lo_ok.all())
        if not lo_ok.all():
            flags.append(f"{a} calibration-side coverage outside [0.90, 0.90+1/(n+1)] in {int((~lo_ok).sum())} rows")
    d_e = deltas["MR-P6"]["delta_empty_rate"]["mean"]
    if abs(d_e) > 0.01:
        flags.append(f"|Delta empty| ADD-P6 vs MR-P6 = {abs(d_e):.4f} > 1pp")
    po = cond["ADD-P6"]["pos_Pleural_Other"]["resplit_mean"]
    if po < 0.20:
        flags.append(f"ADD-P6 PO-positive conditional coverage {po:.3f} < 0.20")
    if r1_info["cap_hits_stop_epoch_200"] > 0:
        flags.append(f"{r1_info['cap_hits_stop_epoch_200']} scorer(s) stop_epoch 200 (cap)")
    for a in GATED:
        if not gates[a]["in_band"]:
            flags.append(f"{a} resplit-mean coverage {gates[a]['resplit_mean_cov']:.4f} outside [{COV_LO},{COV_HI}]")
    for nm, v in cross.items():
        if not v["passed"]:
            flags.append(f"cross-check {nm} failed")
    out["flags"] = flags
    return out


def main():
    t0 = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    metrics = dict(status=None, steps=[])
    try:
        g0, pool = gate0()
        metrics["gate0"] = g0
        metrics["steps"].append("gate0 passed")

        P6, PH, SING = partitions()
        fits = distinct_fits(P6 + PH + SING)
        cost_ref = cost_reference()
        r1, nets, ok = train_and_check(cost_ref)
        metrics["r1"] = dict(n_distinct_fits=int(len(r1)), max_abs_fit_vs_ref=float(r1.abs_fit_vs_ref.max()),
                             max_abs_restored_vs_fit=float(r1.abs_restored_vs_fit.max()),
                             all_epochs_match=bool((r1.epoch_fit == r1.epoch_ref).all()),
                             cap_hits_stop_epoch_200=int(r1.cap_hit_200.sum()),
                             max_stop_epoch=int(r1.stop_epoch.max()), passed=ok)
        if not ok:
            raise Stop("R1_mismatch")
        P6_names = {" | ".join(LABELS[c] for c in c_) for _, c_ in P6}
        P6_sum = float(r1[r1.cluster.isin(P6_names)].ce_fit.sum())
        P6_sum_ref_csv = float(sum(cost_ref[frozenset(c)][0] for _, c in P6))
        metrics["r1"]["p6_sum_refit"] = P6_sum
        metrics["r1"]["p6_sum_cluster_costs_csv_rounded"] = P6_sum_ref_csv
        metrics["r1"]["p6_sum_partitions_json"] = json.load(open(PARTS))["partitions"]["6"]["cost"]
        if abs(P6_sum - 2.532202) > 1e-6 or abs(P6_sum - json.load(open(PARTS))["partitions"]["6"]["cost"]) > 1e-6:
            raise Stop("R1_P6_sum")
        metrics["steps"].append("R1 passed (19 distinct fits; restored-net CE equals fit CE)")

        tabs, Y_pool = nll_tables(nets, pool)
        pool13 = cc.extract_group(pool, LABELS)
        states13 = cc.enumerate_states(N13)
        adm13 = cc.hierarchy_admissible(LABELS, states13)
        yc_pool = cc.state_indices(Y_pool)
        jt6 = joint_table(P6)
        jtH = joint_table(PH)
        jtS = joint_table(SING, enforce=False)  # raw 8192 states by design (SPEC §4)
        if not (len(jt6["combos"]) == len(jtH["combos"]) == 2400 and int(adm13.sum()) == 2400):
            raise Stop("joint_size")
        full_set = set(map(tuple, states13[adm13]))
        if set(map(tuple, jt6["Y"])) != full_set or set(map(tuple, jtH["Y"])) != full_set:
            raise Stop("P6_or_PH_not_full_admissible")
        metrics["joint_tables"] = dict(P6=len(jt6["combos"]), PH=len(jtH["combos"]), singletons=len(jtS["combos"]),
                                       full_admissible_13=int(adm13.sum()),
                                       cross_cluster_edges_P6=e56.edges_across([[LABELS[i] for i in c] for _, c in P6]),
                                       cross_cluster_edges_PH=e56.edges_across([[LABELS[i] for i in c] for _, c in PH]))
        metrics["steps"].append("joint tables built (P6=PH=2400 equal full admissible set)")

        subj = pool.subject_id.to_numpy()
        uniq = np.unique(subj)
        n_conf = g0["n_conf"]
        splits = [(-1, np.arange(n_conf), np.arange(n_conf, len(pool)))]
        for r in range(N_RESPLITS):
            perm = np.random.default_rng(r).permutation(uniq)
            inA = np.isin(subj, perm[: len(uniq) // 2])
            splits.append((r, np.flatnonzero(inA), np.flatnonzero(~inA)))

        arms_def = [
            ("ADD-P6", "add", P6, jt6),
            ("MR-P6", "max", P6, jt6),
            ("ADD-PH", "add", PH, jtH),
            ("ADD-1", "add", SING, jtS),
        ]
        rows, share_rows = [], []
        for r, idxA, idxB in splits:
            yB = Y_pool[idxB]
            masks = {f"pos_{l.replace(' ', '_')}": yB[:, LABELS.index(l)] == 1 for l in RARE}
            masks["any_pos"] = yB.sum(axis=1) >= 1
            for arm, kind, cl, jt in arms_def:
                rec = set_arm(kind, cl, tabs, jt, idxA, idxB, Y_pool)
                proj = rec["proj"]
                extra = dict(mean_size_adm_intersected=float(rec["sizes_adm"].mean()),
                             inadmissible_share_of_retained=float(
                                 1 - rec["sizes_adm"].sum() / max(rec["sizes"].sum(), 1)))
                row = row_summary(arm, r, len(idxA), len(idxB), rec["k"], rec["cal_cov"], rec["sizes"],
                                  rec["covered"], rec["viol"], masks, {rec["cluster_names"][i]: proj[i] for i in proj},
                                  extra=extra)
                rows.append(row)
                if arm == "ADD-P6" and r == -1:
                    # S share at the frozen threshold: per-cluster NLL / S on calibration true states
                    S_A = rec["cal_S_true"]
                    q = rec["thr"]
                    share_full = {}
                    for ci, (_, c) in enumerate(P6):
                        contrib = tabs[tuple(c)]["true"][idxA]
                        share_full[ci] = contrib / S_A
                    near = np.argsort(np.abs(S_A - q))[: max(1, int(round(0.05 * len(S_A))))]
                    for ci, (_, c) in enumerate(P6):
                        share_rows.append(dict(cluster=rec["cluster_names"][ci], share_all_cal=float(share_full[ci].mean()),
                                               share_near_q_5pct=float((share_full[ci][near]).mean()),
                                               n_all=len(S_A), n_near=len(near)))
            # product references on the 13-label space
            gA = e56.subset(pool13, idxA)
            gB = e56.subset(pool13, idxB)
            for nm in ("maxrank_adm", "bonferroni", "sidak"):
                out = ref13(nm, gA, gB, adm13, states13, yc_pool[idxB])
                row = row_summary(nm, r, len(idxA), len(idxB), out["k"], float("nan"), out["sizes"], out["covered"],
                                  out["viol"], masks, None)
                rows.append(row)
        metrics["steps"].append("arms computed on frozen draw + 100 resplits")

        res = pd.DataFrame(rows)
        look, cross = lookups_and_crosscheck(res)
        allres = pd.concat([res, look], ignore_index=True)
        allres.to_csv(os.path.join(EXP_DIR, "add_resplit_metrics.csv"), index=False)
        share_df = pd.DataFrame(share_rows)
        share_df.to_csv(os.path.join(EXP_DIR, "add_p6_share_of_S.csv"), index=False)
        r1_info = metrics["r1"]
        if not all(v["passed"] for v in cross.values()):
            raise Stop("reference_crosscheck_failed " + json.dumps(cross, default=float))
        metrics["summary"] = summarize(res, share_df, r1_info, cross, allres)
        metrics["status"] = "ARMS_COMPUTED"
    except Stop as e:
        metrics["status"] = f"STOPPED: {e}"
        print("STOP:", e, flush=True)
    finally:
        metrics["runtime_s"] = round(time.time() - t0, 1)
        with open(os.path.join(EXP_DIR, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2, default=str)
    return metrics


if __name__ == "__main__":
    main()
