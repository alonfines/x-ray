#!/usr/bin/env python3
"""Exp055 - learned joint-state score, FULL design (follows the Exp054 pilot).

Pre-registered in experiments/055_learned_score_full/README.md before this file was run.
Implements that spec; no tuning, no searching. Reuses unedited: conformal_common.py,
bonferrony.py, max_rank.py, lp_scp.py (fit_structure, state_scores), data/hierarchy.py.
The L0/L1/L2 training loop and the spec-order bookkeeping are copied from
experiments/054_learned_score_pilot/run.py (copy, not edit; that file is not modified).

Usage (from any directory):
    /usr/bin/python3 experiments/055_learned_score_full/run.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

import conformal_common as cc  # noqa: E402
import bonferrony as bf  # noqa: E402
import max_rank as mr  # noqa: E402
import lp_scp as lp  # noqa: E402
from data.hierarchy import CHEXPERT_HIERARCHY  # noqa: E402

STEM = "densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809"
CLUSTERS_YAML = os.path.join(REPO, "configs", "clusters", "exp046.yaml")
PILOT_RESULTS = os.path.join(REPO, "experiments", "054_learned_score_pilot", "results.csv")

ALPHA = 0.10
SEEDS = (0, 1, 2, 3, 4)            # MLP / model seeds; seed 0 = primary = pilot seed
HOLDOUT_FRAC = 0.20
HOLDOUT_SEED = 0                   # fixed across all scorers and seeds
HIDDEN = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 256
PATIENCE = 10
MAX_EPOCHS = 200
PROB_CLIP = 1e-6
N_RESPLITS = 100                   # resplit r = 0..99, default_rng(r)
N_DRAWS_PER_CLUSTER = 20           # random label subsets per designed cluster
DRAW_SEED_BASE = 5000              # default_rng(5000 + idx), idx = 20*cluster_position + draw
ELIGIBLE_MIN_POS = 200             # calibration positives on the all-13-clean conformal split
IDENTITY_TOL = 0.005               # Exp053 identity gate tolerance (fraction of rows)
COV_LO, COV_HI = 0.895, 0.905      # coverage gate on resplit-mean joint coverage
BAND_Z = 1.96

# Expected population (all-13-clean unless stated); asserted, not assumed.
EXPECT = dict(val_rows=18570, val_subj=5898, conf_rows=18754, conf_subj=5926,
              test_rows_full=24394, test_rows=18877, test_subj=5932,
              pool_rows=37631, pool_subj=11858)
DESIGNED_EXPECT = [("cardio_pair", 2), ("lung_core", 3), ("pleural_free", 3), ("lung_wide", 5)]
DESIGNED_EDGES_EXPECT = {"cardio_pair": 1, "lung_core": 2, "pleural_free": 0, "lung_wide": 5}

SCOPE = "all-13-clean studies, 77% of test"


# ---------------------------------------------------------------------------
# helpers (copied from the pilot where noted)
# ---------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def package_versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "pyyaml": yaml.__version__}


def all13_clean(df):
    """Rows with every one of the 13 true labels in {0,1} and every prob finite (pilot, verbatim logic)."""
    y = df[[f"{lab}_true_label" for lab in cc.ALL_LABELS]].to_numpy(dtype=float)
    p = df[[f"{lab}_prob" for lab in cc.ALL_LABELS]].to_numpy(dtype=float)
    keep = np.all(np.isin(y, [0.0, 1.0]), axis=1) & np.all(np.isfinite(p), axis=1)
    return df.loc[keep].reset_index(drop=True)


def hier_violations(df, labels):
    """(violating vectors, rows checked) for rows whose cluster members are all in {0,1} (pilot logic)."""
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


def logit(p):
    p = np.clip(p, PROB_CLIP, 1.0 - PROB_CLIP)
    return np.log(p / (1.0 - p))


def spec_weights(m):
    """Spec target index is sum_l y_l * 2^l, labels in yaml order (label 0 = LSB)."""
    return (1 << np.arange(m)).astype(np.int64)


def subset(g, idx):
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


# ---------------------------------------------------------------------------
# learned scorers (L0 / L1 / L2) - training loop copied from the pilot's train_l1
# ---------------------------------------------------------------------------

def train_scorer(X_tr, y_tr, X_ho, y_ho, S, seed, hidden):
    """L1/L2: Linear(d,64)-ReLU-Linear(64,S). L0: Linear(d,S) (multinomial logistic regression).

    Adam, lr 1e-3, weight_decay 1e-4, batch 256; early stopping on inner-holdout CE with
    patience 10, max 200 epochs, best weights kept. Seed re-applied before init and for the
    minibatch generator.
    """
    torch.manual_seed(seed)
    d_in = X_tr.shape[1]
    if hidden:
        net = nn.Sequential(nn.Linear(d_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, S))
    else:
        net = nn.Sequential(nn.Linear(d_in, S))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    best, best_ep, since, best_state = float("inf"), 0, 0, None
    stop_ep, stopped_early = MAX_EPOCHS, False
    history = []
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        order = torch.randperm(X_tr.shape[0], generator=gen)
        for i in range(0, X_tr.shape[0], BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = ce(net(X_tr[b]), y_tr[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            ho_loss = ce(net(X_ho), y_ho).item()
            tr_loss = ce(net(X_tr), y_tr).item()
        history.append({"epoch": ep, "train_ce": tr_loss, "holdout_ce": ho_loss})
        if ho_loss < best:
            best, best_ep, since = ho_loss, ep, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            since += 1
        if since >= PATIENCE:
            stop_ep, stopped_early = ep, True
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        final_tr = ce(net(X_tr), y_tr).item()
        final_ho = ce(net(X_ho), y_ho).item()
    info = {"best_epoch": best_ep, "stop_epoch": stop_ep, "hit_cap": bool(not stopped_early),
            "epochs_run": len(history), "final_train_ce": final_tr, "final_holdout_ce": final_ho}
    return net, info


def neg_log_softmax_spec(net, X):
    """s(x, y) = -log_softmax(logits)[y] for every joint state, (n, 2^m), spec index order."""
    with torch.no_grad():
        lpr = torch.log_softmax(net(X).double(), dim=1).numpy()
    return -lpr


# ---------------------------------------------------------------------------
# region builders (all in cc state order)
# ---------------------------------------------------------------------------

def threshold_eval(raw, smask, adm, y_cc, idxA, idxB):
    """Split-conformal threshold over joint-state scores; guarded and unguarded regions.

    raw   (N, S) raw score, cc order;  smask = raw with +inf on inadmissible states.
    Threshold: k = ceil((n+1)(1-alpha))-th smallest calibration score of the TRUE state
    (conformal_rank / order_statistic, no interpolation). Admissibility is masked explicitly
    so an infinite threshold cannot re-admit inadmissible states.
    """
    s_true = smask[idxA, y_cc[idxA]]
    n_bad = int(np.count_nonzero(~adm[y_cc[idxA]]))
    assert n_bad == 0, "inadmissible true state in calibration half: Gate 0 violated"
    k = cc.conformal_rank(idxA.size, ALPHA)
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


def refs_eval(gA, gB, adm):
    """Classical references, each as a region over cc states on A (calibration) and B (test).

    indep_per_label : alpha per label, product
    bonferroni      : alpha/m per label, product            (bonferrony.calibrate_per_label)
    sidak           : 1-(1-alpha)^(1/m) per label, product
    maxrank         : max_rank.calibrate_max_rank (Timans Alg. 2), product
    maxrank_adm     : maxrank product intersected with hierarchy-admissible states
    bonferroni_adm  : bonferroni product intersected with hierarchy-admissible states
    All thresholds use k = ceil((n+1)(1-alpha')) via conformal_rank (one convention).
    """
    thr_i, di = bf.calibrate_per_label(gA, ALPHA, "none")
    thr_b, db = bf.calibrate_per_label(gA, ALPHA, "bonferroni")
    thr_s, ds = bf.calibrate_per_label(gA, ALPHA, "sidak")
    thr_m, dm = mr.calibrate_max_rank(gA, ALPHA)
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
    """Locked metric set on test half B, plus calibration-side coverage on A."""
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


# ---------------------------------------------------------------------------
# units: designed clusters and random label subsets
# ---------------------------------------------------------------------------

def build_units(designed, eligible):
    units = []
    for ci, cl in enumerate(designed):
        units.append(dict(family="designed", unit=cl.name, parent=cl.name, draw_id=-1,
                          draw_seed=-1, labels=list(cl.labels), m=cl.m, cluster_idx=ci))
    for ci, cl in enumerate(designed):
        for i in range(N_DRAWS_PER_CLUSTER):
            idx = N_DRAWS_PER_CLUSTER * ci + i
            rng = np.random.default_rng(DRAW_SEED_BASE + idx)
            pick = rng.choice(len(eligible), size=cl.m, replace=False)
            labels = [eligible[p] for p in sorted(pick.tolist())]    # canonical ALL_LABELS order
            units.append(dict(family="random", unit=f"rand_{cl.name}_{i:02d}", parent=cl.name,
                              draw_id=i, draw_seed=DRAW_SEED_BASE + idx, labels=labels, m=cl.m,
                              cluster_idx=ci))
    for u in units:
        u["n_edges"] = n_edges_of(u["labels"])
    return units


def occupancy(val_df, labels):
    """Distinct cluster label vectors among all-13-clean validation rows (projection onto labels)."""
    g = cc.extract_group(val_df, labels)
    return int(np.unique(cc.state_indices(g.y)).size)


def eval_unit(u, ctx):
    """Train this unit's scorers and evaluate every method on every resplit (and the frozen draw)."""
    L = u["labels"]
    m = len(L)
    S = 2 ** m
    w = spec_weights(m)
    states_cc = cc.enumerate_states(m)
    spec_of_cc = (states_cc * w).sum(axis=1)               # cc state -> spec index
    adm = cc.hierarchy_admissible(L, states_cc)            # (S,) cc order
    n_adm = int(adm.sum())

    gp = cc.extract_group(ctx["pool_df"], L)
    assert gp.n == EXPECT["pool_rows"]
    y_cc = cc.state_indices(gp.y)
    assert np.array_equal(spec_of_cc[y_cc], (gp.y * w).sum(axis=1)), "state-order mapping mismatch"

    g_tr = cc.extract_group(ctx["val_tr"], L)
    g_ho = cc.extract_group(ctx["val_ho"], L)
    g_val = cc.extract_group(ctx["val"], L)
    occ = int(np.unique(cc.state_indices(g_val.y)).size)
    tr13 = cc.extract_group(ctx["val_tr"], cc.ALL_LABELS)
    ho13 = cc.extract_group(ctx["val_ho"], cc.ALL_LABELS)
    assert tr13.n == g_tr.n and ho13.n == g_ho.n

    ytr = torch.tensor((g_tr.y * w).sum(axis=1), dtype=torch.long)
    yho = torch.tensor((g_ho.y * w).sum(axis=1), dtype=torch.long)

    scorers, scorer_diag = {}, []
    fams = ("L0", "L1", "L2") if u["family"] == "designed" else ("L1", "L2")
    for fam in fams:
        if fam == "L2":
            Xtr, Xho, Xpool = logit(tr13.probs), logit(ho13.probs), logit(ctx["pool13"].probs)
        else:
            Xtr, Xho, Xpool = logit(g_tr.probs), logit(g_ho.probs), logit(gp.probs)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        Xho_t = torch.tensor(Xho, dtype=torch.float32)
        for seed in SEEDS:
            net, info = train_scorer(Xtr_t, ytr, Xho_t, yho, S, seed, hidden=(fam != "L0"))
            raw_spec = neg_log_softmax_spec(net, torch.tensor(Xpool, dtype=torch.float32))
            raw = raw_spec[:, spec_of_cc]
            smask = np.where(adm[None, :], raw, np.inf)
            scorers[(fam, seed)] = (raw, smask)
            scorer_diag.append(dict(family=u["family"], unit=u["unit"], labels="|".join(L), m=m,
                                    scorer=fam, seed=seed, n_input=Xtr.shape[1], **info,
                                    n_admissible=n_adm, n_occupied_val=occ))

    # LP-SCP admissible (mu = nu = 1): Sigma, Hp, Cp fit on the all-13-clean validation rows
    struct = lp.fit_structure(g_val)
    raw_lp = lp.state_scores(gp.probs, struct, 1.0, 1.0)
    smask_lp = np.where(adm[None, :], raw_lp, np.inf)
    scorer_diag.append(dict(family=u["family"], unit=u["unit"], labels="|".join(L), m=m,
                            scorer="LPSCP", seed=-1, n_input=m, best_epoch=np.nan, stop_epoch=np.nan,
                            hit_cap=False, epochs_run=0, final_train_ce=np.nan, final_holdout_ce=np.nan,
                            n_admissible=n_adm, n_occupied_val=occ,
                            sigma_condition_number=struct.diagnostics["sigma_condition_number"],
                            sigma_ridge=struct.diagnostics["sigma_ridge"],
                            n_observed_label_sets=struct.diagnostics["n_observed_label_sets"]))

    rows = []
    for (r, idxA, idxB) in ctx["splits"]:
        gA = subset(gp, idxA)
        gB = subset(gp, idxB)
        base = dict(family=u["family"], unit=u["unit"], parent=u["parent"], draw_id=u["draw_id"],
                    m=m, n_edges=u["n_edges"], labels="|".join(L), resplit=int(r),
                    n_admissible=n_adm, n_occupied_val=occ, seed=-1)

        for name, (regA, regB, k) in refs_eval(gA, gB, adm).items():
            rows.append(metric_row(dict(base, variant=name, threshold=np.nan, k=k, n_inadm_true_cal=np.nan),
                                   gA, gB, regA, regB))

        for (fam, seed), (raw, smask) in scorers.items():
            out = threshold_eval(raw, smask, adm, y_cc, idxA, idxB)
            for guard in (True, False):
                regA, regB, thr, k, n_bad = out[guard]
                variant = fam if guard else f"{fam}_unguarded"
                rows.append(metric_row(dict(base, variant=variant, seed=seed, threshold=thr, k=k,
                                            n_inadm_true_cal=n_bad), gA, gB, regA, regB))
            if r == -1 and u["family"] == "designed" and fam == "L1" and seed == 0:
                ctx["pilot_check"].append(check_against_pilot(u["unit"], out[True][2], scorers_info(scorer_diag, fam, seed)))

        out = threshold_eval(raw_lp, smask_lp, adm, y_cc, idxA, idxB)
        for guard in (True, False):
            regA, regB, thr, k, n_bad = out[guard]
            variant = "lpscp_adm" if guard else "lpscp_adm_unguarded"
            rows.append(metric_row(dict(base, variant=variant, threshold=thr, k=k, n_inadm_true_cal=n_bad),
                                   gA, gB, regA, regB))

    return rows, scorer_diag


def scorers_info(scorer_diag, fam, seed):
    for d in scorer_diag:
        if d["scorer"] == fam and d["seed"] == seed:
            return d
    raise KeyError((fam, seed))


def check_against_pilot(unit_name, frozen_thr, info):
    """Reproduction check: frozen-draw L1 seed 0 must equal the pilot's L1 row exactly."""
    pilot = pd.read_csv(PILOT_RESULTS)
    row = pilot[(pilot["method"] == "L1") & (pilot["cluster"] == unit_name)]
    assert len(row) == 1, f"pilot L1 row missing for {unit_name}"
    row = row.iloc[0]
    res = dict(unit=unit_name,
               threshold_now=frozen_thr, threshold_pilot=float(row["threshold"]),
               best_epoch_now=int(info["best_epoch"]), best_epoch_pilot=int(row["best_epoch"]),
               stop_epoch_now=int(info["stop_epoch"]), stop_epoch_pilot=int(row["stop_epoch"]),
               holdout_ce_now=float(info["final_holdout_ce"]), holdout_ce_pilot=float(row["final_holdout_ce"]))
    res["match"] = bool(abs(res["threshold_now"] - res["threshold_pilot"]) < 1e-6
                        and res["best_epoch_now"] == res["best_epoch_pilot"]
                        and res["stop_epoch_now"] == res["stop_epoch_pilot"]
                        and abs(res["holdout_ce_now"] - res["holdout_ce_pilot"]) < 1e-6)
    return res


# ---------------------------------------------------------------------------
# identity gate (Exp053 SPEC 8, Amendment 3), classical analogue: m = 1
# ---------------------------------------------------------------------------

def identity_gate(ctx):
    """At m = 1 (single-label clusters) Bonferroni, Sidak, max-rank and independent per-label
    must produce the same set on the same rows. Tolerance IDENTITY_TOL of rows.
    Checked on the frozen draw and all 100 resplits, for all 13 labels.
    """
    rows = []
    for (r, idxA, idxB) in ctx["splits"]:
        for lab in cc.ALL_LABELS:
            g1 = ctx["pool_single"][lab]
            gA = subset(g1, idxA)
            gB = subset(g1, idxB)
            thr = {}
            thr["indep"], _ = bf.calibrate_per_label(gA, ALPHA, "none")
            thr["bonferroni"], _ = bf.calibrate_per_label(gA, ALPHA, "bonferroni")
            thr["sidak"], _ = bf.calibrate_per_label(gA, ALPHA, "sidak")
            thr["maxrank"], _ = mr.calibrate_max_rank(gA, ALPHA)
            setsB = {k: cc.prediction_sets(gB, v) for k, v in thr.items()}
            setsA = {k: cc.prediction_sets(gA, v) for k, v in thr.items()}
            rec = dict(resplit=int(r), label=lab)
            rec["mismatch_bonf_vs_maxrank"] = float(np.mean(np.any(setsB["bonferroni"] != setsB["maxrank"], axis=(1, 2))))
            rec["mismatch_sidak_vs_maxrank"] = float(np.mean(np.any(setsB["sidak"] != setsB["maxrank"], axis=(1, 2))))
            rec["mismatch_indep_vs_bonf"] = float(np.mean(np.any(setsB["indep"] != setsB["bonferroni"], axis=(1, 2))))
            for k in ("indep", "bonferroni", "sidak", "maxrank"):
                rec[f"calside_cov_{k}"] = float(np.mean(np.take_along_axis(setsA[k], gA.y[:, :, None], axis=2)[:, 0, 0]))
            rows.append(rec)
    df = pd.DataFrame(rows)
    mism_cols = ["mismatch_bonf_vs_maxrank", "mismatch_sidak_vs_maxrank", "mismatch_indep_vs_bonf"]
    per_resplit = df.groupby("resplit")[mism_cols].max().max(axis=1)
    frozen_val = float(per_resplit.loc[-1])
    res_vals = per_resplit.drop(index=-1)
    summary = dict(
        frozen_max_mismatch=frozen_val,
        resplits_0_99_min=float(res_vals.min()), resplits_0_99_median=float(res_vals.median()),
        resplits_0_99_max=float(res_vals.max()),
        n_resplits_over_tol=int((res_vals > IDENTITY_TOL).sum()),
        tolerance=IDENTITY_TOL,
        passed=bool(frozen_val <= IDENTITY_TOL and (res_vals <= IDENTITY_TOL).all()),
        calside_cov_mean={k: float(df[f"calside_cov_{k}"].mean()) for k in ("indep", "bonferroni", "sidak", "maxrank")},
    )
    return df, summary


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

METRICS = ["joint_coverage", "cal_side_coverage", "mean_joint_set_size", "median_joint_set_size",
           "log2_over_m_mean", "empty_set_rate", "fully_decisive_rate", "correct_decisive_rate",
           "decisive_accuracy", "hierarchy_violation_rate_decisive", "threshold"]
PAIR_METRICS = ["mean_joint_set_size", "log2_over_m_mean", "correct_decisive_rate", "joint_coverage",
                "median_joint_set_size"]


def summarize(df):
    key = ["family", "unit", "parent", "draw_id", "m", "n_edges", "n_admissible", "labels", "variant", "seed"]
    out = []
    for keyvals, g in df.groupby(key, sort=False):
        rs = g[g.resplit >= 0].sort_values("resplit")
        fr = g[g.resplit == -1]
        rec = dict(zip(key, keyvals))
        rec["n_resplits"] = int(rs.shape[0])
        for mt in METRICS:
            v = rs[mt].to_numpy(dtype=float)
            f = float(fr[mt].iloc[0]) if len(fr) else np.nan
            rec[f"{mt}_mean"] = float(np.nanmean(v))
            rec[f"{mt}_sd"] = float(np.nanstd(v, ddof=1))
            rec[f"{mt}_p2.5"] = float(np.nanpercentile(v, 2.5))
            rec[f"{mt}_p97.5"] = float(np.nanpercentile(v, 97.5))
            rec[f"{mt}_frozen"] = f
            rec[f"{mt}_frozen_pct_le"] = float(np.mean(v <= f)) if np.isfinite(f) else np.nan
        cov = rec["joint_coverage_mean"]
        rec["coverage_gate"] = "pass" if (COV_LO <= cov <= COV_HI) else "coverage-unequal"
        out.append(rec)
    return pd.DataFrame(out)


def paired(rs, pairs):
    """pairs: list of (unit, family, (varA,seedA), (varB,seedB)). Paired within resplit."""
    out = []
    for mt in PAIR_METRICS:
        wide = rs.pivot_table(index="resplit", columns=["unit", "variant", "seed"], values=mt, aggfunc="first")
        for unit, fam, (va, sa), (vb, sb) in pairs:
            a = wide[(unit, va, sa)].to_numpy(dtype=float)
            b = wide[(unit, vb, sb)].to_numpy(dtype=float)
            assert len(a) == N_RESPLITS and len(b) == N_RESPLITS
            d = a - b
            lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            out.append(dict(unit=unit, family=fam, arm=va, seed_arm=sa, comparator=vb, seed_comparator=sb,
                            metric=mt, n_resplits=len(d), mean_diff=float(d.mean()),
                            ci_lo_2p5=float(lo), ci_hi_97p5=float(hi),
                            ci_excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(out)


def run_analysis(df, ctx):
    rs = df[df.resplit >= 0].copy()
    summ = summarize(df)
    summ["coverage_gate"] = summ["coverage_gate"].astype(str)

    cov = summ.set_index(["unit", "variant", "seed"])["joint_coverage_mean"]

    def cov_ok(unit, var, seed):
        return bool(COV_LO <= cov[(unit, var, seed)] <= COV_HI)

    # pairs: designed units (all learned seeds + references) and random draws (L1, L2 seed 0)
    pairs = []
    designed_units = [c for c, _ in DESIGNED_EXPECT]
    for unit in designed_units:
        for fam in ("L0", "L1", "L2"):
            for s in SEEDS:
                for ref in (("maxrank_adm", -1), ("lpscp_adm", -1), ("bonferroni", -1)):
                    pairs.append((unit, "designed", (fam, s), ref))
        pairs.append((unit, "designed", ("L1", 0), ("L0", 0)))
        pairs.append((unit, "designed", ("L2", 0), ("L1", 0)))
    random_units = sorted(set(df.loc[df.family == "random", "unit"]))
    for unit in random_units:
        for fam in ("L1", "L2"):
            for ref in (("maxrank_adm", -1), ("lpscp_adm", -1), ("bonferroni", -1)):
                pairs.append((unit, "random", (fam, 0), ref))
    pair_df = paired(rs, pairs)
    # comparison validity: both arms inside the coverage gate (resplit mean)
    pair_df["both_coverage_gate_pass"] = [
        cov_ok(r.unit, r.arm, r.seed_arm) and cov_ok(r.unit, r.comparator, r.seed_comparator)
        for r in pair_df.itertuples()]
    pair_df["comparison_valid"] = pair_df["both_coverage_gate_pass"]

    # designed vs random control (L1, L2 seed 0, max-rank adm for context), resplit-mean log2/m
    ctl = []
    all_random = summ[summ.family == "random"]
    for cl, _m in DESIGNED_EXPECT:
        dsg = summ[(summ.unit == cl) & (summ.variant.isin(["L1", "L2", "maxrank_adm", "lpscp_adm"]))]
        for var in ["L1", "L2", "maxrank_adm", "lpscp_adm"]:
            sd = 0 if var in ("L1", "L2") else -1
            rowd = dsg[(dsg.variant == var) & (dsg.seed == sd)]
            val = float(rowd["log2_over_m_mean_mean"].iloc[0])
            draws = summ[(summ.family == "random") & (summ.parent == cl) & (summ.variant == var) & (summ.seed == sd)]
            dv = draws["log2_over_m_mean_mean"].to_numpy(dtype=float)
            ne = int(rowd["n_edges"].iloc[0])
            same_draws = draws[draws.n_edges == ne]["log2_over_m_mean_mean"].to_numpy(dtype=float)
            all_same = all_random[(all_random.variant == var) & (all_random.seed == sd) & (all_random.n_edges == ne)]["log2_over_m_mean_mean"].to_numpy(dtype=float)
            def pc(x, q):
                return float(np.percentile(x, q)) if x.size else float("nan")
            ctl.append(dict(designed_unit=cl, variant=var, seed=sd, designed_log2_over_m=val,
                            n_own_draws=int(dv.size),
                            own_draws_p05=pc(dv, 5), own_draws_p25=pc(dv, 25),
                            own_draws_p50=pc(dv, 50), own_draws_p75=pc(dv, 75),
                            own_draws_p95=pc(dv, 95),
                            frac_own_draws_less_efficient=float(np.mean(dv > val)) if dv.size else float("nan"),
                            frac_own_draws_more_efficient=float(np.mean(dv < val)) if dv.size else float("nan"),
                            n_draws_same_edges_own=int(same_draws.size),
                            frac_same_edges_own_less_efficient=float(np.mean(same_draws > val)) if same_draws.size else np.nan,
                            n_draws_same_edges_all=int(all_same.size),
                            frac_same_edges_all_less_efficient=float(np.mean(all_same > val)) if all_same.size else np.nan,
                            n_edges=ne))
    ctl_df = pd.DataFrame(ctl)

    # seed spread (designed units): resplit-mean metrics per seed and the L1 - maxrank_adm delta per seed
    spread = []
    for cl, _m in DESIGNED_EXPECT:
        for fam in ("L0", "L1", "L2"):
            per_seed = []
            for s in SEEDS:
                row = summ[(summ.unit == cl) & (summ.variant == fam) & (summ.seed == s)].iloc[0]
                pr = pair_df[(pair_df.unit == cl) & (pair_df.metric == "mean_joint_set_size") &
                             (pair_df.arm == fam) & (pair_df.seed_arm == s) & (pair_df.comparator == "maxrank_adm")]
                per_seed.append(dict(unit=cl, family=fam, seed=s,
                                     mean_size=float(row["mean_joint_set_size_mean"]),
                                     log2_over_m=float(row["log2_over_m_mean_mean"]),
                                     coverage=float(row["joint_coverage_mean"]),
                                     delta_vs_maxrank_adm=float(pr["mean_diff"].iloc[0])))
            ps = pd.DataFrame(per_seed)
            for rec in per_seed:
                rec["seed_range_mean_size"] = float(ps.mean_size.max() - ps.mean_size.min())
                rec["seed_range_delta_vs_maxrank_adm"] = float(ps.delta_vs_maxrank_adm.max() - ps.delta_vs_maxrank_adm.min())
            spread.extend(per_seed)
    spread_df = pd.DataFrame(spread)

    # headline per designed unit
    head = []
    for cl, m in DESIGNED_EXPECT:
        rec = dict(unit=cl, m=m)
        for var, sd in [("L0", 0), ("L1", 0), ("L2", 0), ("maxrank_adm", -1), ("lpscp_adm", -1),
                        ("bonferroni", -1), ("maxrank", -1), ("indep_per_label", -1), ("sidak", -1)]:
            row = summ[(summ.unit == cl) & (summ.variant == var) & (summ.seed == sd)].iloc[0]
            rec[f"{var}_cov"] = float(row["joint_coverage_mean"])
            rec[f"{var}_size"] = float(row["mean_joint_set_size_mean"])
            rec[f"{var}_log2m"] = float(row["log2_over_m_mean_mean"])
            rec[f"{var}_correct_decisive"] = float(row["correct_decisive_rate_mean"])
            rec[f"{var}_gate"] = str(row["coverage_gate"])
        for ref in ("maxrank_adm", "lpscp_adm"):
            pr = pair_df[(pair_df.unit == cl) & (pair_df.metric == "mean_joint_set_size") &
                         (pair_df.arm == "L1") & (pair_df.seed_arm == 0) & (pair_df.comparator == ref)].iloc[0]
            rec[f"L1_minus_{ref}_size_diff"] = float(pr["mean_diff"])
            rec[f"L1_minus_{ref}_size_ci_lo"] = float(pr["ci_lo_2p5"])
            rec[f"L1_minus_{ref}_size_ci_hi"] = float(pr["ci_hi_97p5"])
        head.append(rec)
    head_df = pd.DataFrame(head)

    return summ, pair_df, ctl_df, spread_df, head_df


def evaluate_triggers(df, summ, ctx, id_summary):
    trig = []
    lrn = summ[summ.variant.isin(["L0", "L1", "L2"])]
    bad_cov = lrn[(lrn.joint_coverage_mean < COV_LO) | (lrn.joint_coverage_mean > COV_HI)]
    if len(bad_cov):
        trig.append(dict(trigger="learned_arm_coverage_outside_band",
                         detail=bad_cov[["family", "unit", "variant", "seed", "joint_coverage_mean"]].to_dict("records")))
    draws = [u for u in ctx["units"] if u["family"] == "random"]
    one_adm = [u["unit"] for u in draws if u["n_admissible_states"] == 1]
    if one_adm:
        trig.append(dict(trigger="random_draw_with_one_admissible_state", detail=one_adm))
    degen = summ[summ.variant.isin(["L0", "L1", "L2", "lpscp_adm"]) & (summ.seed.isin([-1, 0, 1, 2, 3, 4]))]
    dg = degen[np.isclose(degen["median_joint_set_size_mean"], degen["n_admissible"])]
    if len(dg):
        trig.append(dict(trigger="degenerate_set_median_equals_admissible_count",
                         detail=dg[["unit", "variant", "seed"]].to_dict("records")))
    inf_thr = df[(df.resplit == -1) & df.variant.isin(["L0", "L1", "L2", "lpscp_adm"]) & ~np.isfinite(df.threshold)]
    if len(inf_thr):
        trig.append(dict(trigger="infinite_threshold_frozen", detail=inf_thr[["unit", "variant", "seed"]].to_dict("records")))
    num = ["joint_coverage", "mean_joint_set_size", "median_joint_set_size", "log2_over_m_mean",
           "fully_decisive_rate", "correct_decisive_rate", "empty_set_rate"]
    nan_rows = df[df[num].isna().any(axis=1)]
    if len(nan_rows):
        trig.append(dict(trigger="NaN_in_reported_metric", n_rows=int(len(nan_rows)),
                         detail=nan_rows[["unit", "variant", "seed", "resplit"]].head(20).to_dict("records")))
    cap = [d for d in ctx["scorer_diag"] if d.get("hit_cap") is True]
    info = dict(n_scorers_hit_200_cap=len(cap),
                scorers_hit_cap=[dict(unit=d["unit"], scorer=d["scorer"], seed=d["seed"]) for d in cap])
    return trig, info


def write_escalation(trig, extra_lines):
    path = os.path.join(EXP_DIR, "ESCALATION.md")
    lines = ["# Exp055 ESCALATION (automatic, written by run.py)", "",
             "Triggers from the pre-registered list fired during the full run. The run completed; "
             "numbers are reported in README.md but the triggers must be read by Tom before "
             "interpretation.", ""]
    for t in trig:
        lines.append(f"- **{t['trigger']}**: " + json.dumps(t.get("detail", t.get("n_rows", "")), default=str)[:2000])
    lines += ["", *extra_lines]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)

    clusters, spec = cc.load_clusters(CLUSTERS_YAML)
    assert float(spec.get("alpha")) == ALPHA
    assert [(c.name, c.m) for c in clusters] == DESIGNED_EXPECT, [(c.name, c.m) for c in clusters]

    # ---- Gate 0 (frozen CSVs, before any filtering) ----
    splits = cc.load_splits(stem=STEM, with_validation=True, with_train=False, run_gate0=True)
    overlap = cc.gate0_check(splits, raise_on_overlap=True)
    assert all(v == 0 for v in overlap.values()), overlap

    full_rows = {"validation": len(splits.validation), "conformal": len(splits.calibration),
                 "test": len(splits.evaluation)}
    assert full_rows["test"] == EXPECT["test_rows_full"], full_rows

    # ---- population: all-13-clean in every split ----
    val = all13_clean(splits.validation)
    conf13 = all13_clean(splits.calibration)
    test13 = all13_clean(splits.evaluation)
    assert len(val) == EXPECT["val_rows"] and val.subject_id.nunique() == EXPECT["val_subj"], (len(val), val.subject_id.nunique())
    assert len(conf13) == EXPECT["conf_rows"] and conf13.subject_id.nunique() == EXPECT["conf_subj"]
    assert len(test13) == EXPECT["test_rows"] and test13.subject_id.nunique() == EXPECT["test_subj"]
    pool_df = pd.concat([conf13, test13], ignore_index=True)
    assert len(pool_df) == EXPECT["pool_rows"] and pool_df.subject_id.nunique() == EXPECT["pool_subj"]
    assert len(set(conf13.subject_id) & set(test13.subject_id)) == 0
    n_conf = len(conf13)

    # ---- inner holdout (pilot logic, seed 0), shared by every scorer ----
    subjects = np.unique(val["subject_id"].to_numpy())
    rng = np.random.default_rng(HOLDOUT_SEED)
    perm = rng.permutation(subjects)
    n_hold = int(round(HOLDOUT_FRAC * subjects.size))
    hold = perm[:n_hold]
    is_hold = val["subject_id"].isin(set(hold.tolist())).to_numpy()
    val_tr = val[~is_hold].reset_index(drop=True)
    val_ho = val[is_hold].reset_index(drop=True)
    assert len(set(val_tr.subject_id) & set(val_ho.subject_id)) == 0

    # ---- eligibility for random subsets: calibration positives only ----
    yc = conf13[[f"{lab}_true_label" for lab in cc.ALL_LABELS]].to_numpy(dtype=float)
    pos_conf = {lab: int(yc[:, j].sum()) for j, lab in enumerate(cc.ALL_LABELS)}
    eligible = [lab for lab in cc.ALL_LABELS if pos_conf[lab] >= ELIGIBLE_MIN_POS]
    ineligible = [lab for lab in cc.ALL_LABELS if pos_conf[lab] < ELIGIBLE_MIN_POS]
    assert max(c.m for c in clusters) <= len(eligible)

    units = build_units(clusters, eligible)
    for u in units:
        u["n_admissible_states"] = int(cc.hierarchy_admissible(u["labels"], cc.enumerate_states(u["m"])).sum())
        u["n_occupied_val"] = occupancy(val, u["labels"])
    for u in units:
        if u["family"] == "designed":
            assert u["n_edges"] == DESIGNED_EDGES_EXPECT[u["unit"]], (u["unit"], u["n_edges"])

    # ---- Gate 0 on every unit (designed and random) in every split ----
    gate0_hier = {}
    for u in units:
        for name, df_ in [("validation", splits.validation), ("conformal", splits.calibration),
                          ("test", splits.evaluation)]:
            nb, nc = hier_violations(df_, u["labels"])
            gate0_hier[f"{u['unit']}|{name}"] = dict(violations=nb, rows_checked=nc)
            if nb:
                raise SystemExit(f"GATE 0 FAILED: hierarchy violation {name} {u['unit']}: {nb}")

    # ---- splits: frozen draw and 100 subject-level resplits (Exp053 protocol) ----
    subj = pool_df["subject_id"].to_numpy()
    uniq = np.unique(subj)
    splits_list = [(-1, np.arange(n_conf), np.arange(n_conf, len(pool_df)))]
    resplit_sizes = []
    for r in range(N_RESPLITS):
        rng_r = np.random.default_rng(r)
        perm_r = rng_r.permutation(uniq)
        half = len(uniq) // 2
        inA = np.isin(subj, perm_r[:half])
        idxA, idxB = np.flatnonzero(inA), np.flatnonzero(~inA)
        assert len(idxA) + len(idxB) == len(pool_df)
        resplit_sizes.append((len(idxA), len(idxB), len(np.unique(subj[idxA])), len(np.unique(subj[idxB]))))
        splits_list.append((r, idxA, idxB))
    for (nA, nB, sA, sB) in resplit_sizes:
        assert sA == len(uniq) // 2 and sA + sB == len(uniq)

    pool13 = cc.extract_group(pool_df, cc.ALL_LABELS)
    pool_single = {lab: cc.extract_group(pool_df, [lab]) for lab in cc.ALL_LABELS}

    ctx = dict(pool_df=pool_df, val=val, val_tr=val_tr, val_ho=val_ho, pool13=pool13,
               pool_single=pool_single, splits=splits_list, pilot_check=[], units=units,
               scorer_diag=[])

    # ---- identity gate (before anything is interpreted) ----
    id_df, id_summary = identity_gate(ctx)
    id_df.to_csv(os.path.join(EXP_DIR, "identity_gates.csv"), index=False)
    print("identity gate:", json.dumps(id_summary, indent=2))
    if not id_summary["passed"]:
        path = write_escalation([dict(trigger="identity_gate_failed", detail=id_summary)],
                                ["Stopped before the main run (identity gate, pre-registered stop rule).",
                                 "No learned-score or reference metric from the full run was computed."])
        print("IDENTITY GATE FAILED; wrote", path)
        raise SystemExit(2)

    # ---- main loop ----
    all_rows, all_diag, pilot_checks = [], [], []
    for u in units:
        t1 = time.time()
        rows, diag = eval_unit(u, ctx)
        all_rows.extend(rows)
        ctx["scorer_diag"].extend(diag)
        all_diag.extend(diag)
        print(f"[{u['family']:>8}] {u['unit']:<24} m={u['m']} edges={u['n_edges']} "
              f"adm={u['n_admissible_states']:>2}  rows={len(rows)}  {time.time() - t1:.1f}s", flush=True)
    pilot_checks = ctx["pilot_check"]
    if not all(pc["match"] for pc in pilot_checks):
        write_escalation([dict(trigger="pilot_reproduction_mismatch", detail=pilot_checks)], [])
        raise SystemExit("pilot reproduction mismatch; escalated")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(EXP_DIR, "resplit_metrics.csv"), index=False)
    diag_df = pd.DataFrame(all_diag)
    diag_df.to_csv(os.path.join(EXP_DIR, "scorer_diagnostics.csv"), index=False)

    unit_df = pd.DataFrame([{k: (v if not isinstance(v, list) else "|".join(v)) for k, v in u.items()} for u in units])
    unit_df.to_csv(os.path.join(EXP_DIR, "units.csv"), index=False)

    summ, pair_df, ctl_df, spread_df, head_df = run_analysis(df, ctx)
    summ.to_csv(os.path.join(EXP_DIR, "summary_by_arm.csv"), index=False)
    pair_df.to_csv(os.path.join(EXP_DIR, "paired_deltas.csv"), index=False)
    ctl_df.to_csv(os.path.join(EXP_DIR, "random_control.csv"), index=False)
    spread_df.to_csv(os.path.join(EXP_DIR, "seed_spread.csv"), index=False)
    head_df.to_csv(os.path.join(EXP_DIR, "headline.csv"), index=False)

    trig, trig_info = evaluate_triggers(df, summ, ctx, id_summary)
    metrics = dict(
        gate0=dict(overlap_counts=overlap, full_rows=full_rows, all13_clean_rows=dict(
            validation=len(val), conformal=len(conf13), test=len(test13)),
            all13_clean_subjects=dict(validation=int(val.subject_id.nunique()), conformal=int(conf13.subject_id.nunique()),
                                      test=int(test13.subject_id.nunique())),
            pool_rows=len(pool_df), pool_subjects=int(pool_df.subject_id.nunique()),
            hierarchy_violations_all_zero=all(v["violations"] == 0 for v in gate0_hier.values()),
            units_checked=len(units), splits_per_unit=3),
        eligibility=dict(pos_conf_all13_clean=pos_conf, eligible=eligible, ineligible=ineligible,
                         min_positive=ELIGIBLE_MIN_POS),
        identity_gate=id_summary,
        pilot_reproduction=pilot_checks,
        triggers=trig, trigger_info=trig_info,
        holdout=dict(n_validation_subjects=int(subjects.size), n_holdout_subjects=n_hold,
                     n_train_rows=len(val_tr), n_holdout_rows=len(val_ho)),
        resplit_sizes=dict(n_pool_subjects=int(uniq.size), per_resplit_A_B_subjects=resplit_sizes[:3]),
        runtime_seconds=time.time() - t0,
    )
    with open(os.path.join(EXP_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    if trig:
        write_escalation(trig, [f"Scope: {SCOPE}.", "See README.md, Results section."])

    inputs = {os.path.basename(p): sha256(p) for p in [
        os.path.join(cc.OUTPUTS_DIR, STEM + s) for s in ("_validation.csv", "_conformal.csv", "_test.csv")]}
    code = {os.path.relpath(p, REPO): sha256(p) for p in [
        os.path.join(REPO, "conformal_common.py"), os.path.join(REPO, "bonferrony.py"),
        os.path.join(REPO, "max_rank.py"), os.path.join(REPO, "lp_scp.py"),
        os.path.join(REPO, "data", "hierarchy.py"), CLUSTERS_YAML,
        PILOT_RESULTS, os.path.abspath(__file__)]}
    run_info = dict(
        git_head=git("rev-parse", "HEAD"),
        git_working_tree_dirty_entries=len(git("status", "--porcelain").splitlines()),
        seeds=dict(torch_model_seeds=list(SEEDS), holdout_rng=HOLDOUT_SEED, resplit_rng=f"0..{N_RESPLITS-1}",
                   random_subset_rng=f"{DRAW_SEED_BASE}+idx (idx=20*cluster_position+draw)"),
        torch_num_threads=1, package_versions=package_versions(),
        input_sha256=inputs, code_sha256=code, stem=STEM,
        command="/usr/bin/python3 experiments/055_learned_score_full/run.py",
        runtime_seconds=metrics["runtime_seconds"],
    )
    with open(os.path.join(EXP_DIR, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print("done; triggers:", [t["trigger"] for t in trig])
    print("wrote outputs to", EXP_DIR)


if __name__ == "__main__":
    main()
