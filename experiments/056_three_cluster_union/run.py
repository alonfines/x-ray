#!/usr/bin/env python3
"""Exp056 - three-cluster partition of all 13 labels, L1 per cluster, coverage of the union.

Implements experiments/056_three_cluster_union/SPEC.md (Tom Hagan, 2026-11-16) exactly, with
everything not stated there taken from Exp055 (experiments/055_learned_score_full/run.py).
Pre-registered in README.md before any frozen-split metric was computed.

Reused unedited: conformal_common.py, conformal_methods/{max_rank,bonferrony,mlp_score}.py,
data/hierarchy.py. Helpers that mirror Exp055 (identity gate, reference regions, subset,
all-13-clean filter, hierarchy violation count) are re-written here, not imported from the
finished Exp055 script.

Modes:
    /usr/bin/python3 experiments/056_three_cluster_union/run.py --prereg
        label-only quantities for the pre-registration (no probabilities, no metrics)
    /usr/bin/python3 experiments/056_three_cluster_union/run.py
        full run (Gate 0, identity gate, reproduction check, then the experiment)
"""

from __future__ import annotations

import argparse
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
import yaml  # noqa: E402

import conformal_common as cc  # noqa: E402
from conformal_methods import bonferrony as bf  # noqa: E402
from conformal_methods import max_rank as mr  # noqa: E402
from conformal_methods import mlp_score as ms  # noqa: E402  (diagnostic import only, see A1.3)
import importlib.util as _ilu  # noqa: E402
_spec055 = _ilu.spec_from_file_location(
    "exp055_run", os.path.join(REPO, "experiments", "055_learned_score_full", "run.py"))
e55 = _ilu.module_from_spec(_spec055)          # Amendment 1 (A1.1): Exp055's own loop, imported unedited
_spec055.loader.exec_module(e55)
from data.hierarchy import CHEXPERT_HIERARCHY  # noqa: E402

STEM = "densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809"
EXP055_DIR = os.path.join(REPO, "experiments", "055_learned_score_full")
ALPHA = 0.10
SEEDS = (0, 1, 2, 3, 4)            # L1 model seeds on the designed partition (seed 0 primary)
RANDOM_SEED = 0                    # seed on random partitions and for arm D
HOLDOUT_FRAC = 0.20
HOLDOUT_SEED = 0
N_RESPLITS = 100                   # resplit r = 0..99, default_rng(r)
N_RANDOM = 20                      # random partitions, default_rng(5600 + i)
RANDOM_SEED_BASE = 5600
IDENTITY_TOL = 0.005
COV_LO, COV_HI = 0.895, 0.905
DUP_ESCALATE = 5
GUARD_GAP_ESCALATE = 0.02          # 2 pp, on any designed cluster

EXPECT = dict(val_rows=18570, val_subj=5898, conf_rows=18754, conf_subj=5926,
              test_rows_full=24394, test_rows=18877, test_subj=5932,
              pool_rows=37631, pool_subj=11858)

# Designed partition P_H (SPEC section 1). Canonical label order within each cluster.
P_H = [
    ("C1", ["Enlarged Cardiomediastinum", "Cardiomegaly"]),
    ("C2", ["Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis"]),
    ("C3", ["Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]),
]
PH_SIZES = (2, 6, 5)
POS_SUBSETS = ["Pleural Other", "Fracture", "Lung Lesion"]   # conditional-coverage diagnostics
SCOPE = "all-13-clean studies, 77% of test"

ARMS_CLUSTER = ("A", "B", "Bprime", "C")
REFS_13 = ("indep_per_label", "bonferroni", "sidak", "maxrank", "maxrank_adm", "bonferroni_adm")

LABELS = list(cc.ALL_LABELS)
N13 = len(LABELS)


# ---------------------------------------------------------------------------
# small helpers
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
    import lightning
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "lightning": lightning.__version__, "pyyaml": yaml.__version__}


def canon(lab):
    return LABELS.index(lab)


def all13_clean(df):
    """Rows with every one of the 13 true labels in {0,1} and every prob finite (Exp055 logic)."""
    y = df[[f"{lab}_true_label" for lab in LABELS]].to_numpy(dtype=float)
    p = df[[f"{lab}_prob" for lab in LABELS]].to_numpy(dtype=float)
    keep = np.all(np.isin(y, [0.0, 1.0]), axis=1) & np.all(np.isfinite(p), axis=1)
    return df.loc[keep].reset_index(drop=True)


def hier_violations(df, labels):
    """(violating vectors, rows checked) among rows whose members are all in {0,1}."""
    y = df[[f"{lab}_true_label" for lab in labels]].to_numpy(dtype=float)
    ok = np.all(np.isin(y, [0.0, 1.0]), axis=1)
    v = y[ok].astype(int)
    return int((~cc.hierarchy_admissible(labels, v)).sum()), int(ok.sum())


def edges_within(labels):
    """Directed parent->child pairs with both endpoints in `labels`."""
    L = set(labels)
    return int(sum(1 for ch, ps in CHEXPERT_HIERARCHY.items() if ch in L for p in ps if p in L))


def edges_across(parts):
    """Directed parent->child pairs whose endpoints lie in different clusters."""
    where = {lab: i for i, c in enumerate(parts) for lab in c}
    n = 0
    for ch, ps in CHEXPERT_HIERARCHY.items():
        for p in ps:
            if where[ch] != where[p]:
                n += 1
    return n


def subset(g, idx):
    return cc.GroupData(labels=g.labels, probs=g.probs[idx], y=g.y[idx], scores=g.scores[idx],
                        n_rows_before_filter=int(len(idx)),
                        subject_id=None if g.subject_id is None else g.subject_id[idx])


def product_region(group, thr):
    """Product of per-label prediction sets as a region over cc-ordered joint states (n, 2^m)."""
    sets = cc.prediction_sets(group, thr)
    states = cc.enumerate_states(group.m)
    reg = np.ones((group.n, states.shape[0]), dtype=bool)
    for j in range(group.m):
        reg &= sets[:, j, :][:, states[:, j]]
    return reg


def refs_eval_13(gA, gB, adm):
    """Classical 13-label references (Exp055 section 5 convention, no guard). Returns name -> (regA, regB, k)."""
    thr_i, di = bf.calibrate_per_label(gA, ALPHA, "none")
    thr_b, db = bf.calibrate_per_label(gA, ALPHA, "bonferroni")
    thr_s, ds = bf.calibrate_per_label(gA, ALPHA, "sidak")
    thr_m, dm = mr.calibrate_max_rank(gA, ALPHA)
    for d in (di, db, ds, dm):
        if d.get("degenerate", False):
            raise SystemExit("ESCALATE: degenerate 13-label reference threshold")
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


def cond_cov(covered, mask):
    n = int(mask.sum())
    return (float(covered[mask].mean()) if n else np.nan), n


def vector_summary(sizes, covered, viol_full, m):
    """Locked efficiency and coverage metrics for one joint region per row.

    sizes: int (n,), covered: bool (n,), viol_full: bool (n,) True where a decisive row's
    chosen state is hierarchy-inadmissible (None -> NaN).
    """
    ne = sizes >= 1
    dec = sizes == 1
    out = dict(
        joint_coverage=float(covered.mean()),
        mean_joint_set_size=float(sizes.mean()),
        median_joint_set_size=float(np.median(sizes)),
        empty_set_rate=float((sizes == 0).mean()),
        log2_over_m_mean=float(np.log2(sizes[ne]).mean() / m) if ne.any() else np.nan,
        fully_decisive_rate=float(dec.mean()),
        correct_decisive_rate=float((dec & covered).mean()),
        decisive_accuracy=float((dec & covered).sum() / dec.sum()) if dec.any() else np.nan,
    )
    out["hierarchy_violation_rate_decisive"] = (float(viol_full[dec].mean()) if (dec.any() and viol_full is not None)
                                                else np.nan)
    return out


# ---------------------------------------------------------------------------
# partitions
# ---------------------------------------------------------------------------

def partition_key(parts):
    return frozenset(frozenset(c) for c in parts)


def draw_random_partitions():
    """Spec section 1: 20 partitions, default_rng(5600 + i).permutation(ALL_LABELS); positions
    0-1 -> R1, 2-7 -> R2, 8-12 -> R3; reject duplicates of P_H or earlier draws by re-drawing
    with the sub-seed 5600 + i + 1000*j (j = attempt index). All attempts are recorded."""
    seen = {partition_key([c for _, c in P_H])}
    out, log = [], []
    for i in range(N_RANDOM):
        j = 0
        while True:
            seed = RANDOM_SEED_BASE + i + 1000 * j
            perm = np.random.default_rng(seed).permutation(np.array(LABELS, dtype=object))
            p = [str(x) for x in perm]
            parts = [sorted(p[0:2], key=canon), sorted(p[2:8], key=canon), sorted(p[8:13], key=canon)]
            key = partition_key(parts)
            if key in seen:
                log.append(dict(draw=i, attempt=j, seed=seed, duplicate=True))
                j += 1
                continue
            seen.add(key)
            log.append(dict(draw=i, attempt=j, seed=seed, duplicate=False))
            break
        out.append(dict(pid=f"R{i:02d}", family="random", draw=i, seed_used=seed, attempts=j + 1,
                        clusters=[("R1", parts[0]), ("R2", parts[1]), ("R3", parts[2])]))
    return out, log


def build_partitions():
    designed = dict(pid="P_H", family="designed", draw=-1, seed_used=-1, attempts=1, clusters=list(P_H))
    randoms, log = draw_random_partitions()
    return [designed] + randoms, log


# ---------------------------------------------------------------------------
# stop / escalation
# ---------------------------------------------------------------------------

def write_escalation(triggers, lines):
    path = os.path.join(EXP_DIR, "ESCALATION.md")
    out = ["# Exp056 ESCALATION (written by run.py)", "",
           f"Scope: {SCOPE}.", ""]
    for t in triggers:
        out.append(f"- **{t['trigger']}**: " + json.dumps(t.get("detail", ""), default=str)[:3000])
    out += ["", *lines]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


class Stop(Exception):
    def __init__(self, trigger, detail, lines=()):
        super().__init__(trigger)
        self.trigger, self.detail, self.lines = trigger, detail, list(lines)


# ---------------------------------------------------------------------------
# label-only pre-registration quantities (no probabilities, no metrics)
# ---------------------------------------------------------------------------

def prereg_labels():
    out = {}
    paths = {s: os.path.join(cc.OUTPUTS_DIR, STEM + suf)
             for s, suf in (("validation", "_validation.csv"), ("conformal", "_conformal.csv"), ("test", "_test.csv"))}
    out["input_sha256_full"] = {os.path.basename(p): sha256(p) for p in paths.values()}
    lab_cols = [f"{lab}_true_label" for lab in LABELS]
    ids = ["subject_id", "dicom_id"]
    frames = {s: pd.read_csv(p, usecols=ids + lab_cols) for s, p in paths.items()}
    for s, f in frames.items():
        y = f[lab_cols].to_numpy(dtype=float)
        ok = np.all(np.isin(y, [0.0, 1.0]), axis=1)
        out[f"{s}_rows_all13_label_clean"] = int(ok.sum())
        out[f"{s}_subjects_all13_label_clean"] = int(f.loc[ok, "subject_id"].nunique())
    conf = frames["conformal"]
    yc = conf[lab_cols].to_numpy(dtype=float)
    okc = np.all(np.isin(yc, [0.0, 1.0]), axis=1)
    yc = yc[okc].astype(int)
    out["conformal_positives_all13_label_clean"] = {lab: int(yc[:, j].sum()) for j, lab in enumerate(LABELS)}
    # full-hierarchy admissibility of the true conformal states (stop condition for arm D)
    out["conformal_full_hierarchy_violations_label_clean"] = int((~cc.hierarchy_admissible(LABELS, yc)).sum())
    # admissible-state counts and the factorisation claimed in SPEC section 4
    adm13 = cc.hierarchy_admissible(LABELS, cc.enumerate_states(N13))
    lung6 = [LABELS.index(l) for l in ["Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis"]]
    ecm = [0, 1]
    st = cc.enumerate_states(N13)
    n_ecm = int(cc.hierarchy_admissible(["Enlarged Cardiomediastinum", "Cardiomegaly"], cc.enumerate_states(2)).sum())
    n_lung = int(cc.hierarchy_admissible([LABELS[i] for i in lung6], cc.enumerate_states(6)).sum())
    n_root = 2 ** 5
    out["admissible_13"] = int(adm13.sum())
    out["admissible_factorisation"] = dict(ECM=n_ecm, lung6=n_lung, roots5=n_root, product=n_ecm * n_lung * n_root)
    assert int(adm13.sum()) == n_ecm * n_lung * n_root, "admissible factorisation mismatch"
    # designed partition: within-cluster edges, cross-cluster edges, modal-state mass on conformal
    out["PH"] = []
    for name, labs in P_H:
        idx = [LABELS.index(l) for l in labs]
        ys = yc[:, idx]
        codes = cc.state_indices(ys)
        mode_share = float(np.bincount(codes, minlength=2 ** len(idx)).max() / codes.size)
        out["PH"].append(dict(name=name, m=len(labs), within_edges=edges_within(labs),
                              modal_state_mass_conformal=mode_share,
                              admissible=int(cc.hierarchy_admissible(labs, cc.enumerate_states(len(labs))).sum())))
    out["PH_cross_cluster_edges"] = edges_across([c for _, c in P_H])
    return out


# ---------------------------------------------------------------------------
# learned scorers (Lightning port, conformal_methods/mlp_score.py)
# ---------------------------------------------------------------------------

def train_cluster_scorer(labels, seed, ctx):
    """Train L1 on the cluster's m logits. Returns the cluster record (cc state order)."""
    L = list(labels)
    m = len(L)
    S = 2 ** m
    w = e55.spec_weights(m)
    states_cc = cc.enumerate_states(m)
    spec_of_cc = (states_cc * w).sum(axis=1)
    adm = cc.hierarchy_admissible(L, states_cc)

    g_tr = cc.extract_group(ctx["val_tr"], L)
    g_ho = cc.extract_group(ctx["val_ho"], L)
    gp = cc.extract_group(ctx["pool_df"], L)
    assert gp.n == EXPECT["pool_rows"]
    ytr = torch.tensor((g_tr.y * w).sum(axis=1), dtype=torch.long)
    yho = torch.tensor((g_ho.y * w).sum(axis=1), dtype=torch.long)
    Xtr = torch.tensor(e55.logit(g_tr.probs), dtype=torch.float32)
    Xho = torch.tensor(e55.logit(g_ho.probs), dtype=torch.float32)
    Xp = torch.tensor(e55.logit(gp.probs), dtype=torch.float32)

    mod, info = e55.train_scorer(Xtr, ytr, Xho, yho, S, seed, hidden=True)
    raw_spec = e55.neg_log_softmax_spec(mod, Xp)               # (N, S) float64, spec order
    raw = raw_spec[:, spec_of_cc].astype(np.float64)          # cc order
    del raw_spec
    ycc = cc.state_indices(gp.y)
    assert np.array_equal(spec_of_cc[ycc], (gp.y * w).sum(axis=1)), "state-order mapping mismatch"
    return dict(labels=L, m=m, S=S, raw=raw, smask=np.where(adm[None, :], raw, np.inf), adm=adm,
                states=states_cc, ycc=ycc, info=info, seed=seed)


def get_cluster(labels, seed, ctx):
    key = (tuple(labels), seed)
    if key not in ctx["cache"]:
        ctx["cache"][key] = train_cluster_scorer(labels, seed, ctx)
    return ctx["cache"][key]


def cluster_region(e, thr, idx, guard):
    """Region over cc states for row indices idx; admissibility masked explicitly.
    Guard (SPEC section 3): per-cluster argmin added only for rows whose region is empty."""
    reg = e["adm"][None, :] & (e["raw"][idx] <= thr)
    if guard:
        empty = ~reg.any(axis=1)
        if empty.any():
            am = e["smask"][idx].argmin(axis=1)
            reg[np.flatnonzero(empty), am[empty]] = True
    return reg


# ---------------------------------------------------------------------------
# evaluation of a 3-cluster partition (arms A, B, B', C)
# ---------------------------------------------------------------------------

def eval_cluster_partition(part, seed, ctx, rows, crows):
    names = [n for n, _ in part["clusters"]]
    labs = [c for _, c in part["clusters"]]
    K = len(names)
    ents = [get_cluster(l, seed, ctx) for l in labs]
    pos = [[canon(x) for x in l] for l in labs]
    for e, nm in zip(ents, names):
        e["name"] = nm
    pool13_y = ctx["pool13_y"]                                # (N, 13) truth in LABELS order
    for e in ents:
        assert e["adm"][e["ycc"]].all(), f"ESCALATE: true cluster state inadmissible in {part['pid']} {e['name']}"

    for r, idxA, idxB in ctx["splits"]:
        nA, nB = idxA.size, idxB.size
        S_A = np.column_stack([e["raw"][idxA, e["ycc"][idxA]] for e in ents])
        gA = cc.GroupData(labels=names, probs=np.zeros((nA, K)), y=np.zeros((nA, K), dtype=int),
                          scores=S_A, n_rows_before_filter=nA, subject_id=None)
        thr_sets = {
            "A": bf.calibrate_per_label(gA, ALPHA, "none"),
            "B": bf.calibrate_per_label(gA, ALPHA, "bonferroni"),
            "Bprime": bf.calibrate_per_label(gA, ALPHA, "sidak"),
            "C": mr.calibrate_max_rank(gA, ALPHA),
        }
        for arm, (thr, diag) in thr_sets.items():
            if diag.get("degenerate", False):
                raise Stop("degenerate_threshold", dict(pid=part["pid"], arm=arm, resplit=int(r)))
            if not all(np.isfinite(v) for v in thr.values()):
                raise Stop("infinite_threshold", dict(pid=part["pid"], arm=arm, resplit=int(r), thr=thr))
            k = int(diag["k"]) if "k" in diag else int(diag["k_star"])
            for guard in (False, True):
                regA = [cluster_region(e, thr[e["name"]], idxA, guard) for e in ents]
                regB = [cluster_region(e, thr[e["name"]], idxB, guard) for e in ents]
                sizes_c = [reg.sum(axis=1) for reg in regB]
                sizes = np.ones(nB, dtype=np.int64)
                for s_ in sizes_c:
                    sizes = sizes * s_
                cov_c = [reg[np.arange(nB), e["ycc"][idxB]] for reg, e in zip(regB, ents)]
                covered = np.logical_and.reduce(cov_c)
                cov_c_A = [reg[np.arange(nA), e["ycc"][idxA]] for reg, e in zip(regA, ents)]
                covA = np.logical_and.reduce(cov_c_A)

                # hierarchy violation of decisive output (assembled 13-label vector)
                dec = sizes == 1
                viol = np.zeros(nB, dtype=bool)
                if dec.any():
                    V = np.zeros((int(dec.sum()), N13), dtype=int)
                    for reg, e, p_ in zip(regB, ents, pos):
                        ch = reg[dec].argmax(axis=1)
                        V[:, p_] = e["states"][ch]
                    viol[dec] = ~cc.hierarchy_admissible(LABELS, V)

                gname = "guarded" if guard else "unguarded"
                vs = vector_summary(sizes, covered, viol, N13)
                cond = {}
                yB13 = pool13_y[idxB]
                masks = {f"pos_{l.replace(' ', '_')}": yB13[:, LABELS.index(l)] == 1 for l in POS_SUBSETS}
                masks["any_pos"] = yB13.sum(axis=1) >= 1
                for mk, mm in masks.items():
                    cond[f"cov_{mk}"], cond[f"n_{mk}"] = cond_cov(covered, mm)
                rec = dict(pid=part["pid"], family=part["family"], arm=arm, guard=gname, scorer_seed=seed,
                           resplit=int(r), n_cal=nA, n_test=nB, k=k, cal_side_coverage=float(covA.mean()),
                           prod_cluster_coverage=float(np.prod([c.mean() for c in cov_c])),
                           **vs, **cond)
                for nm, t in thr.items():
                    rec[f"thr_{nm}"] = float(t)
                rows.append(rec)

                covs = []
                for c_i, (nm, reg, e, p_) in enumerate(zip(names, regB, ents, pos)):
                    cov_i = cov_c[c_i]
                    covs.append(cov_i)
                    sc = sizes_c[c_i]
                    vs_c = vector_summary(sc, cov_i, None, e["m"])
                    crows.append(dict(pid=part["pid"], family=part["family"], cluster=nm, m=e["m"],
                                      arm=arm, guard=gname, scorer_seed=seed, resplit=int(r),
                                      cal_side_coverage=float(cov_c_A[c_i].mean()),
                                      threshold=float(thr[nm]), **{k_: v_ for k_, v_ in vs_c.items()
                                                                  if k_ != "hierarchy_violation_rate_decisive"}))
                miss = [~c for c in covs]
                for i in range(K):
                    for j in range(i + 1, K):
                        a, b = miss[i].astype(float), miss[j].astype(float)
                        phi = np.nan if (a.std() == 0 or b.std() == 0) else float(np.corrcoef(a, b)[0, 1])
                        rows[-1][f"phi_{names[i]}_{names[j]}"] = phi


# ---------------------------------------------------------------------------
# 13-label materialised regions: references and arm D
# ---------------------------------------------------------------------------

def eval_13_region(rec_base, regA, regB, yA, yB, states13, rows):
    nA, nB = regA.shape[0], regB.shape[0]
    sizes = regB.sum(axis=1)
    covered = regB[np.arange(nB), yB]
    viol = np.zeros(nB, dtype=bool)
    dec = sizes == 1
    if dec.any():
        chosen = states13[regB[dec].argmax(axis=1)]
        viol[dec] = ~cc.hierarchy_admissible(LABELS, chosen)
    vs = vector_summary(sizes, covered, viol, N13)
    return vs, covered, sizes


def refs_for_splits(ctx, rows):
    pool13 = ctx["pool13"]
    adm13 = ctx["adm13"]
    for r, idxA, idxB in ctx["splits"]:
        gA = subset(pool13, idxA)
        gB = subset(pool13, idxB)
        refs = refs_eval_13(gA, gB, adm13)
        for name, (regA, regB, k) in refs.items():
            yA, yB = ctx["y13cc"][idxA], ctx["y13cc"][idxB]
            vs, covered, sizes = eval_13_region(None, regA, regB, yA, yB, ctx["states13"], rows)
            cal = float(regA[np.arange(regA.shape[0]), yA].mean())
            rows.append(dict(pid="none", family="unclustered", arm=name, guard="none", scorer_seed=-1,
                             resplit=int(r), n_cal=int(idxA.size), n_test=int(idxB.size), k=int(k),
                             cal_side_coverage=cal, **vs))


def eval_full13(ctx, rows):
    """Arm D: L1 on all 13 logits (8192 states), one quantile at alpha, unguarded primary and guarded."""
    e = get_cluster(LABELS, RANDOM_SEED, ctx)              # trained once; cached as float64 raw
    raw = e["raw"].astype(np.float32)
    smask = np.where(e["adm"][None, :], raw, np.float32(np.inf))
    del e["raw"], e["smask"]
    yy = ctx["y13cc"]
    for r, idxA, idxB in ctx["splits"]:
        nA, nB = idxA.size, idxB.size
        s_true = raw[idxA, yy[idxA]]
        assert e["adm"][yy[idxA]].all(), "ESCALATE: true 13-label state inadmissible under full hierarchy"
        k = cc.conformal_rank(nA, ALPHA)
        thr = cc.order_statistic(s_true, k)
        if not np.isfinite(thr):
            raise Stop("infinite_threshold", dict(pid="D", resplit=int(r)))
        for guard in (False, True):
            regA = (e["adm"][None, :] & (raw[idxA] <= thr))
            regB = (e["adm"][None, :] & (raw[idxB] <= thr))
            if guard:
                for reg, idx in ((regA, idxA), (regB, idxB)):
                    empty = ~reg.any(axis=1)
                    if empty.any():
                        am = smask[idx].argmin(axis=1)
                        reg[np.flatnonzero(empty), am[empty]] = True
            vs, covered, sizes = eval_13_region(None, regA, regB, yy[idxA], yy[idxB], e["states"], rows)
            cal = float(regA[np.arange(nA), yy[idxA]].mean())
            rows.append(dict(pid="none", family="unclustered", arm="D", guard="guarded" if guard else "unguarded",
                             scorer_seed=RANDOM_SEED, resplit=int(r), n_cal=nA, n_test=nB, k=int(k),
                             cal_side_coverage=cal, thr_13=float(thr), **vs))
    return e


# ---------------------------------------------------------------------------
# identity gate (Exp055 section 7, m = 1, 13 labels)
# ---------------------------------------------------------------------------

def identity_gate(ctx):
    rows = []
    for (r, idxA, idxB) in ctx["splits"]:
        for lab in LABELS:
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

VEC_METRICS = ["joint_coverage", "cal_side_coverage", "prod_cluster_coverage", "mean_joint_set_size",
               "median_joint_set_size", "empty_set_rate", "log2_over_m_mean", "fully_decisive_rate",
               "correct_decisive_rate", "decisive_accuracy", "hierarchy_violation_rate_decisive"]


def group_summary(df, keys, metrics):
    out = []
    for kv, g in df.groupby(keys, sort=False, dropna=False):
        rs = g[g.resplit >= 0].sort_values("resplit")
        fr = g[g.resplit == -1]
        rec = dict(zip(keys, kv if isinstance(kv, tuple) else (kv,)))
        rec["n_resplits"] = int(rs.shape[0])
        for mt in metrics:
            if mt not in rs.columns:
                continue
            v = rs[mt].to_numpy(dtype=float)
            rec[f"{mt}_mean"] = float(np.nanmean(v))
            rec[f"{mt}_sd"] = float(np.nanstd(v, ddof=1))
            rec[f"{mt}_p2.5"] = float(np.nanpercentile(v, 2.5))
            rec[f"{mt}_p97.5"] = float(np.nanpercentile(v, 97.5))
            rec[f"{mt}_frozen"] = float(fr[mt].iloc[0]) if len(fr) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def paired(df, a, b, metric):
    """a, b: dicts of column filters. Within-resplit paired difference a - b over resplits 0..99."""
    def pick(f):
        m = np.ones(len(df), dtype=bool)
        for k_, v_ in f.items():
            m &= (df[k_] == v_).to_numpy()
        sub = df[m & (df.resplit >= 0)].sort_values("resplit")
        assert len(sub) == N_RESPLITS, (f, len(sub))
        return sub[metric].to_numpy(dtype=float)
    d = pick(a) - pick(b)
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    return dict(metric=metric, mean_diff=float(d.mean()), ci_lo_2p5=float(lo), ci_hi_97p5=float(hi),
                ci_excludes_zero=bool(lo > 0 or hi < 0), n=int(d.size))


def gate_status(arm, cov_mean, cluster_cov_means):
    if arm == "A":
        ok = all(COV_LO <= c <= COV_HI for c in cluster_cov_means)
        return "pass" if ok else "outside-band"
    if arm in ("B", "Bprime"):
        return "valid" if cov_mean >= 0.90 else "invalid"
    if arm in ("C", "D", "maxrank_adm"):
        return "pass" if COV_LO <= cov_mean <= COV_HI else "outside-band"
    return "not-gated"


def run_analysis(vec, clu, ctx, dup_count, id_summary):
    res = {}
    V = vec
    # summaries
    vsum = group_summary(V, ["pid", "family", "arm", "guard", "scorer_seed"],
                         VEC_METRICS + ["mean_joint_set_size", "k", "n_cal"])
    csum = group_summary(clu, ["pid", "family", "cluster", "m", "arm", "guard", "scorer_seed"],
                         ["joint_coverage", "log2_over_m_mean", "mean_joint_set_size", "empty_set_rate",
                          "fully_decisive_rate", "correct_decisive_rate", "cal_side_coverage"])
    # gates (unguarded = primary); cluster coverage per partition
    gates = []
    for _, r in vsum.iterrows():
        if r["guard"] != "unguarded":
            continue
        cl = csum[(csum.pid == r["pid"]) & (csum.arm == r["arm"]) & (csum.guard == "unguarded") &
                  (csum.scorer_seed == r["scorer_seed"])]
        ccov = cl["joint_coverage_mean"].tolist() if len(cl) else []
        gates.append(dict(pid=r["pid"], arm=r["arm"], scorer_seed=r["scorer_seed"],
                          joint_cov_mean=r["joint_coverage_mean"],
                          cluster_cov_means=ccov,
                          status=gate_status(r["arm"], r["joint_coverage_mean"], ccov)))
    # the unclustered references and arm D
    for _, r in vsum[(vsum.pid == "none")].iterrows():
        if r["guard"] not in ("unguarded", "none"):
            continue
        st = gate_status(r["arm"], r["joint_coverage_mean"], [])
        gates.append(dict(pid="none", arm=r["arm"], scorer_seed=r["scorer_seed"],
                          joint_cov_mean=r["joint_coverage_mean"], cluster_cov_means=[], status=st))
    res["gates"] = gates
    res["vsum"] = vsum
    res["csum"] = csum
    return res


def compute_guard_gaps(vec, clu):
    out = []
    for (pid, arm, seed), g in vec.groupby(["pid", "arm", "scorer_seed"]):
        if arm not in ARMS_CLUSTER and arm != "D":
            continue
        gu = g[(g.guard == "unguarded") & (g.resplit >= 0)].sort_values("resplit")
        gg = g[(g.guard == "guarded") & (g.resplit >= 0)].sort_values("resplit")
        if not len(gu) or not len(gg):
            continue
        out.append(dict(pid=pid, arm=arm, scorer_seed=seed, level="vector",
                        cluster="all", guard_gap_mean=float((gg.joint_coverage.values - gu.joint_coverage.values).mean())))
    for (pid, arm, cl, seed), g in clu.groupby(["pid", "arm", "cluster", "scorer_seed"]):
        gu = g[(g.guard == "unguarded") & (g.resplit >= 0)].sort_values("resplit")
        gg = g[(g.guard == "guarded") & (g.resplit >= 0)].sort_values("resplit")
        if not len(gu) or not len(gg):
            continue
        out.append(dict(pid=pid, arm=arm, scorer_seed=seed, level="cluster", cluster=cl,
                        guard_gap_mean=float((gg.joint_coverage.values - gu.joint_coverage.values).mean())))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", action="store_true")
    args = ap.parse_args(argv)

    if args.prereg:
        out = prereg_labels()
        with open(os.path.join(EXP_DIR, "prereg_labels.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        return 0

    t0 = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)

    ctx = dict(cache={})
    try:
        # ---- Gate 0 (frozen CSVs, before any filtering) ----
        splits = cc.load_splits(stem=STEM, with_validation=True, with_train=False, run_gate0=True)
        overlap = cc.gate0_check(splits, raise_on_overlap=True)
        if not all(v == 0 for v in overlap.values()):
            raise Stop("gate0_overlap", overlap)
        full_rows = {"validation": len(splits.validation), "conformal": len(splits.calibration),
                     "test": len(splits.evaluation)}
        if full_rows["test"] != 24394:
            raise Stop("gate0_population", full_rows)

        # ---- population: all-13-clean in every split (Exp055 section 2) ----
        val = all13_clean(splits.validation)
        conf13 = all13_clean(splits.calibration)
        test13 = all13_clean(splits.evaluation)
        if not (len(val) == EXPECT["val_rows"] and val.subject_id.nunique() == EXPECT["val_subj"]):
            raise Stop("population_mismatch", dict(split="validation", rows=len(val)))
        if not (len(conf13) == EXPECT["conf_rows"] and conf13.subject_id.nunique() == EXPECT["conf_subj"]):
            raise Stop("population_mismatch", dict(split="conformal", rows=len(conf13)))
        if not (len(test13) == EXPECT["test_rows"] and test13.subject_id.nunique() == EXPECT["test_subj"]):
            raise Stop("population_mismatch", dict(split="test", rows=len(test13)))
        pool_df = pd.concat([conf13, test13], ignore_index=True)
        if not (len(pool_df) == EXPECT["pool_rows"] and pool_df.subject_id.nunique() == EXPECT["pool_subj"]):
            raise Stop("population_mismatch", dict(split="pool", rows=len(pool_df)))
        if len(set(conf13.subject_id) & set(test13.subject_id)):
            raise Stop("gate0_overlap", "pool conformal/test subject overlap")

        # ---- hierarchy Gate 0 on every cluster of every partition, every split ----
        parts, dup_log = build_partitions()
        dup_count = sum(1 for d in dup_log if d["duplicate"])
        gate0_hier = {}
        for part in parts:
            for nm, labs in part["clusters"]:
                for sname, df_ in [("validation", splits.validation), ("conformal", splits.calibration),
                                   ("test", splits.evaluation)]:
                    nb, nc = hier_violations(df_, labs)
                    gate0_hier[f"{part['pid']}|{nm}|{sname}"] = dict(violations=nb, rows_checked=nc)
                    if nb:
                        raise Stop("gate0_hierarchy_violation", dict(pid=part["pid"], cluster=nm, split=sname, n=nb))
        full13_viol = {s: hier_violations(df_, LABELS) for s, df_ in
                       [("validation", splits.validation), ("conformal", splits.calibration), ("test", splits.evaluation)]}

        # ---- inner holdout (Exp055, seed 0) ----
        subjects = np.unique(val["subject_id"].to_numpy())
        rng = np.random.default_rng(HOLDOUT_SEED)
        perm = rng.permutation(subjects)
        n_hold = int(round(HOLDOUT_FRAC * subjects.size))
        hold = perm[:n_hold]
        is_hold = val["subject_id"].isin(set(hold.tolist())).to_numpy()
        val_tr = val[~is_hold].reset_index(drop=True)
        val_ho = val[is_hold].reset_index(drop=True)
        if len(set(val_tr.subject_id) & set(val_ho.subject_id)):
            raise Stop("gate0_overlap", "holdout subject overlap")
        ctx.update(pool_df=pool_df, val=val, val_tr=val_tr, val_ho=val_ho)

        # ---- splits: frozen draw + 100 resplits (Exp055 section 6) ----
        subj = pool_df["subject_id"].to_numpy()
        uniq = np.unique(subj)
        n_conf = len(conf13)
        splits_list = [(-1, np.arange(n_conf), np.arange(n_conf, len(pool_df)))]
        for r in range(N_RESPLITS):
            rng_r = np.random.default_rng(r)
            perm_r = rng_r.permutation(uniq)
            half = len(uniq) // 2
            inA = np.isin(subj, perm_r[:half])
            idxA, idxB = np.flatnonzero(inA), np.flatnonzero(~inA)
            assert len(idxA) + len(idxB) == len(pool_df)
            assert len(np.unique(subj[idxA])) == half
            splits_list.append((r, idxA, idxB))
        ctx["splits"] = splits_list
        ctx["pool13"] = cc.extract_group(pool_df, LABELS)
        ctx["pool13_y"] = ctx["pool13"].y
        ctx["y13cc"] = cc.state_indices(ctx["pool13"].y)
        ctx["states13"] = cc.enumerate_states(N13)
        ctx["adm13"] = cc.hierarchy_admissible(LABELS, ctx["states13"])
        ctx["pool_single"] = {lab: cc.extract_group(pool_df, [lab]) for lab in LABELS}

        # ---- identity gate (Exp055 section 7). Failure is a stop. ----
        id_df, id_summary = identity_gate(ctx)
        id_df.to_csv(os.path.join(EXP_DIR, "identity_gates.csv"), index=False)
        print("identity gate:", json.dumps(id_summary, indent=2), flush=True)
        if not id_summary["passed"]:
            raise Stop("identity_gate_failed", id_summary)

        # ---- reproduction check (SPEC section 2): P_H C1 = Exp055 cardio_pair, L1 seed 0, frozen draw ----
        c1_labels = P_H[0][1]
        e = get_cluster(c1_labels, 0, ctx)
        k = cc.conformal_rank(n_conf, ALPHA)
        s_true = e["raw"][np.arange(n_conf), e["ycc"][:n_conf]]
        thr_now = cc.order_statistic(s_true, k)
        r55 = pd.read_csv(os.path.join(EXP_DIR.replace("056_three_cluster_union", "055_learned_score_full"),
                                       "resplit_metrics.csv"),
                          usecols=["family", "unit", "resplit", "variant", "seed", "threshold"])
        r55 = r55[(r55.family == "designed") & (r55.unit == "cardio_pair") & (r55.resplit == -1) &
                  (r55.variant == "L1") & (r55.seed == 0)]
        d55 = pd.read_csv(os.path.join(EXP_DIR.replace("056_three_cluster_union", "055_learned_score_full"),
                                       "scorer_diagnostics.csv"))
        d55 = d55[(d55.family == "designed") & (d55.unit == "cardio_pair") & (d55.scorer == "L1") & (d55.seed == 0)]
        assert len(r55) == 1 and len(d55) == 1, (len(r55), len(d55))
        ref = dict(threshold_exp055=float(r55.threshold.iloc[0]), threshold_now=float(thr_now),
                   best_epoch_exp055=int(d55.best_epoch.iloc[0]), best_epoch_now=int(e["info"]["best_epoch"]),
                   stop_epoch_exp055=int(d55.stop_epoch.iloc[0]), stop_epoch_now=int(e["info"]["stop_epoch"]),
                   holdout_ce_exp055=float(d55.final_holdout_ce.iloc[0]), holdout_ce_now=float(e["info"]["final_holdout_ce"]))
        ref["match"] = bool(abs(ref["threshold_now"] - ref["threshold_exp055"]) < 1e-6
                            and ref["best_epoch_now"] == ref["best_epoch_exp055"]
                            and ref["stop_epoch_now"] == ref["stop_epoch_exp055"]
                            and abs(ref["holdout_ce_now"] - ref["holdout_ce_exp055"]) < 1e-6)
        print("reproduction check:", json.dumps(ref, indent=2), flush=True)
        ref["training_loop"] = "experiments/055_learned_score_full/run.py::train_scorer (Amendment 1, A1.1)"
        ref["diagnostic_mlp_score_port"] = json.load(open(os.path.join(EXP_DIR, "reproduction_check_mlp_score_diagnostic.json")))
        with open(os.path.join(EXP_DIR, "reproduction_check.json"), "w") as f:
            json.dump(ref, f, indent=2)
        if not ref["match"]:
            raise Stop("reproduction_mismatch", ref,
                       ["Stopped at the reproduction check (SPEC section 9). No experiment metric was computed.",
                        "Training used conformal_methods/mlp_score.py (Lightning) as the spec requires."])

        # ---- main experiment ----
        vec_rows, clu_rows = [], []
        for part in parts:
            t1 = time.time()
            for seed in (SEEDS if part["family"] == "designed" else (RANDOM_SEED,)):
                eval_cluster_partition(part, seed, ctx, vec_rows, clu_rows)
            print(f"[{part['family']:>8}] {part['pid']} clusters={[c for c, _ in part['clusters']]} "
                  f"{time.time() - t1:.1f}s", flush=True)
        t1 = time.time()
        refs_for_splits(ctx, vec_rows)
        print(f"[ refs13 ] {time.time() - t1:.1f}s", flush=True)
        t1 = time.time()
        eval_full13(ctx, vec_rows)
        print(f"[ arm D  ] {time.time() - t1:.1f}s", flush=True)

        vec = pd.DataFrame(vec_rows).rename(columns={"pid": "pid"})
        clu = pd.DataFrame(clu_rows)
        vec.to_csv(os.path.join(EXP_DIR, "resplit_metrics_vector.csv"), index=False)
        clu.to_csv(os.path.join(EXP_DIR, "resplit_metrics_cluster.csv"), index=False)
        pd.DataFrame(dup_log).to_csv(os.path.join(EXP_DIR, "random_partition_draws.csv"), index=False)

        res = run_analysis(vec, clu, ctx, dup_count, id_summary)
        res["vsum"].to_csv(os.path.join(EXP_DIR, "summary_vector.csv"), index=False)
        res["csum"].to_csv(os.path.join(EXP_DIR, "summary_cluster.csv"), index=False)
        pd.DataFrame(res["gates"]).to_csv(os.path.join(EXP_DIR, "gates.csv"), index=False)
        compute_guard_gaps(vec, clu).to_csv(os.path.join(EXP_DIR, "guard_gaps.csv"), index=False)

        # ---- pre-registered decision quantities ----
        dec = {}
        # rule 3 and rule 4 (seed 0, unguarded)
        dec["C_vs_maxrank_adm_log2m"] = paired(
            vec, dict(pid="P_H", arm="C", guard="unguarded", scorer_seed=0),
            dict(pid="none", arm="maxrank_adm", guard="none", scorer_seed=-1), "log2_over_m_mean")
        dec["C_vs_maxrank_adm_empty"] = paired(
            vec, dict(pid="P_H", arm="C", guard="unguarded", scorer_seed=0),
            dict(pid="none", arm="maxrank_adm", guard="none", scorer_seed=-1), "empty_set_rate")
        dec["C_vs_maxrank_adm_size"] = paired(
            vec, dict(pid="P_H", arm="C", guard="unguarded", scorer_seed=0),
            dict(pid="none", arm="maxrank_adm", guard="none", scorer_seed=-1), "mean_joint_set_size")
        dec["C_vs_D_log2m"] = paired(
            vec, dict(pid="P_H", arm="C", guard="unguarded", scorer_seed=0),
            dict(pid="none", arm="D", guard="unguarded", scorer_seed=RANDOM_SEED), "log2_over_m_mean")
        dec["C_vs_D_empty"] = paired(
            vec, dict(pid="P_H", arm="C", guard="unguarded", scorer_seed=0),
            dict(pid="none", arm="D", guard="unguarded", scorer_seed=RANDOM_SEED), "empty_set_rate")
        dec["C_vs_D_size"] = paired(
            vec, dict(pid="P_H", arm="C", guard="unguarded", scorer_seed=0),
            dict(pid="none", arm="D", guard="unguarded", scorer_seed=RANDOM_SEED), "mean_joint_set_size")
        # seed spread (rule 6): C_s - maxrank_adm, per seed
        spread = {}
        for s in SEEDS:
            spread[s] = paired(vec, dict(pid="P_H", arm="C", guard="unguarded", scorer_seed=s),
                               dict(pid="none", arm="maxrank_adm", guard="none", scorer_seed=-1), "log2_over_m_mean")
        dec["seed_spread_C_minus_maxrank_adm"] = {str(s): v["mean_diff"] for s, v in spread.items()}
        # designed vs random (rule 5), arm C seed 0 unguarded, resplit means of log2/13
        cu = res["vsum"]
        def rm(pid, arm, guard, seed, col="log2_over_m_mean_mean"):
            x = cu[(cu.pid == pid) & (cu.arm == arm) & (cu.guard == guard) & (cu.scorer_seed == seed)]
            return float(x[col].iloc[0]) if len(x) else np.nan
        ph_val = rm("P_H", "C", "unguarded", 0)
        rand_vals = [rm(p["pid"], "C", "unguarded", RANDOM_SEED) for p in parts if p["family"] == "random"]
        rand_cov_A = [rm(p["pid"], "A", "unguarded", RANDOM_SEED, "joint_coverage_mean") for p in parts if p["family"] == "random"]
        dec["designed_vs_random"] = dict(
            P_H_log2m_C_seed0=ph_val, random_log2m_C_seed0=rand_vals,
            n_random_with_larger_log2m=int(sum(1 for v in rand_vals if v > ph_val)),
            random_median=float(np.median(rand_vals)),
            arm_A_cov_random=rand_cov_A)
        res_out = dict(
            scope=SCOPE, alpha=ALPHA,
            gate0=dict(overlap_counts=overlap, full_rows=full_rows,
                       all13_clean_rows=dict(validation=len(val), conformal=len(conf13), test=len(test13)),
                       all13_clean_subjects=dict(validation=int(val.subject_id.nunique()), conformal=int(conf13.subject_id.nunique()),
                                                 test=int(test13.subject_id.nunique())),
                       pool_rows=len(pool_df), pool_subjects=int(pool_df.subject_id.nunique()),
                       hierarchy_within_cluster_violations_all_zero=all(v["violations"] == 0 for v in gate0_hier.values()),
                       full13_hierarchy_violations_true_labels=full13_viol),
            identity_gate=id_summary,
            reproduction=ref,
            dup_redraws=dup_count, draws_log=dup_log,
            decisions=dec,
            runtime_seconds=time.time() - t0,
        )
        with open(os.path.join(EXP_DIR, "metrics.json"), "w") as f:
            json.dump(res_out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

        inputs = {os.path.basename(p): sha256(p) for p in [
            os.path.join(cc.OUTPUTS_DIR, STEM + s) for s in ("_validation.csv", "_conformal.csv", "_test.csv")]}
        code = {os.path.relpath(p, REPO): sha256(p) for p in [
            os.path.join(EXP055_DIR, "run.py"),
            os.path.join(REPO, "conformal_common.py"), os.path.join(REPO, "conformal_methods", "bonferrony.py"),
            os.path.join(REPO, "conformal_methods", "max_rank.py"), os.path.join(REPO, "conformal_methods", "mlp_score.py"),
            os.path.join(REPO, "data", "hierarchy.py"), os.path.abspath(__file__)]}
        info = dict(git_head=git("rev-parse", "HEAD"), input_sha256=inputs, code_sha256=code,
                    runtime_seconds=time.time() - t0, package_versions=package_versions(),
                    seeds=dict(model_seeds=list(SEEDS), random_partition_seed_base=RANDOM_SEED_BASE,
                               holdout_rng=HOLDOUT_SEED, resplit_rng="0..99"),
                    command="/usr/bin/python3 experiments/056_three_cluster_union/run.py")
        with open(os.path.join(EXP_DIR, "run_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print("done in", round(time.time() - t0, 1), "s", flush=True)
        return 0

    except Stop as s:
        path = write_escalation([dict(trigger=s.trigger, detail=s.detail)],
                                s.lines or ["Stopped by a hard stop condition (SPEC section 9)."])
        print(f"STOP: {s.trigger}; wrote {path}", flush=True)
        print(json.dumps(s.detail, indent=2, default=str), flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
