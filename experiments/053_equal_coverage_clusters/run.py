"""Exp053 - equal-coverage cluster comparison (SPEC.md + Amendments + Amendments 2).

Pre-registered in experiments/053_equal_coverage_clusters/README.md before any metric was computed.
Amendments 3 (dated, README): max-rank and LP-SCP cluster p-values use the plain calibration count over n+1
for test/candidate/A2 rows (no extra +1); calibration rows self-inclusive. Bonferroni/Sidak unchanged.

Run:
    cd /Users/alonfines/Desktop/x-ray && .venv_exp048/bin/python experiments/053_equal_coverage_clusters/run.py
Options:
    --resplits N          run only the first N resplits (timing); default 100
    --out DIR             output directory (default: this folder)
    --lp-null-frozen-only fallback (SPEC §5): LP-SCP null partitions on the frozen draw only

Reused unedited: conformal_common.py, max_rank.py, bonferrony.py, lp_scp.py, data/hierarchy.py.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import conformal_common as cc   # noqa: E402
import bonferrony               # noqa: E402
import max_rank                 # noqa: E402
import lp_scp                   # noqa: E402

ALPHA = 0.10
M = 13
LABELS = list(cc.ALL_LABELS)
IDX = {lab: i for i, lab in enumerate(LABELS)}
CAP = 7
N_RESPLITS = 100
PO, FR, LL = IDX["Pleural Other"], IDX["Fracture"], IDX["Lung Lesion"]
EDGES = [(IDX[ch], IDX[p]) for ch, ps in cc.CHEXPERT_HIERARCHY.items() for p in ps]
assert len(EDGES) == 7, EDGES
assert len(LABELS) == 13

METHODS_P = ["bonf", "sidak", "maxrank"]
METHODS_ALL = METHODS_P + ["lpscp"]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Gate 0 / data
# ---------------------------------------------------------------------------

SPLITS = cc.load_splits(with_validation=True)  # runs gate0_check (raises on overlap)
GATE = {}
GATE["gate0_overlap_pairs"] = cc.gate0_check(SPLITS, raise_on_overlap=True)
assert all(v == 0 for v in GATE["gate0_overlap_pairs"].values()), GATE["gate0_overlap_pairs"]
assert len(SPLITS.calibration) == 18754 and len(SPLITS.evaluation) == 24394 and len(SPLITS.validation) == 18570
assert SPLITS.calibration.subject_id.nunique() == 5926
assert SPLITS.evaluation.subject_id.nunique() == 6395
assert SPLITS.validation.subject_id.nunique() == 5898
GATE["row_counts"] = {"conformal": len(SPLITS.calibration), "test": len(SPLITS.evaluation), "validation": len(SPLITS.validation)}

CONF13 = cc.extract_group(SPLITS.calibration, LABELS)
TEST13 = cc.extract_group(SPLITS.evaluation, LABELS)
VAL13 = cc.extract_group(SPLITS.validation, LABELS)
assert CONF13.n == 18754, CONF13.n
assert TEST13.n == 18877, TEST13.n
assert VAL13.n == 18570, VAL13.n
assert TEST13.subject_id is not None and len(np.unique(TEST13.subject_id)) == 5932, len(np.unique(TEST13.subject_id))
GATE["clean_counts"] = {"conformal13": CONF13.n, "test13": TEST13.n, "test13_subjects": int(len(np.unique(TEST13.subject_id))),
                        "validation13": VAL13.n}

# Gate 0.5 -----------------------------------------------------------------
pos_conf = CONF13.y.sum(axis=0)
GATE["gate05_positives_conformal"] = {LABELS[j]: int(pos_conf[j]) for j in range(M)}
GATE["gate05_flagged"] = [LABELS[j] for j in range(M) if pos_conf[j] < cc.MIN_POSITIVE_CALIBRATION_ROWS]
assert GATE["gate05_flagged"] == [], GATE["gate05_flagged"]

# Pool: conformal 13-clean then test 13-clean -------------------------------
PR = np.concatenate([CONF13.probs, TEST13.probs])
YR = np.concatenate([CONF13.y, TEST13.y])
SR = np.concatenate([CONF13.scores, TEST13.scores])
SUBJ = np.concatenate([CONF13.subject_id, TEST13.subject_id])
N_CONF = CONF13.n
N_POOL = PR.shape[0]
assert N_POOL == 37631
assert len(set(CONF13.subject_id) & set(TEST13.subject_id)) == 0
S_POOL_SUBJECTS = len(np.unique(SUBJ))
GATE["pooled"] = {"rows": N_POOL, "subjects": S_POOL_SUBJECTS}
assert S_POOL_SUBJECTS == 11858, S_POOL_SUBJECTS
STRAT_TRUE = YR[:, [PO, FR, LL]].any(axis=1)     # used on evaluation rows only
VAL_PROBS = VAL13.probs
VAL_SCORES = VAL13.scores
VAL_Y = VAL13.y

# Validation-side structure -------------------------------------------------
STRUCT_CACHE: dict = {}


def get_struct(cl):
    if cl not in STRUCT_CACHE:
        cols = [LABELS[j] for j in cl]
        g = cc.extract_group(SPLITS.validation, cols)
        STRUCT_CACHE[cl] = lp_scp.fit_structure(g)
    return STRUCT_CACHE[cl]


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

def canon(clusters):
    cl = [tuple(sorted(int(x) for x in c)) for c in clusters]
    return tuple(sorted(cl, key=lambda c: c[0]))


def edges_within(clusters):
    cid = {}
    for k, c in enumerate(clusters):
        for j in c:
            cid[j] = k
    return sum(1 for ch, p in EDGES if cid[ch] == cid[p])


def W_of(clusters, rho):
    tot = 0.0
    for c in clusters:
        for a, b in itertools.combinations(sorted(c), 2):
            tot += rho[a, b]
    return float(tot)


def random_partition(rng, sizes):
    perm = rng.permutation(M)
    out, pos = [], 0
    for s in sizes:
        out.append(tuple(sorted(int(x) for x in perm[pos:pos + s])))
        pos += s
    return canon(out)


CLINICAL_C = canon([
    (IDX["Enlarged Cardiomediastinum"], IDX["Cardiomegaly"]),
    tuple(IDX[x] for x in ["Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis"]),
    (IDX["Pneumothorax"],), (IDX["Pleural Effusion"],), (IDX["Pleural Other"],), (IDX["Fracture"],), (IDX["Support Devices"],),
])
CLINICAL_D = canon([
    (IDX["Enlarged Cardiomediastinum"], IDX["Cardiomegaly"]),
    tuple(IDX[x] for x in ["Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis"]),
    (IDX["Pleural Effusion"], IDX["Pleural Other"]),
    (IDX["Pneumothorax"],), (IDX["Fracture"],), (IDX["Support Devices"],),
])
assert sum(len(c) for c in CLINICAL_C) == 13 and len(CLINICAL_C) == 7
assert sum(len(c) for c in CLINICAL_D) == 13 and len(CLINICAL_D) == 6

# Informed partition: S-avg on validation -----------------------------------
RHO = spearmanr(VAL_SCORES).correlation
RHO = np.asarray(RHO, dtype=float)
assert RHO.shape == (M, M)
DIST = 1.0 - RHO


def informed_merge_sequence():
    clusters = [[i] for i in range(M)]
    snaps = {M: canon(clusters)}
    ties = []
    while len(clusters) > 3:
        best = None
        cands = []
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if len(clusters[a]) + len(clusters[b]) > CAP:
                    continue
                d = float(np.mean([DIST[i, j] for i in clusters[a] for j in clusters[b]]))
                ka, kb = min(clusters[a]), min(clusters[b])
                key = (d, min(ka, kb), max(ka, kb))
                cands.append((d, key, a, b))
        if not cands:
            raise RuntimeError(f"informed procedure stalled at K={len(clusters)} (cap {CAP}); ESCALATE")
        cands.sort(key=lambda x: x[1])
        best_d = cands[0][1][0]
        n_tie = sum(1 for c in cands if c[1][0] == best_d)
        if n_tie > 1:
            ties.append({"K": len(clusters), "n_tied_pairs": n_tie, "distance": best_d})
        _, _, a, b = cands[0]
        merged = clusters[a] + clusters[b]
        clusters = [c for k, c in enumerate(clusters) if k not in (a, b)] + [merged]
        if len(clusters) in (6, 4, 3):
            snaps[len(clusters)] = canon(clusters)
    return snaps, ties


INF_SNAPS, INF_TIES = informed_merge_sequence()
assert set(INF_SNAPS) == {3, 4, 6, 13}, INF_SNAPS.keys()
for K in (3, 4, 6):
    assert max(len(c) for c in INF_SNAPS[K]) <= CAP

# AUC on validation (difficulty control) ------------------------------------
AUC = np.zeros(M)
for j in range(M):
    y = VAL_Y[:, j]
    r = rankdata(VAL_PROBS[:, j])
    n1 = int(y.sum())
    n0 = len(y) - n1
    AUC[j] = (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
AUC_ORDER = sorted(range(M), key=lambda j: (AUC[j], j))   # ascending: hardest first

PARTS = []


def add_part(pid, family, clusters, lp, extra=None):
    cl = canon(clusters)
    assert sorted(j for c in cl for j in c) == list(range(M)), pid
    rec = {"pid": pid, "family": family, "K": len(cl), "clusters": cl,
           "sizes": sorted([len(c) for c in cl], reverse=True),
           "W": W_of(cl, RHO), "edges_within": edges_within(cl), "lp": lp}
    if extra:
        rec.update(extra)
    PARTS.append(rec)
    return rec


add_part("K13_singletons", "ref", [(j,) for j in range(M)], True)
add_part("K1_all", "ref", [tuple(range(M))], True)
add_part("clinC_K7", "clinical", CLINICAL_C, True)
add_part("clinD_K6", "clinical", CLINICAL_D, True)

PROFILES = {K: sorted([len(c) for c in INF_SNAPS[K]], reverse=True) for K in (3, 4, 6)}
for K in (3, 4, 6):
    add_part(f"informed_K{K}", "informed", INF_SNAPS[K], True)

# Difficulty control (E2) ----------------------------------------------------
for K in (3, 4, 6):
    sizes = PROFILES[K]
    clusters, pos = [], 0
    for s in sizes:                                  # largest chunk takes the lowest-AUC labels
        clusters.append(tuple(sorted(AUC_ORDER[pos:pos + s])))
        pos += s
    add_part(f"difficulty_K{K}", "difficulty", clusters, True)

# Anti-clustering (E5) -------------------------------------------------------
ANTI_LOG = {}
for K in (3, 4, 6):
    sizes = PROFILES[K]
    rng = np.random.default_rng(2000)
    best = None
    restart_W = []
    for restart in range(50):
        perm = rng.permutation(M)
        assign = np.zeros(M, dtype=int)
        pos = 0
        for k, s in enumerate(sizes):
            assign[perm[pos:pos + s]] = k
            pos += s

        def Wa(a):
            tot = 0.0
            for i, j in itertools.combinations(range(M), 2):
                if a[i] == a[j]:
                    tot += RHO[i, j]
            return tot

        Wc = Wa(assign)
        n_moves = 0
        while True:
            best_delta, best_pair = 0.0, None
            for i, j in itertools.combinations(range(M), 2):
                if assign[i] == assign[j]:
                    continue
                a2 = assign.copy()
                a2[i], a2[j] = assign[j], assign[i]
                delta = Wa(a2) - Wc
                if delta < best_delta:
                    best_delta, best_pair = delta, (i, j)
            if best_pair is None or best_delta >= -1e-12:
                break
            i, j = best_pair
            assign[i], assign[j] = assign[j], assign[i]
            Wc = Wc + best_delta
            n_moves += 1
        restart_W.append(Wc)
        if best is None or Wc < best[0] - 0.0:
            best = (Wc, assign.copy(), restart)
    Wbest, abest, ridx = best
    clusters = [tuple(sorted(int(i) for i in np.flatnonzero(abest == k))) for k in range(K)]
    add_part(f"anti_K{K}", "anti", clusters, True, {"anti_restart": ridx})
    ANTI_LOG[K] = {"best_restart": ridx, "best_W": float(Wbest),
                   "restart_W_min": float(min(restart_W)), "restart_W_max": float(max(restart_W))}

# Random null (E6): K=3,4,6, 200 each, own RNG seed 1000+idx ---------------------
for K in (3, 4, 6):
    for idx in range(200):
        rng = np.random.default_rng(1000 + idx)
        cl = random_partition(rng, PROFILES[K])
        add_part(f"rand_K{K}_{idx:03d}", "random", cl, idx < 100, {"idx": idx})

# Clinical random (E3) ------------------------------------------------------
for idx in range(100):
    rng = np.random.default_rng(3000 + idx)
    add_part(f"randC_{idx:03d}", "random_clinC", random_partition(rng, sorted([len(c) for c in CLINICAL_C], reverse=True)), False, {"idx": idx})
for idx in range(100):
    rng = np.random.default_rng(4000 + idx)
    add_part(f"randD_{idx:03d}", "random_clinD", random_partition(rng, sorted([len(c) for c in CLINICAL_D], reverse=True)), False, {"idx": idx})

# Matched null (E4): pool of 5000 per profile, stream seeded 1500 (fresh per profile) ---
MATCH_LOG = {}
for K in (3, 4, 6):
    E0 = edges_within(INF_SNAPS[K])
    rng = np.random.default_rng(1500)
    kept = []
    pool_matches = 0
    for _ in range(5000):
        cl = random_partition(rng, PROFILES[K])
        if edges_within(cl) == E0:
            pool_matches += 1
            if len(kept) < 200:
                kept.append(cl)
    MATCH_LOG[K] = {"informed_edges": E0, "pool_matches_of_5000": pool_matches, "kept": len(kept)}
    if len(kept) < 60:
        raise RuntimeError(f"matched null K={K}: only {len(kept)} matches (<60); ESCALATE")
    for i, cl in enumerate(kept):
        add_part(f"matched_K{K}_{i:03d}", "matched", cl, i < 100, {"idx": i})

# Record the partition set (fixed before any metric)
PART_IDS = [p["pid"] for p in PARTS]
assert len(PART_IDS) == len(set(PART_IDS))
log(f"partitions: {len(PARTS)}  informed profiles: {PROFILES}")


# ---------------------------------------------------------------------------
# Core calibration machinery
# ---------------------------------------------------------------------------

def ss_left(a, v):
    return np.searchsorted(a, v, side="left")


def ss_right(a, v):
    return np.searchsorted(a, v, side="right")


def chunk_rows_for(m):
    return max(16, int(2 ** 22 // ((2 ** m) * m)))


def own_scores(rows, cl):
    """Score of each row's own true joint state on cluster cl (lp_scp.state_scores, same formula)."""
    struct = get_struct(cl)
    m = len(cl)
    cols = list(cl)
    out = np.empty(len(rows))
    yst = cc.state_indices(YR[rows][:, cols])
    step = chunk_rows_for(m)
    for s in range(0, len(rows), step):
        rr = rows[s:s + step]
        sc = lp_scp.state_scores(PR[rr][:, cols], struct, lp_scp.MU_PRIMARY, lp_scp.NU_PRIMARY)
        out[s:s + step] = sc[np.arange(len(rr)), yst[s:s + step]]
    return out


def beta_hat(t, n_beta):
    """SPEC §2: largest t-value u with #{t_i <= u} <= floor(alpha(n+1)) - 1 (integer arithmetic)."""
    c = (n_beta + 1) // 10
    ts = np.sort(t)
    u = np.unique(ts)
    counts = np.searchsorted(ts, u, side="right")
    ok = counts <= c - 1
    if not ok.any():
        raise RuntimeError("beta-hat undefined: no t-value satisfies the count bound; ESCALATE")
    k = np.flatnonzero(ok)[-1]
    return float(u[k]), int(counts[k])


class Design:
    """One calibration/evaluation design.

    ref   : rows defining reference counts (LOO when beta rows are inside ref)
    beta  : rows used for t_i and the beta-hat quantile
    add_beta : 0 if beta rows are in ref (LOO, self counted), else 1
    ev    : evaluation rows (test side; always add = 1)
    """

    def __init__(self, name, ref_idx, beta_idx, beta_in_ref, ev_idx):
        self.name = name
        self.ref = np.asarray(ref_idx)
        self.beta = np.asarray(beta_idx)
        self.beta_is_ref = bool(beta_in_ref)
        self.add_beta = 0 if beta_in_ref else 1
        self.ev = np.asarray(ev_idx)
        self.nref = len(self.ref)
        self.nbeta = len(self.beta)
        self.nev = len(self.ev)
        S_ref = SR[self.ref]
        self.sorted_s = [np.sort(S_ref[:, j]) for j in range(M)]
        self.d_ref = np.empty((self.nref, M), dtype=np.int64)          # includes self
        self.d_beta = np.empty((self.nbeta, M), dtype=np.int64)
        for j in range(M):
            self.d_ref[:, j] = self.nref - ss_left(self.sorted_s[j], S_ref[:, j])
            self.d_beta[:, j] = self.add_beta + self.nref - ss_left(self.sorted_s[j], SR[self.beta, j])
        self.d_eval = np.empty((self.nev, M, 2), dtype=np.int64)     # [..., x] for candidate x
        P_e = PR[self.ev]
        for j in range(M):
            s1 = cc.nonconformity(P_e[:, j], np.ones(self.nev, dtype=int))
            s0 = cc.nonconformity(P_e[:, j], np.zeros(self.nev, dtype=int))
            self.d_eval[:, j, 1] = 1 + self.nref - ss_left(self.sorted_s[j], s1)
            self.d_eval[:, j, 0] = 1 + self.nref - ss_left(self.sorted_s[j], s0)
        self.cache: dict = {}

    def R_beta(self, cl):
        return self.d_beta[:, list(cl)].min(axis=1)

    def grid(self, method, cl, which):
        key = ("grid", method, cl, which)
        if key in self.cache:
            return self.cache[key]
        n = self.nref
        R = np.arange(n + 2, dtype=np.float64)
        m = len(cl)
        if method == "bonf":
            f = np.minimum(1.0, m * R / (n + 1))
        elif method == "sidak":
            f = 1.0 - (1.0 - R / (n + 1)) ** m
        elif method == "maxrank":
            sr = np.sort(self.d_ref[:, list(cl)].min(axis=1))
            # Amendments 3: cluster-level count, no extra +1 on test/A2 side; self-inclusive for ref rows
            f = ss_right(sr, R.astype(np.int64)) / (n + 1)
        else:
            raise ValueError(method)
        if not np.all(np.diff(f) >= 0):
            raise RuntimeError(f"non-monotone P grid for {method}")
        self.cache[key] = f
        return f

    def own(self, cl):
        key = ("own", cl)
        if key not in self.cache:
            s_ref = own_scores(self.ref, cl)
            s_beta = s_ref if self.beta_is_ref else own_scores(self.beta, cl)
            self.cache[key] = (np.sort(s_ref), s_beta)
        return self.cache[key]


def run_cell(method, part, D, collect_rows=True):
    """Calibrate (beta-hat on D.beta) and evaluate on D.ev for one (method, partition)."""
    cls = part["clusters"]
    K = len(cls)
    nev = D.nev
    sizes = np.empty((nev, K), dtype=np.int64)
    covk = np.empty((nev, K), dtype=bool)
    has0 = np.zeros((nev, M), dtype=bool)
    has1 = np.zeros((nev, M), dtype=bool)
    yev = YR[D.ev]
    if method in METHODS_P:
        t_cols = [D.grid(method, cl, "beta")[D.R_beta(cl)] for cl in cls]
        t = np.min(np.vstack(t_cols), axis=0)
        beta, _ = beta_hat(t, D.nbeta)
        cal_cov = float(np.mean(t > beta))
        for k, cl in enumerate(cls):
            ge = D.grid(method, cl, "eval")
            i0 = int(np.searchsorted(ge, beta, side="right"))   # first R with P(R) > beta
            rstar = i0 if i0 < len(ge) else 10 ** 9
            size = np.ones(nev, dtype=np.int64)
            cov = np.ones(nev, dtype=bool)
            for j in cl:
                h0 = D.d_eval[:, j, 0] >= rstar
                h1 = D.d_eval[:, j, 1] >= rstar
                has0[:, j] = h0
                has1[:, j] = h1
                size *= (h0.astype(np.int64) + h1)
                cov &= np.where(yev[:, j] == 1, h1, h0)
            sizes[:, k] = size
            covk[:, k] = cov
    elif method == "lpscp":
        t_cols = []
        for cl in cls:
            srt, s_beta = D.own(cl)
            Pb = (D.nref - ss_left(srt, s_beta)) / (D.nref + 1)     # Amendments 3: no extra +1
            t_cols.append(Pb)
        t = np.min(np.vstack(t_cols), axis=0)
        beta, _ = beta_hat(t, D.nbeta)
        cal_cov = float(np.mean(t > beta))
        step = chunk_rows_for(max(len(c) for c in cls))
        for s in range(0, nev, step):
            e = min(nev, s + step)
            rr = D.ev[s:e]
            c = e - s
            for k, cl in enumerate(cls):
                cols = list(cl)
                struct = get_struct(cl)
                srt, _ = D.own(cl)
                sc = lp_scp.state_scores(PR[rr][:, cols], struct, lp_scp.MU_PRIMARY, lp_scp.NU_PRIMARY)
                Pe = (D.nref - ss_left(srt, sc)) / (D.nref + 1)         # Amendments 3: no extra +1
                Mk = Pe > beta
                sizes[s:e, k] = Mk.sum(axis=1)
                states = cc.enumerate_states(len(cl))
                ystate = cc.state_indices(YR[rr][:, cols])
                covk[s:e, k] = Mk[np.arange(c), ystate]
                for p, j in enumerate(cl):
                    has1[s:e, j] = Mk[:, states[:, p] == 1].any(axis=1)
                    has0[s:e, j] = Mk[:, states[:, p] == 0].any(axis=1)
    else:
        raise ValueError(method)
    return {"beta": beta, "cal_cov": cal_cov, "sizes": sizes, "covk": covk, "has0": has0, "has1": has1}


def summarize(res, D):
    sizes, covk, has0, has1 = res["sizes"], res["covk"], res["has0"], res["has1"]
    js = sizes.prod(axis=1)
    cov = covk.all(axis=1)
    decisive = has0 ^ has1
    fd = decisive.all(axis=1)
    pred = has1
    nonempty = js > 0
    out = {
        "beta_hat": res["beta"],
        "cal_cov": res["cal_cov"],
        "joint_cov": float(cov.mean()),
        "cluster_cov_min": float(covk.mean(axis=0).min()),
        "cluster_cov_mean": float(covk.mean()),
        "mean_log2_13": float(np.log2(js[nonempty]).mean() / M) if nonempty.any() else float("nan"),
        "mean_size": float(js.mean()),
        "median_size": float(np.median(js)),
        "fully_dec": float(fd.mean()),
        "empty_rate": float((js == 0).mean()),
        "per_label_dec": float(decisive.mean()),
        "n_eval": int(D.nev),
    }
    if fd.any():
        ok = cc.hierarchy_admissible(LABELS, pred[fd].astype(int))
        out["hv_rate_fd"] = float(1.0 - ok.mean())
        out["dec_acc"] = float((pred[fd].astype(int) == YR[D.ev][fd]).all(axis=1).mean())
    else:
        out["hv_rate_fd"] = float("nan")
        out["dec_acc"] = float("nan")
    strat = STRAT_TRUE[D.ev]
    out["cov_flag"] = float(cov[strat].mean()) if strat.any() else float("nan")
    out["cov_unflag"] = float(cov[~strat].mean()) if (~strat).any() else float("nan")
    out["n_flag"] = int(strat.sum())
    return out


def joint_sets_equal(a, b):
    ja = a["sizes"].prod(axis=1)
    jb = b["sizes"].prod(axis=1)
    same = (a["has0"] == b["has0"]).all(axis=1) & (a["has1"] == b["has1"]).all(axis=1)
    both_empty = (ja == 0) & (jb == 0)
    return ~(same | both_empty)


# ---------------------------------------------------------------------------
# Verification (code-path checks; not science gates)
# ---------------------------------------------------------------------------

def verify_code(D):
    """Naive per-row loops vs vectorised path on a few rows of the frozen design."""
    report = {}
    rng = np.random.default_rng(7)
    part = next(p for p in PARTS if p["pid"] == "informed_K4")
    cl = part["clusters"][0]          # first informed cluster (size <= 7)
    cols = list(cl)
    m = len(cl)
    beta_rows = rng.choice(D.nbeta, size=min(12, D.nbeta), replace=False)
    ev_rows = rng.choice(D.nev, size=min(12, D.nev), replace=False)
    n = D.nref
    # naive d for beta rows and eval candidates, from the reference rows (explicit loops)
    ref_rows = D.ref
    max_err = {"d_beta": 0, "d_eval": 0, "bonf_P": 0.0, "maxrank_P": 0.0, "lp_S": 0.0, "lp_P": 0.0}
    # ranks and P for each beta row (naive)
    for bi in beta_rows:
        row = D.beta[bi]
        for j in cols:
            s_row = SR[row, j]
            cnt = sum(1 for l in ref_rows if SR[l, j] >= s_row)
            add = D.add_beta
            max_err["d_beta"] = max(max_err["d_beta"], abs((add + cnt) - int(D.d_beta[bi, j])))
    # eval candidates naive
    for ei in ev_rows:
        row = D.ev[ei]
        for j in cols:
            for x in (0, 1):
                s = cc.nonconformity(np.array([PR[row, j]]), np.array([x]))[0]
                cnt = sum(1 for l in ref_rows if SR[l, j] >= s)
                max_err["d_eval"] = max(max_err["d_eval"], abs((1 + cnt) - int(D.d_eval[ei, j, x])))
    # bonf / maxrank P naive for a few beta rows against vectorised grid
    Rb_vec = D.R_beta(cl)
    g_b_bonf = D.grid("bonf", cl, "beta")
    g_b_mr = D.grid("maxrank", cl, "beta")
    for bi in beta_rows:
        R_naive_d = [int(D.d_beta[bi, j]) for j in cols]
        R_naive = min(R_naive_d)
        P_bonf_naive = min(1.0, m * R_naive / (n + 1))
        max_err["bonf_P"] = max(max_err["bonf_P"], abs(P_bonf_naive - g_b_bonf[Rb_vec[bi]]))
        # max-rank: count over ref rows of R_l <= R_i; R_l from the d_ref table
        Rl = D.d_ref[:, cols].min(axis=1)
        cnt = int(np.sum(Rl <= R_naive))
        P_mr_naive = cnt / (n + 1)          # Amendments 3
        max_err["maxrank_P"] = max(max_err["maxrank_P"], abs(P_mr_naive - g_b_mr[Rb_vec[bi]]))
    # LP-SCP: own-state and candidate scores vs direct formula on the same rows
    struct = get_struct(cl)
    srt, s_beta = D.own(cl)
    for bi in beta_rows[:4]:
        row = D.beta[bi]
        direct = lp_scp.state_scores(PR[row:row + 1][:, cols], struct, 1.0, 1.0)[0]
        st = cc.state_indices(YR[row:row + 1][:, cols])[0]
        max_err["lp_S"] = max(max_err["lp_S"], abs(direct[st] - s_beta[bi]))
    # full-calibration own-state vs lp_scp.true_state_scores (m <= 6 only: memory)
    if m <= 6:
        g = cc.GroupData(labels=[LABELS[j] for j in cols], probs=PR[D.ref][:, cols], y=YR[D.ref][:, cols],
                         scores=SR[D.ref][:, cols], n_rows_before_filter=len(D.ref))
        ref_true = lp_scp.true_state_scores(g, struct, 1.0, 1.0)
        mine = own_scores(D.ref, cl)
        report["lp_true_state_full_maxabs_m_le_6"] = float(np.max(np.abs(ref_true - mine)))
        assert report["lp_true_state_full_maxabs_m_le_6"] <= 1e-10
    report["naive_vs_vector_maxabs"] = {k: float(v) for k, v in max_err.items()}
    assert max_err["d_beta"] == 0 and max_err["d_eval"] == 0
    assert max_err["bonf_P"] <= 1e-12 and max_err["maxrank_P"] <= 1e-12
    assert max_err["lp_S"] <= 1e-10
    return report


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------

def frozen_design():
    return Design("frozen", np.arange(N_CONF), np.arange(N_CONF), True, np.arange(N_CONF, N_POOL))


def resplit_designs(r):
    """SPEC §5 / E8 / Q1. Returns (headline design, split-arm design, subject info)."""
    rng = np.random.default_rng(r)
    subs = np.unique(SUBJ)
    perm = rng.permutation(subs)
    half = len(subs) // 2
    in_a = np.isin(SUBJ, perm[:half])
    A = np.flatnonzero(in_a)
    B = np.flatnonzero(~in_a)
    subA = np.unique(SUBJ[A])
    perm2 = rng.permutation(subA)           # Q1: second permutation from the same generator
    half2 = len(subA) // 2
    in_a1 = np.isin(SUBJ[A], perm2[:half2])
    A1 = A[in_a1]
    A2 = A[~in_a1]
    head = Design(f"resplit_{r:03d}", A, A, True, B)
    split = Design(f"split_{r:03d}", A1, A2, False, B)
    return head, split, {"nA": len(A), "nB": len(B), "nA1": len(A1), "nA2": len(A2)}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def cell_list():
    out = []
    for p in PARTS:
        methods = METHODS_P + (["lpscp"] if p["lp"] else [])
        for mth in methods:
            out.append((p, mth))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resplits", type=int, default=N_RESPLITS)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--lp-null-frozen-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    log(f"Gate 0 ok: {GATE['gate0_overlap_pairs']} rows={GATE['row_counts']} clean={GATE['clean_counts']}")
    log(f"Gate 0.5 positives (conformal 13-clean): {GATE['gate05_positives_conformal']} flagged={GATE['gate05_flagged']}")
    log(f"informed S-avg: K6={INF_SNAPS[6]} K4={INF_SNAPS[4]} K3={INF_SNAPS[3]} ties={INF_TIES}")

    records = []
    identity = []
    cells = cell_list()

    def run_design(D, design_tag, resplit_id, cell_filter=None):
        t_d = time.time()
        local = {}
        for p, mth in cells:
            if cell_filter is not None and not cell_filter(p, mth):
                continue
            if args.lp_null_frozen_only and mth == "lpscp" and p["family"] in ("random", "matched") and design_tag != "frozen":
                continue
            res = run_cell(mth, p, D)
            s = summarize(res, D)
            rec = {"design": design_tag, "resplit": resplit_id, "pid": p["pid"], "family": p["family"],
                   "K": p["K"], "method": mth, "W": p["W"], "edges_within": p["edges_within"], **s}
            records.append(rec)
            local[(p["pid"], mth)] = res
        # identity gates at K=13 and K=1 (SPEC §8)
        for pid in ("K13_singletons", "K1_all"):
            if (pid, "bonf") in local:
                diffs = {}
                for a, b in (("bonf", "sidak"), ("bonf", "maxrank")):
                    d = joint_sets_equal(local[(pid, a)], local[(pid, b)])
                    diffs[f"{a}_vs_{b}"] = float(d.mean())
                identity.append({"design": design_tag, "resplit": resplit_id, "pid": pid, **diffs,
                                 "cal_cov_bonf": local[(pid, "bonf")]["cal_cov"],
                                 "cal_cov_sidak": local[(pid, "sidak")]["cal_cov"],
                                 "cal_cov_maxrank": local[(pid, "maxrank")]["cal_cov"]})
        return time.time() - t_d

    # Frozen draw first (reference row; all cells)
    FD = frozen_design()
    t_v = time.time()
    verif = verify_code(FD)
    log(f"verification: {verif}  ({time.time() - t_v:.1f}s)")
    tf = run_design(FD, "frozen", -1)
    log(f"frozen draw done in {tf:.1f}s")

    # Context rows (Q4: frozen draw only). Existing reused functions, full conformal calibration.
    context = {}
    for name, fn in [
        ("uncal_bonferroni_K1", lambda: bonferrony.calibrate_per_label(CONF13, ALPHA, "bonferroni")[0]),
        ("uncal_sidak_K1", lambda: bonferrony.calibrate_per_label(CONF13, ALPHA, "sidak")[0]),
        ("uncal_maxrank_K1", lambda: max_rank.calibrate_max_rank(CONF13, ALPHA)[0]),
        ("independent_per_label_alpha", lambda: bonferrony.calibrate_per_label(CONF13, ALPHA, "none")[0]),
    ]:
        thr = fn()
        ev = cc.evaluate(TEST13, thr)
        context[name] = {k: ev[k] for k in ("joint_coverage", "mean_joint_set_size", "median_joint_set_size",
                                            "empty_set_rate", "fully_decisive_rate", "decisive_accuracy",
                                            "hierarchy_violation_rate_decisive") if k in ev}
    log(f"context rows: {context}")

    # Split-calibration arm (informed K=4, all four methods), per resplit
    split_pid = "informed_K4"
    split_records = []

    # Resplits
    per_resplit_time = []
    resplit_meta = []
    for r in range(args.resplits):
        tr = time.time()
        head, split, meta = resplit_designs(r)
        resplit_meta.append({"resplit": r, **meta})
        run_design(head, "resplit", r)
        # split arm (K=4 informed only)
        for mth in METHODS_ALL:
            p = next(pp for pp in PARTS if pp["pid"] == split_pid)
            res = run_cell(mth, p, split)
            s = summarize(res, split)
            split_records.append({"design": "split_arm", "resplit": r, "pid": split_pid, "family": "split_arm",
                                  "K": p["K"], "method": mth, "W": p["W"], "edges_within": p["edges_within"],
                                  "nA1": meta["nA1"], "nA2": meta["nA2"], **s})
        dt = time.time() - tr
        per_resplit_time.append(dt)
        log(f"resplit {r:03d}: {dt:.1f}s  (elapsed {time.time() - t0:.0f}s)  nA={meta['nA']} nB={meta['nB']}")
        if r % 5 == 4:
            pd.DataFrame(records + split_records).to_csv(os.path.join(args.out, "resplit_metrics_partial.csv"), index=False)

    records.extend(split_records)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.out, "resplit_metrics.csv"), index=False)
    pd.DataFrame(identity).to_csv(os.path.join(args.out, "identity_gates.csv"), index=False)

    meta_out = {
        "gate0": GATE,
        "verification": verif,
        "partitions": [{k: (v if k != "clusters" else [[LABELS[j] for j in c] for c in v]) for k, v in p.items()} for p in PARTS],
        "informed": {str(K): [[LABELS[j] for j in c] for c in INF_SNAPS[K]] for K in (3, 4, 6)},
        "informed_ties": INF_TIES,
        "informed_profiles": {str(K): PROFILES[K] for K in (3, 4, 6)},
        "auc_validation": {LABELS[j]: float(AUC[j]) for j in range(M)},
        "anti": ANTI_LOG,
        "matched": MATCH_LOG,
        "context_rows_frozen": context,
        "resplit_meta": resplit_meta,
        "per_resplit_seconds": per_resplit_time,
        "total_seconds_so_far": time.time() - t0,
        "lp_null_frozen_only": args.lp_null_frozen_only,
        "n_resplits": args.resplits,
        "hashes_file": ".prereg_hashes.txt",
    }
    with open(os.path.join(args.out, "run_info.json"), "w") as fh:
        json.dump(meta_out, fh, indent=1, default=float)
    log(f"done: {len(df)} rows; total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
