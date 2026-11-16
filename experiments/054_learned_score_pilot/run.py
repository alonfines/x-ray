#!/usr/bin/env python3
"""Exp054 - learned joint-state score, PILOT / sanity check (arm L1) vs Bonferroni.

Pre-registered in experiments/054_learned_score_pilot/README.md. Implements that spec
exactly; no tuning. Reuses conformal_common (loading, Gate 0, filtering, hierarchy,
evaluate_joint, evaluate, conformal_rank, order_statistic) and bonferrony
(calibrate_per_label). Nothing in the pipeline is modified.

Usage (from any directory):
    /usr/bin/python3 experiments/054_learned_score_pilot/run.py
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

STEM = "densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809"
CLUSTERS_YAML = os.path.join(REPO, "configs", "clusters", "exp046.yaml")

SEED = 0
ALPHA = 0.10
HOLDOUT_FRAC = 0.20
HIDDEN = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 256
PATIENCE = 10
MAX_EPOCHS = 200
PROB_CLIP = 1e-6
Z = 1.96  # two-sided 95% binomial band; the validity gate uses the lower edge only


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def all13_clean(df):
    """Rows with every one of the 13 true labels in {0,1} and every prob finite (spec: all-13-clean)."""
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


def logit(p):
    p = np.clip(p, PROB_CLIP, 1.0 - PROB_CLIP)
    return np.log(p / (1.0 - p))


def spec_weights(m):
    """Spec target index is sum_l y_l * 2^l, labels in yaml order (label 0 = LSB)."""
    return (1 << np.arange(m)).astype(np.int64)


def spec_states(m):
    """(2^m, m) state vectors in spec index order: row i has bit l = (i >> l) & 1."""
    return ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, :]) & 1).astype(int)


def package_versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "pyyaml": yaml.__version__,
    }


# ---------------------------------------------------------------------------
# learned joint-state score (arm L1)
# ---------------------------------------------------------------------------

def train_l1(X_tr, y_tr, X_ho, y_ho, S):
    """MLP m -> 64 -> 2^m, CE loss, Adam, early stopping on holdout CE. Seed 0."""
    m = X_tr.shape[1]
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(m, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, S))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(SEED)

    best, best_ep, since, best_state = float("inf"), 0, 0, None
    stop_ep, stopped_early = MAX_EPOCHS, False
    history = []
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(X_tr.shape[0], generator=gen)
        for i in range(0, X_tr.shape[0], BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = ce(model(X_tr[b]), y_tr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ho_loss = ce(model(X_ho), y_ho).item()
            tr_loss = ce(model(X_tr), y_tr).item()
        history.append({"epoch": ep, "train_ce": tr_loss, "holdout_ce": ho_loss})
        if ho_loss < best:
            best, best_ep, since = ho_loss, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if since >= PATIENCE:
            stop_ep, stopped_early = ep, True
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_tr = ce(model(X_tr), y_tr).item()
        final_ho = ce(model(X_ho), y_ho).item()
    info = {
        "best_epoch": best_ep,
        "stop_epoch": stop_ep,
        "stopped_early": stopped_early,
        "epochs_run": len(history),
        "final_train_ce": final_tr,
        "final_holdout_ce": final_ho,
    }
    return model, info, history


def neg_log_softmax(model, X):
    """s(x, y) = -log_softmax(logits)[y] for every joint state, (n, 2^m), spec order."""
    with torch.no_grad():
        lp = torch.log_softmax(model(X).double(), dim=1).numpy()
    return -lp


def region_l1(s_raw, admissible, threshold, guard):
    """Prediction region over spec-ordered states.

    Admissibility is applied explicitly (hard constraint, s = +inf) so an infinite
    threshold cannot re-admit inadmissible states (inf <= inf).
    """
    n = s_raw.shape[0]
    region = admissible[None, :] & (s_raw <= threshold)
    if guard:
        s_con = np.where(admissible[None, :], s_raw, np.inf)
        region[np.arange(n), s_con.argmin(axis=1)] = True
    return region


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def set_metrics(group, region_spec, spec_of_cc, m, labels_m):
    """Locked metrics via cc.evaluate_joint, plus log2(size)/m over non-empty rows."""
    region_cc = region_spec[:, spec_of_cc]                     # conformal_common state order
    met = cc.evaluate_joint(group, region_cc)
    sizes = region_spec.sum(axis=1)
    ok = sizes >= 1
    met["log2_size_over_m_mean"] = float(np.mean(np.log2(sizes[ok]) / m)) if ok.any() else float("nan")
    met["_sizes"] = sizes
    return met


def binomial_gate(cov, n, alpha=ALPHA):
    half = Z * np.sqrt(alpha * (1 - alpha) / n)
    lower, upper = (1 - alpha) - half, (1 - alpha) + half
    return {"band_lower": lower, "band_upper": upper, "gate_pass_lower": bool(cov >= lower),
            "in_two_sided_band": bool(lower <= cov <= upper)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    clusters, spec = cc.load_clusters(CLUSTERS_YAML)
    assert float(spec.get("alpha")) == ALPHA, "alpha in exp046.yaml must be 0.1"
    assert [c.name for c in clusters] == ["cardio_pair", "lung_core", "pleural_free", "lung_wide"]
    assert [c.m for c in clusters] == [2, 3, 3, 5]

    # ---- Gate 0 (full frozen splits, before any filtering) ----
    splits = cc.load_splits(stem=STEM, with_validation=True, with_train=False, run_gate0=True)
    overlap = cc.gate0_check(splits, raise_on_overlap=True)
    assert all(v == 0 for v in overlap.values()), overlap

    full_rows = {"validation": len(splits.validation), "conformal": len(splits.calibration),
                 "test": len(splits.evaluation)}
    full_subjects = {"validation": int(splits.validation.subject_id.nunique()),
                     "conformal": int(splits.calibration.subject_id.nunique()),
                     "test": int(splits.evaluation.subject_id.nunique())}

    frames = {"validation": splits.validation, "conformal": splits.calibration, "test": splits.evaluation}
    gate0_hier = {}
    for name, df in frames.items():
        for cl in clusters:
            n_bad, n_chk = hier_violations(df, cl.labels)
            gate0_hier[f"{name}|{cl.name}"] = {"violations": n_bad, "rows_checked": n_chk}
            assert n_bad == 0, f"Gate 0 hierarchy violation in {name} / {cl.name}: {n_bad}"

    # ---- population: all-13-clean rows only (spec) ----
    val = all13_clean(splits.validation)
    cal = all13_clean(splits.calibration)
    test = all13_clean(splits.evaluation)
    clean_rows = {"validation": len(val), "conformal": len(cal), "test": len(test)}
    clean_subjects = {"validation": int(val.subject_id.nunique()), "conformal": int(cal.subject_id.nunique()),
                      "test": int(test.subject_id.nunique())}
    n_cal, n_test = len(cal), len(test)

    # ---- inner holdout: 20% of validation subjects (seed 0), shared by all clusters ----
    subjects = np.unique(val["subject_id"].to_numpy())
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(subjects)
    n_hold_subj = int(round(HOLDOUT_FRAC * subjects.size))
    hold_subj = perm[:n_hold_subj]
    is_hold = val["subject_id"].isin(set(hold_subj.tolist())).to_numpy()
    val_tr, val_ho = val[~is_hold].reset_index(drop=True), val[is_hold].reset_index(drop=True)
    assert len(set(val_tr.subject_id) & set(val_ho.subject_id)) == 0
    assert len(val_tr) > 0 and len(val_ho) > 0

    records, metrics = [], {"clusters": {}}
    for cl in clusters:
        labels, m = cl.labels, cl.m
        S = 2 ** m
        w = spec_weights(m)
        st_spec = spec_states(m)
        admissible = cc.hierarchy_admissible(labels, st_spec)          # (2^m,) spec order
        spec_of_cc = (cc.enumerate_states(m) * w).sum(axis=1)           # cc state k -> spec index

        g_tr = cc.extract_group(val_tr, labels)
        g_ho = cc.extract_group(val_ho, labels)
        g_val = cc.extract_group(val, labels)
        g_cal = cc.extract_group(cal, labels)
        g_te = cc.extract_group(test, labels)
        assert g_tr.n == len(val_tr) and g_ho.n == len(val_ho)
        assert g_cal.n == n_cal and g_te.n == n_test

        X_tr = torch.tensor(logit(g_tr.probs), dtype=torch.float32)
        y_tr = torch.tensor((g_tr.y * w).sum(axis=1), dtype=torch.long)
        X_ho = torch.tensor(logit(g_ho.probs), dtype=torch.float32)
        y_ho = torch.tensor((g_ho.y * w).sum(axis=1), dtype=torch.long)

        model, tinfo, hist = train_l1(X_tr, y_tr, X_ho, y_ho, S)

        # ---- conformal threshold on the conformal split ----
        s_cal = neg_log_softmax(model, torch.tensor(logit(g_cal.probs), dtype=torch.float32))
        true_cal = (g_cal.y * w).sum(axis=1)
        n_bad_true_cal = int((~admissible[true_cal]).sum())
        s_cal_con = np.where(admissible[None, :], s_cal, np.inf)
        true_scores = s_cal_con[np.arange(g_cal.n), true_cal]
        k = cc.conformal_rank(g_cal.n, ALPHA)
        threshold = cc.order_statistic(true_scores, k)

        # ---- test evaluation: L1 guarded and unguarded ----
        s_te = neg_log_softmax(model, torch.tensor(logit(g_te.probs), dtype=torch.float32))
        reg_g = region_l1(s_te, admissible, threshold, guard=True)
        reg_u = region_l1(s_te, admissible, threshold, guard=False)
        met_l1 = set_metrics(g_te, reg_g, spec_of_cc, m, labels)
        met_u = set_metrics(g_te, reg_u, spec_of_cc, m, labels)

        # ---- validation occupancy (all-13-clean validation rows, cluster projection) ----
        occ_val = int(len(np.unique((g_val.y * w).sum(axis=1))))

        # ---- Bonferroni on the same rows and splits ----
        thr_b, diag_b = bf.calibrate_per_label(g_cal, ALPHA, "bonferroni")
        met_b = cc.evaluate(g_te, thr_b)
        sizes_b = cc.prediction_sets(g_te, thr_b).sum(axis=2).prod(axis=1)
        ok_b = sizes_b >= 1
        met_b["log2_size_over_m_mean"] = float(np.mean(np.log2(sizes_b[ok_b]) / m)) if ok_b.any() else float("nan")
        met_b["_sizes"] = sizes_b
        met_b["joint_coverage_check"] = float(np.mean(
            cc.prediction_sets(g_te, thr_b)[np.arange(g_te.n)[:, None], np.arange(m)[None, :],
                                            g_te.y].all(axis=1)))

        common = {
            "cluster": cl.name, "m": m, "labels": "|".join(labels), "alpha": ALPHA,
            "n_cal": n_cal, "n_test": n_test,
            "n_states": S, "n_admissible_states": int(admissible.sum()),
            "n_validation_occupied_states": occ_val,
        }
        # validity gate uses the per-row test coverage; band from n_test
        for name, met, extra in [
            ("L1", met_l1, {"threshold": float(threshold), "k": int(k), **tinfo,
                            "n_train_rows": len(val_tr), "n_holdout_rows": len(val_ho),
                            "n_holdout_subjects": int(n_hold_subj),
                            "n_inadmissible_true_states_cal": n_bad_true_cal}),
            ("L1_unguarded", met_u, {"threshold": float(threshold), "k": int(k)}),
            ("bonferroni", met_b, {"threshold": json.dumps({lab: float(thr_b[lab]) for lab in labels}),
                                    "k": int(diag_b["k"]), "alpha_effective": float(diag_b["alpha_effective"])}),
        ]:
            gate = binomial_gate(met["joint_coverage"], n_test)
            row = {**common, "method": name, **extra,
                   "joint_coverage": met["joint_coverage"],
                   "band_lower": gate["band_lower"], "band_upper": gate["band_upper"],
                   "gate_pass_lower": gate["gate_pass_lower"], "in_two_sided_band": gate["in_two_sided_band"],
                   "mean_joint_set_size": met["mean_joint_set_size"],
                   "median_joint_set_size": met["median_joint_set_size"],
                   "log2_size_over_m_mean": met["log2_size_over_m_mean"],
                   "empty_set_rate": met["empty_set_rate"],
                   "fully_decisive_rate": met["fully_decisive_rate"],
                   "decisive_accuracy": met["decisive_accuracy"],
                   "hierarchy_violation_rate_decisive": met["hierarchy_violation_rate_decisive"]}
            records.append(row)
            metrics["clusters"].setdefault(cl.name, {})[name] = {
                k_: (v if isinstance(v, (bool, str, dict, list)) or v is None
                     else float(v))
                for k_, v in row.items() if k_ not in ("cluster", "labels")}

        metrics["clusters"][cl.name]["L1_training_history"] = hist
        metrics["clusters"][cl.name]["bonferroni_per_label_diag"] = {k_: v for k_, v in diag_b.items()
                                                                    if k_ != "positives_per_label"}
        metrics["clusters"][cl.name]["cal_positives_per_label"] = diag_b["positives_per_label"]
        metrics["clusters"][cl.name]["degenerate_L1"] = bool(
            (not np.isfinite(threshold)) or (met_l1["median_joint_set_size"] == admissible.sum()))

    res = pd.DataFrame(records)
    res_path = os.path.join(EXP_DIR, "results.csv")
    res.to_csv(res_path, index=False)

    metrics.update({
        "gate0": {"overlap_counts": overlap, "hierarchy_violations_by_split_cluster": gate0_hier,
                  "full_rows": full_rows, "full_subjects": full_subjects,
                  "all13_clean_rows": clean_rows, "all13_clean_subjects": clean_subjects},
        "holdout": {"n_validation_subjects_clean": int(subjects.size), "n_holdout_subjects": int(n_hold_subj),
                    "n_train_rows": len(val_tr), "n_holdout_rows": len(val_ho)},
        "n_cal": n_cal, "n_test": n_test,
        "runtime_seconds": time.time() - t0,
    })
    with open(os.path.join(EXP_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    inputs = {os.path.basename(p): sha256(p) for p in [
        os.path.join(cc.OUTPUTS_DIR, STEM + s) for s in ("_validation.csv", "_conformal.csv", "_test.csv")]}
    code = {os.path.relpath(p, REPO): sha256(p) for p in [
        os.path.join(REPO, "conformal_common.py"), os.path.join(REPO, "bonferrony.py"),
        os.path.join(REPO, "lp_scp.py"), os.path.join(REPO, "data", "hierarchy.py"),
        CLUSTERS_YAML, os.path.abspath(__file__)]}
    dirty = [l for l in git("status", "--porcelain").splitlines()]
    run_info = {
        "git_head": git("rev-parse", "HEAD"),
        "git_working_tree_dirty_entries": len(dirty),
        "seeds": {"torch": SEED, "numpy": SEED, "inner_holdout_rng": SEED, "dataloader_shuffle_generator": SEED},
        "torch_num_threads": 1,
        "package_versions": package_versions(),
        "input_sha256": inputs,
        "code_sha256": code,
        "stem": STEM,
        "command": "/usr/bin/python3 experiments/054_learned_score_pilot/run.py",
        "runtime_seconds": metrics["runtime_seconds"],
    }
    with open(os.path.join(EXP_DIR, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    # ---- print summary ----
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("Gate 0 overlaps:", overlap)
    print("all-13-clean rows:", clean_rows, "subjects:", clean_subjects)
    print(res[["cluster", "method", "n_cal", "n_test", "joint_coverage", "band_lower", "gate_pass_lower",
               "in_two_sided_band", "mean_joint_set_size", "median_joint_set_size",
               "log2_size_over_m_mean", "fully_decisive_rate", "empty_set_rate", "threshold"]].to_string())
    print("wrote", res_path)


if __name__ == "__main__":
    main()
