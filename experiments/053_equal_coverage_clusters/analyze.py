"""Exp053 analysis: resplit_metrics.csv + run_info.json -> metrics.json and summary CSVs.

Pre-registered analysis (README + SPEC §5–§7). Reads only the run outputs in this folder.
Run: cd /Users/alonfines/Desktop/x-ray && .venv_exp048/bin/python experiments/053_equal_coverage_clusters/analyze.py
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
df = pd.read_csv(os.path.join(HERE, "resplit_metrics.csv"))
info = json.load(open(os.path.join(HERE, "run_info.json")))
ident = pd.read_csv(os.path.join(HERE, "identity_gates.csv"))
PARTS = {p["pid"]: p for p in info["partitions"]}

METHODS_P = ["bonf", "sidak", "maxrank"]
ALL_M = METHODS_P + ["lpscp"]
MNAME = {"bonf": "Bonferroni", "sidak": "Sidak", "maxrank": "max-rank", "lpscp": "LP-SCP"}
TARGET_LO, TARGET_HI = 0.895, 0.905


def pct(a, q):
    return float(np.percentile(a, q))  # linear interpolation (descriptive only, SPEC E7)


res = df[df.design == "resplit"].copy()
frozen = df[df.design == "frozen"].copy()
split = df[df.design == "split_arm"].copy()
NR = res.resplit.nunique()
assert NR == info["n_resplits"], (NR, info["n_resplits"])

# ------------------------------------------------------------------ cell summary (headline)
cells = []
for pid, g in res.groupby(["pid", "method"]):
    pid, mth = pid
    K = int(g.K.iloc[0])
    fam = g.family.iloc[0]
    jc = g.joint_cov.values
    lg = g.mean_log2_13.values
    fz = frozen[(frozen.pid == pid) & (frozen.method == mth)]
    rec = {
        "pid": pid, "method": mth, "family": fam, "K": K, "n_resplits": len(g),
        "joint_cov_mean": jc.mean(), "joint_cov_sd": jc.std(ddof=1),
        "joint_cov_p2.5": pct(jc, 2.5), "joint_cov_p97.5": pct(jc, 97.5),
        "H1_in_band_0.895_0.905": bool(TARGET_LO <= jc.mean() <= TARGET_HI),
        "coverage_unequal": bool(not (TARGET_LO <= jc.mean() <= TARGET_HI)),
        "beta_hat_mean": g.beta_hat.mean(), "beta_hat_sd": g.beta_hat.std(ddof=1),
        "indep_beta_1-0.9^(1/K)": 1 - 0.9 ** (1.0 / K),
        "cal_cov_mean": g.cal_cov.mean(),
        "cluster_cov_mean": g.cluster_cov_mean.mean(), "cluster_cov_min_mean": g.cluster_cov_min.mean(),
        "mean_log2_13_mean": lg.mean(), "mean_log2_13_sd": lg.std(ddof=1),
        "mean_size_mean": g.mean_size.mean(), "median_size_mean": g.median_size.mean(),
        "fully_dec_mean": g.fully_dec.mean(), "empty_rate_mean": g.empty_rate.mean(),
        "per_label_dec_mean": g.per_label_dec.mean(),
        "hv_rate_fd_mean": g.hv_rate_fd.mean(), "dec_acc_mean": g.dec_acc.mean(),
        "cov_flagged_mean": g.cov_flag.mean(), "cov_unflagged_mean": g.cov_unflag.mean(),
        "W": g.W.iloc[0], "edges_within": int(g.edges_within.iloc[0]),
        "frozen_joint_cov": float(fz.joint_cov.iloc[0]) if len(fz) else np.nan,
        "frozen_joint_cov_percentile": float(np.mean(jc <= fz.joint_cov.iloc[0])) if len(fz) else np.nan,
        "frozen_mean_log2_13": float(fz.mean_log2_13.iloc[0]) if len(fz) else np.nan,
        "frozen_mean_log2_percentile": float(np.mean(lg <= fz.mean_log2_13.iloc[0])) if len(fz) else np.nan,
    }
    cells.append(rec)
cell = pd.DataFrame(cells)

# ------------------------------------------------------------------ per-resplit paired deltas
piv = res.pivot_table(index=["resplit"], columns=["pid", "method"], values="mean_log2_13")
size_piv = res.pivot_table(index=["resplit"], columns=["pid", "method"], values="mean_size")
K1 = ("K1_all", "maxrank")


def null_pids(family_key, K, method):
    pids = [pid for pid, p in PARTS.items() if p["family"] == family_key and p["K"] == K]
    if method == "lpscp":
        pids = [pid for pid in pids if PARTS[pid]["lp"]]
    return pids


deltas = []
null_summary = []
for K in (3, 4, 6):
    for mth in ALL_M:
        nulls = null_pids("random", K, mth)
        matched = null_pids("matched", K, mth)
        if not nulls:
            continue
        null_mat = piv[[(pid, mth) for pid in nulls]]
        matched_mat = piv[[(pid, mth) for pid in matched]] if matched else None
        null_avg = null_mat.mean(axis=0).values                    # resplit-average per null partition
        null_med_per_resplit = null_mat.median(axis=1).values
        inf_pid = f"informed_K{K}"
        inf_avg = piv[(inf_pid, mth)].mean()
        null_summary.append({
            "family": "random_primary", "K": K, "method": mth, "N_null": len(nulls),
            "null_mean_avg": float(null_avg.mean()), "null_p5": pct(null_avg, 5), "null_p50": pct(null_avg, 50),
            "informed_mean_avg": float(inf_avg),
            "rank_p": float((1 + np.sum(null_avg <= inf_avg)) / (len(nulls) + 1)),
            "informed_below_null_p5": bool(inf_avg < pct(null_avg, 5)),
        })
        if matched_mat is not None:
            m_avg = matched_mat.mean(axis=0).values
            null_summary.append({
                "family": "matched_secondary", "K": K, "method": mth, "N_null": len(matched),
                "null_mean_avg": float(m_avg.mean()), "null_p5": pct(m_avg, 5), "null_p50": pct(m_avg, 50),
                "informed_mean_avg": float(inf_avg),
                "rank_p": float((1 + np.sum(m_avg <= inf_avg)) / (len(matched) + 1)),
                "informed_below_null_p5": bool(inf_avg < pct(m_avg, 5)),
            })
        # paired deltas per resplit
        inf_series = piv[(inf_pid, mth)].values
        deltas.append({
            "comparison": f"informed_K{K}_{mth} minus null_median(random_K{K})", "K": K, "method": mth,
            "delta_mean_log2": float(np.mean(inf_series - null_med_per_resplit)),
            "delta_p2.5": pct(inf_series - null_med_per_resplit, 2.5),
            "delta_p97.5": pct(inf_series - null_med_per_resplit, 97.5),
        })
        deltas.append({
            "comparison": f"informed_K{K}_{mth} minus K1_maxrank", "K": K, "method": mth,
            "delta_mean_log2": float(np.mean(inf_series - piv[K1].values)),
            "delta_p2.5": pct(inf_series - piv[K1].values, 2.5),
            "delta_p97.5": pct(inf_series - piv[K1].values, 97.5),
        })
        # size ratios (arithmetic mean size) vs K1 max-rank and vs null median (per resplit ratio of means)
        size_inf = size_piv[(inf_pid, mth)].values
        size_k1 = size_piv[K1].values
        deltas.append({
            "comparison": f"mean_size ratio informed_K{K}_{mth} / K1_maxrank", "K": K, "method": mth,
            "delta_mean_log2": float(np.mean(size_inf / size_k1)),
            "delta_p2.5": pct(size_inf / size_k1, 2.5), "delta_p97.5": pct(size_inf / size_k1, 97.5),
        })

# H3 anti-clustering
h3 = []
for K in (3, 4, 6):
    for mth in ["maxrank"]:
        nulls = null_pids("random", K, mth)
        null_avg = piv[[(pid, mth) for pid in nulls]].mean(axis=0).values
        anti_avg = piv[(f"anti_K{K}", mth)].mean()
        inf_avg = piv[(f"informed_K{K}", mth)].mean()
        h3.append({"K": K, "method": mth, "anti_mean_avg": float(anti_avg), "informed_mean_avg": float(inf_avg),
                   "null_median_avg": float(np.median(null_avg)),
                   "above_null_median": bool(anti_avg > np.median(null_avg)),
                   "above_informed": bool(anti_avg > inf_avg),
                   "H3_sign_pass": bool(anti_avg > np.median(null_avg) and anti_avg > inf_avg)})

# H4 dose-response (descriptive)
h4 = []
for K in (3, 4, 6):
    for mth in ALL_M:
        nulls = null_pids("random", K, mth)
        if len(nulls) < 3:
            continue
        W = np.array([PARTS[p]["W"] for p in nulls])
        lg = piv[[(p, mth) for p in nulls]].mean(axis=0).values
        rho, pv = spearmanr(W, lg)
        h4.append({"family": "random_primary", "K": K, "method": mth, "n": len(nulls),
                   "spearman_W_vs_mean_log2": float(rho), "p_descriptive": float(pv)})

# H2 (primary: max-rank, Holm over three cuts)
h2_rows = [r for r in null_summary if r["family"] == "random_primary" and r["method"] == "maxrank"]
h2_rows = sorted(h2_rows, key=lambda r: r["rank_p"])
holm = {}
alpha = 0.05
stop = False
for i, r in enumerate(h2_rows):
    thr = alpha / (len(h2_rows) - i)
    if not stop and r["rank_p"] <= thr:
        holm[r["K"]] = {"p": r["rank_p"], "holm_threshold": thr, "reject": True}
    else:
        stop = True
        holm[r["K"]] = {"p": r["rank_p"], "holm_threshold": thr, "reject": False}

# Matched null counts
matched_counts = {str(K): {"informed_edges_within": info["matched"][str(K)]["informed_edges"],
                           "pool_matches_of_5000": info["matched"][str(K)]["pool_matches_of_5000"],
                           "kept": info["matched"][str(K)]["kept"]} for K in (3, 4, 6)}

# Split-calibration arm (informed K=4)
split_rows = []
for mth in ALL_M:
    s = split[split.method == mth]
    h = res[(res.pid == "informed_K4") & (res.method == mth)].sort_values("resplit")
    s = s.sort_values("resplit")
    split_rows.append({
        "method": mth, "n_resplits": len(s),
        "split_joint_cov_mean": s.joint_cov.mean(), "split_joint_cov_sd": s.joint_cov.std(ddof=1),
        "headline_joint_cov_mean": h.joint_cov.mean(),
        "split_beta_hat_mean": s.beta_hat.mean(), "headline_beta_hat_mean": h.beta_hat.mean(),
        "split_mean_log2_13": s.mean_log2_13.mean(), "headline_mean_log2_13": h.mean_log2_13.mean(),
        "paired_delta_mean_log2(split-headline)": float((s.mean_log2_13.values - h.mean_log2_13.values).mean()),
        "split_frac_in_band": float(np.mean((s.joint_cov >= TARGET_LO) & (s.joint_cov <= TARGET_HI))),
    })
split_df = pd.DataFrame(split_rows)

# Triggers (SPEC §9)
k1sd = res[(res.pid == "K1_all") & (res.method == "maxrank")].joint_cov.std(ddof=1) * 100
triggers = {
    "identity_gate_all_resplits_max_frac": float(ident.bonf_vs_maxrank.max()) if "bonf_vs_maxrank" in ident else None,
    "identity_gate_pass": bool((ident[ident.design.isin(["frozen", "resplit"])].bonf_vs_maxrank <= 0.005).all()),
    "identity_gate_resplits_0_9_pass": bool(ident[(ident.design == "resplit") & (ident.resplit <= 9)].bonf_vs_maxrank.le(0.005).all()),
    "K1_maxrank_resplit_SD_pp": float(k1sd),
    "K1_SD_trigger_fired": bool(k1sd < 0.25 or k1sd > 0.8),
    "pvalue_cells_outside_band": [f"{r['pid']}|{r['method']}" for _, r in cell.iterrows() if r["method"] != "lpscp" and not r["H1_in_band_0.895_0.905"]],
    "lp_cells_outside_band": [f"{r['pid']}|{r['method']}" for _, r in cell.iterrows() if r["method"] == "lpscp" and not r["H1_in_band_0.895_0.905"]],
    "informed_stall_or_cluster_gt7": False,
    "beta_undefined": False,
}
triggers["escalation_fired"] = bool(not triggers["identity_gate_pass"] or triggers["K1_SD_trigger_fired"]
                                    or len(triggers["pvalue_cells_outside_band"]) > 0)

# ------------------------------------------------------------------ write
cell.to_csv(os.path.join(OUT, "cell_summary.csv"), index=False)
frozen_out = frozen.copy()
frozen_out.to_csv(os.path.join(OUT, "frozen_draw_rows.csv"), index=False)
pd.DataFrame(null_summary).to_csv(os.path.join(OUT, "null_summary.csv"), index=False)
pd.DataFrame(deltas).to_csv(os.path.join(OUT, "paired_deltas.csv"), index=False)
pd.DataFrame(h4).to_csv(os.path.join(OUT, "dose_response.csv"), index=False)
split_df.to_csv(os.path.join(OUT, "split_arm_summary.csv"), index=False)
ident.to_csv(os.path.join(OUT, "identity_gates_summary.csv"), index=False)

metrics = {
    "experiment": "053_equal_coverage_clusters",
    "n_resplits": int(NR),
    "gate0": info["gate0"],
    "verification": info["verification"],
    "informed_partitions": info["informed"],
    "informed_profiles": info["informed_profiles"],
    "informed_ties": info["informed_ties"],
    "matched_null_counts": matched_counts,
    "anti_clustering": info["anti"],
    "difficulty_auc_validation": info["auc_validation"],
    "context_rows_frozen": info["context_rows_frozen"],
    "H2_primary_maxrank_holm": holm,
    "H2_null_summary": null_summary,
    "H3_anti": h3,
    "H4_dose_response": h4,
    "paired_deltas": deltas,
    "split_arm": split_rows,
    "triggers": triggers,
    "scope": "studies with no uncertain label on any of the 13 categories, ~77% of test",
    "pop_pool_rows": 37631,
    "alpha": 0.10,
    "caveat_pooling": "Resplit intervals are conditional on the fixed 37,631-row pool; pooling conformal and test understates total uncertainty. The frozen draw is a reference row only (no pass/fail).",
}
with open(os.path.join(OUT, "metrics.json"), "w") as fh:
    json.dump(metrics, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("analysis written; resplits:", NR, "escalation_fired:", triggers["escalation_fired"])
