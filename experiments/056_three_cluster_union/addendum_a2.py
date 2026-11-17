#!/usr/bin/env python3
"""Exp056 Addendum A2 (post-sign-off, descriptive). Does not overwrite main-run outputs.

1. Retrain P_H C1-C3 seed 0 and arm D seed 0 with Exp055's train_scorer (via run.py, unedited import);
   persist best/stop epoch, cap_hit, holdout CE; assert per-resplit thresholds match the main run within 1e-9.
2. Conditional 13-label coverage (unguarded) for D and maxrank_adm vs C, on rows positive for Pleural Other,
   Fracture, Lung Lesion and rows with >= 1 positive; resplit mean and 2.5/97.5 percentiles.
3. Fully-decisive rates for C and D (from the main-run per-resplit columns).
Usage: /usr/bin/python3 experiments/056_three_cluster_union/addendum_a2.py
"""
import json, os, sys
import numpy as np, pandas as pd

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
import run as R  # unedited imports of the main run's helpers (training, splits, product regions)
from conformal_methods import max_rank as mr
from conformal_methods import bonferrony as bf  # noqa: F401
import conformal_common as cc

ALPHA = R.ALPHA
LABELS = R.LABELS
THR_TOL = 1e-9


def stop(trigger, detail):
    path = os.path.join(EXP_DIR, "ESCALATION.md")
    with open(path, "a") as f:
        f.write(f"\n\n## Addendum A2 stop: {trigger}\n\n```\n{json.dumps(detail, default=str, indent=1)[:4000]}\n```\n")
    print("STOP", trigger, detail)
    sys.exit(3)


def main():
    import torch
    torch.set_num_threads(1)
    splits = cc.load_splits(stem=R.STEM, with_validation=True, with_train=False, run_gate0=True)
    assert all(v == 0 for v in cc.gate0_check(splits, raise_on_overlap=True).values())
    val = R.all13_clean(splits.validation)
    conf13 = R.all13_clean(splits.calibration)
    test13 = R.all13_clean(splits.evaluation)
    pool_df = pd.concat([conf13, test13], ignore_index=True)
    assert len(pool_df) == R.EXPECT["pool_rows"]
    subjects = np.unique(val["subject_id"].to_numpy())
    rng = np.random.default_rng(R.HOLDOUT_SEED)
    perm = rng.permutation(subjects)
    hold = perm[:int(round(R.HOLDOUT_FRAC * subjects.size))]
    is_hold = val["subject_id"].isin(set(hold.tolist())).to_numpy()
    ctx = dict(cache={}, pool_df=pool_df,
               val_tr=val[~is_hold].reset_index(drop=True), val_ho=val[is_hold].reset_index(drop=True))
    subj = pool_df["subject_id"].to_numpy()
    uniq = np.unique(subj)
    n_conf = len(conf13)
    splits_list = [(-1, np.arange(n_conf), np.arange(n_conf, len(pool_df)))]
    for r in range(R.N_RESPLITS):
        perm_r = np.random.default_rng(r).permutation(uniq)
        inA = np.isin(subj, perm_r[: len(uniq) // 2])
        splits_list.append((r, np.flatnonzero(inA), np.flatnonzero(~inA)))
    pool13 = cc.extract_group(pool_df, LABELS)
    y13 = cc.state_indices(pool13.y)
    adm13 = cc.hierarchy_admissible(LABELS, cc.enumerate_states(13))
    states13 = cc.enumerate_states(13)

    main_vec = pd.read_csv(os.path.join(EXP_DIR, "resplit_metrics_vector.csv"))
    mC = main_vec[(main_vec.pid == "P_H") & (main_vec.arm == "C") & (main_vec.guard == "unguarded") & (main_vec.scorer_seed == 0)].set_index("resplit").sort_index()
    mD = main_vec[(main_vec.arm == "D") & (main_vec.guard == "unguarded")].set_index("resplit").sort_index()
    assert len(mC) == 101 and len(mD) == 101

    # ---------------- 1. retrain and threshold match ----------------
    diag = []
    P_H = R.P_H
    ents = [R.train_cluster_scorer(l, 0, ctx) for _, l in P_H]
    for (nm, l), e in zip(P_H, ents):
        diag.append(dict(unit=nm, labels="|".join(l), m=len(l), seed=0, **{k: e["info"][k] for k in
                    ("best_epoch", "stop_epoch", "hit_cap", "epochs_run", "final_holdout_ce")}))
    maxdiff_cluster = 0.0
    thr_rows = []
    for r, idxA, idxB in splits_list:
        S_A = np.column_stack([e["raw"][idxA, e["ycc"][idxA]] for e in ents])
        gA = cc.GroupData(labels=[nm for nm, _ in P_H], probs=np.zeros((idxA.size, 3)),
                          y=np.zeros((idxA.size, 3), dtype=int), scores=S_A, n_rows_before_filter=idxA.size, subject_id=None)
        thr, _ = mr.calibrate_max_rank(gA, ALPHA)
        for nm, _ in P_H:
            d = abs(thr[nm] - mC.loc[r, f"thr_{nm}"])
            maxdiff_cluster = max(maxdiff_cluster, d)
            thr_rows.append(dict(arm="C", unit=nm, resplit=r, retrained=thr[nm], main=mC.loc[r, f"thr_{nm}"], abs_diff=d))
    del ents

    dent = R.train_cluster_scorer(LABELS, 0, ctx)
    diag.append(dict(unit="D_L1_full13", labels="|".join(LABELS), m=13, seed=0, **{k: dent["info"][k] for k in
                ("best_epoch", "stop_epoch", "hit_cap", "epochs_run", "final_holdout_ce")}))
    raw32 = dent["raw"].astype(np.float32)
    yy = y13
    maxdiff_D = 0.0
    dregs = {}
    for r, idxA, idxB in splits_list:
        s_true = raw32[idxA, yy[idxA]]
        k = cc.conformal_rank(idxA.size, ALPHA)
        thr = cc.order_statistic(s_true, k)
        d = abs(thr - mD.loc[r, "thr_13"])
        maxdiff_D = max(maxdiff_D, d)
        thr_rows.append(dict(arm="D", unit="D_L1_full13", resplit=r, retrained=thr, main=mD.loc[r, "thr_13"], abs_diff=d))
    pd.DataFrame(thr_rows).to_csv(os.path.join(EXP_DIR, "addendum_a2_threshold_match.csv"), index=False)
    cap = pd.DataFrame(diag)
    cap.to_csv(os.path.join(EXP_DIR, "addendum_a2_cap_diagnostics.csv"), index=False)
    print(cap.to_string(), flush=True)
    print("max |thr diff| cluster C:", maxdiff_cluster, " D:", maxdiff_D, flush=True)
    if maxdiff_cluster > THR_TOL or maxdiff_D > THR_TOL:
        stop("addendum_threshold_mismatch", dict(cluster=maxdiff_cluster, D=maxdiff_D))

    # ---------------- 2. conditional coverage ----------------
    rows = []
    for r, idxA, idxB in splits_list:
        if r < 0:
            pass
        yB = pool13.y[idxB]
        masks = {f"pos_{l.replace(' ', '_')}": yB[:, LABELS.index(l)] == 1 for l in R.POS_SUBSETS}
        masks["any_pos"] = yB.sum(axis=1) >= 1
        # D (unguarded): region from float32 raw, same rule as run.py eval_full13
        thr = cc.order_statistic(raw32[idxA, yy[idxA]], cc.conformal_rank(idxA.size, ALPHA))
        regB = adm13[None, :] & (raw32[idxB] <= thr)
        covD = regB[np.arange(idxB.size), yy[idxB]]
        # maxrank_adm: 13-label max-rank product intersected with admissibility (run.py refs_eval_13)
        gA = R.subset(pool13, idxA)
        gB = R.subset(pool13, idxB)
        thr_m, dm = mr.calibrate_max_rank(gA, ALPHA)
        assert not dm.get("degenerate", False)
        regM = R.product_region(gB, thr_m) & adm13[None, :]
        covM = regM[np.arange(idxB.size), yy[idxB]]
        for arm, cov in (("D", covD), ("maxrank_adm", covM)):
            rec = dict(arm=arm, resplit=r)
            for mk, mm in masks.items():
                rec[f"cov_{mk}"] = float(cov[mm].mean()) if mm.any() else np.nan
                rec[f"n_{mk}"] = int(mm.sum())
            rows.append(rec)
        if r % 20 == 0:
            print("resplit", r, flush=True)
    cdf = pd.DataFrame(rows)
    # C from the main run's per-resplit columns (same masks, unguarded, seed 0)
    for r in range(-1, R.N_RESPLITS):
        rr = mC.loc[r]
        rec = dict(arm="C", resplit=r)
        for mk in ["pos_Pleural_Other", "pos_Fracture", "pos_Lung_Lesion", "any_pos"]:
            rec[f"cov_{mk}"] = float(rr[f"cov_{mk}"])
            rec[f"n_{mk}"] = int(rr[f"n_{mk}"])
        cdf = pd.concat([cdf, pd.DataFrame([rec])], ignore_index=True)
    cdf.to_csv(os.path.join(EXP_DIR, "addendum_a2_conditional_per_resplit.csv"), index=False)

    summ = []
    for arm in ("C", "D", "maxrank_adm"):
        for mk in ["pos_Pleural_Other", "pos_Fracture", "pos_Lung_Lesion", "any_pos"]:
            x = cdf[(cdf.arm == arm) & (cdf.resplit >= 0)]
            v = x[f"cov_{mk}"].to_numpy(float)
            n = x[f"n_{mk}"].to_numpy(float)
            summ.append(dict(arm=arm, subgroup=mk, n_mean=float(n.mean()), n_min=int(n.min()), n_max=int(n.max()),
                             cov_mean=float(np.nanmean(v)), cov_p2_5=float(np.nanpercentile(v, 2.5)),
                             cov_p97_5=float(np.nanpercentile(v, 97.5)), frozen=float(cdf[(cdf.arm == arm) & (cdf.resplit == -1)][f"cov_{mk}"].iloc[0])))
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(EXP_DIR, "addendum_a2_conditional_summary.csv"), index=False)
    print(sdf.round(4).to_string(index=False), flush=True)

    # ---------------- 3. decisive rates (main-run per-resplit columns) ----------------
    dec = {}
    for arm, src in (("C", mC), ("D", mD)):
        s = src[src.index >= 0]
        dec[arm] = dict(fully_decisive_rate_mean=float(s.fully_decisive_rate.mean()),
                        fully_decisive_rate_p2_5=float(np.percentile(s.fully_decisive_rate, 2.5)),
                        fully_decisive_rate_p97_5=float(np.percentile(s.fully_decisive_rate, 97.5)),
                        correct_decisive_rate_mean=float(s.correct_decisive_rate.mean()),
                        decisive_accuracy_mean=float(s.decisive_accuracy.mean()))
    json.dump(dict(cap=diag, threshold_max_abs_diff_cluster=maxdiff_cluster, threshold_max_abs_diff_D=maxdiff_D,
                   threshold_tolerance=THR_TOL, decisive=dec, conditional_summary=summ),
              open(os.path.join(EXP_DIR, "addendum_a2_metrics.json"), "w"), indent=2, default=float)
    print("decisive:", json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
