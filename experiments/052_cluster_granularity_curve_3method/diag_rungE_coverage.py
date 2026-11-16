"""Post-hoc diagnostic (run AFTER the Exp052 main run, to diagnose the Rung E shortfall).

Not part of the pre-registered metric set. Evaluates the already-computed Rung E thresholds
(max-rank cited from Exp051 metrics.json; Bonferroni/Sidak recomputed by bonferrony.py) on
(a) the conformal/calibration split itself (in-sample), and (b) the 13-clean test population.
Question it answers: is the Rung E test shortfall (~0.893 vs 0.90) already present on the
calibration split, or does it appear only on test?

Run:
    cd /Users/alonfines/Desktop/x-ray && .venv_exp048/bin/python experiments/052_cluster_granularity_curve_3method/diag_rungE_coverage.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import conformal_common as cc
import bonferrony

ALL13 = cc.ALL_LABELS
ALPHA = 0.10
splits = cc.load_splits(with_validation=False, run_gate0=True)
cal_all = cc.extract_group(splits.calibration, ALL13)
test_g = cc.extract_group(splits.evaluation, ALL13)
c051 = json.load(open(os.path.join(REPO, "experiments", "051_cluster_granularity_curve", "metrics.json")))
thr_maxrank = {}
for c in c051["rungs"]["E"]["clusters"]:
    thr_maxrank.update(c["thresholds"])
thr_bonf, _ = bonferrony.calibrate_per_label(cal_all, ALPHA, "bonferroni")
thr_sidak, _ = bonferrony.calibrate_per_label(cal_all, ALPHA, "sidak")

def cov(g, thr):
    sets = cc.prediction_sets(g, thr)
    tk = np.take_along_axis(sets, g.y[:, :, None], axis=2)[:, :, 0]
    return float(tk.all(axis=1).mean()), int(g.n)

out = {"alpha": ALPHA, "note": "post-hoc diagnostic, not pre-registered", "rows": {}}
for name, thr in (("maxrank_cited_051", thr_maxrank), ("bonferroni", thr_bonf), ("sidak", thr_sidak)):
    c_cal, n_cal = cov(cal_all, thr)
    c_test, n_test = cov(test_g, thr)
    out["rows"][name] = {"conformal_insample_joint_coverage": c_cal, "n_conformal": n_cal,
                         "test_13clean_joint_coverage": c_test, "n_test": n_test}
    print(f"{name:20s} conformal(in-sample)={c_cal:.4f} (n={n_cal})   test13clean={c_test:.4f} (n={n_test})")
json.dump(out, open(os.path.join(HERE, "diag_rungE_coverage.json"), "w"), indent=2)
