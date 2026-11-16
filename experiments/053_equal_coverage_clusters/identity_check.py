"""Exp053 identity gate (SPEC §8) over the frozen draw and all 100 resplits. p-value methods only.

Imports run.py's module-level pipeline (gates, partitions, designs) unchanged, then evaluates only the
K=13 and K=1 cells for Bonferroni, Sidak and max-rank on each design.
Run: cd /Users/alonfines/Desktop/x-ray && .venv_exp048/bin/python experiments/053_equal_coverage_clusters/identity_check.py
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R

pids = {p["pid"]: p for p in R.PARTS if p["pid"] in ("K13_singletons", "K1_all")}
rows = []

def check(D, tag, r):
    loc = {}
    for pid, p in pids.items():
        for mth in R.METHODS_P:
            loc[(pid, mth)] = R.run_cell(mth, p, D)
    for pid in pids:
        rec = {"design": tag, "resplit": r, "pid": pid}
        for a, b in (("bonf", "sidak"), ("bonf", "maxrank")):
            rec[f"{a}_vs_{b}"] = float(R.joint_sets_equal(loc[(pid, a)], loc[(pid, b)]).mean())
        rec["cal_cov_bonf"] = loc[(pid, "bonf")]["cal_cov"]
        rec["cal_cov_maxrank"] = loc[(pid, "maxrank")]["cal_cov"]
        rows.append(rec)

check(R.frozen_design(), "frozen", -1)
for r in range(R.N_RESPLITS):
    head, _, _ = R.resplit_designs(r)
    check(head, "resplit", r)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "identity_gates_postamendment3_all.csv"), index=False)
for pid in pids:
    s = df[(df.pid == pid) & (df.design == "resplit")]
    f = df[(df.pid == pid) & (df.design == "frozen")]
    print(pid, "resplit bonf_vs_maxrank: min %.5f median %.5f max %.5f  n>0.5%%: %d" % (
        s["bonf_vs_maxrank"].min(), s["bonf_vs_maxrank"].median(), s["bonf_vs_maxrank"].max(), (s["bonf_vs_maxrank"] > 0.005).sum()))
    print(pid, "frozen bonf_vs_maxrank %.5f ; bonf_vs_sidak max %.5f" % (f["bonf_vs_maxrank"].iloc[0], df[df.pid == pid]["bonf_vs_sidak"].max()))
    print(pid, "resplit cal_cov_maxrank mean %.5f sd %.5f" % (s["cal_cov_maxrank"].mean(), s["cal_cov_maxrank"].std()))
