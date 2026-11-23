"""Exp059 driver: full-13 joint conformal, arms A (Bonferroni), B (max-rank), C (MLP full-13), D (MLP singletons).

Thin driver. Every computation is a call into the existing conformal package, except the small
diagnostic (unseen-state share), which uses conformal.data.state_index and scores.mlp.inner_holdout.
Pre-registration, gates and triggers: README.md in this folder. Run from the repo root:

    .venv/bin/python experiments/059_full13_bonf_maxrank_mlp/run.py

Gates are evaluated in the pre-registered order. Gate 2 is checked before any MLP arm is run.
Outputs are written before gates 3-6 are judged, so a failure still leaves the raw numbers on disk.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from conformal import data as D                       # noqa: E402
from conformal import evaluate, partition             # noqa: E402
from conformal import run as R                        # noqa: E402
from conformal.scores.mlp import MLP, inner_holdout   # noqa: E402

ALPHA, N_RES, SEEDS = 0.1, 100, [0, 1, 2, 3, 4]
METRIC_COLS = ["joint_coverage", "median_size", "log2_size_over_m", "mean_size", "empty_rate",
               "share_impossible", "violation_rate_decisive", "decisive_rate"]
COV_LO, COV_HI, EMPTY_MAX = 0.895, 0.905, 0.005
REPRO_MAXRANK_MEAN, REPRO_MAXRANK_ADM, REPRO_BONF_MEAN, POOL_N = 2855.40, 850.00, 3171.9, 37631
SOFT_C0_RANGE = (25.0, 36.0)
PAIRS = [("C0_minus_B", ("C", 0, "max_rank"), ("B", "n/a", "max_rank")),
         ("B_minus_A", ("B", "n/a", "max_rank"), ("A", "n/a", "bonferroni")),
         ("C0_minus_D0", ("C", 0, "max_rank"), ("D", 0, "max_rank"))]

GATES: dict = {}
MLP_INFO: dict = {}


def write_json(name: str, obj) -> None:
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


def gate(name: str, ok: bool, **detail) -> bool:
    GATES[name] = {"pass": bool(ok), **detail}
    write_json("gates.json", GATES)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(detail, default=float)}", flush=True)
    return ok


def need(name: str, ok: bool, reason: str, **detail) -> None:
    """Record a gate; on failure write ESCALATION.md (skeleton, expanded by hand) and stop."""
    if gate(name, ok, **detail):
        return
    with open(os.path.join(HERE, "ESCALATION_rerun.md"), "w") as f:
        f.write(f"# Exp059 ESCALATION (auto skeleton)\n\nGate failed: {name}\n\nReason: {reason}\n\n"
                f"Detail: {json.dumps(detail, default=float)}\n\n"
                "Raw outputs in this folder: resplits.csv, summary.csv, gates.json, metrics.json.\n")
    print(f"ESCALATED at {name}; stopping.", flush=True)
    raise SystemExit(2)


def main() -> None:
    t0 = time.time()

    # Gate 0 and Gate 1. load() runs gate0() (subject/image disjointness) and the hierarchy check.
    try:
        data = D.load()
    except RuntimeError as e:
        need("gate0_disjoint", False, "Gate 0 raised in conformal.data.load", error=str(e))
    counts = {k: int(v.n) for k, v in data.items()}
    gate("gate0_disjoint", True, note="conformal.data.load ran gate0 and the admissibility check without raising")
    pool_n = int(D.pool(data).n)
    need("gate1_pool_count", pool_n == POOL_N, f"pool rows {pool_n} != {POOL_N}",
         pool_n=pool_n, split_clean_counts=counts)

    single = partition.singletons()
    frames = []

    # Arms A and B: one call. Seed is irrelevant for these arms (no fitted scorer).
    ab = R.run(single, "per_label", rules=["bonferroni", "max_rank"], alpha=ALPHA,
               n_resplits=N_RES, seed=0, data=data)
    ab["arm"] = ab.rule.map({"bonferroni": "A", "max_rank": "B"})
    ab["seed"] = "n/a"
    frames.append(ab)
    mA = float(ab[(ab.arm == "A") & (ab.resplit >= 0)].mean_size.mean())
    mB = float(ab[(ab.arm == "B") & (ab.resplit >= 0)].mean_size.mean())
    admB = float(ab[(ab.arm == "B") & (ab.resplit >= 0)].mean_admissible_size.mean())
    fA = float(ab[(ab.arm == "A") & (ab.resplit == -1)].mean_size.iloc[0])
    covA = float(ab[(ab.arm == "A") & (ab.resplit >= 0)].joint_coverage.mean())
    need("gate2_reproduction", round(mB, 2) == REPRO_MAXRANK_MEAN and round(admB, 2) == REPRO_MAXRANK_ADM
         and round(mA, 1) == REPRO_BONF_MEAN,
         "anchor mismatch vs amended Gate 2 (README Amendment 2026-11-17); no MLP number has been read",
         maxrank_mean_size=mB, maxrank_mean_admissible_size=admB, bonferroni_mean_size=mA,
         bonferroni_frozen_mean_size=fA, bonferroni_coverage_resplit_mean=covA,
         targets={"maxrank_mean_size": REPRO_MAXRANK_MEAN, "maxrank_mean_admissible_size": REPRO_MAXRANK_ADM,
                  "bonferroni_mean_size_1dp": REPRO_BONF_MEAN})
    print(f"A/B done in {time.time() - t0:.0f}s", flush=True)

    # Arm C: one 13-label cluster, MLP scorer, seeds run one after another.
    for s in SEEDS:
        scorer = MLP()
        df = R.run([list(D.LABELS)], scorer, rules=["none", "max_rank"], alpha=ALPHA,
                   n_resplits=N_RES, seed=s, data=data)
        df["arm"], df["seed"] = "C", s
        frames.append(df)
        MLP_INFO[f"C_seed{s}"] = {k: (bool(v) if isinstance(v, (np.bool_, bool)) else v) for k, v in scorer.info.items()}
        print(f"C seed {s} done {time.time() - t0:.0f}s info={MLP_INFO[f'C_seed{s}']}", flush=True)
        if s == 0:
            c0_size = float(df[(df.rule == "max_rank") & (df.resplit >= 0)].mean_size.mean())
            if not (SOFT_C0_RANGE[0] <= c0_size <= SOFT_C0_RANGE[1]):
                gate("soft_c0_mean_size_range", False, c0_resplit_mean_size=c0_size, range=SOFT_C0_RANGE)
                with open(os.path.join(HERE, "ESCALATION_rerun.md"), "w") as f:
                    f.write(f"# Exp059 rerun ESCALATION (soft check)\n\nArm C seed 0 resplit-mean mean_size = {c0_size:.2f}, outside the soft range {SOFT_C0_RANGE}. Escalating before interpreting; seeds 1-4 not run.\n")
                raise SystemExit(2)
            gate("soft_c0_mean_size_range", True, c0_resplit_mean_size=c0_size, range=SOFT_C0_RANGE)
        del scorer, df
        gc.collect()

    # Arm D: singletons, MLP per label, seeds run one after another. info = last of 13 fits only.
    for s in SEEDS:
        scorer = MLP()
        df = R.run(single, scorer, rules=["max_rank"], alpha=ALPHA, n_resplits=N_RES, seed=s, data=data)
        df["arm"], df["seed"] = "D", s
        frames.append(df)
        MLP_INFO[f"D_seed{s}"] = {"last_label_fit": D.LABELS[-1],
                                  **{k: (bool(v) if isinstance(v, (np.bool_, bool)) else v) for k, v in scorer.info.items()}}
        print(f"D seed {s} done {time.time() - t0:.0f}s", flush=True)
        del scorer, df
        gc.collect()

    allr = pd.concat(frames, ignore_index=True)
    allr.to_csv(os.path.join(HERE, "resplits.csv"), index=False)
    summ = evaluate.summarize(allr, by=("arm", "seed", "rule"), metric_cols=METRIC_COLS)
    summ.to_csv(os.path.join(HERE, "summary.csv"), index=False)

    # Paired CIs (resplits >= 0), exact pairs fixed in the README.
    def sel(arm, seed, rule):
        return allr[(allr.arm == arm) & (allr.seed == seed) & (allr.rule == rule)]
    paired_rows = []
    for name, (a, b) in [(n, (x, y)) for n, x, y in PAIRS]:
        for metric in ["log2_size_over_m", "median_size"]:
            paired_rows.append({"pair": name, **evaluate.paired(sel(*a), sel(*b), metric)})
    pd.DataFrame(paired_rows).to_csv(os.path.join(HERE, "paired.csv"), index=False)

    # Diagnostic: share of test rows whose true 13-label state never appears among the MLP training rows.
    val = data["validation"]
    ho = inner_holdout(val.subject_id)
    train_states = np.unique(D.state_index(val.y[~ho]))
    pooled = D.pool(data)
    pidx = D.state_index(pooled.y)
    diag_rows = []
    for r, _cal, test in D.draws(data, N_RES):
        diag_rows.append({"resplit": r, "unseen_state_share": float(np.mean(~np.isin(pidx[test], train_states)))})
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(os.path.join(HERE, "diagnostics_unseen_state.csv"), index=False)
    diagnostics = {"mlp_train_rows": int((~ho).sum()), "mlp_holdout_rows": int(ho.sum()),
                   "distinct_train_states_13": int(train_states.size),
                   "unseen_share_frozen": float(diag[diag.resplit == -1].unseen_state_share.iloc[0]),
                   "unseen_share_resplit_mean": float(diag[diag.resplit >= 0].unseen_state_share.mean()),
                   "unseen_share_resplit_min": float(diag[diag.resplit >= 0].unseen_state_share.min()),
                   "unseen_share_resplit_max": float(diag[diag.resplit >= 0].unseen_state_share.max())}

    # Gate 3: identity of none and max_rank in arm C, every draw, every seed.
    ident = {}
    for s in SEEDS:
        c = allr[(allr.arm == "C") & (allr.seed == s)]
        a = c[c.rule == "none"].sort_values("resplit").drop(columns=["rule", "arm", "seed"]).reset_index(drop=True)
        b = c[c.rule == "max_rank"].sort_values("resplit").drop(columns=["rule", "arm", "seed"]).reset_index(drop=True)
        ident[s] = int((((a != b) & ~(a.isna() & b.isna())).any(axis=1)).sum())

    # Coverage and empty-rate table used by gates 4 and 5.
    def row(arm, seed, rule):
        return summ[(summ.arm == arm) & (summ.seed == seed) & (summ.rule == rule)].iloc[0]
    cov = {"A": float(row("A", "n/a", "bonferroni").joint_coverage_mean),
           "B": float(row("B", "n/a", "max_rank").joint_coverage_mean)}
    cov.update({f"C{s}": float(row("C", s, "max_rank").joint_coverage_mean) for s in SEEDS})
    cov.update({f"D{s}": float(row("D", s, "max_rank").joint_coverage_mean) for s in SEEDS})
    empt = {f"{r.arm}{r.seed}_{r.rule}": {"resplit_mean": float(r.empty_rate_mean), "frozen": float(r.empty_rate_frozen)}
            for r in summ.itertuples()}
    per_draw_max_empty = {f"{arm}{seed}_{rule}": float(g.empty_rate.max())
                          for (arm, seed, rule), g in allr.groupby(["arm", "seed", "rule"])}

    # Log-size comparison values, computed for the record; reported only if gate 5 passes.
    pair_out = {name: evaluate.paired(sel(*a), sel(*b), "log2_size_over_m")
                for name, a, b in PAIRS}
    c_log = {s: float(row("C", s, "max_rank").log2_size_over_m_mean) for s in SEEDS}
    spread_c = max(c_log.values()) - min(c_log.values())

    metrics = {
        "alpha": ALPHA, "n_resplits": N_RES, "seeds": SEEDS, "pool_rows": pool_n,
        "split_clean_counts": counts, "population": "all-13-clean rows (~77% of test)",
        "mean_size_A_resplit_mean": mA, "mean_size_B_resplit_mean": mB,
        "coverage_resplit_mean": cov, "empty_rate": empt, "per_draw_max_empty_rate": per_draw_max_empty,
        "identity_differing_draws_by_seed": ident, "mlp_info": MLP_INFO,
        "diagnostics": diagnostics, "paired_log2_size_over_m": pair_out,
        "c_seed_log2_size_over_m_resplit_mean": c_log, "c_seed_spread_log2_size_over_m": spread_c,
        "runtime_s": time.time() - t0,
    }
    write_json("metrics.json", metrics)
    print(f"all arms written {time.time() - t0:.0f}s", flush=True)

    # Gates 3-6, in order.
    need("gate3_identity", all(v == 0 for v in ident.values()),
         "none and max_rank rows differ in arm C; K=1 identity broken", differing_draws_by_seed=ident)

    cov_low = {k: v for k, v in cov.items() if v < COV_LO}
    need("gate4a_coverage_not_below_0.895", not cov_low,
         "joint coverage below 0.895 somewhere: possible bug", coverage_resplit_mean=cov)
    cov_high = {k: v for k, v in cov.items() if v > COV_HI and k != "A"}
    need("gate4b_coverage_not_above_0.905", not cov_high,
         "coverage above 0.905 (C: discreteness plateau; B/D: out of window). Report, do not tune; escalate",
         above_window=cov_high, coverage_resplit_mean=cov)

    bad_empty = {k: v for k, v in empt.items()
                 if max(v["resplit_mean"], v["frozen"]) > EMPTY_MAX}
    need("gate5_empty_rate", not bad_empty,
         "empty_rate > 0.5% in an arm; escalate before log-size comparison", failing=bad_empty)

    c0_hit = bool(MLP_INFO["C_seed0"]["hit_cap"])
    gap = abs(pair_out["C0_minus_B"]["mean_diff"])
    need("gate6_seed_noise", (not c0_hit) and spread_c <= gap,
         "hit_cap for C seed 0, or C seed spread exceeds |C0 - B|: seed noise, not a finding",
         hit_cap_C_seed0=c0_hit, c_seed_spread_log2_size_over_m=spread_c, abs_mean_diff_C0_minus_B=gap)

    print(f"ALL GATES PASSED in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
