# Exp053 — ESCALATION (unpinned items in SPEC.md; nothing run)

**Date:** 2026-11-10, 12:09 IST. **Git HEAD:** `6a26ac848b016e82c22d1bd42f61280740d3d26d` (working tree dirty, as in 051/052). **Status:** no `run.py`, no README pre-registration, no gate run, no metric computed. Stopped at step 1 of the brief, per the rule that an unpinned spec item is escalated, not chosen.

## Spec, one line
Partition the 13 labels into K clusters, run Bonferroni / Šidák / max-rank / LP-SCP inside each cluster at one common β̂ calibrated to 90% joint coverage on pooled conformal ∪ test (37,631 rows, 13-clean), 100 subject-level 50/50 resplits, and compare partitions on log₂(size)/13 at equal coverage.

## Inputs (confirmed present)
- `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_{conformal,validation}.csv` and the bare `…809.csv` (test). Not yet opened or checked; Gate 0 not run.
- Reusable code present: `conformal_common.py`, `max_rank.py`, `bonferrony.py`, `lp_scp.py`, `data/hierarchy.py`.
- `.venv_exp048` (numpy 2.5.4, pandas 3.1.0, scipy 1.18.1).
- Exp051 Gate 0.5 file present: Pleural Other 151, Fracture 304, Lung Lesion 523 conformal positives (matches SPEC §7 item 7).

## Unpinned or inconsistent items (need a ruling before pre-registration)

**E1. Hierarchy-edge count in the matched null (§3, "out of 6").** `data/hierarchy.py` has **7** parent→child edges among the 13 labels: Cardiomegaly←Enlarged Cardiomediastinum; Lung Lesion, Edema, Consolidation, Atelectasis, Pneumonia ←Lung Opacity; Pneumonia←Consolidation. There are **6** distinct child labels, and Pneumonia has two parents. "Out of 6" could mean 6 child nodes or a mistaken total of edges. The matched null selects partitions on this count, so the choice changes the sample. Which count: (a) directed parent→child edges, total 7; (b) child labels with a parent, total 6; (c) something else?

**E2. Difficulty control (§3).** "Sort labels by validation AUC, chunk contiguously into the informed profile." Not pinned: ascending or descending AUC, and which contiguous chunk receives which size from the profile (profile order). These give different partitions for asymmetric profiles.

**E3. Clinical random null, profiles C and D (§3, secondary).** 100 random partitions each, no seed stated. §10 lists only 1000+idx (informed null), 1500 (matched), 2000 (anti-clustering). Need a seed or a rule.

**E4. Methods on the matched null (§3, §4).** §4 says "every partition" gets all four methods, but the "LP-SCP uses the first 100 of each list" restriction is written only for the 200-partition random null. Confirm: LP-SCP on all 200 matched partitions, or first 100? Also "keep 200": first 200 matches in generation order, or a random 200 of the matches? Not stated.

**E5. Anti-clustering swap hill-climb (§3).** Acceptance rule (first-improvement vs. best-improvement swap) and stop rule are not stated. Both reach local minima that can differ. Also confirm one RNG stream for the 50 restarts from seed 2000.

**E6. Random partition seeding (§3, §10).** "Seeds 1000+idx": I read this as one independent RNG per partition, seeded 1000+idx, reused across K=3, 4, 6. Confirm.

**E7. H2 percentile convention (§7).** "Informed below the null's 5th percentile": the interpolation convention (e.g. `np.percentile` linear) is not stated. The rank p-value is pinned; the boundary comparison is not.

**E8. Resplit roles and linkage ties (§5, §3).** I take subject half A = calibration, half B = test, with 11,858 pooled subjects split 5,929 / 5,929. Confirm the orientation. Average-linkage ties are not addressed; I would break them by lowest label index and record any tie.

## Open from Exp052 (relevant to the frozen-draw row)
Exp052's ESCALATION.md (2026-11-01) asked five questions. The K=1 max-rank joint coverage on the 13-clean frozen test was 0.8934 against a 0.90 target. The folder has no recorded ruling. The frozen-draw row in this experiment will show the same population effect, and the 052 explanations (§3a shift vs. subject dependence) are still untested. Please confirm whether 052's question 1 is resolved before I read the frozen-draw row.

## What I need from Tom
1. A ruling on E1 (which count, with the matched-null target).
2. A ruling on E2, E3, E4, E5 (choose the reading, or pin the value).
3. Confirmation of E6–E8 readings, or corrections.
4. Confirm or re-scope the 052 open question as above.

Once these are pinned I will pre-register the README (HEAD, time, every parameter and seed), run Gate 0, Gate 0.5 and the identity gates (§8), time one resplit, and report. Nothing in this folder has been computed beyond the input-file existence check.
