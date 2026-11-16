# Exp051 — Cluster-granularity curve under max-rank (Rungs A–E, 13 → 1)

**Status:** PRE-REGISTERED 2026-11-01, before any metric was computed. Folder number 051 assigned by Alon
(048–050 already occupied). Spec: Tom Hagan, as sent to Jules 2026-11-01.

## What this tests
Granularity axis only. Five nested, strictly coarsening partitions (K = 13, 12, 7, 6, 1), each calibrated
per cluster with max-rank at α = 0.10 with no alpha-splitting. Measures how joint coverage, joint set size,
decisiveness and hierarchy violations move as clusters merge.

**Not THESIS_LOCK_PLAN item 3.** Item 3 compares partition *sources* at matched m. This experiment varies
*granularity* under one hierarchy-informed ladder. It does not answer item 3.

**Relation to 049_granularity_sweep and 050_direct_joint_coverage.** 049 appears to address a similar
granularity-curve question with a different ladder (six levels, 13→1, bypassing the m≤5 guard). This
experiment is independent of 049 and 050: separate ladder, separate pre-registration, no shared results.
Its results must not be treated as independent confirmation of 049 without this distinction stated.

## Inputs (frozen; not regenerated, no inference re-run)
- `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_conformal.csv` (calibration; = "conformal_split")
- `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809.csv` (test)
- `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_validation.csv` (loaded for Gate 0 only)
- Code reused as-is: `conformal_common.py` (loading, Gate 0, score, `extract_group`, `order_statistic`,
  `conformal_rank`, `hierarchy_admissible`, `CHEXPERT_HIERARCHY` via `data/hierarchy.py`),
  `max_rank.py::calibrate_max_rank` (Algorithm 2), `bonferrony.py::calibrate_per_label(correction="none")`
  (the independent per-label split-conformal baseline for the sanity check).
- Git: HEAD `6a26ac848b016e82c22d1bd42f61280740d3d26d`. Working tree has many uncommitted changes
  (see `git status`); none of the reused files were edited for this experiment.

## Locked parameters
- α = 0.10 (joint target 90%, per cluster, no split).
- Score: true-class nonconformity `s = 1−p if y=1 else p`, every row (`cc.nonconformity`).
- Method: max-rank (Algorithm 2) per cluster, independently. No λ (max-rank only).
- Seed: **0** recorded. The pipeline is deterministic (no resampling, no random step); the seed is recorded
  for reproducibility only.

## Partition ladder (pre-registered, nested, do not search)
- **A (K=13, m=1 each):** all 13 singletons.
- **B (K=12):** {Enlarged Cardiomediastinum, Cardiomegaly} m=2; 11 singletons.
- **C (K=7):** B's cardio pair m=2; {Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis} m=6;
  {Pneumothorax}, {Pleural Effusion}, {Pleural Other}, {Fracture}, {Support Devices} m=1 each.
- **D (K=6):** cardio pair m=2; lung group m=6; {Pleural Effusion, Pleural Other} m=2; {Pneumothorax}, {Fracture}, {Support Devices} m=1.
- **E (K=1):** all 13 labels, m=13.

Implementation note: `conformal_common.load_clusters` asserts 2 ≤ m ≤ 5. Rung A (m=1) and Rung E (m=13) are
outside it, so `run.py` constructs `ClusterSpec` objects directly from the pre-registered lists above. The
guard is not modified. This is the same exception Exp049 used, applied here for the same reason.

## Gate 0 (run first)
`cc.load_splits(with_validation=True)` runs the subject and dicom overlap check across the three splits.
Asserts: zero shared subjects and zero shared dicom_ids for every pair. Row counts asserted at 18,754
(conformal), 24,394 (test), 18,570 (validation). Subject counts at 5,926 / 6,395 / 5,898 are asserted too.

## Gate 0.5 (run before any results; does not change partition)
Positive-row counts (`y==1`) per label on the conformal split, for all 13 labels. Flag any count under 150
(`cc.MIN_POSITIVE_CALIBRATION_ROWS`). Also report the post-exclusion positive counts inside each cluster's
calibration group, since `extract_group` can drop rows. No label is excluded or moved because of this.

## Mandatory sanity check (Rung A vs independent per-label calibration)
- Rung A thresholds: `max_rank.calibrate_max_rank` per singleton.
- Reference: `bonferrony.calibrate_per_label(group, 0.10, "none")` per singleton, i.e. independent per-label
  split-conformal at α=0.10 on the same conformal rows.
- Metric: max |difference| over all thresholds. **Pass if ≤ 1e-12** (floating-point tolerance). Otherwise ESCALATION.
- **Interpretation (pre-registered, flagged):** the locked score is a single true-class score, and
  `prediction_sets` applies one threshold per label to both candidate classes. So the "positive" and "negative"
  thresholds of a label are the same number, t_j. The 26-threshold check therefore compares 13 values, each
  against its pair (t_j, t_j). Class-conditional thresholds would be a different score and are not part of this spec.

## Row population (interpretation, pre-registered)
- **Calibration:** `cc.extract_group(conformal, cluster_labels)` per cluster, as implemented. Per the
  RESEARCH_DIRECTION §3a note, `conformal_split` has no uncertain labels, so this equals the full conformal split.
- **Test metrics:** rows with any uncertain (−1) or missing value among the 13 labels are excluded, per the
  locked rule for all rungs. N is the same at every rung. This is the 13-label-clean test subpopulation
  (RESEARCH_DIRECTION §3a scope (b)), not the per-cluster-member population.
- **"Patient" unit:** the locked code and all prior experiments compute per row (one image). The spec says
  "patient". I use **rows** and report surviving row count and surviving subject count. Any other unit is not reported.

## Metrics (test split, per rung)
Let sets(i, j) ⊆ {0,1} be the per-label predicted set. Under max-rank the cluster region is a product, so the
per-label projection rule is exactly the per-label set.
1. **Joint coverage:** row = 1 iff for all 13 labels the true label ∈ set. Average over N rows. Empirical only for B–D (see caveat).
2. **Mean joint set size:** per row, product over all 13 per-label set sizes (equals the product over clusters of cluster-joint sizes). Average over N.
3. **Hierarchy-violation rate, fully decisive:** among rows with all 13 singleton, fraction whose full 13-label
   vector violates any edge in `CHEXPERT_HIERARCHY` (`cc.hierarchy_admissible` on all 13 labels, not per cluster).
4. **Decisive rate (per-label):** over 13×N pairs, fraction with singleton set. Empty sets are not singletons; their rate is reported separately.
5. **Fully-decisive rate (per-row):** fraction of rows with all 13 singletons. Reported as a distinct number from #4.

Also reported (secondary): empty-set rate, decisive accuracy, per-cluster k_star and n_calibration, and for Rung E the
distribution of joint set size (quantiles, count of 0, count of 1, count of 2^13) and per-label set-size distribution.

## Pre-registered escalation triggers (ESCALATION.md, then stop)
- T1: sanity check max |Δthreshold| > 1e-12.
- T2: any Rung B, C or D joint coverage ≥ 0.89. Operationalisation of "implausibly close to or above 90%". Chosen by me to
  make the trigger testable; Tom to confirm.
- T3: any reversal in the B→C→D sequence on joint coverage or on decisive rate. Reported as a finding only after Tom's ruling.
- T4: Rung E joint set size shows a degenerate tail not attributable to a flagged label. Concretely: any non-flagged
  label with empty sets for >0 rows, or a non-finite threshold on a non-flagged label.

## Standing caveats (stated in report, not footnotes)
- Shared-backbone cross-label correlation inflation (EC/CM 0.936→0.997). Any dependence reading partly measures the backbone; conservative direction.
- §3a calibration-population mismatch: the calibration split is 13-clean; the test metric population is 13-clean too, but this is scope (b) and a robustness check, not a headline.
- N-shrinkage from the 13-label listwise exclusion: separate from §3a. Report N per rung.
- For B–D, the within-cluster 90% target does **not** imply any bound on the full 13-label joint coverage. Empirical only.
- Gate 0.5 flagged label(s) are a caveat for Rung E and any cluster containing them.
- No numbers from deleted pre-2026-10-28 folders are cited.

## Reproduce
```
cd /Users/alonfines/Desktop/x-ray && /Users/alonfines/Desktop/x-ray/.venv_exp048/bin/python experiments/051_cluster_granularity_curve/run.py
```
