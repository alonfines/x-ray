# Exp052 — Cluster-granularity curve, 3-method comparison (max-rank, Bonferroni+Šidák, LP-SCP), Rungs A–E

**Status:** PRE-REGISTERED 2026-11-01, before any 052 code was run and before any 052 metric was computed.
Spec: Tom Hagan, finalized and sent to Jules 2026-11-01. Folder number 052 confirmed free on 2026-11-01
(highest existing folder: 051).

## Spec restated (one line)
Extend Exp051 (max-rank only, Rungs A→E) to three methods on the identical 5-rung ladder: max-rank (cited from 051, not rerun),
Bonferroni and Šidák (run, per cluster, one pass each), and LP-SCP (run, `penalised` arm, μ=ν=1, per cluster, no α-splitting).
Report all 5 metrics × 5 rungs × 3 methods with the required caveats; escalate on the four triggers below.

## Inputs (frozen; no inference re-run)
- `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_conformal.csv` (calibration)
- `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_test.csv` (test; note the `_test` suffix)
- `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_validation.csv` (LP-SCP proper-training fit, and Gate 0)
- Cited (not rerun): `experiments/051_cluster_granularity_curve/metrics.json`, `rungs_summary.csv`, `gate05_positive_counts.csv`
- Code reused as-is (not edited): `conformal_common.py`, `bonferrony.py` (`calibrate_per_label`, `correction ∈ {bonferroni, sidak}`), `lp_scp.py` (`fit_structure`, `build_region`, `true_state_scores`, `state_scores` used for verification only), `data/hierarchy.py`.
- Git HEAD: `6a26ac848b016e82c22d1bd42f61280740d3d26d`. Working tree is dirty (see `git status`); none of the reused files were edited for this experiment.
- SHA-256 of cited/reused files at pre-registration: `.prereg_hashes.txt` in this folder.

## Seed
Seed **0**, same convention as Exp051 (`np.random.seed(0)`). The pipeline has no stochastic step; the seed is recorded for reproducibility only.

## Locked parameters
- α = 0.10 (joint target 90%, per cluster, **no α-splitting across clusters**).
- Locked score (max-rank, Bonferroni, Šidák): `s = 1−p if y=1 else p`, every row.
- LP-SCP score: paper eq. 30, `a(x,y) = sqrt(rᵀΣ⁻¹r) + μ·Hp(y) + ν·Cp(y)`, `r = |y − o(x)|`, μ = ν = 1, `penalised` arm (no hard hierarchy constraint).

## Partition ladder (copied verbatim from Exp051 README; do not search)
- **A (K=13):** all 13 singletons, m=1 each.
- **B (K=12):** {Enlarged Cardiomediastinum, Cardiomegaly} m=2; 11 singletons.
- **C (K=7):** B's cardio pair m=2; {Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis} m=6; {Pneumothorax}, {Pleural Effusion}, {Pleural Other}, {Fracture}, {Support Devices} m=1.
- **D (K=6):** cardio pair m=2; lung group m=6; {Pleural Effusion, Pleural Other} m=2; {Pneumothorax}, {Fracture}, {Support Devices} m=1.
- **E (K=1):** all 13 labels, m=13.

## Gate 0 (run first, in this experiment)
`cc.load_splits(with_validation=True)` + `cc.gate0_check`: zero shared subjects and zero shared dicom_ids for all three pairs (conformal|test, conformal|validation, test|validation). Row counts 18,754 / 24,394 / 18,570; subject counts 5,926 / 6,395 / 5,898.
(Already satisfied project-wide; re-asserted here, not rederived beyond the code's own check.)

## Gate 0.5 (reused, not recomputed)
Cite `experiments/051_cluster_granularity_curve/gate05_positive_counts.csv`. Result in 051: no label below 150 positive calibration rows (lowest: Pleural Other 151, Fracture 304). 052 asserts the flag list is empty by reading that file. Note: Pleural Other at 151 sits one row above the floor; this is a caveat for Rung C/D/E, not an exclusion.

## Rung A sanity checks
1. **Bonferroni and Šidák Rung A vs Exp051 max-rank Rung A.** At m=1 both corrections reduce to α, so both should equal the independent per-label threshold, which 051 showed equals its max-rank Rung A threshold. Compared per label against 051's cited Rung A thresholds. **Pass if max |Δ| ≤ 1e-12.** Also: the Rung A joint metrics computed by 052's code for Bonferroni and Šidák must equal 051's cited Rung A metrics (tolerance 1e-12). Mismatch = code-path bug → escalate.
2. **Interpretation carried from 051 (flag (a), verbatim):** the locked score is a single true-class score, so positive and negative thresholds of a label are the same number, t_j. Rung A's "sanity check passed" is a code-path check, not an independent correctness proof.
3. **Flag (b), verbatim:** `experiments/049_granularity_sweep` is a different, independent experiment with a different ladder. Exp052 is not a confirmation of 049.

## Methods

### Max-rank — NOT RERUN
Cited from `experiments/051_cluster_granularity_curve/metrics.json` (rungs A–E) and `rungs_summary.csv`. Same α, same score, same test population (N=18,877), same projection rule.

### Bonferroni and Šidák — RUN, one pass
`bonferrony.calibrate_per_label(cal_cluster_group, 0.10, correction)` for `correction ∈ {"bonferroni", "sidak"}`, independently per cluster.
Levels: `alpha/m` (Bonferroni), `1−(1−α)^(1/m)` (Šidák). Per-label thresholds, locked score.
Joint metrics computed with the same per-label-set machinery as 051 (`cc.prediction_sets`, product over labels).

### LP-SCP — RUN, `penalised` arm, μ=ν=1, per cluster
- **Split roles (per `lp_scp.py` convention):** validation → proper-training (fit Σ, Hp vocabulary, Cp frequencies) via `cc.extract_group(validation, cluster)` and `lp_scp.fit_structure`; conformal → calibration quantile; test → evaluation.
- **Calibration:** k = `cc.conformal_rank(n_cal, α)`, threshold = `cc.order_statistic(true-state scores on conformal, k)`. Same rule as `lp_scp.calibrate_lp_scp`.
- **Region:** `{state : score ≤ threshold}` over the cluster's 2^m states, per test row.
- **Hierarchy:** NOT enforced (`enforce_hierarchy=False`). The `admissible` arm is not run.
- **Non-empty guard (paper eq. 20):** the spec does not pin this. `lp_scp.py` computes both variants. **Both are reported side by side** (`guard_off` = raw region, `guard_on` = best-scoring state unioned in). Neither is designated primary in this README; Tom to rule (see Triggers / open question).
- **Compute (implementation, pre-registered):** `lp_scp.state_scores` materialises an (n, 2^m, m) array. At m=13 on the 18,754-row calibration split that is ≈16 GB, which exceeds this 24 GB machine's practical headroom. 052 computes the **same formula** (same `np.einsum('nsi,ij,nsj->ns', …)`, same `hp`, `cp`, `μ`, `ν`) in row chunks of 512. `lp_scp.py` is not edited. Equivalence is asserted: (i) calibration true-state scores vs `lp_scp.true_state_scores` on the full calibration split for every cluster with m ≤ 6, and on a 300-row subset for m=13, tolerance 1e-10; (ii) region masks vs `lp_scp.build_region` on a 300-row test subset for every cluster, exact equality.

### LP-SCP state-space note (pre-registered; do not redesign)
Rung C/D 6-label cluster: 2⁶ = 64 states. Rung E: 2¹³ = 8,192 states. Both must actually run. A degenerate, near-100%-ambiguous Rung E result is an expected, reportable finding. No cluster-size cap, no ladder change.

### Σ fit diagnostics (pre-registered)
Per cluster, record pre-ridge condition number of Σ. `lp_scp.fit_structure` applies a ridge if cond > `COND_LIMIT` (1e10). 052 also recomputes post-ridge condition number. Ridge applied is reported, not itself a trigger.

## Row population and N
- **Test metrics (all methods, all rungs):** 13-label exclusion via `cc.extract_group(test, ALL13)`. Expected N = **18,877** (rows), 5,932 subjects, from 051. 5,517 rows dropped. Identical across methods by construction; asserted.
- **Calibration:** per cluster, `cc.extract_group(conformal, cluster)`. Expected n = 18,754 at every cluster (no uncertain rows).
- **LP-SCP proper-training fit:** per cluster, `cc.extract_group(validation, cluster)`. Validation has 0 rows with any −1 or NaN in the 13 labels (checked 2026-11-01), so expected n = 18,570 at every cluster. Reported separately per cluster and per rung.
- Subject unit: rows (as 051). Surviving subject count reported for the test population.

## Per-label projection rule (all three methods, every m>1 cluster)
Project retained joint states onto each label: **decisive** iff every retained state agrees on that label's value (singleton); **ambiguous** iff retained states disagree ({0,1}); **empty** iff the cluster's retained set is empty. For per-label methods this is the per-label set. For LP-SCP it is the set of values appearing in the retained region. Applied consistently to metrics 3, 4 and 5.

## Metrics (test split, N=18,877, per method per rung)
1. **Joint coverage (primary, locked):** row covered iff true label of every cluster member is in its retained set (ambiguous counts as containing truth). For LP-SCP: true joint state in every cluster region.
2. **Mean joint set size (primary, locked):** joint states retained per row. Per-label methods: product over all 13 per-label set sizes. LP-SCP: product over clusters of region sizes.
3. **Hierarchy-violation rate of fully-decisive output (primary, locked):** among rows with all 13 labels decisive (projection rule), fraction whose 13-label vector violates `CHEXPERT_HIERARCHY` (`cc.hierarchy_admissible` on ALL13).
4. **Decisive rate (secondary, per-label):** over 13×N (label, row) pairs, fraction decisive. Empty sets are reported separately (empty-label-set rate).
5. **Fully-decisive rate (secondary, per-row):** fraction of rows with all 13 labels decisive. Distinct from #4.

Secondary: decisive accuracy on fully-decisive rows; empty-joint-set row count. Rung E: joint-set-size quantiles (0, 1, 5, 25, 50, 75, 95, 99, 100%), count of size 0, 1, 2¹³, and per-label set-size counts, for every method. Šidák − Bonferroni gap per rung for joint coverage and mean joint set size.

## Required caveats (stated in the report, not footnotes)
- **Score asymmetry (LP-SCP):** LP-SCP does not use `s = 1−p if y=1 else p`. It uses its own Mahalanobis-based score (paper eq. 30) fit on validation. Max-rank and Bonferroni/Šidák use the locked score. Stated next to every LP-SCP number.
- **Structural-boost asymmetry (LP-SCP):** with μ=ν=1, LP-SCP receives the Hp/Cp structural penalty (a dependence/frequency-informed admissibility boost). Max-rank and Bonferroni/Šidák receive no equivalent. Stated next to every LP-SCP number. The two asymmetries are not a clean three-way score comparison.
- **§3a population caveat, compounded (LP-SCP):** Σ/Hp/Cp are fit on the all-13-clean validation population, and the calibration quantile on the all-13-clean conformal population. Same compounding pattern as on record for LP-SCP from Exp047. Stated, not re-derived as new.
- **Empirical-only coverage for B/C/D:** the within-cluster 90% target does not imply any bound on the full 13-label joint coverage. Coverage at B–D is empirical.
- **Rung E Fracture-floor effect:** Fracture has 304 calibration positives (above the 150 floor). Its effect on joint coverage and on the set-size distribution is reported with the full distribution, not only the mean.
- **N-shrinkage per rung** (13-label listwise exclusion; distinct from §3a). N = 18,877 at every rung and every method.
- **Monotonicity not assumed.** B→C→D→E trends are reported as observed.
- **052 ≠ 049.** Exp049 is independent (different ladder). 052 is not a confirmation of it.
- **Backbone-correlation caveat:** one shared-backbone classifier inflates cross-label score correlation (EC/CM 0.936→0.997). Any dependence reading partly measures the backbone; the direction is conservative (understates the independent baseline's failure).
- **Šidák − Bonferroni gap** per rung is RESEARCH_DIRECTION.md's evidence that headroom is dependence, not union-bound looseness. Reported explicitly.
- Gate 0.5 flagged labels: none flagged (see Gate 0.5). Pleural Other at 151 noted.
- No numbers from deleted pre-2026-10-28 folders are cited.

## Pre-registered triggers (ESCALATION.md, then consult Tom; do not decide)
- **T-A (LP-SCP execution):** LP-SCP crashes, or exhausts memory, at m=6 or m=13. Infeasible runtime is operationalised as the LP-SCP stage exceeding **2 hours wall-clock** (the spec does not give a figure; Tom to confirm). A completed but degenerate result is NOT a trigger.
- **T-B (Rung A match):** Bonferroni or Šidák Rung A thresholds differ from 051 max-rank Rung A thresholds by max |Δ| > 1e-12, or Rung A joint metrics differ from 051's cited values by > 1e-12.
- **T-C (Σ conditioning):** any cluster where the pre-ridge condition number is non-finite, or where a ridge was applied and the post-ridge condition number still exceeds `COND_LIMIT` (1e10).
- **T-D (N):** any method's test N ≠ 18,877, or any LP-SCP fit N drops more than 0.5% of 18,570 rows (93 rows) relative to the validation split.
- **T-E (implementation equivalence, added for transparency):** the chunked LP-SCP computation disagrees with `lp_scp.py` beyond the tolerances above. Treated as a code-path bug → escalate.

## Open question for Tom (raised pre-run, not decided here)
The non-empty guard (paper eq. 20) is unpinned by the spec. Both variants are computed and reported side by side. Tom to say which, if either, is the headline arm. Not an execution trigger.

## Reproduce
```
cd /Users/alonfines/Desktop/x-ray && .venv_exp048/bin/python experiments/052_cluster_granularity_curve_3method/run.py
```
Expected runtime: LP-SCP at Rung E dominates; estimated tens of minutes on this machine. Wall-clock reported in `metrics.json`.

---

# POST-RUN RECORD (appended 2026-11-01 after the run; pre-registered text above is unchanged)

**Run:** `run.py` exit 0, 24.6 s total (LP-SCP stage 13.3 s), peak RSS 2.4 GB (`/usr/bin/time -l`). Outputs: `metrics.json`, `rungs_summary_3method.csv`.
**Post-hoc diagnostic (not pre-registered):** `diag_rungE_coverage.py` → `diag_rungE_coverage.json`. Run after seeing the Rung E result, to test whether the Rung E shortfall already appears on the calibration split.

## Gates and checks (all pass)
- Gate 0: zero subject and dicom overlap, all three pairs. Counts 18,754 / 24,394 / 18,570 rows; 5,926 / 6,395 / 5,898 subjects.
- Gate 0.5 (reused from 051, sha256 `9c899a7c…`): no label flagged. Lowest: Pleural Other 151.
- Rung A threshold check, Bonferroni and Šidák vs 051 max-rank Rung A: **max |Δ| = 0.0** (13 thresholds each). Rung A joint metrics identical to 051 (diff 0.0; fully-decisive counts match). **T-B not fired.**
- Test N (shared, 13-label exclusion): **18,877 rows, 5,932 subjects**, identical for every method and rung.
- LP-SCP calibration n = 18,754 per cluster; LP-SCP proper-training fit n = **18,570 per cluster** (validation has zero uncertain rows, no drop).
- Σ conditioning (pre-ridge): 1.0 (m=1) to 121.4 (m=13). **No ridge applied to any cluster.** **T-C not fired.**
- LP-SCP chunked computation equals `lp_scp.py` (calibration scores ≤1e-10 on full split for m≤6, first 300 rows for m=13; test region masks exactly equal on 300-row subsets for every cluster). **T-E not fired.**
- LP-SCP execution: no crash, no memory failure, 13.3 s. **T-A not fired.**
- Seed 0. Git HEAD `6a26ac8`.

## Results — 5 metrics × 5 rungs × 3 methods (+ LP-SCP guard variants)
Metrics: joint coverage (target 0.90); mean joint set size; hierarchy-violation rate of fully-decisive output (hv); per-label decisive rate (dec, 13×N denominator); fully-decisive rate (full, per-row). N = 18,877 for all rows.

**Max-rank — CITED from Exp051, not rerun**
| Rung | K | joint cov | mean joint size | hv | dec | full |
|---|---|---|---|---|---|---|
| A | 13 | 0.4530 | 111.87 | 0.0040 | 0.6787 | 0.0800 |
| B | 12 | 0.4561 | 117.11 | 0.0034 | 0.6745 | 0.0785 |
| C | 7 | 0.5556 | 371.71 | 0.0000 | 0.5243 | 0.0200 |
| D | 6 | 0.5866 | 515.48 | 0.0000 | 0.5019 | 0.0199 |
| E | 1 | **0.8934** | 2718.39 | 0.0000 (2 fully-decisive rows) | 0.2853 | 0.0001 |

**Bonferroni (`correction="bonferroni"`, locked score, per-label thresholds at α/m)**
| Rung | joint cov | mean joint size | hv | dec | full |
|---|---|---|---|---|---|
| A | 0.4530 | 111.87 | 0.0040 | 0.6787 | 0.0800 |
| B | 0.4683 | 134.63 | 0.0037 | 0.6569 | 0.0725 |
| C | 0.5837 | 462.27 | 0.0000 | 0.4895 | 0.0131 |
| D | 0.6199 | 651.39 | 0.0000 | 0.4661 | 0.0131 |
| E | **0.9154** | 3020.99 | NaN (0 fully-decisive rows) | 0.2624 | 0.0000 |

**Šidák (`correction="sidak"`)**
| Rung | joint cov | mean joint size | hv | dec | full |
|---|---|---|---|---|---|
| A | 0.4530 | 111.87 | 0.0040 | 0.6787 | 0.0800 |
| B | 0.4681 | 134.13 | 0.0036 | 0.6573 | 0.0726 |
| C | 0.5818 | 456.54 | 0.0000 | 0.4925 | 0.0140 |
| D | 0.6170 | 636.90 | 0.0000 | 0.4698 | 0.0139 |
| E | **0.9111** | 2961.68 | NaN (0 fully-decisive rows) | 0.2669 | 0.0000 |

**LP-SCP, `penalised` arm, μ=ν=1, guard OFF (raw region).**
**ASYMMETRIES, stated for every LP-SCP number: (1) SCORE: LP-SCP uses its own Mahalanobis score (paper eq. 30) fit on validation, not `s=1−p if y=1 else p`. (2) STRUCTURAL BOOST: μ=ν=1 adds Hp+Cp penalties that max-rank and Bonferroni/Šidák do not receive. These are not a clean three-way score comparison.**
| Rung | joint cov | mean joint size | hv | dec | full | rows with empty joint set |
|---|---|---|---|---|---|---|
| A | 0.4456 | 24.93 | 0.0089 | 0.8044 | 0.2726 | 0 |
| B | 0.4512 | 17.01 | 0.0092 | 0.7982 | 0.2697 | 21 |
| C | 0.5516 | 65.40 | 0.0000 | 0.5647 | 0.0403 | 21 |
| D | 0.5835 | 88.05 | 0.0000 | 0.5420 | 0.0403 | 21 |
| E | **0.8943** | 862.42 | NaN (0 fully-decisive rows) | 0.0252 | 0.0000 | 0 |

**LP-SCP, guard ON (paper eq. 20 union of best state). Same asymmetries apply.** Spec did not pin the guard; both variants reported (see open question).
| Rung | joint cov | mean joint size | hv | dec | full | empty rows |
|---|---|---|---|---|---|---|
| A | 0.4456 | 24.93 | 0.0089 | 0.8044 | 0.2726 | 0 |
| B | 0.4514 | 17.02 | 0.0092 | 0.7984 | 0.2697 | 0 |
| C | 0.5517 | 65.44 | 0.0000 | 0.5649 | 0.0403 | 0 |
| D | 0.5837 | 88.12 | 0.0000 | 0.5422 | 0.0403 | 0 |
| E | 0.8943 | 862.42 | NaN | 0.0252 | 0.0000 | 0 |

Secondary: decisive accuracy on fully-decisive rows, rung A/B/C/D/E — max-rank 0.8252/0.8339/0.8783/0.8803/0.50 (n=2); Bonferroni 0.8252/0.8560/0.8992/0.9028/NaN; Šidák 0.8252/0.8562/0.9015/0.9049/NaN; LP-SCP guard-off 0.7699/0.7739/0.8988/0.8988/NaN.

## Šidák − Bonferroni gap, per rung (RESEARCH_DIRECTION evidence)
| Rung | Δ joint cov | Δ mean joint size | Δ dec | Δ full |
|---|---|---|---|---|
| A | 0.0000 | 0.00 | 0.0000 | 0.0000 |
| B | −0.0002 | −0.50 | +0.0005 | +0.0001 |
| C | −0.0019 | −5.73 | +0.0030 | +0.0008 |
| D | −0.0029 | −14.49 | +0.0037 | +0.0008 |
| E | −0.0043 | −59.31 | +0.0046 | 0.0000 |

Reading: Šidák, being less conservative, gives 0.02–0.43 pp lower joint coverage and 0.04–2.0% smaller joint sets than Bonferroni. The gap is small; it is consistent with, but does not by itself establish, the RESEARCH_DIRECTION claim that headroom is dependence rather than union-bound looseness. That inference is Tom's.

## Rung E — full distribution (the Fracture floor and the K=1 result)
Joint set size quantiles (0 / 1% / 5% / 25% / 50% / 75% / 95% / 99% / 100%):
- Max-rank (051): 1 / 4 / 4 / 128 / 1024 / 4096 / 8192 / 8192 / 8192. Count size 1: 2; size 8192: 3652; size 0: 0.
- Bonferroni: 2 / 4 / 8 / 128 / 2048 / 4096 / 8192 / 8192 / 8192. Size 8192: 4281; size 0 and 1: 0.
- Šidák: 2 / 4 / 4 / 128 / 2048 / 4096 / 8192 / 8192 / 8192. Size 8192: 4159; size 0 and 1: 0.
- LP-SCP guard-off and guard-on (identical here): 15 / 21 / 25 / 71 / 410 / 1207 / 3254 / 5413 / 8192. Size 8192: 6; size 0: 0; size 1: 0.

Per-label set-size counts (size 1 / size 2, of 18,877 rows):
- Fracture: max-rank 1391 / 17486; Bonferroni 993 / 17884; Šidák 1065 / 17812; LP-SCP 1167 / 17710.
- Pleural Other: max-rank 8546 / 10331; Bonferroni 7948 / 10929; Šidák 8048 / 10829; LP-SCP **0 / 18877** (fully ambiguous for every row).

Fracture has 304 calibration positives (above the 150 floor). Its near-universal size-2 sets at K=1 are the Fracture-floor effect carried from 051. The LP-SCP Pleural Other fully-ambiguous result is new here.

## LP-SCP per-cluster diagnostics (validation-fit)
- Σ pre-ridge condition numbers: m=1 clusters 1.0; cardio pair 47.0; pleural pair 1.68; lung m=6 59.0; all-13 121.4. No ridge applied.
- Observed label sets in validation (vs 2^m states): cardio 3/4; pleural 4/4; lung 25/64; all-13 **403/8192**.
- Within-cluster joint coverage (guard off), which is the cluster-level target: m=1 0.8956–0.9085; cardio 0.9070; pleural 0.9036; lung 0.8988; all-13 0.8943.

## Rung E shortfall — post-hoc diagnostic (NOT pre-registered; reported because it bears on whether K=1 holds)
Thresholds at Rung E evaluated on the calibration split (in-sample) and on the 13-clean test population:

| Method | conformal in-sample joint cov (n=18,754) | test 13-clean joint cov (n=18,877) | drop |
|---|---|---|---|
| Max-rank (cited thresholds) | 0.9002 | 0.8934 | −0.68 pp |
| Bonferroni | 0.9204 | 0.9154 | −0.50 pp |
| Šidák | 0.9168 | 0.9111 | −0.57 pp |

Max-rank's in-sample conformal coverage is 0.9002, as designed. The test shortfall is therefore not visible in the calibration split. The same direction and similar size (0.5–0.7 pp) appears for all three methods. This points toward a calibration-to-test distribution shift (the 13-clean test subset versus the conformal split; RESEARCH_DIRECTION §3a scope (b)) or subject-level dependence. **I have not established the cause, and I have not tested either explanation.** The LP-SCP Rung E within-cluster coverage (0.8943) is in the same range.

## Required caveats (stated, per spec)
- **Score asymmetry (LP-SCP):** uses its own Mahalanobis score (paper eq. 30) fit on validation; max-rank and Bonferroni/Šidák use the locked score. Stated next to every LP-SCP number above.
- **Structural-boost asymmetry (LP-SCP):** μ=ν=1 gives Hp+Cp; the other two methods get no equivalent. Stated next to every LP-SCP number above.
- **§3a population, compounded (LP-SCP):** Σ/Hp/Cp fit on the all-13-clean validation population and the quantile on the all-13-clean conformal population. Same compounding pattern as Exp047. Not re-derived as new.
- **Empirical-only coverage for B/C/D:** within-cluster 90% gives no bound on the full 13-label joint coverage. Observed: B 0.46–0.47, C 0.55–0.58, D 0.58–0.62 (all methods). Coverage is far below 0.90 at B–D for every method. Those rungs are not joint-coverage results. They are a diagnostic of how much the ladder costs.
- **Rung E:** see the shortfall section above. Coverage is 0.8934 (max-rank), 0.9154 (Bonferroni), 0.9111 (Šidák), 0.8943 (LP-SCP). Two of the four are below 0.90.
- **N-shrinkage per rung** (13-label listwise exclusion, distinct from §3a): N = 18,877 at every rung and method.
- **Monotonicity not assumed.** Observed: max-rank and Bonferroni coverage rise monotonically B→E. LP-SCP coverage at C (0.5516) is slightly below max-rank (0.5556), and the set-size ordering flips relative to max-rank. Trends are reported, not claimed.
- **052 ≠ 049.** Different ladder, independent. Not a confirmation of 049.
- **Backbone-correlation caveat:** the shared backbone inflates cross-label correlation (EC/CM 0.936→0.997). Any dependence reading partly measures the backbone; the direction is conservative.
- **Flag (a), carried verbatim from 051:** Rung A's sanity check passed trivially, because the locked score makes positive and negative thresholds identical (t_j = t_j). It is a code-path check, not an independent correctness proof.
- **Flag (b), carried verbatim from 051:** `experiments/049_granularity_sweep` is a different, independent experiment with a different ladder. 052 is not a confirmation of it.
- Gate 0.5: none flagged. Pleural Other at 151 is one row above the floor; it sits in clusters C/D/E.
- No numbers from deleted pre-2026-10-28 folders are cited.

## Open question for Tom (not decided by Jules)
1. **Rung E coverage below 0.90 at K=1** for max-rank (0.8934) and LP-SCP (0.8943), where the joint guarantee should hold under exchangeability. Diagnostic: the shortfall is not present in-sample on the calibration split. Is the 13-clean test population valid for the K=1 claim, or is this a §3a/shift problem? See ESCALATION.md.
2. **Non-empty guard (LP-SCP):** the spec does not pin it. Both variants are reported above. Which is the headline arm, if either?
3. **LP-SCP wall-clock limit:** the 2-hour threshold in T-A is my operationalisation of "infeasible runtime". Run took 13 s. Confirm or change.
4. **Chunked LP-SCP computation:** the unchunked `lp_scp.state_scores` needs about 16 GB at m=13 on the calibration split. I computed the same formula in 512-row chunks and verified equality. Confirm this implementation is acceptable.

## Reproduce
```
cd /Users/alonfines/Desktop/x-ray && .venv_exp048/bin/python experiments/052_cluster_granularity_curve_3method/run.py
cd /Users/alonfines/Desktop/x-ray && .venv_exp048/bin/python experiments/052_cluster_granularity_curve_3method/diag_rungE_coverage.py
```

## ADDENDUM — decisive accuracy among fully-decisive rows (POST-HOC, requested by Alon, NOT one of Tom's 5 locked metrics)
Definition (051 convention): among rows where all 13 labels are decisive (per-label projection rule), the fraction whose decisive 13-label vector exactly equals the true 13-label vector. Read from the already-computed `metrics.json` / `rungs_summary_3method.csv` (same definition, no recalibration). Table: `decisive_acc_fully_decisive_addendum.csv`.

| Method | A (n) | B (n) | C (n) | D (n) | E (n) |
|---|---|---|---|---|---|
| Max-rank (cited 051) | 0.8252 (1510) | 0.8339 (1481) | 0.8783 (378) | 0.8803 (376) | 0.5000 (2) — n<50, noisy |
| Bonferroni | 0.8252 (1510) | 0.8560 (1368) | 0.8992 (248) | 0.9028 (247) | undefined (n=0) |
| Šidák | 0.8252 (1510) | 0.8562 (1370) | 0.9015 (264) | 0.9049 (263) | undefined (n=0) |
| LP-SCP guard off ‡ | 0.7699 (5146) | 0.7739 (5091) | 0.8988 (761) | 0.8988 (761) | undefined (n=0) |
| LP-SCP guard on ‡ | 0.7699 (5146) | 0.7739 (5091) | 0.8988 (761) | 0.8988 (761) | undefined (n=0) |

‡ LP-SCP carries both asymmetries (own Mahalanobis score; Hp+Cp structural boost). Max-rank, Bonferroni and Šidák use the locked score with no structural boost.

Reading notes: the n shrinks from thousands at Rung A to a few hundred at C/D and to zero or two at E. Rung C/D accuracies are on a few hundred rows, so differences of about 1 pp there are not resolvable. Rung E entries are undefined or noisy and should not be read. Accuracies rise with rung largely because the fully-decisive subset shrinks to rows that are easier to decide, a selection effect, not an improvement in the method. No new calibration, threshold, or ESCALATION.md change was made.
