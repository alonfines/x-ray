# Exp057 — additive factorized-likelihood score on the m_max = 6 DP partition

**Spec:** `experiments/057_additive_dp_m6/SPEC.md` (Tom Hagan, 2026-11-17). Executed by Jules.
**Status at pre-registration:** pre-registered before any conformal or test metric is computed. See §0 for the blocker found during pre-registration.

## 0. One-line restatement and blocking issue (found at pre-registration)

Restatement: on P6 = {LO, LL, Ed, Cons, Pn, Atel} / {EC, CM, PE} / {PTX, SD} / {PO, Fx}, what is the joint-set size of one split-conformal quantile on S(x,y) = −Σ_c log p̂_c(y_c|x) (each p̂_c renormalized over cluster c's admissible states), versus max-rank on the same scorers (MR-P6), the hierarchy-closed P_H with additive scoring (ADD-PH), and 13 singletons (ADD-1), with maxrank_adm, Bonferroni, Šidák and the Exp056 C/D lookups.

**Blocker.** SPEC §1 requires the scorers to be the models trained by `reports/partition_search/find_partitions.fit_cluster` (import, do not copy). Inspection of that function (`find_partitions.py`, lines 55–86) shows:
1. It returns only `(cols, best_CE, best_epoch)`. It does not return the network `net`, so no p̂_c can be computed from it.
2. It does not restore the best-epoch weights. It trains until `bad >= PATIENCE` and returns with the weights at the stop epoch (`best_epoch + 10`, or the 200-epoch cap). So even with the net exposed, the object would not be the model whose holdout CE was reported, which is what SPEC §1(a) requires.
3. It returns no stop-epoch or cap diagnostic, so the §9 cap check cannot be done from its output.

Making these work needs either a change to the shared function (return the best-epoch `net`) or a re-implemented loop (copying, which SPEC forbids). I am not making either change. Consequence: **the additive arms (§5.1–5.4) cannot be run as specified.** Per the stop rules and the no-variants rule, this is escalated (`ESCALATION.md`). The steps that do not depend on scorers (Gate 0, R1, R2) are run below, because they are cheap and their results are needed for the ruling.

## 1. Inputs (frozen, read-only)
- `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_{validation,conformal}.csv` and `…0.809.csv` (test). Loaded with `conformal_common.load_splits(with_validation=True, run_gate0=True)`.
- `reports/partition_search/partitions.json` key `"6"` (P6, read at run time, not retyped).
- `reports/partition_search/cluster_costs.csv` (R1 reference: holdout CE per cluster).
- `experiments/056_three_cluster_union/resplit_metrics_vector.csv` (R2 reference, and Exp056 C/D lookups for §5.7–5.8).
- `experiments/056_three_cluster_union/run.py` (P_H definition; its helpers `all13_clean`, `hier_violations`, `subset`, `refs_eval_13`, `eval_13_region`, `vector_summary`, `cond_cov`, and the `P_H` list are imported unedited).

## 2. Cluster membership (pre-registered)
- **P6** (`partitions.json["6"]`): C1 {LO, LL, Ed, Cons, Pn, Atel}; C2 {EC, CM, PE}; C3 {PTX, SD}; C4 {PO, Fx}. Admissible states 25 × 6 × 4 × 4 = 2400.
- **P_H** (`exp056.P_H`): C1 {EC, CM} (3 admissible); C2 {LO, LL, Ed, Cons, Pn, Atel} (25); C3 {PTX, PE, PO, Fx, SD} (32). Product 2400.
- **Singletons:** 13 labels, 2 states each, 8192 raw states.
- **Distinct scorer fits:** P6 C1 is the same label set as P_H C2 (shared parenchymal cluster, trained once). Distinct multi-label fits = 4 + 3 − 1 = 6, plus 13 singletons = **19 distinct fits**. SPEC §3 R1 says "20 fits" (4+3+13), which double-counts the shared cluster. I will check all 19 distinct fits and flag the count discrepancy for Tom (see ESCALATION.md).

## 3. Hyperparameters, seed, trainer (inherited, unedited)
`find_partitions.fit_cluster`: Linear(m,64)–ReLU–Linear(64,2^m); Adam lr 1e-3, weight decay 1e-4, batch 256, patience 10, max 200 epochs, clip 1e-6; `torch.manual_seed(0)`; holdout = `default_rng(0)` permutation of validation subjects, 20%. Torch threads = 1. Scorer seed 0 only.

## 4. Gate 0 and population (stop on failure)
- Subject and dicom overlap = 0 for all three split pairs (`cc.gate0_check`).
- All-13-clean populations 18,570 / 18,754 / 18,877 rows; pool 37,631 rows / 11,858 subjects (Exp056 `EXPECT`).
- Hierarchy: no true state inadmissible within any P6 or P_H cluster, in any split.
- Expected cross-cluster hierarchy violations for P6 and P_H = 0 (SPEC §4). Reported, not gated, computed from `edges_across`.

## 5. Reproduction checks (stop on failure)
- **R1:** holdout CE from `fit_cluster` for each of the 19 distinct fits matches `cluster_costs.csv` within 1e-6. P6 sum equals 2.532202 (`partitions.json["6"].cost` = 2.5322017520666122).
- **R2:** for each of the 100 resplits, `maxrank_adm` and `bonferroni` (Exp056 `refs_eval_13`, α/13, unmasked) joint coverage, mean joint size and log₂(size)/13 match Exp056 `resplit_metrics_vector.csv` within 1e-9.

## 6. Splits (pre-registered, same as Exp056)
Frozen draw: calibrate on conformal all-13-clean, evaluate on test all-13-clean. Resplits r = 0..99: `default_rng(r)` permutation of sorted pool subjects; first half = calibration, rest = evaluation. α = 0.10. k = ⌈(n+1)(1−α)⌉; k > n stops.

## 7. Arms (pre-registered, not run because of the §0 blocker)
ADD-P6 (primary), MR-P6, ADD-PH, ADD-1, maxrank_adm, bonferroni α/13, sidak, Exp056 C (P_H, L1 seed 0) and D (L1 13 logits, seed 0) as paired lookups. Metrics, paired deltas, efficiency ratios, conditional-coverage diagnostics, gates and readings as in SPEC §6–8, unchanged.

## 8. Metrics and asserts (pre-registered)
Joint coverage (resplit mean, 2.5/97.5 pct), per-cluster coverage, mean joint size (empty = 0; headline), median size, mean log₂(size)/13 over non-empty rows, empty-set rate, fully-decisive rate, decisive accuracy, hierarchy-violation rate on decisive output (expected 0 for P6/P_H). Paired Δ with 2.5/97.5 pct over 100 resplits. CIs are conditional on the fixed pool and fixed scorer (understated).

Stop rules: Gate 0 failure; R1 or R2 mismatch; inadmissible or infinite true-state score; infinite q̂; NaN in any reported metric. Escalation items: any gated arm outside [0.895, 0.905]; additive empty-set rate > 2%; |Δ empty| > 1pp (ADD-P6 vs MR-P6); any scorer at the 200-epoch cap; PO-positive conditional coverage < 0.20 for ADD-P6.

## 9. Caveats (carried into any result)
- Shared backbone inflates cross-label dependence (EC/CM 0.936 → 0.997; Cons/Pn 0.832 → 0.971). Likely overstates joint gains.
- All-13-clean population only (77% of test).
- Partition and scorers selected on the validation holdout. Coverage unaffected; reported CEs optimistic.
- Single scorer seed 0.
- C1 of P6 has m = 6, outside the locked 2–5 range.
- CIs conditional on the fixed pool and fixed scorer; understated.

## 10. Reuse
Imported, not edited: `conformal_common.py`, `conformal_methods/{max_rank,bonferrony}.py`, `data/hierarchy.py` (via conformal_common), `reports/partition_search/find_partitions.py` (`load`, `fit_cluster`, `init`), `experiments/056_three_cluster_union/run.py` (helpers and P_H). `experiments/055` is imported only transitively by 056.

## 11. Provenance
- Git HEAD at run: `b85b288c0ef354c452e352b81af4c729d0747580` (working tree has untracked files: `reports/`, `experiments/056…`, `experiments/057…`, `conformal_eval.py`, `.venv-extraction/`).
- Python: `/usr/bin/python3`. Seed: 0 (scorer, holdout); resplit seeds r = 0..99.
- Reproduce (current step): `/usr/bin/python3 experiments/057_additive_dp_m6/run.py`

## 12. Run status (appended after execution; pre-registration above is unchanged)
- **Executed:** `/usr/bin/python3 experiments/057_additive_dp_m6/run.py` (≈40 min, dominated by R2). Outputs: `metrics.json`, `r1_holdout_ce.csv`, `r2_resplit_identity.csv`.
- **Gate 0: PASS** (overlap 0; populations and pool as pre-registered; hierarchy violations 0; cross-cluster edges P6 0, P_H 0).
- **R1: PASS** on 19 distinct fits (max |Δ CE| 4.89e-7; epochs identical). Count is 19, not the 20 in SPEC §1; see ESCALATION.md.
- **R2: PASS** (600 checks, max |Δ| 4.5e-13).
- **Additive arms (§5.1–5.4), MR-P6, Exp056 lookups, metrics, deltas, diagnostics: NOT RUN.** Blocked: `fit_cluster` does not return the trained network or restore best-epoch weights (§0). See `ESCALATION.md`.
- `run.py` implements only the Gate 0 / R1 / R2 checks; no additive-score code was written.

## 12a. Pre-registered operational definitions for the additive run (written before the additive arms were executed)
Resumed after the coordinator's ruling: `fit_cluster(cols, return_net=True)` returns the best-epoch net (eval mode); scorer seed 0; 19 distinct fits (P6 C1 = P_H C2, trained once and shared); R2 stands and is not re-run.
- **Scores.** For cluster c, p̂_c(·|x) = softmax(net_c(x) + neg_c) over admissible states; NLL_c(s|x) = −log p̂_c(s|x), +∞ for inadmissible s. Input x = logit of the clipped probabilities, as in `fit_cluster`. S(x,y) = Σ_c NLL_c(y_c|x) over the clusters of the partition.
- **Joint set.** Enumerated over the product of admissible cluster states (2400 for P6 and P_H; asserted equal to the full 13-label admissible set). ADD-1 uses all 8192 raw states (primary); the admissible-intersected count is secondary.
- **Threshold.** q̂ = order statistic k = ⌈(n+1)(1−α)⌉ of S on the calibration true states; k > n or non-finite q̂ stops. Unguarded.
- **MR-P6.** `mr.calibrate_max_rank` on the n×4 matrix of true-state NLLs; set = product of per-cluster regions {s admissible: NLL_c(s) ≤ thr_c}.
- **Per-cluster coverage.** Product arms (MR-P6): exact per-cluster coverage. Additive arms: *projection coverage*, i.e. the true cluster-c state appears among the cluster-c states of some joint state in the set. This is my definition (the spec does not pin one for non-product sets) and is flagged in ESCALATION.md.
- **Share of S.** Per cluster, mean of NLL_c(y_c)/S(y) over (a) all frozen-draw calibration rows and (b) the 5% of frozen-draw calibration rows closest to q̂ in |S − q̂|.
- **Lookups.** Exp056 C (P_H, L1 seed 0, unguarded) and D rows from `resplit_metrics_vector.csv`; same metric definitions (log₂ with m = 13).
- **Reference recomputation.** maxrank_adm, bonferroni (α/13, unmasked) and sidak are recomputed per resplit via the Exp056 helpers and cross-checked to the Exp056 vector CSV within 1e-9 (a cross-check, not a re-run of R2).
- **Stops.** Gate 0, R1 (incl. restored-net CE equality), inadmissible/infinite true-state score, infinite q̂, NaN in a reported metric. Escalation items per SPEC §9 (cap, empty rate, |Δempty|, PO-conditional < 0.20, gated band, unpinned definitions).

## 12b. Results of the additive run (supersedes the §12 "not run" status)
Run: `/usr/bin/python3 experiments/057_additive_dp_m6/run.py`, ≈24 min. Outputs: `metrics.json` (full summary), `add_resplit_metrics.csv` (every arm × resplit, incl. frozen −1), `add_p6_share_of_S.csv`, `r1_returnnet_holdout_ce.csv`. Git HEAD b85b288 + uncommitted Exp057 files; `fit_cluster(return_net=True)` from the coordinator's ruling.

**Gate 0: PASS.** Overlap 0; 18,570 / 18,754 / 18,877; pool 37,631 / 11,858; cross-cluster edges P6 = 0, P_H = 0; joint tables P6 = P_H = 2400 = full admissible set (asserted).
**R1: PASS (19 distinct fits, return_net).** max |Δ fit CE vs cluster_costs.csv| 4.9e-7; restored-net holdout CE equals the fit CE exactly (max Δ 0.0); all best epochs match. Stop epochs: max 96; **0 scorers at the 200-epoch cap.** P6 sum = 2.5322017521 (partitions.json 2.5322017521).
**R2 (standing, not re-run) cross-check: PASS.** Recomputed maxrank_adm, bonferroni and sidak agree with Exp056 `resplit_metrics_vector.csv` for all 101 splits (max |Δ| ≤ 4.5e-13).

### Arms (resplit mean [2.5, 97.5 pct over 100 resplits]; frozen draw in parentheses)
| arm | joint cov | mean joint size | median | mean log₂/13 (non-empty) | empty rate | fully decisive |
|---|---|---|---|---|---|---|
| **ADD-P6** | 0.8998 [0.8903, 0.9096] (0.9006) | **27.23 [26.00, 28.47]** (26.95) | 25.5 [24.0, 27.0] | **0.3447 [0.3400, 0.3493]** | 0.0000 | 0.0000 |
| MR-P6 | 0.8996 [0.8908, 0.9079] (0.9016) | 50.42 [47.37, 53.25] (50.69) | 35.8 [32, 36] | 0.3584 [0.3516, 0.3655] | 0.0017 [0, 0.0051] | 0.0444 |
| ADD-PH | 0.8998 [0.8894, 0.9094] (0.9015) | 27.23 [26.08, 28.34] (27.15) | 25.0 [24, 26] | 0.3440 [0.3399, 0.3481] | 0.0000 | 0.0000 |
| ADD-1 (raw 8192) | 0.8996 [0.8918, 0.9091] (0.9013) | 101.99 [97.49, 108.06] (101.61) | 92.2 [88.0, 97.5] | 0.4743 [0.4703, 0.4802] | 0.0000 | 0.0000 |
| maxrank_adm | 0.8997 [0.8888, 0.9128] (0.8934) | 850.0 [820.3, 877.2] (809.9) | 591.4 [576, 600] | 0.5969 [0.5886, 0.6046] | 0.0000 | 0.0003 |
| bonferroni α/13 (unmasked) | 0.9206 [0.9117, 0.9304] (0.9154) | 3171.9 [3074.5, 3282.5] (3021.0) | 2048 | 0.7463 [0.7384, 0.7529] | 0 | 0 |
| sidak (unmasked) | 0.9170 [0.9078, 0.9271] (0.9111) | 3113.9 [3014.6, 3216.2] (2961.7) | 2048 | 0.7423 [0.7340, 0.7491] | 0 | 0 |
| EXP056 C (lookup) | 0.8998 [0.8921, 0.9090] | 49.40 [47.14, 52.20] (47.93) | 40.5 | 0.3608 [0.3538, 0.3696] | 0 | 0.0490 |
| EXP056 D (lookup) | 0.8999 [0.8908, 0.9100] | 30.06 [28.54, 31.73] (29.65) | 28.4 | 0.3517 [0.3461, 0.3574] | 0 | 0.0002 |

Gates (§7, resplit-mean coverage in [0.895, 0.905]): ADD-P6, ADD-PH, ADD-1, MR-P6, maxrank_adm all **in band**. Bonferroni and Šidák cover ≥ 0.90 (0.921, 0.917) and are not compared.

### Paired deltas, ADD-P6 minus comparator (per-resplit, 2.5/97.5 pct over 100 resplits; all comparison_valid = True)
| comparator | Δ mean size [CI] | Δ log₂/13 [CI] |
|---|---|---|
| MR-P6 (combination rule) | −23.19 [−25.93, −20.78] | **−0.0137 [−0.0198, −0.0080]** |
| ADD-PH (partition) | +0.004 [−0.424, +0.448] | +0.0007 [−0.0009, +0.0024] |
| ADD-1 (clustering) | −74.75 [−79.83, −70.63] | **−0.1296 [−0.1332, −0.1260]** |
| maxrank_adm | −822.8 [−850.1, −793.5] | −0.2522 [−0.2616, −0.2430] |
| EXP056 C (different trainer) | −22.17 [−24.54, −20.26] | −0.0161 [−0.0223, −0.0103] |
| EXP056 D (different trainer) | −2.83 [−3.64, −2.02] | −0.0069 [−0.0098, −0.0041] |

Δ empty rate ADD-P6 vs MR-P6: −0.0017 (|Δ| < 1 pp).

### Efficiency ratios (ADD-P6 / comparator)
- vs bonferroni (unmasked): ratio of mean sizes 0.00859 [0.00808, 0.00913]; geometric 2^(13·Δlog₂/13) = 0.0268 [0.0247, 0.0289].
- vs maxrank_adm (masked): ratio of mean sizes 0.0320 [0.0301, 0.0342]; geometric 0.1031 [0.0947, 0.1120].
- **Caveat:** bonferroni and sidak are unmasked product sets and include inadmissible states. Most of the ratio to bonferroni is the admissibility mask, not the additive score. maxrank_adm is the fair comparator for the masked, product-form baseline.

### Rare-positive conditional joint coverage (test half, resplit mean [CI], n = mean rows per resplit)
| subset | ADD-P6 | MR-P6 | maxrank_adm | EXP056 C (descriptive, resplit mean) |
|---|---|---|---|---|
| Pleural Other positive | **0.182 [0.131, 0.244]** (n 152) | 0.006 [0.000, 0.070] | 0.792 [0.743, 0.853] | 0.191 |
| Fracture positive | 0.422 [0.369, 0.467] (n 331) | 0.000 [0.000, 0.000] | 0.900 [0.876, 0.925] | 0.317 |
| Lung Lesion positive | 0.500 [0.453, 0.537] (n 506) | 0.536 [0.498, 0.577] | 0.790 [0.750, 0.827] | 0.490 |
| any positive | 0.833 [0.818, 0.848] (n 11,268) | 0.833 [0.820, 0.846] | 0.859 [0.845, 0.875] | — |

Pre-registered expectation (rare-positive coverage worse for additive than max-rank) holds only for Lung Lesion (0.500 vs 0.536, CIs overlap). For Pleural Other and Fracture, **the additive score is better than MR-P6**, because MR-P6 almost never covers those positives. Both are far below maxrank_adm. These numbers are the price of the score, not the price of the partition (see the S-share below).

### Per-cluster share of S on the frozen calibration set (ADD-P6)
| cluster | mean share, all 18,754 rows | mean share, 5% of rows nearest q̂ (n = 938) |
|---|---|---|
| Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis | 0.442 | 0.457 |
| EC, Cardiomegaly, Pleural Effusion | 0.273 | 0.314 |
| Pneumothorax, Support Devices | 0.222 | 0.196 |
| Pleural Other, Fracture | 0.063 | 0.033 |

Per-cluster projection coverage, ADD-P6 (resplit mean): parenchymal 0.975; EC/CM/PE 0.993; PTX/SD 0.994; PO/Fx 0.992. For MR-P6 each cluster ≈ 0.973–0.974 (product by construction). For ADD-P6 these are projection coverages (see §12a).

### Secondary, ADD-1 (SPEC §4)
Raw 8192-state set: mean size 101.99. After intersection with the full admissible set: mean 50.73. Share of retained states that are inadmissible: 50.3%. Hierarchy-violation rate on decisive output: 0 (no decisive sets).

### Calibration side and the stated bound
Calibration coverage equals k/n exactly for every ADD arm (max |Δ| = 0). Values range 0.900046 to 0.900098. Spec §2 states the calibration coverage lies in [0.90, 0.90 + 1/(n+1)]. That is **not** what k = ⌈(n+1)(1−α)⌉ gives: k/n lies in [0.90, 0.90 + 1.9/n). 84 of the 100 resplits exceed 0.90 + 1/(n+1) for each ADD arm. The code is correct; the bound in SPEC §2 is off. This is flagged for Tom.

### Flags (SPEC §9)
1. **ADD-P6 PO-positive conditional coverage 0.182 < 0.20.** Exp056 C was 0.191 (resplit mean), so ADD-P6 is not better than C on this subset. Flagged as the spec requires.
2. **Calibration bound** (see above), 84/100 resplits per ADD arm above the stated upper bound. Spec wording, not an implementation error.
3. **Scorer cap:** none. Max stop epoch 96.
4. **Empty-set rate:** ADD arms 0.0000 (< 2%). |Δ empty| ADD-P6 vs MR-P6 = 0.0017 (< 1 pp).
5. **Gated bands:** all in band.
6. **Undefined by spec:** per-cluster coverage for non-product sets (projection definition, §12a); the S-share definition (§12a).

### Caveats (carried into every number)
Scorer seed 0 only; the partition and scorers were selected on the validation holdout (reported CEs optimistic, coverage unaffected); C1 of P6 has m = 6 (outside the locked 2–5 range); shared-backbone cross-label correlation inflates joint gains; all-13-clean population (77% of test); CIs are conditional on the fixed pool and fixed scorers (understated); the Exp056 C/D lookups use a different trainer and a non-renormalized score, so they are descriptive.
