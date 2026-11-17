# Exp057 ESCALATION — additive arms cannot be run as specified (scorer not available)

**Status:** run stopped before any additive arm. No ADD-P6, MR-P6, ADD-PH, ADD-1 or Exp056-lookup metric was computed.
Gate 0, R1 and R2 passed. Stop code: `STOPPED_BEFORE_ADDITIVE_ARMS` (`metrics.json`).

## The problem
SPEC §1 requires the cluster scorers to be the models trained by `reports/partition_search/find_partitions.fit_cluster`,
imported and not copied, and says these are "the exact model whose holdout CE selected the partition". That function
(`find_partitions.py` lines 55–86) does not provide them:

1. **No network is returned.** It returns `(cols, best_CE, best_epoch)`. The trained `net` is local and cannot be used to compute p̂_c.
2. **The weights are not the best-epoch weights.** The loop trains until `bad >= PATIENCE` (10) and returns without restoring
   the best checkpoint. Any object obtained from it would be the stop-epoch model (`best_epoch + 10`, or the 200-epoch cap),
   not the model whose CE was reported. The reported CE (`best_epoch`) and the scoring model would be different models.
3. **No stop epoch or cap flag is returned**, so the §9 "any scorer at the 200-epoch cap" check cannot be done from its output.
   (From R1, all 19 best epochs are < 200; the stop epoch is not recorded.)

I have not worked around this. Options that would work each need a decision outside Jules:
- (a) Amend the shared function to return the best-epoch `net` (restore best weights before return). That edits a
  shared module, so it needs Alon/Winston. Re-running R1 afterwards would confirm the CE is unchanged, since the training
  trajectory is identical; only the returned object changes.
- (b) Re-implement the training loop inside this experiment folder. SPEC forbids copying `fit_cluster`, so this needs your
  explicit sign-off as an amendment.
- (c) Use a different trainer. The Lightning `mlp_score` port is rejected by SPEC §Note, and Exp056 showed it does not reproduce these CEs.

## What did run (validity checks only)
**Gate 0: PASS.** Overlap 0 for all three pairs (subject and dicom). Populations 18,570 / 18,754 / 18,877;
pool 37,631 rows / 11,858 subjects. No true state inadmissible within any P6 or P_H cluster in any split.
Cross-cluster hierarchy edges: P6 = 0, P_H = 0 (no automatic +∞ mask needed).

**R1: PASS on the 19 distinct fits.** Max |Δ holdout CE| = 4.89e-7 (`r1_holdout_ce.csv`). This is within 1e-6, but close to
the 5e-7 rounding of `cluster_costs.csv` (6 decimals), so the tolerance is meaningful only at the rounding level. Epochs match
exactly for all 19 fits. P6 sum recomputed = 2.5322017521, expected 2.532202.

**R2: PASS.** 600 checks (100 resplits × {maxrank_adm, bonferroni} × {joint coverage, mean joint size, log₂/13}).
Max |Δ| = 4.5e-13 against Exp056 `resplit_metrics_vector.csv`. The resplits are identical, so §5 lookups would be paired correctly.

## Count discrepancy (for Tom)
SPEC §1/R1 says "20 fits" (4 + 3 + 13). P6 C1 and P_H C2 are the same label set (the shared parenchymal cluster), so
the distinct count is 4 + 3 − 1 + 13 = **19**. I checked the 19 distinct fits. Please confirm 19 is what you intended.

## Questions for Tom
1. Which of (a), (b), (c) above should I pursue? My recommendation is (a), with Alon or Winston making the one-line change
   to `fit_cluster` and R1 re-run after it.
2. Confirm 19 distinct fits in place of "20".
3. The §9 cap check: do you want the stop epoch added to the returned tuple as part of (a)?

Nothing has been iterated. No variant was tried. The additive score, the arms and the conditional-coverage diagnostics have not been computed.

---

## UPDATE (after the coordinator's ruling: return_net, 19 fits confirmed, additive arms run)
The scorer-access blocker above is resolved upstream. The additive arms were run per SPEC §4–§8 (README §12b). Items for Tom:

1. **SPEC §9 flag, ADD-P6 PO-positive conditional coverage = 0.182 [0.131, 0.244]** (n ≈ 152 per resplit). Below 0.20, and no better than Exp056 C (0.191 resplit mean). Recorded as the spec requires.
2. **Pre-registered expectation not borne out on rare positives.** Tom expected additive to be worse than max-rank on PO/Fx/LL positives. It is better on Pleural Other (0.182 vs 0.006) and Fracture (0.422 vs 0.000), and slightly worse on Lung Lesion (0.500 vs 0.536, CIs overlap). MR-P6 almost never covers PO or Fx positives, so the comparison is not a clean price-of-score reading. Please review.
3. **SPEC §2 calibration bound is inconsistent with k = ⌈(n+1)(1−α)⌉.** Calibration coverage equals k/n exactly (max |Δ| = 0). k/n lies in [0.90, 0.90 + 1.9/n), but SPEC states [0.90, 0.90 + 1/(n+1)]. 84/100 resplits per ADD arm exceed the stated upper bound. The implementation is correct; the bound needs correcting.
4. **Bonferroni and Šidák are unmasked** (SPEC §5.6 "α/13 per label"). Their sizes include inadmissible states (about 3,000 of the 8,192-state product). The ratio ADD-P6 / bonferroni (0.0086 mean-size) is therefore mostly the admissibility mask. The masked comparator is maxrank_adm (ratio 0.032). Please confirm which baseline is locked for the efficiency headline.
5. **Per-cluster coverage for non-product sets is my definition** (projection coverage; README §12a). SPEC does not pin it. Confirm, or specify.
6. **S-share definition is mine** (mean share over all calibration rows and over the 5% nearest q̂; README §12a). Confirm.
7. **Seed 0 only; the 19 fits and the partition were selected on the validation holdout** (caveats as in SPEC §10). No variants were tried.

No stop rule fired. R1 and R2 stand. The R2 cross-check of the recomputed references passes at ≤ 4.5e-13. No scorer reached the 200-epoch cap (max stop epoch 96). Restored-net holdout CE equals the fit CE exactly for all 19 fits.
