# SPEC: Exp057 — additive factorized-likelihood score on the m_max = 6 DP partition (Tom Hagan, 2026-11-17)

Folder: `experiments/057_additive_dp_m6/`. Write the pre-registration into `README.md` before computing any metric on the conformal or test splits.

Reuse these files without editing them: `conformal_common.py`, `conformal_methods/{max_rank,bonferrony}.py`, `data/hierarchy.py`, and `fit_cluster` from `reports/partition_search/find_partitions.py` (import it; do not copy it).

Anything this spec does not mention follows Exp056 / Exp055 exactly: the all-13-clean population, Gate 0, 100 subject-level 50/50 resplits of the 37,631-row pool (same seeds), the frozen-draw reference row, k = ⌈(n+1)(1−α)⌉, the CI convention and the caveat list. α = 0.10 only. Scorer seed 0 only.

## 0. Question
On the DP partition P6, how big is the joint set from one split-conformal quantile on the additive score S(x,y) = −Σ_c log p̂_c(y_c|x)? Each p̂_c is renormalized over cluster c's admissible states. Is it smaller than the alternatives that change exactly one thing: the combination rule, the partition, or no clustering at all?

P6 = {LO, LL, Ed, Cons, Pn, Atel} / {EC, CM, PE} / {PTX, SD} / {PO, Fx}, read from `partitions.json["6"]`. Do not retype it.

## 1. Scorer training data: reuse the inner-train fit, with the holdout used for early stopping
Train each cluster scorer with `find_partitions.fit_cluster`: the same 80/20 inner subject split (`default_rng(0)`), early stopping on the holdout, and the same hyperparameters. Do not refit on all of validation.

Reasons:
- (a) This is the exact model whose holdout CE selected the partition. That gives a hard reproduction check (§3); a refit gives none.
- (b) A full-validation refit has no early-stopping signal. It would need an epoch count chosen from the holdout run, which is an extra choice, and it adds only about 3.7k rows.
- (c) It matches how Exp056 arms C and D were trained, so the lookups in §5 compare like with like.

Validity is not affected either way. The partition and the scorers depend only on validation, which is independent of the conformal/test pool. Reusing the holdout for both selection and early stopping biases only the reported CE, not coverage.

Train these clusters, all with the same function:
- the 4 clusters of P6;
- the 3 clusters of P_H, training the shared parenchymal cluster once;
- the 13 singletons.

## 2. Calibration and test protocol
- **Gate 0 (stop on failure):**
  - Subject and dicom overlap is 0 for all three split pairs.
  - Population is 18,570 / 18,754 / 18,877 rows, and the pool is 37,631 rows / 11,858 subjects, as in Exp056.
  - No true state is inadmissible within any P6 or P_H cluster, in any split.
- **Frozen draw (reference row):** calibrate on all-13-clean conformal, evaluate on all-13-clean test.
- **100 resplits (source of the CIs):** calibrate on half A, evaluate on half B, using Exp056's resplit seeds.
- **Additive arms:**
  - Compute S on the true y for the calibration rows.
  - q̂ is the k-th smallest score, k = ⌈(n+1)(1−α)⌉. If k > n, stop.
  - The set is {y admissible-within-clusters : S(x,y) ≤ q̂}.
  - Unguarded: an empty set stays empty and counts as size 0.
  - Record calibration-side coverage as a sanity check. By construction it lies in [0.90, 0.90 + 1/(n+1)].
- **Population caveat:** every table carries the scope tag "all-13-clean, 77% of test". The §3a caveat stands.

## 3. Reproduction and identity checks (stop on failure)
- **R1:** every retrained cluster's holdout CE matches `cluster_costs.csv` within 1e-6, for all 20 fits in §1. The P6 sum must equal 2.532202. If anything fails, stop and escalate. Do not debug by changing the trainer.
- **R2:** recompute `maxrank_adm` and `bonferroni` on every resplit. Their per-resplit coverage and log₂/13 must match Exp056 `resplit_metrics_vector.csv` within 1e-9. This proves the resplits are identical, which is what makes the §5 lookups pairable.

## 4. Exact set enumeration
P6 contains every hierarchy edge inside some cluster: EC←CM is in cluster 2, and all lung edges are in cluster 1. So the product of within-cluster admissible states is exactly the full admissible set: 25 × 6 × 4 × 4 = 2400 states. P_H gives 3 × 25 × 32 = 2400, the same set. Cross-cluster hierarchy violations are therefore impossible for P6 and P_H, and the no-automatic-+∞ rule does not come into play. State this in the README and report 0.

- Precompute the 2400 × 13 state table once.
- Per row, build S as the sum of the 4 gathered per-cluster NLL vectors, then count S ≤ q̂. Use float64 and batches. That is about 19k × 2400 per evaluation half, which is trivial.
- Singletons: enumerate all 8192 states.
  - Primary size is the raw count, with no mask.
  - Secondary, clearly labelled: the count after intersecting with the full admissible set, and the share of retained states that are inadmissible.

## 5. Arms
New, all on the same scorer family:
1. **ADD-P6 (primary).**
2. **MR-P6:** max-rank across the 4 clusters (Timans Alg. 2). Build the n_A × 4 matrix of true-state renormalized NLLs, then `max_rank.column_ranks` → row max → k* → per-cluster thresholds. The set is the product of the 4 regions. This isolates the combination rule: same scorers and same partition as ADD-P6.
3. **ADD-PH:** additive on P_H. This isolates the partition: does the DP search buy anything over the hierarchy-closed design?
4. **ADD-1:** additive on the 13 singletons. This is the unclustered baseline in the same score family: what clustering buys at fixed score and calibration.

Recomputed (needed for R2, cheap):
5. **maxrank_adm:** the primary classical unclustered comparator.
6. **bonferroni α/13** and **sidak**, per label: the locked baseline for the efficiency ratio.

Lookups, paired by resplit id once R2 passes (no retraining):
7. **Exp056 arm C:** max-rank on P_H, mlp_score L1, seed 0. Mean size 49.4.
8. **Exp056 arm D:** L1 on all 13 logits. Mean size 30.1.

Label both lookups as different-trainer references: they use the non-renormalized score and the Lightning port.

Not included: other m_max, random partitions, other seeds, a guarded variant, other α.

## 6. Metrics (per arm, frozen row plus resplit summary)
- **Validity:**
  - joint 13-label coverage: resplit mean, plus the 2.5 and 97.5 percentiles;
  - per-cluster coverage.
- **Efficiency:**
  - mean joint size, with empty sets counted as 0 (the headline);
  - median size;
  - mean log₂(size)/13 over non-empty rows;
  - empty-set rate;
  - fully-decisive rate and decisive accuracy.
- **Paired deltas:** compute Δ mean size and Δ log₂/13 within each resplit, then take the 2.5 and 97.5 percentiles over the 100 resplits. Pairs: ADD-P6 minus each of MR-P6, ADD-PH, ADD-1, maxrank_adm, Exp056-C and Exp056-D.
- **Efficiency ratios:** ADD-P6 vs bonferroni and vs maxrank_adm, as a ratio of mean sizes and as a geometric-mean ratio, 2^(13·Δlog₂/13).
- **Diagnostics (not gated, but required):**
  - Conditional joint coverage for ADD-P6, MR-P6 and maxrank_adm, on rows positive for PO, Fx and LL, and on rows with ≥1 positive. Report n for each.
    - Expect rare-positive coverage to be worse than under max-rank, because the additive score lets the common clusters absorb the error budget. That is the price of this score, and it must sit next to the size headline.
  - Per-cluster share of S at q̂: which cluster's NLL dominates the score.
- CIs come from the resplits, not a separate bootstrap. They capture calibration and test randomness together, which is the Exp055 convention.

## 7. Gates
- ADD-P6, ADD-PH, ADD-1, MR-P6 and maxrank_adm: resplit-mean joint coverage must be in [0.895, 0.905].
- Bonferroni / Šidák: coverage ≥ 0.90 is valid. Their coverage differs from the other arms, so report their sizes but do not compare them.
- A delta is `comparison_valid` only if both arms pass.

## 8. Pre-registered readings
1. **Headline:** ADD-P6 mean size [CI], median, log₂/13, empty rate, and its ratio to bonferroni and to maxrank_adm.
2. **Combination rule (ADD-P6 minus MR-P6, Δ log₂/13):**
   - CI entirely < 0: "additive beats max-rank on the same partition".
   - CI entirely > 0: the reverse.
   - Otherwise: "no measurable difference".

   Theory expects additive ≤ max-rank. The additive set is the highest-density region of the plug-in joint, which is the smallest set with that plug-in mass.
3. **Partition search (ADD-P6 minus ADD-PH):**
   - CI entirely < 0: "the DP partition helps".
   - With CEs of 2.532 vs 2.540, "no measurable gain" is the likely outcome. If so, state it plainly: the hierarchy partition is already enough at m ≤ 6.
4. **Value of clustering (ADD-P6 minus ADD-1):** CI entirely < 0 means "joint scoring within clusters shrinks sets at equal coverage". Credit this to clustering, since the score family is fixed.
5. **Versus Exp056 C and D:** descriptive, with paired CIs. The trainer differs, so this is not a headline.
6. **Seed:** seed 0 only. Exp056's seed spread was 1.1% of its delta. Carry this as a caveat, not a claim.

## 9. Stop and escalation (write `ESCALATION.md`, do not iterate)
- **Stop on:**
  - Gate 0 failure;
  - an R1 or R2 mismatch;
  - an inadmissible or infinite true-state score in calibration;
  - an infinite q̂;
  - NaN in any reported metric.
- **Record, and escalate at the end:**
  - any gated arm outside its band;
  - an additive arm with empty-set rate > 2%;
  - |Δ empty| > 1pp between ADD-P6 and MR-P6;
  - any scorer reaching the 200-epoch cap;
  - conditional coverage below 0.20 on PO-positive rows for ADD-P6 (Exp056 C was 0.19, so flag it if ADD-P6 is no better);
  - anything ambiguous.
- Do not try other variants: no guard, no other α, no other m_max, no new seeds.

## 10. Caveats (state them in the results)
- The Exp055/056 caveat list.
- The partition and the scorers were selected on the validation holdout. Coverage is unaffected; the reported CEs are optimistic.
- Single scorer seed.
- C1 has m = 6, outside the locked 2–5 range.
- The unified backbone inflates cross-label dependence (standing caveat).

Note: the mlp_score Lightning trainer is rejected for this experiment. It does not renormalize over admissible states. For the additive score that matters: the missing mass is a per-x constant in each cluster, which changes the ranking across x. It also could not reproduce the CEs that selected the partition.
