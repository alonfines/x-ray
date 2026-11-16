# Exp053 — Equal-coverage cluster comparison — FINAL SPEC

**Spec:** Tom Hagan, 2026-11-10, reviewing `SPEC_DRAFT.md`. **Approved by Alon 2026-11-10**,
including the pooling of conformal + test for resplits (item 5).

**One line:** partition the 13 labels into K clusters, run each method inside each cluster
at one common level β̂ calibrated so the combined 13-label prediction has 90% joint
coverage, then compare partitions and methods on set size at that equal coverage.

## 0. Scope and gates
- Frozen model, CSVs and α = 0.10. Nothing is retrained or re-exported.
- Gate 0: zero subject and zero dicom overlap across all three splits, plus the row and
  subject counts asserted in Exp051.
- Gate 0.5: positive counts per label. Flag any label under 150 (Pleural Other has 151 —
  borderline). No label is moved or dropped.
- Population: all-13-clean rows only. Pooled design = conformal (18,754) ∪ test 13-clean
  (18,877) = 37,631 rows.
- Every headline carries the scope "studies with no uncertain label on any of the 13
  categories, ~77% of test". Never stated bare.
- Validation is used only for fitting partitions and LP-SCP structure. Nothing is tuned on
  conformal or test results.

## 1. Score
True-class nonconformity `s = 1−p if y=1 else p`, every row, all methods.

## 2. Calibration and test statistics (integer-count scale)
- Per-label count: calibration row i: `d_ij = 1 + #{l≠i : s_lj ≥ s_ij}` (leave-one-out).
  Test row or candidate vector: `d_j = 1 + #{l in calibration : s_lj ≥ s_j}`.
- Per-label p-value: `p = d/(n+1)`, calibration and test alike.
- `r_ik = min over j in cluster k of p_ij`.
- Cluster p-value P_k per method:
  - Bonferroni: `min(1, m_k·r_k)`.
  - Šidák: `1−(1−r_k)^m_k`.
  - Max-rank: `P_k = (1 + #{l≠i : r_lk ≤ r_ik})/(n+1)` for calibration rows; same formula
    against all n calibration rows for a test row.
  - LP-SCP: eq. 30 `penalised`, μ = ν = 1; Σ, Hp, Cp fit on validation.
    `P_k = (1 + #{l≠i : S_lk ≥ S_ik})/(n+1)`, S = nonconformity of the true joint state.
  - Non-empty guard OFF for all methods.
- Row statistic: `t_i = min_k P_ik`.
- β̂ rule: β̂ = the largest value v among {t_i} such that
  `#{i : t_i ≤ v} ≤ floor(α(n+1)) − 1`. Ties conservative: a set contains a vector iff its
  P exceeds β̂; a miss is t ≤ β̂.
- Sets: cluster-k set = every candidate label vector v with `P_k(v) > β̂`, P_k(v) computed
  as if v were the truth. For the three p-value methods this is a product set. Combined set
  = product of cluster sets. Same β̂ for all clusters.

## 3. Partitions (all fixed before any run)
- References: K=13 singletons; K=1.
- Clinical: Exp051 rung C (K=7: cardio pair; 6-label lung group; five singletons) and rung
  D (K=6: cardio pair; lung group; {Pleural Effusion, Pleural Other}; three singletons).
- Informed (procedure S-avg): Spearman ρ of nonconformity scores s_j on the validation
  split; distance 1−ρ; average linkage with size cap m ≤ 7 (merge the closest pair whose
  union is ≤ 7 labels); stop at K = 3, 4, 6. Report membership and size profile. State
  that this is not the RESEARCH_DIRECTION GCM procedure and is a marginal score correlation.
- Random null (primary): 200 uniformly random partitions per informed size profile
  (K=3, 4, 6), seeds 1000+idx. LP-SCP uses the first 100 of each list. Secondary: 100
  random partitions each for clinical profiles C and D, p-value methods only.
- Matched null (secondary): from a pool of 5,000 random partitions per profile, keep 200
  with the same number of within-cluster hierarchy edges (out of 7 — see Amendments E1) as the informed
  partition. Seed 1500.
- Anti-clustering: same size profile as each informed cut; minimise total within-cluster
  Σρ by swap hill-climb from 50 random restarts (seed 2000); keep the best.
- Difficulty control (descriptive): sort labels by validation AUC, chunk contiguously into
  the informed profile; one partition per cut.
- For every partition report within-cluster dependence `W(P) = Σ within-cluster ρ` and the
  intra-cluster hierarchy-edge count.

## 4. Methods
Bonferroni, Šidák, max-rank, LP-SCP on every partition, as in item 2.
Context rows (no size comparison): uncalibrated K=1 Bonferroni, Šidák, max-rank at α=0.10;
independent per-label at α (validity gap).

## 5. Evaluation design
- Headline: R = 100 subject-level random 50/50 resplits of the pooled 13-clean set, seeds
  0..99. Recompute the whole calibration (ranks, β̂) on each resplit. Deltas paired within
  a resplit.
- Also report the frozen conformal→test draw as one row, with its percentile inside the
  resplit distribution.
- Pooling conformal + test is a deviation from the frozen split roles (approved by Alon).
  Resplit percentile intervals understate total uncertainty — state this.
- LP-SCP fallback: if projected null runtime exceeds 4 h, run the LP-SCP null on the
  frozen split only and label it so.
- Split-calibration exactness arm: informed K=4 only, all four methods, same resplits.
  Split calibration subjects 50/50 into A (reference ranks) and B (β̂ quantile); test rows
  use A only; same β̂ rule with n = |B|. Robustness only.

## 6. Metrics
- Gate: mean joint coverage over resplits, SD, 2.5/97.5 percentiles. Pass if the resplit
  mean is in [0.895, 0.905]. Outside the band: report the cell, mark its size comparisons
  "coverage-unequal". LP-SCP may over-cover from discrete ties — flag, do not suppress.
  No pass/fail on the single frozen draw.
- Per cell: β̂, the independent-clusters prediction 1−0.9^(1/K), per-cluster achieved
  coverage.
- Efficiency: primary = mean log₂(set size)/13 (empty-product rows excluded from this
  mean). Also arithmetic mean joint size (empty = 0), median, fully-decisive rate, empty
  rate. Flag any comparison where empty rates differ by > 0.5 pp. Ratios vs. K=1 max-rank
  and vs. the null median.
- Secondary: hierarchy-violation rate of fully-decisive output; decisive accuracy.
- Diagnostic: coverage stratified by "any of Pleural Other / Fracture / Lung Lesion
  positive".
- Dose-response: Spearman between W(P) and mean log₂ size over the null partitions, per
  method and profile.

## 7. Hypotheses
- H1: resplit-mean coverage in [0.895, 0.905] for every (method, partition) except LP-SCP
  (over-coverage reported, not suppressed).
- H2 (primary): max-rank, informed cut vs. the 200-partition null, one-sided. Informed
  mean log₂ size (resplit average) below the null's 5th percentile; rank
  `p = (1 + #null ≤ informed)/201`; Holm over the three cuts at 0.05. H2 for other methods
  and the matched null: secondary, unadjusted, labelled descriptive.
- H3: anti-clustering above the null median and above informed (sign check).
- H4 (descriptive): dose-response is negative (more within-cluster dependence → smaller
  sets).

## 8. Identity gates (before anything is interpreted)
- At K=13 and K=1, Bonferroni, Šidák and max-rank must produce the same sets; at most 0.5%
  of rows may differ. Otherwise escalate.
- At K=1, report calibration-side coverage; should equal 0.90 up to 1/n.

## 9. Escalation triggers
Any identity gate fails; resplit-mean coverage outside the band for a p-value method;
resplit SD of K=1 max-rank coverage under 0.25 pp or over 0.8 pp; an informed cut yields a
cluster of more than 7 labels; runtime beyond the 4 h fallback.

## 10. Seeds and reproducibility
Resplits 0..99; null 1000+idx; matched null 1500; anti-clustering 2000. Record git HEAD and
pre-registration time before running.

## Amendments — Tom's rulings on Jules's ESCALATION.md (2026-11-10)
- **E1 Matched-null target.** Directed parent→child edges, 7, as in `CHEXPERT_HIERARCHY`
  (`data/hierarchy.py`). An edge is within-cluster when both endpoints share a cluster;
  Pneumonia–Lung Opacity and Pneumonia–Consolidation count separately. Keep random
  partitions whose within-cluster edge count (0–7) equals the informed partition's. If the
  informed count is 0, the matched null is the 0-edge subset. Report the count.
- **E2 Difficulty control.** Sort labels by validation AUC ascending (hardest first; ties
  to lowest label index). Sort profile sizes descending. Fill contiguously: the largest
  chunk takes the lowest-AUC labels, and so on. E.g. profile {6,4,3}: 6 lowest-AUC labels,
  next 4, highest 3.
- **E3 Clinical random null.** Profile C: seed 3000+idx, idx 0..99. Profile D: seed
  4000+idx, idx 0..99. Own RNG per partition. Unmatched on edges; p-value methods only.
- **E4 Matched null.** Bonferroni, Šidák, max-rank on all 200; LP-SCP on the first 100.
  One sequential RNG stream seeded 1500 generating 5,000 partitions per profile; keep the
  first 200 matches in generation order. Fewer than 200 matches: use all, report the count.
  Fewer than 60: escalate.
- **E5 Anti-clustering.** Minimise W(P) = Σ within-cluster Spearman ρ. Move: swap two
  labels in different clusters (sizes fixed). Best-improvement: evaluate all cross-cluster
  swaps, take the largest decrease, accept only if > 1e-12. Stop when no swap improves.
  50 restarts from random assignments with the target profile, one RNG stream seeded 2000.
  Keep the lowest final W; ties to the lowest restart index.
- **E6 Random partition seeding.** Confirmed: own RNG per partition, seed 1000+idx,
  idx 0..199, reused across K = 3, 4, 6.
- **E7 H2 convention.** Rank-based only: p = (1 + #{null ≤ informed})/(N+1); reject at
  p ≤ 0.05 with Holm over the 3 cuts. N = 200 (p-value methods), 100 (LP-SCP). No
  percentile interpolation in the decision. Descriptively report `np.percentile` (linear)
  5th and 50th of the null.
- **E8 Resplit roles and linkage ties.** Resplit r: seed r, random permutation of pooled
  subjects, first floor(S/2) = half A (calibration), rest = half B (test). Linkage ties:
  the pair whose smaller label index is lowest, then lowest second index. Record any tie.

### Exp052 ruling (citable)
- The 13-clean test subset is a valid, population-matched evaluation population for the
  K=1 joint claim (the conformal split has zero uncertain rows on all 13 labels); the §3a
  shift is excluded as the explanation for this subset. The 0.5–0.7 pp calibration-to-test
  drop is not yet attributed; the leading candidate is calibration-draw variance plus
  subject clustering (~3 rows per subject), which 052's row-level SE ignored. Untested.
- Report the K=1 result as a finding. In Exp053 the frozen-draw row gets no pass/fail:
  state its percentile within the resplit distribution and the resplit SD of K=1 max-rank
  coverage; that measurement settles the question.
- 051 Rung E: no re-read beyond attaching this caveat. LP-SCP guard: OFF is the headline;
  guard-on is not run in Exp053. 052's 2 h limit: moot; Exp053 uses its 4 h fallback.
  Chunked LP-SCP computation: acceptable; `lp_scp.py` stays unedited.

### Amendments 2 — Tom's rulings on ESCALATION_2.md (2026-11-10)
- **Q1 Split-calibration arm.** Inner 50/50 subject split = a second permutation from the
  same generator seeded r, drawn after the outer permutation, applied to calibration half
  A: A1 = reference ranks, A2 = β̂ quantile. Test half B unchanged.
- **Q2 Matched null.** Fresh stream seeded 1500 per profile (K = 3, 4, 6 each).
- **Q3 Anti-clustering.** Fresh stream seeded 2000 per profile.
- **Q4 Uncalibrated context rows.** Frozen conformal→test draw only.
- **Bookkeeping permission (this experiment).** Jules may adopt the obvious reading without
  escalating when the item is a pure RNG/bookkeeping convention (stream or seed layout,
  draw order, tie-break order, file naming, column order, rounding, seed-from-index), every
  plausible reading estimates the same quantity with identical partitions and
  hyperparameters, and the reading is written in the README with one line of reasoning
  before any metric is computed. Still escalate if the choice could change which partitions
  are tested, the target coverage, score or method definition, population, split roles,
  hypothesis or its test, or a gate threshold; if it concerns LP-SCP or max-rank
  definitions, the β̂ rule, or the metrics; if choosing would need a look at results; or if
  a recorded reading would need to change after results exist.

### Amendments 3 — Tom's rulings on ESCALATION_3.md (2026-11-10, before any full-run result)
- **Cause of the identity-gate failure:** a defect in §2 of this spec, not in the code. The
  test-side "+1" was applied to the max-rank and LP-SCP cluster p-value but not to
  Bonferroni/Šidák, and the β̂ rule already reserves one count — so max-rank and LP-SCP
  spent one count twice and sat one integer rank level lower. Conservative, not invalid
  (≤ one count in n+1), but it would bias the Bonferroni/Šidák vs max-rank comparison.
  Not accepted as documented; tolerance not loosened.
- **Definitional amendment (supersedes §2 for max-rank and LP-SCP):** for any cluster
  statistic that is a self-inclusive conformal p-value, P = (number of calibration rows at
  least as nonconforming as the row's own statistic)/(n+1). A calibration row's count
  includes itself. A test row or candidate vector gets the plain calibration count, **no
  extra +1**.
  - Max-rank: count of calibration rows with r_lk ≤ the row's own r (for a calibration row
    this equals the old formula; only the test side changes).
  - LP-SCP: count of calibration rows with S_lk ≥ the row's own S.
  - Bonferroni, Šidák and the β̂ rule are unchanged; covered means t > β̂.
- **Before the full run:** recompute the identity gate on the frozen draw and on resplits
  0–9, at K=13 and K=1. Tolerance stays 0.5% of rows. If it still fails, stop and escalate
  — no further variants. Log the pre-amendment numbers (0.832% frozen; 0.229 / 0.615 /
  1.058% resplit min/median/max) in the README, and describe the one-level diagnostic as
  post-hoc.
- **Gate scope:** the identity gate is a mandatory pass on the frozen draw and resplits
  0–9 before the full run, and is reported for all 100 resplits.
- **Matched null at K=6:** 151 partitions accepted under E4. Bonferroni/Šidák/max-rank use
  all 151 with rank p = (1 + #null ≤ informed)/152; LP-SCP uses the first 100. Report the
  matched counts for K=3 and K=4. Secondary and descriptive; not in the Holm family.
- **`lp_scp.py` changed since Exp052:** reuse allowed after a reproduction check. In a
  scratch directory (never overwriting Exp052 outputs), recompute the Exp052 LP-SCP
  guard-off rows for Rung E (K=1) and Rung C from the current file and compare with
  `rungs_summary_3method.csv` (joint coverage, mean joint set size; K=1 coverage 0.8943).
  Pass at ≤ 1e-6 agreement, then record the current sha256 in the README. Otherwise
  escalate with the differences.
- After the amendment and a passing gate, the full 100-resplit run is authorised.

## Caveats that must appear in the writeup
1. Backbone: shared backbone inflates cross-label score correlation (EC/CM 0.936→0.997);
   likely overstates the efficiency gain. The gain is a property of this model's scores.
2. Guarantee vs. efficiency: coverage of the min-over-clusters statistic is controlled to
   O(1/n) via leave-one-out ranks; exactness requires the split arm. Efficiency is empirical.
3. Population scope: 13-clean rows only; silent about the 23% with an uncertain label.
4. Recalibration changes meaning: "Bonferroni"/"Šidák" here are empirically recalibrated;
   at K=1 and K=13 they coincide with max-rank; differences at K>1 partly reflect the
   cross-cluster weighting convention.
5. Partition provenance: informed = marginal score correlation (S-avg), not GCM, Ding or
   RadGraph.
6. Uncertainty: resplit intervals are conditional on a fixed 37,631-row pool.
7. Rare labels: coverage is not class-conditional; report the stratified diagnostic
   (Pleural Other 151, Fracture 304, Lung Lesion 523 calibration positives).
8. LP-SCP: tie over-coverage may break equal coverage; hierarchy co-location may drive its
   gain — the matched null is the check.
9. Pooling: resplits pool conformal and test; the frozen draw is a reference row only.
