# Exp046 — max-rank vs Bonferroni on pre-registered clusters

**Date:** 2026-10-28. **Status:** complete under the locked per-member rule; no rerun
pending. Design ruled by Tom Hagan; caveat wording per his ruling of the same date.

## Setup

- Model: `densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809` (frozen).
- Calibration: conformal split, 18,754 rows / 5,926 subjects.
  Evaluation: test split, 24,394 rows / 6,395 subjects.
- **Gate 0 passed** — zero shared `subject_id` and zero shared `dicom_id` on every pair.
- Score: true-class nonconformity `s = 1-p if y=1 else p`.
- α = 0.1, target joint coverage 0.90.
- Clusters pre-registered in `configs/clusters/exp046.yaml` before any run, m = 2/3/3/5
  with 1/2/0/4 hard hierarchy constraints.
- Rows with any uncertain (−1) **cluster member** excluded per cluster (locked rule).

Code: `conformal_common.py` (shared core), `max_rank.py` (Timans et al. 2025 Alg. 2),
`bonferrony.py` (per-label split conformal; none/bonferroni/sidak), `compare_methods.py`,
`bootstrap_coverage.py`.

## Primary result

Joint coverage and mean joint set size (= number of retained joint states; the product
conversion that makes the per-label and joint families comparable).

| cluster | m | n test | indep@α | Bonferroni | Šidák | max-rank |
|---|---|---|---|---|---|---|
| cardio_pair | 2 | 23,128 | 0.8839 / 2.176 | 0.9421 / 2.623 | 0.9411 / 2.614 | 0.8959 / 2.260 |
| lung_core | 3 | 22,236 | 0.7631 / 2.413 | 0.9083 / 4.245 | 0.9057 / 4.200 | 0.8922 / 3.977 |
| pleural_free | 3 | 23,630 | 0.7528 / 2.126 | 0.9036 / 3.802 | 0.9009 / 3.757 | 0.8982 / 3.726 |
| lung_wide | 5 | 20,318 | 0.7108 / 5.874 | 0.9242 / 16.563 | 0.9219 / 16.396 | 0.8993 / 14.839 |

Independent-at-α has **no joint guarantee by construction** and is the motivating failure,
not a competitor. It collapses to 0.711 at m = 5.

Paired set-size delta, max-rank − Bonferroni, 95% percentile interval over 50
subject-level calibration resamples:

| cluster | m | mean delta | 95% interval | max-rank smaller in |
|---|---|---|---|---|
| cardio_pair | 2 | −0.367 | [−0.409, −0.324] | 100% of resamples |
| lung_core | 3 | −0.273 | [−0.310, −0.228] | 100% |
| pleural_free | 3 | −0.074 | [−0.096, −0.057] | 100% |
| lung_wide | 5 | −1.709 | [−1.942, −1.489] | 100% |

## What this supports

Reproduces the shape recorded in `RESEARCH_DIRECTION.md` §4: the independent baseline has
no joint validity, Bonferroni over-covers, Šidák buys ~0.2pp over Bonferroni, and max-rank
retains smaller joint-state sets at near-nominal coverage. Šidák's near-nil gain is the
evidence that the available headroom is **label dependence**, not union-bound looseness.

The supportable claim is **relative**: max-rank is 10–14% smaller than Bonferroni where
Bonferroni over-covers, and essentially tied on `pleural_free`, the no-constraint cluster.

## What this does not support

- **Absolute informativeness is low.** On `lung_wide`, max-rank retains 14.8 of 32 joint
  states (46% of the state space). Per-row: 19.1% fully decisive, 27.9% entirely
  undecided, mean 3.06 of 5 labels abstained. Both methods are weak in absolute terms;
  the comparison is between two weak methods, not evidence either is good.
- **Decisive accuracy (0.939 on `lung_wide`) applies only to the 19% of rows that decided
  anything.** It is not a general accuracy claim.
- **Hierarchy-violation rate of decisive output is 0.000 for every method** — and this is
  now **resolved rather than open** (Tom Hagan, 2026-10-28, after Exp047 showed the same
  0.000 even for a method with no hard constraint): `data/hierarchy.py` applies hierarchy
  correction at load time, propagating child-positive → parent-positive in the ground
  truth *before* any conformal method runs, so violating combinations are already
  rare-to-absent in the labels every method is scored against. The metric would become
  informative only on a model or dataset without that correction. It is not evidence about
  any calibration method, and should not be presented as one.

## Exchangeability caveat — required alongside any coverage number above

Max-rank is systematically centred **below** nominal: bootstrap means 0.8951 / 0.8917 /
0.8991 / 0.8990 with 78% / 98% / 60% / 60% of resamples under 0.90. Bonferroni is under
target in 0% of resamples (14% on `pleural_free`).

This is **not an implementation defect**. Max-rank's coverage on its own calibration set,
under the thresholds it produces, is 0.9001 / 0.9002 / 0.9001 / 0.9002 — exactly
`ceil((n+1)(1−α))/n`. The machinery hits nominal in-sample to four decimals.

The cause is the calibration/test population mismatch documented in
`RESEARCH_DIRECTION.md` §3a: calibration is all-13-clean (uncertain rows deleted at
split-construction, `data/mimic_split.py:162-164`), evaluation is per-member-clean.
Bonferroni over-covers enough to absorb the same shift, which is why it looked immune —
the bias is masked, not absent.

**Secondary robustness check** (not a replacement, and scope must be stated): restricting
test to studies with no uncertain label on any of the 13 categories (n = 18,877, 77% of
test), max-rank coverage is 0.9019 / 0.9009 / 0.8979 / 0.9010 — at nominal on three of
four. Under the narrower, cleaner population the exchangeability concern disappears and
the result reads as predicted.

## Open

- 200+-resample bootstrap held pending Alon's ruling on the re-inference escalation
  (§3a). If approved, it runs on the corrected calibration, not this one.
- Structured Layer 1 on `lung_wide` (λ = 1.0 frozen) — the next experiment, and the only
  place the hierarchy-violation metric should move. Needs the conditional-dependence graph
  for edge weights, which has the same validation-split defect.
- Max-rank's two-sided behaviour under a corrected calibration population is untested.

## Files

`method_comparison.{csv,md}`, `max_rank_results.csv`, `per_label_results.csv`,
`bootstrap_{raw,summary,size_deltas}.csv`, `thresholds/*.json` (each stamped with the key
it was computed under — method, labels, α, variant, checkpoint stem, calibration
fingerprint).
