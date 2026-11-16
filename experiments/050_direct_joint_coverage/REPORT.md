# Exp050 — Direct joint coverage, no clustering, no correction

**Run:** 2026-11-01. Not a new design choice — a direct, enumeration-free computation Alon
asked for after reviewing Exp049: AND across 13 independently-calibrated per-label
conformal sets (α=0.1 each, `per_label[none]`, no multiplicity correction), computed
per-row instead of via the 8192-state power-set enumeration `all13` uses.

## Headline numbers

| Metric | Value |
|---|---|
| `direct_joint_coverage` | **0.4530** |
| `fully_decisive_rate` | 0.0800 (8.0% of rows have all 13 labels as singletons) |
| `n_eval` | 18,877 |
| `n_dropped_uncertain` | 5,517 (rows with any of the 13 labels uncertain/-1.0 or missing) |

## This is not a new number

Exp049's own L0 aggregate row (`results_aggregate.csv`, level=0) already reports
`joint_coverage = 0.453` for both `max_rank` and `per_label` — that *is* the AND across
13 singleton (m=1) clusters, which is exactly what this experiment computes directly.
Exp050 isolated and re-verified that quantity; it did not discover a new one. See
`experiments/049_granularity_sweep/REPORT.md` (Key Findings §1) for the cross-link.

## Independence comparison — corrected per Tom Hagan's review

**Do not compare 0.453 to 0.9^13.** First, 0.9^13 = 0.254, not 0.194 (an earlier
in-session arithmetic error — flagging so it isn't repeated). Second, and more
importantly, 0.9^13 is the wrong baseline regardless: the 13 per-label coverages are not
exactly 0.90 (they run 0.896–0.911 per the per-label table below), so the correct
independence null is the **product of the actual empirical per-label coverages**:

```
product of 13 empirical per-label coverages = 0.269
observed direct joint coverage              = 0.453
amplification                                = 0.453 / 0.269 ≈ 1.68×
```

**Attribution (Tom's ruling, verbatim substance):** this amplification is not specific to
the shared backbone. Positive dependence among per-label errors mechanically inflates
joint AND-coverage above the independence product for *any* correlated model — if a hard
image tends to be wrong on many labels at once and an easy image tends to be right on
all of them, AND-ing 13 events concentrates mass on "all-right" rather than spreading
combinatorially. A hypothetical independently-trained-per-label ensemble would likely
still show amplification above its own independence baseline, just a smaller one. The
standing backbone-correlation-inflation caveat (CLAUDE.md, RESEARCH_DIRECTION.md:
Enlarged Cardiomediastinum/Cardiomegaly 0.936→0.997 etc.) is why this effect is *larger*
than an independently-trained ensemble would show — it is not why the effect *exists*.
We cannot decompose how much of the 1.68× is real pathology co-occurrence vs.
backbone-amplified correlation without an independently-trained per-label model, which is
out of scope (no per-label retraining — see CLAUDE.md "Per-Label Pipeline (retired)").

## Per-label sanity cross-check vs. Exp049 L0

| Label | Exp050 | Exp049 L0 (`per_label[none]`) | Diff |
|---|---|---|---|
| Enlarged Cardiomediastinum | 0.9018 | 0.8934 | +0.84% |
| Cardiomegaly | 0.9035 | 0.8937 | +1.09% |
| Lung Opacity | 0.9107 | 0.9051 | +0.63% |
| Lung Lesion | 0.9045 | 0.9001 | +0.49% |
| Edema | 0.9051 | 0.8990 | +0.68% |
| Consolidation | 0.8993 | 0.8861 | +1.49% |
| Pneumonia | 0.8962 | 0.8912 | +0.56% |
| Atelectasis | 0.9091 | 0.9036 | +0.61% |
| Pneumothorax | 0.9044 | 0.9012 | +0.36% |
| Pleural Effusion | 0.9069 | 0.9031 | +0.42% |
| Pleural Other | 0.9075 | 0.9064 | +0.12% |
| Fracture | 0.8961 | 0.8970 | −0.10% |
| Support Devices | 0.9060 | 0.9036 | +0.26% |

Cross-check passes (all diffs <1.5%, consistent direction — Exp050's filtering drops a
row if *any* of the 13 is uncertain, a stricter/cleaner subset than per-label-individual
filtering, so slightly higher per-label coverage here is expected).

## Caveats

1. **This is the uncorrected, unclustered baseline — not a method result.** No
   multiplicity correction, no clustering. It is a diagnostic showing why naive per-label
   conformal (treating all 13 labels independently with no correction) fails to deliver
   nominal 90% joint coverage, motivating the clustering/LP-SCP work — not a competing
   arm to be plotted as if it were one.
2. **Says nothing about marginal validity.** Each label's own 90% marginal guarantee is
   assumed to hold (standard conformal guarantee); this number is purely about what
   AND-ing 13 marginal sets does to joint coverage on this test split. It is an empirical
   observation on one test-split draw, not a proven guarantee — same caveat as every
   other point-estimate result in this project (no bootstrap CIs yet, per Exp046–049
   Caveat 6).
3. **Not a second confirming experiment** — see "This is not a new number" above.

## Files

- `PRE_REGISTRATION.md` — locked method, formula, uncertain-handling rule
- `run.py` — reproducible implementation (reuses Exp049's cached `per_label[none]`
  thresholds, no recalibration)
- `results.json` — n_eval, n_dropped_uncertain, direct_joint_coverage,
  fully_decisive_rate, per_label_coverage dict
- Plot: `experiments/049_granularity_sweep/joint_coverage_vs_ambiguity.png` (this
  experiment's point plotted alongside Exp049's L0–L5 series, marked distinctly)

## Reproduce

```bash
cd /Users/alonfines/Desktop/x-ray
python experiments/050_direct_joint_coverage/run.py
```
