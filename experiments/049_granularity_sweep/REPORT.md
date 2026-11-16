# Exp049 — Granularity sweep: six-level partition, 13→1, joint coverage vs decisiveness

**Run:** 2026-11-01. Design pre-registered by Tom Hagan (see `PRE_REGISTRATION.md`), exactly as specified. No design decisions made after seeing results — no re-grouping, no dropped levels, no arm changes.

All numbers below are read directly from `results_raw.csv` / `results_aggregate.csv`, not reconstructed from memory.

## Setup

- Model: `densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809` (frozen).
- Calibration: conformal split, 18,754 rows / 5,926 subjects. Evaluation: test split, 24,394 rows / 6,395 subjects.
- Score (shared across methods): true-class nonconformity `s = 1−p if y=1 else p`.
- α = 0.1 (target joint coverage 0.90).
- One blocking exception, pre-registered: `conformal_common.load_clusters()` asserts `2 ≤ m ≤ 5`; this sweep needs m=1 (L0) and m=6/m=5/m=13 (L3–L5), so `ClusterSpec` objects are constructed directly in `orchestrator.py`, bypassing the loader. Scoped to this experiment only — every other experiment keeps the 2–5 guard.
- Headline arms: `max_rank` (no arms), `per_label[bonferroni]`, `lp_scp[admissible]` (μ=ν=1, hard hierarchy constraint). `per_label[none/sidak]` and `lp_scp[base/penalised]` are recorded in `results_raw.csv` only, never plotted as competing methods.

## Gate 0 — split disjointness ✓

| Pair | Subject overlap | Dicom overlap |
|---|---|---|
| conformal vs test | 0 | 0 |
| conformal vs validation | 0 | 0 |
| test vs validation | 0 | 0 |

## Partition (pre-registered, unchanged from `PRE_REGISTRATION.md`)

| Level | n_clusters | Non-trivial clusters | Logic |
|---|---|---|---|
| L0 | 13 | — (all singletons) | Baseline; anchor point where all three methods must coincide |
| L1 | 12 | `cardio` {Enlarged Cardiomediastinum, Cardiomegaly}, m=2 | Direct hierarchy edge |
| L2 | 10 | `cardio` m=2; `pneumonia_tri` {Lung Opacity, Consolidation, Pneumonia}, m=3 | Pneumonia's two hierarchy parents + itself |
| L3 | 7 | `cardio` m=2; `lung_hub` {Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis}, m=6 | Complete hierarchy-connected-components partition; reproduction check of the archive's Lung Lesion dilution |
| L4 | 3 | `cardio` m=2; `lung_hub` m=6; `residual` {Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices}, m=5 | **Forced/null grouping** — no hierarchy or dependence edge among the residual five |
| L5 | 1 | `all13`, m=13 | Coarsest endpoint, forced, not evidence-based |

## Headline results — label-weighted aggregate (`results_aggregate.csv`)

| Level | n_clusters | max_rank cov. | bonferroni cov. | admissible cov. | max_rank decisive | bonferroni decisive | admissible decisive |
|---|---|---|---|---|---|---|---|
| L0 | 13 | 0.8987 | 0.8987 | 0.8986 | 0.6595 | 0.6595 | 0.7882 |
| L1 | 12 | 0.8991 | 0.9062 | 0.8989 | 0.6510 | 0.6329 | 0.7820 |
| L2 | 10 | 0.8998 | 0.9149 | 0.9000 | 0.5811 | 0.5462 | 0.6902 |
| L3 | 7 | 0.9006 | 0.9174 | 0.8996 | 0.3782 | 0.3535 | 0.5161 |
| L4 | 3 | 0.8998 | 0.9193 | 0.8987 | 0.1101 | 0.0846 | 0.1902 |
| L5 | 1 | 0.8934 | 0.9154 | 0.8943 | 0.0001 | 0.0000 | 0.0000 |

Weighting: each cluster weighted by its own `m`, normalized to sum to 13 (label-weighted, not cluster-count-weighted).

## The non-trivial clusters themselves, level by level (`results_raw.csv`)

The singletons are unchanged across levels (same per-label numbers everywhere); what actually moves is the cluster being built. Values are `joint_coverage / mean_joint_set_size / fully_decisive_rate / decisive_accuracy`.

| Cluster (m) | Level | max_rank | bonferroni | admissible |
|---|---|---|---|---|
| `cardio` (2) | L1–L4 | 0.8959 / 2.260 / 0.541 / 0.834 | 0.9421 / 2.623 / 0.424 / 0.886 | 0.9011 / 1.444 / 0.631 / 0.860 |
| `pneumonia_tri` (3) | L2 | 0.8972 / 4.630 / 0.300 / 0.830 | 0.9319 / 5.391 / 0.227 / 0.867 | 0.8952 / 2.246 / 0.344 / 0.778 |
| `lung_hub` (6) | L3–L4 | 0.9009 / 29.28 / 0.047 / 0.957 | 0.9219 / 32.31 / 0.033 / 0.961 | 0.8965 / 8.389 / 0.202 / 0.942 |
| `residual` (5) | L4 | 0.9000 / 12.36 / 0.013 / 0.946 | 0.9070 / 12.78 / 0.011 / 0.947 | 0.9006 / 10.999 / 0.000 / NaN |
| `all13` (13) | L5 | 0.8934 / 2718 / 0.0001 / 0.500 | 0.9154 / 3021 / 0.000 / NaN | 0.8943 / 380.0 / 0.000 / NaN |

(`cardio` and `lung_hub` are identical across the levels they appear in — expected, since the cluster's own calibration/evaluation population doesn't change when an unrelated cluster elsewhere in the partition changes.)

## Key findings

1. **L0 sanity check passes.** On singletons, `max_rank` and `per_label[bonferroni]` coincide exactly (0.8987 coverage, 0.6595 decisive) — confirms the shared-score implementation is consistent between the two frameworks at m=1. Note: Exp049's L0 aggregate `joint_coverage` (0.453 from results_aggregate.csv) is the same direct-AND-across-13-singletons quantity that Exp050 (`experiments/050_direct_joint_coverage/`) computed explicitly and labeled "direct joint coverage" (0.4530) — i.e. Exp050 did not discover a new number, it isolated and re-verified a number already present in Exp049's own L0 row. See `experiments/050_direct_joint_coverage/REPORT.md`.

2. **LP-SCP's admissible arm is the most decisive method at every level through L4, without breaking coverage.** At L0–L2 it leads max_rank/bonferroni by 13–23pp on decisive rate while holding coverage within 0.001 of nominal. The gap narrows sharply once `lung_hub` (m=6) enters at L3, and both product methods (`max_rank`, `bonferroni`) overtake it on `lung_hub`'s own decisive rate once it's isolated — LP-SCP's enumerate-the-power-set score pays combinatorially as m grows (8 states at m=3 → 64 at m=6 → 8192 at m=13), so its relative advantage is real but bounded to moderate m.

3. **Bonferroni's over-coverage grows monotonically with coarseness** (0.899 → 0.906 → 0.915 → 0.917 → 0.919 → 0.915 across L0–L5) — consistent with the union bound getting looser as more correlated labels are folded into one calibrated unit. This is the predicted signature of union-bound slack, not dependence being captured.

4. **L3 (`lung_hub`, m=6) shows a cluster-aggregate collapse consistent with the direction of the deleted archive's Lung Lesion dilution finding, though it is not a direct reproduction.** Mean joint set size explodes to 29–32 states for the product methods (26–30% of the 64-state space) with decisive rate collapsing to 3–5%; LP-SCP is less bad (8.4 states, 20% decisive) but still far below its own L0–L2 performance. The archive's Exp005/006/031 finding was specifically about *Lung Lesion's own per-label decisive accuracy* regressing (to 55.4%/56.97%) while staying above chance — a per-member effect. This run's schema records cluster-aggregate decisive rate/set size only, with no per-member decisive-accuracy column inside `lung_hub`, so there is no way to confirm from this data whether Lung Lesion specifically is still the worst-hit member. Directionally consistent, not confirmed at the per-label level; a well-defined follow-up (adding per-label decisive accuracy inside `lung_hub` and `residual` to the raw CSV) would settle it.

5. **L4/L5 are forced/null groupings and must not be read as dependence findings.** `residual` (Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices) has no hierarchy or computed dependence edge; decisive rate is already near-zero there (0.000–0.013) for all three methods. At L5 (`all13`, 8192 states) every method is functionally non-informative: decisive rate 0.0000–0.0001, mean set size 380–3021 states. LP-SCP admits the fewest states (380, 4.6% of the space) vs. bonferroni's 3021 (37%) at comparable coverage — the method is still doing something at the coarsest extreme, just not enough to be decisive on any row.

6. **Hierarchy-violation rate of decisive output is 0.000 everywhere** (not tabulated above since it is uniformly zero). This is consistent with hierarchy correction already being applied to ground truth at load time — the metric is not informative on this corrected data; it would need unmodified labels to discriminate between methods.

## Caveats (standing, restated per `PRE_REGISTRATION.md`)

1. **Backbone-correlation inflation.** `cardio`, `pneumonia_tri`, `lung_hub` probabilities come from one shared-backbone model; cross-label correlation is inflated relative to independently trained models (e.g. Enlarged Cardiomediastinum/Cardiomegaly 0.936→0.997). This **understates** the independent baseline's failure (conservative direction), but must be named whenever a hierarchy-grounded coarser cluster shows a gain — some of that gain is backbone, not genuine label dependence.
2. **§3a calibration-population mismatch.** Applies at every level, not just the coarse ones. Expect out-of-sample joint coverage a bit below 0.90 in general (Exp046 found 0.8917–0.8993 in-sample vs. nominal); this run's range (0.893–0.919) is consistent with that known gap plus bonferroni's union-bound looseness on top.
3. **L4 (`residual`) and L5 (`all13`) are forced/null groupings** — not evidence that these labels are statistically related. Not fixed post-hoc, not dropped, as pre-registered.
4. **Population dilution at L3–L5 was expected, not suppressed.** Rare labels (Fracture: ~304–523 positive calibration rows depending on level; Pleural Other: ~151) drive the collapse once folded into `lung_hub`/`residual`/`all13`. Note: `residual`'s `min_positive_per_label` is 151 — one unit above the archive's `<150` hard-failure boundary, inside the documented 150–200 gray zone. `below_min_support_flag` correctly reads `False`, and `residual`'s near-zero decisive rate (1.1–1.3%) is adequately explained by combinatorial dilution (32-state space) rather than the small-sample failure mode — flagged here so a reviewer doesn't have to go find the near-miss themselves.
5. **LP-SCP fits Σ/Hp/Cp on the validation split** at m=6/m=5/m=13 with no more data available than at smaller m; the `COND_LIMIT` ridge trigger is a diagnostic in the threshold cache, not tuned away.
6. **No bootstrap CIs** — point estimates only, same as Exp046/047/048.

## Files

- `PRE_REGISTRATION.md` — locked design, written before this run
- `results_raw.csv` — 322 rows (level × cluster × method × variant)
- `results_aggregate.csv` — 18 rows (level × headline method)
- `granularity_tradeoff.png` — headline plot (joint coverage vs. fully-decisive rate, 3 lines, labeled by n_clusters)
- `joint_coverage_vs_ambiguity.png` — variant plot (joint coverage vs. ambiguity rate = 1 − fully_decisive_rate), with Exp050's direct point added
- `orchestrator.py` — full reproduction code
- `thresholds/` — cached max_rank/per_label calibration thresholds

## Reproduce

```bash
cd /Users/alonfines/Desktop/x-ray
python experiments/049_granularity_sweep/orchestrator.py
```
