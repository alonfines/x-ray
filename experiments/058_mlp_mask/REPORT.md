# Exp058 — MLP scorer with vs without the hierarchy mask

**Question.** Does the MLP joint-state scorer need the hierarchy mask (−∞ on inadmissible states
inside the softmax), or does it learn the hierarchy from data? Without the mask, how much of each
prediction set violates the hierarchy?

**Setup.** DP m_max=6 partition (`classifier/reports/partition_search/partitions.json`):
{Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis} · {Enl. Cardiomediastinum,
Cardiomegaly, Pleural Effusion} · {Pneumothorax, Support Devices} · {Pleural Other, Fracture}.
It is hierarchy-closed: clusters 0 and 1 hold every hierarchy edge (39 and 2 impossible states of 64 and 8).
Arms: `MLP()` (no mask) vs `MLP(renormalize_admissible=True)` (mask), seed 0 in both, fit on validation.
Rules max-rank and Bonferroni, α = 0.1. Sets are **not** filtered to admissible vectors (`admissible=False`).
Frozen draw + 100 subject-level 50/50 resplits of conformal ∪ test. Population: all-13-clean rows (~77% of test).
New metric `share_impossible`: per non-empty set, the fraction of its label vectors that violate the
hierarchy, averaged over test rows. Code: `run.py` (13 s on the Mac).

## Results (mean over 100 resplits, 2.5–97.5% percentile band)

| arm | rule | joint coverage | mean size | median size | share impossible | decisive rate |
|---|---|---|---|---|---|---|
| no mask | max-rank | 0.8996 [0.8905, 0.9082] | 50.44 [47.44, 53.24] | 35.9 | **0** | 0.0399 |
| mask | max-rank | 0.8996 [0.8908, 0.9079] | 50.42 [47.37, 53.25] | 35.8 | 0 | 0.0444 |
| no mask | Bonferroni | 0.9046 [0.8956, 0.9124] | 53.91 [50.54, 56.87] | 37.0 | **0** | 0.0357 |
| mask | Bonferroni | 0.9046 [0.8957, 0.9128] | 53.94 [50.22, 57.33] | 37.0 | 0 | 0.0403 |

Paired (no mask − mask, within resplit): coverage +0.0000 [−0.0010, 0.0012]; mean size +0.01
[−0.42, 0.53] (max-rank), −0.04 [−0.65, 0.45] (Bonferroni) — no difference. Decisive rate −0.0045
[−0.0060, −0.0032] (max-rank): the mask makes slightly more sets a single vector (~0.45 points).
log2(size)/M is higher by 0.0015 without the mask (CI excludes 0), negligible in size.

**Without the mask, no prediction set contains a single impossible label vector, on any resplit.**
This is a hard margin, not luck: the smallest score any impossible state gets on any pooled row is
7.46 (lung cluster) / 6.45 (cardiac cluster), while the largest threshold on any resplit is 4.73 / 3.79.

## Scorer diagnostics (pooled conformal ∪ test rows)

| cluster | arm | holdout CE | test CE | mass on impossible states | top-1 impossible | best / stop epoch |
|---|---|---|---|---|---|---|
| lung (6) | no mask | 1.1388 | 1.1717 | 0.0010 | 0 | 48 / 58 |
| lung (6) | mask | 1.1375 | 1.1699 | 0 | 0 | 48 / 58 |
| cardiac (3) | no mask | 0.7338 | 0.7651 | 0.0003 | 0 | 55 / 65 |
| cardiac (3) | mask | 0.7331 | 0.7642 | 0 | 0 | 55 / 65 |

The unmasked MLP puts 0.1% of its probability on impossible states and never ranks one first; its
cross-entropy is within 0.002 nats of the masked one.

## Reading

The MLP learns the hierarchy by itself: it is free to put mass on impossible states and doesn't, so
the mask buys nothing in coverage or size. **Why it learns it:** the training labels contain zero
hierarchy-violating vectors (hierarchy correction is applied to all splits; Gate 0 checks this), so
"impossible" states are simply never-observed states and cross-entropy drives them down. This
holds for these corrected labels; it says nothing about data where violations occur. A loss that
explicitly penalises impossible states (the planned weighted loss) can therefore only matter for
mass the plain MLP already leaves near zero, on this data.

Caveats: one MLP seed per arm; CIs are over resplits only, conditional on the fixed pool and the
fixed trained scorers, so they understate total uncertainty; population is all-13-clean rows.

**Outcome (2026-11-17).** On this result Alon removed the hierarchy mask from the MLP and the
admissible-set filter from `conformal/` (which also served LP-SCP); the hierarchy is now only measured.
