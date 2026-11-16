# Exp047 — LP-SCP with structural penalties (Katsios & Papadopoulos 2025)

**Date:** 2026-10-28. Design pre-registered by Tom Hagan before the run; µ = ν = 1 fixed
a priori, not selected on the outcome. Same four clusters as Exp046, same α = 0.1, same
test populations, same locked metrics.

**Framework note.** LP-SCP is a *different framework* from `max_rank.py` / `bonferrony.py`,
not another variant of the shared score. Those build m per-label sets and take the
Cartesian product; LP-SCP scores every joint state in the power-set and returns a region
over states directly. Both produce joint label-vector regions, so joint coverage and mean
joint set size compare fairly — but the base score is Mahalanobis + penalties, **not** the
locked `s = 1−p if y=1 else p`. Per-label set size is undefined here.

**Split roles.** validation → proper-training (Σ, Hp vocabulary, Cp frequencies);
conformal → calibration; test → evaluation (locked per-member uncertain-exclusion rule).
Σ, Hp and Cp therefore inherit the standing all-13-clean caveat
(`RESEARCH_DIRECTION.md` §3a). Gate 0 passed.

**Arms.** base (µ=ν=0, Mahalanobis only) · penalised (µ=ν=1, +Hp+Cp) · admissible
(µ=ν=1, +Hp+Cp, +∞ on hierarchy-violating states). Each evaluated with and without the
paper's eq. 20 non-empty guard.

## Primary result (no guard)

| cluster | m | arm | joint coverage | mean joint set size | % of state space | decisive |
|---|---|---|---|---|---|---|
| cardio_pair | 2 | base | 0.8988 | 1.747 | 43.7% | 0.551 |
| | | penalised | 0.9011 | 1.444 | 36.1% | 0.631 |
| | | admissible | 0.9011 | 1.444 | 36.1% | 0.631 |
| lung_core | 3 | base | 0.8949 | 4.345 | 54.3% | 0.213 |
| | | penalised | 0.8907 | 3.381 | 42.3% | 0.262 |
| | | admissible | 0.8907 | 2.789 | 34.9% | 0.272 |
| pleural_free | 3 | base | 0.9035 | 3.642 | 45.5% | 0.291 |
| | | penalised | 0.8989 | 3.263 | 40.8% | 0.307 |
| | | admissible | 0.8989 | 3.263 | 40.8% | 0.307 |
| lung_wide | 5 | base | 0.8966 | 11.490 | 35.9% | 0.200 |
| | | penalised | 0.8978 | 9.086 | 28.4% | 0.252 |
| | | admissible | 0.8978 | **5.438** | **17.0%** | 0.282 |

## Cross-framework comparison, mean joint set size at α = 0.1

LP-SCP and max-rank/Bonferroni differ in **both** framework and base score; this is a
framework-level comparison on each method's native score, as in Katsios & Papadopoulos
2025 against BR-SCP/IR-SCP, **not** a controlled score ablation. A score-matched version
(power-set enumeration on the locked `s = 1−p if y=1 else p`, no Mahalanobis) would be a
useful secondary check and is not required for the comparison to be honest — logged as a
possible future appendix, not a blocker. Ruling: Tom Hagan, 2026-10-28.

| cluster | m | independent@α | Bonferroni | Šidák | max-rank | LP-SCP base | LP-SCP penalised | LP-SCP admissible |
|---|---|---|---|---|---|---|---|---|
| cardio_pair | 2 | 2.176 | 2.623 | 2.614 | 2.260 | 1.747 | 1.444 | 1.444 |
| lung_core | 3 | 2.413 | 4.245 | 4.200 | 3.977 | 4.345 | 3.381 | 2.789 |
| pleural_free | 3 | 2.126 | 3.802 | 3.757 | 3.726 | 3.642 | 3.263 | 3.263 |
| lung_wide | 5 | 5.874 | 16.563 | 16.396 | 14.839 | 11.490 | 9.086 | 5.438 |

Independent@α is not a competitor — it has no joint guarantee (coverage 0.711–0.884).

## Findings

**1. The penalties work, and by less than the paper's headline.** base → penalised cuts
set size by 17.3% / 22.2% / 10.4% / 20.9% across the four clusters, at coverage within
±0.5pp. That is in the range of the paper's *honest* all-24-combination average (6–13pp of
state-space share; we see 7.5 / 12.0 / 4.7 / 7.5 pp), not its cherry-picked top-3 figure
of 15–20pp. Our µ=ν=1 was fixed in advance and not searched, so this is the directly
comparable number.

**2. LP-SCP beats every shared-score method at m = 5, substantially.** `admissible` gives
5.44 joint states against max-rank's 14.84 and Bonferroni's 16.56 — a 63% reduction over
max-rank at comparable coverage (0.8978 vs 0.8993). At m = 2 the ordering is the same
(1.44 vs 2.26). This is the largest efficiency gain measured on this project.

**3. Learned structure does not substitute for encoded structure — it depends on m.**
This was the crux question. At m = 2, `penalised` and `admissible` are *identical to the
digit*: cardio_pair's single inadmissible state (EC=0, CM=1) was never observed in
proper-training, so Hp is already > 0 on exactly that state and the hard constraint adds
nothing. At m = 5 the picture reverses — admissibility cuts a further 40% (9.09 → 5.44),
because Hp is a graded *minimum* Hamming distance, not an exclusion: a state one bit from
an observed set gets a small penalty, and with 19 of 32 states inadmissible those small
penalties do not add up to removal. **So Hp absorbs hierarchy information when the
constraint count is small and the violating state is unobserved, and fails to when the
inadmissible region is large.** `pleural_free` (no constraints) is the control: identical
arms, as expected by construction.

**4. The non-empty guard is nearly inert, and is the primary variant.** Empty regions occur
only on cardio_pair, at 0.11–0.14% of rows; every other cluster is exactly 0.000. The guard
lifts cardio_pair coverage by ~0.1pp (0.8988 → 0.8999 on base) and leaves all other numbers
unchanged to four decimals. **Per Tom's ruling the with-guard numbers are primary** — an
empty region is a coverage failure under the paper's own eq. 20 definition — with the
no-guard column kept as a footnote confirming negligibility. (The primary table above is
the no-guard version; the difference is confined to cardio_pair's third decimal.)

**5. Hierarchy-violation rate of decisive output is 0.000 for every arm**, including
`penalised` with no hard constraint. This is **resolved, not an open question**: the data
pipeline applies hierarchy correction at load time (`data/hierarchy.py`), propagating
child-positive → parent-positive in the ground truth *before* any conformal method runs,
so hierarchy-violating combinations are already rare-to-absent in the labels every method
is scored against. The metric would become informative only on a model or dataset without
that correction. Ruling: Tom Hagan, 2026-10-28; applies equally to Exp046.

**6. Σ is well-conditioned at cluster scale.** Condition numbers 47.0 (cardio_pair), 3.44
(pleural_free), 56.3 (lung_wide). No ridge was applied anywhere. The feared near-singularity
from the shared backbone's ρ≈0.997 correlation did not materialise at m ≤ 5.

**7. Proper-training state coverage, which bounds what Hp can know:** cardio_pair 3 of 4
states observed, lung_core 5 of 8, pleural_free 8 of 8, lung_wide 13 of 32. At m = 5, Hp is
penalising 19 of 32 states as unobserved — and 19 of 32 are also hierarchy-inadmissible.
Those two sets coincide substantially, which is why finding 3 reads the way it does.

## Stratified coverage check — the gate on the headline number

Tom required conditional coverage split by whether a test row's true state was in
validation's observed vocabulary, to rule out Hp spending the error budget unevenly across
strata (marginally valid, locally invalid on the rare combinations). **The check passes,
and shows something stronger than expected: the unobserved stratum is empty.**

| cluster | m | states | observed in validation | observed in test | test-only states |
|---|---|---|---|---|---|
| cardio_pair | 2 | 4 | 3 | 3 | 0 |
| lung_core | 3 | 8 | 5 | 5 | 0 |
| pleural_free | 3 | 8 | 8 | 8 | 0 |
| lung_wide | 5 | 32 | 13 | 13 | 0 |

Validation and test observe **exactly the same** sets of joint states. Every test row's
true state is one Hp assigns zero penalty to, so there is no stratum on which Hp could be
undercovering — the cross-tab is single-stratum by construction, 100% observed, coverage
as reported in the primary table.

## The headline claim — a structural lemma, not a percentage

The 19 states Hp penalises at m = 5 have zero ground-truth support in *either* split. The
top 3 states alone cover 81.1% of test rows (cardio_pair 100%, lung_core 89.8%,
pleural_free 84.5%). Against that, max-rank's 14.84 and Bonferroni's 16.56 retain **more
states than the label distribution contains.** The claim to make is the mechanism, not the
63%:

> Any joint region constructed as a Cartesian product of per-label admissible sets — this
> includes independent, Bonferroni, Šidák, and max-rank calibration — cannot exclude a
> combinatorially-achievable joint state while any contributing label's marginal set
> remains ambiguous. At m = 5, 19 of 32 states have zero observed support in either split;
> the product-constructed methods retain many of them anyway, because doing otherwise would
> require collapsing a label's marginal set to a point, which per-label calibration has no
> mechanism to do selectively per combination. LP-SCP enumerates P(L) directly and can
> price each joint state independently, which is the mechanism — not a tuning artifact —
> behind its smaller regions at this m.

This is a combinatorial fact about the method family, not an empirical property of this
run: a per-label method with arbitrarily well-chosen thresholds still cannot drop an
*interior* point of the product (a combination implausible as a whole though each of its
marginal values is individually plausible). It is a structural ceiling, not a
calibration-sharpness issue.

**Two qualifications belong next to the claim, not in a footnote.**

*First*, the 13-state support figure inherits §3a and must not be read as a fact about
chest-X-ray findings. Within the all-13-clean population (§3a), effective support is 13 of
32 states; this has **not** been verified against the broader per-cluster-member-clean
population. Studies uncertain on labels outside the cluster are plausibly exactly the
atypical cases most likely to produce a combination outside the clean 13 — if so the true
support is larger, and some of the "structurally wasted" budget on product methods is less
wasteful against the real population than this estimate suggests. Read it as a property of
the filtered population, never asserted of the underlying label distribution.

*Second*, the lemma makes a testable prediction: the LP-SCP-vs-product gap should grow with
joint-state sparsity and shrink toward zero where there is no sparsity to exploit.
**Confirmed, with one inversion** (max-rank vs LP-SCP admissible, with guard):

| cluster | m | unobserved states | % unobserved | max-rank | LP-SCP adm. | reduction |
|---|---|---|---|---|---|---|
| pleural_free | 3 | 0/8 | 0.0% | 3.73 | 3.26 | 12.4% |
| cardio_pair | 2 | 1/4 | 25.0% | 2.26 | 1.44 | 36.1% |
| lung_core | 3 | 3/8 | 37.5% | 3.98 | 2.79 | 29.9% |
| lung_wide | 5 | 19/32 | 59.4% | 14.84 | 5.44 | 63.4% |

Spearman ρ = 0.80 over four points. The control case behaves as the lemma requires —
`pleural_free`, with every state observed, gives the smallest gap (12.4%, attributable to
the different base score rather than to sparsity), and the densest-sparsity cluster gives
the largest. `cardio_pair` and `lung_core` are inverted relative to the prediction; with
m differing between them (2 vs 3) and four points total, this is not evidence against the
mechanism, but it is not confirmation of strict monotonicity either. State it as directional
support, not a fitted relationship.

## Exploratory appendix — µ/ν sensitivity (not primary)

Full 5×5 grid, reported whole, as the deliberate corrective to the paper's §4.4
selection-on-outcome. Primary number remains (1,1). Mean joint set size, with-guard:

`lung_wide` (coverage range 0.8961–0.8984 across all 25):

| µ \ ν | 0.00 | 0.25 | 0.50 | 1.00 | 2.00 |
|---|---|---|---|---|---|
| 0.00 | 11.490 | 11.133 | 10.871 | 10.308 | 9.541 |
| 0.25 | 11.159 | 10.815 | 10.554 | 9.990 | 9.244 |
| 0.50 | 10.838 | 10.493 | 10.233 | 9.680 | 8.944 |
| 1.00 | 10.207 | 9.870 | 9.618 | 9.086 | 8.385 |
| 2.00 | 9.053 | 8.745 | 8.517 | 8.044 | 7.432 |

`lung_core` 4.345 → 2.864, `cardio_pair` 1.748 → 1.305, `pleural_free` 3.642 → 3.053 over
the same corners. Observations:

- **Monotone in both weights on every cluster** — no inversion anywhere, unlike the paper's
  Yeast result where Cp at high weight hurt. Both penalties help throughout our grid.
- **µ has literally no effect on `pleural_free`** (identical down every column): all 8
  states are observed, so Hp ≡ 0 and the Hamming penalty cannot do anything. This is the
  mechanism from finding 3, visible directly.
- Coverage is stable in 0.896–0.904 across all 25 configurations on every cluster, so the
  grid is a size/efficiency trade only, not a validity trade.
- (2,2) is best everywhere, consistent with the paper finding µ = 2 in its top
  configurations. **We do not adopt it** — (1,1) was pre-registered and stands as primary.
  Reporting the best grid cell as a headline would reproduce exactly the flaw this appendix
  exists to correct.

## Caveats

- **Decisive accuracy must be reported next to set size in every table — never set size
  alone** (Tom's ruling). It *falls* as sets shrink on lung_core (0.911 → 0.871 → 0.857)
  while the decisive rate rises (0.213 → 0.262 → 0.272), and moves the other way on
  cardio_pair (0.837 → 0.860). Smaller regions and a higher decisive rate do not mean
  better decisions — they mean the method commits more often, and on lung_core committing
  more often costs accuracy among the committed cases. This is marginal coverage being an
  average over rows rather than a per-stratum guarantee; nothing here is wrong, but
  "LP-SCP admissible wins" must not stand without the accuracy line beside it. Same
  pattern as the §4 record for Structured Layer 1 (won on set size, less often decisive).
- Coverage is slightly under nominal on lung_core (0.8907) and the base arm of lung_wide,
  as in Exp046, and for the same diagnosed §3a reason. Not re-litigated here.
- No bootstrap CIs on these deltas yet. Same hold as Exp046.
- µ, ν sensitivity grid not run in this pass; available via `--grid` as a labeled
  exploratory appendix, to be reported whole if run.

## Files

`lp_scp_results.csv` (all arms × guard × cluster, with `is_primary` flag),
`thresholds/lp_scp_*.json`. Code: `lp_scp.py`, shared core `conformal_common.py`
(`enumerate_states`, `state_indices`, `evaluate_joint`).
