# Exp053 — Equal-coverage cluster comparison (SPEC DRAFT, awaiting Tom's review)

**Status:** draft from Alon + main session, 2026-11-10. Not pre-registered. Nothing run.

## Idea (Alon)
Partition the 13 labels into K clusters. Each method runs inside each cluster. The
per-cluster level is raised by exactly enough that the **combined 13-label prediction has
90% joint coverage**. Every method and every partition is then compared at the same overall
coverage, on set size. Hypothesis: a correct choice of clusters is less conservative —
smaller combined sets at the same 90% — than an arbitrary one.

This differs from Exp051/052, which ran every cluster at α = 0.10 with no adjustment, so
overall coverage fell to 0.45–0.59 for K > 1.

## How the bar is raised — one common β per (method, partition)
Each method, inside a cluster, has one dial: nominal per-cluster miss level β.

1. For every calibration row i and cluster k, compute the method's p-value for the true
   label vector, p_ik = the largest β at which cluster k's set still contains the truth:
   - Bonferroni: m_k · min_j p_ij (p_ij = per-label conformal p-value)
   - Šidák: 1 − (1 − min_j p_ij)^m_k
   - max-rank: conformal p-value of the row's max-rank statistic within the cluster
   - LP-SCP: conformal p-value of the row's joint score (eq. 30, `penalised`, μ = ν = 1,
     structure fit on validation, as in Exp052)
2. Row statistic t_i = min_k p_ik (a row is covered only if every cluster covers it).
3. β̂ = the α = 0.10 conformal quantile of t_i over calibration rows (finite-sample
   corrected). Run every cluster at β̂.
4. Same β̂ for all clusters (Alon, 2026-11-10: "same β for starters").

At K = 1, β̂ = α. With K independent clusters, 1 − β̂ ≈ 0.9^(1/K).

## Partitions
- References: K = 13 singletons; K = 1 (all 13).
- Dependence-informed: hierarchical clustering on validation-split dependence between
  labels' nonconformity scores; cut at K = 3, 4, 6. (Statistic and linkage: Tom to pin.)
- Clinical: Exp051 rungs C (K = 7) and D (K = 6).
- Random null: random partitions with the same size profile as each dependence-informed
  cut (count: Tom to pin; proposed 200–500).
- Anti-clustering: most-dependent labels forced into different clusters, same size profile.

## Methods
Bonferroni, Šidák, max-rank, LP-SCP — each on every partition.

## Metrics
- Gate: overall 13-label joint coverage on test ≈ 0.90 for every (method, partition).
- β̂ and the achieved per-cluster coverage (how far the bar was raised).
- Mean joint set size (product over clusters), log₂(size)/13.
- Set-size ratio vs. the same method at K = 1 and vs. the random-null median.
- Secondary: fully-decisive rate, hierarchy-violation rate of fully-decisive output.
- CIs: subject-level bootstrap.

## Hypotheses (to be pre-registered)
- H1: every (method, partition) lands at ≈ 0.90 overall.
- H2: for each method, the dependence-informed partition gives a smaller mean joint set
  than the random null's 5th percentile at the same size profile.
- H3: anti-clustering is above the null median (sign check).

## Open questions for Tom
1. Validity of step 1–3 when the per-label / per-cluster p-values and the β̂ quantile use
   the same calibration rows (in-sample ranks, as max-rank does) — sound, or split the
   calibration set?
2. Exp052's unanswered escalation: K = 1 max-rank covers 0.893 on test vs. 0.900
   in-sample. H1's tolerance depends on the ruling.
3. Population: all-13-clean rows only (test N = 18,877), since the target is the 13-label
   vector?
4. Dependence statistic for the informed partition, linkage, and a cap on m (LP-SCP
   enumerates 2^m states).
5. LP-SCP non-empty guard on or off (open since Exp052).
6. Number of random partitions and bootstrap replicates.
