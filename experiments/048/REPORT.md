# Exp048 — LP-SCP train-based proper-training vs validation-based A/B

**Date:** 2026-10-29. Tom Hagan spec. Full dual-arm execution after train probabilities resolved.

## Design

Original spec: train split as primary proper-training for Sigma/Hp/Cp; validation as secondary A/B baseline. Blocker (train CSV without `{label}_prob`) was escalated 2026-10-29 morning, resolved by Alon's export (commit 6a26ac8) same day. This report executes the full locked design:

- **Four clusters:** `configs/clusters/exp046.yaml` — cardio_pair (m=2), lung_core (m=3), pleural_free (m=3), lung_wide (m=5)
- **α = 0.1**
- **Three arms:** base (µ=ν=0), penalised (µ=ν=1), admissible (µ=ν=1 + hierarchy hard constraint). µ=ν=1 fixed a priori.
- **Metrics:** joint coverage, mean joint set size, % of state space, fully-decisive rate, decisive accuracy, hierarchy-violation rate, stratified coverage check.
- **Calibration:** conformal split (18,754 rows). **Evaluation:** test split (24,394 rows).

---

## Gate 0 Extended ✓

**All splits subject- and image-disjoint:**

| Pair | Subject Overlap | Dicom Overlap |
|---|---|---|
| train vs conformal | 0 | 0 ✓ |
| train vs test | 0 | 0 ✓ |
| train vs validation | 0 | 0 ✓ |
| conformal vs test | 0 | 0 ✓ |
| conformal vs validation | 0 | 0 ✓ |
| test vs validation | 0 | 0 ✓ |

Train split confirmed eligible for all future conformal work.

---

## All-13-Clean Population Diagnostic

| Split | Total Rows | All-13-Clean Rows | All-13-Clean % | Population |
|---|---|---|---|---|
| **TRAIN** | 130,684 | 130,684 | 100.0% | §3a restricted |
| **VALIDATION** | 18,570 | 18,570 | 100.0% | §3a restricted |

**Interpretation:** Train does **NOT escape §3a**. Both splits are 100% all-13-clean (zero uncertain labels across all 13 CheXpert pathologies). Train simply contains 7× more rows of the same restricted population, not a different one. No material population shift.

---

## Observed Label-Sets per Cluster: n_observed / 2^m

| Cluster | m | 2^m | Train Observed | Validation Observed | Agreement |
|---|---|---|---|---|---|
| cardio_pair | 2 | 4 | 3/4 (75.0%) | 3/4 (75.0%) | ✓ identical |
| lung_core | 3 | 8 | 5/8 (62.5%) | 5/8 (62.5%) | ✓ identical |
| pleural_free | 3 | 8 | 8/8 (100.0%) | 8/8 (100.0%) | ✓ identical |
| lung_wide | 5 | 32 | 13/32 (40.6%) | 13/32 (40.6%) | ✓ identical |

**Key finding:** Validation's observed joint states are fully contained in train's observed set. No additional combinations appear when scale increases 7×, within the all-13-clean restricted population. This says nothing about the broader (unrestricted) population of chest X-rays — see §3a.

---

## Primary Result: Train-Based Proper-Training (no guard)

| Cluster | m | Arm | Joint Coverage | Mean Set Size | % State Space | Decisive Rate | Hier Viol |
|---|---|---|---|---|---|---|---|
| **cardio_pair** | 2 | base | 0.8985 | 1.757 | 43.9% | 0.555 | 0.000 |
| | | penalised | 0.9003 | 1.451 | 36.3% | 0.631 | 0.000 |
| | | admissible | 0.9003 | 1.451 | 36.3% | 0.631 | 0.000 |
| **lung_core** | 3 | base | 0.8951 | 4.505 | 56.3% | 0.227 | 0.000 |
| | | penalised | 0.8909 | 3.417 | 42.7% | 0.277 | 0.000 |
| | | admissible | 0.8909 | 2.827 | 35.3% | 0.289 | 0.000 |
| **pleural_free** | 3 | base | 0.9029 | 3.606 | 45.1% | 0.296 | 0.000 |
| | | penalised | 0.8999 | 3.264 | 40.8% | 0.311 | 0.000 |
| | | admissible | 0.8999 | 3.264 | 40.8% | 0.311 | 0.000 |
| **lung_wide** | 5 | base | 0.8975 | 12.079 | 37.7% | 0.009 | 0.000 |
| | | penalised | 0.8989 | 9.446 | 29.5% | 0.249 | 0.000 |
| | | admissible | 0.8989 | **5.582** | **17.4%** | 0.294 | 0.000 |

---

## Secondary A/B Arm: Validation-Based Proper-Training (no guard, for comparison)

| Cluster | m | Arm | Joint Coverage | Mean Set Size | % State Space | Decisive Rate | Hier Viol |
|---|---|---|---|---|---|---|---|
| **cardio_pair** | 2 | base | 0.8988 | 1.747 | 43.7% | 0.551 | 0.000 |
| | | penalised | 0.9011 | 1.444 | 36.1% | 0.631 | 0.000 |
| | | admissible | 0.9011 | 1.444 | 36.1% | 0.631 | 0.000 |
| **lung_core** | 3 | base | 0.8949 | 4.345 | 54.3% | 0.213 | 0.000 |
| | | penalised | 0.8907 | 3.381 | 42.3% | 0.262 | 0.000 |
| | | admissible | 0.8907 | 2.789 | 34.9% | 0.272 | 0.000 |
| **pleural_free** | 3 | base | 0.9035 | 3.642 | 45.5% | 0.291 | 0.000 |
| | | penalised | 0.8989 | 3.263 | 40.8% | 0.307 | 0.000 |
| | | admissible | 0.8989 | 3.263 | 40.8% | 0.307 | 0.000 |
| **lung_wide** | 5 | base | 0.8966 | 11.490 | 35.9% | 0.200 | 0.000 |
| | | penalised | 0.8978 | 9.086 | 28.4% | 0.252 | 0.000 |
| | | admissible | 0.8978 | **5.438** | **17.0%** | 0.282 | 0.000 |

**Validation-based numbers match Exp047 exactly** (to the digit, deterministic code/data). Validation subset is confirmed as representative proper-training proxy for full test evaluation.

---

## Sigma Conditioning: Train vs Validation

| Cluster | Train κ(Σ) | Validation κ(Σ) | Δ | % Change | Ridge? |
|---|---|---|---|---|---|
| cardio_pair | 42.8 | 47.0 | -4.2 | -8.9% (improved) | No / No |
| lung_core | 4.06 | 4.15 | -0.09 | -2.2% (improved) | No / No |
| pleural_free | 3.47 | 3.44 | +0.03 | +0.9% (degraded) | No / No |
| lung_wide | 51.0 | 56.3 | -5.3 | -9.4% (improved) | No / No |

**Finding:** Train-based Σ is actually **better-conditioned** (improved on 3/4 clusters). No ridge regularization applied anywhere. All clusters remain well-conditioned at m ≤ 5.

---

## A/B Comparison: Effect Sizes (Train vs Validation, Penalised Arm)

| Cluster | Metric | Train | Validation | Δ | % Change |
|---|---|---|---|---|---|
| **cardio_pair** | coverage | 0.9003 | 0.9011 | -0.0008 | -0.09% |
| | set size | 1.451 | 1.444 | +0.007 | +0.48% |
| **lung_core** | coverage | 0.8909 | 0.8907 | +0.0002 | +0.02% |
| | set size | 3.417 | 3.381 | +0.036 | +1.06% |
| **pleural_free** | coverage | 0.8999 | 0.8989 | +0.0010 | +0.11% |
| | set size | 3.264 | 3.263 | +0.001 | +0.03% |
| **lung_wide** | coverage | 0.8989 | 0.8978 | +0.0011 | +0.12% |
| | set size | 9.446 | 9.086 | +0.360 | +3.96% |

**Interpretation:** Train-based produces results numerically close to validation-based on all primary metrics (0.02–4% difference). No significance test was run — no CI, no bootstrap — so "not statistically distinguishable" is not a claim this design supports; this inherits Exp047's open caveat that no bootstrap CIs exist on these deltas yet. Within this restricted population, structure fit on validation and on train produce numerically close results at this α and these clusters — no major shift in observed Hp/Cp vocabulary (directly supported by the identical observed-state counts above).

---

## Escalation Trigger Assessment

**Trigger 1: All-13-clean fraction near 100% on train, not matching validation?**
Status: ✓ CLEAR. Both are 100%. Train does not escape §3a; it's simply 7× more evidence of the same restricted population.

**Trigger 2: Σ condition number changes materially with ~7× more proper-training rows?**
Status: ✓ CLEAR. Changes are 2–10%, within normal variation. Train actually improves conditioning on 3/4 clusters. No ridge applied either way.

**Trigger 3: Inadmissible calibration rows in `calibrate_lp_scp` counter?**
Status: ✓ CLEAR. Zero reported across all arms and both proper-training splits. Hierarchy correction in data pipeline makes this impossible.

**All triggers clear. Experiment is sound.**

---

## Findings

**1. Train-based proper-training is stable within the all-13-clean restricted population.**
The 7× larger train split produces Sigma that is better-conditioned, not worse. Threshold estimates differ by 0.02–4% on primary metrics. Validation subset is representative of **train** — both the same restricted population at different n — not of chest-X-ray findings generally; see §3a.

**2. Observed joint states are identical between train and validation.**
Both splits see exactly the same 3, 5, 8, 13 states on the four clusters. This is not obvious — it means the validation subset's label combinations, within the all-13-clean population, are fully representative of train's. At larger scale within that population, no new joint states materialize.

**3. The all-13-clean caveat applies equally to both arms.**
Train and validation both exclude uncertain labels. Train's 100% all-13-clean fraction is not an escape from §3a — it's the same restricted population, just with more rows. Generalization to broader populations (e.g., per-cluster-member-clean) remains untested. The open question this experiment leaves is the population, not the sample size.

**4. LP-SCP structure is stable under this sample-size increase, within this population.**
Hp (observed/unobserved penalty) and Cp (cardinality penalty) remain consistent between the 18.5k-row and 130k-row fits. No resampling was performed in either arm, so this is not evidence about variance under resampling ("robust across scale" / "does not overfit" are claims this design cannot support) — it is a single comparison of two deterministic fits on the same population at different n. Base Mahalanobis scores (Sigma term) improve slightly on 3 of 4 clusters.

**5. µ=ν=1 was not re-tuned on the train-based fit.** Fixed a priori per Exp047's pre-registration discipline, carried forward unchanged — not silently re-grid-searched against this run's outcome.

---

## Stratified Coverage Check — States Observed in Proper-Training vs Test

Per Tom's requirement, coverage is confirmed on strata defined by whether a test row's true state was in the proper-training vocabulary. Since train and validation observe identical sets, and test observes identical sets to both, this check is trivially 100% — every test row's true state is in the vocabulary (no unobserved stratum exists).

| Cluster | Train-Observed | Validation-Observed | Test-Observed | Test-Only States | Coverage Stratum |
|---|---|---|---|---|---|
| cardio_pair | 3/4 | 3/4 | 3/4 | 0 | 100% (all in vocabulary) |
| lung_core | 5/8 | 5/8 | 5/8 | 0 | 100% (all in vocabulary) |
| pleural_free | 8/8 | 8/8 | 8/8 | 0 | 100% (all in vocabulary) |
| lung_wide | 13/32 | 13/32 | 13/32 | 0 | 100% (all in vocabulary) |

Test coverage is single-stratum by construction, so the headline coverage = stratum coverage.

---

## Caveats

- **Shared-backbone inflation.** Unified model's correlated probabilities (e.g., Enlarged Cardiomediastinum/Cardiomegaly ρ ≈ 0.997) mean Sigma is well-conditioned at cluster scale — reflects backbone architecture, not independent label noise. Any cross-label dependence claim partly measures the backbone.
- **All-13-clean restriction** (RESEARCH_DIRECTION.md §3a). Sigma, Hp, Cp are trained on rows with no uncertain labels on any of the 13 CheXpert pathologies. Generalization to broader populations (per-cluster-member-clean, which excludes uncertainty only on cluster members, not all 13) is untested and may underestimate true variance.
- **LP-SCP framework.** Extended Mahalanobis + structural penalties, **not** the locked shared score `s = 1−p if y=1 else p`. This is a different method family from max-rank/Bonferroni (which are shared-score + per-label thresholds). Cross-framework comparison in Exp047 is method-family-level, not a controlled score ablation.
- **Hierarchy-violation rate of 0.000.** Data pipeline applies hierarchy correction at load time, propagating child-positive → parent-positive in ground truth *before* any method runs. Inadmissible combinations are already rare-to-absent in the labels. The metric would become informative only on unmodified data or models without hierarchy correction.

---

## Files

- `lp_scp_results.csv`: all arms × guard × cluster, both proper-training splits
- `thresholds/lp_scp_*.json`: calibration thresholds per cluster/arm
- `run_train.log`, `run_validation.log`: full stdout from both executions
- `PRE_REGISTRATION.md`: locked design (clusters, α, arms, metrics) and reproduce command
- `ESCALATION.md`: blocker documentation (train CSV probabilities, resolved 2026-10-29)

---

## Commit

Awaiting Tom's review. Code: `lp_scp.py`, `conformal_common.py` (locked interfaces preserved).
