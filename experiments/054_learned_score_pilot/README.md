# Exp054 — Learned joint-state score: PILOT / sanity check (PRE-REGISTRATION)

**Status: PILOT.** A sanity check that the pipeline runs, that joint coverage lands near 0.90,
and that set sizes are in a sensible range against Bonferroni. It is **not a result**. Single
split, no CIs, no random-cluster control, no logistic-regression floor. No efficiency claim is
supported by this experiment.

**Authority.** Alon commissioned this pilot directly. Tom Hagan reviewed the broader design
(a learned joint-state score as a controlled comparator, `RESEARCH_DIRECTION.md` §2–3a) but has
**not ruled on this reduced pilot**.

**Spec.** Reduced pilot spec from Alon (2026-11-15), implemented exactly; design nothing.
Reads: `RESEARCH_DIRECTION.md` §2, §3, §3a; `lp_scp.py`; `conformal_common.py`; `bonferrony.py`.

**Git HEAD at pre-registration:** `6a26ac848b016e82c22d1bd42f61280740d3d26d` (working tree dirty;
`run.py`, this README and the reused modules are untracked; nothing committed).
**`run.py` sha256 at pre-registration:** `e93b5eb26400f998edfb7861dacd3f16193111a79c2558d28b567120027f7590`.
**Pre-registration time:** 2026-11-15, before `run.py` was executed. No metric was computed.

## What is tested
Per cluster, the pre-registered learned joint-state score, arm **L1**, versus **Bonferroni** (α/m,
per-label, product set), on the same all-13-clean rows and splits. Also the unguarded L1 set
size (recorded, not primary).

## Inputs (frozen; no inference re-run)
| role | file | sha256 (first 16) |
|---|---|---|
| validation (train MLP) | `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_validation.csv` | `e2c7dfd3087154…` |
| conformal (threshold) | `…_conformal.csv` | `cce3677863ab2a…` |
| test (evaluation) | `…_test.csv` | `ab42e9a895e0b1…` |

Not used: `…_train.csv` (loaded by nothing in this experiment). Full hashes in `run_info.json`.
Reused code (read-only): `conformal_common.py` (sha `9ddae7d9b0b8…`), `bonferrony.py`
(`3b887c7c5e1704…`), `lp_scp.py` (`79123faec33bfb…`, read for the score/guard conventions, not
imported), `data/hierarchy.py` (`86df94816d46f6…`), `configs/clusters/exp046.yaml` (`ed115cb05272…`).

## Clusters (unchanged, `configs/clusters/exp046.yaml`)
| name | m | labels (yaml order) | hard constraints |
|---|---|---|---|
| cardio_pair | 2 | Enlarged Cardiomediastinum, Cardiomegaly | 1 |
| lung_core | 3 | Lung Opacity, Edema, Consolidation | 2 |
| pleural_free | 3 | Pneumothorax, Pleural Effusion, Support Devices | 0 |
| lung_wide | 5 | Lung Opacity, Edema, Consolidation, Pneumonia, Atelectasis | 4 |

## Population
**All-13-clean rows only** in all three splits: every one of the 13 true labels in {0,1} and every
prob finite. Same rows for L1 and Bonferroni. This is the §3a population (the calibration
population the project already uses); test is a broader population only under the per-cluster
exclusion rule, which this pilot does not use. Expected test n ≈ 77% of 24,394 (to be reported as
measured).

## Gate 0 (stop and escalate on any failure)
1. Zero subject and zero dicom overlap across validation / conformal / test, computed on the
   full frozen CSVs before any filtering (`conformal_common.gate0_check`).
2. In every split, zero true label vectors violating the hard constraints for each cluster
   (`conformal_common.hierarchy_admissible`, the same function `lp_scp.py` uses). Checked on all rows
   whose cluster members are each in {0,1} (a superset of the all-13-clean rows, so the check is
   the broader of the two). A 13-label check is not gated.

## Model — arm L1 (one MLP per cluster)
- Input: the cluster's m logits, `log(p/(1−p))` of `{label}_prob`, p clipped to [1e-6, 1−1e-6].
- Network: `Linear(m, 64) → ReLU → Linear(64, 2^m)`.
- Target: state index `Σ_l y_l·2^l`, labels in yaml order (**spec convention: label 0 is the LSB**).
  Region columns are permuted to `conformal_common.enumerate_states` order before `evaluate_joint`.
- Loss: categorical cross-entropy over the 2^m states (not BCE).
- Adam, lr 1e-3, weight_decay 1e-4, batch 256.
- Inner holdout: 20% of the all-13-clean **validation subjects** (`numpy.random.default_rng(0)`
  permutation; `round(0.2·N)` subjects). One subject split, shared by all four clusters.
  Training on the remaining 80%.
- Early stopping: holdout CE; stop when it has not improved for 10 consecutive epochs; max 200
  epochs; keep the best-holdout-loss weights. Stopping epoch, best epoch, and final train/holdout
  CE recorded.
- Seeds: `torch.manual_seed(0)` re-applied before each cluster's model init; `torch.Generator`
  seeded 0 for minibatch order; `torch.set_num_threads(1)`.
- No other tuning, no architecture change, no standardisation (none specified).

## Conformal step — arm L1
- Score: `s(x,y) = −log_softmax(logits)[y]` for every joint state; `s = +∞` on states violating a
  hard constraint, applied after the network.
- Threshold: α = 0.10; calibration score = score of each conformal row's true state (after the
  constraint); `k = ⌈(n+1)(1−α)⌉`; threshold = k-th smallest; no interpolation; `+∞` if k > n.
- Prediction set (**guarded**, primary): `{y admissible : s(x,y) ≤ threshold} ∪ {argmin of constrained s}`.
  Admissibility is masked explicitly, not via `s ≤ threshold`, because with an infinite threshold
  `inf ≤ inf` would re-admit inadmissible states (the `lp_scp.build_region` path has this edge; not
  edited, not used here).
- Unguarded set also recorded (`L1_unguarded` rows in `results.csv`).

## Bonferroni reference
`bonferrony.calibrate_per_label(conformal_group, α, "bonferroni")` (per-label level α/m), evaluated on
test with `conformal_common.evaluate`. Joint set size = product of per-label set sizes. Per-label
sets can be empty under Bonferroni; those rows count as non-covered and are excluded from the
log₂ mean (empty rate reported).

## Metrics (per cluster, per method)
- **Validity gate:** test joint coverage (true state in set, over all m members) ≥ 0.90 − 1.96·√(0.09/n_test).
  Failure → escalate; the set size is not compared.
- Two-sided 95% binomial band around 0.90 for n_test (reported; "in band" is informational).
- Mean joint set size; median; `mean log₂(size)/m` over non-empty rows; empty-set rate;
  fully-decisive rate (size = 1); decisive accuracy; hierarchy-violation rate of fully-decisive output.
- n_cal, n_test, threshold, k, stopping epoch, best epoch, final train/holdout CE, number of joint
  states occupied in validation (distinct cluster state vectors among all-13-clean validation rows),
  number of admissible states.
- L1 vs Bonferroni ratio of mean set sizes: descriptive only (no CI, single split).

## Escalation triggers (pre-registered; no iteration if any fires)
- Gate 0 fails (any overlap or any hierarchy violation).
- L1 or Bonferroni test coverage below the lower gate on any cluster.
- L1 threshold is +∞, or L1 median set size equals the number of admissible states (degenerate).
- Any NaN in a reported metric.

Coverage far from 0.90 in either direction, or a degenerate set, is also an escalation trigger
under the spec's own instruction.

## Standing caveats (repeated in the results section)
1. **Single split, no CIs, no random-cluster control, no logistic-regression floor.** No efficiency
   claim is supported.
2. **§3a population caveat:** the MLP, the score and the threshold are all fit on all-13-clean rows.
3. **Shared-backbone caveat:** all probabilities come from one shared-backbone classifier, which
   inflates cross-label score correlation (EC/CM 0.936 → 0.997 vs independently trained models).
   This likely overstates any efficiency gain from joint calibration; it is not conservative.

## Reproduce
```
/usr/bin/python3 experiments/054_learned_score_pilot/run.py
```
Outputs: `results.csv`, `metrics.json`, `run_info.json` in this folder.
Seed 0 everywhere. Python 3.9.6, torch 2.8.0 (see `run_info.json`).

---

# Results

Run: `/usr/bin/python3 experiments/054_learned_score_pilot/run.py` (stdout in `run_stdout.log`),
2026-11-15, ~2 s. Outputs: `results.csv`, `metrics.json`, `run_info.json`. Pre-registration
hash of `run.py` matches the file that ran.

## Gate 0 — PASSED
- Subject overlap: 0 for all three pairs (conformal|test, conformal|validation, test|validation).
- Dicom overlap: 0 for all three pairs.
- Hierarchy violations of true label vectors: **0** in every split × cluster (12 checks; rows checked in `metrics.json`).
- Rows, full → all-13-clean: validation 18,570 → 18,570; conformal 18,754 → 18,754; test 24,394 → **18,877 (77.4%)**.
- Subjects, all-13-clean: validation 5,898; conformal 5,926; test 5,932.

## Validity gate — PASSED for all 8 rows
Lower gate 0.8957 (= 0.90 − 1.96·√(0.09/18,877)). Two-sided band 0.8957–0.9043.

| cluster | m | L1 joint cov | in band | Bonf joint cov | in band |
|---|---|---|---|---|---|
| cardio_pair | 2 | 0.9017 | yes | **0.9439** | no (over) |
| lung_core | 3 | 0.9036 | yes | **0.9161** | no (over) |
| pleural_free | 3 | 0.9023 | yes | 0.9026 | yes |
| lung_wide | 5 | 0.9023 | yes | **0.9251** | no (over) |

## Efficiency (descriptive, single split, no CIs — no claim supported)

| cluster | m | L1 mean joint size | L1 median | L1 log₂(size)/m | L1 fully-decisive | Bonf mean joint size | Bonf median | Bonf log₂(size)/m | Bonf fully-decisive |
|---|---|---|---|---|---|---|---|---|---|
| cardio_pair | 2 | 1.237 | 1 | 0.119 | 0.763 | 2.529 | 2 | 0.519 | 0.454 |
| lung_core | 3 | 1.993 | 2 | 0.281 | 0.350 | 4.067 | 4 | 0.537 | 0.272 |
| pleural_free | 3 | 2.016 | 2 | 0.270 | 0.418 | 3.589 | 4 | 0.474 | 0.316 |
| lung_wide | 5 | 2.834 | 3 | 0.249 | 0.292 | 16.159 | 16 | 0.647 | 0.166 |

Empty-set rate: 0 for every L1 and Bonferroni row.

Decisive accuracy (on each method's own fully-decisive rows; not a like-for-like comparison of sets):
L1 vs Bonferroni — cardio 0.879 vs 0.898; lung_core 0.914 vs 0.909; pleural_free 0.939 vs 0.963; lung_wide 0.930 vs 0.949.
Hierarchy-violation rate of fully-decisive output: 0 for L1 and Bonferroni in all clusters.

**Read these carefully.**
- The Bonferroni sets are at *different* coverage from L1 on three of four clusters (0.944, 0.916,
  0.925 vs 0.90). Bonferroni's larger sets are partly its over-coverage, which is the known union-bound
  behaviour and matches the prior reproduction target in RESEARCH_DIRECTION.md §4 (0.90–0.94). The
  size comparison is therefore not at equal coverage except on **pleural_free** (0.9023 vs 0.9026),
  where L1 mean size is 2.02 vs Bonferroni 3.59. That single matched-coverage pair is the only
  informative comparison here, and it is one split with no CI.
- Bonferroni's over-coverage (cardio 0.944) is outside the two-sided band. It is not an L1 validity
  failure and is not the pilot's target, so I did not escalate on it; flagged for Tom's reading.

## Learned score (L1) — training and conformal diagnostics

| cluster | admissible / 2^m | states occupied in validation | best epoch | stop epoch | final train CE | final holdout CE | conformal k | threshold | n_cal | n_test |
|---|---|---|---|---|---|---|---|---|---|---|
| cardio_pair | 3 / 4 | 3 | 86 | 96 | 0.470 | 0.451 | 16,880 | 1.1684 | 18,754 | 18,877 |
| lung_core | 5 / 8 | 5 | 34 | 44 | 0.852 | 0.818 | 16,880 | 2.0926 | 18,754 | 18,877 |
| pleural_free | 8 / 8 | 8 | 22 | 32 | 0.822 | 0.821 | 16,880 | 2.2503 | 18,754 | 18,877 |
| lung_wide | 13 / 32 | 13 | 37 | 47 | 1.097 | 1.070 | 16,880 | 2.8915 | 18,754 | 18,877 |

- All four runs stopped by the 10-epoch patience rule (none hit the 200-epoch cap). Stop epoch =
  best + 10.
- Validation occupies **every** admissible state in every cluster (the 13 of 32 at m = 5 is the
  admissible count, not a sparsity finding on the other 19 states).
- Thresholds are finite for all clusters; no degenerate sets.
- Guard inactive: L1 guarded and unguarded sets are identical on every metric in every cluster
  (empty rate 0 without the guard). The argmin state was always already in the threshold set.
- Holdout CE is slightly below train CE in all four clusters (e.g. lung_wide 1.070 vs 1.097). Not
  investigated; the holdout is a random 20% of validation subjects.

## Checks run after the pre-registered metrics (not part of the spec)
- `check_bonferroni_independent.py` recomputes Bonferroni coverage, mean and median size from the raw
  CSVs with numpy only (no conformal_common / bonferrony). Matches `results.csv` exactly on all four
  clusters (e.g. lung_wide 0.925094 / 16.159, pleural_free 0.90258 / 3.589).
- pleural_free and lung_wide L1 coverage are identical (0.902315; both 17,033 of 18,877). Checked:
  thresholds (2.2503 vs 2.8915) and set sizes (2.02 vs 2.83) differ; this is a coincidence.

## Escalation
None of the pre-registered triggers fired (Gate 0 passed; L1 coverage gate passed; no
degenerate or NaN metrics). No `ESCALATION.md` written. No variant was run.

## Caveats (stated in this result)
1. **Pilot. Single split, no CIs, no random-cluster control, no logistic-regression floor.** No
   efficiency claim is supported by these numbers; the matched-coverage pleural_free row is the only
   pair that could be read at all, and only descriptively.
2. **§3a population caveat.** The MLP, the score and the threshold are fit or calibrated only on
   all-13-clean rows. Validation and conformal are entirely all-13-clean in this export (no rows
   removed), so the restriction bites on test: 77.4% of test is evaluated. The 22.6% excluded test
   rows (ambiguous studies) are not represented. Coverage on them is unknown.
3. **Shared-backbone caveat.** All probabilities come from one shared-backbone classifier, which
   inflates cross-label score correlation (EC/CM 0.936 → 0.997 vs independently trained models).
   Higher correlation is what lets joint calibration beat Bonferroni, so this likely **overstates** the
   efficiency gain from clustering. It is not conservative.
4. L1 and Bonferroni coverage differ (above), so the size ratios are not at equal coverage.
