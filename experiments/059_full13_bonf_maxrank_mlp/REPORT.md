# Exp059 — 13-label joint coverage: Bonferroni vs max-rank vs learned MLP scorer

Rerun after the Gate 2 amendment (2026-11-17). Report text written by Jules; saved to file by the main session
(subagent file write was blocked). Numbers verified against `summary.csv`, `paired.csv`, `gates.json`.

**Population:** all-13-clean rows. Pool = 37,631 (conformal 18,754 + test 18,877), ~77% of test. α = 0.10, no guard,
no hierarchy enforcement. Frozen draw + 100 subject-level resplits.

## Gates (all PASS)

| gate | result |
|---|---|
| 0 disjoint + hierarchy check | PASS |
| 1 pool = 37,631 | PASS |
| 2 (amended) reproduction | PASS — max-rank `mean_size` 2855.40 (2855.402); max-rank `mean_admissible_size` 850.00 (849.995); Bonferroni `mean_size` 3171.9 (3171.89), frozen 3021.0 (3020.995), coverage 0.9206 |
| soft: C seed 0 `mean_size` in [25, 36] | PASS, 29.96 |
| 3 identity `none` vs `max_rank` in C | PASS, 0 differing draws, all seeds |
| 4a coverage ≥ 0.895 | PASS (lowest 0.8993, arm D) |
| 4b coverage ≤ 0.905 | PASS (highest 0.9009, C seed 2) |
| 5 `empty_rate` ≤ 0.5% | PASS — arm D resplit mean 0.26%, frozen 0.23% (per-draw max 0.66%, info only) |
| 6 `hit_cap` false (C s0); C seed spread ≤ \|C0 − B\| | PASS — spread 0.0051 vs gap 0.3729 |

## Headline

Resplit mean [2.5, 97.5 pct] (frozen in parentheses). Sizes are unrestricted product sets.

| arm | joint_coverage | median_size | log2_size_over_m | mean_size | empty_rate |
|---|---|---|---|---|---|
| A Bonferroni (per-label, α/13) | 0.9206 [0.9117, 0.9304] (0.9154) | 2048 [2048, 2048] (2048) | 0.7463 [0.7384, 0.7529] (0.7376) | 3172 [3075, 3283] (3021) | 0 |
| B max-rank (per-label) | 0.8997 [0.8888, 0.9128] (0.8934) | 2048 [2048, 2048] (1024) | 0.7240 [0.7156, 0.7309] (0.7147) | 2855 [2755, 2949] (2718) | 0 |
| **C MLP full-13, seed 0 (headline)** | 0.8999 [0.8906, 0.9096] (0.9007) | 28.5 [27, 30] (28) | 0.3511 [0.3457, 0.3562] (0.3503) | 29.96 [28.47, 31.35] (29.74) | 0 |
| D MLP singletons + max-rank, seed 0 (diagnostic) | 0.8993 [0.8909, 0.9084] (0.9001) | 126.1 [94.4, 128] (128) | 0.4594 [0.4455, 0.4711] (0.4537) | 425.4 [387.1, 463.5] (417) | 0.26% |

Column note: 850.00 is max-rank's `mean_admissible_size` (hierarchy-admissible count, measured only; frozen 809.86).
It is **not** max-rank's set size, which is `mean_size` 2855.40. `share_impossible` is 0.66–0.67 for A and B.
Decisive-output violation rate ~1e-5–1e-4 for A and B, 0 for C seed 0.

## Paired CIs

100 resplits, 95% percentile; conditional on the fixed pool and fixed trained scorers (understates total uncertainty).

| pair | metric | mean diff | 95% CI | excludes 0 |
|---|---|---|---|---|
| C0 − B | log2_size_over_m | −0.3729 | [−0.3825, −0.3646] | yes |
| C0 − B | median_size | −2019.5 | [−2021, −2018] | yes |
| B − A | log2_size_over_m | −0.0222 | [−0.0244, −0.0204] | yes |
| B − A | median_size | 0.0 | [0, 0] | no (medians tie at 2048) |
| C0 − D0 | log2_size_over_m | −0.1083 | [−0.1211, −0.0961] | yes |
| C0 − D0 | median_size | −97.6 | [−101, −65.5] | yes |
| B − D0 | log2_size_over_m | +0.2646 | [+0.2530, +0.2729] | yes |
| B − D0 | median_size | +1921.9 | [+1920, +1953.6] | yes |

The B − D0 rows were added after Tom's sign-off (computed with `evaluate.paired` from `resplits.csv`, no rerun).
Decomposition of the C0 − B gap in log2/13 (−0.373): B → D0 (learned per-label score) −0.265, about 70%;
D0 → C0 (joint 13-label score) −0.108, about 30%.

Direction-of-bias notes (Tom):
- D's 0.26% empty sets are dropped from log2/13, which flatters D, so the C0 − D0 gap is, if anything, understated.
- Only the last of D's 13 MLP fits is logged; `hit_cap` is unknown for the other 12. Coverage is unaffected; an
  under-trained per-label fit would inflate D's sets and overstate the C0 − D0 gap.
- Not the best number on record: Exp057's additive score on the DP m_max=6 partition had log2/13 0.3447 (mean size
  27.2), on par with C. One cluster of 13 is not claimed optimal.

## Claim (Tom, signed off 2026-11-17)

On MIMIC-CXR (DenseNet-121 unified model, frozen probabilities; all-13-clean rows, about 77% of test), targeting
joint coverage of the full 13-label vector at α = 0.10, over 100 subject-level calibration/test resplits, split
conformal with a learned joint score (a small MLP from the 13 logits to a softmax over all 8,192 label vectors,
trained on a separate validation split) reaches 0.900 joint coverage with a median set of 28.5 label vectors
(geometric mean ≈ 24; mean 30.0). Per-label conformal with max-rank (coverage 0.900) and Bonferroni (0.921,
conservative) both give a median of 2,048 vectors (geometric means ≈ 680 and ≈ 830). That is about 29× smaller in
geometric-mean size at equal coverage (paired 95% CI on log2(size)/13: −0.383 to −0.365). A control applying the same
learned scorer per label, combined by max-rank (geometric mean ≈ 63), splits the gain: about 11× from learning the
score from data, about 2.7× more from scoring the labels jointly (CI −0.121 to −0.096). The learned scorer never put a
hierarchy-violating vector in a decisive set without being told the hierarchy. CIs are conditional on the fixed pool
and trained scorers; coverage is exchangeable at subject level, so row-level coverage is approximate. Stable across
five MLP seeds.

## Seed spread

- C coverage, seeds 0–4: 0.8999–0.9001; log2/13: 0.3495–0.3547 (range 0.0051) vs C0 − B gap 0.37.
- D coverage 0.8993–0.8994; log2/13 0.4594–0.4598.

## Prior replication

Exp056 arm D (masked, old Lightning trainer, seed 0): coverage 0.8999, mean size 30.06 [28.54, 31.73], frozen 29.65,
median 28.4. Exp059 C seed 0 (no enforcement, `conformal/`): coverage 0.8999, mean size 29.96 [28.47, 31.35], frozen
29.74, median 28.5. The 13-label MLP has a prior masked number, so this arm is a replication under the no-enforcement
convention, not a first measurement.

## Diagnostics

- MLP training rows 14,853 (holdout 3,717); 367 distinct 13-label training states.
- Test rows whose true 13-label state is unseen in MLP training: frozen 1.08%, resplit mean 1.07% (0.85%–1.30%).
- Arm C holdout CE seeds 0–4: 2.6819, 2.6911, 2.6847, 2.6847, 2.6846; best epoch 18–25; stop 28–35; `hit_cap` False.
- Arm D `info` is from the last of 13 fits only (Support Devices, holdout CE 0.381–0.382); not representative.

## Caveats

- All-13-clean population (~77% of test); uncertain-label rows not covered.
- Arms C/D train on ~14.9k labelled validation rows; A/B use none.
- Subject-level splits: coverage exchangeable at subject level; row-level coverage approximate.
- CIs conditional on the fixed pool and trained scorers.
- Not run: DP m_max=6 partition with MLP (next step).

## Provenance

git 98b8269 (`conformal/` untracked; hashes in `code_hashes.txt`). Reproduce from repo root:
`.venv/bin/python experiments/059_full13_bonf_maxrank_mlp/run.py 2>&1 | tee experiments/059_full13_bonf_maxrank_mlp/run.log`.
Runtime ~14 minutes.
