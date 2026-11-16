# Exp055 — Learned joint-state score, FULL design (PRE-REGISTRATION)

**Scope tag (every table and caption):** all-13-clean studies, 77% of test.

**What this is.** The full Exp054 design (Tom Hagan's spec, as amended in his ruling on the pilot;
Alon approved the full run 2026-11-15). This is the **full design following the 054 pilot**
(`experiments/054_learned_score_pilot/`). The pilot is a sanity check and is not part of this result.
The pilot touched test rows. **Hyperparameters were frozen before the full run. They were not changed
after seeing the pilot's test numbers.** The L1 training recipe is the pilot's, unchanged.

**Git HEAD at pre-registration:** `6a26ac848b016e82c22d1bd42f61280740d3d26d` (working tree dirty; this folder
and the reused modules are untracked; nothing committed).
**`run.py` sha256 at pre-registration:** `6f589bb4911e4732dd87988755a7c0f58d4a9908984803a39141c75a61647448`.
**Pre-registration time:** 2026-11-15, ~10:40 IST, before `run.py` was executed on any real split. Before this
point only (a) positive counts on the all-13-clean conformal split (needed to fix eligibility, calibration
labels only) and (b) a synthetic self-test that reads no frozen CSV were run. No metric from the frozen
splits was computed.

**Spec in one line.** Four designed clusters plus 20 random label subsets per cluster; learned joint-state
scores L0/L1/L2 (5 seeds each) and classical references, all calibrated at α = 0.10 with the
⌈(n+1)(1−α)⌉ rule, evaluated on 100 subject-level 50/50 resplits of the pooled all-13-clean conformal+test
population, with a frozen-draw reference row, identity gate, and pre-registered decision rules.

---

## 1. Inputs (frozen; no inference re-run)

| role | file (under `results/outputs/mimic/`) | rows (all-13-clean) | sha256 (first 16) |
|---|---|---|---|
| validation (scorers, LP-SCP structure) | `densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_validation.csv` | 18,570 (5,898 subj) | `e2c7dfd308715440` |
| conformal (calibration half of the frozen draw) | `…_conformal.csv` | 18,754 (5,926 subj) | `cce3677863ab2a24` |
| test (evaluation half of the frozen draw) | `…_test.csv` | 18,877 of 24,394 (5,932 subj) | `ab42e9a895e0b1c3` |

Reused, unedited: `conformal_common.py` (`9ddae7d9b0b81147`), `bonferrony.py` (`3b887c7c5e1704936`),
`max_rank.py`, `lp_scp.py` (`79123faec33bfb9f7`, the same hash recorded for Exp053 and the pilot), `data/hierarchy.py`
(`86df94816d46f64b`), `configs/clusters/exp046.yaml` (`ed115cb05272a721`), and `experiments/054_learned_score_pilot/results.csv`
(used only for the reproduction check, §10). Full hashes in `run_info.json`.

## 2. Population and Gate 0

- Population: **all-13-clean rows only in every split** (every one of the 13 true labels in {0,1}, every prob finite).
  Validation 18,570 → 18,570; conformal 18,754 → 18,754; test 24,394 → 18,877 (77.4%).
- **Gate 0 (stop and escalate on failure):**
  1. zero subject and zero dicom overlap across the three frozen CSVs (`cc.gate0_check`, full frames, before filtering);
  2. zero true-label vectors violating the hierarchy, in every split (full frames), for every designed cluster and every
     random draw (rows whose unit members are all in {0,1}; `hier_violations`, pilot logic).
- Pooled design for resplits: conformal 13-clean ∪ test 13-clean = **37,631 rows, 11,858 subjects** (asserted; disjoint).

## 3. Clusters

**Designed (unchanged, `configs/clusters/exp046.yaml`):**

| name | m | labels | directed hierarchy edges within cluster (= hard-constraint terms) |
|---|---|---|---|
| cardio_pair | 2 | Enlarged Cardiomediastinum, Cardiomegaly | 1 |
| lung_core | 3 | Lung Opacity, Edema, Consolidation | 2 |
| pleural_free | 3 | Pneumothorax, Pleural Effusion, Support Devices | 0 |
| lung_wide | 5 | Lung Opacity, Edema, Consolidation, Pneumonia, Atelectasis | 5 |

*Note (discrepancy with the 054 pilot README):* the pilot and the yaml note say lung_wide has "4" hard constraints. That
is the number of **constrained children**. `hierarchy_admissible` applies one term per directed parent→child pair, and
Pneumonia has two parents (Lung Opacity, Consolidation), so lung_wide has **5** terms. This experiment records and
stratifies by **directed edges (5)**. The admissible count is 13 of 32 under either reading (the pilot's figure).

**Random control (spec §Clusters):**
- Eligible labels: ≥ 200 positive rows in the all-13-clean **conformal** split (calibration positives only).
  Counts: Pleural Other 151 (**ineligible**); all other 12 labels eligible (lowest eligible: Fracture 304).
  Eligible pool = the other 12 labels.
- For each designed cluster (yaml order: cardio_pair = position 0, lung_core 1, pleural_free 2, lung_wide 3),
  **20 draws**, draw i (0..19) gets global index `idx = 20·position + i`, RNG `numpy.random.default_rng(5000 + idx)`,
  `rng.choice(eligible_pool, size=m, replace=False)`, labels returned in canonical order. Subsets, not partitions;
  draws are independent across clusters (idx is global; see reading R3).
- Per draw recorded: labels, m, directed-edge count (hard constraints), admissible states, validation-occupied states,
  Gate 0 violation counts in each split. Hierarchy admissibility is applied within each draw automatically.

## 4. Score arms (each yields a region over 2^m joint states)

Common input: the unit's m logits, `log(p/(1−p))`, p clipped to [1e-6, 1−1e-6]; no standardisation. Target: joint state.
Spec bookkeeping (as pilot): state index `Σ y_l 2^l`, label 0 = LSB; regions are permuted to `conformal_common` state
order before `evaluate_joint`.

- **L0** (designed clusters only): multinomial logistic regression, `Linear(m, 2^m)`, softmax over states.
- **L1** (primary learned arm; designed and random): `Linear(m, 64) → ReLU → Linear(64, 2^m)`.
- **L2** (secondary; designed and random): identical to L1, input = all 13 logits (ALL_LABELS order).
- **Training (L0, L1, L2 — identical to the pilot's L1):** categorical cross-entropy over states; Adam, lr 1e-3,
  weight_decay 1e-4, batch 256; inner holdout = 20% of all-13-clean **validation subjects** (`default_rng(0)`
  permutation of sorted unique subjects, `round(0.2·5898) = 1,180` subjects), fixed across all seeds and units;
  training on the remaining 80%. Early stopping on holdout CE, patience 10, max 200 epochs (cap recorded if hit,
  not changed), best-holdout weights kept.
- **Seeds:** model seeds 0–4 (`torch.manual_seed(seed)` before init; `torch.Generator` seeded `seed` for minibatch order;
  `torch.set_num_threads(1)`). **Seed 0 is primary and equals the pilot's seed.**
- **Random draws:** L1 and L2 trained with seeds 0–4 (L0 is not run on draws).
- **Score and admissibility:** `s(x,y) = −log_softmax(net(x))[y]`; `s = +∞` on hierarchy-inadmissible states, applied
  after the network.
- **LP-SCP admissible (μ = ν = 1, as Exp047):** `lp_scp.fit_structure` on all-13-clean validation rows of the unit's
  labels (Σ, Hp, Cp), `lp_scp.state_scores(·, μ=1, ν=1)`, `+∞` on inadmissible states. Fitted once per unit, fixed
  across resplits.

## 5. Calibration rule, guard, references

- **Learned arms and LP-SCP (α = 0.10 only):** `k = ⌈(n+1)(1−α)⌉` (`conformal_common.conformal_rank`), threshold =
  k-th smallest calibration score of each calibration row's **true** state (no interpolation; `+∞` if k > n, which
  would be a stop). Prediction region = `{admissible y : s(x,y) ≤ threshold}` **guarded** (primary: ∪ argmin over
  admissible states) and **unguarded** (recorded). Admissibility is masked explicitly, not via `s ≤ threshold`.
- **References (classical, one convention):** every threshold is `k = ⌈(n+1)(1−α′)⌉` from `conformal_rank` with
  n = calibration-half size; joint sets are products of per-label sets, joint size = product of per-label sizes;
  **no guard** (empty sets allowed and counted):
  - `indep_per_label`: α per label (validity gap only, not a competitor);
  - `bonferroni`: α/m (`bonferrony.calibrate_per_label(…, "bonferroni")`); legacy reference;
  - `sidak`: 1−(1−α)^(1/m);
  - `maxrank`: `max_rank.calibrate_max_rank` (Timans et al. Algorithm 2, k_star);
  - `maxrank_adm`: product set ∩ hierarchy-admissible states (**primary comparator**);
  - `bonferroni_adm`: Bonferroni product ∩ admissible.
  The Exp053 β̂ recalibration is **not** used here (the spec expects classical Bonferroni/Šidák to over-cover and fail the
  coverage gate).
- **"+1" convention (Amendment 3, applied as specified for these classical references):** the calibration count n is the
  calibration-half size; the rank index is `⌈(n+1)(1−α′)⌉` everywhere; a test row or candidate state gets the plain
  indicator (coverage = true state in region), with no extra +1 applied on top. Same k for every method at the same α′.
  See reading R1 for the identity gate that tests this.

## 6. Resplits (Exp053 protocol, E8 / Amendment 3)

- **Frozen draw (reference row only, no inference):** calibration = conformal 13-clean (18,754), test = test 13-clean (18,877).
- **Headline resplits r = 0..99:** `rng = numpy.random.default_rng(r)`; `perm = rng.permutation(sorted unique pooled subjects)`;
  first `⌊S/2⌋ = 5,929` subjects = calibration half A, the rest (5,929) = test half B; rows follow subjects.
  Every calibration (thresholds, ranks, k) is recomputed on A; coverage/size are evaluated on all B rows.
- Paired deltas are within resplit. **CIs = 2.5/97.5 percentiles over the 100 resplits** of the paired differences.
- **Caveat (stated with every CI):** the CIs are conditional on the fixed pooled 37,631-row population and on the fixed
  trained scorers, and therefore **understate total uncertainty** (Exp053 caveat). Pooling conformal and test is a
  deviation from the frozen split roles, as in Exp053.

## 7. Identity gate (run first; stop and escalate on failure)

The Exp053 identity gate was defined at K = 13 and K = 1 in the β̂ framework, which this experiment does not use. The
classical analogue at which Bonferroni, Šidák, max-rank and independent per-label must coincide is **m = 1**
(single-label clusters; reading R1). For each of the 13 labels, on the frozen draw and on all 100 resplits:
- mismatch = fraction of test-half-B rows whose per-label prediction set differs between (bonferroni, maxrank),
  (sidak, maxrank), (indep, bonferroni);
- **pass iff every mismatch ≤ 0.5% of rows** (Exp053 tolerance) on the frozen draw and every resplit;
- also reported: calibration-side coverage at m = 1 for each method (expected ≈ 0.90).

## 8. Metrics (per arm × unit × seed × resplit; summarised over resplits)

- **Coverage:** joint coverage (true state in the region, all members); mean, SD, 2.5/97.5 percentiles over resplits;
  calibration-side coverage; threshold and k for the learned arms and LP-SCP.
- **Primary efficiency:** mean joint set size (empty = 0; product for product-set references); mean log₂(size)/m over
  non-empty rows; median size; fully-decisive rate (size = 1); **correct-decisive rate = fully-decisive rate × decisive
  accuracy** (the like-for-like accuracy metric; not raw decisive accuracy).
- **Secondary:** raw decisive accuracy; hierarchy-violation rate of fully-decisive output; empty-set rate.
- **Ratios:** paired deltas (arm − comparator) vs `maxrank_adm` (primary), `lpscp_adm` (guarded) and `bonferroni`
  (legacy), with 2.5/97.5 percentile CIs over resplits; resplit-mean ratios are shown for readability only.
- **Frozen-draw row:** each metric's frozen value and its percentile within the resplit distribution (not for inference).

## 9. Gates and comparison validity

- **Coverage gate:** an arm whose resplit-mean joint coverage is outside [0.895, 0.905] is labelled **coverage-unequal**;
  its sizes are reported but not compared. Bonferroni and Šidák are expected to fail; that is acceptable.
  A paired comparison is marked `comparison_valid` only when **both** arms pass.
- **Calibration-side coverage** is reported; it is diagnostic, not a gate.

## 10. Reproduction check (asserted)

On the frozen draw, the L1 seed-0 model for each designed cluster must reproduce the pilot's L1 row exactly:
threshold within 1e-6, best epoch and stop epoch equal, holdout CE within 1e-6 (pilot values: cardio_pair 1.168434,
best 86 / stop 96; lung_core 2.0926, 34 / 44; pleural_free 2.2503, 22 / 32; lung_wide 2.8915, 37 / 47; values read from
`results.csv`, not retyped). Mismatch → escalate.

## 11. Seeds

Model seeds 0–4 (scorers); holdout `default_rng(0)`; random draws `default_rng(5000 + idx)`; resplits `default_rng(r)`,
r = 0..99. Seed spread is reported as its own table (`seed_spread.csv`), **not** folded into the CIs.

## 12. Pre-registered decision rules (operational definitions recorded before the run)

Decision quantity for "smaller" = **mean joint set size** (spec primary); log₂(size)/m is reported alongside.
"Equal coverage" = both arms pass the coverage gate; the coverage difference is reported.

1. **L1 vs L0:** L1 "beats" L0 if the paired CI of (L1_s0 − L0_s0) mean size is entirely below 0 and both pass
   the gate. Otherwise the neural part adds nothing and L0 is the learned-score result.
2. **Headline (L1 vs max-rank ∩ admissible):** a smaller set at equal coverage with a paired CI excluding 0
   (CI entirely below 0, both gates passing) is the headline efficiency result, scoped to all-13-clean studies.
3. **Designed vs random:** designed clusters are "better" under L1 (or L2) if the designed resplit-mean log₂(size)/m is
   below the median of its own 20 draws (lower = more efficient). Otherwise say so plainly. Stratified: same
   directed-edge count, own 20 draws and all 80 draws. Descriptive percentiles only, no hypothesis test.
4. **L2 vs L1:** a significant gain for L2 (paired CI of L2_s0 − L1_s0 entirely below 0) is attributed to
   out-of-cluster information, not to clustering.
5. **Seed spread:** "comparable" = the seed range (max − min over seeds 0–4) of the resplit-mean delta
   (L1_s − max-rank ∩ admissible) is at least 50% of |seed-0 delta|. If comparable, the claim is downgraded.

## 13. Asserts and escalation triggers

**Asserts (abort):** Gate 0 (overlaps; hierarchy violations per split × unit); population counts
(18,570 / 18,754 / 18,877 rows; 5,898 / 5,926 / 5,932 subjects; pool 37,631 rows, 11,858 subjects); designed clusters
equal the yaml names and m; directed-edge counts match §3; every calibration true state admissible (Gate 0 consequence);
identity gate; pilot reproduction; `decisive rate × decisive accuracy` consistency (1e-12).

**Escalation (write `ESCALATION.md`, do not iterate):** any Gate 0 or identity-gate failure; any random draw with one
admissible state; any learned-arm seed (L0/L1/L2, designed or random) whose resplit-mean coverage is outside
[0.895, 0.905]; degenerate sets (resplit-mean median size = admissible count) or an infinite threshold; any NaN in a
reported metric; any reproduction mismatch; anything ambiguous. The run completes and records every trigger;
`run.py` writes `ESCALATION.md` automatically if any non-stop trigger fires.

**200-epoch cap:** recorded per scorer (`scorer_diagnostics.csv`), not an escalation by itself, and the cap is not changed.

## 14. Readings not pinned by the spec (recorded before any metric)

- **R1 — identity gate instance.** The spec gate is Exp053's (K = 13 and K = 1, β̂ framework). Here the classical
  equivalent is m = 1 with the same 0.5% tolerance (§7). This is the only analogue at which Bonferroni, Šidák and
  max-rank must coincide exactly, so it is the test of the k convention in §5. If Tom meant a different degenerate case,
  this is the item to revisit before results are interpreted.
- **R2 — L0 training.** The spec gives L0 as "multinomial logistic regression" without a training recipe. It is run as
  `Linear(m, 2^m)` under the identical learned-arm recipe (the spec's "Learned-arm rules" apply to all learned arms), with
  the same holdout and early stopping.
- **R3 — draw index across clusters.** `default_rng(5000 + idx)` with idx = 20·cluster-position + i. Using the same idx
  range for every cluster would give lung_core and pleural_free (both m = 3, same eligible pool) identical draws, which
  contradicts "drawn independently per cluster."
- **R4 — random-draw arms.** L1 and L2 (spec decision rule 3 names both), seeds 0–4, plus all references (the spec says
  "run L1 and the references"); L0 not run on draws.
- **R5 — hard-constraint count = directed parent→child edges** (per hierarchy term), not constrained children (see §3).
- **R6 — "all draws".** Stratified controls use (a) the designed cluster's own 20 draws and (b) all 80 draws with the same
  edge count.
- **R7 — "equal coverage" and "comparable"** are operationalised as in §12 (rules 2 and 5). Both are my readings.
- **R8 — references are classical** (Exp047 convention), not the Exp053 β̂ recalibration (§5).
- **R9 — LP-SCP structure population.** Σ/Hp/Cp fit on the all-13-clean validation rows. For this export, validation has
  no removed rows, so this equals the full validation set.
- **R10 — λ.** Not applicable: no structured Layer-1 score is part of this experiment.

## 15. Standing caveats (repeated in the results)

1. **Backbone:** one shared-backbone classifier inflates cross-label score correlation (EC/CM 0.936 → 0.997 vs
   independently trained models). This likely **overstates** any efficiency gain from joint calibration; it is not
   conservative.
2. **§3a population:** the scorers, LP-SCP structure, thresholds and every resplit are fit or calibrated on all-13-clean
   rows. The 22.6% of test with an uncertain label is not represented; coverage there is unknown.
3. **Occupied-state counts** (validation-occupied, admissible) are properties of the filtered population, not of how
   findings co-occur.
4. **Conformal controls coverage only;** efficiency is empirical. design A (LP-SCP-style label powerset with a learned base
   classifier) has **no novelty claim**.
5. **CIs** are conditional on the fixed pool and the fixed trained scorers (understated; see §6).

## 16. Reproduce

```
/usr/bin/python3 experiments/055_learned_score_full/run.py
```
(Python 3.9.6; numpy 1.24.3; pandas 2.0.3; torch 2.8.0; pyyaml 6.0.1; see `run_info.json`.)
Outputs in this folder: `identity_gates.csv`, `units.csv`, `resplit_metrics.csv` (per resplit, per unit, per arm, plus
the frozen row with `resplit = -1`), `scorer_diagnostics.csv`, `summary_by_arm.csv`, `paired_deltas.csv`,
`random_control.csv`, `seed_spread.csv`, `headline.csv`, `metrics.json`, `run_info.json`, `run_stdout.log`, and
`ESCALATION.md` if a trigger fires.

---

# Results

*(Appended after the run. Pre-registered text above is unchanged.)*

**Run:** `/usr/bin/python3 experiments/055_learned_score_full/run.py`, exit 0, 990 s, 2026-11-15. stdout in
`run_stdout.log`; `run_info.json` records HEAD `6a26ac8` (dirty tree), seeds, package versions and input/code hashes.
`run.py` hash matched the pre-registered `6f589bb4…`. **Scope on every table: all-13-clean studies, 77% of test.**

## Gates

| gate | result |
|---|---|
| Gate 0 — subject / dicom overlap, 3 splits (full frames) | **PASS**: 0 / 0 for all three pairs |
| Gate 0 — hierarchy violations of true labels, 84 units × 3 splits | **PASS**: 0 everywhere |
| Population (all-13-clean) | validation 18,570 / 5,898 subj; conformal 18,754 / 5,926; test 18,877 / 5,932 (of 24,394); pool 37,631 / 11,858 |
| Eligibility (calibration positives ≥ 200) | 12 eligible; **Pleural Other ineligible** (151) |
| Identity gate (m = 1, 13 labels, frozen + 100 resplits, tol 0.5%) | **PASS**: max mismatch 0.000% on every split; calibration-side coverage 0.9001 |
| Pilot reproduction (L1 seed 0, frozen draw, 4 clusters) | **PASS, exact**: thresholds 1.168434 / 2.092575 / 2.250284 / 2.891493; best/stop epochs 86/96, 34/44, 22/32, 37/47 |
| Coverage gate, designed learned arms (L0/L1/L2 × 5 seeds × 4 clusters) | **PASS**: all 60 resplit-mean coverages in [0.8989, 0.9007] |
| Coverage gate, random-draw learned arms | **FAIL, 2 draws** (see Escalation): `rand_cardio_pair_15` and `rand_cardio_pair_19` (same label set {Lung Lesion, Fracture}), L1 and L2, all 10 seed-arms, coverage 0.9557 |
| Hard-constraint count (directed edges) | cardio 1, lung_core 2, pleural_free 0, lung_wide 5 (asserted) |
| 200-epoch cap | 6 scorers hit it, **all L0**: cardio_pair seeds 0–3, lung_wide seeds 3–4. Recorded, not changed. |

## Headline table — designed clusters, seed 0 (scope: all-13-clean studies, 77% of test)

Coverage: resplit mean (SD over 100 resplits). Sizes: resplit mean. "Coverage-unequal" arms are reported but not compared.

| cluster (m) | arm | joint cov (SD) | mean joint size | log₂(size)/m | median | fully-decisive | correct-decisive | decisive acc | hier. viol. |
|---|---|---|---|---|---|---|---|---|---|
| cardio_pair (2) | **L1** | 0.9005 (0.0043) | **1.238** | 0.119 | 1 | 0.762 | 0.668 | 0.877 | 0 |
| | L0 | 0.9006 (0.0043) | 1.238 | 0.119 | 1 | 0.762 | 0.669 | 0.878 | 0 |
| | L2 | 0.9005 (0.0042) | 1.241 | 0.120 | 1 | 0.759 | 0.666 | 0.877 | 0 |
| | max-rank ∩ adm | 0.8997 (0.0050) | 1.816 | 0.328 | 1 | 0.570 | 0.478 | 0.839 | 0 |
| | LP-SCP adm | 0.9012 (0.0042) | 1.412 | 0.191 | 1 | 0.661 | 0.570 | 0.862 | 0 |
| | Bonferroni | 0.9421 (0.0032) | 2.551 | 0.525 | 2.88 | 0.449 | 0.401 | 0.893 | 0 — *coverage-unequal* |
| lung_core (3) | **L1** | 0.8992 (0.0042) | **1.990** | 0.281 | 2 | 0.348 | 0.317 | 0.910 | 0 |
| | L0 | 0.8992 (0.0041) | 1.991 | 0.281 | 2 | 0.348 | 0.317 | 0.909 | 0 |
| | L2 | 0.8993 (0.0045) | 1.985 | 0.281 | 2 | 0.346 | 0.315 | 0.911 | 0 |
| | max-rank ∩ adm | 0.8995 (0.0057) | 2.707 | 0.387 | 2.99 | 0.335 | 0.295 | 0.882 | 0 |
| | LP-SCP adm | 0.8997 (0.0051) | 2.705 | 0.390 | 2 | 0.296 | 0.261 | 0.884 | 0 |
| | Bonferroni | 0.9163 (0.0050) | 4.108 | 0.543 | 4 | 0.266 | 0.242 | 0.911 | 0 — *coverage-unequal* |
| pleural_free (3) | **L1** | 0.9005 (0.0059) | **2.007** | 0.269 | 2 | 0.417 | 0.390 | 0.934 | 0 |
| | L0 | 0.9005 (0.0058) | 2.008 | 0.272 | 2 | 0.403 | 0.379 | 0.940 | 0 |
| | L2 | 0.9004 (0.0056) | 2.014 | 0.271 | 2 | 0.413 | 0.387 | 0.936 | 0 |
| | max-rank ∩ adm | 0.9002 (0.0060) | 3.556 | 0.470 | 4 | 0.317 | 0.303 | 0.957 | 0 |
| | LP-SCP adm | 0.9005 (0.0066) | 3.135 | 0.425 | 2.79 | 0.340 | 0.322 | 0.948 | 0 |
| | Bonferroni | 0.9048 (0.0059) | 3.624 | 0.479 | 4 | 0.313 | 0.300 | 0.958 | 0 (in band) |
| lung_wide (5) | **L1** | 0.8991 (0.0043) | **2.828** | 0.248 | 3 | 0.290 | 0.269 | 0.926 | 0 |
| | L0 | 0.8990 (0.0047) | 2.840 | 0.252 | 3 | 0.269 | 0.251 | 0.931 | 0 |
| | L2 | 0.8991 (0.0043) | 2.834 | 0.253 | 3 | 0.257 | 0.240 | 0.933 | 0 |
| | max-rank ∩ adm | 0.8991 (0.0053) | 6.386 | 0.418 | 6 | 0.284 | 0.261 | 0.920 | 0 |
| | LP-SCP adm | 0.9000 (0.0055) | 5.366 | 0.373 | 4.26 | 0.289 | 0.267 | 0.925 | 0 |
| | Bonferroni | 0.9271 (0.0041) | 16.557 | 0.658 | 16 | 0.158 | 0.150 | 0.950 | 0 — *coverage-unequal* |

Calibration-side coverage is 0.9001 for every learned arm; LP-SCP adm is 0.9012 on cardio_pair and 0.9001 on the other three. Empty-set rate is 0 for every arm in the table.

## Paired deltas — mean joint set size (L1 seed 0; 100 resplits; CI = 2.5/97.5 percentile)

Scope: all-13-clean studies, 77% of test. Each comparison below has both arms inside the coverage gate.

| cluster | L1 − max-rank ∩ adm | L1 − LP-SCP adm | L1 − L0 | L2 − L1 | L1 − Bonferroni |
|---|---|---|---|---|---|
| cardio_pair | **−0.578** [−0.629, −0.541] | **−0.174** [−0.202, −0.151] | +0.001 [−0.002, 0.003] | +0.002 [−0.006, 0.009] | not compared (coverage-unequal) |
| lung_core | **−0.717** [−0.770, −0.660] | **−0.716** [−0.771, −0.653] | −0.001 [−0.011, 0.010] | −0.005 [−0.017, 0.008] | not compared (coverage-unequal) |
| pleural_free | **−1.548** [−1.625, −1.480] | **−1.128** [−1.199, −1.050] | −0.000 [−0.017, 0.019] | +0.007 [−0.017, 0.024] | −1.616 [−1.694, −1.546] |
| lung_wide | **−3.558** [−3.692, −3.426] | **−2.538** [−2.673, −2.399] | −0.012 [−0.038, 0.012] | +0.006 [−0.021, 0.035] | not compared (coverage-unequal) |

Paired correct-decisive-rate deltas, L1 seed 0 (scope as above):

| cluster | − max-rank ∩ adm | − LP-SCP adm | − L0 |
|---|---|---|---|
| cardio_pair | +0.190 [+0.172, +0.212] | +0.099 [+0.086, +0.116] | −0.001 [−0.003, +0.001] |
| lung_core | +0.022 [+0.015, +0.027] | +0.056 [+0.045, +0.066] | +0.001 [−0.003, +0.004] |
| pleural_free | +0.086 [+0.080, +0.092] | +0.067 [+0.062, +0.073] | −0.002 [−0.006, +0.001] |
| lung_wide | +0.008 [−0.000, +0.015] | +0.002 [−0.007, +0.009] | −0.004 [−0.007, −0.001] |

## Frozen-draw reference row (not for inference; scope as above)

| cluster | arm | frozen cov | frozen size | size percentile in resplit distribution | threshold (frozen) |
|---|---|---|---|---|---|
| cardio_pair | L1 | 0.9017 | 1.237 | 0.51 | 1.1684 (= pilot) |
| lung_core | L1 | 0.9036 | 1.993 | 0.54 | 2.0926 (= pilot) |
| pleural_free | L1 | 0.9023 | 2.016 | 0.65 | 2.2503 (= pilot) |
| lung_wide | L1 | 0.9023 | 2.834 | 0.56 | 2.8915 (= pilot) |
| cardio_pair | max-rank ∩ adm | 0.9019 | 1.798 | 0.12 | — |
| lung_wide | max-rank ∩ adm | 0.9010 | 6.302 | 0.16 | — |

## Seed spread — resplit-mean mean joint size, seeds 0–4 (scope as above)

| cluster | family | range over seeds (size) | range of (arm − max-rank ∩ adm) over seeds | seed-0 delta | range / \|seed-0 delta\| |
|---|---|---|---|---|---|
| cardio_pair | L0 | 0.0004 | 0.0004 | −0.579 | 0.07% |
| cardio_pair | L1 | 0.0006 | 0.0006 | −0.578 | 0.10% |
| cardio_pair | L2 | 0.0041 | 0.0041 | −0.576 | 0.71% |
| lung_core | L1 | 0.0031 | 0.0031 | −0.717 | 0.43% |
| pleural_free | L1 | 0.0064 | 0.0064 | −1.548 | 0.41% |
| lung_wide | L1 | 0.0122 | 0.0122 | −3.558 | 0.34% |

Per the pre-registered rule 5 the seed range is far below the L1 − max-rank delta, so **the claim is not downgraded on seed grounds.** Seed spread is not in the CIs.

## Designed vs random control (L1 seed 0; log₂(size)/m, lower = more efficient)

Scope: all-13-clean studies, 77% of test. The table is the pre-registered comparison (all 20 own draws). A second
column excludes coverage-unequal draws, per the coverage-gate rule (post-run descriptive, not a new comparison).

Fractions are the share of draws with a **larger** log₂/m than the designed cluster (bigger sets), i.e. the share the designed
cluster is more efficient than. Higher = designed is more efficient.

| designed cluster | designed | own-draw median (all 20) | fraction of own draws with larger log₂/m than designed (all 20) | same, gate-passing draws only | same-edge stratum: fraction (own / all 80 draws) |
|---|---|---|---|---|---|
| cardio_pair | 0.119 | 0.220 | 0.70 | 0.78 (18 draws; gate-pass median 0.230) | 4 own, 22 all — 0.75 / 0.86 |
| lung_core | 0.281 | 0.203 | 0.15 | 0.15 (20 draws) | 0 own, 3 all — n/a / 1.00 (n=3) |
| pleural_free | 0.269 | 0.275 | 0.55 | 0.55 (20 draws) | 13 own, 53 all — 0.62 / 0.40 |
| lung_wide | 0.248 | 0.291 | 0.75 | 0.75 (20 draws) | 1 own, 1 all — 0.00 / 0.00 |

Reading (plain): **under L1 and L2 the designed clusters are not reliably more efficient than random same-m subsets.**
cardio_pair and lung_wide beat the median of their own draws; pleural_free (no constraints) is near the median; lung_core
is *less* efficient than 17 of its 20 random draws (designed 0.281 vs own-draw median 0.203). The lung_core own pool has no
2-edge draw (its draws have 0 or 1 edges), and the three random draws in the whole 80 that share its 2-edge count are all
more efficient than the designed cluster (fraction 1.00, n = 3, too small to read). So the lung_core deficit against its own
pool is largely an edge-count and label-prevalence effect. The own-pool comparison is confounded; see Anomalies.

**The learned-score gain is not clustering-specific.** L1 seed 0 is smaller than max-rank ∩ adm on **78 of 78** valid random
subsets (every paired CI below 0; median mean-size delta −2.29) and smaller than LP-SCP adm on **78 of 78** (median −1.49). The
two random draws with invalid coverage are excluded from these 78 by the gate.

## Interpretation under the pre-registered decision rules

1. **L1 vs L0:** no paired CI excludes 0 on any designed cluster (§paired table). Per rule 1 **L1 does not beat L0**; the
   neural component adds nothing measurable, and **L0 is the learned-score result.** The L0 result carries the 200-cap caveat
   (6 capped L0 scorers).
2. **Headline (L1 vs max-rank ∩ adm):** smaller set at equal coverage with paired CI excluding 0 on all four designed clusters
   (both arms inside [0.895, 0.905]; coverage differences ≤ 0.0008). **Met.** L0 gives the same result at seed 0: mean-size delta vs
   max-rank ∩ adm −0.579 [−0.629, −0.542] (cardio), −0.716 [−0.767, −0.665] (lung_core), −1.548 [−1.621, −1.483] (pleural_free),
   −3.546 [−3.688, −3.416] (lung_wide), all CIs below 0. Scoped to all-13-clean studies; subject to the caveats below.
3. **Designed vs random:** not better in general (see control section). Stated plainly: this run does **not** show that the
   designed clusterings are more efficient than random same-m subsets under L1/L2. It does show the learned-score advantage over
   the references holds on random subsets too.
4. **L2 vs L1:** no gain on any cluster (CIs include 0). Nothing to attribute to out-of-cluster information.
5. **Seed spread:** far below the L1 − max-rank delta; not downgraded on this rule.

## Anomalies and things Tom must see

- **Coverage-gate failure on two random draws (escalated).** `rand_cardio_pair_15` and `rand_cardio_pair_19` are both the label
  set {Lung Lesion, Fracture}. For these, L1 and L2 (all 5 seeds) return a single state (size 1.000) and cover 0.9557. The all-negative
  state is 95.6% of the all-13-clean conformal rows (95.5% of test; post-run diagnostic on the frozen CSVs). The learned score puts
  p ≥ 0.9 on that state for essentially every row, so the 90% calibration quantile falls inside the atom at 0.956: calibration-side
  coverage is also 0.9565. This is an atom, not a bug and not invalid (coverage ≥ 0.90), but it fails the two-sided gate and the
  learned set size of 1.0 is over-coverage, not efficiency. Max-rank ∩ adm and Bonferroni on the same draw are at 0.9007 and 0.9034.
- **Duplicate draws.** The 80 random draws contain 72 unique label sets (e.g. {Lung Lesion, Fracture} ×2, {Lung Opacity, Edema} ×3). The
  pre-registered procedure does not deduplicate; duplicates are counted as separate draws, which weights those sets twice.
- **Random draws are not matched on prevalence or difficulty.** Draws frequently contain Lung Lesion (2.8% positive), Fracture (1.6%)
  and other rare labels, which produce small sets regardless of structure. The pre-registered control is "same m, random labels"; it
  does not control for label difficulty, so the own-pool comparison above is confounded (the Exp053 difficulty control is the analogue
  that was run there and is not run here).
- **Pilot table discrepancy.** The 054 pilot lists lung_wide with 4 hard constraints; the directed-edge count is 5 (Pneumonia has two
  parents). Admissible count (13 of 32) is the same under either reading.
- **L0 non-convergence.** Six L0 runs hit the 200-epoch cap (cardio seeds 0–3; lung_wide seeds 3–4). The cap was not changed. L0 and L1
  sizes agree within 0.013 on every cluster (gaps −0.0006, 0.0011, 0.0001, 0.0124), so the cap does not change the reading, but the
  L0 numbers are from capped runs.
- **Holdout CE ordering is not uniform.** L1 has the lowest holdout CE on cardio, lung_core and lung_wide; L0 is lowest on pleural_free
  (0.8191 vs L1 0.8216, L2 0.8254). Not investigated; no decision depends on it.
- **Bonferroni** over-covers on cardio (0.942), lung_core (0.916) and lung_wide (0.927) and is coverage-unequal there (as expected); on
  pleural_free it is in band (0.905) and the comparison is valid.
- **LP-SCP adm, cardio:** frozen-draw coverage 0.9078 vs resplit mean 0.9012; a single draw, not gated, reported as is.
- **Gate 0 and identity pass; no NaN in any reported metric; no infinite threshold; no degenerate set (resplit-mean median = admissible
  count) on any designed unit.**

## Caveats that must accompany every efficiency number

1. **Shared backbone.** All probabilities come from one shared-backbone classifier, which inflates cross-label score correlation
   (EC/CM 0.936 → 0.997). This likely **overstates** every efficiency gain above; it is not conservative.
2. **§3a population.** Scorers, LP-SCP structure, thresholds and all resplits are fit or calibrated on all-13-clean rows. The 22.6%
   of test with an uncertain label is not represented.
3. **Occupied-state counts** (validation-occupied = admissible for the designed clusters: 3, 5, 8, 13) are properties of the filtered
   population, not of how findings co-occur.
4. **Conformal controls coverage only;** the efficiency gains are empirical, not guaranteed. Design A is LP-SCP-style label powerset with
   a learned base classifier; **no novelty claim.**
5. **CIs are conditional** on the fixed pool and the fixed trained scorers, so they understate total uncertainty.
