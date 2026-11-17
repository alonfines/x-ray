# Exp056 — three-cluster partition of all 13 labels, L1 per cluster, coverage of the union (PRE-REGISTRATION)

**Scope tag (every table and caption):** all-13-clean studies, 77% of test (18,877 of 24,394).
At m = 13 the per-member exclusion rule of SPEC §3a is the same as all-13-clean, so the population is the
natural one. The 22.6% of test with an uncertain label is not represented; coverage there is unknown.

**Spec:** `experiments/056_three_cluster_union/SPEC.md` (Tom Hagan, 2026-11-16; Alon approved the run the same day).
Everything the spec does not state follows Exp055 (`experiments/055_learned_score_full/README.md`, `run.py`).

**One-line restatement.** Partition the 13 labels into three clusters (P_H: 2/6/5, hierarchy-closed, fit on no data),
train an L1 joint-state scorer per cluster, combine the three cluster regions as a product, and measure how often the
true 13-label vector is covered (arms A, B, B′, C, guarded and unguarded), against 13-label unclustered references,
a learned 13-label scorer (arm D), and 20 random 2/6/5 partitions, on 100 subject-level 50/50 resplits of the 37,631-row
pool.

**Git HEAD at pre-registration:** `b85b288c0ef354c452e352b81af4c729d0747580` (branch main; working tree dirty: this folder
and other untracked files; nothing committed).
**`run.py` sha256 at pre-registration (original, pre-escalation):** `383566696a93b7fb2fa70420bd3538a397364969ab0964857aa73907b8c6970e`.
**`run.py` sha256 for the amended run (Amendment 1, the run reported below):** `97a4c90c940f8ec487fdec5f3f46d00555570b447051af3d7eb939d5fd5c8660`.
The amendment changes only the training call (A1.1) and the reproduction diagnostic record (A1.3). Exp055 `run.py` sha256 (imported, unedited): `6f589bb4911e4732dd87988755a7c0f58d4a9908984803a39141c75a61647448`.
**Pre-registration time:** 2026-11-16, before `run.py` was executed in full mode. The only executions before this point:
(a) `run.py --prereg` (label columns and ids only, no probabilities, no metrics; output `prereg_labels.json`), and
(b) an import of `run.py` to list the random draws (label-only). No frozen-split probability was read into a metric.

---

## 1. Inputs (frozen; no inference re-run)

| role | file (under `results/outputs/mimic/`) | sha256 (full) |
|---|---|---|
| validation (scorer training, inner holdout) | `densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_validation.csv` | `e2c7dfd3087154406d5fd79ae0284dad883926092538d80e854b13b2ed3b4119` |
| conformal (calibration half of the frozen draw) | `…_conformal.csv` | `cce3677863ab2a24a673339d7ab71ce1ef8a27c0e88ead9c7f9d4625bbaa59db` |
| test (evaluation half of the frozen draw) | `…_test.csv` | `ab42e9a895e0b1c31f758bd420d935607ea216a7968472e089a597309ff5d3f5` |

Full validation hash: `e2c7dfd3087154406d5fd79ae0284dad883926092538d80e854b13b2ed3b4119`.
Exp055 reference files read by the reproduction check (not retyped): `experiments/055_learned_score_full/resplit_metrics.csv`,
`experiments/055_learned_score_full/scorer_diagnostics.csv`.

**Reused, unedited:** `conformal_common.py` (loading, Gate 0, `GroupData`, `prediction_sets`, `conformal_rank`,
`order_statistic`, `hierarchy_admissible`, `enumerate_states`, `state_indices`), `conformal_methods/bonferrony.py`
(`calibrate_per_label`, corrections none / bonferroni / sidak), `conformal_methods/max_rank.py` (`calibrate_max_rank`,
Timans Alg. 2 on a 3-column score matrix), `conformal_methods/mlp_score.py` (`train_scorer`, `neg_log_softmax_spec`,
`logit`, `spec_weights`), `data/hierarchy.py` (`CHEXPERT_HIERARCHY`).
**Not reused:** `conformal_common.load_clusters` (asserts m ≤ 5, so cannot load C2 with m = 6). The P_H labels are
written in `run.py` from SPEC §1 instead; `configs/clusters/exp046.yaml` is not used.
**Re-written here (mirror Exp055, not imported from the finished script):** `all13_clean`, `hier_violations`, `subset`,
`product_region`, `refs_eval` (13-label version), `identity_gate`, threshold/guard logic.

## 2. Population and Gate 0

- Population: all-13-clean rows in every split (every true label in {0,1}, every prob finite), as Exp055.
  Label-only counts (prereg): validation 18,570 / 5,898 subj; conformal 18,754 / 5,926; test 18,877 / 5,932 (of 24,394).
  Pool (conformal ∪ test) = **37,631 rows, 11,858 subjects** (asserted in `run.py`).
- **Gate 0 (stop and escalate on failure):** (1) zero subject and zero dicom overlap across the three frozen CSVs
  (`cc.gate0_check`, full frames); (2) zero true-label hierarchy violations in every split, for every cluster of every
  partition (P_H and the 20 random partitions, within-cluster edges), and the full 13-label count is recorded.
  Label-only prereg check: **conformal true states, all-13-clean, full 13-label hierarchy violations = 0.**

## 3. Partitions (fit on no data)

**Designed, primary: P_H (hierarchy-closed, canonical label order inside each cluster).**

| cluster | m | labels | directed edges within cluster | admissible states | modal-state mass (conformal, all-13-clean) |
|---|---|---|---|---|---|
| C1 cardiac | 2 | Enlarged Cardiomediastinum, Cardiomegaly | 1 | 3 | 0.7924 |
| C2 parenchymal | 6 | Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis | 6 | 25 | 0.6033 |
| C3 pleural/other | 5 | Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices | 0 | 32 | 0.5781 |

Cross-cluster directed edges for P_H: **0** (every hierarchy edge is inside one cluster, so the product of
admissible cluster regions is admissible under the full 13-label hierarchy). Full 13-label admissible states: **2,400 of
8,192** = 3 × 25 × 32 (factorisation asserted, §4).

**Random control: 20 partitions, size profile 2/6/5.**
- Partition i (0..19): `np.random.default_rng(5600 + i).permutation(ALL_LABELS)` in canonical order. Positions 0–1 → R1,
  2–7 → R2, 8–12 → R3; labels within each cluster in canonical order.
- Duplicate rule: a draw equal (as an unordered partition) to P_H or an earlier draw is redrawn with sub-seed
  `5600 + i + 1000·j`, j = attempt index (reading R-S1 below). **Duplicate redraws: 0** (all 20 draws accepted on attempt 1).
- Cross-cluster hierarchy edges are not applied for any arm (SPEC §1). Cross-cluster directed-edge counts per random partition:

| draw | seed | R1 (2) | R2 (6) | R3 (5) | cross-cluster edges |
|---|---|---|---|---|---|
| R00 | 5600 | Pneumothorax, Fracture | EC, LO, Consolidation, Pneumonia, Pleural Effusion, Support Devices | CM, Lung Lesion, Edema, Atelectasis, Pleural Other | 4 |
| R01 | 5601 | Lung Lesion, Pleural Effusion | EC, CM, Edema, Atelectasis, Pleural Other, Fracture | LO, Consolidation, Pneumonia, Pneumothorax, Support Devices | 3 |
| R02 | 5602 | Atelectasis, Pleural Other | LO, Consolidation, Pneumonia, Pleural Effusion, Fracture, Support Devices | EC, CM, Lung Lesion, Edema, Pneumothorax | 3 |
| R03 | 5603 | Lung Lesion, Fracture | EC, Edema, Pneumonia, Atelectasis, Pleural Other, Support Devices | CM, LO, Consolidation, Pneumothorax, Pleural Effusion | 6 |
| R04 | 5604 | CM, Pneumothorax | EC, LO, Edema, Consolidation, Fracture, Support Devices | Lung Lesion, Pneumonia, Atelectasis, Pleural Effusion, Pleural Other | 5 |
| R05 | 5605 | Pneumothorax, Pleural Effusion | EC, Consolidation, Pneumonia, Atelectasis, Pleural Other, Support Devices | CM, LO, Lung Lesion, Edema, Fracture | 4 |
| R06 | 5606 | EC, Fracture | CM, LO, Lung Lesion, Pneumothorax, Pleural Effusion, Support Devices | Edema, Consolidation, Pneumonia, Atelectasis, Pleural Other | 5 |
| R07 | 5607 | Edema, Support Devices | Lung Lesion, Consolidation, Pneumonia, Pleural Effusion, Pleural Other, Fracture | EC, CM, LO, Atelectasis, Pneumothorax | 4 |
| R08 | 5608 | Pleural Effusion, Support Devices | Edema, Pneumonia, Atelectasis, Pneumothorax, Pleural Other, Fracture | EC, CM, LO, Lung Lesion, Consolidation | 4 |
| R09 | 5609 | Pneumonia, Fracture | LO, Edema, Consolidation, Atelectasis, Pneumothorax, Pleural Effusion | EC, CM, Lung Lesion, Pleural Other, Support Devices | 3 |
| R10 | 5610 | EC, CM | Lung Lesion, Consolidation, Pneumothorax, Pleural Effusion, Pleural Other, Support Devices | LO, Edema, Pneumonia, Atelectasis, Fracture | 3 |
| R11 | 5611 | Consolidation, Pleural Effusion | LO, Pneumonia, Atelectasis, Pneumothorax, Fracture, Support Devices | EC, CM, Lung Lesion, Edema, Pleural Other | 4 |
| R12 | 5612 | LO, Consolidation | EC, CM, Lung Lesion, Pneumothorax, Pleural Other, Support Devices | Edema, Pneumonia, Atelectasis, Pleural Effusion, Fracture | 5 |
| R13 | 5613 | CM, Support Devices | EC, LO, Pneumonia, Pneumothorax, Pleural Other, Fracture | Lung Lesion, Edema, Consolidation, Atelectasis, Pleural Effusion | 6 |
| R14 | 5614 | EC, Pneumonia | CM, Lung Lesion, Edema, Atelectasis, Pleural Effusion, Pleural Other | LO, Consolidation, Pneumothorax, Fracture, Support Devices | 6 |
| R15 | 5615 | Edema, Pleural Effusion | Consolidation, Atelectasis, Pneumothorax, Pleural Other, Fracture, Support Devices | EC, CM, LO, Lung Lesion, Pneumonia | 4 |
| R16 | 5616 | Pneumothorax, Pleural Other | LO, Lung Lesion, Edema, Atelectasis, Pleural Effusion, Fracture | EC, CM, Consolidation, Pneumonia, Support Devices | 2 |
| R17 | 5617 | EC, LO | Lung Lesion, Edema, Pneumothorax, Pleural Other, Fracture, Support Devices | CM, Consolidation, Pneumonia, Atelectasis, Pleural Effusion | 6 |
| R18 | 5618 | Pleural Effusion, Support Devices | Lung Lesion, Consolidation, Atelectasis, Pneumothorax, Pleural Other, Fracture | EC, CM, LO, Edema, Pneumonia | 4 |
| R19 | 5619 | EC, Edema | LO, Lung Lesion, Consolidation, Pleural Effusion, Pleural Other, Fracture | CM, Pneumonia, Atelectasis, Pneumothorax, Support Devices | 5 |

Within-cluster directed edges are applied as admissibility for every cluster (as for P_H). The random draws are
independent of label prevalence, so they are not matched on difficulty (same caveat as Exp055).

**Pleural Other (151 calibration positives)** is included in P_H (the ≥200 floor does not apply). Conditional coverage on
PO-positive, Fracture-positive (304) and Lung-Lesion-positive (523) rows is a diagnostic, not gated.

## 4. Scorers

- **L1 per cluster**, Exp055 recipe: `Linear(m,64)→ReLU→Linear(m→2^m)` on the cluster's m logits (`log(p/(1−p))`,
  p clipped to [1e-6, 1−1e-6]). Trained with `conformal_methods/mlp_score.train_scorer` (Lightning port, per the spec).
  Adam lr 1e-3, weight decay 1e-4, batch 256, inner holdout = 20% of all-13-clean validation subjects
  (`default_rng(0)` permutation of sorted unique subjects; 1,180 subjects), early stopping on holdout CE, patience 10,
  max 200 epochs (cap recorded, not changed), best weights kept.
  Score `s(x,y) = −log_softmax(net(x))[y]`; `+∞` on hierarchy-inadmissible states, applied after the network.
- **P_H: seeds 0–4** for all three clusters (seed 0 primary). **Random partitions: seed 0 only.** No L0, no L2.
- **Arm D (L1_full13):** same recipe on all 13 logits, 8,192 output states, `+∞` on states inadmissible under the full
  hierarchy; seed 0 only; one quantile at α = 0.10. Scores are stored as float32 (spec: "float32") for the per-row size count.
- **Reproduction check (SPEC §2, asserted):** P_H C1 = `cardio_pair` of Exp055. Frozen draw (calibration = conformal,
  test = test), L1 seed 0: threshold within 1e-6 of Exp055 `resplit_metrics.csv` (L1, resplit −1, seed 0);
  best epoch and stop epoch equal to `scorer_diagnostics.csv`; final holdout CE within 1e-6. Mismatch = stop.

## 5. Arms for combining clusters (all at α = 0.10, all thresholds from the calibration half A, k = ⌈(n+1)(1−α′)⌉)

In every arm the 13-label set is R1 × R2 × R3 (product of cluster regions); size = product of cluster region sizes; a row
is covered iff the true state of every cluster is in its region (`joint_coverage`).

- **A — naive**, α_c = 0.10 per cluster (`bonferrony.calibrate_per_label(…, "none")`).
- **B — Bonferroni across clusters**, α_c = 0.10/3 (`…, "bonferroni"`). **B′ — Šidák**, α_c = 1 − 0.9^(1/3) (`…, "sidak"`).
- **C — max-rank across clusters (primary jointly-valid arm).** The n_A × 3 matrix of true-state L1 scores on half A is
  passed to `max_rank.calibrate_max_rank` (Timans Alg. 2, clusters as columns): row-wise max of column ranks → k* → column
  order statistics at k*.
- **D — L1_full13**, one 13-label quantile (§4).
- **Unguarded variant = primary.** **Guarded variant = secondary:** per-cluster argmin of the (admissible) score added **only
  for rows whose cluster region is empty** (SPEC §3). For D the same rule applies to its single region.

## 6. Unclustered references on the 13-label vector (Exp055 §5 convention, no guard)

`indep_per_label` (α per label), `bonferroni` (α/13), `sidak`, `maxrank` (13 columns), **`maxrank_adm`** = product set
∩ full-hierarchy admissible (**primary classical comparator**), `bonferroni_adm`. All with k = ⌈(n+1)(1−α′)⌉ via `conformal_rank`.

## 7. Identity gate (SPEC §5)

Exp055 m = 1 gate rerun as is: 13 single-label clusters; Bonferroni, Šidák, max-rank and independent per-label must
coincide; mismatch ≤ 0.5% of rows on the frozen draw and every one of the 100 resplits. Failure = stop.

## 8. Resplits and frozen row (SPEC §0, Exp055 §6)

- Frozen draw (reference row only): calibration = conformal 13-clean (18,754), test = test 13-clean (18,877).
- Resplits r = 0..99: `default_rng(r)`; permutation of sorted unique pooled subjects; first 5,929 subjects = half A
  (calibration), rest = half B (evaluation); rows follow subjects. Every threshold, rank and k is recomputed on A; coverage
  and size on all B rows.
- **CIs** = 2.5/97.5 percentiles over the 100 resplits of within-resplit paired differences. Caveat (stated with every CI):
  conditional on the fixed 37,631-row pool and the fixed trained scorers; understates total uncertainty.

## 9. Metrics

- **Coverage:** 13-label joint coverage; per-cluster coverage; product of per-cluster coverages; pairwise φ between the three
  cluster miss indicators; calibration-side coverage; thresholds and k / k*.
- **Primary efficiency:** mean log₂(size)/13 over non-empty rows, with the empty-set rate beside it. Also mean joint size
  (empty = 0), median size, fully-decisive rate, correct-decisive rate (decisive × correct), decisive accuracy,
  hierarchy-violation rate of decisive output. Same per cluster with log₂(size)/m_c.
- **Diagnostics:** 13-label coverage on rows positive for PO, Fracture, Lung Lesion, and on rows with ≥ 1 positive (n reported);
  guard gap = guarded − unguarded coverage per cluster and on the vector; modal-state mass per cluster on conformal (§3, recorded
  before the run).

## 10. Gates (unguarded variants; SPEC §7)

- **A:** each of the three per-cluster coverages in [0.895, 0.905] (resplit mean). 13-label coverage reported, not gated.
- **B / B′:** 13-label coverage ≥ 0.90 (resplit mean) = valid; coverage-unequal sizes reported but not compared.
- **C, D, maxrank_adm:** 13-label coverage in [0.895, 0.905].
- A comparison is `comparison_valid` only if both arms pass. A random partition that fails a gated arm is recorded and
  excluded from that comparison.

## 11. Pre-registered decision rules (SPEC §8), with operational readings

1. **Alon's question (descriptive).** Arm A's 13-label coverage (resplit mean, CI) next to 0.70, 0.729 and 0.90. State plainly
   whether "90% per cluster" gives 90% on the vector.
2. **Cost of guaranteeing the vector.** Arm B's coverage excess over 0.90 and its size ratio vs C (descriptive).
3. **Headline: C vs maxrank_adm.** "Clustered L1 beats unclustered max-rank" only if the paired CI of Δ mean log₂(size)/13
   (C − maxrank_adm) is entirely < 0 and both arms pass. Report Δ empty-set rate; flag if |Δ empty| > 1 pp.
4. **C vs D.** CI < 0 → clustering beats one 13-label learned scorer; CI > 0 → clustering costs efficiency. Stated either way.
5. **Designed vs random (arm C, seed 0, unguarded).** P_H's resplit-mean log₂(size)/13 against the 20 random partitions.
   "Beats median" if below the median. "More efficient than random" only if below ≥ 18 of 20. Also arm A 13-label coverage
   across the 20 random partitions.
6. **Seed spread (P_H, arm C):** Exp055 rule 5 unchanged: comparable if the range over seeds 0–4 of the resplit-mean delta
   (C_s − maxrank_adm) is ≥ 50% of |seed-0 delta|; if comparable, the claim is downgraded.

## 12. Stop and escalation (SPEC §9; `run.py` writes `ESCALATION.md` on any stop)

**Hard stops:** Gate 0 failure; identity-gate failure; reproduction mismatch; any true calibration state inadmissible
(within-cluster for any cluster, or full-hierarchy for D); infinite threshold; NaN in a reported core metric.
**Recorded and escalated at the end:** any gated arm outside its band on P_H; guard gap > 2 pp on any designed cluster;
D fails to train or is degenerate (median size = admissible count = 2,400); duplicate redraws > 5; anything ambiguous.
**Not an escalation by itself:** 200-epoch cap hits (recorded per scorer; cap not changed).

## 13. Readings not pinned down by the spec (recorded before the run)

- **R-S1 — duplicate redraw seed.** SPEC says "increment a sub-seed, record it". Implemented as `5600 + i + 1000·j` (j =
  attempt index), so no redraw collides with another partition's base seed. Attempts are logged in `random_partition_draws.csv`.
  Zero redraws occurred.
- **R-S2 — guard rule (equivalent, Amendment 1 A1.5).** Exp055's always-union and SPEC §3's only-when-empty guards give the
  **same region**. The argmin s* of the admissible scores satisfies s* ≤ t whenever any admissible state has s ≤ t, so a
  non-empty region {adm, s ≤ t} already contains the argmin. The two guards differ only on empty rows, where both add the argmin.
  Equivalent, not a deviation.
- **R-S3 — "primary seed".** Seed 0 is the primary seed for P_H rules (as in Exp055); seeds 0–4 are reported in full.
- **R-S4 — arm A and B use the L1 seed-0 scorers** for their headline rows; other seeds reported.
- **R-S5 — P_H within-cluster edge count.** C2 has 6 directed edges (Lung Opacity → Lung Lesion, Edema, Consolidation,
  Atelectasis; Pneumonia → Lung Opacity, Consolidation). Admissible states of C2 = 25.
- **R-S6 — `load_clusters` cap.** `conformal_common.load_clusters` enforces m ≤ 5 and cannot load C2 (m = 6). P_H is written in
  `run.py` from SPEC §1. No shared module is edited.
- **R-S7 — arm D storage.** Scores in float32 (spec), so ties at float32 precision are resolved as float32 comparisons.
- **R-S8 — training loop (superseded by Amendment 1, A1.1).** The Lightning port is not used for any reported number; see Amendment 1.
  Original note: `mlp_score.py` is documented in its own docstring as *not bit-identical* to Exp055's training
  loop (Lightning minibatch RNG differs). The spec nevertheless requires the reproduction check against Exp055 values to
  match to 1e-6. See §14 of the results; if the check fails, the run stops (SPEC §9).

## 14. Caveats (carried into the results)

1. **Shared backbone.** One shared-backbone model inflates cross-label correlation (EC/CM 0.936 → 0.997; Consolidation/Pneumonia
   0.832 → 0.971). Likely overstates efficiency gains from joint calibration; not conservative.
2. **Population.** Scorers, thresholds and all resplits use the all-13-clean rows only; 22.6% of test is not represented.
3. **Random partitions are not matched on prevalence or difficulty.** Several draws contain Lung Lesion (2.8%), Fracture (1.6%),
   Pleural Other (0.8%) and other rare labels, which give small sets regardless of structure.
4. **Conformal controls coverage only;** efficiency is empirical. Design: LP-SCP-style label powerset with a learned base
   classifier; no novelty claim.
5. **CIs are conditional** on the fixed pool and fixed trained scorers; understated.
6. **C2 is m = 6**, outside the locked 2–5 range in RESEARCH_DIRECTION (SPEC §10).
7. **Cross-cluster hierarchy is not enforced for random partitions** (SPEC §1); their product sets can contain states that
   violate cross-cluster edges, and that is reported as hierarchy-violation rate on decisive output.
8. **Arm A has no guarantee by construction** (SPEC §3).

## 15. Reproduce

```
/usr/bin/python3 experiments/056_three_cluster_union/run.py
```
(Python 3.9.6; numpy 1.24.3; pandas 2.0.3; torch 2.8.0; lightning 2.0.9; see `run_info.json`.)
Label-only pre-registration: `/usr/bin/python3 experiments/056_three_cluster_union/run.py --prereg` → `prereg_labels.json`.
Outputs: `identity_gates.csv`, `reproduction_check.json`, `resplit_metrics_vector.csv`, `resplit_metrics_cluster.csv`,
`random_partition_draws.csv`, `summary_vector.csv`, `summary_cluster.csv`, `gates.csv`, `guard_gaps.csv`, `metrics.json`,
`run_info.json`, `run_stdout.log`, and `ESCALATION.md` if a stop or trigger fires.

---

## Amendment 1 (post-escalation, before any arm metric)

Tom Hagan's ruling, after the reproduction stop (`ESCALATION.md`): option 1, resume with this amendment, no other spec changes.

- **A1.1 Training loop.** Every learned scorer (P_H C1–C3 seeds 0–4; all 20 random partitions, seed 0; arm D L1_full13, seed 0)
  is trained with Exp055's `train_scorer`, imported unedited from `experiments/055_learned_score_full/run.py` (sha256
  `6f589bb4911e4732dd87988755a7c0f58d4a9908984803a39141c75a61647448`, recorded in `run_info.json`). `conformal_methods/mlp_score.py`
  is **not** used for any reported number. This replaces the §4 "Lightning port" training line.
- **A1.2 Reproduction check.** Unchanged in content (P_H C1 = Exp055 cardio_pair, L1 seed 0, frozen draw; threshold, best/stop
  epoch and holdout CE within 1e-6). Exp055 values are read from its files, not retyped. The holdout CE is
  0.4509621858596802 (Tom's "1.168434" was the threshold, in the wrong field). The diagnostic (Exp055 loop on identical inputs)
  is the pass, and the same check is rerun inside `run.py` as the assert.
- **A1.3 Diagnostic row.** The `mlp_score.py` result is recorded in `reproduction_check.json` under
  `diagnostic_mlp_score_port` (threshold 1.1704441, best/stop epoch 114/124, holdout CE 0.4502637). It is used for nothing.
  Tom: statistically equivalent to seed noise, not bit-identical; a docstring issue, not a validity issue.
- **A1.4 No edits to shared modules.** `conformal_methods/mlp_score.py` and all other shared modules are untouched.
- **A1.5 R-S2 confirmed equivalent** (§13).

# Results

*(Appended after the run. Pre-registered text above is unchanged.)*

**Run:** `/usr/bin/python3 experiments/056_three_cluster_union/run.py`, stdout in `run_stdout.log`. Exit: **STOPPED at the
reproduction check** (SPEC §9 hard stop). `ESCALATION.md` written. `run.py` sha256 matched the pre-registered `383566…`.

| gate / check | result |
|---|---|
| Gate 0 — subject / dicom overlap, 3 splits | **PASS** (0 / 0 all pairs) |
| Gate 0 — within-cluster hierarchy violations, all clusters × 3 splits | **PASS** (0) |
| Identity gate (m = 1, 13 labels, frozen + 100 resplits, tol 0.5%) | **PASS** (max mismatch 0.000%; calibration-side coverage 0.9001) |
| Reproduction check (P_H C1 = Exp055 cardio_pair, L1 seed 0, frozen draw) | **FAIL**: threshold 1.170444 vs 1.168434; best/stop epoch 114/124 vs 86/96; holdout CE 0.450264 vs 0.450962 |
| Main experiment (arms A/B/B′/C/D, references, random partitions) | **not run** (hard stop) |

Diagnosis: `diag_reproduction_exp055_loop.py` reproduces Exp055 exactly with Exp055's training loop on the same inputs, so the
gap is the Lightning port (`conformal_methods/mlp_score.py`). See `ESCALATION.md` for the options for Tom.

**No results section for decision rules 1–6 exists, because no metric was computed.** The pre-registration above is unchanged.

## Results (Amendment 1 run; scope: all-13-clean studies, 77% of test)

**Run:** `/usr/bin/python3 experiments/056_three_cluster_union/run.py`, exit 0, **3,150 s** (13-label references 2,791 s; arm D 149 s).
stdout `run_stdout.log`; `run_info.json` records the amended `run.py` hash and the imported Exp055 `run.py` hash.

**Gates**

| gate | result |
|---|---|
| Gate 0, subject / dicom overlap, 3 splits | PASS (0 / 0) |
| Gate 0, within-cluster hierarchy violations, all clusters × 3 splits | PASS (0) |
| Identity gate (m = 1, 13 labels, frozen + 100 resplits) | PASS (max mismatch 0.000%) |
| Reproduction check (P_H C1 = Exp055 cardio_pair, L1 s0, Exp055 loop) | PASS, exact: threshold 1.1684340, best/stop 86/96, holdout CE 0.4509622 |
| Gated arms, P_H seed 0 (A per cluster, C, D, maxrank_adm) | all PASS; B and B′ valid (≥ 0.90) |
| Gated arms, 20 random partitions (A per cluster, C) | all PASS (C 13-label coverage 0.8991–0.8999) |
| Duplicate redraws | 0 |
| NaN in any core metric | none |
| Guard gap, designed clusters | 0.000 on every designed cluster and vector (unguarded = guarded) |

**Decision rules (resplit mean; CI = 2.5/97.5 percentile over 100 resplits, paired within resplit)**

1. **Alon's question.** Arm A (α = 0.10 per cluster) gives per-cluster coverage C1 0.9005, C2 0.8989, C3 0.9002, all in band.
   13-label coverage **0.7382 [0.7246, 0.7512]**, against Bonferroni floor 0.70, independence 0.729 (inside the CI), and the
   wanted 0.90. **"90% per cluster" does not give 90% on the vector; it gives ~74%.** Cross-cluster miss correlation under A
   (φ C1–C2 0.036, C1–C3 0.059, C2–C3 0.023) is why the product (0.729) is not the realised value.
2. **Cost of guaranteeing the vector.** B: 13-label coverage **0.9047 [0.8965, 0.9127]**, excess +0.0047 over 0.90.
   Size vs C: mean size 52.45 vs 49.40 (ratio 1.06); log₂(size)/13 0.3695 vs 0.3608. B′ (Šidák): 0.9015, log₂/13 0.3637.
3. **Headline, C vs maxrank_adm.** Δ mean log₂(size)/13 = **−0.2362 [−0.2443, −0.2273]**, CI entirely < 0, both arms pass
   (C 0.8998; maxrank_adm 0.8997). Empty-set rate Δ = 0.0 (both 0). Mean joint size 49.4 vs 850.0.
   **Clustered L1 beats unclustered max-rank on this run.**
4. **C vs D.** Δ log₂(size)/13 = **+0.0091 [+0.0034, +0.0164]**, CI > 0. Both arms pass (D 0.8999). Mean joint size 49.4 (C) vs
   30.1 (D). **Clustering costs efficiency relative to one 13-label learned scorer.** D's fully-decisive rate is 0.0002, so its
   small sets are almost all size-2+ (median 28.4, admissible 2,400; not degenerate).
5. **Designed vs random (arm C, seed 0, unguarded).** P_H log₂(size)/13 = 0.3608. Random partitions: median 0.4201, range
   0.3710–0.4682. P_H is **below all 20** (beats median; more efficient than random by the ≥ 18-of-20 rule). Arm A
   13-label coverage across the 20 random partitions: 0.742–0.770, all below 0.90 (as expected for an unguarded per-cluster arm).
   **Scope caveat:** random partitions do not enforce cross-cluster hierarchy edges (P_H has 0; random draws have 2–6), and
   are not matched on prevalence or difficulty, so this comparison mixes clustering with structure and label difficulty.
6. **Seed spread (P_H, C − maxrank_adm, seeds 0–4):** −0.2362, −0.2335, −0.2351, −0.2362, −0.2335. Range 0.0027 =
   **1.1%** of |seed-0 delta|, far below 50%. Not comparable-downgraded; the headline is not downgraded on seed grounds.

**Arm table, P_H seed 0, unguarded (resplit means; 13-label unless noted)**

| arm | joint cov [CI] | mean size | median size | log₂/13 | empty | fully-dec. | correct-dec. | dec. acc. | viol. dec. |
|---|---|---|---|---|---|---|---|---|---|
| A (α 0.10 / cluster) | 0.7382 [0.7246, 0.7512] | 11.84 | 5.99 | 0.1951 | 0 | 0.2108 | 0.1842 | 0.8740 | 0 |
| B (Bonferroni 0.10/3) | 0.9047 [0.8965, 0.9127] | 52.45 | 45.20 | 0.3695 | 0 | 0.0432 | 0.0411 | 0.9501 | 0 |
| B′ (Šidák) | 0.9015 [0.8936, 0.9101] | 50.40 | 41.90 | 0.3637 | 0 | 0.0470 | 0.0446 | 0.9497 | 0 |
| **C (max-rank across clusters)** | 0.8998 [0.8921, 0.9090] | 49.40 | 40.49 | **0.3608** | 0 | 0.0490 | 0.0465 | 0.9498 | 0 |
| D (L1 on 13 logits) | 0.8999 [0.8908, 0.9100] | 30.06 | 28.40 | 0.3517 | 0 | 0.0002 | 0.0002 | 1.0000 | 0 |
| maxrank_adm (classical, 13) | 0.8997 [0.8888, 0.9128] | 850.00 | 591.36 | 0.5969 | 0 | 0.0003 | 0.0002 | 0.7744 | 0 |
| maxrank (classical, 13) | 0.8997 | 2855.40 | 2048 | 0.7240 | 0 | 0.0001 | 0.0000 | 0.5162 | 0 |
| bonferroni α/13 (classical) | 0.9206 [0.9117, 0.9304] | 3171.89 | 2048 | 0.7463 | 0 | 0 | 0 | 0 | 0 |
| sidak (classical) | 0.9170 | 3113.95 | — | 0.7423 | 0 | 0 | 0 | 0 | 0 |
| indep_per_label (validity gap only) | 0.4441 | 106.90 | 16 | 0.3205 | 0 | 0.0790 | 0.0650 | 0.8229 | 0.0066 |

Per-cluster coverage (P_H seed 0): A 0.9005 / 0.8989 / 0.9002; B 0.9669 / 0.9663 / 0.9668; B′ 0.9656 / 0.9650 / 0.9656;
C 0.9650 / 0.9644 / 0.9650. Calibration-side coverage: A 0.7388, C 0.9001, D 0.9001.
(Classical references: decisive accuracy is low for maxrank and bonferroni, as expected for large unconditional sets.)

**Diagnostic (not gated): conditional 13-label coverage, arm C, P_H seed 0.** Rows positive for Pleural Other: 0.191 (n ≈ 152);
Fracture: 0.317 (n ≈ 330); Lung Lesion: 0.490 (n ≈ 506); any positive: 0.833 (n ≈ 11,268). Arm A on the same rows: 0.026, 0.005,
0.116, 0.568. **Rare-positive rows are poorly covered by every arm; the joint coverage is an average over mostly-negative rows.**
This is a finding to report, not a failure of the gate.

**Predicted vs realised cluster coverage under C.** Product of cluster coverages 0.8981; realised 13-label coverage 0.8998. φ
between cluster misses 0.026 (C1–C2), 0.020 (C1–C3), 0.004 (C2–C3).

**Guard gaps.** 0.000 on all designed clusters and the vector. Random partition R00 arm A vector guard gap 0.024, R03 0.045,
R16 0.030 (random partitions; not compared).

**Escalation.** `ESCALATION.md` from the first stop is **resolved by Amendment 1** and is kept as the record of the stop. No
pre-registered non-stop trigger fired on this run. One recording gap is flagged there: the 200-epoch cap hits per scorer were
not persisted by this `run.py`. I did not rerun to recover them.

**Caveats carried into every number:** shared-backbone correlation (EC/CM 0.936 → 0.997) likely overstates joint gains; the
all-13-clean population excludes 22.6% of test; CIs conditional on the fixed pool and trained scorers (understated); C2 is m = 6,
outside the locked 2–5 range; cross-cluster hierarchy not enforced for random partitions; arm A has no guarantee by construction.

## Addendum A2 (post-sign-off, descriptive)

Tom signed off the main run conditional on this addendum, which is required before claim 4 (C vs D) is cited. Descriptive only;
no new decisions. Script: `addendum_a2.py` (imports `run.py` unedited for training, splits and product regions). Outputs are new
files and do not overwrite the main run: `addendum_a2_cap_diagnostics.csv`, `addendum_a2_threshold_match.csv`,
`addendum_a2_conditional_per_resplit.csv`, `addendum_a2_conditional_summary.csv`, `addendum_a2_metrics.json`, `addendum_a2_stdout.log`.
Reproduce: `/usr/bin/python3 experiments/056_three_cluster_union/addendum_a2.py` (≈ 10 min).

**1. Cap diagnostics and threshold match.** Retrained with Exp055's `train_scorer`, seed 0:

| unit | m | best epoch | stop epoch | hit 200 cap | epochs run | holdout CE |
|---|---|---|---|---|---|---|
| C1 | 2 | 86 | 96 | no | 96 | 0.450962 |
| C2 | 6 | 48 | 58 | no | 58 | 1.139609 |
| C3 | 5 | 38 | 48 | no | 48 | 0.954851 |
| D (L1_full13) | 13 | 19 | 29 | no | 29 | 2.691522 |

No cap hits. **Threshold match:** all 101 splits (frozen draw + 100 resplits), C1–C3 max-rank thresholds and D thresholds,
max |difference| from the main run = 8.9e-16. Assert at 1e-9 passed. The C1 values reproduce Exp055's `cardio_pair` exactly.

**2. Conditional 13-label coverage (unguarded, resplit mean [2.5, 97.5 percentile]; n = mean rows per resplit).**

| subgroup (n) | C | D | maxrank_adm |
|---|---|---|---|
| Pleural Other-positive (n 152) | 0.191 [0.130, 0.252] | 0.169 [0.121, 0.222] | 0.792 [0.743, 0.853] |
| Fracture-positive (n 331) | 0.317 [0.255, 0.382] | 0.423 [0.372, 0.469] | 0.900 [0.876, 0.925] |
| Lung Lesion-positive (n 506) | 0.490 [0.448, 0.537] | 0.483 [0.445, 0.525] | 0.790 [0.751, 0.827] |
| ≥ 1 positive (n 11,268) | 0.833 [0.821, 0.847] | 0.833 [0.819, 0.849] | 0.859 [0.845, 0.875] |

Read plainly: on the rare-positive rows, C and D both cover far less than 0.90, and maxrank_adm covers much more (it is
much larger, so it covers these rows at about 0.79–0.90). Joint coverage on these subgroups does not track the 0.90 overall
figure. The subgroup-level numbers are not gated and are not a validity claim.

**3. Fully-decisive rates (resplit mean [2.5, 97.5]).** C **0.0490 [0.0431, 0.0557]** (correct-decisive 0.0465; decisive accuracy
0.9498). D **0.00020 [0.00005, 0.00042]** (correct-decisive 0.00020; decisive accuracy 1.000, from 0.02% of rows). So
C's efficiency gain over D is not carried by decisive output. Its sets are not mostly size 1, and its 13-label sizes come from
the product of per-cluster sets.

**4. Random partitions:** not retrained (per A2 item 4).

---

## Tom Hagan sign-off (2026-11-16)

Signed off. Claim 4 cleared for citation after Addendum A2 (no cap hit; thresholds reproduce to 8.9e-16).
Scope on every claim: all-13-clean studies, 77% of test; shared-backbone caveat applies.

1. **Arm A (naive, 0.10 per cluster):** 13-label coverage 0.738 [0.725, 0.751], not 0.90. Cluster misses are near-independent (φ ≤ 0.026), so coverage ≈ 0.9³.
2. **Arm B (Bonferroni across clusters):** 0.9047, only 6% larger than C (descriptive, as pre-registered). With 3 near-independent clusters the combination rule barely matters.
3. **C vs maxrank_adm:** ≈ 8.4× smaller in geometric mean (Δ log₂/13 = −0.236 [−0.244, −0.227]). Credit joint learned scoring, not clustering. **Must state:** smaller sets, much worse coverage on rare positives (learned 0.17–0.49 vs maxrank_adm 0.79–0.90).
4. **C vs D:** D slightly smaller in geometric-mean size (Δ log₂/13 = +0.0091 [+0.0034, +0.0164], ≈ 8%). C is fully decisive on 4.9% of rows [4.3, 5.6] (decisive accuracy 0.950); D on 0.02%. D wins on typical size; C is the only one that commits. Report both; no single winner. (Earlier line "neither is ever decisive" retracted.) D's early stop (best epoch 19, holdout CE 2.69) is the pre-registered rule and doesn't affect the result.
5. **P_H vs random:** more efficient than all 20 random 2/6/5 partitions under within-cluster-only admissibility. Cannot separate grouping from keeping cross-cluster hierarchy constraints. (Difficulty-matching is not a confound: same 13 labels.)
6. **Seed spread:** 1.1%, not downgraded.

**Mandatory caveat:** C conditional coverage on rare positives: Pleural Other 0.19, Fracture 0.32, Lung Lesion 0.49, any positive 0.83. The marginal 0.90 is earned mostly on all-negative rows.

**Next experiment (Tom):** label-/class-conditional (Mondrian) joint coverage.
