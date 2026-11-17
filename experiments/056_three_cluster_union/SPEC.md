# SPEC — Exp056: three-cluster partition of all 13 labels, L1 scorer, coverage of the combined set

Author: Tom Hagan, 2026-11-16. Approved to run by Alon, 2026-11-16 ("yes, run it").
Alon's request: "using L1 to create 3 clusters. the 3 clusters combined should contain all 13 labels with no overlap. and I want to see how the unification of cluster cover." Read as: score each of 3 clusters with L1 (not: derive clusters from L1).

Tom's side ruling on Exp055 Q2: the over-coverage on `rand_cardio_pair_15` was produced by the guard (L1_unguarded coverage 0.9025, calibration-side 0.9001, empty rate 6.3%; guarded 0.956), not by an atom. **Unguarded is primary going forward.**

Folder: `experiments/056_three_cluster_union/`. Pre-register in `README.md` before any frozen-split metric is computed. Reuse without editing: `conformal_common.py`, `conformal_methods/{max_rank,bonferrony,mlp_score}.py`, `data/hierarchy.py`. Everything not stated below follows Exp055 exactly: inputs, the all-13-clean population, Gate 0, 100 subject-level 50/50 resplits of the 37,631-row pool, the frozen-draw reference row, the k = ⌈(n+1)(1−α)⌉ rule, the CI convention and the caveat list.

## 0. Question
Each cluster is calibrated on its own. The 13-label prediction set is the Cartesian product of the three cluster regions. What fraction of rows have the true 13-label vector inside that set? And what does it cost to make that fraction ≥ 0.90?

Scope tag on every table: all-13-clean studies, 77% of test. Note in the README: at m = 13, the per-member exclusion rule in §3a is the same as all-13-clean, so for this experiment the population is the natural one, not a restriction, although the 22.6% caveat still stands.

## 1. Partitions
**Designed, primary: P_H, the hierarchy-closed partition.** Pre-registered and fit on no data.
- C1 cardiac (m=2): Enlarged Cardiomediastinum, Cardiomegaly.
- C2 parenchymal (m=6): Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis.
- C3 pleural/other (m=5): Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices.

Rationale: each hierarchy edge falls inside one cluster, so admissibility is enforced exactly and the product of admissible cluster regions is admissible. C2 has m=6, outside the 2–5 range in RESEARCH_DIRECTION (64 states with ~14.8k training rows is fine for L1; calibration is one scalar quantile). No data-driven partition is fitted.

**Pleural Other (151 calibration positives).** Included; the ≥200 floor does not apply. Conditional coverage on PO-positive rows is reported as a diagnostic (§6), not gated. Fracture (304) and Lung Lesion get the same diagnostic.

**Random control: 20 partitions with the same size profile (2/6/5).**
- Partition i (0..19): `default_rng(5600+i).permutation(ALL_LABELS canonical order)`. Positions 0–1 → R1, 2–7 → R2, 8–12 → R3. Labels within each cluster in canonical order.
- Reject and redraw (increment a sub-seed, record it) if the draw equals P_H or an earlier draw as an unordered partition.
- Within-cluster hierarchy edges are applied as admissibility. Cross-cluster edges are not applied, for any arm. Record cross-cluster edge counts per partition.

## 2. Scorer
- L1 per cluster, Exp055 recipe unchanged: `Linear(m,64)→ReLU→Linear(64,2^m)` on the cluster's m logits; same inner holdout (20% of validation subjects, `default_rng(0)`), Adam, early stopping, score `−log_softmax[y]`, +∞ on inadmissible states.
- Train via `conformal_methods/mlp_score.py`.
- Seeds 0–4 on P_H. Seed 0 only on random partitions.
- No L0, no L2.

**Reproduction check (assert):** P_H C1 equals Exp055 cardio_pair. On the frozen draw, L1 seed 0 must reproduce Exp055's threshold, best/stop epochs and holdout CE within 1e-6. Read these values from Exp055 files; do not retype them. A mismatch is stop-and-escalate.

## 3. Arms for combining the clusters
In every arm, the 13-label set is R1×R2×R3; size = product of the three cluster region sizes; a row is covered iff the true state of every cluster is in its region. Each arm has an **unguarded variant (primary)** and a **guarded variant (secondary**; per-cluster argmin added when a cluster region is empty).

- **A — naive, α_c = 0.10 per cluster.** No 0.90 guarantee on the vector (Bonferroni lower bound 0.70; independence prediction 0.9³ = 0.729).
- **B — Bonferroni across clusters, α_c = 0.10/3.** Guaranteed ≥ 0.90.
  - **B′ — Šidák, α_c = 1−0.9^(1/3):** recorded.
- **C — max-rank across clusters (primary jointly-valid arm).** n_A×3 matrix of true-state L1 scores on calibration half A; `max_rank.column_ranks` → row-wise max → k* at α=0.10 → t_c = order statistic of column c at k*. Timans Alg. 2 with clusters as columns.
- **D — L1_full13 (unclustered joint learned scorer, secondary).** L1 on all 13 logits → 8192 states, +∞ on states inadmissible under the full hierarchy, one quantile at α=0.10. Seed 0 only. Size = count of admissible states with s ≤ t; enumerate per row in batches, float32. If it hits the 200-epoch cap, record; do not change the cap.

## 4. Unclustered references on the 13-label vector (classical, Exp055 convention, no guard)
- Per-label `indep` at α (validity gap only).
- `bonferroni` at α/13; `sidak`.
- `maxrank` (13 columns); **`maxrank_adm`** = product set ∩ full-hierarchy admissibility (**primary classical comparator**).
- Admissible count factorises over hierarchy components: enumerate {EC,CM} (4 states) and the six-label lung family (64 states), multiply by the per-label sizes of the 5 root labels without hierarchy edges.

## 5. Identity gate
Exp055 m=1 gate rerun as is (13 labels, frozen draw + 100 resplits, tolerance 0.5%). Failure is a stop.

## 6. Metrics (per arm × partition × seed × resplit; summarised over resplits)
**Coverage:** 13-label joint coverage; per-cluster coverage; product of the three per-cluster coverages; pairwise φ between the three cluster miss indicators; calibration-side coverage; thresholds and k / k*.

**Efficiency:** **primary: mean log₂(size)/13 over non-empty rows, empty-set rate always beside it.** Mean joint size (empty = 0), median size, fully-decisive rate, correct-decisive rate. Same per cluster (log₂/m_c).

**Diagnostics:** conditional 13-label coverage on rows positive for PO, Fracture, Lung Lesion, and on rows with ≥1 positive (report n). Guard gap = guarded − unguarded coverage, per cluster and on the vector. Modal-state mass per cluster on conformal (labels only; record before the run).

**CIs:** 2.5/97.5 percentiles over the 100 resplits of within-resplit paired differences, with the Exp055 conditionality caveat.

## 7. Gates (unguarded variants)
- **A:** each of the three per-cluster coverages in [0.895, 0.905]. 13-label coverage reported, not gated.
- **B / B′:** 13-label coverage ≥ 0.90 (resplit mean) = valid; coverage-unequal, sizes reported but not compared.
- **C, D, maxrank_adm:** 13-label coverage in [0.895, 0.905].
- A comparison is `comparison_valid` only if both arms pass.

## 8. Pre-registered decision rules
1. **Alon's question (descriptive).** Arm A's 13-label coverage (mean, CI) next to 0.70, 0.729, 0.90. State plainly whether "90% per cluster" gives 90% on the vector.
2. **Cost of guaranteeing the vector.** Arm B's coverage excess over 0.90 and its size ratio vs C (descriptive).
3. **Headline: C vs maxrank_adm.** "Clustered L1 beats unclustered max-rank" only if the paired CI of Δ mean log₂(size)/13 is entirely < 0 and both arms pass. Report Δ empty-set rate; flag if |Δ empty| > 1pp.
4. **C vs D.** CI < 0 → clustering beats a single 13-label learned scorer; CI > 0 → clustering costs efficiency. Say it plainly either way.
5. **Designed vs random (arm C, seed 0).** P_H's percentile among 20 random partitions on log₂/13. "Beats median" if below median; "more efficient than random" only if below ≥18 of 20. Also report arm A coverage across random partitions.
6. **Seed spread (P_H, arm C):** Exp055 rule 5 unchanged.

## 9. Escalation (write `ESCALATION.md`, do not iterate)
Stop: Gate 0 failure; identity-gate failure; reproduction mismatch; any true calibration state inadmissible; infinite threshold; NaN in a reported metric.

Record and escalate at the end: any gated arm outside its band (on a random partition, record and exclude from that comparison); guard gap > 2pp on any designed cluster; D fails to train or is degenerate (median size = admissible count); duplicate-partition redraws > 5; anything ambiguous.

Readings not pinned down by this spec go in a §Readings, as in Exp055 §14.

## 10. Caveats (in the results)
Exp055 list, plus: arm A has no guarantee by construction; cross-cluster hierarchy not enforced for random partitions; C2 is m=6, outside the locked 2–5 range.
