# Exp053 — Equal-coverage cluster comparison (PRE-REGISTRATION)

**Spec:** `SPEC.md` (Tom Hagan; approved by Alon 2026-11-10), incl. Amendments (E1–E8), Exp052 ruling, Amendments 2 (Q1–Q4 and bookkeeping permission).
**Earlier escalations:** `ESCALATION.md` — resolved by the Amendments. `ESCALATION_2.md` — resolved by Amendments 2. Both kept in place.
**Git HEAD at pre-registration:** `6a26ac848b016e82c22d1bd42f61280740d3d26d` (working tree dirty; `run.py` and this README are untracked; the run is not committed).
**Pre-registration time:** 2026-11-10 ~12:20 IST, before any metric or gate was computed.
**Seeds:** resplits r = 0..99 (`default_rng(r)`); random null `default_rng(1000+idx)`, idx 0..199; clinical random C `3000+idx`, D `4000+idx`, idx 0..99; matched null `default_rng(1500)` fresh per profile; anti-clustering `default_rng(2000)` fresh per profile. Frozen draw: no randomness.
**Hashes of reused files and inputs:** `.prereg_hashes.txt` in this folder.

**Provenance flag (recorded before run):** `lp_scp.py` has sha256 `79123fa…` now, versus `81e2834…` recorded in Exp052's `.prereg_hashes.txt` (file mtime 2026-11-04, after 052 ran on 2026-11-01). It is reused as-is per the spec; I did not edit it and cannot reconstruct the 052 version. The LP-SCP formula in the current file matches the one 052 describes (`sqrt(rᵀΣ⁻¹r) + μHp + νCp`, μ=ν=1). Flagged in the writeup.

## What is tested
Partition the 13 labels into K clusters; run four methods (Bonferroni, Šidák, max-rank, LP-SCP) inside each cluster at one common β̂ chosen so the combined 13-label prediction has 90% joint coverage; compare partitions on set size at equal coverage. Population: 13-clean rows only. Every headline carries the scope "studies with no uncertain label on any of the 13 categories, ~77% of test".

## Inputs (frozen; no inference re-run)
- Calibration: `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_conformal.csv` (18,754 rows, 5,926 subjects)
- Test: `…809_test.csv` (24,394 rows, 6,395 subjects); 13-clean subset 18,877 rows / 5,932 subjects (from 052)
- Validation: `…809_validation.csv` (18,570 rows, 5,898 subjects); used only for partitions and LP-SCP structure
- Pooled design: conformal (18,754) ∪ test-13-clean (18,877) = **37,631 rows**, 11,858 subjects (S = 11,858; half = 5,929)
- Reused unedited: `conformal_common.py` (loading, Gate 0, `extract_group`, `nonconformity`, `enumerate_states`, `hierarchy_admissible`, `CHEXPERT_HIERARCHY` via `data/hierarchy.py`, `evaluate`), `max_rank.py::calibrate_max_rank` and `bonferrony.py::calibrate_per_label` (context rows only), `lp_scp.py` (`fit_structure`, `state_scores`), `data/hierarchy.py`.

## Score, ranks, p-values (all methods; integer-count scale)
- Score s = 1−p if y=1 else p, every row.
- For a calibration pool A (size n), label j: for a row i, `d_ij = #{l∈A : s_lj ≥ s_ij}` (includes i itself; equals 1 + #{l≠i}).
- For a test row or candidate vector v, label j: `d_j(v_j) = 1 + #{l∈A : s_lj ≥ s_j(v_j)}`, with s_j(1) = 1−p_j, s_j(0) = p_j.
- p_ij = d_ij/(n+1). `R_ik = min_{j∈k} d_ij`; `r_ik = R_ik/(n+1)`.
- Cluster P-values (cal row i / candidate v):
  - Bonferroni: `min(1, m_k·R/(n+1))`.
  - Šidák: `1 − (1 − R/(n+1))^{m_k}`.
  - Max-rank: `#{l∈A : R_lk ≤ R_ik}/(n+1)` for cal (self included) and `(1 + #{l∈A : R_lk ≤ R_k(v)})/(n+1)` for candidates.
  - LP-SCP: S = `lp_scp.state_scores` (μ=ν=1, structure from validation, hierarchy not enforced, no non-empty guard). `P_ik = #{l∈A : S_lk ≥ S_ik}/(n+1)` for cal; `P_k(v) = (1 + #{l∈A : S_lk ≥ S_k(v)})/(n+1)` for candidates. S_ik is the score of the row's own true state, taken from the same `state_scores` call (identical formula; bookkeeping, see below).
- Row statistic `t_i = min_k P_ik` over calibration rows (LOO).

## β̂ rule (exact)
With n = |A| and c = ⌊α(n+1)⌋ = ⌊(n+1)/10⌋ (computed as an integer), β̂ = the largest value u among the distinct {t_i} such that `#{i : t_i ≤ u} ≤ c − 1`. If no value qualifies, the run ESCALATES (no fallback). Ties conservative: a cluster contains v iff P_k(v) > β̂; row i is covered iff t_test > β̂.

## Sets and metrics
- Cluster set = `{v : P_k(v) > β̂}`; joint set = product of cluster sets; joint size = product of cluster set sizes (empty product = 0; excluded from log₂ mean).
- Per-label decisive: label j of cluster k is a singleton if exactly one value appears in the cluster set (projection rule; for p-value methods this equals the product set).
- Metrics per cell and resplit (test half B, 13-clean): joint coverage (true vector in every cluster set); per-cluster coverage; β̂; mean log₂(size)/13 over non-empty rows; arithmetic mean size (empty=0); median size; fully-decisive rate (all 13 singletons); empty rate; hierarchy-violation rate among fully-decisive output (`hierarchy_admissible` on all 13); decisive accuracy on fully-decisive rows; per-label decisive rate (diagnostic); calibration-side coverage (fraction of A rows with t_i > β̂).
- Diagnostic: coverage stratified by "any of Pleural Other / Fracture / Lung Lesion positive" (true label).
- Ratio and delta reporting: paired within resplit; CIs = 2.5 / 97.5 percentiles over resplits.

## Resplits (headline)
- For r in 0..99: `rng = default_rng(r)`; permutation of the sorted unique pooled subject IDs; first 5,929 = half A (calibration), rest = half B (test). Full calibration (LOO ranks, β̂) recomputed on A; evaluation on all B rows.
- Split-calibration arm (K=4 informed only, all four methods, same 100 resplits): inner permutation drawn from the same `rng` after the outer permutation (Amendments 2, Q1). Sorted unique subject IDs of A permuted; first ⌊|A|/2⌋ = A1 (reference ranks: d, R, S from A1 only, with no LOO since rows are disjoint), rest = A2 (β̂ quantile). β̂ from A2 rows with n = |A2|; test = B with P computed from A1 only (n = |A1|). Robustness only.
- Frozen draw: calibration = conformal 13-clean (LOO), test = test 13-clean. One row per cell, no pass/fail; reported with its percentile within the resplit distribution of each metric.

## Partitions (all fixed before any metric)
- **References:** K=13 singletons; K=1 (all 13).
- **Clinical:** C (K=7): {EC,CM}, {LO,LL,Edema,Consolidation,Pneumonia,Atelectasis}, {PTX}, {PE}, {PO}, {Fracture}, {SD}. D (K=6): {EC,CM}, {LO,LL,Edema,Consolidation,Pneumonia,Atelectasis}, {PE,PO}, {PTX}, {Fracture}, {SD}.
- **Informed (S-avg):** Spearman ρ of validation nonconformity score columns (all 18,570 rows; no uncertain labels so no exclusion). Distance 1−ρ. Average linkage; merge the closest pair (mean pairwise distance) whose union ≤ 7 labels. Single merge sequence; snapshots at K=6, 4, 3. Ties: pair whose cluster-minimum label index is lowest, then the other cluster's minimum. Ties recorded. If the procedure stalls above K=3, escalate. Stated: this is a marginal score-correlation partition, not the RESEARCH_DIRECTION GCM procedure.
- **Random null (primary):** 200 per informed profile (K=3,4,6); `default_rng(1000+idx)`; random permutation of 13 labels chunked into the profile sizes. LP-SCP on idx 0..99.
- **Random null C/D (secondary):** 100 each, p-value methods only; seeds 3000+idx (C), 4000+idx (D).
- **Matched null (secondary):** per informed profile, pool of 5,000 random partitions from `default_rng(1500)` (fresh per profile); keep the first 200 whose within-cluster directed hierarchy-edge count (7 edges total, `CHEXPERT_HIERARCHY`, both endpoints in one cluster) equals the informed partition's. If fewer than 200 match: use all and report the count; fewer than 60 → escalate. p-value methods on all kept; LP-SCP on the first 100 kept.
- **Anti-clustering:** same profile as each informed cut; minimise W(P) = Σ within-cluster ρ; initial partition = random permutation chunked into the profile (`default_rng(2000)`, fresh per profile), 50 restarts; move = best-improvement swap of two labels in different clusters (sizes fixed), accept if decrease > 1e-12, stop when none; keep lowest final W, ties to lowest restart index. All four methods.
- **Difficulty control (descriptive):** labels sorted by validation AUC ascending (ties to lowest index); profile sizes sorted descending; contiguous fill (largest chunk = lowest-AUC labels). One partition per cut; all four methods.
- For every partition: W(P) and within-cluster hierarchy-edge count.
- LP-SCP structure per cluster: `fit_structure(extract_group(validation, cluster))`, labels in `ALL_LABELS` order. Σ conditioning and ridge use are recorded.

## Context rows (Q4: frozen draw only, no size comparison)
Uncalibrated K=1 Bonferroni, Šidák, max-rank at α = 0.10 (`bonferrony.calibrate_per_label` and `max_rank.calibrate_max_rank` on the conformal split, `cc.evaluate` on test), and independent per-label at α (`correction="none"`).

## Methods on partitions
All four methods on references, informed, anti, difficulty, clinical C/D (LP-SCP) and matched/random as above. LP-SCP null on the first 100 per profile. If the LP-SCP null projects over 4 h, fallback = LP-SCP null on the frozen split only, labelled.

## Gates (run first, in order)
- **Gate 0:** `cc.load_splits(with_validation=True)` — zero shared subjects and dicom IDs for all pairs (checked by `gate0_check`); counts asserted 18,754 / 24,394 / 18,570 rows and 5,926 / 6,395 / 5,898 subjects; 13-clean test 18,877 rows / 5,932 subjects; pooled subjects disjoint between conformal and test; pooled rows 37,631.
- **Gate 0.5:** positive counts per label on the conformal 13-clean split; flag < 150 (`cc.MIN_POSITIVE_CALIBRATION_ROWS`); expected: none flagged (Pleural Other 151, Fracture 304, Lung Lesion 523). No label dropped or moved.
- **Identity gates (SPEC §8):** at K=13 and K=1, Bonferroni, Šidák and max-rank produce identical sets; ≤ 0.5% of rows may differ (checked on every resplit and on the frozen draw). At K=1, calibration-side coverage reported (should be 0.90 up to 1/n).
- **Code verification (not a science gate):** vectorised P-values and LP-SCP cal scores compared with a naive per-row loop on random subsets (exact for counts; tolerance 1e-10 for LP-SCP); own-state scores vs `lp_scp.true_state_scores` on full calibration for m ≤ 6 and a 300-row subset for m = 13.

## Pre-registered escalation triggers (SPEC §9)
Any identity gate failure; resplit-mean joint coverage outside [0.895, 0.905] for a p-value method (cell reported, size comparisons marked "coverage-unequal"); resplit SD of K=1 max-rank coverage < 0.25 pp or > 0.8 pp; an informed cut with a cluster > 7 labels; runtime over the 4 h fallback (→ fallback, not stop); β̂ undefined; matched-null matches < 60; informed procedure stalls above K=3.

## Hypotheses and tests (SPEC §7)
- **H1:** resplit-mean coverage in [0.895, 0.905] for every (method, partition) except LP-SCP.
- **H2 (primary):** max-rank, informed cuts vs the 200-partition null (resplit-average mean log₂/13). `p = (1 + #{null ≤ informed})/201` (E7, rank-based only). Holm over the three cuts at 0.05. Other methods and matched null: secondary, unadjusted, descriptive. Descriptive `np.percentile` (linear) 5th and 50th of the null reported.
- **H3:** anti-clustering above null median and above informed (sign check).
- **H4 (descriptive):** Spearman(W(P), mean log₂ size) negative over the null, per method and profile.

## Bookkeeping readings (Amendments 2 permission; each is an RNG/bookkeeping convention, same partitions and hyperparameters under any plausible reading)
1. RNG = `numpy.random.default_rng(seed)`; the bookkeeping permission covers the generator choice, which only changes which random draws are used.
2. Random partition = random permutation of label indices chunked into the profile sizes (uniform over set partitions with that size profile).
3. Subject permutation is taken over the sorted unique subject IDs, so the permutation does not depend on row order.
4. Pool order = conformal rows then test-13-clean rows.
5. Spearman via `scipy.stats.spearmanr` on the 13 score columns; AUC via Mann–Whitney rank formula.
6. Anti-clustering best-improvement computed by full recomputation of W for each swap (no approximation).
7. Matched pool partitions stored in canonical form (clusters sorted by minimum label index).
8. Floating-point comparisons use the same expression for calibration and test, so equal-valued P's compare equal.

## Caveats that must appear in the writeup
1. Backbone: shared backbone inflates cross-label score correlation (EC/CM 0.936→0.997); likely overstates the efficiency gain. The gain is a property of this model's scores.
2. Guarantee vs efficiency: coverage of the min-over-clusters statistic is controlled to O(1/n) via leave-one-out ranks; exactness requires the split arm. Efficiency is empirical.
3. Population scope: 13-clean rows only; silent about the 23% with an uncertain label.
4. Recalibration changes meaning: Bonferroni/Šidák here are empirically recalibrated; at K=1 and K=13 they coincide with max-rank; differences at K>1 partly reflect the cross-cluster weighting convention.
5. Partition provenance: informed = marginal score correlation (S-avg), not GCM, Ding or RadGraph.
6. Uncertainty: resplit intervals are conditional on the fixed 37,631-row pool, and pooling conformal + test means the resplit percentile intervals understate total uncertainty.
7. Rare labels: coverage is not class-conditional; stratified diagnostic reported (Pleural Other 151, Fracture 304, Lung Lesion 523).
8. LP-SCP: tie over-coverage may break equal coverage; hierarchy co-location may drive its gain — the matched null is the check. LP-SCP uses a Mahalanobis score with Hp/Cp structure fit on validation (score and structural asymmetry vs. the other three methods).
9. Pooling: resplits pool conformal and test; the frozen draw is a reference row only.
10. `lp_scp.py` hash differs from Exp052's record (see Provenance flag).
11. Matched-null edge-count target: 7 directed edges (SPEC E1; Amendments).

## Results
Written after the run, in this file below the line "POST-RUN RECORD", without altering the text above.

---

# POST-PREREGISTRATION RECORD (2026-11-10, ~13:00 IST)

- **Status: STOPPED at the identity gate (SPEC §8, §9). The full 100-resplit run was not executed.** See `ESCALATION_3.md`.
- Gate 0: passed. Gate 0.5: passed (none flagged; Pleural Other 151). Code verification: passed (`run.py`, `verify_code`).
- Identity gate: **failed** on the frozen draw (Bonferroni vs max-rank 0.832% of rows) and on 76/100 resplits (range 0.229–1.058%). Šidák identical to Bonferroni. Outputs: `identity_gates_all.csv`, produced by `identity_check.py`.
- Timing: one resplit with split arm 112.5 s, projected ~3.2 h for 100 resplits. Under the 4 h limit, so no LP-SCP fallback.
- Partition count 1,364 (matched null K=6 kept 151 of 200 because only 151 of 5,000 pool partitions matched the 3-edge target; E4 rule, ≥60).
- Pre-run code changes (all before any reported number): removed O(n²) naive check and dead code; no change to any formula. The timing run wrote to `/private/tmp/claude-501/scratch_exp053/` and is not part of the record.
- Pre-registration text above is unchanged. The `lp_scp.py` hash discrepancy with Exp052 is unchanged and flagged.

---

# AMENDMENT 3 (dated 2026-11-10, ~12:35 IST; recorded before any post-amendment result exists)

**Ruling:** Tom, `ESCALATION_3.md`; spec text in `SPEC.md` "Amendments 3" (supersedes SPEC §2 for max-rank and LP-SCP cluster p-values).

**Pre-amendment identity-gate numbers (logged, not deleted):** Bonferroni vs max-rank, K=13 and K=1 (Šidák = Bonferroni exactly):
- frozen draw **0.832%** of rows; resplits 0–99 min / median / max **0.229% / 0.615% / 1.058%**; 76/100 resplits above 0.5%.
- Raw output: `identity_gates_preamendment_all.csv`. Code: pre-amendment `run.py` version (cluster-level "+1" on test side for max-rank and LP-SCP).
- **One-level diagnostic (POST-HOC, not pre-registered):** at K=1 on resplits 0–2, max-rank and Bonferroni thresholds differ by one integer R-level (185 vs 186; 190 vs 191; 187 vs 188). Tom's diagnosis: the test-side "+1" was applied to the max-rank/LP-SCP cluster p-value but not to Bonferroni/Šidák, and the β̂ rule already reserves one count.

**Amended definition (max-rank and LP-SCP cluster p-values only):**
- For a self-inclusive calibration row: `P = #{calibration rows at least as nonconforming as own statistic}/(n+1)` (own row included). Unchanged from before.
- For a test row or candidate vector: `P = #{calibration rows at least as nonconforming}/(n+1)`, **no extra +1**.
  - Max-rank: count of ref rows with `R_l ≤ R(v)`, where `R(v) = min_j d_j(v_j)` and the per-label `d_j` keeps its `+1` (per-label p-values unchanged).
  - LP-SCP: count of ref rows with `S_l ≥ S(v)`.
- Bonferroni, Šidák, per-label `d` and the β̂ rule are unchanged. Coverage is still `t > β̂`.
- **Reading adopted for the split-calibration arm (flagged):** A2 rows are β̂-calibration rows that are not in the reference set A1. They are treated as test-like rows against A1, i.e. the plain count with no extra +1 for the cluster p-values, and the per-label `d` keeps its `+1` (A2 rows are not self-inclusive). This keeps A2 exchangeable with the test rows. Robustness only.
- **Note (my reading, to be checked, not a finding):** the test-side cluster p-value now carries one count less than the standard split-conformal p-value `(1+#)/(n+1)`. The validity argument rests on the β̂ rule reserving that count. The empirical coverage in the run is the check.

**Gate plan (pre-run, per Amendments 3):**
1. Reproduce Exp052 LP-SCP guard-off Rung E and Rung C from the current `lp_scp.py` in a scratch directory; compare with `rungs_summary_3method.csv` at ≤ 1e-6. Pass → record sha256 below. Fail → escalate.
2. Recompute the identity gate on the frozen draw and resplits 0–9 at K=13 and K=1 (tolerance 0.5% of rows). Report all 100 resplits. Fail → stop, escalate, no further variants.
3. Full 100-resplit run, if 1 and 2 pass. Run in background.

**Matched null (Amendments 3):** K=6 uses all 151 kept partitions for Bonferroni/Šidák/max-rank with rank p = (1+#null ≤ informed)/152; LP-SCP uses the first 100. Counts for K=3 and K=4 reported. Secondary, descriptive, outside the Holm family.

## Amendment 3 pre-run checks (recorded before the full run)
- **lp_scp.py reproduction (PASS):** current file sha256 `79123faec33bfb9f7ef336359eaf03ce9b7759e7e377d11fc7eb35dcf52d94a1`. Exp052 guard-off rows recomputed in scratch (`/private/tmp/claude-501/scratch_exp053/repro/lp_repro.py`): Rung C coverage 0.5515706944959474 / size 65.39603750595964, Rung E coverage 0.8942628595645494 / size 862.4215712242411; both differ from `rungs_summary_3method.csv` by 0.0 (threshold 1e-6). Exp052 outputs untouched.
- **Identity gate, post-Amendment 3 (PASS):** `identity_check.py` → `identity_gates_postamendment3_all.csv`. Frozen draw: 0.000%. Resplits 0–9: max 0.037% (K=13, resplit 4). All 100 resplits: max 0.048%, none above 0.5%. Šidák identical to Bonferroni. Calibration-side K=1 coverage unchanged in kind (resplit mean 0.90028).
- Pre-amendment record retained: `identity_gates_preamendment_all.csv`.
- Full 100-resplit run authorised and started after these checks.

---

# POST-RUN RECORD (appended after the full run; pre-registered text and Amendment 3 unchanged)

**Run:** `run.py --resplits 100` completed, 10,908 s (~3.0 h), exit 0. Log: `run_stdout.log`. Outputs: `resplit_metrics.csv` (475,605 rows), `run_info.json`, `identity_gates.csv`. Analysis: `analyze.py` → `metrics.json`, `cell_summary.csv`, `frozen_draw_rows.csv`, `null_summary.csv`, `paired_deltas.csv`, `dose_response.csv`, `split_arm_summary.csv`, `identity_gates_summary.csv`.
**Post-amendment identity gate:** frozen 0.000%; resplits 0–9 max 0.037%; all 100 max 0.048% (`identity_gates_postamendment3_all.csv`). Pass.
**lp_scp.py:** sha256 `79123faec33bfb9f7ef336359eaf03ce9b7759e7e377d11fc7eb35dcf52d94a1`; reproduction of Exp052 guard-off Rung C and E: pass (differences 0.0).
**Triggers (SPEC §9), evaluated in `metrics.json`:** identity gate pass; K=1 max-rank resplit SD 0.573 pp (inside 0.25–0.8); all p-value and LP-SCP cells inside [0.895, 0.905]; no informed cluster > 7; β̂ defined everywhere; runtime under 4 h. **No escalation fired.**
**Partitions:** 1,364 (matched K=6 kept 151; K=3 and K=4 kept 200 each, from 441 and 394 pool matches of 5,000).

## Informed partitions (S-avg on validation; no ties)
Labels by ALL_LABELS index: EC Enlarged Cardiomediastinum, CM Cardiomegaly, LO Lung Opacity, LL Lung Lesion, Ed Edema, Co Consolidation, Pn Pneumonia, At Atelectasis, PTX Pneumothorax, PE Pleural Effusion, PO Pleural Other, Fx Fracture, SD Support Devices.
- K=3: {EC,CM,LO,Ed,At,PE} | {LL,PO,Fx} | {Co,Pn,PTX,SD}. W(P)=14.03; hierarchy edges within cluster 4 (of 7).
- K=4: {EC,CM,LO,Ed,At,PE} | {LL,PO} | {Co,Pn,PTX,SD} | {Fx}. W(P)=13.38; edges 4.
- K=6: {EC,CM,Ed,PE} | {LO,At} | {LL,PO} | {Co,Pn} | {PTX,SD} | {Fx}. W(P)=7.28; edges 3.
- Reference: K=1 W=31.93, edges 7; K=13 W=0, edges 0. Anti-clustering W: K=3 8.03, K=4 6.84, K=6 2.44 (best restarts 1, 1, 2).

## Headline cell table (resplit mean over 100; joint coverage target 0.90)
Coverage SD, β̂ and the independent prediction 1−0.9^(1/K) are per cell; `mean log2/13` excludes empty rows; `size` is the arithmetic mean (empty = 0); fully-decisive and empty rates are per row.

| partition (K) | method | joint cov (SD) | β̂ | 1−0.9^(1/K) | mean log2/13 | mean size | fully-dec | empty |
|---|---|---|---|---|---|---|---|---|
| K13 singletons (13) | Bonf / Šidák / max-rank | 0.9002 (0.0057) | 0.0099 | 0.0081 | 0.7245 | 2862 | 0.00006 | 0 |
| K13 singletons (13) | LP-SCP | 0.8999 (0.0055) | 0.0101 | 0.0081 | 0.5638 | 1230 | 0.0309 | 0 |
| K=1 all | Bonf | 0.9002 | 0.1280 | 0.1000 | 0.7245 | 2862 | 0.00006 | 0 |
| K=1 all | Šidák | 0.9002 | 0.1207 | 0.1000 | 0.7245 | 2862 | 0.00006 | 0 |
| K=1 all | max-rank | 0.9002 (0.0057) | 0.0997 | 0.1000 | 0.7245 | 2862 | 0.00006 | 0 |
| K=1 all | LP-SCP | 0.9006 (0.0053) | 0.0999 | 0.1000 | 0.6469 | 911 | 0 | 0 |
| clinical C (7) | Bonf / Šidák / max-rank | 0.9005 / 0.9005 / 0.9003 | 0.0166 / 0.0166 / 0.0154 | 0.0149 | 0.746 / 0.746 / 0.741 | 2813 / 2812 / 2816 | ~0 | 0 |
| clinical C (7) | LP-SCP | 0.9004 | 0.0152 | 0.0149 | 0.644 | 951 | 0 | 0 |
| clinical D (6) | Bonf / Šidák / max-rank | 0.9006 / 0.9005 / 0.9006 | 0.0198 / 0.0198 / 0.0180 | 0.0174 | 0.744 / 0.744 / 0.740 | 2888 / 2886 / 2897 | ~0 | 0 |
| clinical D (6) | LP-SCP | 0.9008 | 0.0179 | 0.0174 | 0.636 | 977 | 0 | 0 |
| **informed K=3** | Bonf / Šidák / max-rank | 0.9006 / 0.9005 / 0.9003 | 0.0422 / 0.0415 / 0.0360 | 0.0345 | 0.7269 / 0.7268 / 0.7256 | 2854 / 2853 / 2862 | 0.00017 / 0.00017 / 0.00010 | 0 |
| informed K=3 | LP-SCP | 0.9001 | 0.0355 | 0.0345 | 0.6289 | 878 | 0 | 0 |
| **informed K=4** | Bonf / Šidák / max-rank | 0.9006 / 0.9004 / 0.9006 | 0.0307 / 0.0305 / 0.0273 | 0.0260 | 0.7399 / 0.7395 / 0.7366 | 2914 / 2911 / 2902 | 0.00106 / 0.00104 / 0.00077 | 0 |
| informed K=4 | LP-SCP | 0.9003 | 0.0266 | 0.0260 | 0.6117 | 820 | 0 | 0 |
| **informed K=6** | Bonf / Šidák / max-rank | 0.9005 / 0.9005 / 0.9004 | 0.0210 / 0.0209 / 0.0185 | 0.0174 | 0.7322 / 0.7321 / 0.7275 | 2985 / 2982 / 2990 | 0.00023 / 0.00023 / 0.00011 | 0 |
| informed K=6 | LP-SCP | 0.9003 | 0.0189 | 0.0174 | 0.5778 | 897 | 0.00002 | 0 |
| anti K=3 | max-rank | 0.9002 | 0.0409 | 0.0345 | 0.7280 | 2845 | 0.00002 | 0 |
| anti K=4 | max-rank | 0.9001 | 0.0299 | 0.0260 | 0.7361 | 2777 | 0.00002 | 0 |
| anti K=6 | max-rank | 0.9002 | 0.0208 | 0.0174 | 0.7273 | 2883 | 0.00003 | 0 |
| difficulty K=3 / 4 / 6 | max-rank | 0.9002 / 0.8999 / 0.9001 | 0.0376 / 0.0279 / 0.0198 | per K | 0.7290 / 0.7380 / 0.7289 | 2851 / 2795 / 2803 | ~0 | 0 |

Full table for every (method, partition): `cell_summary.csv`. Per-cell SD, 2.5/97.5 percentiles, the H1 flag and the stratified coverage are there.

## Null and matched-null summaries (resplit-averaged mean log2/13; `null_summary.csv`)
| family | K | method | N | null mean | null p5 | null p50 | informed | rank p |
|---|---|---|---|---|---|---|---|---|
| random (primary) | 3 | max-rank | 200 | 0.7270 | 0.7230 | 0.7272 | 0.7256 | 0.259 |
| random | 4 | max-rank | 200 | 0.7377 | 0.7300 | 0.7382 | 0.7366 | 0.363 |
| random | 6 | max-rank | 200 | 0.7304 | 0.7241 | 0.7309 | 0.7275 | 0.224 |
| random | 3 / 4 / 6 | LP-SCP | 100 | 0.708 / 0.706 / 0.675 | 0.658 / 0.664 / 0.621 | 0.714 / 0.710 / 0.679 | 0.629 / 0.612 / 0.578 | 0.0099 each |
| matched (secondary) | 3 | max-rank | 200 | 0.7272 | 0.7226 | 0.7275 | 0.7256 | 0.219 |
| matched | 4 | max-rank | 200 | 0.7381 | 0.7326 | 0.7385 | 0.7366 | 0.318 |
| matched | 6 | max-rank | 151 | 0.7305 | 0.7248 | 0.7311 | 0.7275 | 0.145 |
| matched | 3 / 4 / 6 | LP-SCP | 100 | 0.683 / 0.689 / 0.651 | 0.638 / 0.647 / 0.612 | 0.683 / 0.688 / 0.650 | 0.629 / 0.612 / 0.578 | 0.020 / 0.010 / 0.010 |

## Hypotheses
- **H1 (coverage band [0.895, 0.905]):** all 4,092 p-value cells (resplit-mean joint coverage 0.89959–0.90107) and all 613 LP-SCP cells (0.89934–0.90109) in band. No "coverage-unequal" cells. Pass. (LP-SCP is not over-covering here.)
- **H2 (primary; max-rank, informed vs 200-partition random null; rank p = (1+#null≤informed)/201; Holm at 0.05):** rank p = K3 0.259, K4 0.363, K6 0.224. Holm: no rejection (thresholds 0.0167, 0.025, 0.05 against 0.224 / 0.259 / 0.363). **Not supported.** Informed max-rank is not below the null 5th percentile at any K.
  - Descriptive (not in Holm): Bonf/Šidák rank p 0.37–0.58, not significant. LP-SCP (N=100, /101): rank p 0.0099 at each K, informed below the null p5 (unadjusted, secondary). Matched null max-rank p 0.145–0.318.
- **H3 (anti-clustering above null median and informed, max-rank):** K=3 passes (anti 0.7280 > null median 0.7272 and > informed 0.7256). K=4 and K=6 fail (anti below informed and below the null median). **Mixed.**
- **H4 (descriptive, Spearman W vs mean log2/13 over null):** for LP-SCP, negative at all K (K3 −0.31, K4 −0.39, K6 −0.23; p 0.002, <0.001, 0.023). For max-rank, Bonferroni and Šidák, positive (max-rank K3 +0.08, K4 +0.15, K6 +0.14; Bonf K4 +0.21, K6 +0.22). **Direction matches H4 for LP-SCP only.**

## Paired efficiency deltas (per resplit; 2.5/97.5 percentiles over 100; `paired_deltas.csv`)
- max-rank, informed vs K=1 max-rank: K3 Δlog2/13 +0.0010 [−0.0009, +0.0027], size ratio 1.000 [0.991, 1.009]; K4 +0.0121 [+0.0091, +0.0146], size ratio 1.014 [0.999, 1.029]; K6 +0.0029 [+0.0008, +0.0052], size ratio 1.045 [1.032, 1.057].
- max-rank, informed minus random null median: K3 −0.0017 [−0.0031, −0.0003]; K4 −0.0016 [−0.0037, +0.0009]; K6 −0.0034 [−0.0055, −0.0013].
- Bonferroni, informed K4 vs K1 max-rank: +0.0153 [+0.0125, +0.0179], size ratio 1.018 [1.003, 1.033].
- LP-SCP, informed vs K1 max-rank: K3 −0.096 [−0.103, −0.089] (size ratio 0.307); K4 −0.113; K6 −0.147. LP-SCP informed minus LP random null median: −0.086 / −0.099 / −0.101 (CIs exclude 0). See caveats: this is a different score with a structural boost, not a clustering gain.

## Split-calibration arm (informed K=4; A split 50/50; 100 resplits; robustness only)
| method | split joint cov | headline joint cov | split β̂ | headline β̂ | paired Δlog2/13 (split−headline) |
|---|---|---|---|---|---|
| Bonf | 0.9010 | 0.9006 | 0.0305 | 0.0307 | +0.0010 |
| Šidák | 0.9008 | 0.9004 | 0.0304 | 0.0305 | +0.0009 |
| max-rank | 0.9010 | 0.9006 | 0.0272 | 0.0273 | +0.0010 |
| LP-SCP | 0.9011 | 0.9003 | 0.0262 | 0.0266 | +0.0015 |
Split-arm joint coverage 0.9008–0.9011 (SD 0.007). All in the band.

## Frozen draw (reference row; no pass/fail; percentile = share of resplit values ≤ frozen)
- K=1 max-rank: joint cov **0.8942**, percentile 0.15. Exp052 pre-amendment value was 0.8934; the amendment shifts it by +0.0008. Frozen mean log2/13 0.7154, percentile 0.02 (the frozen draw sits at the low end of the resplit log2 distribution; not an independent draw, since test rows are in the pool).
- K=1 LP-SCP: joint cov 0.8943 (matches Exp052), percentile 0.10.
- K=4 informed max-rank: joint cov 0.8946, percentile 0.15; mean log2/13 0.7266.
- Full rows: `frozen_draw_rows.csv`.

## Context rows (frozen draw only; reused code; `run_info.json`)
Uncalibrated α=0.10, K=1, conformal → test 13-clean: Bonf joint cov 0.9154, Šidák 0.9111, max-rank 0.8934 (pre-amendment code path, `max_rank.calibrate_max_rank`), independent per-label at α: see `run_info.json`.

## Stratified diagnostic (rare labels; SPEC caveat 7)
Joint coverage on rows with a true positive for Pleural Other, Fracture or Lung Lesion (resplit mean) vs others: K=1 max-rank 0.827 vs 0.904; K=1 LP-SCP 0.732 vs 0.910; informed K=4 max-rank 0.823 vs 0.905. Coverage is not class-conditional; these rows are under-covered.

## Secondary efficiency and decisiveness (for the record)
- Empty joint-set rate: 0 for every p-value cell (verified: no p-value cell with empty_rate > 0); ≤0.00036 for LP-SCP (max cell). No cell pair differs by > 0.5 pp (flag none).
- Fully-decisive rows are very rare for p-value methods (0.006% at K=13 and K=1; 0.01–0.1% for informed cuts, roughly 1–20 rows). Hierarchy-violation and decisive-accuracy figures on these rows are not reliable. LP-SCP has 3.1% fully decisive at K=13, 0% at K≤6.

## Caveats (verbatim list from SPEC; all apply)
1. Backbone: shared backbone inflates cross-label score correlation (EC/CM 0.936→0.997); likely overstates the efficiency gain. The gain is a property of this model's scores.
2. Guarantee vs efficiency: coverage of the min-over-clusters statistic is controlled to O(1/n) via leave-one-out ranks; exactness requires the split arm. Efficiency is empirical.
3. Population scope: 13-clean rows only; silent about the 23% with an uncertain label. Scope: studies with no uncertain label on any of the 13 categories, ~77% of test.
4. Recalibration changes meaning: Bonferroni/Šidák here are empirically recalibrated; at K=1 and K=13 they coincide with max-rank; differences at K>1 partly reflect the cross-cluster weighting convention.
5. Partition provenance: informed = marginal score correlation (S-avg), not GCM, Ding or RadGraph.
6. Uncertainty: resplit intervals are conditional on a fixed 37,631-row pool, and pooling conformal + test understates total uncertainty.
7. Rare labels: coverage is not class-conditional (see stratified diagnostic above).
8. LP-SCP: tie over-coverage may break equal coverage (LP-SCP joint coverage is within band here); hierarchy co-location may drive its gain — the matched null is the check. Here LP-SCP informed cuts are below the matched LP null (rank p 0.010–0.020), so co-location alone does not explain it, but the LP score differs from the others (Mahalanobis + Hp/Cp boost).
9. Pooling: resplits pool conformal and test; the frozen draw is a reference row only.

**Standing caveat (CLAUDE.md):** one shared-backbone classifier; efficiency numbers likely overstate the clustering gain.
**Amendment 3 interpretation (flagged):** the test-side cluster p-value now carries one count less than the standard split-conformal `(1+#)/(n+1)`. Empirical coverage is on target (0.9001–0.9008 across cells), which is consistent with the β̂ rule reserving that count, but it is not a proof of validity.
