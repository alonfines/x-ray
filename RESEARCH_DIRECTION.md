# RESEARCH_DIRECTION.md

The committed design for the experiments. Tom reads this when a consult is about *this*
project; general conformal-prediction theory questions need nothing from here.

**Refocused 2026-11-10.** The question is now the **efficiency of clustering for conformal
prediction**. Everything below is label-level and runs on cached probabilities from one
frozen model. No retraining, no re-inference.

---

## The question

**How much does grouping labels into clusters, and calibrating each cluster jointly, make
conformal prediction sets smaller (more efficient) at the same joint coverage — and which
clustering procedure buys the most?**

Coverage is the validity gate; efficiency is the result. A method or clustering that does
not hold joint coverage at 1−α has no efficiency to report.

Three parts:
1. **Clustering procedure** — how labels are grouped (which labels belong together, and m).
2. **Joint calibration** — how a cluster is calibrated so the guarantee is over the label
   *vector*, not per label.
3. **Efficiency comparison** — set size against an unclustered baseline and against random
   clusters of the same sizes, at equal α, m and score.

Max-rank (Timans et al. 2025) is the primary joint-calibration method. The structured
product-of-potentials score is the comparison that shows *when* logical structure buys
efficiency beyond a multiplicity correction.

## Model and data — frozen

One checkpoint: `densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809`.
13 labels (No Finding excluded), hierarchy correction on, trained on full MIMIC-CXR.

Cached probability CSVs in `results/outputs/mimic/`, same 29-column shape
(`subject_id, study_id, dicom_id` + `{label}_prob` / `{label}_true_label`):

| split | file suffix | rows | subjects |
|---|---|---|---|
| validation | `_validation.csv` | 18,570 | 5,898 |
| conformal | `_conformal.csv` | 18,754 | 5,926 |
| test | `.csv` | 24,394 | 6,395 |

**Splits are subject-disjoint** — `data/mimic_split.py` stratifies on subject-level label
maxima and splits subject IDs; zero subject and zero dicom overlap was verified. Every
experiment still re-runs the overlap check as its Gate 0 before computing anything.

**Split roles:** clustering procedures are fit on **validation**; thresholds are calibrated
on **conformal**; coverage and efficiency are measured on **test**. Nothing about a
partition is chosen by looking at conformal or test results.

**Backbone caveat — state it in every efficiency claim.** All probabilities come from one
shared-backbone classifier, which inflates cross-label score correlation (EC/CM 0.936 →
0.997 vs. independently trained models). Higher correlation is exactly what lets joint
calibration beat Bonferroni, so this caveat is **not conservative for efficiency**: it
likely *overstates* the set-size gain from clustering. Report the gain as a property of
this model's scores, not of the labels themselves.

## Locked design

### 1. Clustering procedures
A cluster is a small set of labels (m = 2–5) calibrated as one joint object. Candidate
procedures to compare:

- **Conditional-dependence graph** — GCM on residuals `r = y − p̃` after per-label isotonic
  calibration fit on validation; keep edges with conditional correlation |ρ| ≥ 0.10
  (effect-size floor, never a bare p-value — at n≈19k, significance clears on √n alone);
  exclude the 6 hierarchy parent–child pairs, which are correlated by construction.
- **Ding et al. (2023)** — k-means on quantile embeddings of nonconformity scores, with a
  null cluster for labels under ~200 own-positive calibration rows. The data-driven
  comparator.
- **RadGraph co-occurrence graph** — corpus-level relation graph over findings, used once,
  offline, to define a label graph. *Open slot:* not in the repo yet; say which source a
  result used.
- **Hierarchy** (`data/hierarchy.py`) — supplies hard admissibility constraints regardless
  of which procedure defines the clusters.
- **Random partition (control, mandatory)** — random clusters with the same size profile as
  the procedure under test, multiple seeds. Without it, "clustering helps" cannot be told
  apart from "any grouping helps."

Procedures and their parameters are **pre-registered before running**, with m stated.
Comparing pre-registered procedures is the point; tuning a partition against conformal or
test results is forbidden.

### 2. Joint calibration
- **Score (all methods share it):** true-class nonconformity `s = 1−p if y=1 else p`,
  defined on every row. This is what makes FWER control over m columns *exactly* joint
  coverage of the label vector. BCOPS dual thresholds cannot be used for joint work — they
  calibrate presence and absence on different row sets.
- **max-rank** (Timans et al. 2025, Algorithm 2): row-wise l∞ of ranks, rank quantile,
  column-wise order statistic. Primary method.
- **Unclustered baselines (in every comparison):** independent per-label at α (not jointly
  valid — reported to show the validity gap, not as a competitor), Bonferroni at α/m, Šidák
  at 1−(1−α)^(1/m).
- **Structured Layer 1** (comparator): `s(state) = −Σ log φ_l − λ Σ_E e_jk·agree(·,·)`,
  `s = +∞` on states violating a hard hierarchy constraint; one split-conformal quantile
  over joint states. λ = 1.0, never tuned post hoc.
- Hard hierarchy constraints are **admissibility**, never a soft term.
- Edge weights `e_jk ∈ [−1,1]`: the conditional ρ above. Below ~30–50 co-occurrences, use a
  fixed-magnitude ±0.3 sign rather than a noisy estimate.

### 3. Metric set — locked
**Validity gate:** joint coverage (true label of every member in its set) ≥ 1−α within a
binomial tolerance. A configuration that fails the gate is reported as failed; its set
size is not compared.

**Primary (efficiency):**
- mean joint set size = number of joint states retained; for per-label methods, the
  *product* of per-label set sizes (without this conversion the families are not comparable);
- normalized size log₂(set size)/m, so clusters of different m are comparable;
- efficiency ratio vs. the unclustered Bonferroni baseline and vs. the random-partition
  control, at equal α, m and score;
- set-size distribution: median and fully-decisive rate (share of single-state sets).

**Uncertainty:** confidence intervals on every efficiency delta from repeated subject-level
resplits (or a subject-level bootstrap). A single split does not support a set-size claim.

**Secondary:** hierarchy-violation rate of fully-decisive output; decisive accuracy.

Retired: marginal per-label coverage; cluster-assertion coverage on `Z_c = OR over members`
(it collapses a cluster to one binary and cannot distinguish a method that gets the label
*vector* right from one that does not).

Rows with any uncertain (−1) member are excluded from joint metrics.

### 3a. Calibration population caveat — standing, affects every result on this project
`conformal_split.csv` contains **only rows with no uncertain (−1) value on any of the 13
CheXpert labels** — dropped at split-construction time, `data/mimic_split.py:162-164`,
before inference ever ran. All conformal thresholds are therefore calibrated on this
restricted population (n = 18,754). Test evaluation uses the per-cluster-member exclusion
rule above, which is a **broader** population (n = 20,318–23,630 depending on m). The two
are not exchangeable, and the gap is not hypothetical: Exp046 measured max-rank at exactly
nominal in-sample (0.9001–0.9002 on all four clusters) and 0.8917–0.8993 out-of-sample,
with the mismatch as the identified cause.

Any coverage claim must therefore either (a) state this population mismatch explicitly as
a limitation, or (b) be evaluated against the all-13-clean-restricted test subpopulation
as a secondary check — and if (b), state the scope ("studies with no uncertain label on
any of the 13 categories, 77% of test"), never bare. (b) is a robustness check only, never
a headline: it is silent about the harder 23%, where uncertainty correlates with genuinely
ambiguous findings and joint calibration plausibly matters most.

Efficiency claims that rest on the **effective support of the joint label vector** inherit
this caveat too. Exp047 measures 13 of 32 states occupied at m = 5 and uses that sparsity
to explain LP-SCP's efficiency — but it is measured on the all-13-clean population. The
true support may be larger. State it as a property of the filtered population, never as a
fact about how chest-X-ray findings co-occur.

**Methods that fit their score function on a split inherit this bias twice** — once in the
score, once in the threshold. LP-SCP (Exp047) fits Σ, Hp and Cp on validation, so its
nonconformity measure *and* its quantile are both built on all-13-clean populations.
Measured magnitude stays inside the same ~1pp band.

The eventual fix is a corrected re-export of conformal **and** validation probabilities
preserving per-cluster-member filtering — one forward pass on the frozen checkpoint, not
retraining. Both splits must be re-exported in the same pass or the bug simply relocates.
**Not yet approved: escalated to Alon, 2026-10-28.**

### 4. Prior observations — reproduction targets, not evidence
From a run deleted on 2026-10-28; if a rerun disagrees, the rerun wins. At α=0.1:

- Independent per-label calibration had **no joint validity**: ≈0.66 at m=5, ≈0.76 at m=3
  against a 0.90 target.
- max-rank landed on nominal (≈0.90) where Bonferroni/Šidák over-covered (0.90–0.94), with
  modestly smaller sets than Bonferroni.
- Šidák bought almost nothing over Bonferroni (a few tenths of a pp): the efficiency
  headroom is **label dependence and admissibility**, not union-bound looseness. Keep Šidák
  in every comparison for this reason.
- Structured beat max-rank on set size **only where hard constraints exist**, monotonically
  in their number. The honest claim is conditional, never "structured wins."
- Structured won on mean set size but was *less* often fully decisive than Bonferroni,
  whose set-size distribution is bimodal. Report both.

### 5. Open items
- Efficiency vs. cluster size m: where does joint calibration stop paying for itself
  (state-space growth 2^m vs. calibration rows per occupied state)?
- Each clustering procedure vs. the random-partition control at equal m, with CIs.
- Whether RadGraph, conditional-dependence and Ding clusters differ in efficiency at equal m.
- Soft edges were exercised in only one cluster; the ψ term is under-tested.
- λ never swept.

### 6. What not to do
- Don't propose retraining or re-inference beyond the approved re-export above.
- Don't report an efficiency number without its coverage gate, its baseline and its CI.
- Don't search partitions or parameters on conformal/test results.
- Don't quote numbers from the deleted pre-2026-10-28 experiments as results.
- Keep the two guarantees separate in writing: conformal machinery controls coverage;
  efficiency gains are empirical, never a proven guarantee.

## Historical record
Folders 032–045 were deleted in the 2026-10-28 clean restart (backup tarball:
`~/Desktop/xray_pre_reset_backup_20261028_104145.tar.gz`). Do not Glob for or cite them.
Current experiments live in `experiments/046` onward; new work takes the next sequential
number.

The older archive `MAX_RANK_EXPERIMENTS_SUMMARY.md` (Exp001–031) matters for one fact:
**max-rank fails for labels with <150–200 positive calibration samples** (Fracture,
reproducibly). That is a data-size limit, not a tuning problem, and it constrains which
clusters are worth pre-registering.
