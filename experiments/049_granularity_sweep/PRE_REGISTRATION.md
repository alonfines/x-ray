# Exp049 — Granularity sweep: six-level partition, 13→1, joint coverage vs decisiveness

**Pre-registered:** 2026-11-01. Tom Hagan spec, exactly as specified. No design decisions post-facto.

## One blocking fact

`conformal_common.load_clusters()` hard-asserts `2 <= m <= 5` on every `ClusterSpec`. Tom's spec requires m=1 (13 singletons) and m=6, m=13 (beyond 5) at some levels. This assertion is bypassed deliberately by constructing `ClusterSpec` objects directly in the orchestrator, not via the YAML loader. This is a one-time pre-registered exception; every other experiment retains the 2–5 guard.

## Partition design — pre-registered, grounded in `data/hierarchy.py`

| Level | n_clusters | Clusters (name: labels, m) | Logic |
|---|---|---|---|
| L0 | 13 | 13 singletons, m=1 each | Baseline — no clustering, independent per-label. Anchor point: all three methods must coincide numerically here (sanity check). |
| L1 | 12 | `cardio`: {Enlarged Cardiomediastinum, Cardiomegaly}, m=2; 11 singles | Direct hierarchy parent–child edge, the single most literally coupled pair in the model (shared-backbone ρ 0.936→0.997). |
| L2 | 10 | `cardio` m=2; `pneumonia_tri`: {Lung Opacity, Consolidation, Pneumonia}, m=3; 8 singles | Pneumonia's two hierarchy parents + itself — second documented hierarchy structure. |
| L3 | 7 | `cardio` m=2; `lung_hub`: {Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis}, m=6; 5 singles (Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices) | Absorbs the remaining Lung-Opacity children. This is the complete hierarchy-connected-components partition — no hierarchy edges remain to merge on. Also exactly the 6-member family the deleted archive found population-dilutes (Lung Lesion) under naive pooling and under max-rank (Exp001/005/031) — a direct reproduction check. |
| L4 | 3 | `cardio` m=2; `lung_hub` m=6; `residual`: {Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices}, m=5 | Null/control grouping — these five have no documented hierarchy or (unbuilt) dependence edge among them. Forced merge, not a dependence claim. Flag in every table/plot touching this level. |
| L5 | 1 | `all13`: all 13 labels, m=13 | Coarsest endpoint. Forced, not evidence-based. Predicted to show severe dilution/abstention, especially Fracture and Pleural Other. |

Cluster-count sequence: 13, 12, 10, 7, 3, 1 — real spacing, each step's logic stated, no search.

## Arms per method (pre-registered, no search)

- **max_rank**: no arms. One run per level.
- **per_label (bonferrony)**: run all three corrections (`none`, `bonferroni`, `sidak`) at every level. Headline arm = `bonferroni`. `none` is the motivating invalid baseline — record it, never plot it as a competing method line. `sidak` recorded for the appendix.
- **lp_scp**: run all three arms (`base`, `penalised`, `admissible`) at every level. Headline arm = `admissible` (μ=ν=1, hard hierarchy constraint). `base`/`penalised` recorded for completeness, not plotted.

Headline plot = exactly 3 lines: `max_rank`, `per_label[bonferroni]`, `lp_scp[admissible]`.

## Alpha and metrics

- **α = 0.1** (repo convention, confirmed)
- **Locked score:** true-class nonconformity `s = 1−p if y=1 else p` (all methods share this)
- **Primary metrics:** joint coverage, mean joint set size, hierarchy-violation rate of fully-decisive output
- **Secondary metrics:** fully-decisive rate, decisive accuracy

## CSV schema — pre-registered

**Raw collector CSV** (`results_raw.csv`), one row per (level, cluster, method, variant):

Columns: `level` (0–5), `n_clusters` (13/12/10/7/3/1), `cluster_name`, `m`, `method` (`max_rank`/`per_label`/`lp_scp`), `variant` (`""`/`none`/`bonferroni`/`sidak`/`base`/`penalised`/`admissible`), `alpha`, `alpha_effective` (bonferrony only), `n_eval`, `n_dropped_uncertain`, `joint_coverage`, `mean_joint_set_size`, `mean_label_set_size` (NaN where undefined, i.e. lp_scp), `fully_decisive_rate`, `decisive_accuracy`, `hierarchy_violation_rate_decisive`, `degenerate` (k>n flag, bonferrony), `min_positive_per_label` (min over the cluster's `positives_per_label()`), `below_min_support_flag` (True if `min_positive_per_label < 150`).

**Aggregate CSV** (`results_aggregate.csv`), one row per (level, method+headline-variant) — exactly what the plot reads:

Columns: `level`, `n_clusters`, `method`, `joint_coverage_wmean`, `fully_decisive_rate_wmean`.

Aggregation rule: weight each cluster by its own `m`, normalized so weights sum to 13 (every label appears in exactly one cluster per level, by partition construction — this is a label-weighted mean, not a cluster-count-weighted mean).

## Plot spec — pre-registered

**Plot** (`granularity_tradeoff.png`):
- x-axis: `fully_decisive_rate_wmean`. y-axis: `joint_coverage_wmean`.
- One line per method (3 lines, headline arms only), points ordered by `n_clusters` descending (13→1), each point labeled with its `n_clusters` value directly on the plot.
- Horizontal reference line at y = 1−α = 0.90.
- Caption: "bonferrony = Bonferroni arm; lp_scp = admissible arm (μ=ν=1, hard hierarchy constraint); independent/Šidák/base/penalised arms are in results_raw.csv only. L4 and L5 clusters are forced/null groupings, not dependence claims."
- House plot-design guideline (18×11in, ≥28pt titles, ≥22pt axis labels, ≥18pt tick labels, ≥17pt annotations).

## Evaluation and calibration splits

- **Calibration:** conformal split, 18,754 rows / 5,926 subjects (frozen CSV `_conformal.csv`)
- **Evaluation:** test split, 24,394 rows / 6,395 subjects (frozen CSV `_test.csv`)
- **Rows dropped:** Any row with uncertain (−1) or missing member label (locked exclusion rule for joint metrics)

## Gate 0

Run before any number is computed: train/conformal/test subject_id and dicom_id disjointness. Report in REPORT.md.

## Caveats (to be restated in REPORT.md)

1. Backbone-correlation inflation (cardio pair, pneumonia triangle, lung hub) — understates rather than inflates the independent baseline's failure, conservative, but must be named every time a coarser-but-hierarchy-grounded cluster shows a gain.
2. §3a calibration-population mismatch — applies at every level; expect out-of-sample joint coverage a bit below nominal everywhere (Exp046 found 0.8917–0.8993 vs. 0.90 in-sample).
3. L4 (`residual`) and L5 (`all13`) are forced/null groupings — never write "the model found these labels related."
4. Expect — don't force, don't suppress — the known population-dilution failure to reappear once Fracture/Pleural Other enter a multi-member cluster at L4/L5.
5. LP-SCP fits Σ/Hp/Cp on validation at m=6/m=5/m=13 with no more data than smaller clusters — report the COND_LIMIT ridge trigger as a diagnostic, don't tune it away.

## Out of scope

- Do not modify the pipeline (train.py, test.py, data/, losses/, configs/)
- Do not reimplement scoring logic — reuse existing functions from conformal_common.py, max_rank.py, bonferrony.py, lp_scp.py
- No parameter search, no arm selection on outcome, no dropping levels because they look bad

## Input files

- Frozen CSVs: `/Users/alonfines/Desktop/x-ray/results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_{conformal,test,validation}.csv`
- Hierarchy: `/Users/alonfines/Desktop/x-ray/data/hierarchy.py`
- Calibration functions: `max_rank.py`, `bonferrony.py`, `lp_scp.py`

## Reproduce

```bash
cd /Users/alonfines/Desktop/x-ray
python experiments/049_granularity_sweep/orchestrator.py
```

Outputs:
- `experiments/049_granularity_sweep/results_raw.csv`
- `experiments/049_granularity_sweep/results_aggregate.csv`
- `experiments/049_granularity_sweep/granularity_tradeoff.png`
- `experiments/049_granularity_sweep/REPORT.md`

## Git commit

TBD (will add after successful run)
