# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DenseNet-based chest X-ray classification model using the **MIMIC-CXR-JPG** dataset. The project trains a unified multi-label deep learning model to classify 13 pathological conditions from chest radiographs, with hierarchy label correction and conformal prediction calibration.

**Current state:** Unified model trained on full MIMIC dataset with hierarchy correction; its probabilities are frozen. Research now studies how clustering labels makes joint conformal prediction sets smaller (see Research Direction). MIMIC is the main dataset; other multi-label datasets (Yeast first, 2026-11-23) are added to test that the clustering gain carries over.

## Repository Layout (multi-dataset, 2026-11-23)

```
x-ray/
├── models/<dataset>/     # code that produces probabilities for one dataset (download, train, export)
│   ├── mimic/            # DenseNet pipeline (was classifier/ until 2026-11-23)
│   └── yeast/            # fetch.py (OpenML 40597 → data/yeast/raw/); model not written yet
├── data/<dataset>/       # data only, git-ignored CSVs
│   ├── raw/              # the dataset as downloaded (Yeast: raw/yeast.csv)
│   └── probs/            # frozen {stem}_{train,validation,conformal,test}.csv, the input to conformal/
├── conformal/            # dataset-agnostic; picks a dataset by name from data.DATASETS
└── experiments/NNN_<dataset>_<slug>/   # from Exp061 on, the dataset is in the folder name
```

**Rules:**
1. Every dataset's `probs/` CSVs share one schema: a group column, a row-id column, and
   `{label}_prob` / `{label}_true_label` per label, in the same 4 splits (train for the model,
   validation for scorers, conformal + test pooled for calibration/test draws).
2. A dataset is registered once in `conformal/data.py` `DATASETS`: labels, hierarchy (`{}` if none),
   group column, id column. `load(dataset=...)` returns `Data` carrying its `labels` and `hierarchy`;
   `run.run(..., dataset=...)` passes it through. No other module knows which dataset it is on.
3. Gate 0 runs for every dataset (group- and row-disjoint splits; hierarchy check when one exists).
   With no hierarchy, `share_impossible` and `violation_rate_decisive` are trivially 0.
4. Data never lives in `models/` and code never lives in `data/`.

| dataset | rows | labels | hierarchy | group | status |
|---|---|---|---|---|---|
| mimic | 192k images | 13 CheXpert | yes | `subject_id` | probs frozen |
| yeast | 2,417 | 14 (`label_1`–`label_14`), 103 features | none | `id` | raw only, no model yet |

## Conformal Package (`conformal/`)

All conformal research code, shared by every dataset. It reads only the frozen CSVs in `data/<dataset>/probs/`, never a model.
Run locally with `.venv/bin/python`. One idea organises it: **partition → score per cluster → rule
across clusters → product region**. Classical per-label conformal is the special case "every label its
own cluster, `per_label` score".

```
conformal/
├── data.py        # DATASETS registry; load(dataset=) splits + Gate 0 (group/row disjoint, no
│                  # hierarchy-violating true vectors); all-labels-clean population; joint states;
│                  # admissibility; draws (frozen + group-level resplits)
├── partition.py   # singletons(), random_partition(), search() (exact DP min-CE partition per m_max)
├── scores/        # fit(train, labels, seed) -> f(data) = (n, 2^m) state scores
│   ├── per_label.py   # true-class score, one-label clusters
│   ├── lp_scp.py      # Katsios & Papadopoulos structural-penalty score (mu, nu)
│   └── mlp.py         # learned joint-state MLP (plain torch); no hierarchy information
├── quantile.py    # k = ceil((n+1)(1-alpha)); calibrate(scores, alpha, rule),
│                  # rule = none | bonferroni | sidak | max_rank (Timans Alg. 2)
├── regions.py     # per-cluster regions (optional non-empty guard), joint product without enumeration;
│                  # counts the set's hierarchy-admissible vectors (measured, never enforced)
├── evaluate.py    # metrics (incl. share_impossible), resplit summaries, paired CIs
└── run.py         # run(partition, score, rules, alpha, guard, n_resplits) -> DataFrame
```

```python
from conformal import run, partition, evaluate
df = run.run(partition.singletons(), "per_label", rules=["bonferroni", "max_rank"])   # MIMIC (default)
evaluate.summarize(df)
# another dataset: run.run(partition.singletons(DATASETS["yeast"].labels), ..., dataset="yeast")
```

**Fixed conventions:** population is all-labels-clean rows in every split (for MIMIC ~77% of test; say so with every
MIMIC result). Scorers are fit on validation; calibration/test draws are the frozen conformal/test split
(`resplit = -1`) plus subject-level 50/50 resplits of the pooled conformal ∪ test rows (`default_rng(r)`).
**No hierarchy enforcement anywhere (2026-11-17).** Scores learn label dependencies from the data; the
hierarchy mask in the MLP and the admissible-set filter were removed after Exp058 showed the unmasked
MLP never puts an impossible vector in a set. The hierarchy is only *measured* (`share_impossible`,
`violation_rate_decisive`). This applies to LP-SCP too: its Σ, Hp and Cp are fit on validation only, from
the observed label vectors (never `models.mimic.data.hierarchy`), with no +∞ scores. No scorer is ever fed the
hierarchy, so the MLP's learning of it stands on its own. Gate 0 still checks that no true label vector violates it.

**Verified on 2026-11-23 after the multi-dataset move:** per-label (4 rules, 20 resplits), LP-SCP
μ=ν=2 and MLP arms give bit-identical results before and after.

**Verified on 2026-11-17 against the old code:** per-label none/Bonferroni/Šidák/max-rank, max-rank
∩ admissible (option since removed) and LP-SCP match exactly on the frozen draw. The 13-label max-rank references reproduce
Exp056 exactly over 100 resplits (sizes 2855.40 and 850.00). The MLP arms reproduce Exp056 within seed
noise (arm C: coverage 0.8998, size 49.6 vs 49.4). The DP search gives the saved partitions, and the
MLP holdout CE is identical to six decimals.
The old modules (`conformal_common.py`, `conformal_eval.py`, `compare_methods.py`,
`bootstrap_coverage.py`, `conformal_methods/`, `find_partitions.py`) were deleted; recoverable from git `98b8269`.

## MIMIC Model (`models/mimic/`)

Everything that trains, tests, or runs the DenseNet lives in `models/mimic/` (named `classifier/`
until 2026-11-23) — separate from the conformal research code in `conformal/`, which only reads the
frozen probability CSVs. Run its scripts as modules from the repo root (`python -m models.mimic.<script>`).
The cluster checkout and its runai commands must use the new path.

```
models/mimic/
├── train.py, test.py, inference.py, densenet_model.py, export_split_scores.py
├── download_mimic_cxr.py, mimic-cxr_scripts/   # data acquisition / prep
├── configs/             # config.yaml (base), config_hier_unified.yaml (unified hier model)
├── losses/              # BCE loss
├── data/                # dataset package (below)
└── reports/             # MIMIC free-text reports (files/, *.zip: git-ignored, credentialed),
                         # partition_search/ (DP label partitions), mlp_pipeline.{png,svg}
```

The conformal side imports the MIMIC labels and hierarchy from `models.mimic.data`.

### Data Package (`models/mimic/data/`)

```
models/mimic/data/
├── __init__.py          # Shared constants (ALL_CHEXPERT_LABELS), parse_labels_config(), get_data_module() factory
├── split_utils.py       # Shared stratification & statistics helpers
├── hierarchy.py         # Disease hierarchy definition & label correction
├── mimic.py             # MIMICDataset + MIMICDataModule
└── mimic_split.py       # MIMIC splitting script (scans disk for existing files, filters by view)
```

- **`get_data_module(config_path)`** is the main entry point. Returns the MIMIC DataModule.
- **`parse_labels_config(config)`** returns `(pathologies, use_all_labels)` based on `config['labels']`.
- Consumers (`train.py`, `test.py`, `export_split_scores.py`) import from `models.mimic.data`.

## Data Structure

### CSV Organization

Split CSVs under `csv_files/mimic/`:

```
csv_files/mimic/
├── train_split.csv
├── valid_split.csv
├── conformal_split.csv
└── test_split.csv
```

### MIMIC-CXR-JPG Raw Data

Located at `mimic.data_dir` (models/mimic/configs/config.yaml): `/gpfs0/tamyr/projects/data/MIMIC-CXR/`

```
MIMIC-CXR/
├── csv/
│   ├── mimic-cxr-2.0.0-metadata.csv    # Image-level metadata (dicom_id, subject_id, study_id, ViewPosition)
│   ├── mimic-cxr-2.0.0-chexpert.csv    # Study-level CheXpert labels (subject_id, study_id, 14 labels)
│   └── mimic-cxr-2.1.0-test-set-labeled.csv
└── files/
    └── p{XX}/p{subject_id}/s{study_id}/{dicom_id}.jpg
```

### CSV Format

Uses the 14 CheXpert pathology labels:
- No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices
- Values: 1.0 (positive), 0.0 (negative), -1.0 (uncertain), empty (unlabeled)

**MIMIC CSVs** have `dicom_id`, `subject_id`, `study_id` columns; image paths are constructed from these.

### Dataset Splitting Strategy

The dataset is split into four stratified sets:

1. **Train Set**: 70% for model training
2. **Validation Set**: 10% for validation (early stopping)
3. **Conformal Set**: 10% for conformal prediction calibration
4. **Test Set**: 10% for final held-out evaluation

**Label Filtering:**
- Train/Validation/Conformal splits: Uncertain labels (-1.0) are **excluded**. Only positive (1.0) and negative (0.0) labels are retained for clean training data.
- Test split: Uncertain (-1.0) labels are **preserved** at load time (`clean_uncertain=False`) and filtered out per-pathology when computing AUC.

**MIMIC-specific:**
- Filtered to frontal views only (AP + PA) via `mimic.views` in config.yaml
- `mimic_split.py` scans the disk for actually downloaded files before splitting (download may be ongoing)
- Labels are joined from metadata CSV (image-level) + CheXpert CSV (study-level) on `(subject_id, study_id)`

**Creating splits:**
```bash
python -m models.mimic.data.mimic_split
```

All splits use Mondrian-style stratification based on top pathology prevalence.

## Development Workflow

**HPC Environment - Login Node vs. Compute Nodes**

The login node is for data transfer and job submission ONLY. Do NOT run Python scripts, training, or computationally intensive operations on the login node. All work must be submitted to compute nodes via `runai`.

1. Edit and commit code on the login node
2. Push to git repository
3. Submit training jobs to compute nodes using `runai` (never run training directly on the login node)
4. Monitor job status with runai commands

**Local environment (Mac):** `.venv/` (Python 3.14, git-ignored) has numpy, pandas, scipy,
scikit-learn, torch and torchxrayvision, enough to run the conformal code on the frozen CSVs
(`.venv/bin/python ...`). It lacks the `lightning` package, so `models/mimic/` training and
inference run on the cluster only.

## Quick Start

**For Running on Compute Nodes (via runai):**

1. Train the model:
```bash
python -m models.mimic.train --config models/mimic/configs/config_hier_unified.yaml
```

2. Run test evaluation:
```bash
python -m models.mimic.test --config models/mimic/configs/config_hier_unified.yaml
```

3. Export train/validation/conformal scores:
```bash
python -m models.mimic.export_split_scores --config models/mimic/configs/config_hier_unified.yaml
```

Outputs go to `data/mimic/probs/` (paths are anchored on `data.working_dir` in the config).
The old model-based per-class calibration (`calculate_conformal_pred.py`, `conformal_calibration/`)
was deleted 2026-11-17; a general `computing_conformal_quantile` module is planned on the conformal side.

## Hierarchy Label Correction

CheXpert labels suffer from a labeling artifact: when a child finding is positive but the parent was "not mentioned" in the report, the parent gets NaN (mapped to 0). This creates contradictory training signals (e.g., Edema=1 but Lung Opacity=0). Analysis of the MIMIC training set shows 52-96% of child-positive samples have missing parent labels.

**Hierarchy correction** propagates child-positive → parent-positive. When `hierarchy_correction.enabled: true`, the DataModule corrects labels in the DataFrame after loading (all splits including test).

**Disease hierarchy** (defined in `models/mimic/data/hierarchy.py`):
- Enlarged Cardiomediastinum ← Cardiomegaly
- Lung Opacity ← Lung Lesion, Edema, Consolidation, Atelectasis
- Lung Opacity + Consolidation ← Pneumonia

**Implementation:**
- `models/mimic/data/hierarchy.py`: `apply_hierarchy_correction_to_df()` corrects a DataFrame in-place. Only definite positives (1.0) trigger propagation — uncertain (-1) and NaN do not.
- `models/mimic/data/mimic.py`: DataModule reads `hierarchy_correction.enabled` from config and applies correction in `setup()`.
- `train.py`: Adds `-hier-` suffix to checkpoint filenames when hierarchy correction is enabled.

## Key Implementation Notes

Paths below without a prefix are under `models/mimic/`.

- **data/__init__.py**: Exports `ALL_CHEXPERT_LABELS`, `parse_labels_config()`, and `get_data_module()` factory.
- **data/mimic.py**: `MIMICDataset` + `MIMICDataModule`. Builds image paths from `(subject_id, study_id, dicom_id)` columns. All images come from a single `files_dir`.
- **data/mimic_split.py**: Joins metadata + labels, filters to configured views (AP/PA), scans disk for existing files, then splits. Re-runnable as download progresses.
- **data/split_utils.py**: Shared Mondrian stratification logic and split statistics printing.
- **data/hierarchy.py**: Disease hierarchy (`CHEXPERT_HIERARCHY`), `apply_hierarchy_correction_to_df()`.
- **configs/config.yaml**: Base config. `mimic:` section has MIMIC paths and `views` filter. `labels: "all"` enables 13-label mode (excluding No Finding) with No Finding preprocessing.
- **configs/config_hier_unified.yaml**: Hierarchy-enabled config for the unified model. Points to the hier checkpoint and enables hierarchy correction.
- **config.yaml** (cont.): Optimizer hyperparameters under `optimizer:`, pretrained-backbone loading under `loss:` (`pretrained_checkpoint`, `reinit_classifier`), checkpoint for inference under `evaluation:`.
- **losses/**: BCE loss implementation. `get_loss_function(weighted, pos_weight)` factory; `calculate_pos_weights()` computes pos_weights from training labels.
- **densenet_model.py**: `CXRDenseNet` model class (DenseNet wrapper with torchxrayvision pretrained weights).
- **train.py**: Trains on train_split, validates on valid_split. Uses `get_data_module()` to load the dataset. Supports `reinit_classifier` (re-initialize the classifier head when loading a pretrained checkpoint) and WandB logging via `WandbLogger`.
- **inference.py**: Shared helpers used by `test.py` and `export_split_scores.py`: `load_model()` (checkpoint lookup + eval-mode model + device), `predict()`, `build_results_df()`.
- **test.py**: Runs inference on test_split, saves `{stem}_test.csv` to `data/mimic/probs/`, prints per-label AUC (filtering uncertain labels).
- **plots/**: the kept method-comparison figures (PNG only, no plotting scripts). Pre-2026-10-28 plots are only in
  `~/Desktop/xray_pre_reset_backup_20261028_104145.tar.gz`.
- **models/mimic/reports/partition_search/**: `partitions.json` and `cluster_costs.csv`, the saved output of the DP partition search (now `conformal.partition.search`). `models/mimic/reports/files/` and `models/mimic/reports/*.zip` hold the MIMIC free-text reports — git-ignored, credentialed data, never commit.

## Trained Model

Single unified checkpoint: `checkpoints/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809.ckpt`
- Trained on full MIMIC-CXR dataset with hierarchy correction enabled
- Test inference CSV: `data/mimic/probs/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_test.csv`

All four splits are exported, flat in `data/mimic/probs/`, named off the checkpoint
stem (`{stem}_{train,validation,conformal,test}.csv`). Each has
`subject_id, study_id, dicom_id` + `{label}_prob` / `{label}_true_label` for 13 labels:

| split | rows | subjects |
|---|---|---|
| train | 130,684 | 41,416 |
| validation | 18,570 | 5,898 |
| conformal | 18,754 | 5,926 |
| test | 24,394 | 6,395 |

**Splits are subject-disjoint.** `models/mimic/data/mimic_split.py` stratifies on subject-level
label maxima (`groupby('subject_id')[pathologies].max()`) and splits subject IDs, then
maps rows back — verified zero subject and zero image overlap across all three pairs.
This fixed a subject-level leakage problem in the earlier partial-data splits (32.8% of
test subjects also appeared in calibration), which had invalidated the coverage
guarantees of every experiment built on them. Every new experiment re-verifies the overlap
as its Gate 0.

- **export_split_scores.py**: Exports train / validation / conformal predictions for the unified
  model (`--splits` selects a subset). `python -m models.mimic.export_split_scores --config models/mimic/configs/config_hier_unified.yaml`.

## Per-Label Pipeline (retired and deleted)

The per-label one-vs-rest pathway is **retired and will not be retrained** (Alon,
2026-10-21). All conformal work runs on unified-model probabilities. Standing caveat: the
unified model's shared backbone inflates cross-label probability correlation (Enlarged
Cardiomediastinum/Cardiomegaly 0.936 → 0.997, Consolidation/Pneumonia 0.832 → 0.971), so
any dependence claim computed on it partly measures the backbone — state this, don't
retrain around it.

Its code (`utils.py`, `export_scores.py`, `train_per_label.py`, `calibrate_per_label.py`,
`test_per_label.py`, `configs/per_label*/`) was deleted in the /organize cleanup; it is
recoverable from git commit `3662897`. Its checkpoints were partial-data and leaky; don't
resurrect them.

## MIMIC-CXR Data Utilities

### Download

- **`models/mimic/download_mimic_cxr.py`**: Downloads MIMIC-CXR-JPG from PhysioNet via wget. Supports `--subset p10,p11` for partial downloads, `--views AP,PA` filtering, skip-existing via `.downloaded_manifest`, `--verify` mode, and parallel downloads.

### Prep Scripts (`models/mimic/mimic-cxr_scripts/`)

- **`check_mimic_coverage.py`**: Verifies how many study records have matching images on disk; reports view distribution.
- **`create_filtered_mimic_csv.py`**: Scans disk for existing files, applies "No Finding correction" (zeros out other pathologies when No Finding=1.0), saves `mimic_cxr_filtered.csv`.
- **`filter_to_pa_views.py`**: Filters `mimic_cxr_filtered.csv` to frontal views (PA/AP) using metadata CSV.
- **`reorganize_mimic_cxr.py`**: Moves flat-downloaded JPGs into the proper `files/p{XX}/p{subject_id}/s{study_id}/` hierarchy.

## Results & Visualization

`data/mimic/probs/` holds the four frozen probability CSVs (train / validation /
conformal / test) — these are the inputs to all research work and are not regenerated.
`plots/` holds the three kept method-comparison figures — `maxrank_vs_bonferroni.png`,
`maxrank_vs_bonferroni_vs_lpscp_mu2nu2.png`, `mlp_vs_maxrank_vs_bonferroni_vs_lpscp.png`. Their
plotting scripts were deleted (2026-11-17); the numbers behind them are in the experiment reports
(046, 047, 055). New figures follow the Plot Design Guideline below.

## Research Direction

The focus (as of 2026-11-10) is the **efficiency of clustering for conformal prediction**:
how much clustering labels and calibrating each cluster jointly shrinks prediction sets at
the same joint coverage, on the unified model's cached probabilities. The committed design
lives in the latest experiment `SPEC.md`/`README.md` files. `pdfs/` (the thesis proposal
`MOST_2026July1.txt`, the max-rank paper text and reference PDFs), `third_party/CheXbert` and
`configs/clusters/` were removed 2026-11-17 as part of the renovation (pdfs and cluster
configs recoverable from git `98b8269`); `RESEARCH_DIRECTION.md`, `THESIS_LOCK_PLAN.md` and
`MAX_RANK_EXPERIMENTS_SUMMARY.md` are retired (deleted 2026-11-17, recoverable from git `b85b288`).

Latest: Exp056 (three-cluster union), Exp057 (additive factorized score on the m_max=6 DP
partition from `models/mimic/reports/partition_search/`; blocked at pre-registration, see its `ESCALATION.md`),
Exp058 (MLP scorer with vs without the hierarchy mask on the DP m_max=6 partition: no difference;
the unmasked MLP never puts an impossible vector in a set). Exp058 onward run on `conformal/`.

**Experiments are reports only (2026-11-17).** All code, CSV/JSON results and logs in
`experiments/046`–`057` were deleted as convoluted and error-prone; only the markdown
(REPORT/README, SPEC, PRE_REGISTRATION, ESCALATION) and plots remain. Treat their numbers as
findings to build on, not something reproducible from this repo. The code is recoverable from
git `98b8269`; the result CSVs were git-ignored and are gone. New experiments are written on the
streamlined shared modules.

**Clean restart, 2026-10-28.** Folders 032–045, the Exp032–042 summaries, and all
plots/plotting scripts were deliberately deleted. Current experiments are
`experiments/046` onward. Backup tarball:
`~/Desktop/xray_pre_reset_backup_20261028_104145.tar.gz`.

## Agents

- **Tom Hagan** (`.claude/agents/tom-hagan.md`): conformal-prediction and mathematical-
  statistics consultant. Does NOT write code. Rules on validity of coverage guarantees,
  score design, joint vs. marginal coverage, clustering choices, and how to state a claim
  honestly. Reads the implemented algorithms and experiment SPECs on demand. Invoke for design decisions and for
  reviewing Jules's results or escalations.
- **Jules** (`.claude/agents/jules.md`): experiment runner. Executes exactly Tom's spec on
  the frozen CSVs, pre-registers before running, reports honest numbers. Designs nothing —
  no choosing clusters, metrics, methods, α or λ. Never modifies the pipeline. Escalates via
  `ESCALATION.md` rather than iterating variants.

- **Winston Wolf** (`.claude/agents/winston-wolf.md`): the cleaner. After an experiment is
  done and Tom has signed off, generalizes its reusable code (scorers, loaders, metrics)
  into the `conformal/` package. Refactor only: verifies numbers are
  unchanged, keeps current import paths working, never edits finished experiment folders,
  never deletes without Alon's approval.

**Workflow (cleanup step):** after Tom signs off on a Jules experiment, invoke Winston on
that folder; surface his move table and deletion candidates to Alon.

**Workflow:** Tom designs an experiment (grounded in the implemented algorithms and prior experiment reports) → Jules
pre-registers and runs it in `experiments/<NNN>_<slug>/`, reporting joint coverage (validity
gate) and efficiency (joint set size vs. unclustered and random-cluster baselines, with CIs)
→ sound result feeds Tom's next decision;
unclear or unsound result becomes an `ESCALATION.md` for Tom.

**Plot Design Guideline:** All plots must be readable by a 50-year-old woman without glasses, looking at a small laptop screen through a Zoom call. This means:
- Large figure size (18×11 inches minimum)
- Bold, oversized fonts: titles ≥28pt, axis labels ≥22pt, tick labels ≥18pt, annotations ≥17pt
- High-contrast color palettes (e.g., `mako`)
- Percentage annotations directly above each bar
- Y-axis gridlines for readability
- Rotated x-axis labels (45°) to avoid overlap
- White background, 300 DPI export
