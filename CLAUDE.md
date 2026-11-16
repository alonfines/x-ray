# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DenseNet-based chest X-ray classification model using the **MIMIC-CXR-JPG** dataset. The project trains a unified multi-label deep learning model to classify 13 pathological conditions from chest radiographs, with hierarchy label correction and conformal prediction calibration.

**Current state:** Unified model trained on full MIMIC dataset with hierarchy correction. Next step: experiment with different conformal prediction methods.

## Data Package (`data/`)

All dataset logic lives in the `data/` package:

```
data/
├── __init__.py          # Shared constants (ALL_CHEXPERT_LABELS), parse_labels_config(), get_data_module() factory
├── split_utils.py       # Shared stratification & statistics helpers
├── hierarchy.py         # Disease hierarchy definition & label correction
├── mimic.py             # MIMICDataset + MIMICDataModule
└── mimic_split.py       # MIMIC splitting script (scans disk for existing files, filters by view)
```

- **`get_data_module(config_path)`** is the main entry point. Returns the MIMIC DataModule.
- **`parse_labels_config(config)`** returns `(pathologies, use_all_labels)` based on `config['labels']`.
- All consumer scripts (`train.py`, `test.py`, `calculate_conformal_pred.py`) import from `data` package.

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

Located at `mimic.data_dir` (config.yaml): `/gpfs0/tamyr/projects/data/MIMIC-CXR/`

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
PYTHONPATH=. python data/mimic_split.py
```

All splits use Mondrian-style stratification based on top pathology prevalence.

## Development Workflow

**HPC Environment - Login Node vs. Compute Nodes**

The login node is for data transfer and job submission ONLY. Do NOT run Python scripts, training, or computationally intensive operations on the login node. All work must be submitted to compute nodes via `runai`.

1. Edit and commit code on the login node
2. Push to git repository
3. Submit training jobs to compute nodes using `runai` (never run `python train.py` directly on login node)
4. Monitor job status with runai commands

## Quick Start

**For Running on Compute Nodes (via runai):**

1. Train the model:
```bash
python train.py --config config_hier_unified.yaml
```

2. Calibrate conformal thresholds:
```bash
python calculate_conformal_pred.py --config config_hier_unified.yaml
```

3. Run test evaluation:
```bash
python test.py --config config_hier_unified.yaml
```

Outputs go to `conformal_calibration/mimic/hier_unified/` (thresholds, confusion matrices) and `results/outputs/mimic/` (test CSV).

## Hierarchy Label Correction

CheXpert labels suffer from a labeling artifact: when a child finding is positive but the parent was "not mentioned" in the report, the parent gets NaN (mapped to 0). This creates contradictory training signals (e.g., Edema=1 but Lung Opacity=0). Analysis of the MIMIC training set shows 52-96% of child-positive samples have missing parent labels.

**Hierarchy correction** propagates child-positive → parent-positive. When `hierarchy_correction.enabled: true`, the DataModule corrects labels in the DataFrame after loading (all splits including test).

**Disease hierarchy** (defined in `data/hierarchy.py`):
- Enlarged Cardiomediastinum ← Cardiomegaly
- Lung Opacity ← Lung Lesion, Edema, Consolidation, Atelectasis
- Lung Opacity + Consolidation ← Pneumonia

**Implementation:**
- `data/hierarchy.py`: `apply_hierarchy_correction_to_df()` corrects a DataFrame in-place. Only definite positives (1.0) trigger propagation — uncertain (-1) and NaN do not.
- `data/mimic.py`: DataModule reads `hierarchy_correction.enabled` from config and applies correction in `setup()`.
- `train.py`: Adds `-hier-` suffix to checkpoint filenames when hierarchy correction is enabled.

## Key Implementation Notes

- **data/__init__.py**: Exports `ALL_CHEXPERT_LABELS`, `parse_labels_config()`, and `get_data_module()` factory.
- **data/mimic.py**: `MIMICDataset` + `MIMICDataModule`. Builds image paths from `(subject_id, study_id, dicom_id)` columns. All images come from a single `files_dir`.
- **data/mimic_split.py**: Joins metadata + labels, filters to configured views (AP/PA), scans disk for existing files, then splits. Re-runnable as download progresses.
- **data/split_utils.py**: Shared Mondrian stratification logic and split statistics printing.
- **data/hierarchy.py**: Disease hierarchy (`CHEXPERT_HIERARCHY`), `apply_hierarchy_correction_to_df()`.
- **config.yaml**: Base config. `mimic:` section has MIMIC paths and `views` filter. `labels: "all"` enables 13-label mode (excluding No Finding) with No Finding preprocessing.
- **config_hier_unified.yaml**: Hierarchy-enabled config for the unified model. Points to the hier checkpoint, enables hierarchy correction, and outputs to `conformal_calibration/mimic/hier_unified/`.
- **config.yaml**: Optimizer hyperparameters under `optimizer:`. Checkpoint loading under `evaluation:`. Output paths under `output:`.
- **losses/**: BCE loss implementation. `get_loss_function()` factory with auto-computed pos_weights.
- **densenet_model.py**: `CXRDenseNet` model class (DenseNet wrapper with torchxrayvision pretrained weights).
- **train.py**: Trains on train_split, validates on valid_split. Uses `get_data_module()` to load the dataset. Supports `reinit_classifier` (re-initialize the classifier head when loading a pretrained checkpoint) and WandB logging via `WandbLogger`.
- **test.py**: Runs inference on test_split, saves predictions CSV to `results/outputs/`, prints per-label AUC (filtering uncertain labels).
- **calculate_conformal_pred.py**: Conformal prediction calibration using conformal_split.
- **results/scripts/**: empty as of the 2026-10-28 clean restart — every plotting script was
  deleted along with `results/images/`. New figures are written fresh; the old ones are in
  `~/Desktop/xray_pre_reset_backup_20261028_104145.tar.gz` if a reference is needed.

## Trained Model

Single unified checkpoint: `checkpoints/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809.ckpt`
- Trained on full MIMIC-CXR dataset with hierarchy correction enabled
- Test inference CSV: `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809.csv`
- Conformal calibration: `conformal_calibration/mimic/hier_unified/`

All three splits are exported, flat in `results/outputs/mimic/`, named off the checkpoint
stem (`{stem}.csv` for test, `{stem}_conformal.csv`, `{stem}_validation.csv`). Each has
`subject_id, study_id, dicom_id` + `{label}_prob` / `{label}_true_label` for 13 labels:

| split | rows | subjects |
|---|---|---|
| train | — | — |
| validation | 18,570 | 5,898 |
| conformal | 18,754 | 5,926 |
| test | 24,394 | 6,395 |

**Splits are subject-disjoint.** `data/mimic_split.py` now stratifies on subject-level
label maxima (`groupby('subject_id')[pathologies].max()`) and splits subject IDs, then
maps rows back — verified zero subject and zero image overlap across all three pairs.
This fixed a subject-level leakage problem in the earlier partial-data splits (32.8% of
test subjects also appeared in calibration), which had invalidated the coverage
guarantees of every experiment built on them. Every new experiment re-verifies the overlap
as its Gate 0.

- **export_split_scores.py**: Exports validation + conformal predictions for the unified
  model. `python export_split_scores.py --config config_hier_unified.yaml`.

## Per-Label Pipeline (retired, kept on disk only)

The per-label one-vs-rest pathway is **retired and will not be retrained** (Alon,
2026-10-21). All conformal work runs on unified-model probabilities. Standing caveat: the
unified model's shared backbone inflates cross-label probability correlation (Enlarged
Cardiomediastinum/Cardiomegaly 0.936 → 0.997, Consolidation/Pneumonia 0.832 → 0.971), so
any dependence claim computed on it partly measures the backbone — state this, don't
retrain around it.

Files still present but unused: `utils.py`, `export_scores.py`, `train_per_label.py`,
`calibrate_per_label.py`, `test_per_label.py`, `configs/per_label/`,
`configs/per_label_hier/`. Their checkpoints were partial-data and leaky; don't resurrect
them.

- **export_scores.py**: Exports conformal + validation probabilities with subject IDs.
  `__main__` runs `export_unified_hier()` then `export_per_label_hier()`; labels without
  a checkpoint are skipped with a `[SKIP]` line. `export_per_label()` (non-hier) exists
  but is not called from `__main__`.

## MIMIC-CXR Data Utilities

### Download

- **`download_mimic_cxr.py`**: Downloads MIMIC-CXR-JPG from PhysioNet via wget. Supports `--subset p10,p11` for partial downloads, `--views AP,PA` filtering, skip-existing via `.downloaded_manifest`, `--verify` mode, and parallel downloads.

### Prep Scripts (`mimic-cxr_scripts/`)

- **`check_mimic_coverage.py`**: Verifies how many study records have matching images on disk; reports view distribution.
- **`create_filtered_mimic_csv.py`**: Scans disk for existing files, applies "No Finding correction" (zeros out other pathologies when No Finding=1.0), saves `mimic_cxr_filtered.csv`.
- **`filter_to_pa_views.py`**: Filters `mimic_cxr_filtered.csv` to frontal views (PA/AP) using metadata CSV.
- **`reorganize_mimic_cxr.py`**: Moves flat-downloaded JPGs into the proper `files/p{XX}/p{subject_id}/s{study_id}/` hierarchy.

## Results & Visualization

`results/outputs/mimic/` holds the three frozen probability CSVs (validation / conformal /
test) — these are the inputs to all research work and are not regenerated.
`results/images/` and `results/scripts/` were emptied in the 2026-10-28 restart.

## Research Direction

The focus (as of 2026-11-10) is the **efficiency of clustering for conformal prediction**:
how much clustering labels and calibrating each cluster jointly shrinks prediction sets at
the same joint coverage, on the unified model's cached probabilities. `RESEARCH_DIRECTION.md`
holds the committed design (clustering procedures, score, methods, metric set, caveats).

**Clean restart, 2026-10-28.** Folders 032–045, the Exp032–042 summaries, and all
plots/plotting scripts were deliberately deleted. Current experiments are
`experiments/046` onward. Backup tarball:
`~/Desktop/xray_pre_reset_backup_20261028_104145.tar.gz`.
`MAX_RANK_EXPERIMENTS_SUMMARY.md` (Exp001–031) is the older archive.

## Agents

- **Tom Hagan** (`.claude/agents/tom-hagan.md`): conformal-prediction and mathematical-
  statistics consultant. Does NOT write code. Rules on validity of coverage guarantees,
  score design, joint vs. marginal coverage, clustering choices, and how to state a claim
  honestly. Reads `RESEARCH_DIRECTION.md` on demand. Invoke for design decisions and for
  reviewing Jules's results or escalations.
- **Jules** (`.claude/agents/jules.md`): experiment runner. Executes exactly Tom's spec on
  the frozen CSVs, pre-registers before running, reports honest numbers. Designs nothing —
  no choosing clusters, metrics, methods, α or λ. Never modifies the pipeline. Escalates via
  `ESCALATION.md` rather than iterating variants.

**Workflow:** Tom designs an experiment (grounded in `RESEARCH_DIRECTION.md`) → Jules
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
