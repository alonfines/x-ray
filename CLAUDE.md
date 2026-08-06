# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DenseNet-based chest X-ray classification model supporting both **CheXpert** and **MIMIC-CXR-JPG** datasets. The project trains a deep learning model to classify multiple pathological conditions from chest radiographs.

**Key Goal:** Build and test the training pipeline locally before uploading to production servers.

## Data Package (`data/`)

All dataset logic lives in the `data/` package:

```
data/
├── __init__.py          # Shared constants (ALL_CHEXPERT_LABELS), parse_labels_config(), get_data_module() factory
├── split_utils.py       # Shared stratification & statistics helpers
├── chexpert.py          # CheXpertDataset + CheXpertDataModule
├── chexpert_split.py    # CheXpert splitting script
├── mimic.py             # MIMICDataset + MIMICDataModule
└── mimic_split.py       # MIMIC splitting script (scans disk for existing files, filters by view)
```

- **`get_data_module(config_path)`** is the main entry point. It reads `config['dataset']` (`"chexpert"` or `"mimic"`) and returns the appropriate DataModule.
- **`parse_labels_config(config)`** returns `(pathologies, use_all_labels)` based on `config['labels']`.
- All consumer scripts (`train.py`, `test.py`, `calculate_conformal_pred.py`) import from `data` package.

## Data Structure

### CSV Organization

Split CSVs are organized by dataset under `csv_files/`:

```
csv_files/
├── chexpert/
│   ├── train_split.csv
│   ├── valid_split.csv
│   ├── conformal_split.csv
│   ├── test_split.csv
│   ├── results.csv
│   └── conformal_predictions.csv
└── mimic/
    ├── train_split.csv
    ├── valid_split.csv
    ├── conformal_split.csv
    └── test_split.csv
```

### CheXpert Raw Data

Located at `data.chexpert_dir` (config.yaml). Contains `train/` and `valid/` image directories plus raw `train.csv` and `valid.csv`.

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

Both datasets use the same 14 CheXpert pathology labels:
- No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices
- Values: 1.0 (positive), 0.0 (negative), -1.0 (uncertain), empty (unlabeled)

**CheXpert CSVs** have a `Path` column for image paths.
**MIMIC CSVs** have `dicom_id`, `subject_id`, `study_id` columns; image paths are constructed from these.

### Dataset Splitting Strategy

Both datasets are split into four stratified sets with the same ratios:

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
# CheXpert
PYTHONPATH=. python data/chexpert_split.py

# MIMIC-CXR
PYTHONPATH=. python data/mimic_split.py
```

All splits use Mondrian-style stratification based on top pathology prevalence.

## Development Workflow

**HPC Environment - Login Node vs. Compute Nodes**

The login node is for data transfer and job submission ONLY. Do NOT run Python scripts, training, or computationally intensive operations on the login node. All work must be submitted to compute nodes via `runai`.

1. Edit and commit code on the login node
2. Push to git repository
3. Submit training jobs to compute nodes using `runai` (never run `python3 train.py` directly on login node)
4. Monitor job status with runai commands

## Quick Start

**For Code Development (on login node):**
- Edit and commit scripts to git
- Run only minimal validation scripts if absolutely necessary
- Use `runai` to submit all actual training/processing jobs

**For Running on Compute Nodes (via runai):**

1. Create the stratified splits:
```bash
# CheXpert
runai submit split_job python3 -c "import sys; sys.path.insert(0,'.'); exec(open('data/chexpert_split.py').read())"

# MIMIC-CXR
runai submit split_job python3 -c "import sys; sys.path.insert(0,'.'); exec(open('data/mimic_split.py').read())"
```

2. Train the model:
```bash
runai submit train_job python3 train.py
```

3. Run test evaluation:
```bash
runai submit test_job python3 test.py
```

**Configuration:**
- Set `dataset: "chexpert"` or `dataset: "mimic"` in config.yaml to switch datasets
- All scripts automatically use the correct DataModule via `get_data_module()`
- Update paths in the `data:` or `mimic:` sections of config.yaml as needed

## Hierarchical Conditional Training (Pham et al. 2020)

Implements the method from "Interpreting chest X-rays via CNNs that exploit hierarchical disease dependencies and uncertainty labels."

**Components:**
- `data/hierarchy.py`: Disease hierarchy definition (`CHEXPERT_HIERARCHY`), conditional mask, and `apply_hierarchical_inference()`.
- `data/conditional_dataset.py`: `ConditionalSubset` wrapper that filters to parent-positive samples for Phase 1.
- `train_hierarchical.py`: Two-phase training orchestrator (Phase 1: conditional subset, Phase 2: frozen backbone + full data).

**Uncertain label strategies** (set via `training.uncertain_strategy` in config.yaml):
- `u_zeros` (default, backward-compatible): -1 → 0.0
- `u_ones`: -1 → 1.0
- `u_zeros_lsr`: -1 → random U(0.0, 0.3)
- `u_ones_lsr`: -1 → random U(0.55, 0.85)

**Running hierarchical training:**
```bash
# Set in config.yaml:
#   conditional_training.enabled: true
#   training.uncertain_strategy: u_ones_lsr
#   hierarchical_inference.enabled: true
python3 train_hierarchical.py
```

**Config toggles (all default to false/baseline):**
- `conditional_training.enabled`: Enable two-phase CT training
- `conditional_training.phase1_epochs` / `phase2_epochs`: Epochs per phase
- `conditional_training.phase1_lr` / `phase2_lr`: Learning rates per phase
- `hierarchical_inference.enabled`: Multiply conditional probs along hierarchy at inference

## Key Implementation Notes

- **data/__init__.py**: Exports `ALL_CHEXPERT_LABELS`, `parse_labels_config()`, and `get_data_module()` factory. The factory reads `config['dataset']` to return the correct DataModule.
- **data/chexpert.py**: `CheXpertDataset` + `CheXpertDataModule`. Loads from four split CSVs. Maps image paths from the `Path` column, routing to `train/` or `valid/` image directories.
- **data/mimic.py**: `MIMICDataset` + `MIMICDataModule`. Builds image paths from `(subject_id, study_id, dicom_id)` columns. All images come from a single `files_dir`.
- **data/chexpert_split.py**: Splits CheXpert raw CSVs into train/validation/conformal/test. Filters uncertain labels from non-test sets.
- **data/mimic_split.py**: Joins metadata + labels, filters to configured views (AP/PA), scans disk for existing files, then splits. Re-runnable as download progresses.
- **data/split_utils.py**: Shared Mondrian stratification logic and split statistics printing.
- **config.yaml**: `dataset` field selects active dataset. `data:` section has CheXpert paths. `mimic:` section has MIMIC paths and `views` filter. `labels: "all"` enables 13-label mode (excluding No Finding) with No Finding preprocessing.
- **config.yaml**: `loss.type` selects the training loss: `"bce"` or `"auc_margin"` (AUC Margin Loss from Yuan et al. 2021).
- **config.yaml**: Optimizer hyperparameters under `optimizer:`. Checkpoint loading under `evaluation:`. Output paths under `output:`.
- **losses/**: Loss implementations. `get_loss_function()` factory, `AUCMarginLoss`, BCE with pos_weights.
- **densenet_model.py**: `CXRDenseNet` model class (DenseNet wrapper with torchxrayvision pretrained weights). Has `freeze_backbone()` / `unfreeze_all()` for conditional training Phase 2.
- **train.py**: Trains on train_split, validates on valid_split. Uses `get_data_module()` to load the correct dataset.
- **train_hierarchical.py**: Two-phase conditional training orchestrator (Pham et al. 2020). Phase 1: conditional subset, Phase 2: frozen backbone.
- **test.py**: Runs inference on test_split, saves predictions to CSV, prints per-label AUC (filtering uncertain labels). Supports hierarchical inference toggle.
- **data/hierarchy.py**: Disease hierarchy (`CHEXPERT_HIERARCHY`), `get_conditional_mask()`, `apply_hierarchical_inference()`.
- **data/conditional_dataset.py**: `ConditionalSubset` dataset wrapper for Phase 1 filtering.
- **calculate_conformal_pred.py**: Conformal prediction calibration using conformal_split.
- **test_analyze.py**: Analyzes results from test.py output without requiring re-runs.
