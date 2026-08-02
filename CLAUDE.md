# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a proof-of-concept implementation of a DenseNet-based chest X-ray classification model using the CheXpert dataset. The project trains a deep learning model to classify multiple pathological conditions from chest radiographs.

**Key Goal:** Build and test the training pipeline locally before uploading to production servers.

## Data Structure

The `chexpert/` directory mirrors the production server structure:

```
chexpert/
├── train.csv              # (Original) Source for train/validation/conformal splits
├── valid.csv              # (Original) Becomes test_split.csv (held-out test set)
├── train_split.csv        # Stratified train set (training only)
├── valid_split.csv        # Stratified validation set (validation during training)
├── conformal_split.csv    # Stratified conformal set (held-out for conformal calibration)
├── test_split.csv         # Test set from valid.csv (final evaluation)
├── train/                 # Training images organized by patient ID
│   └── patient*/          # Individual patient folders containing X-ray images
└── valid/                 # Test images
    └── patient*/          # Individual patient folders
```

### CSV Format

All CSVs contain:
- **Path**: Image path reference
- **Demographics**: Sex, Age
- **View Info**: Frontal/Lateral, AP/PA orientation
- **Labels**: 14 binary/ternary pathology conditions:
  - No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices
  - Values: 1.0 (positive), 0.0 (negative), -1.0 (uncertain), empty (unlabeled)

### Dataset Splitting Strategy

Raw `train.csv` and `valid.csv` are processed into four stratified sets:

1. **Train Set**: Model training (70% of filtered train.csv)
2. **Validation Set**: Model validation during training (10% of filtered train.csv)
3. **Conformal Set**: Held-out data for conformal prediction calibration (10% of filtered train.csv)
4. **Test Set**: Final held-out evaluation set (entire valid.csv, unfiltered)

**Label Filtering:**
- Train/Validation/Conformal splits: Uncertain labels (-1.0) are **excluded**. Only positive (1.0) and negative (0.0) labels are retained for clean training data.
- Test split: Uses data from the original dataset splits. Note: the test set **does** contain uncertain (-1.0) labels. These are preserved at load time (`clean_uncertain=False`) and filtered out per-pathology when computing AUC.

**Creating splits:**
```bash
runai submit split_job python3 data_split.py
```

This script:
- Reads raw train.csv and valid.csv
- Filters train.csv to remove samples with uncertain (-1.0) labels across any pathology
- Performs stratified splits on filtered train.csv: 70% train / 10% validation / 10% conformal
- Uses entire valid.csv (unfiltered) as the test set with no modification
- Ensures similar label distributions across train/validation/conformal sets via Mondrian-style stratification
- Prints final sample counts for each split
- Creates four new CSV files used by data.py

All splits are stratified based on total positive pathologies per sample to maintain label distribution balance.

## Development Workflow

**HPC Environment - Login Node vs. Compute Nodes**

⚠️ **CRITICAL:** The login node is for data transfer and job submission ONLY. Do NOT run Python scripts, training, or computationally intensive operations on the login node. All work must be submitted to compute nodes via `runai`.

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
runai submit split_job python3 data_split.py
```

2. Train the model:
```bash
runai submit train_job python3 train.py
```

3. Run conformal prediction (calibration + test evaluation):
```bash
runai submit test_job python3 test.py
```

**Configuration:**
- Update `conformal_split_size` in config.yaml based on your dataset size
- Update `chexpert_dir` path if needed
- The scripts automatically scale to handle the full dataset

## Key Implementation Notes

- **data.py**: `CXRDataModule` loads from four split CSVs (train_split.csv, valid_split.csv, conformal_split.csv, test_split.csv)
- **data.py**: Includes `conformal_dataloader()` for conformal calibration and `test_dataloader()` for final evaluation
- **data.py**: Correctly maps image directories (train/val/conformal use train_img_dir, test uses valid_img_dir)
- **config.yaml**: Contains `conformal_split_size` parameter (number of samples to reserve for conformal prediction)
- **data_split.py**: Combines train.csv and valid.csv, then splits into train/validation/conformal/test. Filters uncertain -1.0 labels from train/validation/conformal sets only. Test set retains all labels (including -1.0 uncertain) from the split data.
- **config.yaml**: `use_all_labels: True` enables 13-label mode (all CheXpert pathologies excluding No Finding) with No Finding preprocessing. When `False`, uses the 5 labels listed in `use_labels`.
- **config.yaml**: `loss.type` selects the training loss: `"bce"` (default, BCEWithLogitsLoss with optional task_weights) or `"auc_margin"` (AUC Margin Loss from Yuan et al. 2021, arXiv:2012.03173). AUC Margin Loss directly optimizes AUC via a min-max surrogate with learnable primal-dual parameters (a, b, α per label). Margin `m` is tunable via `loss.auc_margin.margin` (paper tunes from {0.3, 0.5, 0.7, 1.0}). Imratio (class priors) is computed automatically from the training CSV.
- **densenet_model.py**: Contains `AUCMarginLoss` class implementing equation (8) from the paper, and `get_loss_function()` factory that returns the configured loss.
- **train.py**: Trains on train_split, validates on valid_split. When using AUC Margin Loss, includes loss parameters (a, b, α) in the optimizer, applies gradient ascent on α via hook, and projects α ≥ 0 after each step.
- **test.py**: Runs inference on test_split, saves predictions to CSV, and prints per-label AUC (filtering out uncertain labels). Does not perform calibration.
- **calculate_conformal_pred.py**: Separate workflow for conformal prediction calibration using conformal_split
- **test_analyze.py**: Analyzes results from test.py output without requiring re-runs — use for evaluation and visualization after inference is complete
