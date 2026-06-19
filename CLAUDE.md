# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a proof-of-concept implementation of a DenseNet-based chest X-ray classification model using the CheXpert dataset. The project trains a deep learning model to classify multiple pathological conditions from chest radiographs.

**Key Goal:** Build and test the training pipeline locally before uploading to production servers.

## Data Structure

The `chexpert/` directory mirrors the production server structure:

```
chexpert/
├── train.csv              # (Original) Training set metadata
├── valid.csv              # (Original) Validation set metadata
├── train_split.csv        # Stratified train set (training only)
├── valid_split.csv        # Stratified validation set (validation during training)
├── conformal_split.csv    # Stratified conformal set (held-out for conformal prediction)
├── train/                 # Training images organized by patient ID
│   └── patient*/          # Individual patient folders containing X-ray images
└── valid/                 # Validation/test images
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

Raw `train.csv` and `valid.csv` are split into three stratified sets:

1. **Train Set**: Model training (configurable percentage of non-conformal data, default 80%)
2. **Validation Set**: Model validation during training (remaining non-conformal data, default 20%)
3. **Conformal Set**: Held-out data for conformal prediction (size set by `conformal_split_size` in config.yaml)

**Creating splits:**
```bash
python3 data_split.py
```

This script:
- Reads raw train.csv and valid.csv
- Combines all samples
- Stratified split: extracts conformal set first (using `conformal_split_size` parameter)
- Stratified split: divides remaining data 80% train / 20% validation
- Ensures similar label distributions across all three sets
- Creates three new CSV files used by data.py

The splitting is stratified based on total positive pathologies per sample to maintain label distribution balance.

## Development Workflow

**Local Development → Git → Production Servers**

1. Build and test all scripts locally in this directory
2. Commit to git with proper documentation
3. Push to git repository
4. Pull on production servers and run training pipeline

## Quick Start

1. **Create the stratified splits:**
   ```bash
   python3 data_split.py
   ```
   This generates `train_split.csv`, `valid_split.csv`, and `conformal_split.csv` in the chexpert/ directory.

2. **Run training:**
   ```bash
   python3 train.py
   ```
   Trains model on train_split.csv, validates on valid_split.csv, and saves checkpoints.

3. **Configure for production:**
   - Update `conformal_split_size` in config.yaml based on your full dataset size
   - Update `chexpert_dir` path for production servers
   - The splitting script will automatically scale to handle the full dataset

## Key Implementation Notes

- **data.py**: `CXRDataModule` loads from split CSVs (train_split.csv, valid_split.csv, conformal_split.csv)
- **data.py**: Includes `conformal_dataloader()` method for accessing the held-out conformal set
- **config.yaml**: Contains `conformal_split_size` parameter (number of samples to reserve for conformal prediction)
- **data_split.py**: Handles stratified splitting with support for mixed train/valid sources in each split
