# Exp048 Escalation — Missing train probabilities

**Date:** 2026-10-29. Gates: 0 (data availability).

## Issue

Tom's Exp048 spec requests using `_train.csv` as the proper-training split for fitting Sigma, Hp, and Cp in LP-SCP. However, the available train CSV contains **only true labels**, not predicted probabilities:

```
train CSV columns: subject_id, study_id, dicom_id, {label}_true_label (13 labels)
validation CSV columns: subject_id, study_id, dicom_id, {label}_prob, {label}_true_label (13 labels each)
conformal CSV columns: (same as validation — has both prob and true_label)
test CSV columns: (same as validation — has both prob and true_label)
```

LP-SCP's `fit_structure()` function requires probabilities on the proper-training split to compute:
1. **Sigma** (covariance of error vectors `r = |y - o(x)|`): requires both true labels and predicted probabilities
2. **Hp** (observed label-sets vocabulary): can use true labels only
3. **Cp** (cardinality distribution): can use true labels only

Without the predicted probabilities on the train split, Sigma cannot be computed.

## Documentation note

CLAUDE.md § "Trained Model" states: "All three splits are exported, flat in `results/outputs/mimic/`, named off the checkpoint stem (`{stem}.csv` for test, `{stem}_conformal.csv`, `{stem}_validation.csv`). Each has `subject_id, study_id, dicom_id` + `{label}_prob` / `{label}_true_label`..."

The documentation explicitly lists only three frozen probability CSVs (validation, conformal, test). The train split's CSV exists but contains only ground-truth labels, not predicted probabilities. The commit message for the train CSV (57a9f84) says "add train GT CSV" — GT = ground truth only.

## Constraint

Per CLAUDE.md hard rules: "Never modify the pipeline. `train*.py`, `test*.py`, `calibrate_per_label.py`, `calculate_conformal_pred.py`, `densenet_model.py`, `utils.py`, `export*.py`, `data/`, `losses/`, `configs/` are read-only."

The `export_split_scores.py` script only exports validation and conformal splits and cannot be modified to add train support.

## Feasible Options

1. **Use validation as proper-training (primary):** Matches Exp047 exactly; this will run immediately and produce valid results.
2. **Manual export of train probabilities:** If Alon runs `export_split_scores.py` or a new export script on the frozen checkpoint to generate train probabilities, place the CSV at `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_train.csv` with both `_prob` and `_true_label` columns, then re-run Exp048.
3. **Defer train-based arm:** Mark train-based LP-SCP as a future experiment (Exp049?) pending availability of train probabilities.

## Question for Tom

What should Exp048 do?

A. Proceed now with validation-based proper-training as the primary arm (Exp047 repeat), skip the train-based A/B variant, and report validation results?

B. Wait for Alon to export train probabilities, then run both arms?

C. Something else?

**Impact:** Without Tom's direction, Exp048 is blocked on data availability.
