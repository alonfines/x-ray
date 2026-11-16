# PRE_REGISTRATION: Direct Joint Coverage (Exp050)

**Date:** 2026-11-01  
**Method:** Direct enumeration-free joint coverage of 13 independent per-label conformal sets at α=0.1.

## Specification

1. **Input:**
   - Test split CSV: `results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_test.csv`
   - Per-label calibrated thresholds (α=0.1, correction=none): 13 JSON files in `experiments/049_granularity_sweep/thresholds/per_label_<Label>_a0.1000_none.json`

2. **Score:** True-class nonconformity (locked in conformal_common.py):
   - `s = 1 - p` if `y == 1`
   - `s = p` if `y == 0`

3. **Per-label prediction sets** (3-way, per row, for each label independently):
   - Load the label's threshold `t` from its cached JSON
   - For each row's predicted probability `p` and true label `y`:
     - If `s ≤ t`, include both classes {0, 1} (uncertain)
     - If `s > t`, keep only the predicted class (singleton set based on argmax(p, 1-p))
   - Equivalently: set contains class `c` iff `class_score(p)[c] ≤ t`, where `class_score(p) = [p, 1-p]`

4. **Uncertain ground truth handling:**
   - Per conformal_common.extract_group() (locked in conformal_common.py):
     - Drop entire rows where ANY of the 13 labels has uncertain (-1.0) or missing (NaN) ground truth
     - Operate only on remaining rows (n_eval defined below)
   - No per-label filtering; a row is all-or-nothing

5. **Direct joint coverage (enumeration-free):**
   - For each row in the filtered test set:
     - For each of the 13 labels:
       - Check: does the per-label prediction set contain the true label?
       - Record boolean indicator: `label_survives[row, label]`
     - `row_survives = AND over all 13 labels of label_survives[row, :]`
   - `direct_joint_coverage = fraction of rows where row_survives`

6. **Metrics reported:**
   - `n_eval`: number of rows after filtering uncertain ground truth
   - `n_dropped_uncertain`: number of rows dropped (any label with -1.0 or NaN)
   - `direct_joint_coverage`: the main metric
   - Per-label coverage for sanity check: fraction of rows where each label survives (reported separately, should roughly match Exp049 level 0 per_label[none] coverage)

## Reproduction command

```bash
python run.py
```

## Implementation notes

- Load thresholds from `experiments/049_granularity_sweep/thresholds/` directly (no re-calibration)
- Use the same score formula as conformal_common.py (unlocked, must match)
- Filter rows the same way as conformal_common.extract_group() (locked, all 13 labels must be definite)
- Compute per-row AND, not enumeration over 2^13=8192 states
- Cross-check: per-label[none] coverage from this direct computation should match Exp049 results_raw.csv level=0, method=per_label, variant=none (within floating-point rounding)
