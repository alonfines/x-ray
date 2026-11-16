# Exp048 — LP-SCP validation-based proper-training (train-based arm blocked)

**Pre-registered (revised):** 2026-10-29. Tom Hagan spec, post-escalation ruling. Original spec called for train-based proper-training; `_train.csv` has no `{label}_prob` columns, blocking the primary arm. This run: validation-based proper-training (exact Exp047 re-execution) + extended Gate 0 (train/conformal/test disjointness) for provenance. Zero of two A/B arms available. Clusters/α/arms/metrics unchanged — see `experiments/047/REPORT.md` for full design rationale.

## Locked (unchanged from Exp046/047)

- **Four clusters:** `configs/clusters/exp046.yaml` → `cardio_pair` (m=2), `lung_core` (m=3), `pleural_free` (m=3), `lung_wide` (m=5)
- **α = 0.1**
- **Three arms:** base (µ=ν=0), penalised (µ=ν=1, no hard constraint), admissible (µ=ν=1 + hierarchy hard constraint). µ=ν=1 fixed a priori.
- **Both guard variants computed:** no-guard is primary (Tom's 2026-10-28 ruling), with-guard as footnote.
- **Metrics:** joint coverage, mean joint set size, % of state space, fully-decisive rate, decisive accuracy, hierarchy-violation rate of decisive output, stratified coverage check (states observed in proper-training vs observed in test).
- **Calibration split:** `_conformal.csv` (18,754 rows, 5,926 subjects)
- **Evaluation split:** `test.csv` (24,394 rows, 6,395 subjects)
- **No µ/ν grid:** `--grid` not required this pass.

## Actual scope (Tom's 2026-10-29 ruling)

**Only run:** Validation-based proper-training (exact Exp047 repeat), with extended Gate 0.

- **Proper-training split:** `_validation.csv` (18,570 rows, 5,898 subjects)
- **Rationale:** Train-based arm blocked by missing `{label}_prob` columns in train CSV (ground-truth only). Rerun validation-based setup for Gate-0/provenance cleanliness.
- **Implementation:** Call `lp_scp.py --proper-training-split validation` (default in code).
- **Result:** Should match Exp047 to the digit (same data, same deterministic method, no seed variation). Exact match required; any delta is a bug.

## Gate 0, extended

**Existing check:** conformal, test, validation disjointness ✓ (per Exp047)

**New check (no probs needed, runs free):**
- Train vs conformal: subject_id and dicom_id disjointness
- Train vs test: subject_id and dicom_id disjointness

Report in REPORT.md before results tables. This is the only new content — validates the train split for future use once probabilities are exported.

## Out of scope

- Do not touch `max_rank.py`, `bonferrony.py`, or Exp046
- Do not touch `conformal_common.py`'s locked score/exclusion/metrics
- Do not attempt to fix §3a calibration-population caveat itself — that is separate, unapproved work

## Reproduce

```bash
cd /Users/alonfines/Desktop/x-ray

# Run validation-based proper-training (only available arm)
python lp_scp.py --clusters configs/clusters/exp046.yaml --experiment experiments/048
```

Outputs: `experiments/048/REPORT.md`, `experiments/048/lp_scp_results.csv`, `experiments/048/thresholds/`

## Input files

- Frozen CSVs: `/Users/alonfines/Desktop/x-ray/results/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_{train,validation,conformal,test}.csv`
- Clusters: `/Users/alonfines/Desktop/x-ray/configs/clusters/exp046.yaml`
- Hierarchy: `/Users/alonfines/Desktop/x-ray/data/hierarchy.py`

## Git commit

TBD (will add after successful run)
