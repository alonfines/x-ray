# Exp053 — ESCALATION 3 (identity gate SPEC §8 fails; full run NOT executed)

**Date:** 2026-11-10, ~13:00 IST. **Git HEAD:** `6a26ac848b016e82c22d1bd42f61280740d3d26d` (dirty tree). **Folder:** `experiments/053_equal_coverage_clusters/`.

## Trigger
SPEC §8: at K=13 and K=1, Bonferroni, Šidák and max-rank must produce the same sets, at most 0.5% of rows differing. §9: any identity gate failure → escalate. The brief: stop on an identity-gate failure. Stopped. The 100-resplit run was not started.

## What was run
- Gate 0 passed (zero subject/dicom overlap on all pairs; 18,754 / 24,394 / 18,570 rows; 5,926 / 6,395 / 5,898 subjects; 13-clean test 18,877 / 5,932 subjects; pooled 37,631 rows / 11,858 subjects).
- Gate 0.5 passed: no label below 150 positives (Pleural Other 151, Fracture 304, Lung Lesion 523; none flagged).
- Code verification passed (naive per-row loops = vectorised counts and P-values exactly; LP-SCP own-state scores = `lp_scp.true_state_scores` to 0.0 on the full calibration set for m ≤ 6).
- `identity_check.py` (p-value methods only; LP-SCP not involved) over the frozen draw and all 100 resplits. Output: `identity_gates_all.csv` in this folder.

## Result (identity gate)

| design | Bonferroni vs max-rank, fraction of rows with different joint sets (K=13 and K=1 identical) |
|---|---|
| frozen conformal→test | **0.832%** |
| resplits 0–99: min / median / max | 0.229% / 0.615% / 1.058% |
| resplits above 0.5% | **76 / 100** |

Bonferroni vs Šidák: 0.000% on every design (identical sets). Calibration-side coverage at K=1 is 0.90018 (frozen) and 0.90029 (resplit 0); averaged over the 100 resplits it is 0.90028 (SD 0.015 pp) for max-rank, and the Bonferroni value is the same to the printed precision. The calibration side is not the cause.

## Post-trigger diagnostic (not pre-registered; for Tom's ruling only)
- K=1, resplits 0–2: the max-rank and Bonferroni cutoffs differ by exactly one integer R-level (R*=185 vs 186; 190 vs 191; 187 vs 188), with β̂ on different scales (max-rank 0.0997; Bonferroni 0.1282). The disagreeing rows sit at the single R-level boundary. I have not verified the full mechanism. The most likely cause is the test-side "1 +" count against the calibration-side self-inclusive count in the max-rank formula, interacting with a tie group at the threshold; this is a hypothesis to check, not a finding.
- K=1 max-rank test joint coverage over the 100 resplits: mean 0.90069, **SD 0.574 pp** (inside the 0.25–0.8 pp band; no SD trigger). Frozen draw 0.8934 (same as Exp052).

## Runtime (for the record)
One resplit including the split-calibration arm: **112.5 s**. Projected 100 resplits ≈ 3.2 h, plus the frozen draw (~2 min). Under the 4 h limit; no LP-SCP fallback needed.

## Also noted
- Matched null K=6: only 151 of 5,000 pool partitions match the 3-edge informed count, so 151 (not 200) are kept (the rule for <200 matches; ≥60 so no escalation). Partition count is 1,364, not 1,413. K=3 and K=4 kept 200 each.
- `lp_scp.py` sha256 differs from Exp052's record (flagged in README).

## Question for Tom
1. Ruling needed on SPEC §8. The gate as written is not met: max-rank and Bonferroni/Šidák-as-recalibrated differ on ~0.6% of rows at K=1 and K=13. Options for you to choose (I have not chosen): (a) accept the gate failure as a documented finding and authorise the full run; (b) change the tolerance; (c) change a definition (the max-rank count convention) — this is a definitional change and would need its own pre-registration; (d) other.
2. If (a), confirm whether the gate applies to the frozen draw as well as the resplits (0.83% frozen vs 0.23–1.06% resplits).
3. Confirm the 151-match K=6 matched null is acceptable under E4.

Nothing has been run beyond the gates, verification, identity check and the one-resplit timing. Full-run outputs have not been produced. The timing run output is in `/private/tmp/claude-501/scratch_exp053/` (not part of the record).
