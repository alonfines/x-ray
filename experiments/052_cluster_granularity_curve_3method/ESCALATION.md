# Exp052 — ESCALATION (Rung E joint coverage below 0.90 at K=1)

**Date:** 2026-11-01. **Folder:** `experiments/052_cluster_granularity_curve_3method/`. **Git HEAD:** `6a26ac8`. **Seed:** 0. **α:** 0.10.

## What was run
Pre-registered 3-method granularity curve (max-rank cited from Exp051; Bonferroni and Šidák, and LP-SCP `penalised` μ=ν=1, run) on the 5-rung ladder. All pre-registered triggers T-A through T-E **did not fire**. Gate 0, Gate 0.5, Rung A checks, and N checks all pass. Full numbers are in `README.md` (post-run record) and `metrics.json`.

## What came out, and why it is a problem
At **Rung E (K=1, one joint cluster of all 13 labels)**, the joint guarantee should hold under exchangeability, and the target is 0.90:

| Method | test joint coverage (N=18,877) |
|---|---|
| Max-rank (cited, Exp051) | **0.8934** |
| LP-SCP, guard off / on | **0.8943** / 0.8943 |
| Bonferroni | 0.9154 |
| Šidák | 0.9111 |

Max-rank and LP-SCP are both below 0.90. With SE ≈ 0.0022 that is about 3 SE below target for max-rank. The two Bonferroni-family methods are above target. Max-rank at K=1 is a single rank-quantile over the 13-label row maximum, so it is the case with the most direct finite-sample claim. Its shortfall is therefore the most important thing to rule on.

**Post-hoc diagnostic (not pre-registered):** the same Rung E thresholds, evaluated on the calibration split in-sample, give max-rank 0.9002 (as designed), Bonferroni 0.9204, Šidák 0.9168. Each drops by 0.5–0.7 pp on the 13-clean test population. The shortfall is therefore absent in-sample and is a calibration-to-test drop. The likely explanations are (a) a population shift between the conformal split and the 13-clean test subset (RESEARCH_DIRECTION §3a scope (b)), or (b) subject-level dependence, since rows come ~3 per subject. **I have not tested either explanation and have not established the cause.** I did not iterate on alpha, partitions, or subsets to make the number look better.

Rung A checks are exact (Bonferroni and Šidák Rung A thresholds equal 051's max-rank thresholds, max |Δ| = 0.0). So this is not a code-path bug in the Bonferroni family. The LP-SCP chunked computation was checked against `lp_scp.py` on the full split for m≤6 and on subsets for m=13. It is not a bug in the score path either.

## Questions for Tom
1. **K=1 validity.** Is the 13-clean test subset (N=18,877, dropping the 5,517 rows with any −1) a valid evaluation population for the K=1 joint claim? Is the 0.5–0.7 pp calibration-to-test drop a §3a shift, subject dependence, or something else? Should the K=1 result be reported as a finding or held pending a population check?
2. **Scope of the Rung E shortfall.** Max-rank's 0.8934 was already in Exp051 and was not flagged there. Should 051's Rung E be re-read in light of this?
3. **Non-empty guard (LP-SCP).** The spec does not pin the eq. 20 guard. Both variants are reported. Which, if either, is headline? Guard-on and guard-off differ only at B–D (21 empty rows at B–D with guard off).
4. **Pre-registration of T-A runtime limit.** I set 2 h wall-clock as the infeasibility threshold. The run took 13 s. Confirm or change.
5. **Chunked LP-SCP computation.** The unchunked `lp_scp.state_scores` needs about 16 GB at m=13 on the calibration split. I computed the identical formula in 512-row chunks, verified equality, and did not edit `lp_scp.py`. Confirm this is acceptable.

## What I have not done
- No changes to any pipeline file.
- No re-running or editing of Exp051.
- No choice of partition, α, μ, ν, arm, or guard. The guard question is open, not decided.
- No interpretation of the Šidák-vs-Bonferroni gap as evidence for the dependence claim. That is Tom's call.

## Standing caveat
One shared-backbone classifier inflates cross-label correlation (EC/CM 0.936→0.997). The direction is conservative, understating the independent baseline's failure.
