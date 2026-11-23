# Exp059: full-13 joint conformal, Bonferroni vs max-rank vs MLP (pre-registration)

Status: PRE-REGISTERED before any run. Spec from Tom Hagan (signed off, changes folded in); run requested by Alon.

## Question
Joint coverage and joint set size on the full 13-label vector, alpha = 0.10, no guard, no hierarchy
enforcement. Frozen conformal/test draw (resplit -1) plus 100 subject-level resplits (r = 0..99).

**Population: all-13-clean rows only.** Pool = conformal + test = 37,631 rows (18,754 + 18,877), about 77%
of test. Every result is on this population; the ~23% of test rows with an uncertain label are not represented.

## Arms
| id | partition | score | rules | seeds |
|---|---|---|---|---|
| A Bonferroni | `partition.singletons()` | `"per_label"` | `["bonferroni"]` | n/a |
| B max-rank | `partition.singletons()` | `"per_label"` | `["max_rank"]` | n/a |
| C MLP full-13 (primary learned) | `[list(LABELS)]` (one cluster, 8,192 states) | `MLP()` | `["none", "max_rank"]` (identity check) | 0,1,2,3,4 |
| D MLP singletons (secondary/diagnostic) | `partition.singletons()` | `MLP()` | `["max_rank"]` | 0,1,2,3,4 |

A and B run in one call with `rules=["bonferroni","max_rank"]`. The MLP `seed` is passed as `run(..., seed=s)`
(run() forwards its `seed` to `scorer.fit`). Seeds run strictly one after another, with `gc.collect()` between.
Headline = seed 0 of C (pre-registered). Seeds 1-4 are reported as spread only (range of resplit-mean coverage,
range of resplit-mean log2 size / 13). The best seed is never singled out.

## Metrics (per arm, per seed)
- Validity gate: joint coverage (resplit mean, 2.5/97.5 percentile, frozen value).
- Headline pair: median set size, mean of log2(size)/13 (`log2_size_over_m`).
- Secondary: mean set size (reproduction anchor only), empty_rate, share_impossible, violation_rate_decisive, decisive_rate.
- Summaries via `evaluate.summarize(df, by=("arm","seed","rule"), metric_cols=[joint_coverage, median_size, log2_size_over_m, mean_size, empty_rate, share_impossible, violation_rate_decisive, decisive_rate])`.
- Paired CIs via `evaluate.paired` on resplits >= 0, metrics `log2_size_over_m` and `median_size`:
  C(seed 0) - B, B - A, C(seed 0) - D(seed 0).
- Every CI is conditional on the fixed pool and the fixed trained scorers, so it understates total uncertainty.

## Diagnostics
- Share of test rows (frozen draw, and resplit mean) whose true 13-label state never appears among the MLP training
  rows (validation minus `inner_holdout(validation.subject_id)`), via `data.state_index`.
- Arm C per seed: `MLP().info` (holdout_ce, best_epoch, stop_epoch, hit_cap) of the last fit. For arm D, `info` reflects
  only the last of 13 fits; this is stated, not extended.
- Note: `hit_cap` in the existing code is `bad < PATIENCE` evaluated at loop exit; it is reported as the code defines it.

## Gates and escalation triggers (in order; stop at the first failure, write ESCALATION.md)
1. Gate 0 passes (`conformal.data.load` runs it) and pool count = 37,631.
2. Reproduction, exact: arm A mean set size resplit mean = 2855.40, arm B = 850.00 (100 resplits, rounded to 2 dp).
   Mismatch -> stop before reading any MLP number. (Stage ordering: A/B run and are checked before C/D are read.)
3. Identity: in arm C, `none` and `max_rank` rows identical on every draw (frozen + 100 resplits), every seed.
   Mismatch -> stop.
4. Coverage (resplit mean): B, every C seed, every D seed in [0.895, 0.905]; A >= 0.895.
   Any value below 0.895 -> stop (possible bug). C above 0.905 -> discreteness plateau: report, do not tune,
   randomise or re-seed, escalate.
5. empty_rate > 0.5% in any arm -> escalate before computing the log-size comparison. No handling rule is chosen here.
   Clarification fixed before running: a row counts as failing if its resplit-mean empty_rate OR its frozen-draw empty_rate exceeds 0.005.
   The largest single-draw empty_rate is reported as information only, not used as a trigger.
6. hit_cap True for C seed 0, OR the C seed range of resplit-mean log2 size / 13 is wider than |mean paired difference C(s0) - B|
   -> escalate (seed noise, not a finding).
7. Report exactly these arms. No extra rules, alpha, seeds, guards or partitions.

## Asserts on effect sizes
Gates above are the only assertions. Paired CIs are reported with their `excludes_zero` flag; no bare p-values.

## Caveats (stated in REPORT.md with every result)
- All-13-clean population (~77% of test).
- Arms C/D train their MLP on ~14k validation rows (validation minus inner holdout); A/B use no training data.
- Splits are subject-disjoint: coverage is exchangeable at the subject level; row-level coverage is approximate.
- Not in this experiment: the DP m_max=6 partition with MLP (next step, not run).

## Inputs (frozen, read-only)
- `classifier/outputs/mimic/densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_validation.csv`
  sha256 e2c7dfd3087154406d5fd79ae0284dad883926092538d80e854b13b2ed3b4119
- `..._conformal.csv` sha256 cce3677863ab2a24a673339d7ab71ce1ef8a27c0e88ead9c7f9d4625bbaa59db
- `..._test.csv` sha256 ab42e9a895e0b1c31f758bd420d935607ea216a7968472e089a597309ff5d3f5
- Code: `conformal/` package (untracked in git at HEAD; file hashes in `code_hashes.txt` in this folder).

## Reuse (existing modules only; no new scorer, quantile, region, metric or loader code)
`conformal.data` (`load`, `pool`, `draws`, `state_index`, `LABELS`), `conformal.partition` (`singletons`),
`conformal.run` (`run`), `conformal.evaluate` (`summarize`, `paired`), `conformal.scores.mlp` (`MLP`, `inner_holdout`).

## Reproduce
- Git commit: 98b8269 (HEAD at pre-registration). `conformal/` and `classifier/` changes are uncommitted in the working tree; hashes recorded.
- Seeds: resplits `default_rng(r)`, r = 0..99; frozen draw r = -1; MLP seeds 0..4 (init and batch order), inner holdout `default_rng(0)` fixed in code.
- Command (repo root): `.venv/bin/python experiments/059_full13_bonf_maxrank_mlp/run.py 2>&1 | tee experiments/059_full13_bonf_maxrank_mlp/run.log`

## Amendment 2026-11-17 (after ESCALATION.md on Gate 2; ruling by Tom Hagan)
The first run stopped at Gate 2 (see ESCALATION.md, kept unchanged). Tom's ruling: confirm, with amendments. The anchor mislabel was in the brief, not the code.
- **Gate 2 (amended), 100 resplits, resplit means:**
  - Max-rank: `mean_size` = 2855.40 and `mean_admissible_size` = 850.00, exact to 2 dp.
  - Bonferroni: `mean_size` = 3171.9 (1 dp). Reference values from Exp057 README line 105: frozen draw 3021.0, resplit-mean joint coverage 0.9206. Frozen and coverage are reported as a check, not gated.
- **Headline stays the unrestricted product set.** 850.00 is a measured column only (`mean_admissible_size`, hierarchy-admissible vectors, never enforced). Every number in the report names its column. 850 is never presented as max-rank's set size.
- **New soft check (not a hard gate):** arm C seed 0, resplit-mean `mean_size` should fall roughly in [25, 36]. Outside that range, escalate before interpreting.
- **Report comparison:** Exp056 arm D (masked MLP, old Lightning trainer, seed 0): coverage 0.8999, mean size 30.06 [28.54, 31.73], frozen 29.65, median 28.4. The 13-label MLP has a prior masked number, so this arm is a replication under the no-enforcement convention, not a first measurement.
- **Unchanged:** the identity check, coverage bands, empty_rate > 0.5%, hit_cap and seed-spread triggers, and the arms and seeds. No extra arms.
- The rerun writes any new escalation to `ESCALATION_rerun.md`, so the gate-2 `ESCALATION.md` is not overwritten. Run log and gates from the stopped run are replaced by the rerun.
