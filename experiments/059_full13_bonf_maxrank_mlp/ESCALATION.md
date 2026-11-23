# Exp059 ESCALATION: Gate 2 (reproduction) failed; no MLP arm was run

## What ran
- `run.py` (this folder), pre-registered in `README.md` before running. Gate 0 passed (`conformal.data.load`), Gate 1 passed (pool = 37,631 all-13-clean rows: validation 18,570, conformal 18,754, test 18,877).
- Arms A (Bonferroni, `per_label` singletons) and B (max-rank, `per_label` singletons) in one `conformal.run.run` call, alpha = 0.10, no guard, frozen draw + 100 resplits. Source: `run.log`, `gates.json`.
- The script stopped at Gate 2 as pre-registered. Arm C and D were not run. No MLP number was read or written.

## What it gave
Resplit means over r = 0..99 (rounded to 2 dp where relevant):

| arm (rule) | mean_size | mean_admissible_size | median_size |
|---|---|---|---|
| A (bonferroni) | 3171.89 | 941.57 | 2048 |
| B (max_rank) | 2855.40 | 850.00 | 2048 |

Frozen draw (resplit -1): bonferroni mean_size 3021.00, max_rank 2718.39.

## Why it is not good
The pre-registered anchors were: arm A mean_size = 2855.40, arm B mean_size = 850.00. Neither matches as assigned:
- Bonferroni (A) gives 3171.89, not 2855.40.
- max-rank (B) gives mean_size 2855.40 (matches the number) and mean_admissible_size 850.00 (matches the other number).

So both anchor values are reproduced exactly, but by the same arm (max-rank), in two different columns: `mean_size` and `mean_admissible_size`. The spec's arm-to-anchor mapping appears to be wrong, or the 850.00 anchor refers to the admissible count rather than the set size. This is a question about which quantity the anchor measures, not a numerical failure: the code reproduces both numbers.

I did not change the gate, relabel the arms, or pick the column that makes it pass. The mean_admissible_size reading above is a post-hoc observation from the package's existing output, computed once for this escalation, not a basis for the gate.

Note that CLAUDE.md records the same two numbers (2855.40 and 850.00) for "the 13-label max-rank references" in Exp056, which is consistent with the admissible-count reading of the second number.

## Question for Tom
1. Which quantity should Gate 2 check? Options: (a) max-rank `mean_size` = 2855.40 and max-rank `mean_admissible_size` = 850.00, with Bonferroni unanchored; (b) something else you specify.
2. Should the 850.00 anchor be treated as the admissible count (measured, not enforced), with the headline set size still the unrestricted product? That matters for the headline efficiency pair.
3. Once the gate is fixed, I will rerun from scratch with the same pre-registered arms, seeds and gates. I have not started that.

## Files in this folder
`README.md` (pre-registration), `run.py`, `run.log` (full log of the stopped run), `gates.json` (Gates 0, 1, 2), `code_hashes.txt`. No `resplits.csv`, `summary.csv`, `metrics.json` or `REPORT.md`: the run stopped before those were written.
