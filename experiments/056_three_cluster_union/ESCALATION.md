# Exp056 ESCALATION — reproduction check failed (hard stop, SPEC §9)

**Status:** run stopped at the reproduction check (SPEC §2 / §9). **No experiment metric was computed.** No arm,
partition, coverage, or efficiency number from the frozen splits is reported. Gate 0 and the identity gate passed.

## What ran
- Gate 0: subject and dicom overlap 0/0 for all three pairs; hierarchy-violation count 0 for every cluster in every split.
  Population 18,570 / 18,754 / 18,877 rows (all-13-clean); pool 37,631 rows, 11,858 subjects.
- Identity gate (Exp055 §7, m = 1, 13 labels, frozen + 100 resplits): **PASS**, max mismatch 0.000%.
- Reproduction check (SPEC §2): P_H C1 = Exp055 `cardio_pair`, L1 seed 0, frozen draw, trained with
  `conformal_methods/mlp_score.train_scorer` (the spec's required trainer).

| quantity | Exp055 (`cardio_pair`, L1 s0) | Exp056 C1 via `mlp_score.py` | match (tol 1e-6) |
|---|---|---|---|
| threshold (frozen, guarded/unguarded same) | 1.1684340 | 1.1704441 | **no** |
| best epoch | 86 | 114 | **no** |
| stop epoch | 96 | 124 | **no** |
| final holdout CE | 0.4509622 | 0.4502637 | **no** |

(Exp056 values: `reproduction_check.json`; Exp055 values read from `experiments/055_learned_score_full/resplit_metrics.csv`
and `scorer_diagnostics.csv`.)

## Diagnosis (diagnostic only, not a result)
`diag_reproduction_exp055_loop.py` (in this folder) trains Exp055's own `train_scorer` (imported unedited from
`experiments/055_learned_score_full/run.py`) on the same C1 inputs built the same way (same population, same
`default_rng(0)` holdout, same logits, same spec-order targets). Output:
`best_epoch 86, stop_epoch 96, final_holdout_ce 0.4509621858596802, threshold 1.1684340136620388`.
So **the inputs are identical to Exp055 and Exp055's training loop reproduces exactly.** The mismatch is entirely the
Lightning port in `conformal_methods/mlp_score.py`. Its own docstring says it is "statistically equivalent, not
bit-identical" to Exp055 because minibatch RNG draws differ (Lightning `DataLoader` shuffling vs Exp055's
`torch.randperm` on a `torch.Generator`). The trajectory is different, not off by one: stop − best = 10 (patience) in both.

## Question for Tom (I have not chosen)
The spec requires (a) training via `conformal_methods/mlp_score.py` and (b) a 1e-6 reproduction of Exp055's C1 values.
These cannot both hold with the module as written. Options, for your ruling:
1. **Train with Exp055's loop** (verbatim copy of `train_scorer`, which the diagnostic shows reproduces exactly), keeping
   every other spec item. This departs from "train via mlp_score.py" and needs your sign-off.
2. **Port the Exp055 RNG order into `mlp_score.py`** so it is bit-identical. That is a change to a shared module; I will not
   edit it (hard rule 1). It needs Alon/Winston.
3. **Relax the reproduction check** to a statistical match. Not what the spec says; would need a spec amendment.
4. Something else.

Nothing has been iterated. The main experiment (`run.py` loop after the check) was not executed. I will not proceed
until you rule.

## Also noted
- Guard rule (SPEC §3: argmin added only when a region is empty) differs from Exp055 (always union the argmin). Recorded as
  reading R-S2 in README; flagged for your confirmation before the run.
- `conformal_common.load_clusters` cannot load C2 (m = 6); P_H is written in `run.py` from SPEC §1. No shared module edited.

---

## RESOLUTION (Amendment 1, Tom Hagan's ruling option 1)

Resolved. The reproduction check now passes with Exp055's own training loop (threshold, best/stop epoch and holdout CE match
exactly). The full run was executed after Amendment 1 (`run.py` sha256 `97a4c90c940f8ec487fdec5f3f46d00555570b447051af3d7eb939d5fd5c8660`).
No pre-registered non-stop trigger fired. The results are in `README.md` (Results section).

**Open recording gap (for Tom):** this `run.py` does not persist the per-scorer 200-epoch cap diagnostics
(`hit_cap`, epochs run). Scorer training is deterministic, so they can be recovered by re-training the scorers, but I have not
rerun. Tell me if you want that.
