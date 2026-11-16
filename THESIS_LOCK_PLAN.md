# Thesis Work Plan

**Rewritten 2026-10-28**, superseding the 2026-08-31 lock-month plan. Living checklist —
append status inline, don't rewrite history.

Two decisions by Alon on 2026-10-28:
1. **Scope.** The thesis is **label-level clustering + joint conformal coverage on the
   unified model.** The MOST proposal's statement-extraction / omission-detection pathway is
   the grant's later phase and is out of the thesis. Design detail: `RESEARCH_DIRECTION.md`.
2. **Clean restart.** All experiment folders (032–045), the Exp032–042 summaries, and all
   derived plots and plotting scripts were deleted. Backup:
   `~/Desktop/xray_pre_reset_backup_20261028_104145.tar.gz`. New work starts at
   `experiments/046`. There are currently **no citable results.**

What survives: the three frozen probability CSVs in `results/outputs/mimic/`, the training
and calibration pipeline, `conformal_calibration/mimic/`, and
`MAX_RANK_EXPERIMENTS_SUMMARY.md`.

## Remaining work, in priority order

1. **[ ] Rebuild the headline experiment from scratch** (`experiments/046`): Gate 0
   split-overlap check, then independent / Bonferroni / Šidák / max-rank / structured on
   3–4 pre-registered clusters, scored on the locked metric set (joint coverage, mean joint
   set size, hierarchy-violation rate of decisive output). One α, one λ, one seed.
   Everything else below depends on this existing.
2. **[ ] Recompute the conditional-dependence edge graph** (isotonic → GCM residuals →
   |ρ| ≥ 0.10 floor, hierarchy pairs excluded). Needed as input to the structured method's
   soft edges and as a partition source for item 3.
3. **[ ] The partition comparison, on the locked joint metrics.** RadGraph vs.
   conditional-dependence vs. Ding vs. hierarchy vs. hand-picked, at **matched m**. This is
   what makes "clustering" a contribution rather than a preprocessing step; the previous
   attempt answered the wrong question by scoring partitions on cluster-assertion coverage.
4. **[ ] Decide the RadGraph question and close the open slot.** Either produce a
   RadGraph-derived label graph and include it in item 3, or write the paragraph recording
   that it was dropped and why. Don't leave it ambiguous — an examiner will ask why a
   relation extractor was named and never used.
5. **[ ] Bootstrap CIs on the headline set-size deltas.** Cheap, and the previous run never
   had them.
6. **[ ] One cluster at m≥4 that exercises soft edges and hard constraints together.** The
   ψ term has always been under-tested relative to the weight the write-up puts on it.
7. **[ ] Write the paragraphs that pre-empt the obvious objections**, rather than leaving
   them to be found: (a) the unified-model shared-backbone caveat and why its direction is
   conservative; (b) max-rank's <150–200-positive failure floor (Fracture) as a data-size
   limit; (c) why cluster-assertion coverage was retired; (d) structured-vs-max-rank being
   conditional on hard constraints, not a uniform win.

Explicitly out of scope: retraining anything, per-label models, report-text statement
extraction, CheXbert, omission detection, MR-RATE, α/λ sweeps beyond one sensitivity check.

## Division of labour

Tom designs and rules on validity; Jules runs the spec and reports numbers. Jules does not
choose clusters, metrics, or methods — if a spec is ambiguous he escalates rather than
picking. See `.claude/agents/`.
