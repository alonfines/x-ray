# Exp055 ESCALATION — for Tom Hagan

**Scope:** all-13-clean studies, 77% of test. Pre-registration: `README.md` (this folder). Run completed, exit 0.

## What was run
Full design per `README.md`: four designed clusters (cardio_pair, lung_core, pleural_free, lung_wide), 20 random
label subsets per designed cluster (80 total), learned scorers L0/L1/L2 × seeds 0–4, classical references
(indep, Bonferroni, Šidák, max-rank, max-rank ∩ adm, Bonferroni ∩ adm), LP-SCP adm (μ = ν = 1), α = 0.10, 100
subject-level resplits of the pooled all-13-clean conformal + test population, frozen-draw reference row.

## What came out
- Gate 0 PASS (zero subject/dicom overlap; zero hierarchy violations, 84 units × 3 splits).
- Identity gate (m = 1) PASS (0.000% mismatch on frozen and all 100 resplits).
- Pilot reproduction PASS (L1 seed 0 exact on all four designed clusters).
- Designed clusters: every learned seed-arm (L0/L1/L2 × 5 seeds × 4 clusters) is inside [0.895, 0.905] (resplit mean).
- L1 seed 0 vs max-rank ∩ adm: mean-size deltas −0.578 / −0.717 / −1.548 / −3.558 (cardio / lung_core / pleural_free /
  lung_wide), all paired 95% CIs below 0, both arms inside the coverage gate.
- L1 vs L0: no CI excludes 0 on any designed cluster (pre-registered rule 1: L0 is the learned-score result).

## Why this is escalated
Pre-registered trigger "any learned-arm seed whose resplit-mean coverage is outside [0.895, 0.905]" **fired** on two
random draws:

- `rand_cardio_pair_15` and `rand_cardio_pair_19`, the **same label set {Lung Lesion, Fracture}** (duplicate draws).
- L1 and L2, all five seeds (20 seed-arms). Resplit-mean joint coverage **0.9557** (SD 0.002). Mean size exactly 1.000.

Mechanism (post-run diagnostic on the frozen CSVs, not a pre-registered check): the all-negative state carries 95.6%
of the all-13-clean conformal rows (95.5% of test). The learned score puts p ≥ 0.9 on that state for essentially every
row, so the 90% quantile lands inside that atom (threshold 0.1056). Calibration-side coverage is 0.9565 and test coverage
0.9557. The region is therefore always the single modal state. This is valid (coverage ≥ 0.90) but over-covers by about
5.6 pp, so the two-sided gate fails and the size-1 sets are not an efficiency result. Max-rank ∩ adm and Bonferroni on the same
draw cover 0.9007 and 0.9034 at sizes 2.80 and 2.83.

No designed-cluster learned arm is affected. Those two draws are excluded from the 78-of-78 random-subset comparisons
by the coverage gate (both arms must pass).

## Specific questions for Tom
1. **Does this trigger block the headline?** The trigger is literal (any learned seed, designed or random). My reading:
   it concerns random draws only, the designed results pass, and the random-draw comparisons exclude the two
   coverage-unequal draws by the gate rule. Confirm that reading, or say the headline is withheld until it is resolved.
2. **Atom-driven over-coverage.** Is a learned score that collapses to the modal state under a 95.6% atom an
   accepted over-coverage case, or should the spec add a rule for discrete atoms (e.g. randomised or tie-aware
   thresholds)? I have not changed anything; this would need a new pre-registration.
3. **Duplicate random draws.** 80 draws give 72 unique label sets (pre-registered procedure, not deduplicated). Keep
   as is, or should the control be redrawn without replacement on label sets?
4. **Designed vs random.** Under L1/L2 the designed clusters are not reliably more efficient than random same-m
   subsets. lung_core is less efficient than 17 of 20 own draws; pleural_free sits near the median; cardio and lung_wide
   beat their medians. The random draws are not matched on prevalence or difficulty, so the own-pool comparison is
   confounded. The learned-score advantage over max-rank ∩ adm and LP-SCP adm holds on 78 of 78 valid random subsets.
   Is the random control as pre-registered enough, or should it be matched on prevalence?
5. **Readings I made (README §14).** R1 identity gate instance (m = 1); R2 L0 trained with the L1 recipe; R3 global draw
   index; R5 hard constraints counted as directed edges (lung_wide = 5, not the pilot's 4); R7 "equal coverage" and
   "comparable" operationalisations. Confirm or change before the result is cited.
6. **L0 at the 200-epoch cap.** Six L0 scorers hit the cap (cardio seeds 0–3, lung_wide seeds 3–4). Cap unchanged per
   spec. The headline learned-score result is L0 (rule 1), so this needs acknowledging in any citation.

## Not escalated (for completeness)
No NaN in a reported metric, no infinite threshold, no degenerate set, no random draw with one admissible state (minimum
admissible count is 3). Bonferroni coverage-unequal on three of four designed clusters, as the spec expects.

Scope: all-13-clean studies, 77% of test. Shared-backbone caveat applies to every efficiency number above.
