# Exp053 — ESCALATION 2 (new unpinned items after the Amendments; nothing run)

**Date:** 2026-11-10, 12:13 IST. **Git HEAD:** `6a26ac848b016e82c22d1bd42f61280740d3d26d`.

I re-read SPEC.md including the Amendments and checked every seed and context mention. Four items are still unpinned. Per the brief, I have not chosen. Each has a proposed reading below for a one-word confirmation. Nothing has been run.

**Q1. Split-calibration arm (§5), inner subject split.** The arm splits the calibration subjects 50/50 into A and B. No seed is given. §10 and E8 pin seeds only for the resplit permutation, not for this inner split.
- Proposed reading: inside resplit r, the inner split uses a second permutation drawn from the same generator seeded r, after the outer permutation.
- Alternative: a separate fixed seed (value to be given).

**Q2. RNG stream scope, matched null (E4).** "One sequential RNG stream seeded 1500 generating 5,000 partitions per profile." Ambiguous whether each profile (K=3, 4, 6) gets its own stream seeded 1500 or one stream runs across all three profiles.
- Proposed reading: a fresh stream seeded 1500 per profile.

**Q3. RNG stream scope, anti-clustering (E5).** "One RNG stream seeded 2000" for the 50 restarts. Same ambiguity across the three profiles.
- Proposed reading: a fresh stream seeded 2000 per profile.

**Q4. Uncalibrated context rows (§4).** "Uncalibrated K=1 Bonferroni, Šidák, max-rank at α=0.10; independent per-label at α." Not stated on which draw these are evaluated.
- Proposed reading: the frozen conformal→test draw only, with no resplits.
- Alternative: all 100 resplits, reported as means.

**What is ready once confirmed:** `run.py` design (integer-count ranks, LOO, product sets, LP-SCP chunked with `lp_scp.state_scores`), README pre-registration with HEAD and time, Gate 0 / 0.5 / identity gates, a timed resplit, and the runtime projection against the 4 h fallback. Nothing in the primary design is blocked beyond these four items.
