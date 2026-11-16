#!/usr/bin/env python3
"""Exp050: Direct joint coverage computation without enumeration.

Computes joint coverage of 13 independent per-label conformal sets (α=0.1, no correction)
on the test split. No clustering, no enumeration of 2^13 states — just direct per-row AND.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

# Add repo root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import conformal_common as cc

EXPERIMENT_DIR = os.path.dirname(__file__)
ALPHA = 0.1
ALL_LABELS = cc.ALL_LABELS

# Paths
TEST_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              "results", "outputs", "mimic",
                              "densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809_test.csv")
THRESHOLDS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "049_granularity_sweep", "thresholds")


def load_per_label_threshold(label: str, alpha: float = 0.1) -> float:
    """Load the cached per-label[none] threshold for a label."""
    path = os.path.join(THRESHOLDS_DIR, f"per_label_{label}_a{alpha:.4f}_none.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"threshold not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data["thresholds"][label]


def main():
    print("=" * 80)
    print("Exp050 — Direct joint coverage (no enumeration)")
    print("=" * 80)

    # Gate 0: load and check splits
    print("\nGate 0: loading and checking splits...")
    splits = cc.load_splits(stem=None, with_validation=False, with_train=False, run_gate0=True)
    print(f"  Test (evaluation): {splits.evaluation.shape[0]} rows")
    print("  Gate 0 PASSED")

    # Load test data
    print("\nLoading test CSV...")
    df_test = pd.read_csv(TEST_CSV_PATH)
    print(f"  {df_test.shape[0]} total rows")

    # Filter rows: drop any row where any label is uncertain (-1) or missing (NaN)
    print("\nFiltering: dropping rows with uncertain or missing ground truth...")
    y_raw = df_test[[f"{lab}_true_label" for lab in ALL_LABELS]].to_numpy(dtype=float)
    probs = df_test[[f"{lab}_prob" for lab in ALL_LABELS]].to_numpy(dtype=float)

    keep = np.all(np.isfinite(y_raw) & (y_raw >= 0.0), axis=1) & np.all(np.isfinite(probs), axis=1)
    n_dropped = (~keep).sum()

    y = y_raw[keep].astype(int)
    p = probs[keep]
    n_eval = y.shape[0]

    print(f"  Rows after filtering: {n_eval}")
    print(f"  Rows dropped (uncertain or missing): {n_dropped}")

    # Load per-label thresholds
    print(f"\nLoading {len(ALL_LABELS)} per-label[none] thresholds (α={ALPHA})...")
    thresholds = {}
    for lab in ALL_LABELS:
        thresholds[lab] = load_per_label_threshold(lab, ALPHA)
    print(f"  Loaded {len(thresholds)} thresholds")

    # Compute per-label prediction sets and coverage
    print("\nComputing per-label prediction sets and joint coverage...")

    # Nonconformity scores: s = 1-p if y==1, else p
    scores = np.where(y == 1, 1.0 - p, p)  # shape (n_eval, 13)

    # Per-label prediction sets: class c is in set iff class_score(p)[c] <= t
    # class_score(p) = [p, 1-p]
    class_scores_0 = p  # score of class 0
    class_scores_1 = 1.0 - p  # score of class 1

    # For each label, determine which classes are in the set
    sets_0 = np.zeros((n_eval, len(ALL_LABELS)), dtype=bool)  # class 0 in set?
    sets_1 = np.zeros((n_eval, len(ALL_LABELS)), dtype=bool)  # class 1 in set?

    for j, lab in enumerate(ALL_LABELS):
        t = thresholds[lab]
        sets_0[:, j] = class_scores_0[:, j] <= t
        sets_1[:, j] = class_scores_1[:, j] <= t

    # Per-label coverage: true label in its prediction set
    # y[:, j] is 0 or 1 for each label
    label_survives = np.where(y == 0, sets_0, sets_1)  # shape (n_eval, 13)

    # Per-label coverage (fraction of rows where label survives)
    per_label_coverage = label_survives.mean(axis=0)

    # Joint coverage: all 13 labels survive
    row_survives = label_survives.all(axis=1)  # shape (n_eval,)
    direct_joint_coverage = row_survives.mean()

    # Fully decisive rate: fraction of rows where ALL 13 labels' prediction sets are singletons
    # A row is fully decisive if, for every label, exactly one class is in its prediction set
    # This means: for each label, either (sets_0 and not sets_1) or (not sets_0 and sets_1)
    fully_decisive = np.all((sets_0 & ~sets_1) | (~sets_0 & sets_1), axis=1)  # shape (n_eval,)
    fully_decisive_rate = fully_decisive.mean()

    print(f"\n  Direct joint coverage: {direct_joint_coverage:.10f}")
    print(f"  Fully decisive rate: {fully_decisive_rate:.10f}")
    print(f"  Per-label coverage (sanity check):")
    for j, lab in enumerate(ALL_LABELS):
        print(f"    {lab:30s} {per_label_coverage[j]:.10f}")

    # Save results
    results = {
        "n_eval": int(n_eval),
        "n_dropped_uncertain": int(n_dropped),
        "direct_joint_coverage": float(direct_joint_coverage),
        "fully_decisive_rate": float(fully_decisive_rate),
        "per_label_coverage": {lab: float(per_label_coverage[j]) for j, lab in enumerate(ALL_LABELS)},
    }

    results_path = os.path.join(EXPERIMENT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {results_path}")

    return results


if __name__ == "__main__":
    main()
