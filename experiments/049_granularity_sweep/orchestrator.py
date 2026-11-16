#!/usr/bin/env python3
"""Granularity sweep orchestrator: 6-level partition, 13→1, evaluate all methods.

Constructs ClusterSpec objects directly (bypassing load_clusters' 2-5 assertion) to span
m = 1, 2, 3, 5, 6, 13 as pre-registered in PRE_REGISTRATION.md.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import conformal_common as cc
import max_rank
import bonferrony
import lp_scp

EXPERIMENT_DIR = os.path.dirname(__file__)
ALPHA = 0.1
ALL_LABELS = cc.ALL_LABELS


def build_clusters_for_level(level: int) -> tuple[List[cc.ClusterSpec], int]:
    """Construct ClusterSpec objects for level 0-5, bypassing load_clusters' m=2..5 guard.

    NOTE: ClusterSpec objects constructed directly (Exp049 ONLY) to bypass
    conformal_common.load_clusters' m=2..5 assertion. This guard is intentional and required
    for all other experiments. It is lifted only for this granularity sweep, which spans
    m=1, m=6, and m=13 per Tom Hagan's pre-registered spec.
    """
    if level == 0:
        clusters = [cc.ClusterSpec(name=lab, labels=[lab], note="singleton") for lab in ALL_LABELS]
        return clusters, 13
    elif level == 1:
        clusters = [
            cc.ClusterSpec(name="cardio", labels=["Enlarged Cardiomediastinum", "Cardiomegaly"], note="m=2"),
        ]
        reserved = {"Enlarged Cardiomediastinum", "Cardiomegaly"}
        for lab in ALL_LABELS:
            if lab not in reserved:
                clusters.append(cc.ClusterSpec(name=lab, labels=[lab], note="singleton"))
        return clusters, 12
    elif level == 2:
        clusters = [
            cc.ClusterSpec(name="cardio", labels=["Enlarged Cardiomediastinum", "Cardiomegaly"], note="m=2"),
            cc.ClusterSpec(name="pneumonia_tri", labels=["Lung Opacity", "Consolidation", "Pneumonia"], note="m=3"),
        ]
        reserved = {"Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Consolidation", "Pneumonia"}
        for lab in ALL_LABELS:
            if lab not in reserved:
                clusters.append(cc.ClusterSpec(name=lab, labels=[lab], note="singleton"))
        return clusters, 10
    elif level == 3:
        clusters = [
            cc.ClusterSpec(name="cardio", labels=["Enlarged Cardiomediastinum", "Cardiomegaly"], note="m=2"),
            cc.ClusterSpec(
                name="lung_hub",
                labels=["Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis"],
                note="m=6"
            ),
        ]
        reserved = {
            "Enlarged Cardiomediastinum", "Cardiomegaly",
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis"
        }
        for lab in ALL_LABELS:
            if lab not in reserved:
                clusters.append(cc.ClusterSpec(name=lab, labels=[lab], note="singleton"))
        return clusters, 7
    elif level == 4:
        clusters = [
            cc.ClusterSpec(name="cardio", labels=["Enlarged Cardiomediastinum", "Cardiomegaly"], note="m=2"),
            cc.ClusterSpec(
                name="lung_hub",
                labels=["Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis"],
                note="m=6"
            ),
            cc.ClusterSpec(
                name="residual",
                labels=["Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"],
                note="m=5"
            ),
        ]
        return clusters, 3
    elif level == 5:
        clusters = [
            cc.ClusterSpec(name="all13", labels=ALL_LABELS, note="m=13"),
        ]
        return clusters, 1
    else:
        raise ValueError(f"unknown level {level}")


def run_level(level: int, splits: cc.Splits, cache: cc.ThresholdCache) -> List[Dict]:
    """Run one level: max_rank, bonferrony, lp_scp on all clusters."""
    clusters, n_clusters = build_clusters_for_level(level)
    records = []

    for cluster in clusters:
        cal = cc.extract_group(splits.calibration, cluster.labels)
        ev = cc.extract_group(splits.evaluation, cluster.labels)
        min_pos = min(cal.positives_per_label().values())
        below_min = min_pos < 150

        # max_rank
        thresholds, diagnostics, _ = cache.get_or_compute(
            "max_rank", cluster, ALPHA, lambda: max_rank.calibrate_max_rank(cal, ALPHA)
        )
        metrics = cc.evaluate(ev, thresholds)
        records.append({
            "level": level, "n_clusters": n_clusters, "cluster_name": cluster.name, "m": cluster.m,
            "method": "max_rank", "variant": "", "alpha": ALPHA, "alpha_effective": ALPHA,
            "n_eval": metrics["n_eval"], "n_dropped_uncertain": metrics["n_dropped_uncertain"],
            "joint_coverage": metrics["joint_coverage"], "mean_joint_set_size": metrics["mean_joint_set_size"],
            "mean_label_set_size": np.nan, "fully_decisive_rate": metrics["fully_decisive_rate"],
            "decisive_accuracy": metrics["decisive_accuracy"],
            "hierarchy_violation_rate_decisive": metrics["hierarchy_violation_rate_decisive"],
            "degenerate": diagnostics.get("degenerate", False), "min_positive_per_label": min_pos,
            "below_min_support_flag": below_min,
        })

        # per_label (bonferrony)
        for correction in ["none", "bonferroni", "sidak"]:
            thresholds, diagnostics, _ = cache.get_or_compute(
                "per_label", cluster, ALPHA,
                lambda c=correction: bonferrony.calibrate_per_label(cal, ALPHA, c),
                variant=correction,
            )
            metrics = cc.evaluate(ev, thresholds)
            alpha_eff = bonferrony.effective_alpha(ALPHA, cluster.m, correction)
            records.append({
                "level": level, "n_clusters": n_clusters, "cluster_name": cluster.name, "m": cluster.m,
                "method": "per_label", "variant": correction, "alpha": ALPHA, "alpha_effective": alpha_eff,
                "n_eval": metrics["n_eval"], "n_dropped_uncertain": metrics["n_dropped_uncertain"],
                "joint_coverage": metrics["joint_coverage"], "mean_joint_set_size": metrics["mean_joint_set_size"],
                "mean_label_set_size": metrics["mean_label_set_size"], "fully_decisive_rate": metrics["fully_decisive_rate"],
                "decisive_accuracy": metrics["decisive_accuracy"],
                "hierarchy_violation_rate_decisive": metrics["hierarchy_violation_rate_decisive"],
                "degenerate": diagnostics.get("degenerate", False), "min_positive_per_label": min_pos,
                "below_min_support_flag": below_min,
            })

        # lp_scp
        train = cc.extract_group(splits.validation, cluster.labels)
        struct = lp_scp.fit_structure(train)
        for arm_name, mu, nu, enforce_hier in lp_scp.ARMS:
            threshold, diagnostics = lp_scp.calibrate_lp_scp(cal, struct, ALPHA, mu, nu, enforce_hierarchy=enforce_hier)
            region = lp_scp.build_region(ev, struct, threshold, mu, nu, enforce_hierarchy=enforce_hier, nonempty_guard=False)
            metrics = cc.evaluate_joint(ev, region)
            records.append({
                "level": level, "n_clusters": n_clusters, "cluster_name": cluster.name, "m": cluster.m,
                "method": "lp_scp", "variant": arm_name, "alpha": ALPHA, "alpha_effective": ALPHA,
                "n_eval": metrics["n_eval"], "n_dropped_uncertain": metrics["n_dropped_uncertain"],
                "joint_coverage": metrics["joint_coverage"], "mean_joint_set_size": metrics["mean_joint_set_size"],
                "mean_label_set_size": np.nan, "fully_decisive_rate": metrics["fully_decisive_rate"],
                "decisive_accuracy": metrics["decisive_accuracy"],
                "hierarchy_violation_rate_decisive": metrics["hierarchy_violation_rate_decisive"],
                "degenerate": diagnostics.get("degenerate", False), "min_positive_per_label": min_pos,
                "below_min_support_flag": below_min,
            })

    return records


def main():
    """Orchestrate all levels."""
    print("=" * 80)
    print("Exp049 — Granularity sweep: 13→1 clustering")
    print("=" * 80)

    print("\nGate 0: loading and checking splits...")
    splits = cc.load_splits(stem=None, with_validation=True, with_train=True, run_gate0=True)
    print(f"  Calibration: {splits.calibration.shape[0]} rows")
    print(f"  Evaluation: {splits.evaluation.shape[0]} rows")
    print("  Gate 0 PASSED")

    cache = cc.ThresholdCache(EXPERIMENT_DIR, splits, force=False)
    all_records = []

    for level in range(6):
        print(f"\nLevel {level}/5: running all methods...", end=" ", flush=True)
        records = run_level(level, splits, cache)
        all_records.extend(records)
        print(f"({len(records)} rows)")

    # Write raw results
    raw_df = pd.DataFrame(all_records)
    raw_path = os.path.join(EXPERIMENT_DIR, "results_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nWrote results_raw.csv ({len(raw_df)} rows)")

    # Compute aggregate
    agg_records = []
    for level in range(6):
        level_data = raw_df[raw_df["level"] == level]
        n_clusters = level_data["n_clusters"].iloc[0]
        for method in ["max_rank", "per_label", "lp_scp"]:
            variant = {"max_rank": "", "per_label": "bonferroni", "lp_scp": "admissible"}[method]
            subset = level_data[(level_data["method"] == method) & (level_data["variant"] == variant)]
            if len(subset) == 0:
                continue
            weights = subset["m"].values / 13.0
            agg_records.append({
                "level": level, "n_clusters": n_clusters, "method": method,
                "joint_coverage_wmean": (subset["joint_coverage"].values * weights).sum(),
                "fully_decisive_rate_wmean": (subset["fully_decisive_rate"].values * weights).sum(),
            })
    agg_df = pd.DataFrame(agg_records)
    agg_path = os.path.join(EXPERIMENT_DIR, "results_aggregate.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Wrote results_aggregate.csv ({len(agg_df)} rows)")

    # Plot
    print(f"Creating plot...")
    create_plot(agg_df, os.path.join(EXPERIMENT_DIR, "granularity_tradeoff.png"))

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)
    return raw_df, agg_df


def create_plot(agg_df: pd.DataFrame, output_path: str):
    """Create granularity_tradeoff.png."""
    fig, ax = plt.subplots(figsize=(18, 11), dpi=100)
    colors = {"max_rank": "#1f77b4", "per_label": "#ff7f0e", "lp_scp": "#2ca02c"}
    labels_plot = {"max_rank": "max_rank", "per_label": "bonferrony", "lp_scp": "lp_scp"}

    for method in ["max_rank", "per_label", "lp_scp"]:
        subset = agg_df[agg_df["method"] == method].sort_values("level", ascending=False)
        if len(subset) > 0:
            x = subset["fully_decisive_rate_wmean"].values
            y = subset["joint_coverage_wmean"].values
            ax.plot(x, y, marker="o", linewidth=3, markersize=10, color=colors[method],
                   label=labels_plot[method])
            for xx, yy, nc in zip(x, y, subset["n_clusters"].values):
                ax.annotate(f"{int(nc)}", xy=(xx, yy), xytext=(5, 5), textcoords="offset points",
                           fontsize=17, fontweight="bold", color=colors[method])

    ax.axhline(y=0.90, color="gray", linestyle="--", linewidth=2, alpha=0.7, label="target (1-α=0.90)")
    ax.set_xlabel("Fully Decisive Rate (label-weighted mean)", fontsize=22, fontweight="bold")
    ax.set_ylabel("Joint Coverage (label-weighted mean)", fontsize=22, fontweight="bold")
    ax.set_title("Granularity Tradeoff: Joint Coverage vs Decisiveness (6-level sweep)",
                fontsize=28, fontweight="bold", pad=20)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_facecolor("white")
    ax.legend(fontsize=20, loc="best", framealpha=0.95)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.80, 1.02)

    caption = (
        "bonferrony = Bonferroni arm; lp_scp = admissible arm (μ=ν=1, hard hierarchy constraint); "
        "independent/Šidák/base/penalised arms in results_raw.csv only. "
        "L4/L5 forced/null groupings, not dependence claims."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=15, style="italic", wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
