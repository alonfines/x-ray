"""Head-to-head comparison of max-rank vs Bonferroni on pre-registered clusters.

Runs both calibration methods over the same clusters, on the same splits, through the
same threshold cache, and emits one merged table of the two locked primary metrics:
joint coverage and mean joint set size.

Deliberately narrow:
  * two methods only - max-rank and per-label Bonferroni. Independent-at-alpha and
    Šidák are available via `bonferrony.py --corrections`, not shown here.
  * joint set size only. Per-label size is in the per-method CSVs; only the joint
    number is comparable across method families.

Usage
-----
    python compare_methods.py --clusters configs/clusters/exp046.yaml --experiment experiments/046

Writes `method_comparison.csv` and `method_comparison.md` into the experiment directory,
alongside the per-method result CSVs the two modules write themselves.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd

import bonferrony
import conformal_common as cc
import max_rank

METRICS = ["joint_coverage", "mean_joint_set_size"]


def build_comparison(max_rank_df: pd.DataFrame, bonf_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Merge the two per-method frames into one row per cluster."""
    left = max_rank_df.set_index("cluster")
    right = bonf_df[bonf_df["variant"] == "bonferroni"].set_index("cluster")

    rows: List[dict] = []
    for cluster in left.index:
        mr, bf = left.loc[cluster], right.loc[cluster]
        if int(mr["n_eval"]) != int(bf["n_eval"]):
            # both methods filter identically, so this would mean a cache/spec mismatch
            raise RuntimeError(f"cluster {cluster!r}: evaluation rows differ between methods "
                               f"({int(mr['n_eval'])} vs {int(bf['n_eval'])})")
        rows.append({
            "cluster": cluster,
            "m": int(mr["m"]),
            "n_eval": int(mr["n_eval"]),
            "target_coverage": 1 - alpha,
            "maxrank_coverage": mr["joint_coverage"],
            "bonferroni_coverage": bf["joint_coverage"],
            "maxrank_joint_size": mr["mean_joint_set_size"],
            "bonferroni_joint_size": bf["mean_joint_set_size"],
            "size_delta": mr["mean_joint_set_size"] - bf["mean_joint_set_size"],
            "size_ratio": mr["mean_joint_set_size"] / bf["mean_joint_set_size"],
            "labels": mr["labels"],
        })
    return pd.DataFrame(rows).sort_values("m").reset_index(drop=True)


def render_markdown(df: pd.DataFrame, alpha: float, clusters_path: str, stem: str) -> str:
    head = (
        f"# max-rank vs Bonferroni\n\n"
        f"- clusters: `{clusters_path}` (pre-registered)\n"
        f"- alpha: {alpha} -> target joint coverage {1 - alpha:.2f}\n"
        f"- model: `{stem}`\n"
        f"- calibration: conformal split; evaluation: test split; Gate 0 passed\n"
        f"- score: true-class nonconformity `s = 1-p if y=1 else p`\n"
        f"- rows with any uncertain (-1) member label excluded per cluster\n\n"
        f"| cluster | m | n | target | max-rank cov | Bonf. cov | max-rank \\|set\\| | "
        f"Bonf. \\|set\\| | delta | ratio |\n"
        f"|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r.cluster} | {r.m} | {r.n_eval:,} | {r.target_coverage:.2f} | "
        f"{r.maxrank_coverage:.4f} | {r.bonferroni_coverage:.4f} | "
        f"{r.maxrank_joint_size:.3f} | {r.bonferroni_joint_size:.3f} | "
        f"{r.size_delta:+.3f} | {r.size_ratio:.3f} |\n"
        for r in df.itertuples()
    )
    note = (
        "\nJoint set size = number of retained joint states (product of per-label set "
        "sizes), the only size metric comparable across method families. A negative "
        "delta means max-rank is the smaller set at equal or better coverage. No "
        "confidence intervals: the deltas are point estimates on one test split.\n"
    )
    return head + body + note


def print_table(df: pd.DataFrame) -> None:
    print(f"\n{'cluster':<16}{'m':>3}{'n':>8}{'target':>8}"
          f"{'maxrank cov':>13}{'bonf cov':>10}{'maxrank |s|':>13}{'bonf |s|':>10}"
          f"{'delta':>9}{'ratio':>8}")
    print("-" * 98)
    for r in df.itertuples():
        print(f"{r.cluster:<16}{r.m:>3}{r.n_eval:>8}{r.target_coverage:>8.2f}"
              f"{r.maxrank_coverage:>13.4f}{r.bonferroni_coverage:>10.4f}"
              f"{r.maxrank_joint_size:>13.3f}{r.bonferroni_joint_size:>10.3f}"
              f"{r.size_delta:>+9.3f}{r.size_ratio:>8.3f}")


def run(
    clusters_path: str,
    experiment_dir: str,
    alpha: float | None = None,
    stem: str | None = None,
    force: bool = False,
    run_gate0: bool = True,
) -> pd.DataFrame:
    _, spec = cc.load_clusters(clusters_path)
    alpha = alpha if alpha is not None else float(spec.get("alpha", 0.1))

    # load the CSVs once and hand the same Splits to both methods
    splits = cc.load_splits(stem=stem, run_gate0=run_gate0)

    mr_df = max_rank.run(clusters_path, experiment_dir, alpha=alpha, force=force, splits=splits)
    bf_df = bonferrony.run(clusters_path, experiment_dir, alpha=alpha,
                           corrections=["bonferroni"], force=force, splits=splits)

    df = build_comparison(mr_df, bf_df, alpha)
    print_table(df)

    os.makedirs(experiment_dir, exist_ok=True)
    df.to_csv(os.path.join(experiment_dir, "method_comparison.csv"), index=False)
    md_path = os.path.join(experiment_dir, "method_comparison.md")
    with open(md_path, "w") as f:
        f.write(render_markdown(df, alpha, clusters_path, splits.stem))
    print(f"\nwrote {os.path.join(experiment_dir, 'method_comparison.csv')}\nwrote {md_path}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cc.add_common_args(p)
    a = p.parse_args()
    run(a.clusters, a.experiment, alpha=a.alpha, stem=a.stem, force=a.force,
        run_gate0=not a.skip_gate0)


if __name__ == "__main__":
    main()
