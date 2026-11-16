"""Bootstrap the calibration draw, to put uncertainty on coverage and set size.

The question this answers: max-rank's guarantee is marginal over the random calibration
draw, so one realized split can land below 1-alpha without anything being wrong. Is the
observed undercoverage a draw that fell low, or is the method systematically centred
below nominal (which would indicate an implementation problem)?

Method: resample the calibration split with replacement **by subject** - images cluster
within subjects (~3 per subject here), so row-level resampling would understate the
variance of a calibration quantile - recalibrate each method on the resample, and
evaluate on the fixed test split. The spread of test coverage across resamples is the
sampling distribution of the calibration draw.

Reported per cluster and method: mean coverage, its bootstrap SE, a percentile interval,
and the share of resamples landing below 1-alpha. Plus a percentile interval on the
max-rank minus Bonferroni mean-joint-set-size delta, paired within each resample
(RESEARCH_DIRECTION.md section 5 lists this CI as missing).

Caveats this does not fix: the test split is held fixed, so test-side sampling error is
not included; and resampling the calibration set is not the same as drawing a fresh
independent calibration set, so the interval is an approximation of the latter.

Usage
-----
    python bootstrap_coverage.py --clusters configs/clusters/exp046.yaml \
        --experiment experiments/046 --n-boot 200 --seed 0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import bonferrony
import conformal_common as cc
import max_rank


def bootstrap_cluster(
    cluster: cc.ClusterSpec,
    cal: cc.GroupData,
    ev: cc.GroupData,
    alpha: float,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One row per (resample, method) with the two primary metrics."""
    rows: List[Dict] = []
    for b in range(n_boot):
        boot = cc.resample_by_subject(cal, rng)

        mr_th, _ = max_rank.calibrate_max_rank(boot, alpha)
        bf_th, _ = bonferrony.calibrate_per_label(boot, alpha, "bonferroni")

        for method, th in (("max_rank", mr_th), ("bonferroni", bf_th)):
            m = cc.evaluate(ev, th)
            rows.append({
                "cluster": cluster.name,
                "m": cluster.m,
                "boot": b,
                "method": method,
                "joint_coverage": m["joint_coverage"],
                "mean_joint_set_size": m["mean_joint_set_size"],
            })
    return pd.DataFrame(rows)


def summarize(boot: pd.DataFrame, point: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Collapse the resamples into one summary row per (cluster, method)."""
    target = 1 - alpha
    out: List[Dict] = []
    for (cluster, method), g in boot.groupby(["cluster", "method"], sort=False):
        cov, size = g["joint_coverage"].to_numpy(), g["mean_joint_set_size"].to_numpy()
        pt = point[(point["cluster"] == cluster) & (point["method"] == method)]
        out.append({
            "cluster": cluster,
            "m": int(g["m"].iloc[0]),
            "method": method,
            "n_boot": len(g),
            "point_coverage": float(pt["joint_coverage"].iloc[0]) if len(pt) else float("nan"),
            "boot_mean_coverage": float(cov.mean()),
            "boot_se_coverage": float(cov.std(ddof=1)),
            "boot_cov_lo95": float(np.percentile(cov, 2.5)),
            "boot_cov_hi95": float(np.percentile(cov, 97.5)),
            "frac_below_target": float((cov < target).mean()),
            "point_joint_size": float(pt["mean_joint_set_size"].iloc[0]) if len(pt) else float("nan"),
            "boot_mean_joint_size": float(size.mean()),
            "boot_size_lo95": float(np.percentile(size, 2.5)),
            "boot_size_hi95": float(np.percentile(size, 97.5)),
        })
    return pd.DataFrame(out)


def summarize_deltas(boot: pd.DataFrame) -> pd.DataFrame:
    """Paired max-rank minus Bonferroni set-size delta, per resample."""
    wide = boot.pivot_table(index=["cluster", "m", "boot"], columns="method",
                            values="mean_joint_set_size").reset_index()
    out: List[Dict] = []
    for (cluster, m), g in wide.groupby(["cluster", "m"], sort=False):
        d = (g["max_rank"] - g["bonferroni"]).to_numpy()
        out.append({
            "cluster": cluster,
            "m": int(m),
            "mean_size_delta": float(d.mean()),
            "delta_lo95": float(np.percentile(d, 2.5)),
            "delta_hi95": float(np.percentile(d, 97.5)),
            "frac_maxrank_smaller": float((d < 0).mean()),
        })
    return pd.DataFrame(out)


def print_report(summary: pd.DataFrame, deltas: pd.DataFrame, alpha: float) -> None:
    print(f"\nCoverage across calibration resamples (target {1 - alpha:.2f}, "
          f"test split fixed)")
    print(f"{'cluster':<15}{'m':>3}{'method':>12}{'point':>9}{'boot mean':>11}{'SE':>8}"
          f"{'95% interval':>20}{'% below':>9}")
    print("-" * 87)
    for r in summary.itertuples():
        ci = f"[{r.boot_cov_lo95:.4f}, {r.boot_cov_hi95:.4f}]"
        print(f"{r.cluster:<15}{r.m:>3}{r.method:>12}{r.point_coverage:>9.4f}"
              f"{r.boot_mean_coverage:>11.4f}{r.boot_se_coverage:>8.4f}{ci:>20}"
              f"{r.frac_below_target * 100:>8.1f}%")

    print(f"\nPaired set-size delta (max-rank - Bonferroni), per resample")
    print(f"{'cluster':<15}{'m':>3}{'mean delta':>12}{'95% interval':>22}{'% maxrank smaller':>20}")
    print("-" * 72)
    for r in deltas.itertuples():
        ci = f"[{r.delta_lo95:+.3f}, {r.delta_hi95:+.3f}]"
        print(f"{r.cluster:<15}{r.m:>3}{r.mean_size_delta:>+12.3f}{ci:>22}"
              f"{r.frac_maxrank_smaller * 100:>19.1f}%")


def run(
    clusters_path: str,
    experiment_dir: str,
    alpha: float | None = None,
    n_boot: int = 200,
    seed: int = 0,
    stem: str | None = None,
    run_gate0: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clusters, spec = cc.load_clusters(clusters_path)
    alpha = alpha if alpha is not None else float(spec.get("alpha", 0.1))
    splits = cc.load_splits(stem=stem, run_gate0=run_gate0)
    rng = np.random.default_rng(seed)

    # point estimates on the real (un-resampled) calibration split, for reference
    point_rows, boot_frames = [], []
    for cluster in clusters:
        cal = cc.extract_group(splits.calibration, cluster.labels)
        ev = cc.extract_group(splits.evaluation, cluster.labels)

        for method, th in (
            ("max_rank", max_rank.calibrate_max_rank(cal, alpha)[0]),
            ("bonferroni", bonferrony.calibrate_per_label(cal, alpha, "bonferroni")[0]),
        ):
            m = cc.evaluate(ev, th)
            point_rows.append({"cluster": cluster.name, "method": method,
                               "joint_coverage": m["joint_coverage"],
                               "mean_joint_set_size": m["mean_joint_set_size"]})

        print(f"bootstrapping {cluster.name} (m={cluster.m}, {n_boot} resamples)...")
        boot_frames.append(bootstrap_cluster(cluster, cal, ev, alpha, n_boot, rng))

    boot = pd.concat(boot_frames, ignore_index=True)
    point = pd.DataFrame(point_rows)
    summary = summarize(boot, point, alpha)
    deltas = summarize_deltas(boot)
    print_report(summary, deltas, alpha)

    os.makedirs(experiment_dir, exist_ok=True)
    for name, df in (("bootstrap_raw", boot), ("bootstrap_summary", summary),
                     ("bootstrap_size_deltas", deltas)):
        path = os.path.join(experiment_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path}")
    return summary, deltas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cc.add_common_args(p)
    p.add_argument("--n-boot", type=int, default=200, help="number of calibration resamples")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (record it with the result)")
    a = p.parse_args()
    run(a.clusters, a.experiment, alpha=a.alpha, n_boot=a.n_boot, seed=a.seed,
        stem=a.stem, run_gate0=not a.skip_gate0)


if __name__ == "__main__":
    main()
