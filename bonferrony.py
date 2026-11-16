"""Per-label split conformal with analytic multiplicity corrections.

Three variants, all identical except for the level each column is calibrated at:

    none        alpha            independent per-label calibration. NOT jointly valid -
                                 this is the motivating failure the thesis rests on, kept
                                 in every comparison as the baseline that breaks.
    bonferroni  alpha / m        union bound over the m members.
    sidak       1-(1-alpha)^(1/m)  union bound sharpened under independence.

Šidák stays in every run on purpose: it buys only a few tenths of a pp over Bonferroni,
which is the evidence that the available headroom is label dependence and admissibility,
not union-bound looseness.

Usage
-----
    python bonferrony.py --clusters configs/clusters/example.yaml --experiment experiments/046
    python bonferrony.py ... --corrections bonferroni sidak

Importable
----------
    from bonferrony import calibrate_per_label, effective_alpha, run
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import conformal_common as cc

METHOD = "per_label"
CORRECTIONS = ("none", "bonferroni", "sidak")


def effective_alpha(alpha: float, m: int, correction: str) -> float:
    """Per-column miscoverage level implied by a correction."""
    if correction == "none":
        return alpha
    if correction == "bonferroni":
        return alpha / m
    if correction == "sidak":
        return 1.0 - (1.0 - alpha) ** (1.0 / m)
    raise ValueError(f"unknown correction {correction!r}; expected one of {CORRECTIONS}")


def calibrate_per_label(
    group: cc.GroupData, alpha: float, correction: str
) -> tuple[Dict[str, float], Dict]:
    """One split-conformal quantile per column, at the corrected level."""
    n = group.n
    if n == 0:
        raise RuntimeError("no usable calibration rows for this group")

    a_eff = effective_alpha(alpha, group.m, correction)
    k = cc.conformal_rank(n, a_eff)

    thresholds = {lab: cc.order_statistic(group.scores[:, j], k)
                  for j, lab in enumerate(group.labels)}
    diagnostics = {
        "n_calibration": n,
        "m": group.m,
        "correction": correction,
        "alpha_effective": a_eff,
        "k": k,
        # k > n means the level is finer than the calibration set can resolve: the
        # threshold is +inf and both classes are always kept (trivially valid, useless).
        "degenerate": k > n,
        "positives_per_label": group.positives_per_label(),
    }
    return thresholds, diagnostics


def run(
    clusters_path: str,
    experiment_dir: str,
    alpha: float | None = None,
    corrections: Sequence[str] = CORRECTIONS,
    stem: str | None = None,
    force: bool = False,
    run_gate0: bool = True,
    splits: cc.Splits | None = None,
) -> "cc.pd.DataFrame":
    """Calibrate + evaluate every (cluster, correction) pair. Returns the results frame."""
    bad = [c for c in corrections if c not in CORRECTIONS]
    if bad:
        raise ValueError(f"unknown corrections {bad}; expected from {CORRECTIONS}")

    clusters, spec = cc.load_clusters(clusters_path)
    alpha = alpha if alpha is not None else float(spec.get("alpha", 0.1))
    splits = splits or cc.load_splits(stem=stem, run_gate0=run_gate0)
    cache = cc.ThresholdCache(experiment_dir, splits, force=force)

    records: List[Dict] = []
    for cluster in clusters:
        cal = cc.extract_group(splits.calibration, cluster.labels)
        ev = cc.extract_group(splits.evaluation, cluster.labels)

        for correction in corrections:
            thresholds, diagnostics, cached = cache.get_or_compute(
                METHOD, cluster, alpha,
                lambda c=cal, k=correction: calibrate_per_label(c, alpha, k),
                variant=correction,
            )
            if diagnostics.get("degenerate"):
                print(f"!! WARNING [{METHOD}/{correction}]: cluster {cluster.name!r} needs "
                      f"order statistic {diagnostics['k']} of only {diagnostics['n_calibration']} "
                      f"calibration rows; thresholds are +inf and every class is retained.")

            rec = {
                "method": METHOD,
                "variant": correction,
                "cluster": cluster.name,
                "m": cluster.m,
                "alpha": alpha,
                "alpha_effective": diagnostics["alpha_effective"],
                "labels": "|".join(cluster.labels),
                "n_calibration": cal.n,
                "jointly_valid": correction != "none",
                "thresholds_cached": cached,
            }
            rec.update(cc.evaluate(ev, thresholds))
            rec.update({f"threshold[{lab}]": thresholds[lab] for lab in cluster.labels})
            records.append(rec)

    df = cc.results_to_frame(records)
    cc.print_coverage_table(df, alpha)
    if "none" in corrections:
        print("\nnote: variant 'none' has no joint guarantee by construction - it is the "
              "baseline failure, not a competitor.")
    cc.save_results(df, experiment_dir, f"{METHOD}_results")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cc.add_common_args(p)
    p.add_argument("--corrections", nargs="+", default=list(CORRECTIONS), choices=CORRECTIONS,
                   help="which corrections to run (default: all three)")
    a = p.parse_args()
    run(a.clusters, a.experiment, alpha=a.alpha, corrections=a.corrections, stem=a.stem,
        force=a.force, run_gate0=not a.skip_gate0)


if __name__ == "__main__":
    main()
