"""Label-Powerset Split Conformal Prediction with structural penalties.

Katsios & Papadopoulos, "Incorporating Structural Penalties in Multi-label Conformal
Prediction", COPA 2025 (PMLR 266:1-20), extending Papadopoulos (2014) LP-SCP and the
Mahalanobis measure of Katsios & Papadopoulos (2024).

This is a *different framework* from max_rank.py / bonferrony.py, not another variant of
the shared score. Those build m per-label sets and take the Cartesian product; LP-SCP
scores every joint state in the power-set directly and returns a region over states. Both
produce joint label-vector regions, so joint coverage and mean joint set size compare
fairly - but the base score is not the locked `s = 1-p if y=1 else p`.

Score (paper eq. 30), for instance x and candidate joint state y:

    a(x, y) = sqrt(r^T Sigma^-1 r) + mu * Hp(y) + nu * Cp(y),    r = |y - o(x)|

  Sigma  m x m covariance of error vectors on the proper-training split
  Hp(y)  min normalised Hamming distance from y to any observed training label-set
         (eq. 25) - 0 if that exact combination was seen
  Cp(y)  1 - (empirical frequency of Card(y) in training) (eq. 29)

Split roles (Tom Hagan's ruling, 2026-10-28): validation -> proper-training (Sigma, Hp
vocabulary, Cp frequencies), conformal -> calibration, test -> evaluation. Sigma, Hp and
Cp therefore inherit the standing all-13-clean calibration-population caveat
(RESEARCH_DIRECTION.md section 3a).

Three pre-registered arms:
    base        mu = nu = 0           Mahalanobis only
    penalised   mu = nu = 1           + Hp + Cp, no hard constraint
    admissible  mu = nu = 1 + hier    + Hp + Cp, score = +inf on hierarchy-violating states

mu = nu = 1 is fixed a priori, not selected on the outcome - the paper's own weights were
grid-searched and reported top-3, which is selection on the result. `--grid` runs the full
sensitivity table as a labeled exploratory appendix, reported whole, never cherry-picked.

Non-empty guard: the paper's eq. 20 unions the argmax-p-value label-set into the region so
it is never empty. Both variants are computed. Because the argmin-score state is in the
region whenever the region is non-empty, unioning it always is equivalent to adding it
only when the region would be empty.

Usage
-----
    python lp_scp.py --clusters configs/clusters/exp046.yaml --experiment experiments/047
    python lp_scp.py ... --grid            # exploratory mu/nu sensitivity appendix
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import conformal_common as cc

# ALL_LABELS from conformal_common for diagnostics
_ALL_LABELS = cc.ALL_LABELS

METHOD = "lp_scp"

# Fixed a priori. Do not tune these on a result.
MU_PRIMARY = 1.0
NU_PRIMARY = 1.0

GRID = [0.0, 0.25, 0.5, 1.0, 2.0]

# Sigma is regularised only if it is worse-conditioned than this, and the ridge is stated.
COND_LIMIT = 1e10


def compute_all13_clean_diagnostic(df: pd.DataFrame, cluster_labels: Sequence[str]) -> Dict[str, float]:
    """Diagnostic: fraction of rows with no uncertain (-1) values on all 13 labels.

    Returns:
        Dict with keys: n_total, n_cluster_clean, n_all13_clean, all13_clean_fraction
    """
    n_total = len(df)

    # Rows clean on cluster's own labels (what extract_group filters)
    cluster_cols = [f"{lab}_true_label" for lab in cluster_labels]
    cluster_clean_mask = np.all(np.isfinite(df[cluster_cols].to_numpy(dtype=float)) &
                                (df[cluster_cols].to_numpy(dtype=float) >= 0.0), axis=1)
    n_cluster_clean = int(cluster_clean_mask.sum())

    # Rows clean on all 13 labels (§3a all-13-clean population)
    all13_cols = [f"{lab}_true_label" for lab in _ALL_LABELS]
    all13_clean_mask = np.all(np.isfinite(df[all13_cols].to_numpy(dtype=float)) &
                              (df[all13_cols].to_numpy(dtype=float) >= 0.0), axis=1)
    n_all13_clean = int(all13_clean_mask.sum())

    all13_clean_fraction = n_all13_clean / n_total if n_total > 0 else 0.0

    return {
        "n_total": n_total,
        "n_cluster_clean": n_cluster_clean,
        "n_all13_clean": n_all13_clean,
        "all13_clean_fraction": all13_clean_fraction,
    }


@dataclass
class TrainingStructure:
    """Everything the score needs from the proper-training split, for one cluster."""

    labels: List[str]
    sigma_inv: np.ndarray       # (m, m)
    hp: np.ndarray              # (2^m,) Hamming penalty per joint state
    cp: np.ndarray              # (2^m,) cardinality penalty per joint state
    diagnostics: Dict


def fit_structure(train: cc.GroupData) -> TrainingStructure:
    """Sigma, Hp and Cp from the proper-training split (paper section 3, step 1)."""
    m = train.m
    states = cc.enumerate_states(m)

    # Sigma: covariance of error vectors r = |y - o(x)| on proper-training rows.
    r = np.abs(train.y - train.probs)
    sigma = np.cov(r, rowvar=False)
    sigma = np.atleast_2d(sigma)

    cond = float(np.linalg.cond(sigma))
    ridge = 0.0
    if not np.isfinite(cond) or cond > COND_LIMIT:
        # documented ridge, never a silent pseudo-inverse. Floored so a
        # zero-trace (all-constant-error) Sigma doesn't leave a 0 ridge.
        trace_val = float(np.trace(sigma))
        ridge = 1e-6 * trace_val / m if trace_val > 0 else 1e-6
        sigma = sigma + ridge * np.eye(m)
    sigma_inv = np.linalg.inv(sigma)

    # Hp: min normalised Hamming distance from each state to any observed label-set.
    observed = np.unique(train.y, axis=0)
    ham = (states[:, None, :] != observed[None, :, :]).sum(axis=2) / m
    hp = ham.min(axis=1)

    # Cp: 1 - empirical frequency of the state's cardinality.
    card_train = train.y.sum(axis=1)
    freq = np.bincount(card_train, minlength=m + 1) / train.n
    cp = 1.0 - freq[states.sum(axis=1)]

    return TrainingStructure(
        labels=train.labels,
        sigma_inv=sigma_inv,
        hp=hp,
        cp=cp,
        diagnostics={
            "n_proper_training": train.n,
            "sigma_condition_number": cond,
            "sigma_ridge": ridge,
            "n_observed_label_sets": int(observed.shape[0]),
            "n_states": int(2 ** m),
            "unobserved_states": int((hp > 0).sum()),
            "hp_min": float(hp.min()), "hp_max": float(hp.max()),
            "cp_min": float(cp.min()), "cp_max": float(cp.max()),
        },
    )


def state_scores(
    probs: np.ndarray, struct: TrainingStructure, mu: float, nu: float
) -> np.ndarray:
    """Extended Mahalanobis score of every joint state, shape (n, 2^m)."""
    states = cc.enumerate_states(probs.shape[1])
    r = np.abs(states[None, :, :] - probs[:, None, :])           # (n, S, m)
    quad = np.sum((r @ struct.sigma_inv) * r, axis=-1)
    return np.sqrt(np.maximum(quad, 0.0)) + mu * struct.hp[None, :] + nu * struct.cp[None, :]


def true_state_scores(group: cc.GroupData, struct: TrainingStructure, mu: float, nu: float) -> np.ndarray:
    """Score of each row's own (true) joint state - the calibration scores."""
    all_scores = state_scores(group.probs, struct, mu, nu)
    return all_scores[np.arange(group.n), cc.state_indices(group.y)]


def inadmissible_mask(labels: Sequence[str]) -> np.ndarray:
    """(2^m,) True where the joint state violates a hard hierarchy constraint."""
    states = cc.enumerate_states(len(labels))
    return ~cc.hierarchy_admissible(labels, states)


class _LpScpCacheWithSplit(cc.ThresholdCache):
    """Extends ThresholdCache to include proper_training_split in cache keys.

    Prevents cache collisions when running both train and validation versions.
    """

    def __init__(self, experiment_dir: str, splits: cc.Splits, proper_training_split: str, force: bool = False):
        super().__init__(experiment_dir, splits, force)
        self.proper_training_split = proper_training_split

    def _path(self, method: str, cluster: str, alpha: float, variant: str) -> str:
        tag = (f"{method}_{cluster}_a{alpha:.4f}"
               + (f"_{variant}" if variant else "")
               + f"_proper={self.proper_training_split}")
        return os.path.join(self.dir, tag + ".json")

    def _key(self, method: str, cluster: cc.ClusterSpec, alpha: float, variant: str) -> Dict:
        key = super()._key(method, cluster, alpha, variant)
        key["proper_training_split"] = self.proper_training_split
        return key


def calibrate_lp_scp(
    cal: cc.GroupData,
    struct: TrainingStructure,
    alpha: float,
    mu: float,
    nu: float,
    enforce_hierarchy: bool = False,
) -> tuple[float, Dict]:
    """One split-conformal quantile over joint-state scores.

    Returns the scalar threshold. Equivalent to the paper's descending-order index
    i'_eps = floor(eps(n+1)): the k-th smallest with k = ceil((n+1)(1-eps)) is the
    same element, and reusing conformal_rank keeps one convention across the repo.
    """
    scores = true_state_scores(cal, struct, mu, nu)

    bad = 0
    if enforce_hierarchy:
        # A calibration row whose own true state is inadmissible would get +inf and
        # inflate the threshold. Hierarchy correction should make this impossible;
        # count it rather than assume it.
        bad = int((~cc.hierarchy_admissible(cal.labels, cal.y)).sum())

    k = cc.conformal_rank(cal.n, alpha)
    threshold = cc.order_statistic(scores, k)
    return threshold, {
        "n_calibration": cal.n,
        "k": k,
        "threshold": threshold,
        "degenerate": k > cal.n,
        "inadmissible_calibration_rows": bad,
        "mu": mu, "nu": nu, "enforce_hierarchy": enforce_hierarchy,
        **struct.diagnostics,
    }


def build_region(
    group: cc.GroupData,
    struct: TrainingStructure,
    threshold: float,
    mu: float,
    nu: float,
    enforce_hierarchy: bool = False,
    nonempty_guard: bool = False,
) -> np.ndarray:
    """Boolean (n, 2^m) region mask over joint states."""
    scores = state_scores(group.probs, struct, mu, nu)
    if enforce_hierarchy:
        scores = scores.copy()
        scores[:, inadmissible_mask(group.labels)] = np.inf

    region = scores <= threshold
    if nonempty_guard:
        # paper eq. 20: union the best-scoring state so the region is never empty
        region[np.arange(group.n), scores.argmin(axis=1)] = True
    return region


ARMS = [
    ("base", 0.0, 0.0, False),
    ("penalised", MU_PRIMARY, NU_PRIMARY, False),
    ("admissible", MU_PRIMARY, NU_PRIMARY, True),
]


def run(
    clusters_path: str,
    experiment_dir: str,
    alpha: float | None = None,
    stem: str | None = None,
    force: bool = False,
    run_gate0: bool = True,
    grid: bool = False,
    proper_training_split: str = "train",
    splits: cc.Splits | None = None,
) -> pd.DataFrame:
    """Run LP-SCP with configurable proper-training split (train or validation).

    Args:
        proper_training_split: "train" or "validation" — which split to use for Sigma, Hp, Cp.
    """
    clusters, spec = cc.load_clusters(clusters_path)
    alpha = alpha if alpha is not None else float(spec.get("alpha", 0.1))

    # Load splits: always conformal + test; always validation; optionally train
    splits = splits or cc.load_splits(stem=stem, with_validation=True, with_train=True, run_gate0=run_gate0)

    if proper_training_split == "train":
        if splits.train is None:
            raise RuntimeError("LP-SCP with --proper-training-split=train needs the train split")
        proper_training_frame = splits.train
    elif proper_training_split == "validation":
        if splits.validation is None:
            raise RuntimeError("LP-SCP with --proper-training-split=validation needs the validation split")
        proper_training_frame = splits.validation
    else:
        raise ValueError(f"unknown proper_training_split {proper_training_split!r}")

    # Split-aware cache to prevent collisions between train and validation runs
    cache = _LpScpCacheWithSplit(experiment_dir, splits, proper_training_split, force=force)

    print(f"\n=== Gate 0 extended: train/conformal/test disjointness ===")
    if splits.train is not None:
        report = cc.gate0_check(splits, raise_on_overlap=True)
        for key, n in sorted(report.items()):
            if n > 0:
                print(f"  ERROR: {key} has {n} overlaps")
            elif "train" in key:
                print(f"  ✓ {key}: {n} overlaps")

    # All-13-clean diagnostics (per Tom's spec, now possible with train probs)
    print(f"\n=== All-13-clean population diagnostic ===")
    for split_name in ["train", "validation"]:
        split_df = splits.train if split_name == "train" else splits.validation
        if split_df is None:
            continue
        if split_name == "train" and not any("_prob" in c for c in split_df.columns):
            continue  # skip if train has no probs
        diag = compute_all13_clean_diagnostic(split_df, _ALL_LABELS)
        print(f"\n{split_name.upper()}:")
        print(f"  (a) Total rows: {diag['n_total']}")
        print(f"  (c) All-13-clean: {diag['n_all13_clean']} / {diag['n_total']} ({diag['all13_clean_fraction']:.1%})")

    print(f"\n=== Running LP-SCP on {proper_training_split} proper-training split ===")

    records: List[Dict] = []
    for cluster in clusters:
        train = cc.extract_group(proper_training_frame, cluster.labels)
        cal = cc.extract_group(splits.calibration, cluster.labels)
        ev = cc.extract_group(splits.evaluation, cluster.labels)
        struct = fit_structure(train)

        # Per-cluster all-13-clean diagnostic
        cluster_diag = compute_all13_clean_diagnostic(proper_training_frame, cluster.labels)

        n_bad_states = int(inadmissible_mask(cluster.labels).sum())
        print(f"\n{cluster.name} (m={cluster.m}, {2 ** cluster.m} states, "
              f"{n_bad_states} hierarchy-inadmissible)")
        ridge = struct.diagnostics["sigma_ridge"]
        ridge_note = f" (ridge {ridge:.3g} applied)" if ridge else " (no ridge)"
        print(f"  Sigma condition number "
              f"{struct.diagnostics['sigma_condition_number']:.3g}{ridge_note}")
        print(f"  {struct.diagnostics['n_observed_label_sets']} of {2 ** cluster.m} states "
              f"observed in proper-training; Hp>0 on {struct.diagnostics['unobserved_states']}")
        print(f"  All-13-clean on {proper_training_split}: "
              f"{cluster_diag['n_all13_clean']} / {cluster_diag['n_total']} "
              f"({cluster_diag['all13_clean_fraction']:.1%})")
        if n_bad_states == 0:
            print("  note: no hard constraints in this cluster - 'admissible' arm is "
                  "identical to 'penalised' by construction")

        combos = ([(f"grid_mu{mu}_nu{nu}", mu, nu, False) for mu in GRID for nu in GRID]
                  if grid else [])
        for arm, mu, nu, hier in ARMS + combos:
            threshold, diag, _cached = cache.get_or_compute(
                METHOD, cluster, alpha,
                lambda c=cal, s=struct, m_=mu, n_=nu, h=hier: calibrate_lp_scp(c, s, alpha, m_, n_, h),
                variant=arm,
            )
            if diag["inadmissible_calibration_rows"]:
                print(f"  !! WARNING [{arm}]: {diag['inadmissible_calibration_rows']} "
                      f"calibration rows have hierarchy-violating true labels; their "
                      f"scores are +inf and the threshold is inflated.")

            for guard in (False, True):
                region = build_region(ev, struct, threshold, mu, nu, hier, guard)
                rec = {
                    "method": METHOD,
                    "proper_training_split": proper_training_split,
                    "variant": arm + ("+guard" if guard else ""),
                    "arm": arm,
                    "nonempty_guard": guard,
                    "cluster": cluster.name,
                    "m": cluster.m,
                    "alpha": alpha,
                    "mu": mu, "nu": nu,
                    "enforce_hierarchy": hier,
                    "is_primary": arm in {a[0] for a in ARMS},
                    "labels": "|".join(cluster.labels),
                    "n_calibration": cal.n,
                    "n_proper_training": train.n,
                    "threshold": threshold,
                    "sigma_condition_number": struct.diagnostics["sigma_condition_number"],
                    "sigma_ridge": struct.diagnostics["sigma_ridge"],
                    "n_inadmissible_states": n_bad_states,
                }
                rec.update(cc.evaluate_joint(ev, region))
                records.append(rec)

    df = cc.results_to_frame(records)

    # Combine with existing results from other proper-training-split runs (deterministic merge)
    results_path = os.path.join(experiment_dir, f"{METHOD}_results.csv")
    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        # Remove rows with same (cluster, arm, nonempty_guard, proper_training_split) to avoid duplicates
        dedup_cols = ["cluster", "arm", "nonempty_guard", "proper_training_split"]
        mask = ~existing_df[dedup_cols].apply(tuple, axis=1).isin(df[dedup_cols].apply(tuple, axis=1))
        existing_df = existing_df[mask]
        df = pd.concat([existing_df, df], ignore_index=True)

    # Report only the primary arm
    primary = df[df["is_primary"]]
    cc.print_coverage_table(primary, alpha)

    print("\nLP-SCP is a different framework from the shared-score methods: the base score "
          "is Mahalanobis + penalties, not s = 1-p if y=1 else p. Sigma/Hp/Cp inherit the "
          "all-13-clean caveat (RESEARCH_DIRECTION.md section 3a).")
    cc.save_results(df, experiment_dir, f"{METHOD}_results")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cc.add_common_args(p)
    p.add_argument("--grid", action="store_true",
                   help="also run the full mu/nu sensitivity grid as an exploratory, "
                        "non-primary appendix (reported whole, never cherry-picked)")
    p.add_argument("--proper-training-split", type=str, default="validation", choices=["train", "validation"],
                   help="which split to use as proper-training for Sigma/Hp/Cp (default: validation)")
    a = p.parse_args()

    # Load splits once, then run for the requested split
    splits = cc.load_splits(stem=a.stem, with_validation=True, with_train=True, run_gate0=not a.skip_gate0)

    run(a.clusters, a.experiment, alpha=a.alpha, stem=a.stem, force=a.force,
        run_gate0=False,  # Already done above
        grid=a.grid, proper_training_split=a.proper_training_split, splits=splits)


if __name__ == "__main__":
    main()
