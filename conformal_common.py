"""Shared machinery for joint conformal experiments on the frozen unified-model CSVs.

Everything that is *not* a calibration method lives here: locating and loading the
frozen probability CSVs, the Gate 0 subject-overlap check, the locked nonconformity
score, per-experiment threshold caching, and the locked metric set.

The two method modules (`max_rank.py`, `bonferroni.py`) import from this file and
supply only a `calibrate(...) -> {label: threshold}` function plus a CLI.

Locked decisions (see RESEARCH_DIRECTION.md) encoded here:
  * score            s = 1 - p if y == 1 else p   (true-class nonconformity, every row)
  * calibration set  *_conformal.csv ;  evaluation set  *.csv (test)
  * metrics          joint coverage, mean joint set size, hierarchy-violation rate of
                     fully-decisive output; secondary: fully-decisive rate, decisive accuracy
  * exclusions       rows with any uncertain (-1) or missing member label are dropped
                     from a group's metrics

BCOPS thresholds are deliberately not supported: they calibrate presence and absence on
different row sets and cannot express joint coverage.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from data.hierarchy import CHEXPERT_HIERARCHY

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(REPO_ROOT, "results", "outputs", "mimic")

# The 13 labels of the unified model (No Finding excluded), in CSV column order.
ALL_LABELS: List[str] = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# max-rank is not usable below this many positive calibration rows for a label
# (MAX_RANK_EXPERIMENTS_SUMMARY.md, Exp001-031: Fracture fails reproducibly).
MIN_POSITIVE_CALIBRATION_ROWS = 150

SPLIT_SUFFIX = {"test": "_test.csv", "conformal": "_conformal.csv", "validation": "_validation.csv", "train": "_train.csv"}


def resolve_checkpoint_stem(outputs_dir: str = OUTPUTS_DIR) -> str:
    """Find the single checkpoint stem that the frozen CSVs are named after."""
    cands = sorted(glob.glob(os.path.join(outputs_dir, "*_conformal.csv")))
    if not cands:
        raise FileNotFoundError(f"no *_conformal.csv in {outputs_dir}")
    if len(cands) > 1:
        raise RuntimeError(
            "more than one checkpoint's CSVs present; pass --stem explicitly:\n  "
            + "\n  ".join(os.path.basename(c) for c in cands)
        )
    return os.path.basename(cands[0])[: -len("_conformal.csv")]


def split_path(split: str, stem: str | None = None, outputs_dir: str = OUTPUTS_DIR) -> str:
    if split not in SPLIT_SUFFIX:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_SUFFIX)}")
    stem = stem or resolve_checkpoint_stem(outputs_dir)
    return os.path.join(outputs_dir, stem + SPLIT_SUFFIX[split])


# ---------------------------------------------------------------------------
# Loading + Gate 0
# ---------------------------------------------------------------------------


@dataclass
class Splits:
    """The frozen probability CSVs, loaded once and reused across groups/methods."""

    stem: str
    calibration: pd.DataFrame
    evaluation: pd.DataFrame
    validation: pd.DataFrame | None = None
    train: pd.DataFrame | None = None
    fingerprints: Dict[str, str] = field(default_factory=dict)

    def frame(self, split: str) -> pd.DataFrame:
        return {"conformal": self.calibration, "test": self.evaluation, "validation": self.validation, "train": self.train}[split]


def _fingerprint(path: str, df: pd.DataFrame) -> str:
    """Cheap content key: row count + a hash of the identity columns."""
    ids = df[["subject_id", "study_id", "dicom_id"]].to_numpy(dtype="U")
    h = hashlib.sha256(ids.tobytes()).hexdigest()[:16]
    return f"{os.path.basename(path)}:{len(df)}:{h}"


def load_splits(
    stem: str | None = None,
    outputs_dir: str = OUTPUTS_DIR,
    with_validation: bool = False,
    with_train: bool = False,
    run_gate0: bool = True,
) -> Splits:
    """Load calibration (conformal) + evaluation (test) CSVs and run Gate 0.

    The train split (if loaded) may only have true_label columns (ground-truth),
    not predicted probabilities. Other splits must have both.
    """
    stem = stem or resolve_checkpoint_stem(outputs_dir)
    wanted = ["conformal", "test"] + (["validation"] if with_validation else []) + (["train"] if with_train else [])
    frames, prints = {}, {}
    for split in wanted:
        path = split_path(split, stem, outputs_dir)
        df = pd.read_csv(path)

        # Train split may only have true_label columns; other splits must have both prob and true_label
        if split == "train":
            missing = [c for lab in ALL_LABELS for c in (f"{lab}_true_label",) if c not in df.columns]
        else:
            missing = [c for lab in ALL_LABELS for c in (f"{lab}_prob", f"{lab}_true_label") if c not in df.columns]

        if missing:
            raise RuntimeError(f"{path} is missing columns: {missing[:6]}")
        frames[split] = df
        prints[split] = _fingerprint(path, df)

    splits = Splits(
        stem=stem,
        calibration=frames["conformal"],
        evaluation=frames["test"],
        validation=frames.get("validation"),
        train=frames.get("train"),
        fingerprints=prints,
    )
    if run_gate0:
        gate0_check(splits)
    return splits


def gate0_check(splits: Splits, raise_on_overlap: bool = True) -> Dict[str, int]:
    """Gate 0: every split pair must be subject- and image-disjoint.

    Run before any number is computed. Subject-level leakage silently invalidates the
    coverage guarantee, so this raises rather than warns.
    """
    present = {k: v for k, v in
               {"conformal": splits.calibration, "test": splits.evaluation, "validation": splits.validation, "train": splits.train}.items()
               if v is not None}
    report: Dict[str, int] = {}
    problems = []
    names = sorted(present)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for key in ("subject_id", "dicom_id"):
                n = len(set(present[a][key]) & set(present[b][key]))
                report[f"{a}|{b}:{key}"] = n
                if n:
                    problems.append(f"{a} vs {b}: {n} shared {key}")
    if problems and raise_on_overlap:
        raise RuntimeError("Gate 0 FAILED - splits are not disjoint:\n  " + "\n  ".join(problems))
    return report


# ---------------------------------------------------------------------------
# Score + group extraction
# ---------------------------------------------------------------------------


@dataclass
class GroupData:
    """Probabilities, labels and nonconformity scores for one group on one split."""

    labels: List[str]
    probs: np.ndarray       # (n, m) predicted P(y=1)
    y: np.ndarray           # (n, m) in {0, 1}
    scores: np.ndarray      # (n, m) true-class nonconformity
    n_rows_before_filter: int
    subject_id: np.ndarray | None = None   # (n,) for subject-level resampling

    @property
    def n(self) -> int:
        return self.probs.shape[0]

    @property
    def m(self) -> int:
        return len(self.labels)

    def positives_per_label(self) -> Dict[str, int]:
        return {lab: int(self.y[:, j].sum()) for j, lab in enumerate(self.labels)}


def nonconformity(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Locked true-class score: s = 1 - p where y == 1, s = p where y == 0."""
    return np.where(y == 1, 1.0 - probs, probs)


def class_scores(probs: np.ndarray) -> np.ndarray:
    """Scores of both candidate classes, shape (n, m, 2) indexed [.., .., class]."""
    return np.stack([probs, 1.0 - probs], axis=-1)


def extract_group(df: pd.DataFrame, labels: Sequence[str]) -> GroupData:
    """Pull a group's probs/labels out of a split, dropping unusable rows.

    A row is dropped if any member label is uncertain (-1) or missing - the locked
    exclusion rule for joint metrics.
    """
    unknown = [lab for lab in labels if lab not in ALL_LABELS]
    if unknown:
        raise ValueError(f"not model labels: {unknown}")

    probs = df[[f"{lab}_prob" for lab in labels]].to_numpy(dtype=float)
    y_raw = df[[f"{lab}_true_label" for lab in labels]].to_numpy(dtype=float)

    keep = np.all(np.isfinite(y_raw) & (y_raw >= 0.0), axis=1) & np.all(np.isfinite(probs), axis=1)
    return GroupData(
        labels=list(labels),
        probs=probs[keep],
        y=y_raw[keep].astype(int),
        scores=nonconformity(probs[keep], y_raw[keep].astype(int)),
        n_rows_before_filter=len(df),
        subject_id=df["subject_id"].to_numpy()[keep] if "subject_id" in df.columns else None,
    )


def resample_by_subject(group: GroupData, rng: np.random.Generator) -> GroupData:
    """Bootstrap a group by resampling *subjects* with replacement.

    Images cluster within subjects (~3 per subject in these splits), so row-level
    resampling would understate the sampling variance of a calibration quantile.
    """
    if group.subject_id is None:
        raise RuntimeError("group has no subject_id; cannot resample by subject")
    subjects, inverse = np.unique(group.subject_id, return_inverse=True)
    by_subject = [np.flatnonzero(inverse == i) for i in range(subjects.size)]
    picked = rng.integers(0, subjects.size, size=subjects.size)
    idx = np.concatenate([by_subject[i] for i in picked])
    return GroupData(
        labels=group.labels,
        probs=group.probs[idx],
        y=group.y[idx],
        scores=group.scores[idx],
        n_rows_before_filter=group.n_rows_before_filter,
        subject_id=group.subject_id[idx],
    )


def conformal_rank(n: int, alpha: float) -> int:
    """Finite-sample split-conformal order statistic index k in 1..n.

    k = ceil((n + 1)(1 - alpha)); k > n means no finite threshold exists at this
    alpha for this n (the caller must then keep both classes).
    """
    return int(math.ceil((n + 1) * (1.0 - alpha)))


def order_statistic(values: np.ndarray, k: int) -> float:
    """k-th smallest of `values` (1-indexed); +inf if k exceeds the sample size."""
    if k > values.size:
        return float("inf")
    return float(np.partition(values, k - 1)[k - 1])


# ---------------------------------------------------------------------------
# Prediction sets + locked metrics
# ---------------------------------------------------------------------------


def prediction_sets(group: GroupData, thresholds: Dict[str, float]) -> np.ndarray:
    """Per-label prediction sets as a boolean (n, m, 2) mask over classes {0, 1}."""
    t = np.array([thresholds[lab] for lab in group.labels], dtype=float)
    return class_scores(group.probs) <= t[None, :, None]


def hierarchy_admissible(labels: Sequence[str], vectors: np.ndarray) -> np.ndarray:
    """Per-row admissibility of decisive label vectors under CHEXPERT_HIERARCHY.

    A vector violates the hierarchy when a child is predicted positive while a parent
    *present in the same group* is predicted negative. Parents outside the group are
    unconstrained - the group cannot speak to them.
    """
    idx = {lab: j for j, lab in enumerate(labels)}
    ok = np.ones(vectors.shape[0], dtype=bool)
    for child, parents in CHEXPERT_HIERARCHY.items():
        if child not in idx:
            continue
        child_pos = vectors[:, idx[child]] == 1
        for parent in parents:
            if parent in idx:
                ok &= ~(child_pos & (vectors[:, idx[parent]] == 0))
    return ok


def enumerate_states(m: int) -> np.ndarray:
    """All 2^m joint label vectors, shape (2^m, m), in binary-counting order."""
    return ((np.arange(2 ** m)[:, None] >> np.arange(m)[::-1]) & 1).astype(int)


def state_indices(y: np.ndarray) -> np.ndarray:
    """Index of each row's label vector within enumerate_states(m)."""
    m = y.shape[1]
    weights = (1 << np.arange(m)[::-1]).astype(np.int64)
    return (y * weights).sum(axis=1)


def evaluate_joint(group: GroupData, region: np.ndarray) -> Dict[str, float]:
    """Locked metrics for a method that returns a region over joint states directly.

    `region` is a boolean (n, 2^m) mask over enumerate_states(m). This is the
    label-powerset analogue of `evaluate()`: the region is not a product of per-label
    sets, so per-label set sizes are undefined and only the joint metrics apply.
    """
    states = enumerate_states(group.m)
    true_idx = state_indices(group.y)

    size = region.sum(axis=1).astype(float)
    covered = region[np.arange(group.n), true_idx] if group.n else np.zeros(0, bool)
    decisive = size == 1

    metrics = {
        "n_eval": float(group.n),
        "n_dropped_uncertain": float(group.n_rows_before_filter - group.n),
        "n_states": float(2 ** group.m),
        "joint_coverage": float(covered.mean()) if group.n else float("nan"),
        "mean_joint_set_size": float(size.mean()) if group.n else float("nan"),
        "median_joint_set_size": float(np.median(size)) if group.n else float("nan"),
        "empty_set_rate": float((size == 0).mean()) if group.n else float("nan"),
        "fully_decisive_rate": float(decisive.mean()) if group.n else float("nan"),
    }

    if decisive.any():
        chosen = states[region[decisive].argmax(axis=1)]
        metrics["decisive_accuracy"] = float((chosen == group.y[decisive]).all(axis=1).mean())
        metrics["hierarchy_violation_rate_decisive"] = float(
            (~hierarchy_admissible(group.labels, chosen)).mean()
        )
    else:
        metrics["decisive_accuracy"] = float("nan")
        metrics["hierarchy_violation_rate_decisive"] = float("nan")

    # share of the state space retained - the paper's cross-dataset normalisation
    metrics["mean_region_fraction"] = metrics["mean_joint_set_size"] / 2 ** group.m
    return metrics


def evaluate(group: GroupData, thresholds: Dict[str, float]) -> Dict[str, float]:
    """The locked metric set for one group under one set of thresholds."""
    sets = prediction_sets(group, thresholds)                   # (n, m, 2)
    per_label_size = sets.sum(axis=2)                           # (n, m)

    # Joint coverage: the true class of every member is retained.
    true_kept = np.take_along_axis(sets, group.y[:, :, None], axis=2)[:, :, 0]
    covered = true_kept.all(axis=1)

    # Joint set size = number of retained joint states = product of per-label sizes.
    # This is the conversion that makes per-label and joint families comparable.
    joint_size = per_label_size.prod(axis=1).astype(float)

    empty = per_label_size.min(axis=1) == 0                     # some label kept no class
    decisive = per_label_size.max(axis=1) == 1                  # all singletons

    metrics = {
        "n_eval": float(group.n),
        "n_dropped_uncertain": float(group.n_rows_before_filter - group.n),
        "joint_coverage": float(covered.mean()) if group.n else float("nan"),
        "mean_joint_set_size": float(joint_size.mean()) if group.n else float("nan"),
        "median_joint_set_size": float(np.median(joint_size)) if group.n else float("nan"),
        # average per-label set size (mean over rows and members, in [0, 2]) - the
        # per-label companion to the joint size, not a substitute for it: only the
        # joint number is comparable across the per-label and joint-state families.
        "mean_label_set_size": float(per_label_size.mean()) if group.n else float("nan"),
        "empty_set_rate": float(empty.mean()) if group.n else float("nan"),
        "fully_decisive_rate": float(decisive.mean()) if group.n else float("nan"),
    }

    if decisive.any():
        vectors = sets[decisive][:, :, 1].astype(int)           # the singleton choice
        y_dec = group.y[decisive]
        metrics["decisive_accuracy"] = float((vectors == y_dec).all(axis=1).mean())
        metrics["hierarchy_violation_rate_decisive"] = float(
            (~hierarchy_admissible(group.labels, vectors)).mean()
        )
    else:
        metrics["decisive_accuracy"] = float("nan")
        metrics["hierarchy_violation_rate_decisive"] = float("nan")

    metrics.update({f"set_size[{lab}]": float(per_label_size[:, j].mean())
                    for j, lab in enumerate(group.labels)})
    return metrics


# ---------------------------------------------------------------------------
# Cluster specs
# ---------------------------------------------------------------------------


@dataclass
class ClusterSpec:
    name: str
    labels: List[str]
    note: str = ""

    @property
    def m(self) -> int:
        return len(self.labels)


def load_clusters(path: str) -> tuple[List[ClusterSpec], Dict]:
    """Read a pre-registered cluster YAML.

    Expected shape:

        alpha: 0.1                      # optional default, CLI overrides
        clusters:
          cardio:
            labels: [Enlarged Cardiomediastinum, Cardiomegaly]
            note: hierarchy pair, m=2
          lung_core:
            labels: [Lung Opacity, Edema, Consolidation]
    """
    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    raw = spec.get("clusters")
    if not raw:
        raise ValueError(f"{path} has no 'clusters:' mapping")

    clusters = []
    for name, body in raw.items():
        labels = body["labels"] if isinstance(body, dict) else body
        note = body.get("note", "") if isinstance(body, dict) else ""
        if not (2 <= len(labels) <= 5):
            raise ValueError(f"cluster {name!r} has m={len(labels)}; the locked design allows m=2..5")
        clusters.append(ClusterSpec(name=name, labels=list(labels), note=note))
    return clusters, spec


# ---------------------------------------------------------------------------
# Per-experiment threshold cache
# ---------------------------------------------------------------------------


class ThresholdCache:
    """Caches calibrated thresholds under an experiment directory.

    One JSON per (method, cluster, alpha, variant). Each file stores the key it was
    computed under - method, sorted labels, alpha, variant, calibration-split
    fingerprint - and a cached entry is reused only on an exact key match, so a
    regenerated CSV or an edited cluster invalidates itself instead of silently
    handing back wrong thresholds.
    """

    def __init__(self, experiment_dir: str, splits: Splits, force: bool = False):
        self.dir = os.path.join(experiment_dir, "thresholds")
        os.makedirs(self.dir, exist_ok=True)
        self.splits = splits
        self.force = force

    def _path(self, method: str, cluster: str, alpha: float, variant: str) -> str:
        tag = f"{method}_{cluster}_a{alpha:.4f}" + (f"_{variant}" if variant else "")
        return os.path.join(self.dir, tag + ".json")

    def _key(self, method: str, cluster: ClusterSpec, alpha: float, variant: str) -> Dict:
        return {
            "method": method,
            "cluster": cluster.name,
            "labels": sorted(cluster.labels),
            "alpha": round(float(alpha), 10),
            "variant": variant,
            "stem": self.splits.stem,
            "calibration_fingerprint": self.splits.fingerprints["conformal"],
        }

    def get_or_compute(
        self,
        method: str,
        cluster: ClusterSpec,
        alpha: float,
        compute: Callable[[], tuple[Dict[str, float], Dict]],
        variant: str = "",
    ) -> tuple[Dict[str, float], Dict, bool]:
        """Return (thresholds, diagnostics, was_cached)."""
        path = self._path(method, cluster.name, alpha, variant)
        key = self._key(method, cluster, alpha, variant)
        if not self.force and os.path.exists(path):
            with open(path) as f:
                blob = json.load(f)
            if blob.get("key") == key:
                return blob["thresholds"], blob.get("diagnostics", {}), True

        thresholds, diagnostics = compute()
        with open(path, "w") as f:
            json.dump({"key": key, "thresholds": thresholds, "diagnostics": diagnostics}, f, indent=2)
        return thresholds, diagnostics, False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def warn_small_positive_support(group: GroupData, method: str, minimum: int = MIN_POSITIVE_CALIBRATION_ROWS) -> List[str]:
    """Loud warning for labels below the positive-support floor that breaks max-rank."""
    thin = {lab: n for lab, n in group.positives_per_label().items() if n < minimum}
    lines = []
    for lab, n in sorted(thin.items(), key=lambda kv: kv[1]):
        lines.append(
            f"!! WARNING [{method}]: {lab} has only {n} positive calibration rows "
            f"(< {minimum}). Rank-based joint calibration is known to fail at this support "
            f"(MAX_RANK_EXPERIMENTS_SUMMARY.md). Treat this cluster's numbers as unreliable."
        )
    for line in lines:
        print(line)
    return lines


def results_to_frame(records: Iterable[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    lead = [c for c in ["method", "variant", "cluster", "m", "alpha", "alpha_effective",
                        "n_eval", "joint_coverage", "mean_joint_set_size",
                        "mean_label_set_size",
                        "hierarchy_violation_rate_decisive", "fully_decisive_rate",
                        "decisive_accuracy"] if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]


def print_coverage_table(df: pd.DataFrame, alpha: float) -> None:
    print(f"\n{'cluster':<18}{'m':>3}{'variant':>14}{'n':>8}{'joint cov':>11}"
          f"{'(target)':>10}{'mean joint':>12}{'mean label':>12}{'decisive':>10}{'hier viol':>11}")
    print("-" * 109)
    for _, r in df.iterrows():
        print(f"{r['cluster']:<18}{int(r['m']):>3}{str(r.get('variant', '') or '-'):>14}"
              f"{int(r['n_eval']):>8}{r['joint_coverage']:>11.4f}{1 - alpha:>10.2f}"
              f"{r['mean_joint_set_size']:>12.3f}"
              # undefined for joint-state methods: their region is not a product of
              # per-label sets, so there is no per-label size to average
              f"{r.get('mean_label_set_size', float('nan')):>12.3f}"
              f"{r['fully_decisive_rate']:>10.3f}"
              f"{r['hierarchy_violation_rate_decisive']:>11.3f}")


def save_results(df: pd.DataFrame, experiment_dir: str, name: str) -> str:
    out = os.path.join(experiment_dir, f"{name}.csv")
    os.makedirs(experiment_dir, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return out


def add_common_args(parser) -> None:
    """CLI flags shared by both method scripts."""
    parser.add_argument("--clusters", required=True,
                        help="pre-registered cluster YAML (see configs/clusters/)")
    parser.add_argument("--experiment", required=True,
                        help="experiment directory; thresholds and results are written here")
    parser.add_argument("--alpha", type=float, default=None,
                        help="miscoverage level; overrides 'alpha' in the cluster YAML (default 0.1)")
    parser.add_argument("--stem", default=None, help="checkpoint stem of the frozen CSVs (auto-detected)")
    parser.add_argument("--force", action="store_true", help="recompute thresholds, ignoring the cache")
    parser.add_argument("--skip-gate0", action="store_true",
                        help="skip the subject-overlap check (do not use for reported results)")
