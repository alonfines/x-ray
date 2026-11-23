"""Frozen probability CSVs, Gate 0, the joint-state space, and calibration/test resplits.

Every dataset is registered in DATASETS and stored as data/<name>/probs/{stem}_{split}.csv with
`{label}_prob` / `{label}_true_label` per label, plus a group column (rows of a group always share a
split) and a row-id column. load(dataset=...) returns Data that carries its labels and hierarchy.

Population: all-labels-clean rows (every true label in {0,1}, every probability finite), in every
split. For MIMIC that is ~77% of test: rows with an uncertain label are not represented and coverage
there is unknown. State this with every MIMIC result.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.mimic.data import ALL_CHEXPERT_LABELS
from models.mimic.data.hierarchy import CHEXPERT_HIERARCHY

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ("train", "validation", "conformal", "test")
PROB_CLIP = 1e-6


@dataclass(frozen=True)
class Dataset:
    name: str
    labels: list[str]                      # model labels, CSV order
    hierarchy: dict[str, list[str]]        # child -> parents; {} when the dataset has none
    group_col: str                         # rows sharing it never straddle splits (subject for MIMIC)
    id_col: str                            # one row = one id (image for MIMIC)

    @property
    def probs_dir(self) -> str:
        return os.path.join(REPO_ROOT, "data", self.name, "probs")


DATASETS = {
    "mimic": Dataset("mimic", list(ALL_CHEXPERT_LABELS), CHEXPERT_HIERARCHY, "subject_id", "dicom_id"),
    "yeast": Dataset("yeast", [f"label_{i}" for i in range(1, 15)], {}, "id", "id"),
}
LABELS: list[str] = DATASETS["mimic"].labels          # default label set (MIMIC's 13)
OUTPUTS_DIR = DATASETS["mimic"].probs_dir


@dataclass
class Data:
    """Rows of one split (or a subset of rows) of one dataset, all its labels."""

    probs: np.ndarray          # (n, L) predicted P(y=1)
    y: np.ndarray              # (n, L) in {0, 1}
    group_id: np.ndarray       # (n,) subject for MIMIC, row id for datasets without groups
    labels: list[str] = None   # column order of probs / y
    hierarchy: dict = None     # child -> parents, only measured, never enforced

    def __post_init__(self):
        if self.labels is None:
            self.labels = LABELS
        if self.hierarchy is None:
            self.hierarchy = CHEXPERT_HIERARCHY

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def subject_id(self) -> np.ndarray:    # old name, kept for experiments 058-060
        return self.group_id

    def rows(self, idx: np.ndarray) -> "Data":
        return Data(self.probs[idx], self.y[idx], self.group_id[idx], self.labels, self.hierarchy)

    @staticmethod
    def concat(parts: list["Data"]) -> "Data":
        arrs = (np.concatenate([getattr(p, f) for p in parts]) for f in ("probs", "y", "group_id"))
        return Data(*arrs, parts[0].labels, parts[0].hierarchy)


def csv_path(split: str, outputs_dir: str = OUTPUTS_DIR) -> str:
    """The one {stem}_{split}.csv in outputs_dir."""
    hits = glob.glob(os.path.join(outputs_dir, f"*_{split}.csv"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one *_{split}.csv in {outputs_dir}, found {len(hits)}")
    return hits[0]


def load(splits=("validation", "conformal", "test"), dataset: str = "mimic",
         outputs_dir: str | None = None) -> dict[str, Data]:
    """Load splits, run Gate 0 on the full frames, return the all-labels-clean rows of each."""
    ds = DATASETS[dataset]
    frames = {s: pd.read_csv(csv_path(s, outputs_dir or ds.probs_dir)) for s in splits}
    gate0(frames, (ds.group_col, ds.id_col))
    out = {}
    for s, df in frames.items():
        p = df[[f"{l}_prob" for l in ds.labels]].to_numpy(float)
        y = df[[f"{l}_true_label" for l in ds.labels]].to_numpy(float)
        keep = np.all(np.isin(y, [0.0, 1.0]), axis=1) & np.all(np.isfinite(p), axis=1)
        out[s] = Data(p[keep], y[keep].astype(int), df[ds.group_col].to_numpy()[keep], ds.labels, ds.hierarchy)
        bad = int((~admissible(ds.labels, out[s].y, ds.hierarchy)).sum())
        if bad:
            raise RuntimeError(f"Gate 0 FAILED: {bad} {s} rows have a hierarchy-violating true label vector")
    return out


def gate0(frames: dict[str, pd.DataFrame], keys=("subject_id", "dicom_id")) -> None:
    """Every split pair must be group- and row-disjoint; leakage voids the coverage guarantee."""
    names = sorted(frames)
    problems = [f"{a} vs {b}: {len(set(frames[a][k]) & set(frames[b][k]))} shared {k}"
                for i, a in enumerate(names) for b in names[i + 1:] for k in dict.fromkeys(keys)
                if set(frames[a][k]) & set(frames[b][k])]
    if problems:
        raise RuntimeError("Gate 0 FAILED - splits are not disjoint:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# Joint states and the label hierarchy
# ---------------------------------------------------------------------------

def states(m: int) -> np.ndarray:
    """All 2^m label vectors, (2^m, m), binary-counting order (first label = most significant bit)."""
    return ((np.arange(2 ** m)[:, None] >> np.arange(m)[::-1]) & 1).astype(int)


def state_index(y: np.ndarray) -> np.ndarray:
    """Row index of each label vector within states(m)."""
    return y @ (1 << np.arange(y.shape[1])[::-1])


def admissible(labels: list[str], vectors: np.ndarray, hierarchy: dict = CHEXPERT_HIERARCHY) -> np.ndarray:
    """False where a child is positive while a parent in `labels` is negative (parents outside are free)."""
    idx = {l: j for j, l in enumerate(labels)}
    ok = np.ones(len(vectors), dtype=bool)
    for child, parents in hierarchy.items():
        for parent in parents:
            if child in idx and parent in idx:
                ok &= ~((vectors[:, idx[child]] == 1) & (vectors[:, idx[parent]] == 0))
    return ok


def logits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, PROB_CLIP, 1 - PROB_CLIP)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Calibration / test draws
# ---------------------------------------------------------------------------

def pool(data: dict[str, Data]) -> Data:
    """Conformal rows followed by test rows: the pool every draw is cut from."""
    return Data.concat([data["conformal"], data["test"]])


def draws(data: dict[str, Data], n_resplits: int):
    """Yield (r, calibration_idx, test_idx), row indices into pool(data).

    r = -1 is the frozen draw (conformal / test CSVs). r = 0..n_resplits-1 permute the pool's sorted
    unique groups (subjects) with default_rng(r) and give the first half to calibration; rows follow their group.
    """
    n_conf, pooled = data["conformal"].n, pool(data)
    yield -1, np.arange(n_conf), np.arange(n_conf, pooled.n)
    subjects = np.unique(pooled.group_id)
    for r in range(n_resplits):
        in_cal = np.isin(pooled.group_id, np.random.default_rng(r).permutation(subjects)[: subjects.size // 2])
        yield r, np.flatnonzero(in_cal), np.flatnonzero(~in_cal)
