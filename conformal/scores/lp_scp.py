"""Label-powerset score with structural penalties (Katsios & Papadopoulos, COPA 2025, eq. 30).

    a(x, y) = sqrt(r^T Sigma^-1 r) + mu * Hp(y) + nu * Cp(y),    r = |y - p(x)|

    Sigma  covariance of the error vectors |y - p| on the proper-training split
    Hp(y)  min normalised Hamming distance from y to any label vector seen in training (0 if seen)
    Cp(y)  1 - training frequency of y's cardinality

Sigma, Hp and Cp are fit on the validation split only (run.py passes data["validation"]). Hp and Cp
come from the label vectors observed there, never from models.mimic.data.hierarchy. No +inf scores and
no admissibility filter: the hierarchy is never given to any scorer (Alon, 2026-11-22), so the MLP's
learning of it stands on its own.

mu = nu = 0 is the plain Mahalanobis score. mu = nu = 1 was fixed a priori in Exp047; don't
tune them on a result.
"""

from __future__ import annotations

import numpy as np

from ..data import Data, states

COND_LIMIT = 1e10     # Sigma gets a stated ridge only if worse-conditioned than this
CHUNK = 2048          # rows per block when scoring, to bound memory at large m


class LPSCP:
    def __init__(self, mu: float = 1.0, nu: float = 1.0):
        self.mu, self.nu = mu, nu

    def fit(self, train: Data, labels: list[str], seed: int = 0):
        cols = [train.labels.index(l) for l in labels]
        m, S = len(cols), states(len(cols))
        p, y = train.probs[:, cols], train.y[:, cols]

        sigma = np.atleast_2d(np.cov(np.abs(y - p), rowvar=False))
        cond = np.linalg.cond(sigma)
        if not np.isfinite(cond) or cond > COND_LIMIT:
            tr = float(np.trace(sigma))
            sigma = sigma + (1e-6 * tr / m if tr > 0 else 1e-6) * np.eye(m)
        sigma_inv = np.linalg.inv(sigma)

        seen = np.unique(y, axis=0)
        hp = ((S[:, None, :] != seen[None]).sum(axis=2) / m).min(axis=1)
        cp = 1.0 - (np.bincount(y.sum(axis=1), minlength=m + 1) / len(y))[S.sum(axis=1)]
        penalty = self.mu * hp + self.nu * cp

        def score(d: Data) -> np.ndarray:
            out = np.empty((d.n, len(S)))
            for a in range(0, d.n, CHUNK):
                r = np.abs(S[None] - d.probs[a:a + CHUNK, cols][:, None, :])       # (b, 2^m, m)
                out[a:a + CHUNK] = np.sqrt(np.maximum(np.sum((r @ sigma_inv) * r, axis=-1), 0.0)) + penalty
            return out
        return score
