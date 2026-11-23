"""Learned joint-state score: an MLP from the cluster's logits to its 2^m joint states.

    Linear(m, 64) - ReLU - Linear(64, 2^m),   s(x, y) = -log softmax(net(logit p(x)))[y]

Trained on the validation split: 20% of its subjects (default_rng(0), fixed across seeds) are the
inner holdout for early stopping. Adam lr 1e-3, weight decay 1e-4, batch 256, patience 10,
max 200 epochs, best-holdout weights kept. `seed` sets the initialisation and batch order.

No hierarchy information is given to the scorer: it learns which joint states are possible from
the data (Exp058: unmasked and masked MLPs give the same sets, and no set holds an impossible vector).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from ..data import Data, logits, state_index

HIDDEN, LR, WEIGHT_DECAY, BATCH, PATIENCE, MAX_EPOCHS, HOLDOUT = 64, 1e-3, 1e-4, 256, 10, 200, 0.2


def inner_holdout(subject_id: np.ndarray) -> np.ndarray:
    """Boolean mask of the inner-holdout rows (20% of subjects, default_rng(0))."""
    subj = np.random.default_rng(0).permutation(np.unique(subject_id))
    return np.isin(subject_id, subj[: int(round(HOLDOUT * subj.size))])


class MLP:
    def __init__(self):
        self.info: dict = {}            # training diagnostics of the last fit

    def fit(self, train: Data, labels: list[str], seed: int = 0):
        cols = [train.labels.index(l) for l in labels]
        m = len(cols)

        X = torch.tensor(logits(train.probs[:, cols]), dtype=torch.float32)
        y = torch.tensor(state_index(train.y[:, cols]))
        ho = torch.tensor(inner_holdout(train.group_id))
        Xtr, ytr, Xho, yho = X[~ho], y[~ho], X[ho], y[ho]

        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(m, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 2 ** m))
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss = lambda Xb, yb: nn.functional.cross_entropy(net(Xb), yb)
        gen = torch.Generator().manual_seed(seed)

        best, best_epoch, best_state, bad = float("inf"), 0, None, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            for b in torch.randperm(len(Xtr), generator=gen).split(BATCH):
                opt.zero_grad()
                loss(Xtr[b], ytr[b]).backward()
                opt.step()
            with torch.no_grad():
                ce = loss(Xho, yho).item()
            if ce < best - 1e-12:
                best, best_epoch, best_state, bad = ce, epoch, copy.deepcopy(net.state_dict()), 0
            elif (bad := bad + 1) >= PATIENCE:
                break
        net.load_state_dict(best_state)
        net.eval()
        self.info = {"holdout_ce": best, "best_epoch": best_epoch, "stop_epoch": epoch,
                     "hit_cap": bad < PATIENCE}

        def score(d: Data) -> np.ndarray:
            with torch.no_grad():
                x = torch.tensor(logits(d.probs[:, cols]), dtype=torch.float32)
                return -torch.log_softmax(net(x).double(), dim=1).numpy()
        return score
