"""Learned joint-state score: an MLP over per-label logits (extracted from Exp055).

A scorer is trained to predict the joint label state y in {0,1}^m (one of S = 2^m classes)
from the model's per-label logits. The nonconformity score of a state is its negative
log-softmax, s(x, y) = -log p(y | x); a joint region is then built with the usual
split-conformal threshold (see `conformal_common.conformal_rank` / `order_statistic`).

    L0  Linear(d, S)                       multinomial logistic regression
    L1  Linear(d, 64)-ReLU-Linear(64, S)   MLP on the cluster's own m logits
    L2  same MLP, input = all 13 logits

Ported to PyTorch Lightning from experiments/055_learned_score_full/run.py (which is left
unmodified, as pre-registered). Hyperparameters are the pre-registered ones, but the training
loop is now a Lightning Trainer, so minibatch RNG draws differ: results are statistically
equivalent, not bit-identical, to Exp055's. Do not use this to reproduce Exp055 numbers.

Importable
----------
    from conformal_methods.mlp_score import logit, spec_weights, train_scorer, neg_log_softmax_spec
"""

from __future__ import annotations

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

HIDDEN = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 256
PATIENCE = 10
MAX_EPOCHS = 200
PROB_CLIP = 1e-6


class ScorerModule(pl.LightningModule):
    """L1/L2 (hidden=True): Linear(d,64)-ReLU-Linear(64,S). L0 (hidden=False): Linear(d,S)."""

    def __init__(self, d_in: int, S: int, hidden: bool):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(d_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, S))
                    if hidden else nn.Sequential(nn.Linear(d_in, S)))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.ce(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch   # the whole holdout is one batch, so val_ce is the exact holdout CE
        self.log("val_ce", self.ce(self(x), y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


class _KeepBest(Callback):
    """Remember the weights (in memory) and epoch with the lowest val_ce."""

    def __init__(self):
        self.best, self.best_epoch, self.best_state = float("inf"), 0, None

    def on_validation_epoch_end(self, trainer, pl_module):
        val = trainer.callback_metrics["val_ce"].item()
        if val < self.best:
            self.best, self.best_epoch = val, trainer.current_epoch + 1
            self.best_state = {k: v.clone() for k, v in pl_module.state_dict().items()}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, PROB_CLIP, 1.0 - PROB_CLIP)
    return np.log(p / (1.0 - p))


def spec_weights(m: int) -> np.ndarray:
    """Target index is sum_l y_l * 2^l, labels in yaml order (label 0 = LSB)."""
    return (1 << np.arange(m)).astype(np.int64)


def train_scorer(X_tr, y_tr, X_ho, y_ho, S, seed, hidden):
    """Train one scorer with a Lightning Trainer. Returns (module, info).

    Adam, lr 1e-3, weight_decay 1e-4, batch 256; early stopping on inner-holdout CE with
    patience 10, max 200 epochs, best weights kept. `seed` seeds init and the minibatch order.
    """
    pl.seed_everything(seed, workers=True)
    module = ScorerModule(X_tr.shape[1], S, hidden)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    holdout_loader = DataLoader(TensorDataset(X_ho, y_ho), batch_size=len(X_ho))

    stopper = EarlyStopping(monitor="val_ce", patience=PATIENCE, mode="min")
    keeper = _KeepBest()
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, callbacks=[stopper, keeper], accelerator="cpu", devices=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0)
    trainer.fit(module, train_loader, holdout_loader)

    module.load_state_dict(keeper.best_state)
    module.eval()
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        final_tr = ce(module(X_tr), y_tr).item()
        final_ho = ce(module(X_ho), y_ho).item()
    stopped_early = stopper.stopped_epoch > 0
    epochs_run = trainer.current_epoch
    info = {"best_epoch": keeper.best_epoch, "stop_epoch": epochs_run if stopped_early else MAX_EPOCHS,
            "hit_cap": not stopped_early, "epochs_run": epochs_run,
            "final_train_ce": final_tr, "final_holdout_ce": final_ho}
    return module, info


def neg_log_softmax_spec(net, X) -> np.ndarray:
    """s(x, y) = -log_softmax(logits)[y] for every joint state, (n, 2^m), spec index order."""
    with torch.no_grad():
        lpr = torch.log_softmax(net(X).double(), dim=1).numpy()
    return -lpr
