"""Diagnostic only (not an experiment result): Exp055's own train_scorer on the Exp056 C1 inputs."""
import importlib.util, sys
import numpy as np, torch
REPO = "/Users/alonfines/Desktop/x-ray"
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("e55", REPO + "/experiments/055_learned_score_full/run.py")
e55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e55)
import conformal_common as cc
torch.set_num_threads(1)
STEM = "densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809"
splits = cc.load_splits(stem=STEM, with_validation=True, with_train=False, run_gate0=True)
val = e55.all13_clean(splits.validation)
conf13 = e55.all13_clean(splits.calibration)
test13 = e55.all13_clean(splits.evaluation)
subjects = np.unique(val["subject_id"].to_numpy())
perm = np.random.default_rng(0).permutation(subjects)
hold = perm[:int(round(0.2 * subjects.size))]
is_hold = val["subject_id"].isin(set(hold.tolist())).to_numpy()
val_tr = val[~is_hold].reset_index(drop=True); val_ho = val[is_hold].reset_index(drop=True)
L = ["Enlarged Cardiomediastinum", "Cardiomegaly"]
w = e55.spec_weights(2)
g_tr = cc.extract_group(val_tr, L); g_ho = cc.extract_group(val_ho, L)
ytr = torch.tensor((g_tr.y * w).sum(axis=1), dtype=torch.long)
yho = torch.tensor((g_ho.y * w).sum(axis=1), dtype=torch.long)
Xtr = torch.tensor(e55.logit(g_tr.probs), dtype=torch.float32)
Xho = torch.tensor(e55.logit(g_ho.probs), dtype=torch.float32)
net, info = e55.train_scorer(Xtr, ytr, Xho, yho, 4, 0, hidden=True)
print("Exp055 train_scorer info:", info)
# frozen-draw threshold on conformal (A) using the same cc-order mapping as Exp055
gc = cc.extract_group(conf13, L)
states = cc.enumerate_states(2); spec_of_cc = (states * w).sum(axis=1)
raw_spec = e55.neg_log_softmax_spec(net, torch.tensor(e55.logit(gc.probs), dtype=torch.float32))
s_true = raw_spec[np.arange(gc.n), (gc.y * w).sum(axis=1)]
k = cc.conformal_rank(gc.n, 0.10)
print("threshold on conformal (k=%d):" % k, cc.order_statistic(s_true, k))
