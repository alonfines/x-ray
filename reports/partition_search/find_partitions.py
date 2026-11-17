"""Find the best label partition for each m_max in {3,4,5,6}. No conformal step, no test data.

Method (Tom's ruling, 2026-11-17):
  1. For every candidate cluster (every label subset of size <= 6), train the L1 joint scorer
     (Linear(m,64)-ReLU-Linear(64,2^m) on the cluster's own logits) on the validation split's
     80% inner-train subjects, and record its cross-entropy on the 20% inner-holdout subjects.
     Softmax is renormalised over the cluster's hierarchy-admissible states.
  2. A partition's cost = sum of its clusters' holdout CE (= factorised joint log-loss).
  3. Exact minimum-cost partition with cluster size <= m_max, by DP over all 2^13 label subsets.

Uses validation only (all-13-clean rows). Conformal and test splits are not read.
Run: /usr/bin/python3 reports/partition_search/find_partitions.py
"""
import itertools, json, os, sys, time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
OUT = os.path.dirname(os.path.abspath(__file__))
M_MAX_GRID = [3, 4, 5, 6]
SEED = 0

# L1 hyperparameters (conformal_methods/mlp_score.py)
HIDDEN, LR, WD, BATCH, PATIENCE, MAX_EPOCHS, CLIP = 64, 1e-3, 1e-4, 256, 10, 200, 1e-6


def load():
    import conformal_common as cc
    splits = cc.load_splits(with_validation=True, with_train=False, run_gate0=True)
    val = splits.validation
    labels = list(cc.ALL_LABELS)
    y = val[[f"{l}_true_label" for l in labels]].to_numpy(float)
    p = val[[f"{l}_prob" for l in labels]].to_numpy(float)
    keep = np.all(np.isin(y, [0.0, 1.0]), 1) & np.all(np.isfinite(p), 1)
    val = val.loc[keep].reset_index(drop=True)
    y, p = y[keep].astype(int), p[keep]
    p = np.clip(p, CLIP, 1 - CLIP)
    x = np.log(p / (1 - p))
    subj = np.unique(val.subject_id.to_numpy())
    perm = np.random.default_rng(0).permutation(subj)          # Exp055/056 inner holdout
    hold = set(perm[: int(round(0.2 * subj.size))].tolist())
    is_ho = val.subject_id.isin(hold).to_numpy()
    return labels, x, y, is_ho


def init(data):
    global D
    import torch
    torch.set_num_threads(1)
    D = data


def fit_cluster(cols, return_net=False):
    """Default: (cols, best_holdout_ce, best_epoch). return_net=True additionally returns
    (stop_epoch, net) with the best-epoch weights restored. Training is identical either way."""
    import copy, torch, torch.nn as nn
    import conformal_common as cc
    labels, x, y, is_ho = D
    cols = list(cols)
    m, S = len(cols), 2 ** len(cols)
    states = cc.enumerate_states(m)
    adm = torch.tensor(cc.hierarchy_admissible([labels[c] for c in cols], states))
    yi = torch.tensor(cc.state_indices(y[:, cols]))
    X = torch.tensor(x[:, cols], dtype=torch.float32)
    tr, ho = torch.tensor(~is_ho), torch.tensor(is_ho)
    Xtr, ytr, Xho, yho = X[tr], yi[tr], X[ho], yi[ho]

    torch.manual_seed(SEED)
    net = nn.Sequential(nn.Linear(m, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, S))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    neg = torch.where(adm, 0.0, float("-inf"))                 # renormalise over admissible states
    ce = lambda Xb, yb: nn.functional.cross_entropy(net(Xb) + neg, yb)
    g = torch.Generator().manual_seed(SEED)
    best, best_ep, bad, best_state = float("inf"), 0, 0, None
    for ep in range(1, MAX_EPOCHS + 1):
        for b in torch.randperm(len(Xtr), generator=g).split(BATCH):
            opt.zero_grad(); ce(Xtr[b], ytr[b]).backward(); opt.step()
        with torch.no_grad():
            v = ce(Xho, yho).item()
        if v < best - 1e-12:
            best, best_ep, bad = v, ep, 0
            if return_net:
                best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if not return_net:
        return tuple(cols), best, best_ep
    net.load_state_dict(best_state)
    net.eval()
    return tuple(cols), best, best_ep, ep, net


def best_partition(cost, n, m_max):
    """Exact DP: f[mask] = min over clusters c containing mask's lowest label of cost[c] + f[mask\\c]."""
    full = (1 << n) - 1
    f = np.full(1 << n, np.inf); f[0] = 0.0
    arg = [None] * (1 << n)
    for mask in range(1, full + 1):
        low = mask & -mask
        sub = mask
        while sub:
            if sub & low and sub in cost and bin(sub).count("1") <= m_max:
                v = cost[sub] + f[mask ^ sub]
                if v < f[mask]:
                    f[mask], arg[mask] = v, sub
            sub = (sub - 1) & mask
    parts, mask = [], full
    while mask:
        parts.append(arg[mask]); mask ^= arg[mask]
    return f[full], parts


def main():
    data = load()
    labels, x, y, is_ho = data
    n = len(labels)
    cands = [c for k in range(1, max(M_MAX_GRID) + 1) for c in itertools.combinations(range(n), k)]
    print(f"rows {len(y)} (train {(~is_ho).sum()}, holdout {is_ho.sum()}); {len(cands)} candidate clusters", flush=True)
    t0 = time.time()
    with Pool(os.cpu_count(), initializer=init, initargs=(data,)) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(fit_cluster, cands, chunksize=4)):
            res.append(r)
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(cands)}  {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "cluster_costs.csv"), "w") as fh:
        fh.write("cluster,size,holdout_ce,best_epoch\n")
        for cols, ce_, ep in sorted(res, key=lambda r: (len(r[0]), r[0])):
            fh.write(f"\"{' | '.join(labels[c] for c in cols)}\",{len(cols)},{ce_:.6f},{ep}\n")
    cost = {sum(1 << c for c in cols): ce_ for cols, ce_, _ in res}

    singles = sum(cost[1 << c] for c in range(n))
    out = {"independent_singletons_cost": singles, "partitions": {}}
    for mm in M_MAX_GRID:
        tot, parts = best_partition(cost, n, mm)
        clusters = [[labels[c] for c in range(n) if s >> c & 1] for s in parts]
        clusters.sort(key=len, reverse=True)
        out["partitions"][mm] = {"cost": tot, "gain_vs_singletons": singles - tot, "clusters": clusters}
        print(f"\nm_max={mm}  cost={tot:.4f}  (singletons {singles:.4f})")
        for c in clusters:
            print("   ", c)
    with open(os.path.join(OUT, "partitions.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
