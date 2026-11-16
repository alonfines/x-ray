# max-rank vs Bonferroni

- clusters: `configs/clusters/exp046.yaml` (pre-registered)
- alpha: 0.1 -> target joint coverage 0.90
- model: `densenet-alllabels-bce-hier-full-epoch=09-val_auroc_mean=0.809`
- calibration: conformal split; evaluation: test split; Gate 0 passed
- score: true-class nonconformity `s = 1-p if y=1 else p`
- rows with any uncertain (-1) member label excluded per cluster

| cluster | m | n | target | max-rank cov | Bonf. cov | max-rank \|set\| | Bonf. \|set\| | delta | ratio |
|---|---|---|---|---|---|---|---|---|---|
| cardio_pair | 2 | 23,128 | 0.90 | 0.8959 | 0.9421 | 2.260 | 2.623 | -0.363 | 0.862 |
| lung_core | 3 | 22,236 | 0.90 | 0.8922 | 0.9083 | 3.977 | 4.245 | -0.268 | 0.937 |
| pleural_free | 3 | 23,630 | 0.90 | 0.8982 | 0.9036 | 3.726 | 3.802 | -0.076 | 0.980 |
| lung_wide | 5 | 20,318 | 0.90 | 0.8993 | 0.9242 | 14.839 | 16.563 | -1.724 | 0.896 |

Joint set size = number of retained joint states (product of per-label set sizes), the only size metric comparable across method families. A negative delta means max-rank is the smaller set at equal or better coverage. No confidence intervals: the deltas are point estimates on one test split.
