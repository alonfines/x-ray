"""Joint conformal prediction over label clusters, on the frozen unified-model CSVs.

The pipeline, one module per step:

    data       load the frozen splits, Gate 0, all-13-clean population, calibration/test resplits
    partition  which labels are calibrated together (singletons, given clusters, random, DP search)
    scores     a nonconformity score over each cluster's 2^m joint states (per_label, lp_scp, mlp)
    quantile   split-conformal quantiles, and the rule that splits alpha across clusters
               (none / bonferroni / sidak / max_rank)
    regions    per-cluster regions; the joint set is their product
    evaluate   joint coverage, set size, decisiveness; summaries and paired CIs over resplits
    run        one call: (partition, score, combine rule, alpha) -> per-resplit metrics

Classical per-label conformal is the special case "every label its own cluster, per_label score".
"""
