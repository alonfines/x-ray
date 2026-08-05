"""Shared utilities for dataset splitting (stratification, statistics)."""

import pandas as pd


def get_stratification_target(df, pathologies):
    """Create a Mondrian-style stratification target for conformal prediction.

    Stratifies by presence/absence of top pathologies to ensure balanced class distribution.
    Returns None if stratification is not feasible.
    """
    labels = df[pathologies].fillna(0).astype(int)

    top_n = min(3, len(pathologies))
    prevalence = labels.sum(axis=0)
    top_pathologies_idx = prevalence.nlargest(top_n).index.tolist()

    strat_groups = []
    for idx in range(len(df)):
        group_str = ""
        for path_idx in top_pathologies_idx:
            group_str += "1" if labels.iloc[idx][path_idx] > 0 else "0"
        strat_groups.append(group_str)

    strat_target = pd.Series(strat_groups, index=df.index)

    value_counts = strat_target.value_counts()
    min_samples = value_counts.min()

    # If any group has only 1 sample, fall back to coarser stratification
    if min_samples < 2 and len(value_counts) > 1:
        most_prevalent_idx = prevalence.idxmax()
        strat_target = (labels[most_prevalent_idx] > 0).astype(str)

    if strat_target.value_counts().min() < 2:
        return None

    return strat_target


def print_split_statistics(splits, pathologies):
    """Print label distribution, conformal readiness check, and final statistics.

    Args:
        splits: list of (name, dataframe) tuples
        pathologies: list of pathology column names
    """
    # Label distribution
    print(f"\nLabel distribution across splits:")
    for name, df in splits:
        if len(df) > 0:
            print(f"\n{name}:")
            for pathology in pathologies:
                if pathology in df.columns:
                    positive = (df[pathology] == 1.0).sum()
                    uncertain = (df[pathology] == -1.0).sum()
                    negative = (df[pathology] == 0.0).sum()
                    print(f"  {pathology:<30} {positive:>5} positive, {uncertain:>5} uncertain, {negative:>5} negative")

    # Conformal readiness check
    conformal_splits = [(name, df) for name, df in splits if 'conformal' in name.lower()]
    if conformal_splits:
        conformal_name, conformal_df = conformal_splits[0]
        print(f"\n" + "=" * 70)
        print("CONFORMAL CALIBRATION READINESS CHECK")
        print("=" * 70)
        print(f"\nFor per-class conformal prediction, recommend ~100+ positive samples per pathology.")
        print(f"Your conformal set has {len(conformal_df)} total samples.\n")

        min_positives = float('inf')
        for pathology in pathologies:
            if pathology in conformal_df.columns:
                positives = (conformal_df[pathology] == 1.0).sum()
                min_positives = min(min_positives, positives)
                status = "OK" if positives >= 100 else "LOW" if positives >= 50 else "TOO FEW"
                print(f"  {pathology:<30} {positives:>4} positives {status}")

        if min_positives < 50:
            print(f"\nWARNING: Conformal set has insufficient samples for stable per-class quantiles.")
        elif min_positives < 100:
            print(f"\nNOTE: Some classes are below 100 positives. Results may be borderline.")
        else:
            print(f"\nConformal set appears adequate for per-class calibration.")

    # Final statistics
    print(f"\n" + "=" * 70)
    print("FINAL SPLIT STATISTICS")
    print("=" * 70)
    total = 0
    for name, df in splits:
        print(f"{name + ' set:':<20} {len(df):>6} samples")
        total += len(df)
    print(f"{'-'*70}")
    print(f"{'Total:':<20} {total:>6} samples")
    print("=" * 70)

    # Calculate pos_weights from training set
    train_splits = [(name, df) for name, df in splits if 'train' in name.lower()]
    if train_splits:
        train_name, train_df = train_splits[0]
        print(f"\n" + "=" * 70)
        print("CALCULATED POS_WEIGHTS FOR config.yaml")
        print("=" * 70)
        print(f"\npos_weight = num_negatives / num_positives per class\n")

        pos_weights = []
        for pathology in pathologies:
            if pathology in train_df.columns:
                num_positives = (train_df[pathology] == 1.0).sum()
                num_negatives = (train_df[pathology] == 0.0).sum()
                pos_weight = num_negatives / (num_positives + 1e-8)
                prevalence = 100 * num_positives / len(train_df)
                pos_weights.append(pos_weight)
                print(f"  {pathology:<30} pos_weight: {pos_weight:.3f}  ({num_positives:>5} pos, {prevalence:>5.1f}% prevalence)")

        if pos_weights:
            max_weight = max(pos_weights)
            normalized_weights = [w / max_weight for w in pos_weights]
            print(f"\nNormalized pos_weights (rarest=1.0):")
            print(f"task_weights:")
            for pathology, weight in zip(pathologies, normalized_weights):
                print(f"  - {weight:.3f} # {pathology}")
            print("=" * 70)
