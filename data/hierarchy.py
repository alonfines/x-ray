"""Disease hierarchy for CheXpert/MIMIC-CXR pathology labels.

Based on Pham et al. (2020) "Interpreting chest X-rays via CNNs that exploit
hierarchical disease dependencies and uncertainty labels."

The hierarchy encodes parent-child relationships among the 14 CheXpert labels.
Root-level labels have no parents. Child labels are conditioned on their parents
being positive (e.g., Pneumonia requires Consolidation and Lung Opacity to be positive).
"""

import pandas as pd

# Maps each label to its ordered list of parents (root → child direction).
# Labels with no parents have an empty list.
CHEXPERT_HIERARCHY = {
    "Enlarged Cardiomediastinum": [],
    "Cardiomegaly": ["Enlarged Cardiomediastinum"],
    "Lung Opacity": [],
    "Lung Lesion": ["Lung Opacity"],
    "Edema": ["Lung Opacity"],
    "Consolidation": ["Lung Opacity"],
    "Pneumonia": ["Lung Opacity", "Consolidation"],
    "Atelectasis": ["Lung Opacity"],
    "Pneumothorax": [],
    "Pleural Effusion": [],
    "Pleural Other": [],
    "Fracture": [],
    "Support Devices": [],
}


def apply_hierarchy_correction_to_df(df: pd.DataFrame) -> None:
    """Propagate child-positive -> parent-positive labels in a DataFrame (in-place).

    When a child label is definitively positive (1.0) but its parent is NaN
    (not mentioned) or 0.0, the parent is set to 1.0.  This corrects the
    labeling artifact where radiologists annotate a specific finding without
    explicitly noting its logically implied parent.

    Only definite positives (1.0) trigger propagation — uncertain (-1.0)
    and NaN values do not.
    """
    for child, parents in CHEXPERT_HIERARCHY.items():
        if child not in df.columns or not parents:
            continue
        child_positive = df[child] == 1.0
        for parent in parents:
            if parent in df.columns:
                mask = child_positive & (df[parent].isna() | (df[parent] == 0.0))
                df.loc[mask, parent] = 1.0


