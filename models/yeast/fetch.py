"""Download the multi-label Yeast dataset (OpenML 40597) to data/yeast/raw/yeast.csv.

2,417 rows: `id`, 103 numeric features Att1..Att103, 14 binary labels label_1..label_14.
No hierarchy. Run from the repo root: python -m models.yeast.fetch
"""

import os

import pandas as pd
from sklearn.datasets import fetch_openml

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "yeast", "raw", "yeast.csv")


def main() -> None:
    # parser="liac-arff": the default pandas parser fails on this file's TRUE/FALSE categories
    d = fetch_openml(data_id=40597, as_frame=True, parser="liac-arff")
    X = d.data.astype(float)
    Y = (d.target == "TRUE").astype(int)
    Y.columns = [f"label_{i}" for i in range(1, Y.shape[1] + 1)]
    df = pd.concat([X, Y], axis=1)
    df.insert(0, "id", range(len(df)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {df.shape} to {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
