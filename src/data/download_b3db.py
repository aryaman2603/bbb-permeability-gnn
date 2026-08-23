"""Download and inspect the B3DB dataset (external validation set).
Unlike BBBP, B3DB is much larger (~7,800 categorical labels) and uses
different column names — inspect before assuming structure.
"""

from pathlib import Path
from B3DB import B3DB_DATA_DICT

RAW_DIR = Path("data/raw/B3DB")


def load_and_inspect():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Available B3DB subsets:", list(B3DB_DATA_DICT.keys()))

    df_cls = B3DB_DATA_DICT["B3DB_classification"]
    print(f"\nClassification set shape: {df_cls.shape}")
    print(f"Columns: {list(df_cls.columns)}")
    print(f"\nFirst few rows:\n{df_cls.head()}")

    # save raw CSV for our own pipeline to consume later
    out_path = RAW_DIR / "B3DB_classification.csv"
    df_cls.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    return df_cls


if __name__ == "__main__":
    load_and_inspect()