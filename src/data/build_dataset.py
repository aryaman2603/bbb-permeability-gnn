

from pathlib import Path
import torch

from src.data.graph_builder import build_all
from src.data.split import scaffold_split

PROCESSED_DIR = Path("data/processed")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data_list, failed = build_all()
    train_idx, val_idx, test_idx = scaffold_split(data_list)

    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]

    torch.save(train_data, PROCESSED_DIR / "train.pt")
    torch.save(val_data, PROCESSED_DIR / "val.pt")
    torch.save(test_data, PROCESSED_DIR / "test.pt")

    print(f"Saved: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    print(f"Failed to parse (excluded): {len(failed)}")
    print(f"Written to {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    main()