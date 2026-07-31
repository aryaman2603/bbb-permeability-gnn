"""Scaffold-based train/val/test split (70/15/15, seed=42).
Groups molecules by Murcko scaffold so no scaffold appears in more than one split.
"""

import random
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold


def scaffold_split(data_list, seed: int = SEED):
    """Returns (train_idx, val_idx, test_idx) — indices into data_list."""
    scaffold_to_indices = defaultdict(list)
    for idx, data in enumerate(data_list):
        scaffold = get_scaffold(data.smiles)
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffold groups by size, descending (largest groups placed first)
    scaffold_groups = list(scaffold_to_indices.values())
    random.Random(seed).shuffle(scaffold_groups)  # shuffle ties for reproducibility
    scaffold_groups.sort(key=len, reverse=True)

    n_total = len(data_list)
    n_train_cutoff = int(TRAIN_FRAC * n_total)
    n_val_cutoff = int((TRAIN_FRAC + VAL_FRAC) * n_total)

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(train_idx) + len(group) <= n_train_cutoff:
            train_idx += group
        elif len(train_idx) + len(val_idx) + len(group) <= n_val_cutoff:
            val_idx += group
        else:
            test_idx += group

    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    from src.data.graph_builder import build_all

    data_list, _ = build_all()
    train_idx, val_idx, test_idx = scaffold_split(data_list)

    print(f"Train: {len(train_idx)} ({len(train_idx)/len(data_list):.1%})")
    print(f"Val:   {len(val_idx)} ({len(val_idx)/len(data_list):.1%})")
    print(f"Test:  {len(test_idx)} ({len(test_idx)/len(data_list):.1%})")

    # sanity check: label balance per split
    for name, idxs in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        labels = [data_list[i].y.item() for i in idxs]
        pos = sum(labels)
        print(f"{name} label balance: {pos}/{len(labels)} positive ({pos/len(labels):.1%})")