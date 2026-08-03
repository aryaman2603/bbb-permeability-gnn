"""Morgan circular fingerprint extractor for BBBP molecules.

Each molecule (given as a SMILES string) is converted to a 2048-bit
Morgan fingerprint at radius 2 (equivalent to ECFP4).

These fingerprints serve as the feature matrix X for the Random Forest
baseline — no graph representation needed.

Background
----------
Morgan/ECFP fingerprints work by iteratively hashing each atom's local
environment out to `radius` bonds. At radius 2, each bit represents a
unique substructure that fits within a 2-bond radius around some atom.
The result is a fixed-length bit vector that encodes what chemical
environments are *present* in the molecule, but not how they connect
at a global level (that's why GNNs can theoretically do better).
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import torch

RDLogger.DisableLog("rdApp.*")

# Hyperparameters — frozen for reproducibility
RADIUS = 2      # ECFP4: radius-2 captures up to 2 bonds from each atom
N_BITS = 2048   # length of bit vector (standard choice for drug-like molecules)


def smiles_to_morgan(smiles: str, radius: int = RADIUS, n_bits: int = N_BITS) -> np.ndarray | None:
    """Convert a SMILES string to a Morgan fingerprint bit vector.

    Args:
        smiles:  SMILES representation of the molecule.
        radius:  Morgan algorithm radius (2 = ECFP4).
        n_bits:  Length of the output fingerprint bit vector.

    Returns:
        np.ndarray of shape (n_bits,) with dtype float32, or None if SMILES
        fails to parse.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def build_fingerprint_dataset(data_list) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) numpy arrays from a list of PyG Data objects.

    Each Data object already has .smiles and .y attributes from Person A's
    graph-building pipeline. We re-extract fingerprints from SMILES here —
    this path is entirely independent of the PyG node features.

    Args:
        data_list: list of PyG Data objects (loaded from .pt files).

    Returns:
        X: np.ndarray of shape (N, n_bits), float32 — fingerprint matrix.
        y: np.ndarray of shape (N,), int — binary labels (0 or 1).
    """
    X_list, y_list = [], []
    skipped = 0

    for data in data_list:
        fp = smiles_to_morgan(data.smiles)
        if fp is None:
            skipped += 1
            continue
        X_list.append(fp)
        y_list.append(int(data.y.item()))

    if skipped:
        print(f"  Warning: {skipped} molecule(s) skipped — SMILES failed to parse")

    X = np.stack(X_list, axis=0)    # shape: (N, 2048)
    y = np.array(y_list, dtype=int)  # shape: (N,)
    return X, y


if __name__ == "__main__":
    print("Loading frozen dataset...")
    train_data = torch.load("data/processed/train.pt", weights_only=False)
    val_data   = torch.load("data/processed/val.pt",   weights_only=False)
    test_data  = torch.load("data/processed/test.pt",  weights_only=False)

    print("Building Morgan fingerprints (radius=2, 2048 bits)...")
    X_train, y_train = build_fingerprint_dataset(train_data)
    X_val,   y_val   = build_fingerprint_dataset(val_data)
    X_test,  y_test  = build_fingerprint_dataset(test_data)

    print(f"\nX_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"X_val  : {X_val.shape}     y_val  : {y_val.shape}")
    print(f"X_test : {X_test.shape}    y_test : {y_test.shape}")
    print(f"\nLabel balance — train: {y_train.mean():.1%} BBB+")
    print(f"Label balance — val  : {y_val.mean():.1%} BBB+")
    print(f"Label balance — test : {y_test.mean():.1%} BBB+")
    print(f"\nSample fingerprint (first 20 bits): {X_train[0, :20]}")
    print(f"Non-zero bits in sample: {X_train[0].sum():.0f} / {N_BITS}")
