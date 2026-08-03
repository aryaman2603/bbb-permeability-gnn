import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import torch

RDLogger.DisableLog("rdApp.*")

RADIUS = 2      
N_BITS = 2048   


def smiles_to_morgan(smiles: str, radius: int = RADIUS, n_bits: int = N_BITS) -> np.ndarray | None:
  
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def build_fingerprint_dataset(data_list) -> tuple[np.ndarray, np.ndarray]:
  
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

    X = np.stack(X_list, axis=0)    
    y = np.array(y_list, dtype=int)  
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
