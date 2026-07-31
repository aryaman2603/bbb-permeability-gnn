"""SMILES -> PyG Data objects, using RDKit featurization.
Skips and logs any SMILES that fail to parse.
"""

from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit import RDLogger

from src.data.featurize import atom_features, bond_features

RDLogger.DisableLog("rdApp.*")  # suppress RDKit parse warnings; we log failures ourselves

CSV_PATH = Path("data/raw/bbbp/raw/BBBP.csv")  


def smiles_to_data(smiles: str, label: float) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    xs = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(xs, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)


def build_all(csv_path: Path = CSV_PATH) -> tuple[list[Data], list[str]]:
    df = pd.read_csv(csv_path)
    data_list = []
    failed_smiles = []

    for _, row in df.iterrows():
        smiles = row["smiles"]
        label = float(row["p_np"])  # BBBP label column
        data = smiles_to_data(smiles, label)
        if data is None:
            failed_smiles.append(smiles)
        else:
            data_list.append(data)

    print(f"Built {len(data_list)} graphs, {len(failed_smiles)} SMILES failed to parse")
    return data_list, failed_smiles


if __name__ == "__main__":
    data_list, failed = build_all()
    print(f"\nSample graph: {data_list[0]}")
    if failed:
        print(f"\nFailed SMILES ({len(failed)}):")
        for s in failed:
            print(f"  {s}")