

from pathlib import Path
from torch_geometric.datasets import MoleculeNet

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_bbbp():
    """Load the raw BBBP dataset (PyG auto-downloads on first call)."""
    dataset = MoleculeNet(root=str(RAW_DIR), name="BBBP")
    return dataset


def inspect(dataset):
    print(f"Number of molecules: {len(dataset)}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of edge features: {dataset.num_edge_features}")
    print(f"Number of classes: {dataset.num_classes}")

    sample = dataset[0]
    print("\nSample molecule:")
    print(f"  SMILES: {sample.smiles}")
    print(f"  Label (y): {sample.y}")
    print(f"  x shape: {sample.x.shape}")
    print(f"  edge_index shape: {sample.edge_index.shape}")

    labels = [int(d.y.item()) for d in dataset]
    pos = sum(labels)
    print(f"\nLabel distribution: {pos} positive / {len(labels) - pos} negative "
          f"({pos/len(labels):.1%} BBB+)")


if __name__ == "__main__":
    dataset = load_bbbp()
    inspect(dataset)