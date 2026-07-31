"""RDKit-based atom/bond featurization for BBBP molecules.

Builds explicit, documented feature vectors — independent of PyG's
default MoleculeNet featurizer.

Vocabularies below are data-driven: derived from frequency analysis
of the actual BBBP dataset (see analyze_features.py), not generic
defaults. Rare categories (<0.1% occurrence) are folded into an
"other" bucket rather than given a dedicated one-hot slot.
"""

from rdkit import Chem
import torch

# --- Vocabularies for one-hot encodings (data-driven, see analyze_features.py) ---
ATOM_LIST = ['C', 'N', 'O', 'Cl', 'S', 'F', 'Br']  # + "other" bucket (covers 99.84% of atoms)
DEGREE_LIST = [0, 1, 2, 3, 4]  # all meaningfully represented, no "other" needed in practice
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]  # + "other" bucket (SP3D/SP3D2 have zero occurrences in BBBP)

BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _one_hot(value, choices):
    """One-hot encode value against choices, with a trailing 'other' bucket."""
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def atom_features(atom: Chem.Atom) -> list[float]:
    features = []
    features += _one_hot(atom.GetSymbol(), ATOM_LIST)                  # 8
    features += _one_hot(atom.GetDegree(), DEGREE_LIST)                # 6
    features.append(float(atom.GetFormalCharge()))                     # 1
    features += _one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST)  # 4
    features.append(float(atom.GetIsAromatic()))                       # 1
    features.append(float(atom.IsInRing()))                            # 1
    features.append(float(atom.GetTotalNumHs()))                       # 1
    return features  # total length: 22... see note below


def bond_features(bond: Chem.Bond) -> list[float]:
    features = []
    features += _one_hot(bond.GetBondType(), BOND_TYPE_LIST)  # 5
    features.append(float(bond.GetIsConjugated()))            # 1
    features.append(float(bond.IsInRing()))                   # 1
    return features  # total length: 7


def get_feature_dims() -> tuple[int, int]:
    """Return (node_feature_dim, edge_feature_dim) for model init."""
    node_dim = (
        (len(ATOM_LIST) + 1)            # atom symbol one-hot + other
        + (len(DEGREE_LIST) + 1)        # degree one-hot + other
        + 1                              # formal charge
        + (len(HYBRIDIZATION_LIST) + 1) # hybridization one-hot + other
        + 1                              # aromatic
        + 1                              # ring membership
        + 1                              # total numH
    )
    edge_dim = (
        (len(BOND_TYPE_LIST) + 1)       # bond type one-hot + other
        + 1                              # conjugated
        + 1                              # ring membership
    )
    return node_dim, edge_dim


if __name__ == "__main__":
    # quick sanity check on a sample molecule
    smiles = "CC(C)NCC(O)COc1cccc2ccccc12"
    mol = Chem.MolFromSmiles(smiles)
    print(f"SMILES: {smiles}")
    print(f"Num atoms: {mol.GetNumAtoms()}, Num bonds: {mol.GetNumBonds()}")

    atom0 = mol.GetAtomWithIdx(0)
    feats = atom_features(atom0)
    print(f"\nAtom 0 ({atom0.GetSymbol()}) features (len={len(feats)}):")
    print(feats)

    bond0 = mol.GetBondWithIdx(0)
    bfeats = bond_features(bond0)
    print(f"\nBond 0 features (len={len(bfeats)}):")
    print(bfeats)

    print(f"\nget_feature_dims(): {get_feature_dims()}")