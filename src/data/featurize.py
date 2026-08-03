

from rdkit import Chem
import torch


ATOM_LIST = ['C', 'N', 'O', 'Cl', 'S', 'F', 'Br']  
DEGREE_LIST = [0, 1, 2, 3, 4]  
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
] 

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
    return features  


def bond_features(bond: Chem.Bond) -> list[float]:
    features = []
    features += _one_hot(bond.GetBondType(), BOND_TYPE_LIST)  # 5
    features.append(float(bond.GetIsConjugated()))            # 1
    features.append(float(bond.IsInRing()))                   # 1
    return features  # total length: 7


def get_feature_dims() -> tuple[int, int]:
    """Return (node_feature_dim, edge_feature_dim) for model init."""
    node_dim = (
        (len(ATOM_LIST) + 1)            
        + (len(DEGREE_LIST) + 1)        
        + 1                              
        + (len(HYBRIDIZATION_LIST) + 1) 
        + 1                              
        + 1                             
        + 1                            
    )
    edge_dim = (
        (len(BOND_TYPE_LIST) + 1)      
        + 1                             
        + 1                             
    )
    return node_dim, edge_dim


if __name__ == "__main__":

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