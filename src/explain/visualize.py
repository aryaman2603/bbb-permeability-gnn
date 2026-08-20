
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import Draw


def pyg_edges_to_bond_scores(data, edge_scores):
    """
    Convert PyG directed-edge importance scores
    into one score per chemical bond.
    """

    bond_scores = defaultdict(list)

    for i in range(data.edge_index.shape[1]):

        src = int(data.edge_index[0, i].item())
        dst = int(data.edge_index[1, i].item())

        bond = tuple(sorted((src, dst)))

        bond_scores[bond].append(
            float(edge_scores[i])
        )

    final_scores = {}

    for bond, scores in bond_scores.items():
        final_scores[bond] = float(max(scores))

    return final_scores


def get_top_atoms(node_scores, fraction=0.20):
    """
    Get the most important atoms.
    """

    num_atoms = len(node_scores)

    num_top = max(
        1,
        int(num_atoms * fraction)
    )

    ranked = node_scores.argsort()[::-1]

    # IMPORTANT:
    # Convert NumPy integers to normal Python integers
    return [int(x) for x in ranked[:num_top]]


def get_top_bonds(bond_scores, fraction=0.20):
    """
    Get the most important chemical bonds.
    """

    sorted_bonds = sorted(
        bond_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    num_top = max(
        1,
        int(len(sorted_bonds) * fraction)
    )

    return [
        (tuple(int(x) for x in bond), float(score))
        for bond, score in sorted_bonds[:num_top]
    ]


def rdkit_bond_indices(mol, important_bonds):
    """
    Convert atom-pair bonds into RDKit bond indices.
    """

    important_pairs = {
        tuple(int(x) for x in bond)
        for bond, _ in important_bonds
    }

    indices = []

    for bond in mol.GetBonds():

        pair = tuple(sorted((
            int(bond.GetBeginAtomIdx()),
            int(bond.GetEndAtomIdx()),
        )))

        if pair in important_pairs:
            indices.append(int(bond.GetIdx()))

    return indices


def draw_explanation(
    smiles,
    node_scores,
    bond_scores,
    fraction=0.20,
    size=(900, 700),
):
    """
    Draw the molecule with important
    atoms and bonds highlighted.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    top_atoms = get_top_atoms(
        node_scores,
        fraction,
    )

    top_bonds = get_top_bonds(
        bond_scores,
        fraction,
    )

    bond_indices = rdkit_bond_indices(
        mol,
        top_bonds,
    )

    # Extra safety for RDKit
    top_atoms = [int(x) for x in top_atoms]
    bond_indices = [int(x) for x in bond_indices]

    image = Draw.MolToImage(
        mol,
        size=size,
        highlightAtoms=top_atoms,
        highlightBonds=bond_indices,
    )

    return image, top_atoms, top_bonds
