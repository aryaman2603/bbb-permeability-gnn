

from collections import Counter
import pandas as pd
from rdkit import Chem

CSV_PATH = "data/raw/BBBP/raw/BBBP.csv"  


def analyze():
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows in CSV: {len(df)}")

    atom_counter = Counter()
    degree_counter = Counter()
    hybridization_counter = Counter()
    bond_type_counter = Counter()
    formal_charge_counter = Counter()
    numH_counter = Counter()

    failed = 0
    for smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed += 1
            continue
        for atom in mol.GetAtoms():
            atom_counter[atom.GetSymbol()] += 1
            degree_counter[atom.GetDegree()] += 1
            hybridization_counter[str(atom.GetHybridization())] += 1
            formal_charge_counter[atom.GetFormalCharge()] += 1
            numH_counter[atom.GetTotalNumHs()] += 1
        for bond in mol.GetBonds():
            bond_type_counter[str(bond.GetBondType())] += 1

    print(f"Failed to parse: {failed} SMILES")

    def report(name, counter):
        total = sum(counter.values())
        print(f"\n{name} (n={total}):")
        for k, v in counter.most_common():
            print(f"  {k}: {v} ({v/total:.2%})")

    report("Atom symbols", atom_counter)
    report("Degree", degree_counter)
    report("Hybridization", hybridization_counter)
    report("Formal charge", formal_charge_counter)
    report("Total numH", numH_counter)
    report("Bond types", bond_type_counter)


if __name__ == "__main__":
    analyze()