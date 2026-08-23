"""Run inference on B3DB (external dataset) using BBBP-trained GAT/GIN models.

Deduplicates against BBBP train/val/test first — B3DB is compiled from ~50
sources and likely overlaps with BBBP, so without this step we'd partly be
re-testing on training data rather than getting a genuine generalization check.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger

from src.data.graph_builder import smiles_to_data
from src.eval.metrics import evaluate_model, print_metrics, save_metrics
from src.explain.aggregate_explanations import load_model  # reuses config auto-discovery

RDLogger.DisableLog("rdApp.*")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B3DB_CSV = Path("data/raw/B3DB/B3DB_classification.csv")
RESULTS_DIR = Path("results/external/b3db")


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_bbbp_smiles_set() -> set[str]:
    """Canonical SMILES from all of our frozen BBBP train/val/test splits."""
    smiles_set = set()
    for split in ["train", "val", "test"]:
        data_list = torch.load(f"data/processed/{split}.pt", weights_only=False)
        for data in data_list:
            canon = canonical_smiles(data.smiles)
            if canon:
                smiles_set.add(canon)
    return smiles_set


def load_and_dedup_b3db() -> pd.DataFrame:
    df = pd.read_csv(B3DB_CSV)
    print(f"Raw B3DB classification rows: {len(df)}")

    df = df.dropna(subset=["SMILES", "BBB+/BBB-"]).copy()
    df["canonical_smiles"] = df["SMILES"].apply(canonical_smiles)
    n_invalid = df["canonical_smiles"].isna().sum()
    df = df.dropna(subset=["canonical_smiles"])
    print(f"Dropped {n_invalid} unparseable SMILES -> {len(df)} remain")

    bbbp_smiles = get_bbbp_smiles_set()
    print(f"BBBP frozen splits contain {len(bbbp_smiles)} unique canonical SMILES")

    overlap_mask = df["canonical_smiles"].isin(bbbp_smiles)
    n_overlap = overlap_mask.sum()
    df = df[~overlap_mask].copy()
    print(f"Removed {n_overlap} molecules overlapping with BBBP -> {len(df)} genuinely external")

    df["label"] = (df["BBB+/BBB-"].str.strip() == "BBB+").astype(int)
    print(f"Label balance: {df['label'].mean():.1%} BBB+")

    return df


def build_b3db_graphs(df: pd.DataFrame) -> list:
    data_list = []
    failed = 0
    for _, row in df.iterrows():
        data = smiles_to_data(row["SMILES"], float(row["label"]))
        if data is None:
            failed += 1
        else:
            data_list.append(data)
    print(f"Built {len(data_list)} graphs, {failed} failed featurization")
    return data_list


@torch.no_grad()
def run_inference(model, data_list: list) -> tuple:
    import numpy as np
    model.eval()
    all_probs, all_labels = [], []
    for data in data_list:
        data = data.to(DEVICE)
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
        logit = model(data.x, data.edge_index, data.edge_attr, batch)
        prob = torch.sigmoid(logit).cpu().item()
        all_probs.append(prob)
        all_labels.append(data.y.item())
    return np.array(all_labels), np.array(all_probs)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_dedup_b3db()
    data_list = build_b3db_graphs(df)

    for model_name, ckpt_path in [
        ("gat", "results/nethra/gat_best.pt"),
        ("gin", "results/nethra/gin_best.pt"),
    ]:
        print(f"\n{'='*60}\nRunning {model_name.upper()} on external B3DB set\n{'='*60}")
        model = load_model(model_name, ckpt_path).to(DEVICE)

        y_true, y_prob = run_inference(model, data_list)
        metrics = evaluate_model(y_true, y_prob)
        print_metrics(metrics, model_name=f"{model_name.upper()} (external B3DB)", split="b3db_external")
        save_metrics(metrics, model_name=f"{model_name}_b3db_external", split="external", out_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()