import random
from itertools import combinations

import numpy as np
import torch

from src.explain.gnn_explainer import create_explainer, explain_molecule, get_node_scores
from src.explain.visualize import get_top_atoms
from src.explain.aggregate_explanations import load_model, DEVICE, SEED

N_MOLECULES = 20


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compare_models(model_ckpts: dict[str, str], fraction: float = 0.20):
    """model_ckpts: e.g. {'gat': 'results/.../gat_best.pt', 'gin': '...'}"""
    random.seed(SEED)
    test_data = torch.load("data/processed/test.pt", weights_only=False)
    sample = random.sample(test_data, min(N_MOLECULES, len(test_data)))

    models = {name: load_model(name, ckpt) for name, ckpt in model_ckpts.items()}
    explainers = {name: create_explainer(m) for name, m in models.items()}

    pairwise_scores = {pair: [] for pair in combinations(models.keys(), 2)}

    for data in sample:
        data = data.to(DEVICE)
        top_atoms_per_model = {}
        for name, explainer in explainers.items():
            explanation = explain_molecule(explainer, data)
            node_scores = get_node_scores(explanation)
            top_atoms_per_model[name] = set(get_top_atoms(node_scores, fraction))

        for (m1, m2) in pairwise_scores:
            pairwise_scores[(m1, m2)].append(
                jaccard(top_atoms_per_model[m1], top_atoms_per_model[m2])
            )

    print(f"Cross-model explanation agreement (Jaccard overlap, n={len(sample)} molecules):")
    for pair, scores in pairwise_scores.items():
        print(f"  {pair[0]} vs {pair[1]}: {np.mean(scores):.3f} (avg overlap)")


if __name__ == "__main__":
    # adjust paths to your actual checkpoints
    compare_models({
        "gat": "results/nethra/gat_best.pt",
        "gin": "results/nethra/gin_best.pt",
        # "gcn": "results/person_a/gcn_best.pt",
        # "graphsage": "results/person_a/graphsage_best.pt",
    })