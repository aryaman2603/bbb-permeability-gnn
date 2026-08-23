import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from src.data.featurize import atom_features
from src.explain.gnn_explainer import create_explainer, explain_molecule, get_node_scores
from src.explain.visualize import get_top_atoms
import json
from pathlib import Path
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PER_CATEGORY = 15  # how many molecules to sample per TP/TN/FP/FN bucket





def find_params_file(ckpt_path: str) -> Path | None:
    """
    Look for a matching best_params.json next to the checkpoint.
    e.g. results/nethra/gin_best.pt -> results/nethra/gin_best_params.json
         results/person_b/optuna/gin_optuna_best.pt -> .../gin_best_params.json
    """
    ckpt = Path(ckpt_path)
    model_name = ckpt.stem.split("_")[0]  # 'gin_best' -> 'gin', 'gin_optuna_best' -> 'gin'
    candidate = ckpt.parent / f"{model_name}_best_params.json"
    return candidate if candidate.exists() else None


def infer_gin_config(state_dict: dict) -> dict:
    """Fallback: infer hidden_channels/num_layers from tensor shapes if no params file exists."""
    num_layers = len({k.split(".")[1] for k in state_dict if k.startswith("convs.")})
    hidden_channels = state_dict["convs.0.mlp.0.weight"].shape[0]
    return {"hidden_channels": hidden_channels, "num_layers": num_layers}


def load_model(model_name: str, ckpt_path: str, model_kwargs: dict | None = None):
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)

    if model_kwargs is None:
        params_file = find_params_file(ckpt_path)
        if params_file is not None:
            with open(params_file) as f:
                full_config = json.load(f)
            best_params = full_config["best_params"]
            # only keep the constructor args each model class actually accepts
            if model_name == "gat":
                model_kwargs = {
                    "hidden_channels": best_params["hidden_channels"],
                    "num_layers": best_params["num_layers"],
                    "dropout": best_params["dropout"],
                    "num_heads": best_params["num_heads"],
                }
            elif model_name == "gin":
                model_kwargs = {
                    "hidden_channels": best_params["hidden_channels"],
                    "num_layers": best_params["num_layers"],
                    "dropout": best_params["dropout"],
                }
            else:
                model_kwargs = {}
            print(f"Loaded config from {params_file.name}: {model_kwargs}")
        elif model_name == "gin":
            model_kwargs = infer_gin_config(state_dict)
            print(f"No params file found, inferred from checkpoint shapes: {model_kwargs}")
        else:
            model_kwargs = {}
            print(f"No params file found for {model_name}, using constructor defaults — "
                  f"this may cause a shape mismatch if the checkpoint used a tuned config.")

    if model_name == "gat":
        from src.models.gat import GATClassifier
        model = GATClassifier(**model_kwargs)
    elif model_name == "gin":
        from src.models.gin import GINClassifier
        model = GINClassifier(**model_kwargs)
    elif model_name == "gcn":
        from src.models.gcn import GCNClassifier
        model = GCNClassifier(**model_kwargs)
    elif model_name == "graphsage":
        from src.models.graphsage import GraphSAGEClassifier
        model = GraphSAGEClassifier(**model_kwargs)
    else:
        raise ValueError(model_name)

    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def categorize_predictions(model, test_data, threshold: float = 0.5) -> dict[str, list]:
    """Bucket every test molecule into TP/TN/FP/FN based on model output."""
    buckets = defaultdict(list)
    with torch.no_grad():
        for data in test_data:
            data = data.to(DEVICE)
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
            logit = model(data.x, data.edge_index, data.edge_attr, batch)
            prob = torch.sigmoid(logit).item()
            pred = int(prob >= threshold)
            true = int(data.y.item())

            if pred == 1 and true == 1:
                buckets["TP"].append(data)
            elif pred == 0 and true == 0:
                buckets["TN"].append(data)
            elif pred == 1 and true == 0:
                buckets["FP"].append(data)
            else:
                buckets["FN"].append(data)
    return buckets


def compute_fidelity(model, data, top_atom_idx: list[int]) -> float:
    """
    Fidelity: mask out the top-important atoms (zero their features) and
    measure how much the prediction probability drops. Higher = explanation
    is actually load-bearing for the model's decision, not decorative.
    """
    with torch.no_grad():
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
        orig_prob = torch.sigmoid(
            model(data.x, data.edge_index, data.edge_attr, batch)
        ).item()

        x_masked = data.x.clone()
        x_masked[top_atom_idx] = 0.0
        masked_prob = torch.sigmoid(
            model(x_masked, data.edge_index, data.edge_attr, batch)
        ).item()

    return abs(orig_prob - masked_prob)


def atom_symbol_from_features(x_row: torch.Tensor) -> str:
    """Reverse-map a 22-dim node feature vector back to its atom symbol
    (positions 0-7 are the atom one-hot: C,N,O,Cl,S,F,Br,other)."""
    from src.data.featurize import ATOM_LIST
    idx = int(x_row[:len(ATOM_LIST) + 1].argmax().item())
    return ATOM_LIST[idx] if idx < len(ATOM_LIST) else "other"


def run_aggregate_analysis(model_name: str, ckpt_path: str, fraction: float = 0.20):
    random.seed(SEED)
    test_data = torch.load("data/processed/test.pt", weights_only=False)
    model = load_model(model_name, ckpt_path)
    explainer = create_explainer(model)

    buckets = categorize_predictions(model, test_data)
    print(f"Test set breakdown — TP:{len(buckets['TP'])} TN:{len(buckets['TN'])} "
          f"FP:{len(buckets['FP'])} FN:{len(buckets['FN'])}")

    results = {}
    for category, molecules in buckets.items():
        sample = random.sample(molecules, min(N_PER_CATEGORY, len(molecules)))
        if not sample:
            continue

        fidelities, sparsities, atom_type_counts = [], [], Counter()

        for data in sample:
            explanation = explain_molecule(explainer, data)
            node_scores = get_node_scores(explanation)
            top_atoms = get_top_atoms(node_scores, fraction=fraction)

            fidelities.append(compute_fidelity(model, data, top_atoms))
            sparsities.append(len(top_atoms) / data.x.size(0))

            for idx in top_atoms:
                atom_type_counts[atom_symbol_from_features(data.x[idx])] += 1

        results[category] = {
            "n_molecules": len(sample),
            "mean_fidelity": float(np.mean(fidelities)),
            "mean_sparsity": float(np.mean(sparsities)),
            "top_atom_types": atom_type_counts.most_common(),
        }

        print(f"\n{category} (n={len(sample)}):")
        print(f"  Mean fidelity: {np.mean(fidelities):.4f}  "
              f"(higher = explanation atoms are more load-bearing)")
        print(f"  Mean sparsity: {np.mean(sparsities):.4f}  "
              f"(fraction of molecule highlighted)")
        print(f"  Most-flagged atom types: {atom_type_counts.most_common(5)}")

    return results


def run_random_control(model_name: str, fraction: float = 0.20, n_samples: int = 15):
    """
    Sanity check: explain a model with RANDOMIZED (untrained) weights on the
    same molecules. If explanations look similarly structured to the trained
    model's, that's a red flag GNNExplainer isn't reflecting learned behavior.
    """
    random.seed(SEED)
    test_data = torch.load("data/processed/test.pt", weights_only=False)
    sample = random.sample(test_data, min(n_samples, len(test_data)))

    if model_name == "gat":
        from src.models.gat import GATClassifier
        model = GATClassifier()
    elif model_name == "gin":
        from src.models.gin import GINClassifier
        model = GINClassifier()
    else:
        raise ValueError(f"Add {model_name} to run_random_control")

    model.to(DEVICE).eval()  # NOTE: untrained, random init weights
    explainer = create_explainer(model)

    fidelities, sparsities = [], []
    for data in sample:
        data = data.to(DEVICE)
        explanation = explain_molecule(explainer, data)
        node_scores = get_node_scores(explanation)
        top_atoms = get_top_atoms(node_scores, fraction=fraction)
        fidelities.append(compute_fidelity(model, data, top_atoms))
        sparsities.append(len(top_atoms) / data.x.size(0))

    print(f"\n[RANDOM-WEIGHT CONTROL] {model_name}:")
    print(f"  Mean fidelity: {np.mean(fidelities):.4f}")
    print(f"  Mean sparsity: {np.mean(sparsities):.4f}")
    print("  Compare these to the trained model's fidelity above — "
          "if similar, explanations may not reflect learned behavior.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["gat", "gin", "gcn", "graphsage"])
    parser.add_argument("--ckpt", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--fraction", type=float, default=0.20)
    args = parser.parse_args()

    run_aggregate_analysis(args.model, args.ckpt, args.fraction)
    run_random_control(args.model, args.fraction)