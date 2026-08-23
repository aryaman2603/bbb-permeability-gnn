
import torch
from torch_geometric.explain import Explainer, GNNExplainer


def create_explainer(model):
    """
    Create a GNNExplainer configured for
    graph-level binary classification.
    """

    return Explainer(
        model=model,
        algorithm=GNNExplainer(
            epochs=100,
            lr=0.01,
        ),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "binary_classification",
            "task_level": "graph",
            "return_type": "raw",
        },
    )


def explain_molecule(explainer, data):
    """
    Generate a GNNExplainer explanation
    for one molecular graph.
    """

    batch = torch.zeros(
        data.x.size(0),
        dtype=torch.long,
        device=data.x.device,
    )

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        batch=batch,
    )

    return explanation


def get_node_scores(explanation):
    """
    Extract one importance score per atom, by summing feature-level
    importances across each atom's 22-dim feature mask.
    """

    node_mask = (
        explanation.node_mask
        .detach()
        .cpu()
        .numpy()
    )

    if node_mask.ndim == 2:
        # node_mask_type="attributes" -> [num_atoms, num_features]
        # collapse to one score per atom
        return node_mask.sum(axis=1)

    return node_mask.squeeze()


def get_edge_scores(explanation):
    """
    Extract importance scores for PyG edges.
    """

    return (
        explanation.edge_mask
        .detach()
        .cpu()
        .numpy()
    )
