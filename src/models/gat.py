"""Graph Attention Network (GAT) for BBB permeability prediction.

Key Idea
--------
GAT improves on GCN by replacing the fixed, normalized neighbor aggregation
with a *learned attention mechanism*. For each atom i, it computes an
attention score α_{ij} for every neighbor j, reflecting how "important"
that neighbor is when updating atom i's representation.

Formally (single head):
    e_{ij} = LeakyReLU( a^T · [W·h_i || W·h_j] )
    α_{ij} = softmax_j(e_{ij})
    h_i'   = σ( Σ_j α_{ij} · W · h_j )

Multi-head attention runs K independent attention heads and
concatenates (or averages) their outputs.

Architecture
------------
    Input  : node features (22-dim), edge features (7-dim)
    Layer 1: GATConv(22 → 64, heads=4, concat=True)  → 256-dim per node
    Layer 2: GATConv(256 → 64, heads=4, concat=True) → 256-dim per node
    Layer 3: GATConv(256 → 64, heads=4, concat=False)→  64-dim per node (avg)
    Pool   : global mean pool                         →  64-dim per graph
    MLP    : 64 → 64 → ReLU → Dropout → 1            → logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 22,       # node feature dim — must match featurize.py
        hidden_channels: int = 64,
        num_heads: int = 4,          # attention heads per conv layer
        num_layers: int = 3,
        dropout: float = 0.3,
        edge_dim: int = 7,           # edge feature dim — must match featurize.py
    ):
        """
        Args:
            in_channels:     Dimension of input node features (22).
            hidden_channels: Width of hidden representations per head (64).
            num_heads:       Number of independent attention heads (4).
            num_layers:      Number of GATConv layers (3).
            dropout:         Dropout rate used both in attention weights and MLP (0.3).
            edge_dim:        Dimension of edge features (7), passed to GATConv.
        """
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # Determine input/output channels for this layer
            in_ch = in_channels if i == 0 else hidden_channels * num_heads

            if i < num_layers - 1:
                # Hidden layers: concat=True → output dim = hidden_channels * num_heads
                conv = GATConv(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=True,       # concatenate head outputs
                    dropout=dropout,   # dropout on attention coefficients (regularisation)
                    edge_dim=edge_dim, # incorporate bond features into attention
                    add_self_loops=True,
                )
                out_ch = hidden_channels * num_heads
            else:
                # Final layer: concat=False → average heads → output dim = hidden_channels
                conv = GATConv(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=False,      # average attention heads for final representation
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
                out_ch = hidden_channels

            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(out_ch))

        # Classifier MLP: graph-level representation → binary prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),   # raw logit — apply sigmoid externally or use BCEWithLogitsLoss
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:          Node feature matrix, shape (total_nodes_in_batch, 22).
            edge_index: COO-format graph connectivity, shape (2, total_edges).
            edge_attr:  Edge feature matrix, shape (total_edges, 7).
            batch:      Batch assignment vector, shape (total_nodes,).
                        batch[i] = graph index that node i belongs to.

        Returns:
            Logits tensor of shape (batch_size,). Apply sigmoid to get probabilities.
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)       # ELU: smoother than ReLU, avoids dead neurons
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Aggregate all atom embeddings into a single molecule-level vector
        x = global_mean_pool(x, batch)      # shape: (batch_size, hidden_channels)

        # Predict BBB permeability
        out = self.classifier(x).squeeze(-1)  # shape: (batch_size,)
        return out


if __name__ == "__main__":
    """Quick architecture sanity check (no data needed)."""
    model = GATClassifier()
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")
