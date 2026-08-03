"""Graph Isomorphism Network (GIN) for BBB permeability prediction.

Key Idea
--------
GIN is theoretically the most powerful standard GNN — it is as expressive
as the Weisfeiler-Lehman (WL) graph isomorphism test. While GCN uses a
symmetric normalisation and GAT uses attention, GIN uses a *learnable MLP*
and a *learnable ε parameter*:

    h_v^(k) = MLP^(k)( (1 + ε^(k)) · h_v^(k-1) + Σ_{u ∈ N(v)} h_u^(k-1) )

The key insight: by using SUM aggregation (instead of MEAN), GIN can
distinguish graphs that look identical to MEAN-based methods.

Edge Feature Handling
---------------------
Standard GINConv ignores edge features. We write a custom GINConvWithEdge
layer that projects edge features to node dim and adds them to neighbor
messages before aggregation — a lightweight but effective approach.

Architecture
------------
    Input  : node features (22-dim), edge features (7-dim)
    Layer 1: GINConvWithEdge(22 → 64)   — MLP with BN + ReLU
    Layer 2: GINConvWithEdge(64 → 64)
    Layer 3: GINConvWithEdge(64 → 64)
    Pool   : mean pool || max pool → concat → 128-dim per graph
    MLP    : 128 → 64 → ReLU → Dropout → 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class GINConvWithEdge(MessagePassing):
    """GIN convolution layer that incorporates edge features.

    Extends the standard GINConv formula with edge information:
        message(j → i) = ReLU( h_j + W_e · e_{ij} )

    where W_e projects edge features to node feature space before addition.
    """

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, eps: float = 0.0):
        """
        Args:
            node_dim: Input node feature dimension.
            edge_dim: Edge feature dimension.
            out_dim:  Output node feature dimension.
            eps:      Initial value for learnable ε parameter.
        """
        super().__init__(aggr="add")  # GIN uses SUM aggregation (not mean/max)

        # Linear projection: edge_dim → node_dim (so we can add to h_j)
        self.edge_proj = nn.Linear(edge_dim, node_dim)

        # The GIN MLP applied after aggregation
        # BatchNorm inside MLP is crucial for training stability
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

        # ε is learnable — starts at 0 (symmetric) and adapts during training
        self.eps = nn.Parameter(torch.tensor([eps]))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          Node features, shape (N, node_dim).
            edge_index: Graph edges, shape (2, E).
            edge_attr:  Edge features, shape (E, edge_dim).
        Returns:
            Updated node features, shape (N, out_dim).
        """
        # Aggregate messages from neighbors (calls self.message internally)
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # GIN update: (1 + ε) * self + aggregated_neighbors
        out = self.mlp((1 + self.eps) * x + agg)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Construct message from neighbor j, incorporating edge features.

        Args:
            x_j:       Neighbor node features, shape (E, node_dim).
            edge_attr: Edge features for each edge, shape (E, edge_dim).
        Returns:
            Messages shape (E, node_dim).
        """
        edge_emb = self.edge_proj(edge_attr)      # (E, node_dim)
        return F.relu(x_j + edge_emb)             # inject bond info into message


class GINClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 22,       # node feature dim — must match featurize.py
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        edge_dim: int = 7,           # edge feature dim — must match featurize.py
    ):
        """
        Args:
            in_channels:     Input node feature dimension (22).
            hidden_channels: Hidden layer width (64).
            num_layers:      Number of GINConv layers (3).
            dropout:         Dropout rate in classifier head (0.3).
            edge_dim:        Edge feature dimension (7).
        """
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(
                GINConvWithEdge(
                    node_dim=in_ch,
                    edge_dim=edge_dim,
                    out_dim=hidden_channels,
                )
            )

        # Dual pooling: concatenate mean + max → richer graph representation
        # Mean captures average atom properties; Max captures extreme features
        pool_dim = hidden_channels * 2  # 128

        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # raw logit
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
            x:          Node feature matrix, shape (total_nodes, 22).
            edge_index: Graph connectivity in COO format, shape (2, total_edges).
            edge_attr:  Edge feature matrix, shape (total_edges, 7).
            batch:      Node-to-graph assignment vector, shape (total_nodes,).

        Returns:
            Logits, shape (batch_size,). Apply sigmoid for probabilities.
        """
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Dual pooling: mean + max, then concatenate
        x_mean = global_mean_pool(x, batch)          # (batch_size, 64)
        x_max  = global_max_pool(x, batch)           # (batch_size, 64)
        x = torch.cat([x_mean, x_max], dim=1)         # (batch_size, 128)

        out = self.classifier(x).squeeze(-1)          # (batch_size,)
        return out


if __name__ == "__main__":
    """Quick architecture sanity check (no data needed)."""
    model = GINClassifier()
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")
