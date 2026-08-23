import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGEClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 22,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggr: str = "mean",  # SAGEConv's neighbor-aggregation function: 'mean', 'max', or 'lstm'
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_ch, hidden_channels, aggr=aggr))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,  # unused: vanilla SAGEConv doesn't take edge features either
        batch: torch.Tensor,
    ) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        out = self.classifier(x).squeeze(-1)
        return out


if __name__ == "__main__":
    model = GraphSAGEClassifier()
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")