import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 22,      
        hidden_channels: int = 64,
        num_heads: int = 4,          
        num_layers: int = 3,
        dropout: float = 0.3,
        edge_dim: int = 7,           
    ):
    
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels * num_heads

            if i < num_layers - 1:
                conv = GATConv(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=True,     
                    dropout=dropout,   
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
                out_ch = hidden_channels * num_heads
            else:
                conv = GATConv(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=False,     
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
                out_ch = hidden_channels

            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(out_ch))

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
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
      
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)    
            x = F.dropout(x, p=self.dropout, training=self.training)

      
        x = global_mean_pool(x, batch)     

        out = self.classifier(x).squeeze(-1)  
        return out


if __name__ == "__main__":
    model = GATClassifier()
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")
