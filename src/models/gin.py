import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class GINConvWithEdge(MessagePassing):
   

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, eps: float = 0.0):
       
        super().__init__(aggr="add")  


        self.edge_proj = nn.Linear(edge_dim, node_dim)


        self.mlp = nn.Sequential(
            nn.Linear(node_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )


        self.eps = nn.Parameter(torch.tensor([eps]))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
       
      
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        out = self.mlp((1 + self.eps) * x + agg)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
     
        edge_emb = self.edge_proj(edge_attr)      
        return F.relu(x_j + edge_emb)            


class GINClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 22,       
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        edge_dim: int = 7,           
    ):
        
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

   
        pool_dim = hidden_channels * 2 

        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, 64),
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
      
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = global_mean_pool(x, batch)         
        x_max  = global_max_pool(x, batch)           
        x = torch.cat([x_mean, x_max], dim=1)     

        out = self.classifier(x).squeeze(-1)         
        return out


if __name__ == "__main__":
    model = GINClassifier()
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")
