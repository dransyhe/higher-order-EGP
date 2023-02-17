import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SumConv(MessagePassing):
    def __init__(self, emb_dim, mlp=False):
        super(SumConv, self).__init__(aggr='add')
        self.mlp = mlp
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        if self.mlp:
            x = self.linear(x)
        return x

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

