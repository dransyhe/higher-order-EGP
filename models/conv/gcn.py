import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr=None, expander_node_mask=None):
        x = self.linear(x)
        if edge_attr is not None:
            edge_embedding = self.bond_encoder(edge_attr)
        else:
            edge_embedding = None

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # set expander_node_feature to 0-vector
        if expander_node_mask is not None:
            expander_node_mask = expander_node_mask.unsqueeze(dim=-1)
            expander_node_mask = expander_node_mask.expand(expander_node_mask.shape[0],
                                                           x.shape[1])
            x = torch.where(expander_node_mask > 0, x, 0.0)

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        if edge_attr is not None:
            return norm.view(-1, 1) * F.relu(x_j + edge_attr)
        else:
            return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

