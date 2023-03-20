import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, task, flow=None):
        if flow is None:
            super(GCNConv, self).__init__(aggr='add')
        else:
            super(GCNConv, self).__init__(aggr = "add", flow = flow)

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        if task == "mol":
            self.edge_encoder = BondEncoder(emb_dim = emb_dim)
        elif task == "ppo":
            self.edge_encoder = torch.nn.Linear(7, emb_dim)
        elif task == "code2":
            self.edge_encoder = torch.nn.Linear(2, emb_dim)
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, edge_attr=None, masking=False, expander_node_mask=None, update_nodes="original"):
        x = self.linear(x)
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)

        else:
            edge_embedding = None

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # set expander_node_feature to 0-vector
        if masking:
            x = x * expander_node_mask

        out = self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

        if update_nodes == "expander":
            # Don't update original nodes on left -> right
            out = (1 - expander_node_mask) * out + expander_node_mask * x
        elif update_nodes == "original":
            # Don't update hyperedge nodes on right -> left
            out = expander_node_mask * out + (1 - expander_node_mask) * x

        return out

    def message(self, x_j, edge_attr, norm):
        if edge_attr is not None:
            return norm.view(-1, 1) * F.relu(x_j + edge_attr)
        else:
            return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

