import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder


### GIN convolution along the graph structure
# NOTE - This has been changed for consistency with 'run_tree_neighbours_match' task and shouldn't be run with
# OGB
class GINConv(MessagePassing):
    def __init__(self, emb_dim, task, flow=None):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        if flow is None:
            super(GINConv, self).__init__(aggr = "add")
        else:
            super(GINConv, self).__init__(aggr = "add", flow = flow)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim),
                                       torch.nn.BatchNorm1d(emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        if task == "mol":
            self.edge_encoder = BondEncoder(emb_dim=emb_dim)
        elif task == "ppo":
            self.edge_encoder = torch.nn.Linear(7, emb_dim)
        elif task == "code2":
            self.edge_encoder = torch.nn.Linear(2, emb_dim)
        elif task == "tree_neighbours_match":
            # TODO: Double check that they don't use edge encodings
            self.edge_encoder = None
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, edge_attr=None, expander_node_mask=None):
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = None

        # set expander_node_feature to 0-vector
        if expander_node_mask is not None:
            x = x * expander_node_mask

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return F.relu(x_j + edge_attr)
        else:
            return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


