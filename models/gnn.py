import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from models.conv.gcn import GCNConv
from models.conv.gin import GINConv
from models.conv.summation import SumConv
from models.conv.tree_neighbours_gin import TreeNeighboursGINConv


# GNN to generate node embedding
# NOTE - Adapted to only run on tree_neighbours_match task for speed
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, task, residual=False, gnn_type='gin', tree_neighbours_dim0=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.task = task
        self.num_layer = num_layer
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self._keys_encoder = torch.nn.Embedding(num_embeddings=tree_neighbours_dim0 + 1, embedding_dim=emb_dim)
        self._values_encoder = torch.nn.Embedding(num_embeddings=tree_neighbours_dim0 + 1, embedding_dim=emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, task))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, task))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.layer_norms.append(torch.nn.LayerNorm(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch, roots = batched_data.x, batched_data.edge_index, batched_data.edge_attr, \
            batched_data.batch, batched_data.root_mask
        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self._keys_encoder(x_key)
        x_val_embed = self._values_encoder(x_val)
        h = x_key_embed + x_val_embed
        h_list = [h]

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            h = h + h_list[layer]
            h = self.layer_norms[layer](h)

            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation


class GNN_node_expander(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, task, residual=True, gnn_type='gin', expander_edge_handling="learn-features",
                 tree_neighbours_dim0=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node_expander, self).__init__()
        self.num_layer = num_layer
        self.task = task
        ### add residual connection or not
        self.residual = residual
        self.expander_edge_handling = expander_edge_handling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self._keys_encoder = torch.nn.Embedding(num_embeddings=tree_neighbours_dim0 + 1, embedding_dim=emb_dim)
        self._values_encoder = torch.nn.Embedding(num_embeddings=tree_neighbours_dim0 + 1, embedding_dim=emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.expander_left_convs = torch.nn.ModuleList()
        self.expander_left_batch_norms = torch.nn.ModuleList()
        self.expander_right_convs = torch.nn.ModuleList()
        self.expander_right_batch_norms = torch.nn.ModuleList()
        self.summation = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, task))
                if self.expander_edge_handling not in ["summation", "summation-mlp"]:
                    self.expander_left_convs.append(GINConv(emb_dim, task, flow="source_to_target"))
                self.expander_right_convs.append(GINConv(emb_dim, task, flow="source_to_target"))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, task))
                if self.expander_edge_handling not in ["summation", "summation-mlp"]:
                    self.expander_left_convs.append(GCNConv(emb_dim, task, flow="source_to_target"))
                self.expander_right_convs.append(GCNConv(emb_dim, task, flow="source_to_target"))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

            # TODO: Perhaps add layer norms between expanding layers too?
            self.layer_norms.append(torch.nn.LayerNorm(emb_dim))

            if self.expander_edge_handling in ["summation", "summation-mlp"]:
                self.summation.append(
                    SumConv(emb_dim, mlp=True if self.expander_edge_handling == "summation-mlp" else False))

            self.expander_left_batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.expander_right_batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def propagate(self, conv, bn, h, edge_index, edge_attr=None, expander_node_mask=None, masking=False, update_nodes="original"):
        h_residual = h
        h = conv(h, edge_index, edge_attr, masking, expander_node_mask, update_nodes)
        h = bn(h)
        h = F.relu(h)

        if self.residual:
            h = h + h_residual
        return h

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch, roots, expander_edge_index, expander_node_mask, num_nodes = batched_data.x, \
            batched_data.edge_index, batched_data.edge_attr, batched_data.batch, batched_data.root_mask, \
            batched_data.expander_edge_index, batched_data.expander_node_mask, batched_data.num_nodes
        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self._keys_encoder(x_key)
        x_val_embed = self._values_encoder(x_val)
        h = x_key_embed + x_val_embed

        expander_node_mask = expander_node_mask.unsqueeze(dim=-1)
        expander_node_mask = expander_node_mask.expand(expander_node_mask.shape[0],
                                                       h.shape[1])
        h = h * expander_node_mask
        h_list = [h]
        for layer in range(self.num_layer):
            # Propagation on the original graph
            h = self.propagate(self.convs[layer],
                               self.batch_norms[layer],
                               h_list[layer], edge_index, edge_attr, expander_node_mask=expander_node_mask, masking=False,
                               update_nodes="original")

            # Propagation on the expander graph
            # from left to right.
            if self.expander_edge_handling in ["summation", "summation-mlp"]:
                h = h * expander_node_mask
                h_edge = self.summation[layer](h, expander_edge_index)
                h = h + h_edge * (1 - expander_node_mask)
            else:
                if self.expander_edge_handling == "masking":
                    masking = True
                else:
                    masking = False
                h = self.propagate(self.expander_left_convs[layer],
                                   self.expander_left_batch_norms[layer],
                                   h, expander_edge_index,
                                   expander_node_mask=expander_node_mask, masking=masking, update_nodes="expander")

            # from right to left
            reverse_expander_edge_index = expander_edge_index[[1, 0]]
            h = self.propagate(self.expander_right_convs[layer],
                               self.expander_right_batch_norms[layer],
                               h, reverse_expander_edge_index, expander_node_mask=expander_node_mask, masking=False,
                               update_nodes="original")

            h = self.layer_norms[layer](h)
            # TODO: (can have other options) now only saves h at the end of three propagations
            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation


class GNN(torch.nn.Module):

    def __init__(self, task, num_layer, emb_dim, gnn_type='gin', residual=True, expander=False,
                 expander_edge_handling="learn-features", tree_neighbours_dim0=None, tree_neighbours_out_dim=None):
        '''
            num_tasks (int): number of labels to be predicted
            TODO: virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.task = task
        self.tree_neighbours_out_layer = torch.nn.Linear(in_features=emb_dim, out_features=tree_neighbours_out_dim + 1,
                                                         bias=False)
        self.expander = expander

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if not expander:
            self.gnn_node = GNN_node(num_layer, emb_dim, task=task, residual=residual, gnn_type=gnn_type,
                                     tree_neighbours_dim0=tree_neighbours_dim0)
        else:
            self.gnn_node = GNN_node_expander(num_layer, emb_dim, task=task, residual=residual, gnn_type=gnn_type,
                                              expander_edge_handling=expander_edge_handling, tree_neighbours_dim0=tree_neighbours_dim0)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        # if self.expander:
        #     # Replace batch[i] to -1 where expander_node_mask indicates it is an expander_edge_node
        #     # +1 due to scatter function requires indices to be non-negative
        #     batch = torch.where(batched_data.expander_node_mask > 0,
        #                         batched_data.batch, -1) + 1
        #     # Slice off h_graph[0] which was the aggregation of all expander_edge_node
        #     h_graph = self.pool(h_node, batch)[1:, :]
        # else:
        #     h_graph = self.pool(h_node, batched_data.batch)

        roots = batched_data.root_mask
        root_nodes = h_node[roots]
        logits = self.tree_neighbours_out_layer(root_nodes)
        return logits


class TreeNeighboursGNN(torch.nn.Module):
    def __init__(self, gnn_type, num_layers, dim0, h_dim, out_dim, layer_norm, use_activation, use_residual):
        super(TreeNeighboursGNN, self).__init__()
        self.gnn_type = gnn_type
        self.use_layer_norm = layer_norm
        self.use_activation = use_activation
        self.use_residual = use_residual
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_layers = num_layers
        self.layer0_keys = torch.nn.Embedding(num_embeddings=dim0 + 1, embedding_dim=h_dim)
        self.layer0_values = torch.nn.Embedding(num_embeddings=dim0 + 1, embedding_dim=h_dim)
        self.layers = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(TreeNeighboursGINConv(
                torch.nn.Sequential(torch.nn.Linear(h_dim, h_dim), torch.nn.BatchNorm1d(h_dim), torch.nn.ReLU(),
                                    torch.nn.Linear(h_dim, h_dim), torch.nn.BatchNorm1d(h_dim), torch.nn.ReLU())))
        if self.use_layer_norm:
            for i in range(num_layers):
                self.layer_norms.append(torch.nn.LayerNorm(h_dim))

        self.out_dim = out_dim
        self.out_layer = torch.nn.Linear(in_features=h_dim, out_features=out_dim + 1, bias=False)

    def forward(self, data):
        x, edge_index, batch, roots = data.x, data.edge_index, data.batch, data.root_mask

        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed

        for i in range(self.num_layers):
            layer = self.layers[i]
            new_x = x
            edges = edge_index
            new_x = layer(new_x, edges)
            if self.use_activation:
                new_x = F.relu(new_x)
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        root_nodes = x[roots]
        logits = self.out_layer(root_nodes)
        return logits


if __name__ == '__main__':
    GNN(num_class=10)
