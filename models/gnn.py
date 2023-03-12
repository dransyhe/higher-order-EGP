import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from models.conv.gcn import GCNConv
from models.conv.gin import GINConv
from models.conv.summation import SumConv


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, task, node_encoder=None, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin',
                 tree_neighbours_dim0=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.task = task
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if task == "mol":
            self.node_encoder = AtomEncoder(emb_dim)
        elif task == "ppa":
            self.node_encoder = torch.nn.Embedding(1, emb_dim)
        elif task == "code2":
            assert (node_encoder is not None)
            self.node_encoder = node_encoder
        elif task == "tree_neighbours_match":
            self._keys_encoder = torch.nn.Embedding(num_embeddings=tree_neighbours_dim0 + 1, embedding_dim=emb_dim)
            self._values_encoder = torch.nn.Embedding(num_embeddings=tree_neighbours_dim0 + 1, embedding_dim=emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, task))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, task))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        if self.task == "tree_neighbours_match":
            x, edge_index, edge_attr, batch, roots = batched_data.x, batched_data.edge_index, batched_data.edge_attr, \
                batched_data.batch, batched_data.root_mask
            x_key, x_val = x[:, 0], x[:, 1]
            x_key_embed = self._keys_encoder(x_key)
            x_val_embed = self._values_encoder(x_val)
            h = x_key_embed + x_val_embed
        else:
            x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
            h = self.node_encoder(x)
        ### computing input node embedding

        h_list = [h]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum" and self.task != "tree_neighbours_match":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        else:
            # TODO: I believe that the original implementation (https://github.com/tech-srl/bottleneck/blob/main/models/graph_model.py)
            #       only uses the last node representations, not sum
            raise Exception("tree_neighbours_match should only be used with the last ")

        return node_representation


class GNN_node_expander(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, task, node_encoder=None, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin',
                       expander_edge_handling="learn-features"):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node_expander, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.task = task
        ### add residual connection or not
        self.residual = residual
        self.expander_edge_handling = expander_edge_handling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if task == "mol":
            self.node_encoder = AtomEncoder(emb_dim)
        elif task == "ppa":
            self.node_encoder = torch.nn.Embedding(1, emb_dim)
        elif task == "code2":
            assert (node_encoder is not None)
            self.node_encoder = node_encoder

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.expander_left_convs = torch.nn.ModuleList()
        self.expander_left_batch_norms = torch.nn.ModuleList()
        self.expander_right_convs = torch.nn.ModuleList()
        self.expander_right_batch_norms = torch.nn.ModuleList()
        self.summation = torch.nn.ModuleList()


        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, task))
                if layer != num_layer - 1:
                    if self.expander_edge_handling not in ["summation", "summation-mlp"]:
                        self.expander_left_convs.append(GINConv(emb_dim, task, flow="source_to_target"))
                    self.expander_right_convs.append(GINConv(emb_dim, task, flow="source_to_target"))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, task))
                if layer != num_layer - 1:
                    if self.expander_edge_handling not in ["summation", "summation-mlp"]:
                        self.expander_left_convs.append(GCNConv(emb_dim, task, flow="source_to_target"))
                    self.expander_right_convs.append(GCNConv(emb_dim, task, flow="source_to_target"))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

            if layer != num_layer - 1:
                if self.expander_edge_handling in ["summation", "summation-mlp"]:
                    self.summation.append(SumConv(emb_dim, mlp=True if self.expander_edge_handling == "summation-mlp" else False))

                self.expander_left_batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
                self.expander_right_batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def propagate(self, conv, bn, h, edge_index, edge_attr=None, expander_node_mask=None, no_act=False):
        h_residual = h
        h = conv(h, edge_index, edge_attr, expander_node_mask)
        h = bn(h)
        if no_act:
            h = F.dropout(h, self.drop_ratio, training=self.training)
        else:
            h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        if self.residual:
            h += h_residual
        return h

    def forward(self, batched_data):
        ### computing input node embedding
        if self.task == "code2":
            # It has an additional node_depth
            x, edge_index, edge_attr, node_depth, batch, expander_edge_index, expander_node_mask, num_nodes = \
                batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.node_depth,  batched_data.batch, \
                batched_data.expander_edge_index, batched_data.expander_node_mask, batched_data.num_nodes
            h = self.node_encoder(x, node_depth.view(-1, ))
        else:
            x, edge_index, edge_attr, batch, expander_edge_index, expander_node_mask, num_nodes = \
                batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch, \
                batched_data.expander_edge_index, batched_data.expander_node_mask, batched_data.num_nodes
            h = self.node_encoder(x)

        expander_node_mask = expander_node_mask.unsqueeze(dim=-1)
        expander_node_mask = expander_node_mask.expand(expander_node_mask.shape[0],
                                                       h.shape[1])
        h = h * expander_node_mask
        h_list = [h]
        for layer in range(self.num_layer):
            # Propagation on the original graph
            no_act = False
            if layer == self.num_layer - 1:
                no_act = True
            h = self.propagate(self.convs[layer],
                               self.batch_norms[layer],
                               h_list[layer], edge_index, edge_attr, no_act=no_act)

            # Propagation on the expander graph
            # from left to right. We don't do this in
            # the final layer.
            if layer != self.num_layer - 1:
                if self.expander_edge_handling in ["summation", "summation-mlp"]:
                    h = h * expander_node_mask
                    h_edge = self.summation[layer](h, expander_edge_index)
                    h = h + h_edge
                else:
                    if self.expander_edge_handling == "learn-features":
                        pass_expander_node_mask = None
                    else:
                        pass_expander_node_mask = expander_node_mask
                    h = self.propagate(self.expander_left_convs[layer],
                                       self.expander_left_batch_norms[layer],
                                       h, expander_edge_index,
                                       expander_node_mask=pass_expander_node_mask)
                # from right to left
                reverse_expander_edge_index = expander_edge_index[[1, 0]]
                h = self.propagate(self.expander_right_convs[layer],
                                   self.expander_right_batch_norms[layer],
                                   h, reverse_expander_edge_index)

            # TODO: (can have other options) now only saves h at the end of three propagations
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


class GNN(torch.nn.Module):

    def __init__(self, task, num_class, max_seq_len=None, node_encoder=None, num_layer=5, emb_dim=300,
                 gnn_type='gin', residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 expander=False, expander_edge_handling="learn-features", tree_neighbours_dim0=None,
                 tree_neighbours_out_dim=None):
        '''
            num_tasks (int): number of labels to be predicted
            TODO: virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.task = task
        if self.task == "code2":
            assert (max_seq_len is not None)
            self.max_seq_len = max_seq_len
        self.graph_pooling = graph_pooling
        self.tree_neighbours_out_layer = torch.nn.Linear(in_features=emb_dim, out_features=tree_neighbours_out_dim + 1,
                                                         bias=False)
        self.expander = expander

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if not expander:
            self.gnn_node = GNN_node(num_layer, emb_dim, task=task, node_encoder=node_encoder, JK=JK,
                                     drop_ratio=drop_ratio, residual=residual, gnn_type=gnn_type, tree_neighbours_dim0=tree_neighbours_dim0)
        else:
            self.gnn_node = GNN_node_expander(num_layer, emb_dim, task=task, node_encoder=node_encoder, JK=JK,
                                              drop_ratio=drop_ratio, residual=residual,
                                              gnn_type=gnn_type, expander_edge_handling=expander_edge_handling)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            if self.task == "code2":
                self.graph_pred_linear_list = []
                for i in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(2*emb_dim, self.num_class))
            else:
                self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_class)
        else:
            if self.task == "code2":
                self.graph_pred_linear_list = []
                for i in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, self.num_class))
            else:
                self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        if self.expander:
            # Replace batch[i] to -1 where expander_node_mask indicates it is an expander_edge_node
            # +1 due to scatter function requires indices to be non-negative
            batch = torch.where(batched_data.expander_node_mask > 0,
                                batched_data.batch, -1) + 1
            # Slice off h_graph[0] which was the aggregation of all expander_edge_node
            h_graph = self.pool(h_node, batch)[1:, :]
        else:
            h_graph = self.pool(h_node, batched_data.batch)

        if self.task == "code2":
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.graph_pred_linear[i](h_graph))
            return pred_list
        elif self.task == "tree_neighbours_match":
            roots = batched_data.root_mask
            root_nodes = h_node[roots]
            logits = self.tree_neighbours_out_layer(root_nodes)
            return logits
        else:
            return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    GNN(num_class=10)
