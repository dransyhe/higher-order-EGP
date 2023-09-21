import torch
from models.conv.gin import GINConv
from models.conv.summation import SumConv


# GNN to generate node embedding
# NOTE - Adapted to only run on tree_neighbours_match task for speed
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, task, residual=True, gnn_type='gin', tree_neighbours_dim0=None):
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
        self.layer_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, task))
            else:
                raise ValueError('Only GIN currently supported.')

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
            if self.residual:
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
        self.layer_norms = torch.nn.ModuleList()
        self.expander_left_convs = torch.nn.ModuleList()
        self.expander_left_layer_norms = torch.nn.ModuleList()
        self.expander_right_convs = torch.nn.ModuleList()
        self.expander_right_layer_norms = torch.nn.ModuleList()
        self.summation = torch.nn.ModuleList()

        for _ in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, task))
                self.layer_norms.append(torch.nn.LayerNorm(emb_dim))
                if self.expander_edge_handling not in ["summation", "summation-mlp"]:
                    self.expander_left_convs.append(GINConv(emb_dim, task, flow="source_to_target"))
                    self.expander_left_layer_norms.append(torch.nn.LayerNorm(emb_dim))
                self.expander_right_convs.append(GINConv(emb_dim, task, flow="source_to_target"))
                self.expander_right_layer_norms.append(torch.nn.LayerNorm(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))           

            if self.expander_edge_handling in ["summation", "summation-mlp"]:
                self.summation.append(
                    SumConv(emb_dim, mlp=True if self.expander_edge_handling == "summation-mlp" else False))
                self.expander_left_layer_norms.append(torch.nn.LayerNorm(emb_dim))


    def propagate(self, conv, h, edge_index, edge_attr=None, expander_node_mask=None, masking=False, update_nodes="original"):
        h_residual = h
        h = conv(h, edge_index, edge_attr, masking, expander_node_mask, update_nodes)

        if self.residual:
            if update_nodes == "expander":
                # Don't update original nodes on left -> right
                h = h + (1 - expander_node_mask) * h_residual
            elif update_nodes == "original":
                # Don't update hyperedge nodes on right -> left
                h = h + expander_node_mask * h_residual
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
                               h_list[layer], edge_index, edge_attr, expander_node_mask=expander_node_mask, masking=False,
                               update_nodes="original")
            h = self.layer_norms[layer](h)

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
                                   h, expander_edge_index,
                                   expander_node_mask=expander_node_mask, masking=masking, update_nodes="expander")
            h = self.expander_left_layer_norms[layer](h)

            # from right to left
            reverse_expander_edge_index = expander_edge_index[[1, 0]]
            h = self.propagate(self.expander_right_convs[layer],
                               h, reverse_expander_edge_index, expander_node_mask=expander_node_mask, masking=False,
                               update_nodes="original")

            h = self.expander_right_layer_norms[layer](h)
            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation


class GNN(torch.nn.Module):

    def __init__(self, task, num_layer, emb_dim, gnn_type='gin', residual=True, expander=False,
                 expander_edge_handling="learn-features", tree_neighbours_dim0=None, tree_neighbours_out_dim=None):
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
        roots = batched_data.root_mask
        root_nodes = h_node[roots]
        logits = self.tree_neighbours_out_layer(root_nodes)
        return logits
