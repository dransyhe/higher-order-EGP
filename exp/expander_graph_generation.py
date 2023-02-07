import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce


def add_expander_edges_via_perfect_matchings(hypergraph_order: int,
                                             data: Data):
    """
    Augments graph in 'data' with a new bipartite graph representation of a hypergraph for use as an expander graph.
    For each node in the original graph, we add a node in the bipartite graph. We then generate 'hypergraph_order'
    perfect matchings of the resulting bipartite graph, and store these edges in the 'expander_edge_index' attribute
    of the 'data'

    :param hypergraph_order: number of perfect matchings to generate. This is approximately the order of the resulting
                             'hypergraph', though some 'edges' may be of a lower order as we don't enforce that the
                             matchings are disjoint.
    :param data: graph to be augmented
    :return: updated graph with additional 'expander_edge_index' attribute representing edges for expander graph
    """
    new_data = data
    num_nodes = data.x.shape[0]

    source_nodes = torch.tensor([])
    destination_nodes = torch.tensor([])
    for i in range(hypergraph_order):
        left_nodes = torch.tensor([j for j in range(num_nodes)])
        right_nodes = torch.tensor([num_nodes + j for j in range(num_nodes)])
        left_perm = torch.randperm(left_nodes.shape[0])
        right_perm = torch.randperm(right_nodes.shape[0])
        source_nodes = torch.concat((source_nodes, left_nodes[left_perm]))
        destination_nodes = torch.concat((destination_nodes, right_nodes[right_perm]))

    expander_edge_index = torch.cat([source_nodes[None, ...], destination_nodes[None, ...]], dim=0)
    expander_edge_index = coalesce(expander_edge_index)  # Remove duplicate edges
    new_data['expander_edge_index'] = expander_edge_index
    return new_data
