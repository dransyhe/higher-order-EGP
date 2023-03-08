import networkx as nx
import math
import numpy as np
import random
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, convert


def add_expander_edges_via_perfect_matchings(hypergraph_order: int,
                                             data: Data):
    """
    Augments graph in 'data' with a new bipartite graph representation of a hypergraph for use as an expander graph.
    For each node in the original graph, we add a node in the bipartite graph. We then generate 'hypergraph_order'
    perfect matchings of the resulting bipartite graph, and store these edges in the 'expander_edge_index' attribute
    of the 'data'. We add the expander graph 'edge nodes' to the original graph nodes in 'data.x', and add an
    'expander_node_mask' attribute, where expander_node_mask[i] == 1 if data.x[i] is a node belonging to the original
    graph, and is 0 if data.x[i] is an 'edge node' belonging to the expander graph.

    :param hypergraph_order: number of perfect matchings to generate. This is approximately the order of the resulting
                             'hypergraph', though some 'edges' may be of a lower order as we don't enforce that the
                             matchings are disjoint.
    :param data: graph to be augmented
    :return: updated graph with additional attributes for expander graph
    """
    new_data = data
    num_nodes = data.x.shape[0]
    expander_graph_edge_nodes = torch.zeros(data.x.shape, dtype=data.x.dtype)
    expander_graph_x = torch.concat((data.x, expander_graph_edge_nodes))
    new_num_nodes = expander_graph_x.shape[0]

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
    expander_edge_index = expander_edge_index.to(torch.int64)

    ones = torch.ones(num_nodes)
    zeros = torch.zeros(num_nodes)
    expander_node_mask = torch.concat((ones, zeros))
    new_data['expander_edge_index'] = expander_edge_index
    new_data['expander_node_mask'] = expander_node_mask
    new_data['x'] = expander_graph_x
    new_data['num_nodes'] = new_num_nodes
    return new_data


def add_expander_edges_via_ramanujan_bipartite_graph(hypergraph_order: int,
                                                     random_seed: int,
                                                     data: Data):
    """
     Augments graph in 'data' with a new bipartite graph representation of a hypergraph for use as an expander graph.
     For each node in the original graph, we add a node in the bipartite graph. We then generate 'hypergraph_order' disjoint
     perfect matchings of the resulting bipartite graph, and store these edges in the 'expander_edge_index' attribute
     of the 'data'. Each node in the bipartite expander graph is of degree 'hypergraph_order'. We add the expander graph
     'edge nodes' to the original graph nodes in 'data.x', and add an 'expander_node_mask' attribute, where
     expander_node_mask[i] == 1 if data.x[i] is a node belonging to the original graph, and is 0 if data.x[i] is an
     'edge node' belonging to the expander graph. We test to ensure that the bipartite expander graph produced is fully
     connected and satisfies the Ramanujan property (https://en.wikipedia.org/wiki/Ramanujan_graph) making it a good
     candidate for an expander graph.

     :param hypergraph_order: number of perfect matchings to generate. This is the order of the resulting 'hypergraph'.
     :param random_seed: random seed for generating the expander graphs.
     :param data: graph to be augmented
     :return: updated graph with additional attributes for expander graph
     """
    new_data = data
    num_nodes = data.x.shape[0]
    expander_graph_edge_nodes = torch.zeros(data.x.shape, dtype=data.x.dtype)
    expander_graph_x = torch.concat((data.x, expander_graph_edge_nodes))
    new_num_nodes = expander_graph_x.shape[0]

    connected = False
    ramanujan = False
    random_seed_offset = 0
    while (not connected) and (not ramanujan):
        random.seed(random_seed + random_seed_offset)
        all_destination_nodes = []
        for i in range(hypergraph_order):
            valid_shuffle = False
            while not valid_shuffle:
                destination_nodes = [num_nodes + j for j in range(num_nodes)]
                random.shuffle(destination_nodes)
                if i == 0 or num_nodes < hypergraph_order:
                    # If there are fewer nodes than the order of the hypergraph, we can't avoid duplicate edges in the
                    # constructed bipartite graph
                    valid_shuffle = True
                else:
                    # Checks to ensure that all hyperedges consist of 'hypergraph_order' unique nodes so bipartite graph is regular
                    valid_shuffle = True
                    for j in range(hypergraph_order - 1, 0, -1):
                        nodes_in_edge = all_destination_nodes[-j:] + destination_nodes[:hypergraph_order - j]
                        valid_shuffle = valid_shuffle and (len(nodes_in_edge) == len(set(nodes_in_edge)))
            all_destination_nodes.extend(destination_nodes)

        all_source_nodes = [i for i in range(num_nodes)]
        all_source_nodes = torch.tensor(np.repeat(all_source_nodes, hypergraph_order).tolist())
        all_destination_nodes = torch.tensor(all_destination_nodes)

        expander_edge_index = torch.cat([all_source_nodes[None, ...], all_destination_nodes[None, ...]], dim=0)
        expander_edge_index = coalesce(expander_edge_index)
        expander_edge_index = expander_edge_index.to(torch.int64)
        if num_nodes >= hypergraph_order:
            # If there are more (or equal) nodes in the original graph than the hypergraph order, then there should
            # be 'hypergraph_order' * 'num_nodes' unique edges in the graph
            assert expander_edge_index.shape[1] == hypergraph_order * num_nodes
        graph_data = Data(edge_index=expander_edge_index, num_nodes=num_nodes * 2)
        nx_graph = convert.to_networkx(graph_data)
        connected = nx.is_connected(nx_graph.to_undirected())
        adj_matrix = nx.adjacency_matrix(nx_graph.to_undirected())
        adj_eigenvalues = np.sort(np.linalg.eigvals(adj_matrix.toarray()))
        second_largest_eigenvalue = max(abs(adj_eigenvalues[1]), adj_eigenvalues[-2])
        ramanujan = second_largest_eigenvalue <= 2 * math.sqrt(hypergraph_order - 1)
        random_seed_offset += 1

    ones = torch.ones(num_nodes)
    zeros = torch.zeros(num_nodes)
    expander_node_mask = torch.concat((ones, zeros))
    new_data['expander_edge_index'] = expander_edge_index
    new_data['expander_node_mask'] = expander_node_mask
    new_data['x'] = expander_graph_x
    new_data['num_nodes'] = new_num_nodes
    return new_data
