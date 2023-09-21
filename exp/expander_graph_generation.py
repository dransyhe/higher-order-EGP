import networkx as nx
import math
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, convert


# NOTE - Below has been adapted to only work on tree_neighbours_match dataset
def add_expander_edges_via_ramanujan_bipartite_graph(hypergraph_order: int,
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
     :param data: graph to be augmented
     :return: updated graph with additional attributes for expander graph
     """
    new_data = data
    num_nodes = data.x.shape[0]
    expander_graph_edge_nodes = torch.zeros(data.x.shape, dtype=data.x.dtype)
    expander_graph_x = torch.concat((data.x, expander_graph_edge_nodes))
    new_num_nodes = expander_graph_x.shape[0]
    new_root_mask = torch.tensor([True] + [False] * (new_num_nodes - 1))

    connected = False
    ramanujan = False
    while not (connected and ramanujan):
        all_destination_nodes = torch.tensor([])
        for i in range(hypergraph_order):
            valid_shuffle = False
            while not valid_shuffle:
                destination_nodes = torch.tensor([num_nodes + j for j in range(num_nodes)])
                rand_perm = torch.randperm(destination_nodes.shape[0])
                destination_nodes = destination_nodes[rand_perm]
                if i == 0 or num_nodes < hypergraph_order:
                    # If there are fewer nodes than the order of the hypergraph, we can't avoid duplicate edges in the
                    # constructed bipartite graph
                    valid_shuffle = True
                else:
                    # Checks to ensure that all hyperedges consist of 'hypergraph_order' unique nodes so bipartite graph is regular
                    valid_shuffle = True
                    for j in range(hypergraph_order - 1, 0, -1):
                        nodes_in_edge = torch.cat((all_destination_nodes[-j:], destination_nodes[:hypergraph_order - j]))
                        valid_shuffle = valid_shuffle and (torch.unique(nodes_in_edge).shape[0] == nodes_in_edge.shape[0])
            all_destination_nodes = torch.cat((all_destination_nodes, destination_nodes))

        all_source_nodes = [i for i in range(num_nodes)]
        all_source_nodes = torch.tensor(np.repeat(all_source_nodes, hypergraph_order).tolist())

        expander_edge_index = torch.cat((all_source_nodes[None, ...], all_destination_nodes[None, ...]), dim=0)
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

    assert ramanujan and connected
    ones = torch.ones(num_nodes)
    zeros = torch.zeros(num_nodes)
    expander_node_mask = torch.concat((ones, zeros))
    new_data['expander_edge_index'] = expander_edge_index
    new_data['expander_node_mask'] = expander_node_mask
    new_data['x'] = expander_graph_x
    new_data['num_nodes'] = new_num_nodes
    new_data['root_mask'] = new_root_mask
    return new_data
