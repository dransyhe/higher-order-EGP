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
        destination_node_permutations = []
        # Generate `hypergraph_order` disjoint perfect matchings
        for matching_num in range(hypergraph_order):
            disjoint_matching = False
            destination_nodes = torch.tensor([num_nodes + j for j in range(num_nodes)])
            while not disjoint_matching:
                rand_perm = torch.randperm(destination_nodes.shape[0])
                destination_nodes = destination_nodes[rand_perm]
                if num_nodes < hypergraph_order:
                    # If there are fewer nodes than the order of the hypergraph, we can't avoid duplicate edges in the
                    # constructed bipartite graph
                    disjoint_matching = True
                else:
                    # Checks to ensure that the matching is disjoint to all previous
                    # matchings so that the generated hypergraph is regular
                    disjoint_matching = True
                    for i in range(matching_num):
                        disjoint_matching = disjoint_matching and (destination_nodes != destination_node_permutations[i]).all()
                        if not disjoint_matching:
                            break
            destination_node_permutations.append(destination_nodes)

        all_destination_nodes = torch.hstack(destination_node_permutations)
        all_source_nodes = torch.randperm(num_nodes)
        all_source_nodes = all_source_nodes.repeat(hypergraph_order)

        expander_edge_index = torch.cat((all_source_nodes[None, ...], all_destination_nodes[None, ...]), dim=0)
        expander_edge_index = coalesce(expander_edge_index)
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
