import networkx as nx
import math
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, convert, to_dense_adj


def add_expander_edges_via_perfect_matchings(hypergraph_order: int,
                                             dataset: str,
                                             data: Data):
    """
    Augments graph in 'data' with a new bipartite graph representation of a hypergraph for use as an expander graph.
    For each node in the original graph, we add a node in the bipartite graph. We then generate 'hypergraph_order'
    disjoint perfect matchings of the resulting bipartite graph, and store these edges in the 'expander_edge_index' attribute
    of the 'data'. We add the expander graph 'edge nodes' to the original graph nodes in 'data.x', and add an
    'expander_node_mask' attribute, where expander_node_mask[i] == 1 if data.x[i] is a node belonging to the original
    graph, and is 0 if data.x[i] is an 'edge node' belonging to the expander graph.

    :param hypergraph_order: number of perfect matchings to generate. This is the order of the resulting 'hypergraph'.
    :param dataset: dataset which is being augmented
    :param data: graph to be augmented
    :return: updated graph with additional attributes for expander graph
    """
    if dataset == "ppa":
        # ppa dataset requires manual addition of node features
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    new_data = data
    num_nodes = data.x.shape[0]
    expander_graph_edge_nodes = torch.zeros(data.x.shape, dtype=data.x.dtype)
    expander_graph_x = torch.concat((data.x, expander_graph_edge_nodes))
    new_num_nodes = expander_graph_x.shape[0]

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
    expander_edge_index = coalesce(expander_edge_index)  # Remove duplicate edges
    if num_nodes >= hypergraph_order:
            # If there are more (or equal) nodes in the original graph than the hypergraph order, then there should
            # be 'hypergraph_order' * 'num_nodes' unique edges in the graph
            assert expander_edge_index.shape[1] == hypergraph_order * num_nodes

    ones = torch.ones(num_nodes)
    zeros = torch.zeros(num_nodes)
    expander_node_mask = torch.concat((ones, zeros))
    new_data['expander_edge_index'] = expander_edge_index
    new_data['expander_node_mask'] = expander_node_mask
    new_data['x'] = expander_graph_x
    new_data['num_nodes'] = new_num_nodes
    if dataset == "code2":
        # In code2 nodes have an additional "node_depth" feature. We set this to 0 for the expander graph edge nodes, but
        # it could be initialised to any value as these nodes have their features set to 0 at the start of training.
        expander_graph_edge_node_depths = torch.zeros(data.node_depth.shape, dtype=data.node_depth.dtype)
        expander_graph_node_depths = torch.concat((data.node_depth, expander_graph_edge_node_depths))
        new_data['node_depth'] = expander_graph_node_depths
    return new_data


def add_expander_edges_via_ramanujan_bipartite_graph(hypergraph_order: int,
                                                     dataset: str,
                                                     data: Data):
    """
     Augments graph in 'data' with a new bipartite graph representation of a hypergraph for use as an expander graph.
     For each node in the original graph, we add a node in the bipartite graph. We then generate 'hypergraph_order' disjoint
     perfect matchings of the resulting bipartite graph, and store these edges in the 'expander_edge_index' attribute
     of the 'data'. Each node in the bipartite expander graph is of degree 'hypergraph_order'. We add the expander graph
     'edge nodes' to the original graph nodes in 'data.x', and add an 'expander_node_mask' attribute, where
     expander_node_mask[i] == 1 if data.x[i] is a node belonging to the original graph, and is 0 if data.x[i] is an
     'edge node' belonging to the expander graph. We test to ensure that the bipartite expander graph produced is fully
     connected and satisfies the Ramanujan property (https://en.wikipedia.org/wiki/Ramanujan_graph), making it a good
     candidate for an expander graph.

     :param hypergraph_order: number of perfect matchings to generate. This is the order of the resulting 'hypergraph'.
     :param dataset: dataset which is being augmented
     :param data: graph to be augmented
     :return: updated graph with additional attributes for expander graph
     """
    if dataset == "ppa":
        # ppa dataset requires manual addition of node features
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    new_data = data
    num_nodes = data.x.shape[0]
    expander_graph_edge_nodes = torch.zeros(data.x.shape, dtype=data.x.dtype)
    expander_graph_x = torch.concat((data.x, expander_graph_edge_nodes))
    new_num_nodes = expander_graph_x.shape[0]

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
        if dataset != "code2" and dataset != "ppa":
            # calculating eigenvalues of adjacency matrix is too slow for code2 and ppa graphs
            graph_data = Data(edge_index=expander_edge_index, num_nodes=num_nodes * 2)
            nx_graph = convert.to_networkx(graph_data)
            connected = nx.is_connected(nx_graph.to_undirected())
            adj_matrix = nx.adjacency_matrix(nx_graph.to_undirected())
            adj_eigenvalues = np.sort(np.linalg.eigvals(adj_matrix.toarray()))
            second_largest_eigenvalue = max(abs(adj_eigenvalues[1]), adj_eigenvalues[-2])
            ramanujan = second_largest_eigenvalue <= 2 * math.sqrt(hypergraph_order - 1)
        else:
            raise ValueError("Ramanujan property cannot currently be checked for code2 and ppa graphs due to computational costs.")

    assert ramanujan and connected
    ones = torch.ones(num_nodes)
    zeros = torch.zeros(num_nodes)
    expander_node_mask = torch.concat((ones, zeros))
    new_data['expander_edge_index'] = expander_edge_index
    new_data['expander_node_mask'] = expander_node_mask
    new_data['x'] = expander_graph_x
    new_data['num_nodes'] = new_num_nodes
    if dataset == "code2":
        # In code2 nodes have an additional "node_depth" feature. We set this to 0 for the expander graph edge nodes, but
        # it could be initialised to any value as these nodes have their features set to 0 at the start of training.
        expander_graph_edge_node_depths = torch.zeros(data.node_depth.shape, dtype=data.node_depth.dtype)
        expander_graph_node_depths = torch.concat((data.node_depth, expander_graph_edge_node_depths))
        new_data['node_depth'] = expander_graph_node_depths
    return new_data


def add_expander_edges_via_perfect_matchings_shortest_paths_heuristics(hypergraph_order: int,
                                                                       dataset: str,
                                                                       data: Data):
    """
    Augments graph in 'data' with a new bipartite graph representation of a hypergraph for use as an expander graph.
    For each node in the original graph, we add a node in the bipartite graph. We then generate 'hypergraph_order'
    perfect matchings of the resulting bipartite graph, and store these edges in the 'expander_edge_index' attribute
    of the 'data'. The perfect matchings use a heuristic based on the sum of shortest paths between each pair of nodes
    connected by each hyperedge.
    We add the expander graph 'edge nodes' to the original graph nodes in 'data.x', and add an
    'expander_node_mask' attribute, where expander_node_mask[i] == 1 if data.x[i] is a node belonging to the original
    graph, and is 0 if data.x[i] is an 'edge node' belonging to the expander graph.

    :param hypergraph_order: number of perfect matchings to generate. This is approximately the order of the resulting
                             'hypergraph', though some 'edges' may be of a lower order as we don't enforce that the
                             matchings are disjoint.
    :param data: graph to be augmented
    :param dataset: dataset which is being augmented
    :return: updated graph with additional attributes for expander graph
    """
    if dataset == "ppa":
        # ppa dataset requires manual addition of node features
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    new_data = data
    num_nodes = data.x.shape[0]
    expander_graph_edge_nodes = torch.zeros(data.x.shape, dtype=data.x.dtype)
    expander_graph_x = torch.concat((data.x, expander_graph_edge_nodes))
    new_num_nodes = expander_graph_x.shape[0]

    # Compute adjacency matrix
    adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)[0]
    # Replace 0s with inf
    dis = torch.where(adj == 0, float('inf'), adj)
    # Replace diagonal values with 0
    for i in range(num_nodes):
        dis[i][i] = 0.
    # Compute shortest-paths using Floyd–Warshall algorithm
    for k in range(num_nodes):
        for i in range(num_nodes):
            if k != i:
                for j in range(num_nodes):
                    if i != j and j != k and dis[i][k] + dis[k][j] < dis[i][j]:
                        dis[i][j] = dis[i][k] + dis[k][j]

    # The first perfect-matching is random
    left_nodes = torch.tensor([i for i in range(num_nodes)])
    right_nodes = torch.tensor([num_nodes + i for i in range(num_nodes)])
    left_perm = torch.randperm(left_nodes.shape[0])
    right_perm = torch.randperm(right_nodes.shape[0])
    source_nodes = left_nodes[left_perm]
    destination_nodes = right_nodes[right_perm]

    # Compute sum_dist
    # sum_dist[right][left] stores the sum of shortest paths between left to all nodes connected to right
    sum_dist = torch.zeros((num_nodes, num_nodes))
    for (left, right) in zip(left_perm, right_perm):
        for k in range(num_nodes):
            sum_dist[right][k] += dis[k][left]

    # Generate the rest (hypergraph_order - 1) perfect-matchings using sum_dist as heuristics
    nodes = [i for i in range(num_nodes)]
    for i in range(hypergraph_order - 1):
        # In each iteration, we builds a perfect-matching using heuristic stored in sum_dist
        # We choose k-top maximum values of sum_dist[right][left]
        # We also make sure these k-top have distinct (right, left) pair
        left_set = set(nodes)
        right_set = set(nodes)
        # sum_dist stacked to one dimension for torch.sort
        _, sorted_edge_index = torch.sort(sum_dist.view(1, -1), descending=True)
        sorted_edge_index = sorted_edge_index.squeeze(0)
        j = 0
        left_nodes = []
        right_nodes = []
        while len(left_set) > 0:
            index = sorted_edge_index[j].item()
            # Recalculate index due to sum_dist stacked to one direction
            left = index % num_nodes
            right = int(index / num_nodes)
            # Check if (left, right) are unique
            if left in left_set and right in right_set:
                for k in range(num_nodes):
                    sum_dist[right][k] += dis[k][left]
                left_set.remove(left)
                right_set.remove(right)
                left_nodes += [left]
                right_nodes += [right]
            j += 1
        left_nodes = torch.Tensor(left_nodes)
        right_nodes = torch.Tensor(right_nodes) + num_nodes
        source_nodes = torch.cat((source_nodes, left_nodes))
        destination_nodes = torch.cat((destination_nodes, right_nodes))

    source_nodes = source_nodes.to(torch.long)
    destination_nodes = destination_nodes.to(torch.long)

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
    if dataset == "code2":
        # In code2 nodes have an additional "node_depth" feature. We set this to 0 for the expander graph edge nodes, but
        # it could be initialised to any value as these nodes have their features set to 0 at the start of training.
        expander_graph_edge_node_depths = torch.zeros(data.node_depth.shape, dtype=data.node_depth.dtype)
        expander_graph_node_depths = torch.concat((data.node_depth, expander_graph_edge_node_depths))
        new_data['node_depth'] = expander_graph_node_depths
    return new_data


def add_expander_edges_via_perfect_matchings_access_time_heuristics(hypergraph_order: int,
                                                                    dataset: str,
                                                                    data: Data):
    """
    Augments graph in 'data' with a new bipartite graph representation of a hypergraph for use as an expander graph.
    For each node in the original graph, we add a node in the bipartite graph. We then generate 'hypergraph_order'
    perfect matchings of the resulting bipartite graph, and store these edges in the 'expander_edge_index' attribute
    of the 'data'. The perfect matchings use a heuristic based on the sum of access times between each pair of nodes
    connected by each hyperedge. The access time was computed solving the set of linear equations as described in
    Theorem 3.1 of
    https://arxiv.org/pdf/1208.2171.pdf#:~:text=Let%20n%20be%20the%20number,can%20be%20proven%20using%20induction.
    We add the expander graph 'edge nodes' to the original graph nodes in 'data.x', and add an
    'expander_node_mask' attribute, where expander_node_mask[i] == 1 if data.x[i] is a node belonging to the original
    graph, and is 0 if data.x[i] is an 'edge node' belonging to the expander graph.

    :param hypergraph_order: number of perfect matchings to generate. This is approximately the order of the resulting
                             'hypergraph', though some 'edges' may be of a lower order as we don't enforce that the
                             matchings are disjoint.
    :param data: graph to be augmented
    :param dataset: dataset which is being augmented
    :return: updated graph with additional attributes for expander graph
    """
    if dataset == "ppa":
        # ppa dataset requires manual addition of node features
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    new_data = data
    num_nodes = data.x.shape[0]
    expander_graph_edge_nodes = torch.zeros(data.x.shape, dtype=data.x.dtype)
    expander_graph_x = torch.concat((data.x, expander_graph_edge_nodes))
    new_num_nodes = expander_graph_x.shape[0]

    # Compute adjacency matrix
    adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)[0]
    assert (torch.all(adj == adj.T))
    # Replace diagonal values with 0
    # for i in range(num_nodes):
        # adj[i][i] = 0.
    # Compute coefficient (left) and constants (right) for the set of linear equations
    coefs = []
    consts = []
    for i in range(num_nodes):
        neighbours = adj[i].nonzero()
        if len(neighbours) != 0:
            degree_inverse = 1.0 / len(neighbours)
        else:
            degree_inverse = 0.0
        for j in range(num_nodes):
            coef = torch.zeros((num_nodes * num_nodes,))
            if i == j:
                coef[i * num_nodes + j] = 1.
                const = torch.Tensor([0.])
            else:
                for k in neighbours:
                    coef[k * num_nodes + j] = degree_inverse
                coef[i * num_nodes + j] = -1.
                const = torch.Tensor([-1.])
            coefs += [coef]
            consts += [const]

    coefs = torch.stack(coefs)
    consts = torch.cat(consts)
    # Solve the set of linear equations
    try:
        h = torch.linalg.solve(coefs, consts)
    except RuntimeError:
        # When coefs is a singular matrix, we revert to perfect-matchings without heuristics
        return add_expander_edges_via_perfect_matchings(
            hypergraph_order = hypergraph_order,
            dataset = dataset,
            data = data)
    # Reshape h to get the matrix
    h = h.reshape(num_nodes, num_nodes)

    # The first perfect-matching is random
    left_nodes = torch.tensor([i for i in range(num_nodes)])
    right_nodes = torch.tensor([num_nodes + i for i in range(num_nodes)])
    left_perm = torch.randperm(left_nodes.shape[0])
    right_perm = torch.randperm(right_nodes.shape[0])
    source_nodes = left_nodes[left_perm]
    destination_nodes = right_nodes[right_perm]

    # Compute sum_h
    # sum_h[right][left] stores the sum of access times between left to all nodes connected to right
    sum_h = torch.zeros((num_nodes, num_nodes))
    for (left, right) in zip(left_perm, right_perm):
        for k in range(num_nodes):
            sum_h[right][k] += h[k][left]

    # Generate the rest (hypergraph_order - 1) perfect-matchings using sum_h as heuristics
    nodes = [i for i in range(num_nodes)]
    for i in range(hypergraph_order - 1):
        # In each iteration, we builds a perfect-matching using heuristic stored in sum_h
        # We choose k-top maximum values of sum_h[right][left]
        # We also make sure these k-top have distinct (right, left) pair
        left_set = set(nodes)
        right_set = set(nodes)
        # sum_h stacked to one dimension for torch.sort
        _, sorted_edge_index = torch.sort(sum_h.view(1, -1), descending=True)
        sorted_edge_index = sorted_edge_index.squeeze(0)
        j = 0
        left_nodes = []
        right_nodes = []
        while len(left_set) > 0:
            index = sorted_edge_index[j].item()
            # Recalculate index due to sum_h stacked to one direction
            left = index % num_nodes
            right = int(index / num_nodes)
            # Check if (left, right) are unique
            if left in left_set and right in right_set:
                for k in range(num_nodes):
                    sum_h[right][k] += h[k][left]
                left_set.remove(left)
                right_set.remove(right)
                left_nodes += [left]
                right_nodes += [right]
            j += 1
        left_nodes = torch.Tensor(left_nodes)
        right_nodes = torch.Tensor(right_nodes) + num_nodes
        source_nodes = torch.cat((source_nodes, left_nodes))
        destination_nodes = torch.cat((destination_nodes, right_nodes))

    source_nodes = source_nodes.to(torch.long)
    destination_nodes = destination_nodes.to(torch.long)

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
    if dataset == "code2":
        # In code2 nodes have an additional "node_depth" feature. We set this to 0 for the expander graph edge nodes, but
        # it could be initialised to any value as these nodes have their features set to 0 at the start of training.
        expander_graph_edge_node_depths = torch.zeros(data.node_depth.shape, dtype=data.node_depth.dtype)
        expander_graph_node_depths = torch.concat((data.node_depth, expander_graph_edge_node_depths))
        new_data['node_depth'] = expander_graph_node_depths
    return new_data
