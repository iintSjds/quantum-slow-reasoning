"""
Graph conversion utilities for QuCoNet - bidirectional conversion between NetworkX graphs and shift operators.

This module provides functions to:
1. Convert NetworkX graphs to shift operators for quantum walks
2. Convert shift operators back to NetworkX graphs
3. Validate that graphs and shift operators are legal for unitary quantum walks

Legal graph requirements for unitary quantum walks:
- Must be d-regular (all nodes have same degree K)
- Must be undirected (symmetric adjacency)
- Must have unique edge port mapping for shift operation
"""

import torch
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Union


def is_graph_legal(G: nx.Graph) -> Tuple[bool, str]:
    """
    Check if a NetworkX graph is legal for unitary quantum walks.

    Args:
        G: NetworkX graph to validate

    Returns:
        (is_legal, message): Tuple of boolean legality status and explanatory message
    """
    # Check if graph is undirected
    if G.is_directed():
        return False, "Graph must be undirected for unitary quantum walks"

    # Check if graph is simple (no self-loops or multiple edges)
    if nx.number_of_selfloops(G) > 0:
        return False, "Graph must be simple (no self-loops or multiple edges)"

    # Check for multiple edges by comparing edge count with simple edge count
    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        return False, "Graph must be simple (no self-loops or multiple edges)"

    # Check for empty graph
    if G.number_of_nodes() == 0:
        return False, "Graph has no nodes"

    # Check for isolated nodes
    if any(G.degree(node) == 0 for node in G.nodes()):
        return False, "Graph has isolated nodes (degree 0)"

    # Check if graph is connected (recommended but not strictly required)
    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        # This is a warning, not an error
        print("Warning: Graph is not connected. Quantum walk may not explore all nodes.")

    # Check if graph is regular (all nodes have same degree)
    degrees = [G.degree(node) for node in G.nodes()]
    if len(set(degrees)) > 1:
        return False, f"Graph must be d-regular. Found degrees: {set(degrees)}"

    return True, "Graph is legal for unitary quantum walks"


def graph_to_adjacency_list(G: nx.Graph) -> List[List[int]]:
    """
    Convert NetworkX graph to adjacency list format used by QuCoNet.

    The adjacency list is ordered by node index, and neighbors are sorted
    to ensure consistent ordering for shift operator construction.

    Args:
        G: NetworkX graph

    Returns:
        adjacency_list: List where adjacency_list[i] contains sorted neighbors of node i

    Raises:
        ValueError: If graph nodes are not labeled 0 to N-1
    """
    N = G.number_of_nodes()

    # Check if nodes are labeled 0 to N-1
    expected_nodes = set(range(N))
    actual_nodes = set(G.nodes())
    if actual_nodes != expected_nodes:
        raise ValueError(f"Graph nodes must be labeled 0 to {N-1}, got {sorted(actual_nodes)}")

    # Build adjacency list with sorted neighbors
    adjacency_list = []
    for node in range(N):
        neighbors = sorted(list(G.neighbors(node)))
        adjacency_list.append(neighbors)

    return adjacency_list


def graph_to_shift_operator_maps(G: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert NetworkX graph to shift operator maps for quantum walks.

    This creates the node and coin mapping tensors used by ShiftOperator.

    Args:
        G: NetworkX graph (must be legal)

    Returns:
        (shift_node_map, shift_coin_map): Tensors for shift operator construction

    Raises:
        ValueError: If graph is not legal
    """
    # Validate graph
    is_legal, message = is_graph_legal(G)
    if not is_legal:
        raise ValueError(f"Graph is not legal: {message}")

    N = G.number_of_nodes()
    K = G.degree(0)  # All nodes have same degree (regular graph)

    # Get adjacency list
    adjacency_list = graph_to_adjacency_list(G)

    # Build shift maps
    shift_node_map = torch.zeros(N, K, dtype=torch.long)
    shift_coin_map = torch.zeros(N, K, dtype=torch.long)

    for node_x in range(N):
        neighbors = adjacency_list[node_x]

        for coin_i, node_y in enumerate(neighbors):
            # Find the coin index j that maps back from y to x
            neighbors_of_y = adjacency_list[node_y]
            try:
                coin_j = neighbors_of_y.index(node_x)
            except ValueError:
                # This should not happen for undirected graphs
                raise ValueError(f"Node {node_y} (neighbor of {node_x}) doesn't have {node_x} as neighbor")

            shift_node_map[node_x, coin_i] = node_y
            shift_coin_map[node_x, coin_i] = coin_j

    return shift_node_map, shift_coin_map


def shift_operator_maps_to_graph(shift_node_map: torch.Tensor, shift_coin_map: torch.Tensor) -> nx.Graph:
    """
    Convert shift operator maps back to NetworkX graph.

    Args:
        shift_node_map: Node mapping tensor of shape (N, K)
        shift_coin_map: Coin mapping tensor of shape (N, K)

    Returns:
        G: NetworkX graph reconstructed from shift operator

    Raises:
        ValueError: If shift operator maps are inconsistent
    """
    N, K = shift_node_map.shape

    # Create empty graph
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Build adjacency list from shift maps
    adjacency_list = [[] for _ in range(N)]

    for node_x in range(N):
        for coin_i in range(K):
            node_y = shift_node_map[node_x, coin_i].item()
            coin_j = shift_coin_map[node_x, coin_i].item()

            # Add edge if not already present
            if node_y not in adjacency_list[node_x]:
                adjacency_list[node_x].append(node_y)

                # Verify consistency: shift from y with coin_j should go back to x
                expected_x = shift_node_map[node_y, coin_j].item()
                expected_coin = shift_coin_map[node_y, coin_j].item()

                if expected_x != node_x or expected_coin != coin_i:
                    raise ValueError(
                        f"Inconsistent shift operator at ({node_x}, {coin_i}) -> ({node_y}, {coin_j}). "
                        f"Expected return: ({expected_x}, {expected_coin}), got: ({node_x}, {coin_i})"
                    )

    # Add edges to graph
    for node_x in range(N):
        for node_y in adjacency_list[node_x]:
            G.add_edge(node_x, node_y)

    return G


def validate_shift_operator_maps(shift_node_map: torch.Tensor, shift_coin_map: torch.Tensor) -> Tuple[bool, str]:
    """
    Validate that shift operator maps define a legal quantum walk.

    Args:
        shift_node_map: Node mapping tensor of shape (N, K)
        shift_coin_map: Coin mapping tensor of shape (N, K)

    Returns:
        (is_valid, message): Tuple of boolean validity status and explanatory message
    """
    # Check tensor shapes
    if shift_node_map.shape != shift_coin_map.shape:
        return False, f"Shape mismatch: node_map {shift_node_map.shape} vs coin_map {shift_coin_map.shape}"

    N, K = shift_node_map.shape

    # Check that all node indices are valid
    if torch.any(shift_node_map < 0) or torch.any(shift_node_map >= N):
        return False, f"Invalid node indices in shift_node_map (must be 0 to {N-1})"

    # Check that all coin indices are valid
    if torch.any(shift_coin_map < 0) or torch.any(shift_coin_map >= K):
        return False, f"Invalid coin indices in shift_coin_map (must be 0 to {K-1})"

    # Check bijectivity (unitarity requirement)
    # Each mapping should be unique in the combined (node, coin) space
    for node_x in range(N):
        for coin_i in range(K):
            node_y = shift_node_map[node_x, coin_i].item()
            coin_j = shift_coin_map[node_x, coin_i].item()

            # Check reverse mapping
            try:
                expected_x = shift_node_map[node_y, coin_j].item()
                expected_coin = shift_coin_map[node_y, coin_j].item()

                if expected_x != node_x or expected_coin != coin_i:
                    return False, (
                        f"Non-bijective mapping at ({node_x}, {coin_i}) -> ({node_y}, {coin_j}). "
                        f"Reverse maps to ({expected_x}, {expected_coin})"
                    )
            except IndexError:
                return False, f"Invalid indices in reverse mapping check"

    # Check that the graph is regular by reconstructing it
    try:
        G = shift_operator_maps_to_graph(shift_node_map, shift_coin_map)
        is_regular = len(set(G.degree(n) for n in G.nodes())) == 1
        if not is_regular:
            return False, "Graph reconstructed from shift operator is not regular"
    except Exception as e:
        return False, f"Failed to reconstruct graph from shift operator: {str(e)}"

    return True, "Shift operator maps are valid"


def create_regular_graph(N: int, K: int, graph_type: str = "random") -> nx.Graph:
    """
    Create a regular graph for quantum walks.

    Args:
        N: Number of nodes
        K: Degree of each node (must be even for random regular graphs)
        graph_type: Type of graph to create ("random", "cycle", "complete", "torus")

    Returns:
        G: Regular NetworkX graph

    Raises:
        ValueError: If parameters are invalid for the requested graph type
    """
    if K >= N:
        raise ValueError(f"Degree K={K} must be less than number of nodes N={N}")

    if graph_type == "random":
        # Create random regular graph (K must be even for N*K to be even)
        if (N * K) % 2 != 0:
            raise ValueError(f"N*K={N}*{K}={N*K} must be even for regular graph")
        G = nx.random_regular_graph(K, N)

    elif graph_type == "cycle":
        # Create cycle graph (K=2)
        if K != 2:
            raise ValueError(f"Cycle graph requires K=2, got K={K}")
        G = nx.cycle_graph(N)

    elif graph_type == "complete":
        # Create complete graph (K=N-1)
        if K != N - 1:
            raise ValueError(f"Complete graph requires K=N-1={N-1}, got K={K}")
        G = nx.complete_graph(N)

    elif graph_type == "torus":
        # Create toroidal grid graph (regular grid with wraparound)
        # For degree 4, we need a 2D torus
        if K != 4:
            raise ValueError(f"Torus graph requires K=4, got K={K}")

        # Find grid dimensions
        side = int(np.sqrt(N))
        if side * side != N:
            raise ValueError(f"Torus graph requires N to be a perfect square, got N={N}")

        # Create toroidal grid (periodic boundary conditions)
        G = nx.grid_graph(dim=[side, side], periodic=True)
        # Relabel nodes to integers
        mapping = {(i, j): i * side + j for i in range(side) for j in range(side)}
        G = nx.relabel_nodes(G, mapping)

    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Validate the created graph
    is_legal, message = is_graph_legal(G)
    if not is_legal:
        raise ValueError(f"Created graph is not legal: {message}")

    return G


def visualize_graph_with_ports(G: nx.Graph, shift_node_map: torch.Tensor,
                               shift_coin_map: torch.Tensor,
                               figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualize a graph with port assignments for shift operator.

    This is useful for debugging and understanding the shift operator structure.

    Args:
        G: NetworkX graph
        shift_node_map: Node mapping tensor
        shift_coin_map: Coin mapping tensor
        figsize: Figure size for plotting
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for visualization. Install with: pip install matplotlib")
        return

    plt.figure(figsize=figsize)

    # Use spring layout for visualization
    pos = nx.spring_layout(G, k=2/np.sqrt(G.number_of_nodes()), iterations=50)

    # Draw graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, font_weight='bold')

    # Add port labels to edges
    N, K = shift_node_map.shape
    edge_labels = {}

    for node_x in range(N):
        for coin_i in range(K):
            node_y = shift_node_map[node_x, coin_i].item()
            coin_j = shift_coin_map[node_x, coin_i].item()

            # Add bidirectional port labels
            if (node_x, node_y) not in edge_labels:
                edge_labels[(node_x, node_y)] = f"{coin_i}↔{coin_j}"
            else:
                # If edge already has a label, append new port info
                existing_label = edge_labels[(node_x, node_y)]
                if f"{coin_i}↔{coin_j}" not in existing_label:
                    edge_labels[(node_x, node_y)] += f", {coin_i}↔{coin_j}"

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    plt.title(f"Graph with Shift Operator Port Assignments\nN={N}, K={K}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def analyze_graph_properties(G: nx.Graph) -> Dict[str, Union[int, float, List]]:
    """
    Analyze graph properties relevant to quantum walks.

    Args:
        G: NetworkX graph

    Returns:
        properties: Dictionary of graph properties
    """
    # Basic properties
    N = G.number_of_nodes()
    M = G.number_of_edges()
    K = G.degree(0) if N > 0 else 0

    # Check if graph is bipartite
    is_bipartite = nx.is_bipartite(G)

    # Calculate diameter
    try:
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        diameter = None  # Graph is not connected

    # Calculate average clustering coefficient
    avg_clustering = nx.average_clustering(G)

    # Calculate spectral gap (second largest eigenvalue of normalized Laplacian)
    try:
        L = nx.normalized_laplacian_matrix(G)
        eigenvals = np.linalg.eigvals(L.toarray())
        eigenvals_sorted = np.sort(eigenvals)
        spectral_gap = eigenvals_sorted[1] if len(eigenvals_sorted) > 1 else 0
    except:
        spectral_gap = None

    # Degree sequence
    degree_sequence = sorted([G.degree(n) for n in G.nodes()], reverse=True)

    properties = {
        'num_nodes': N,
        'num_edges': M,
        'degree': K,
        'is_bipartite': is_bipartite,
        'diameter': diameter,
        'avg_clustering': avg_clustering,
        'spectral_gap': spectral_gap,
        'degree_sequence': degree_sequence,
        'is_connected': nx.is_connected(G),
        'num_connected_components': nx.number_connected_components(G)
    }

    return properties


def find_qa_pairs_with_distance(graph, batch_size=1, min_distance=3, max_distance=6):
    """Find multiple distinct QA pairs with appropriate distance.

    This utility function searches for distinct question-answer pairs in a graph
    with specified distance constraints. It's useful for creating training data
    for QuCoNet that ensures QA pairs are neither too close (trivial) nor too
    far (impossible within max_steps).

    Args:
        graph: NetworkX graph to search for QA pairs
        batch_size: Number of QA pairs to find
        min_distance: Minimum shortest path distance between Q and A
        max_distance: Maximum shortest path distance between Q and A

    Returns:
        List of tuples (start_node, target_node) representing QA pairs

    Note:
        - Pairs are guaranteed to be distinct (no duplicates or reverses)
        - If not enough pairs are found within constraints, fills remaining with
          random valid pairs
        - Graph must be connected to find pairs at specific distances
    """
    N = graph.number_of_nodes()
    qa_pairs = []
    attempts = 0
    max_attempts = batch_size * 100  # Allow more attempts per pair

    while len(qa_pairs) < batch_size and attempts < max_attempts:
        start = np.random.randint(0, N)
        target = np.random.randint(0, N)

        if start == target:
            attempts += 1
            continue

        try:
            distance = nx.shortest_path_length(graph, start, target)
            if min_distance <= distance <= max_distance:
                pair = (start, target)
                # Ensure distinct pairs
                if pair not in qa_pairs and (target, start) not in qa_pairs:
                    qa_pairs.append(pair)
        except nx.NetworkXNoPath:
            pass

        attempts += 1

    if len(qa_pairs) < batch_size:
        print(f"Warning: Only found {len(qa_pairs)} QA pairs out of requested {batch_size}")
        # Fill remaining with random valid pairs
        while len(qa_pairs) < batch_size:
            start = np.random.randint(0, N)
            target = np.random.randint(0, N)
            if start != target:
                pair = (start, target)
                if pair not in qa_pairs and (target, start) not in qa_pairs:
                    qa_pairs.append(pair)

    return qa_pairs


def save_graph_with_qa_pairs(graph: nx.Graph, qa_pairs: list, filepath: str):
    """Save a NetworkX graph and QA pairs to a file.

    This function saves both the graph structure and QA pairs to a single file
    using PyTorch's serialization format.

    Args:
        graph: NetworkX graph to save (must be legal for quantum walks)
        qa_pairs: List of (start_node, target_node) tuples
        filepath: Path to save the file (should end in .pt)

    Raises:
        ValueError: If graph is not legal for quantum walks
        IOError: If file cannot be written

    Example:
        >>> G = create_regular_graph(50, 3)
        >>> qa_pairs = find_qa_pairs_with_distance(G, batch_size=5)
        >>> save_graph_with_qa_pairs(G, qa_pairs, 'graph_qa.pt')
        >>> print(f"Saved graph with {G.number_of_nodes()} nodes and {len(qa_pairs)} QA pairs")
    """
    # Validate graph legality
    is_legal, message = is_graph_legal(graph)
    if not is_legal:
        raise ValueError(f"Cannot save illegal graph: {message}")

    # Create simple data dictionary to save
    data = {
        'graph': graph,
        'qa_pairs': qa_pairs,
        'metadata': {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'degree': graph.degree(0) if graph.number_of_nodes() > 0 else 0,
            'num_qa_pairs': len(qa_pairs)
        }
    }

    # Save with PyTorch
    try:
        torch.save(data, filepath)
    except Exception as e:
        raise IOError(f"Failed to save file {filepath}: {str(e)}")


def load_graph_with_qa_pairs(filepath: str):
    """Load a NetworkX graph and QA pairs from a file.

    This function loads a graph and QA pairs that were previously saved with
    save_graph_with_qa_pairs.

    Args:
        filepath: Path to the saved .pt file

    Returns:
        tuple: (graph, qa_pairs) where graph is a NetworkX graph and
               qa_pairs is a list of (start_node, target_node) tuples

    Raises:
        FileNotFoundError: If file does not exist
        IOError: If file cannot be read or is corrupted
        ValueError: If loaded data is invalid

    Example:
        >>> G, qa_pairs = load_graph_with_qa_pairs('graph_qa.pt')
        >>> print(f"Loaded {G.number_of_nodes()} nodes and {len(qa_pairs)} QA pairs")
        >>> print(f"Graph degree: {G.degree(0)}, Connected: {nx.is_connected(G)}")
    """
    # Load data with weights_only=False since we trust our own saved files
    try:
        data = torch.load(filepath, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise IOError(f"Failed to load file {filepath}: {str(e)}")

    # Validate data structure
    required_keys = ['graph', 'qa_pairs']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Corrupted file: missing '{key}' key")

    graph = data['graph']
    qa_pairs = data['qa_pairs']

    # Validate loaded graph
    is_legal, message = is_graph_legal(graph)
    if not is_legal:
        raise ValueError(f"Loaded graph is not legal: {message}")

    # Return graph and QA pairs
    return graph, qa_pairs