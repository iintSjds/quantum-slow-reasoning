"""
Sliding Puzzle (2x3) to Graph Mapping Module.

This module provides utilities to map a 2x3 sliding puzzle to an N=120, K=3 regular graph
that can be used with QuCoNet quantum walks.

The puzzle has 6 tiles (1-5 and 0 representing the empty space/hole).
Only states with the hole in the middle column (indices 1 or 4) and matching the winning
state's parity are valid graph nodes. This results in N=120 unique states forming a 
3-regular graph (one of the two parity components).

Note: The full puzzle has 240 states split into two 120-state components by parity.
This module only works with the component containing the winning state (even parity).
All moves (V, L, R) are odd permutations and flip parity, but since we only use one
component, moves always stay within the component.

Moves:
- V (Vertical): Swap hole with tile above/below (direct move)
- L (Left): 3-step rotation through left corners (compound move)
- R (Right): 3-step rotation through right corners (compound move)
"""

import itertools
import networkx as nx
from typing import List, Tuple, Dict, Optional


# Winning state configuration (even parity: 0 inversions among non-zero tiles)
WINNING_STATE = [1, 2, 3, 4, 0, 5]

# Winning parity: states must have same parity as winning state to be valid
WINNING_PARITY = 0  # Even parity

# Board layout (indices):
# 0 1 2
# 3 4 5
#
# Middle column: indices 1 and 4
# Corners: indices 0, 2, 3, 5


def is_valid_puzzle_state(state: List[int]) -> bool:
    """
    Check if a state is a valid puzzle state (hole in middle column AND correct parity).

    Args:
        state: List of 6 integers representing the puzzle state

    Returns:
        True if state is valid (hole at index 1 or 4, and correct parity), False otherwise
    """
    if len(state) != 6:
        return False
    if sorted(state) != [0, 1, 2, 3, 4, 5]:
        return False
    hole_pos = state.index(0)
    if hole_pos not in [1, 4]:
        return False
    # Check parity matches winning state
    return count_inversions(state) % 2 == WINNING_PARITY


def state_to_node_index(state: List[int], state_to_index: Dict[Tuple[int, ...], int]) -> int:
    """
    Convert a puzzle state to its node index in the graph.

    Args:
        state: List of 6 integers representing the puzzle state
        state_to_index: Dictionary mapping state tuples to node indices

    Returns:
        Node index in the graph

    Raises:
        ValueError: If state is not valid or not in the mapping
    """
    if not is_valid_puzzle_state(state):
        raise ValueError(f"Invalid puzzle state: {state}. Hole must be at index 1 or 4.")

    state_tuple = tuple(state)
    if state_tuple not in state_to_index:
        raise ValueError(f"State {state} not found in valid state space.")

    return state_to_index[state_tuple]


def node_index_to_state(node_idx: int, index_to_state: Dict[int, Tuple[int, ...]]) -> List[int]:
    """
    Convert a node index to its puzzle state.

    Args:
        node_idx: Node index in the graph
        index_to_state: Dictionary mapping node indices to state tuples

    Returns:
        List of 6 integers representing the puzzle state

    Raises:
        ValueError: If node index is out of range
    """
    if node_idx not in index_to_state:
        raise ValueError(f"Node index {node_idx} out of range.")

    return list(index_to_state[node_idx])


def apply_move(state: List[int], move: str) -> List[int]:
    """
    Apply a move (V, L, or R) to a puzzle state.

    Args:
        state: Current puzzle state as list of 6 integers
        move: Move to apply ('V', 'L', or 'R')

    Returns:
        New state after applying the move

    Raises:
        ValueError: If state is invalid or move is unknown
    """
    if not is_valid_puzzle_state(state):
        raise ValueError(f"Cannot apply move to invalid state: {state}")

    move = move.upper()
    if move not in ['V', 'L', 'R']:
        raise ValueError(f"Unknown move: {move}. Must be 'V', 'L', or 'R'.")

    new_state = state.copy()
    hole_pos = state.index(0)

    if move == 'V':
        # Vertical swap: hole swaps with position in other center
        # 1 <-> 4
        new_pos = 4 if hole_pos == 1 else 1
        new_state[hole_pos], new_state[new_pos] = new_state[new_pos], new_state[hole_pos]

    elif move == 'L':
        # Left cycle - 3-step compound move through left corners
        # Moves hole from position 1 to 4 (or 4 to 1) via left corners 0 and 3
        if hole_pos == 1:
            # Hole at top center (position 1) -> moves to bottom center (position 4)
            # Step 1: swap 1 and 0 (hole moves to 0)
            new_state[1], new_state[0] = new_state[0], new_state[1]
            # Step 2: swap 0 and 3 (hole moves to 3)
            new_state[0], new_state[3] = new_state[3], new_state[0]
            # Step 3: swap 3 and 4 (hole moves to 4)
            new_state[3], new_state[4] = new_state[4], new_state[3]
        else:  # hole_pos == 4
            # Hole at bottom center (position 4) -> moves to top center (position 1)
            # Step 1: swap 4 and 3 (hole moves to 3)
            new_state[4], new_state[3] = new_state[3], new_state[4]
            # Step 2: swap 3 and 0 (hole moves to 0)
            new_state[3], new_state[0] = new_state[0], new_state[3]
            # Step 3: swap 0 and 1 (hole moves to 1)
            new_state[0], new_state[1] = new_state[1], new_state[0]

    elif move == 'R':
        # Right cycle - 3-step compound move through right corners
        # Moves hole from position 1 to 4 (or 4 to 1) via right corners 2 and 5
        if hole_pos == 1:
            # Hole at top center (position 1) -> moves to bottom center (position 4)
            # Step 1: swap 1 and 2 (hole moves to 2)
            new_state[1], new_state[2] = new_state[2], new_state[1]
            # Step 2: swap 2 and 5 (hole moves to 5)
            new_state[2], new_state[5] = new_state[5], new_state[2]
            # Step 3: swap 5 and 4 (hole moves to 4)
            new_state[5], new_state[4] = new_state[4], new_state[5]
        else:  # hole_pos == 4
            # Hole at bottom center (position 4) -> moves to top center (position 1)
            # Step 1: swap 4 and 5 (hole moves to 5)
            new_state[4], new_state[5] = new_state[5], new_state[4]
            # Step 2: swap 5 and 2 (hole moves to 2)
            new_state[5], new_state[2] = new_state[2], new_state[5]
            # Step 3: swap 2 and 1 (hole moves to 1)
            new_state[2], new_state[1] = new_state[1], new_state[2]

    return new_state


def apply_moves(state: List[int], moves: List[str]) -> List[int]:
    """
    Apply a sequence of moves to a puzzle state.

    Args:
        state: Initial puzzle state
        moves: List of moves to apply in sequence

    Returns:
        Final state after applying all moves
    """
    current_state = state.copy()
    for move in moves:
        current_state = apply_move(current_state, move)
    return current_state


def count_inversions(state: List[int]) -> int:
    """
    Count the number of inversions in a state.

    An inversion is a pair of tiles (a, b) where a appears before b
    but a > b. The empty tile (0) is not counted.

    Args:
        state: List of integers

    Returns:
        Number of inversions
    """
    # Remove the empty tile (0) from consideration
    tiles = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions


def generate_all_valid_states() -> List[Tuple[int, ...]]:
    """
    Generate all valid puzzle states (N=120) with winning parity.

    Valid states are those where:
    1. The hole (0) is in the middle column (indices 1 or 4)
    2. The state has the same parity as the winning state (even parity)

    This results in exactly 120 states forming one connected component
    of the sliding puzzle graph.

    Note: States with wrong parity are unreachable from the winning state
    and are excluded from this module.

    Returns:
        List of all valid states as tuples, sorted lexicographically (N=120)
    """
    valid_states = []
    tiles = [1, 2, 3, 4, 5]

    # Generate all permutations of tiles 1-5
    for perm in itertools.permutations(tiles):
        # Hole at index 1 (top middle)
        state_1 = list(perm)
        state_1.insert(1, 0)
        if count_inversions(state_1) % 2 == WINNING_PARITY:
            valid_states.append(tuple(state_1))

        # Hole at index 4 (bottom middle)
        state_4 = list(perm)
        state_4.insert(4, 0)
        if count_inversions(state_4) % 2 == WINNING_PARITY:
            valid_states.append(tuple(state_4))

    return sorted(valid_states)


def create_sliding_puzzle_graph() -> Tuple[nx.Graph, Dict[Tuple[int, ...], int], Dict[int, Tuple[int, ...]]]:
    """
    Create the N=120, K=3 regular graph representing the sliding puzzle state space
    (winning parity component only).

    Returns:
        Tuple of:
        - G: NetworkX Graph with 120 nodes, each with degree 3
        - state_to_index: Dictionary mapping state tuples to node indices (0-119)
        - index_to_state: Dictionary mapping node indices to state tuples

    The graph is a 3-regular graph where:
    - Each node represents a valid puzzle state (hole in middle column, winning parity)
    - Edges represent moves V, L, or R
    - Each node has exactly 3 neighbors (one for each move type)
    - All states are reachable from the winning state

    Note: This module only uses the winning parity component (120 states).
    The other parity component (120 states) is excluded as unreachable.
    """
    # Generate all valid states (winning parity only)
    valid_states = generate_all_valid_states()
    N = len(valid_states)
    assert N == 120, f"Expected 120 valid states with winning parity, got {N}"

    # Create bidirectional mappings
    state_to_index = {state: idx for idx, state in enumerate(valid_states)}
    index_to_state = {idx: state for idx, state in enumerate(valid_states)}

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Add edges for each move
    moves = ['V', 'L', 'R']
    for state in valid_states:
        node_idx = state_to_index[state]
        state_list = list(state)

        for move in moves:
            new_state = apply_move(state_list, move)
            new_state_tuple = tuple(new_state)
            new_node_idx = state_to_index[new_state_tuple]

            # Add edge (undirected, so both directions will be added)
            G.add_edge(node_idx, new_node_idx, move=move)

    # Verify graph properties
    assert G.number_of_nodes() == 120, f"Expected 120 nodes, got {G.number_of_nodes()}"
    degrees = set(degree for node, degree in G.degree())
    assert degrees == {3}, f"Expected all nodes to have degree 3, got degrees: {degrees}"

    return G, state_to_index, index_to_state


def format_state_2d(state: List[int]) -> str:
    """
    Format a puzzle state as a 2D grid string.

    Args:
        state: List of 6 integers

    Returns:
        String representation of the 2x3 grid
    """
    if len(state) != 6:
        return "Invalid state"

    # Map 1D to 2D: [0,1,2,3,4,5] -> [[0,1,2],[3,4,5]]
    grid = [state[0:3], state[3:6]]

    lines = []
    for row in grid:
        row_str = ' '.join(str(x) if x != 0 else ' ' for x in row)
        lines.append(f"|{row_str}|")

    return '\n'.join(lines)


def is_winning_state(state: List[int]) -> bool:
    """
    Check if the state is the winning configuration.

    Args:
        state: Puzzle state to check

    Returns:
        True if state matches [1,2,3,4,0,5], False otherwise
    """
    return state == WINNING_STATE


def get_winning_node_index(state_to_index: Dict[Tuple[int, ...], int]) -> int:
    """
    Get the node index of the winning state.

    Args:
        state_to_index: State to index mapping

    Returns:
        Node index of winning state [1,2,3,4,0,5]
    """
    return state_to_index[tuple(WINNING_STATE)]


def generate_random_valid_state(seed: Optional[int] = None) -> List[int]:
    """
    Generate a random valid puzzle state with winning parity.

    Args:
        seed: Optional random seed for reproducibility

    Returns:
        Random valid puzzle state as list of 6 integers (winning parity only)
    """
    import random

    if seed is not None:
        random.seed(seed)

    # All states have winning parity by construction
    valid_states = generate_all_valid_states()
    return list(random.choice(valid_states))


class SlidingPuzzleGraph:
    """
    Convenience class encapsulating the sliding puzzle graph and its mappings.

    This class provides easy access to:
    - The NetworkX graph (N=120, K=3) with winning parity only
    - State<->index conversions
    - Move application
    - State formatting

    Note: This class only works with the winning parity component (120 states).
    States with wrong parity are rejected as illegal.
    """

    def __init__(self):
        """Initialize the sliding puzzle graph with winning parity component only."""
        self.G, self.state_to_index, self.index_to_state = create_sliding_puzzle_graph()
        self.N = 120
        self.K = 3
        self.winning_state = WINNING_STATE
        self.winning_node = self.state_to_index.get(tuple(WINNING_STATE), None)
        # All states have winning parity by construction
        self.winning_parity = WINNING_PARITY

    def state_to_node(self, state: List[int]) -> int:
        """Convert state to node index."""
        return state_to_node_index(state, self.state_to_index)

    def node_to_state(self, node_idx: int) -> List[int]:
        """Convert node index to state."""
        return node_index_to_state(node_idx, self.index_to_state)

    def apply_move(self, state: List[int], move: str) -> List[int]:
        """Apply a move to a state."""
        return apply_move(state, move)

    def apply_moves(self, state: List[int], moves: List[str]) -> List[int]:
        """Apply multiple moves to a state."""
        return apply_moves(state, moves)

    def format_state(self, state: List[int]) -> str:
        """Format state as 2D grid."""
        return format_state_2d(state)

    def is_winning(self, state: List[int]) -> bool:
        """Check if state is winning."""
        return is_winning_state(state)

    def get_state_parity(self, state: List[int]) -> int:
        """Get parity of a state (0 for even, 1 for odd)."""
        return count_inversions(state) % 2

    def is_valid_parity(self, state: List[int]) -> bool:
        """
        Check if state has same parity as winning state.

        Only states with the correct parity are reachable from the winning state.
        Returns True if state has winning parity, False otherwise.
        """
        return self.get_state_parity(state) == self.winning_parity

    def validate_state(self, state: List[int]) -> None:
        """
        Validate that a state is legal (correct parity).

        Raises:
            ValueError: If state is invalid or has wrong parity.
        """
        if not is_valid_puzzle_state(state):
            raise ValueError(
                f"Invalid puzzle state: {state}. "
                f"States must have hole in middle column (indices 1 or 4) "
                f"AND have {('even' if self.winning_parity == 0 else 'odd')} parity "
                f"(same as winning state {self.winning_state})."
            )

    def get_adjacency_list(self) -> List[List[int]]:
        """Get adjacency list for use with QuCoNet."""
        from .graph import graph_to_adjacency_list
        return graph_to_adjacency_list(self.G)

    def get_parity_component(self, even_parity: bool = True) -> Tuple[List[int], nx.Graph]:
        """
        Get the graph component (only winning parity component is available).

        Args:
            even_parity: Must be True (only even-parity component with winning state exists).
                        If False, raises ValueError.

        Returns:
            Tuple of (node_indices, graph) where:
            - node_indices: List of node indices [0, 1, ..., 119]
            - graph: NetworkX graph with 120 nodes

        Raises:
            ValueError: If even_parity is False (odd parity component not available).
        """
        if not even_parity:
            raise ValueError(
                "Odd parity component is not available. "
                "This module only works with the winning parity component (N=120)."
            )

        # Return all nodes and the full graph
        component_nodes = list(range(self.N))
        return component_nodes, self.G

    def get_random_state(self, seed: Optional[int] = None, valid_parity_only: bool = True) -> List[int]:
        """
        Generate a random valid state with winning parity.

        Args:
            seed: Optional random seed for reproducibility
            valid_parity_only: Deprecated parameter (always True). Kept for API compatibility.

        Returns:
            Random valid puzzle state as list of 6 integers (winning parity only)
        """
        import random

        if seed is not None:
            random.seed(seed)

        # All states have winning parity by construction
        valid_states = generate_all_valid_states()
        return list(random.choice(valid_states))
