"""
QuCoNet - Quantum Concept Network

A PyTorch implementation of quantum walks on graphs for reinforcement learning-based graph traversal.
"""

__version__ = "0.1.0"
__author__ = "Xiansheng Cai"

from .modules import CoinOperator, ShiftOperator, QuantumCoNet
from .operators_ar import CoinOperatorAR, ShiftOperatorAR
from .quconet_ar import QuantumCoNetAR
from .trainer import QuCoNetTrainer, create_default_config, load_config_from_file, save_config_to_file
from .visualization import (
    plot_quantum_state_on_graph,
    plot_quantum_state_comparison,
    plot_probability_distribution,
    create_quantum_walk_animation,
    animate_quantum_walk_steps,
    extract_node_probabilities
)
from .slidingpuzzle import (
    SlidingPuzzleGraph,
    create_sliding_puzzle_graph,
    apply_move,
    apply_moves,
    is_valid_puzzle_state,
    is_winning_state,
    WINNING_STATE,
    format_state_2d,
    generate_random_valid_state,
    count_inversions,
    state_to_node_index,
    node_index_to_state
)

__all__ = [
    "CoinOperator",
    "ShiftOperator",
    "QuantumCoNet",
    "QuCoNetTrainer",
    "create_default_config",
    "load_config_from_file",
    "save_config_to_file",
    # Visualization functions
    "plot_quantum_state_on_graph",
    "plot_quantum_state_comparison",
    "plot_probability_distribution",
    "create_quantum_walk_animation",
    "animate_quantum_walk_steps",
    "extract_node_probabilities",
    # Sliding puzzle
    "SlidingPuzzleGraph",
    "create_sliding_puzzle_graph",
    "apply_move",
    "apply_moves",
    "is_valid_puzzle_state",
    "is_winning_state",
    "WINNING_STATE",
    "format_state_2d",
    "generate_random_valid_state",
    "count_inversions",
    "state_to_node_index",
    "node_index_to_state",
]