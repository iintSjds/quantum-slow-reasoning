"""
Visualization utilities for QuCoNet quantum walks.

This module provides functions to visualize quantum states on graphs,
including static plots and animations of quantum state evolution.
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional, Tuple, Dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import os
import warnings

# Suppress warnings from matplotlib animations
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.animation')


def extract_node_probabilities(psi: torch.Tensor) -> np.ndarray:
    """
    Extract node probabilities from quantum state vector.

    Args:
        psi: Quantum state tensor of shape:
             - Legacy: (N, K), (batch, N, K), or (N*K,) flattened
             - AR mode: (B, N, K, K, K, ..., K) with M coin dimensions

    Returns:
        node_probs: Node probabilities of shape:
                    - AR mode: (B, N) - maintains batch dimension
                    - Legacy: (N,) for single state, (batch, N) for batched
    """
    # Handle AR mode: sum over all coin dimensions
    if psi.dim() > 3:
        # AR mode: (B, N, K, K, K, ..., K)
        # Sum over all K dimensions (all dimensions except batch and position)
        sum_dims = list(range(2, psi.dim()))  # All dimensions except 0 (batch) and 1 (N)
        node_probs = torch.sum(torch.abs(psi)**2, dim=sum_dims)
        # Result: (B, N) - keep batch dimension
    elif psi.dim() == 3:  # (batch, N, K)
        node_probs = torch.sum(torch.abs(psi)**2, dim=-1)
        # Optionally squeeze if batch=1 for backward compatibility
        if psi.shape[0] == 1:
            node_probs = node_probs[0]
    elif psi.dim() == 2:  # (N, K) or possibly flattened (N*K,)
        if psi.shape[1] == 1:  # Edge case: (N, 1)
            node_probs = torch.abs(psi[:, 0])**2
        else:  # Normal case: (N, K) - sum over K dimension
            node_probs = torch.sum(torch.abs(psi)**2, dim=-1)
    elif psi.dim() == 1:  # Flattened (N*K,)
        # Try to infer dimensions - assume it's a flattened (N, K) state
        total_dim = psi.shape[0]
        # For quantum walks, K is typically small (2-5), so find N and K
        # such that N*K = total_dim and the state can be reshaped
        # Heuristic: assume K <= sqrt(total_dim)
        K_guess = min(5, int(torch.sqrt(torch.tensor(total_dim)).item()))
        if total_dim % K_guess == 0:
            N = total_dim // K_guess
            psi_reshaped = psi.view(N, K_guess)
            node_probs = torch.sum(torch.abs(psi_reshaped)**2, dim=-1)
        else:
            raise ValueError(f"Cannot infer valid (N, K) dimensions from flattened state of size {total_dim}")
    else:
        raise ValueError(f"Unexpected psi shape: {psi.shape}")

    # Convert to numpy for return
    return node_probs.detach().cpu().numpy()


def plot_quantum_state_on_graph(
    ax: Axes,
    model,
    psi: torch.Tensor,
    node_size: int = 500,
    cmap: str = 'gray_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    with_labels: bool = True,
    font_size: int = 10,
    font_color: str = 'blue',
    edge_color: str = 'gray',
    edge_width: float = 1.0,
    title: Optional[str] = None,
    colorbar: bool = True,
    pos: Optional[Dict[int, np.ndarray]] = None,
    node_edge_color: str = 'black',
    node_edge_width: float = 2.0
) -> Dict[int, np.ndarray]:
    """
    Plot a quantum state on a graph with node colors representing probabilities.

    Args:
        ax: Matplotlib axis to plot on
        model: QuCoNet model containing graph structure
        psi: Quantum state vector
        node_size: Size of nodes in plot
        cmap: Colormap for node colors (default: 'gray_r' - reversed grayscale)
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        with_labels: Whether to show node labels
        font_size: Size of node labels
        font_color: Color of node labels
        edge_color: Color of edges
        edge_width: Width of edges
        title: Optional title for the plot
        colorbar: Whether to show colorbar
        pos: Optional pre-computed node positions
        node_edge_color: Color of node boundaries
        node_edge_width: Width of node boundaries

    Returns:
        pos: Node positions dictionary

    Note:
        The default 'gray_r' colormap maps:
        - Black (0) = Maximum probability (1.0)
        - White (1) = Minimum probability (0.0)
        This provides better visibility with black node boundaries.
    """
    # Extract node probabilities
    node_probs = extract_node_probabilities(psi)

    # Ensure 1D array
    if node_probs.ndim > 1:
        node_probs = node_probs.flatten()

    # Get graph from model
    N = model.N
    adj_list = model.shift.adjacency_list if hasattr(model.shift, 'adjacency_list') else None

    if adj_list is None:
        # Reconstruct graph from shift maps
        node_map = model.shift.shift_node_map.cpu()
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(model.K):
                neighbor = node_map[i, j].item()
                G.add_edge(i, neighbor)
    else:
        # Build graph from adjacency list
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for node, neighbors in enumerate(adj_list):
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

    # Compute node positions if not provided
    if pos is None:
        pos = nx.spring_layout(G, k=5/np.sqrt(N), iterations=100, seed=42)

    # Normalize probabilities for color mapping
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.max(node_probs) if np.max(node_probs) > 0 else 1

    # Create node colors
    node_colors = node_probs

    # Draw the graph
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=edge_width, alpha=0.5)

    # Draw nodes with colors based on probabilities
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors=node_edge_color,
        linewidths=node_edge_width
    )

    # Add node labels
    if with_labels:
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=font_size,
            font_color=font_color,
            font_weight='bold'
        )

    # Add colorbar
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Probability')

    # Set title
    if title is not None:
        ax.set_title(title)

    # Remove axes
    ax.set_axis_off()

    return pos


def create_quantum_walk_animation(
    model,
    trajectory: List[torch.Tensor],
    filename: str = 'quantum_walk_evolution.gif',
    fps: int = 2,
    dpi: int = 100,
    figsize: Tuple[int, int] = (8, 8),
    colorbar: bool = True,
    **plot_kwargs
) -> str:
    """
    Create an animated GIF showing quantum state evolution on a graph.

    Args:
        model: QuCoNet model
        trajectory: List of quantum states at each time step
        filename: Output filename for the GIF
        fps: Frames per second for the animation
        dpi: Resolution of the animation
        figsize: Figure size (width, height)
        colorbar: Whether to show colorbar (added only once)
        **plot_kwargs: Additional arguments for plot_quantum_state_on_graph

    Returns:
        Path to the created GIF file
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Extract all probabilities to determine color scale
    all_probs = []
    for psi in trajectory:
        probs = extract_node_probabilities(psi)
        if probs.ndim > 1 and probs.shape[0] == 1:
            probs = probs[0]
        all_probs.append(probs)

    all_probs = np.array(all_probs)
    vmin = 0
    vmax = np.max(all_probs) if np.max(all_probs) > 0 else 1

    # Get initial positions
    pos = None

    # Animation update function
    def update(frame):
        ax.clear()
        psi = trajectory[frame]
        title = f'Quantum Walk - Step {frame}/{len(trajectory)-1}'
        nonlocal pos
        pos = plot_quantum_state_on_graph(
            ax, model, psi,
            title=title,
            vmin=vmin,
            vmax=vmax,
            pos=pos,
            colorbar=False,  # Don't add colorbar in each frame
            **plot_kwargs
        )

    # Create animation
    anim = animation.FuncAnimation(
        fig, update,
        frames=len(trajectory),
        interval=1000//fps,
        blit=False,
        repeat=True
    )

    # Add colorbar only once after animation is created
    if colorbar:
        # Create a separate axis for colorbar
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=plot_kwargs.get('cmap', 'gray_r'),
                                  norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Probability')

    # Save animation
    filepath = os.path.join(os.getcwd(), filename)
    anim.save(filepath, writer='pillow', dpi=dpi, fps=fps)
    plt.close(fig)

    return filepath


def plot_quantum_state_comparison(
    model,
    states: List[torch.Tensor],
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    **plot_kwargs
) -> Figure:
    """
    Plot multiple quantum states side by side for comparison.

    Args:
        model: QuCoNet model
        states: List of quantum states to compare
        labels: Optional labels for each state
        figsize: Figure size (width, height)
        **plot_kwargs: Additional arguments for plotting

    Returns:
        matplotlib Figure object
    """
    n_states = len(states)
    fig, axes = plt.subplots(1, n_states, figsize=figsize)

    if n_states == 1:
        axes = [axes]

    # Get global color scale
    all_probs = []
    for psi in states:
        probs = extract_node_probabilities(psi)
        if probs.ndim > 1 and probs.shape[0] == 1:
            probs = probs[0]
        all_probs.append(probs)

    all_probs = np.array(all_probs)
    vmin = 0
    vmax = np.max(all_probs) if np.max(all_probs) > 0 else 1

    # Get positions from first plot
    pos = None

    # Plot each state
    for i, (psi, ax) in enumerate(zip(states, axes)):
        label = labels[i] if labels else f'State {i}'
        pos = plot_quantum_state_on_graph(
            ax, model, psi,
            title=label,
            vmin=vmin,
            vmax=vmax,
            pos=pos,
            colorbar=(i == n_states - 1),  # Only last subplot gets colorbar
            **plot_kwargs
        )

    plt.tight_layout()
    return fig


def plot_probability_distribution(
    psi: torch.Tensor,
    figsize: Tuple[int, int] = (10, 6),
    show_top_k: int = 10
) -> Figure:
    """
    Plot probability distribution as a bar chart.

    Args:
        model: QuCoNet model
        psi: Quantum state vector
        figsize: Figure size (width, height)
        show_top_k: Number of top probability nodes to highlight

    Returns:
        matplotlib Figure object
    """
    # Extract probabilities
    node_probs = extract_node_probabilities(psi)
    if node_probs.ndim > 1 and node_probs.shape[0] == 1:
        node_probs = node_probs[0]

    N = len(node_probs)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Plot all probabilities
    colors = plt.cm.hot(node_probs / np.max(node_probs))
    bars = ax1.bar(range(N), node_probs, color=colors)

    # Highlight top K nodes
    if show_top_k > 0:
        top_k_indices = np.argsort(node_probs)[-show_top_k:][::-1]
        for i, idx in enumerate(top_k_indices):
            bars[idx].set_edgecolor('red')
            bars[idx].set_linewidth(2)
            ax1.annotate(f'{i+1}', xy=(idx, node_probs[idx]),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold')

    ax1.set_xlabel('Node')
    ax1.set_ylabel('Probability')
    ax1.set_title('Quantum State Probability Distribution')
    ax1.set_xlim(-0.5, N - 0.5)

    # Plot top K probabilities separately
    if show_top_k > 0 and len(top_k_indices) > 0:
        top_k_probs = node_probs[top_k_indices]
        ax2.bar(range(len(top_k_indices)), top_k_probs,
                color='red', alpha=0.7)
        ax2.set_xlabel(f'Top {show_top_k} Nodes (Ranked)')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Top {show_top_k} Highest Probability Nodes')

        # Add value labels
        for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
            ax2.annotate(f'{prob:.3f}', xy=(i, prob),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom')

    plt.tight_layout()
    return fig


def animate_quantum_walk_steps(
    model,
    start_node: int,
    target_node: int,
    max_steps: int = 20,
    initial_state: str = 'uniform',
    filename: Optional[str] = None,
    **animation_kwargs
) -> str:
    """
    Run a quantum walk and create an animation of the state evolution.

    Args:
        model: QuCoNet model
        start_node: Starting node
        target_node: Target node
        max_steps: Maximum number of steps
        initial_state: Type of initial state ('uniform', 'random', 'basis')
        filename: Output filename (auto-generated if None)
        **animation_kwargs: Arguments for create_quantum_walk_animation

    Returns:
        Path to the created animation file
    """
    # Create initial state
    initial_psi = model.generate_batched_initial_states([start_node], initial_state)

    # Track state evolution
    trajectory = [initial_psi.clone()]
    psi = initial_psi.clone()

    # Run quantum walk step by step
    for _ in range(max_steps):
        # Perform quantum step
        psi = model.quantum_step(psi)

        # Measure and project (but keep the full state)
        _, psi_projected = model.measure_and_project(psi,
            torch.tensor([target_node], device=psi.device))

        # Store the projected state
        trajectory.append(psi_projected.clone())
        psi = psi_projected

    # Create animation
    if filename is None:
        filename = f'quantum_walk_{start_node}_to_{target_node}.gif'

    filepath = create_quantum_walk_animation(
        model, trajectory, filename, **animation_kwargs
    )

    return filepath