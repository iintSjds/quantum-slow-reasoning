#!/usr/bin/env python3
"""
Generate graph_qa.pt files for QuCoNet experiments.

This script creates a random regular graph and finds QA pairs with specified distance constraints,
then saves them to a .pt file for use in training experiments.

Usage:
    python generate_graph_qa.py -N 80 -K 3 -M 8 -B 96 --path_min 4 --path_max 6
    python generate_graph_qa.py -N 240 -K 3 -M 8 -B 96 --path_min 4 --path_max 6
    python generate_graph_qa.py -N 720 -K 3 -M 8 -B 96 --path_min 4 --path_max 6

Arguments:
    -N: Number of nodes in the graph
    -K: Degree of each node (K-regular graph)
    -M: Maximum steps per episode (used in filename, not for graph generation)
    -B: Number of QA pairs to generate
    --path_min: Minimum path distance between Q and A
    --path_max: Maximum path distance between Q and A
    --seed: Random seed for reproducibility (default: 42)
    --output_dir: Output directory (default: current directory)
"""

import argparse
import sys
import os
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quconet.graph import (
    create_regular_graph,
    graph_to_adjacency_list,
    analyze_graph_properties,
    find_qa_pairs_with_distance,
    save_graph_with_qa_pairs
)
import networkx as nx
import torch
import numpy as np


def generate_graph_qa(N, K, M, B, path_min, path_max, seed=42, output_dir='.'):
    """Generate a graph with QA pairs and save to .pt file.
    
    Args:
        N: Number of nodes
        K: Degree of each node
        M: Maximum steps (for filename)
        B: Number of QA pairs
        path_min: Minimum path distance
        path_max: Maximum path distance
        seed: Random seed
        output_dir: Output directory
    
    Returns:
        Path to saved file
    """
    print("="*70)
    print("GRAPH & QA PAIR GENERATOR")
    print("="*70)
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n1. Creating {N}-node {K}-regular graph...")
    G = create_regular_graph(N, K, "random")
    
    # Analyze graph
    props = analyze_graph_properties(G)
    print(f"\n2. Graph properties:")
    print(f"   - Nodes: {props['num_nodes']}")
    print(f"   - Edges: {props['num_edges']}")
    print(f"   - Degree: {props['degree']}")
    print(f"   - Diameter: {props['diameter']}")
    print(f"   - Connected: {props['is_connected']}")
    
    # Find QA pairs
    print(f"\n3. Finding {B} QA pairs with distance in [{path_min}, {path_max}]...")
    qa_pairs = find_qa_pairs_with_distance(G, batch_size=B, min_distance=path_min, max_distance=path_max)
    
    if len(qa_pairs) < B:
        print(f"   Warning: Only found {len(qa_pairs)} QA pairs (requested {B})")
        print(f"   Consider increasing path range or using a larger graph")
    
    print(f"   Found {len(qa_pairs)} QA pairs:")
    for i, (q, a) in enumerate(qa_pairs[:10]):  # Show first 10
        distance = nx.shortest_path_length(G, q, a)
        print(f"   - Pair {i+1}: Q={q} → A={a} (distance: {distance})")
    
    if len(qa_pairs) > 10:
        print(f"   ... and {len(qa_pairs) - 10} more")
    
    # Create filename
    # Pattern: quconet_ar_demo_N{N}_K{K}_M{M}_B{B}_Q{start}_A{target}_graph_qa.pt
    first_q, first_a = qa_pairs[0]
    filename = f"quconet_ar_demo_N{N}_K{K}_M{M}_B{B}_Q{first_q}_A{first_a}_graph_qa.pt"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Save graph and QA pairs
    print(f"\n4. Saving to {filepath}...")
    save_graph_with_qa_pairs(G, qa_pairs, filepath)
    
    # Verify save
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath) / 1024  # KB
        print(f"   ✓ Saved successfully ({file_size:.1f} KB)")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated file: {filepath}")
    print(f"\nYou can now use this file for training:")
    print(f"  python real_world_demo_ar.py -f {filepath} -B {B}")
    
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description='Generate graph_qa.pt files for QuCoNet experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for N=80 with path distance 4-6
  python generate_graph_qa.py -N 80 -K 3 -M 8 -B 96 --path_min 4 --path_max 6
  
  # Generate for N=240 with custom seed
  python generate_graph_qa.py -N 240 -K 3 -M 8 -B 96 --path_min 4 --path_max 6 --seed 123
  
  # Generate for N=720 and save to specific directory
  python generate_graph_qa.py -N 720 -K 3 -M 8 -B 96 --path_min 4 --path_max 6 --output_dir ./graphs
        """
    )
    
    parser.add_argument('-N', type=int, required=True,
                        help='Number of nodes in the graph (e.g., 80, 240, 720)')
    parser.add_argument('-K', type=int, default=3,
                        help='Degree of each node (default: 3)')
    parser.add_argument('-M', type=int, default=8,
                        help='Maximum steps per episode (default: 8)')
    parser.add_argument('-B', type=int, required=True,
                        help='Number of QA pairs to generate')
    parser.add_argument('--path_min', type=int, required=True,
                        help='Minimum path distance between Q and A')
    parser.add_argument('--path_max', type=int, required=True,
                        help='Maximum path distance between Q and A')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.N <= 0:
        parser.error("N must be positive")
    if args.K <= 0:
        parser.error("K must be positive")
    if args.B <= 0:
        parser.error("B must be positive")
    if args.path_min > args.path_max:
        parser.error("path_min must be <= path_max")
    if args.path_min < 1:
        parser.error("path_min must be >= 1")
    
    generate_graph_qa(
        N=args.N,
        K=args.K,
        M=args.M,
        B=args.B,
        path_min=args.path_min,
        path_max=args.path_max,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
