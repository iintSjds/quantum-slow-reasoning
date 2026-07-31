#!/usr/bin/env python3
"""Distance-D random-regular QA pools for the N-scaling frustration study.

For each seed, build a fresh K-regular graph on N nodes and collect B *distinct*
(Q,A) pairs at shortest-path distance EXACTLY D -- fixing per-instance difficulty
(a length-D route within M steps) so that varying N isolates the capacity knee.
Saved as randreg_N{N}_K{K}_M{M}_B{B}_D{D}_seed{seed}.pt with the same schema as
the sliding-puzzle pools ({graph, qa_pairs, metadata}); the sweep slices first-B
train / last-num_val valid, so B must be the FULL pool size (= maxB + num_val).

Efficient: BFS shells (single_source_shortest_path_length) + set dedup -> O(B),
so B~1500 at N=960 is a few seconds (the stock rejection sampler is O(B^2)).

  python conet/gen_randreg_frust.py -N 960 -B 1600 -D 6 --seeds 1 2 3 4
"""
import argparse, os, random
import numpy as np
import networkx as nx
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quconet.graph import create_regular_graph, save_graph_with_qa_pairs


def dist_pairs(G, D, B, rng):
    """B distinct (s,t) at shortest-path distance exactly D, via BFS shells."""
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    seen, pairs = set(), []
    for s in nodes:
        lengths = nx.single_source_shortest_path_length(G, s, cutoff=D)
        shell = [t for t, d in lengths.items() if d == D]
        rng.shuffle(shell)
        for t in shell:
            key = (s, t) if s < t else (t, s)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((int(s), int(t)))
            if len(pairs) >= B:
                return pairs
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("-K", type=int, default=3)
    ap.add_argument("-M", type=int, default=8)
    ap.add_argument("-B", type=int, required=True, help="FULL pool size = maxB + num_val")
    ap.add_argument("-D", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--output_dir", default="from4090/expr4/graph_qa")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    for seed in a.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = random.Random(seed)
        G = create_regular_graph(a.N, a.K, "random")
        pairs = dist_pairs(G, a.D, a.B, rng)
        fn = f"randreg_N{a.N}_K{a.K}_M{a.M}_B{a.B}_D{a.D}_seed{seed}.pt"
        fp = os.path.join(a.output_dir, fn)
        save_graph_with_qa_pairs(G, pairs, fp)
        tag = "OK " if len(pairs) >= a.B else "SHORT"
        print(f"  [{tag}] {fn}: {len(pairs)}/{a.B} dist-{a.D} pairs, diam {nx.diameter(G)}")


if __name__ == "__main__":
    main()
