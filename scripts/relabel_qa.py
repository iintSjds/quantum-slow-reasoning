#!/usr/bin/env python3
"""Build node-relabeled copies of sliding-puzzle QA pools.

The question channel encodes node LABELS; a uniformly random relabeling
preserves the graph and the tasks but scrambles every label digit.  If the
trained accuracy depends only on graph structure, training on relabeled
pools must reproduce the native results within seed spread.

Usage (from repo root):
    python docs/discussion/scripts/relabel_qa.py --seeds 1-8 --perms 2 \
        --out archive/expr4/graph_qa_relabel
Writes <prefix>_seed<S>_perm<P>.pt with metadata recording the permutation.
"""
import argparse, os
import networkx as nx
import numpy as np
import torch

PREFIX = "sliding_puzzle_N120_K3_M8_B192_D6"


def relabel(src, perm_seed):
    d = torch.load(src, map_location="cpu", weights_only=False)
    g, qa = d["graph"], d["qa_pairs"]
    nodes = sorted(g.nodes())
    rng = np.random.default_rng(perm_seed)
    perm = rng.permutation(len(nodes))
    mapping = {nodes[i]: int(perm[i]) for i in range(len(nodes))}
    g2 = nx.relabel_nodes(g, mapping, copy=True)
    qa2 = [(mapping[int(q)], mapping[int(a)]) for q, a in qa]
    meta = dict(d.get("metadata", {}))
    meta["relabel_perm_seed"] = perm_seed
    meta["relabel_mapping"] = [mapping[n] for n in nodes]
    return {"graph": g2, "qa_pairs": qa2, "metadata": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-dir", default="archive/expr4/graph_qa")
    ap.add_argument("--out", default="archive/expr4/graph_qa_relabel")
    ap.add_argument("--seeds", default="1-8")
    ap.add_argument("--perms", type=int, default=2)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    os.makedirs(a.out, exist_ok=True)
    for s in range(lo, hi + 1):
        src = os.path.join(a.qa_dir, f"{PREFIX}_seed{s}.pt")
        for p in range(1, a.perms + 1):
            dst = os.path.join(a.out, f"{PREFIX}_seed{s}_perm{p}.pt")
            torch.save(relabel(src, 1000 * s + p), dst)
            print("wrote", dst)


if __name__ == "__main__":
    main()
