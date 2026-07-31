#!/usr/bin/env python3
"""Extend the distance-6 sliding-puzzle capacity pools to a larger max-B while
preserving BOTH the training prefix and the held-out suffix, so large-B runs
stitch to the cached B<=704 sweep and share one fixed held-out set.

The existing pool ``sliding_puzzle_N120_K3_M8_B768_D6_seed{s}.pt`` is a list of
768 distance-6 (Q,A) pairs; the capacity sweep trains on the first B and
evaluates on the last ``--num-val`` (=64). To grow max-B without disturbing
either end we splice new distance-6 pairs *between* the train region and the
held-out tail:

    new = old[:768-num_val] + [maxtrain-(768-num_val) fresh pairs] + old[-num_val:]

so train=first B is identical to the old pool for B<=768-num_val, the last 64
held-out pairs are byte-identical, and B can now run up to maxtrain.

    python conet/extend_sliding_pool.py --maxtrain 1216 --num-val 64 --seeds 1 2 3 4
"""
import argparse, os, copy, random
import torch
import networkx as nx


def all_dist_pairs(G, D):
    """All unordered node pairs at graph distance exactly D, as (min,max) keys."""
    keys = set()
    for s in G.nodes():
        for t, d in nx.single_source_shortest_path_length(G, s, cutoff=D).items():
            if d == D and s != t:
                keys.add((s, t) if s < t else (t, s))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-dir", default="from4090/expr4/graph_qa")
    ap.add_argument("--in-prefix", default="sliding_puzzle_N120_K3_M8_B768_D6")
    ap.add_argument("--maxtrain", type=int, default=1216,
                    help="largest training-set size B the extended pool must support")
    ap.add_argument("--num-val", type=int, default=64)
    ap.add_argument("--D", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    a = ap.parse_args()

    total = a.maxtrain + a.num_val
    out_prefix = f"sliding_puzzle_N120_K3_M8_B{total}_D6"
    for s in a.seeds:
        src = os.path.join(a.qa_dir, f"{a.in_prefix}_seed{s}.pt")
        d = torch.load(src, map_location="cpu", weights_only=False)
        G = d["graph"]
        old = [(int(q), int(a_)) for q, a_ in d["qa_pairs"]]
        assert len(old) >= a.num_val, f"pool {src} smaller than num_val"

        old_keys = {(q, a_) if q < a_ else (a_, q) for q, a_ in old}
        assert len(old_keys) == len(old), "old pool has duplicate unordered pairs"

        # verify every old pair is at distance D
        bad = [(q, a_) for q, a_ in old if nx.shortest_path_length(G, q, a_) != a.D]
        assert not bad, f"{len(bad)} old pairs not at distance {a.D}"

        train_head = old[:len(old) - a.num_val]        # old[:704]
        held_out   = old[len(old) - a.num_val:]         # old[704:768] (fixed valid set)
        need = a.maxtrain - len(train_head)             # fresh distance-D pairs to add
        assert need >= 0, "maxtrain smaller than existing train region"

        pool_keys = all_dist_pairs(G, a.D)
        avail = sorted(pool_keys - old_keys)            # deterministic order before shuffle
        rng = random.Random(1000 + s)
        rng.shuffle(avail)
        if need > len(avail):
            raise SystemExit(f"seed {s}: need {need} new dist-{a.D} pairs but only "
                             f"{len(avail)} available (total dist-{a.D} = {len(pool_keys)})")
        fresh = [tuple(rng.sample([u, v], 2)) for (u, v) in avail[:need]]  # random direction

        new_qa = train_head + fresh + held_out
        assert len(new_qa) == total
        # integrity: distinct, all distance-D, prefix + held-out preserved
        nk = {(q, a_) if q < a_ else (a_, q) for q, a_ in new_qa}
        assert len(nk) == total, "extended pool has duplicates"
        assert new_qa[:len(train_head)] == train_head, "train prefix changed"
        assert new_qa[-a.num_val:] == held_out, "held-out tail changed"

        out = copy.deepcopy(d)
        out["qa_pairs"] = [tuple(p) for p in new_qa]
        meta = out.get("metadata", {}) or {}
        meta = dict(meta); meta["B"] = total; meta["extended_from"] = os.path.basename(src)
        meta["maxtrain"] = a.maxtrain; meta["num_val_reserved"] = a.num_val
        out["metadata"] = meta
        dst = os.path.join(a.qa_dir, f"{out_prefix}_seed{s}.pt")
        torch.save(out, dst)
        print(f"seed {s}: {len(old)} -> {total} pairs "
              f"(+{need} fresh dist-{a.D}; {len(avail)} were available) -> {dst}")


if __name__ == "__main__":
    main()
