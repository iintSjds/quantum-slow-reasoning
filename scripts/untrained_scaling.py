#!/usr/bin/env python3
"""How the untrained (uniform) walker's held-out convertibility scales with
graph size and task difficulty, versus trained systems.

Context: on the D6/M8 puzzle at N=120 the untrained walker
under blind amplification tops every trained system held-out.  Question:
is that a small-graph artifact?  Two axes:

A. Size (random-regular N=120/240/480/960, B=32): uniform p on the exact
   archived held-out pairs, converted blind at n, vs the cached trained
   families (grover_rr = native n=1; cbestk_exact_rr = classical) and a
   freshly trained two-sided imported-target control at N=480.
B. Difficulty (puzzle, all ordered pairs at graph distance d): uniform p
   falls with d; at d=7 (M=9) compare with the archived transformer-coin
   g2/g3 runs.

Run:  python scripts/untrained_scaling.py
"""
import os, sys, json, glob
from collections import deque
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twosided_target_control import exact_success_probs, blind, bok, train_twosided

HERE = os.path.dirname(os.path.abspath(__file__))
DATAROOT = os.environ.get("QSR_ROOT", os.path.abspath(os.path.join(HERE, "..")))
F4090 = os.path.join(DATAROOT, "from4090")
OUT = os.path.join(HERE, "_sweep_out")

RR_DIRS = {120: "grover_sweep_rr", 240: "grover_sweep_rr_N240",
           480: "grover_sweep_rr_N480", 960: "grover_sweep_rr_N960"}
RR_CACHE = {120: "p_cache_rr.json", 240: "p_cache_rr_N240.json",
            480: "p_cache_rr_N480.json", 960: "p_cache_rr_N960.json"}
M, B = 8, 32


def rr_seed_assets(N, seed):
    for fam in ("cbestkX2", "cstdX1"):
        hits = glob.glob(f"{F4090}/{RR_DIRS[N]}/conet_adam_randreg_N{N}_*_seed{seed}_B{B}_*/"
                         f"{fam}_s{seed}_B{B}_*results.json")
        if hits:
            d = json.load(open(hits[0]))
            sd = torch.load(hits[0].replace("_results.json", "_best_model.pt"),
                            map_location="cpu", weights_only=False)["model_state_dict"]
            return sd["node_neighbors"].long(), \
                [tuple(q) for q in d["train_qa"]], [tuple(q) for q in d["valid_qa"]]
    return None


def uniform_p(nbr, qa, horizon=M):
    N = nbr.shape[0]
    probs = torch.full((N, 3), 1.0 / 3.0)
    with torch.no_grad():
        return exact_success_probs(probs, nbr, qa, horizon).numpy()


def cached(N, fam, split):
    c = json.load(open(os.path.join(OUT, RR_CACHE[N])))
    return [np.asarray(v) for k, v in c.items()
            if k.split("|")[0] == fam and k.split("|")[2] == str(B)
            and k.split("|")[3] == split]


def part_A():
    print("=" * 76)
    print("A. size axis (random-regular, B=32, held-out): blind conversion at n")
    print("=" * 76)
    rows = {}
    for N in (120, 240, 480, 960):
        assets = [rr_seed_assets(N, s) for s in range(1, 9)]
        assets = [a for a in assets if a]
        if not assets:
            print(f"N={N}: no archived runs found, skipped")
            continue
        uva = [uniform_p(nbr, va) for nbr, _, va in assets]
        fams = sorted({k.split("|")[0] for k in
                       json.load(open(os.path.join(OUT, RR_CACHE[N])))})
        line = {"seeds": len(assets), "uniform mean p": np.mean([u.mean() for u in uva])}
        for n in (1, 2, 3, 4):
            line[f"untrained G{n}"] = np.mean([blind(u, n).mean() for u in uva])
        for fam, lab in (("grover_rr", "native g1 blind1"),):
            if fam in fams:
                line[lab] = np.mean([blind(p, 1).mean() for p in cached(N, fam, "valid")])
        for fam, lab in (("cbestk_exact_rr", "classical bo3"),):
            if fam in fams:
                line[lab] = np.mean([bok(p, 3).mean() for p in cached(N, fam, "valid")])
        # deep native ladder exists only at N=120
        for nn, fam in ((2, "grover2_rr"), (3, "grover3_rr"), (4, "grover4_rr")):
            if fam in fams:
                line[f"native g{nn} blind{nn}"] = np.mean(
                    [blind(p, nn).mean() for p in cached(N, fam, "valid")])
        rows[N] = line
        print(f"\nN={N}  (families cached: {fams})")
        for k, v in line.items():
            print(f"  {k:>18}: {v:.4f}" if isinstance(v, float) else f"  {k:>18}: {v}")
    # two-sided control at N=480, c=p*(1), 8 seeds
    print("\n-- two-sided imported-target control at N=480 (c=0.25, 8 seeds) --")
    tr_h, va_h = [], []
    for s in range(1, 9):
        a = rr_seed_assets(480, s)
        if not a:
            continue
        nbr, tq, vq = a
        logits = train_twosided(nbr, tq, 0.25)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            tr_h.append(exact_success_probs(probs, nbr, tq, M).numpy())
            va_h.append(exact_success_probs(probs, nbr, vq, M).numpy())
    if tr_h:
        print(f"  train mean p {np.mean([t.mean() for t in tr_h]):.4f}; "
              f"train blind1 {np.mean([blind(t,1).mean() for t in tr_h]):.3f}; "
              f"heldout mean p {np.mean([v.mean() for v in va_h]):.4f}; "
              f"heldout blind1 {np.mean([blind(v,1).mean() for v in va_h]):.3f}")
    return rows


def bfs_distances(nbr):
    N = nbr.shape[0]
    adj = [set(nbr[i].tolist()) for i in range(N)]
    D = np.full((N, N), -1, dtype=int)
    for s in range(N):
        D[s, s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if D[s, y] < 0:
                    D[s, y] = D[s, x] + 1
                    q.append(y)
    return D


def tcoin_family(tag, n):
    vals = []
    for f in sorted(glob.glob(f"{F4090}/tcoin_merged/results/tcoin_{tag}_s*_B32_history.json")):
        h = json.load(open(f))
        fin = h["history"][-1]
        if fin["epoch"] != h["args"]["epochs"]:
            continue
        vals.append(blind(np.array(fin["valid"]["p"]), n).mean())
    return np.array(vals)


def part_B():
    print("\n" + "=" * 76)
    print("B. difficulty axis (puzzle, uniform walker over the full distance class)")
    print("=" * 76)
    rj = glob.glob(f"{F4090}/grover_sweep/conet_adam_sliding_puzzle_N120*_seed1_B{B}_*/"
                   f"ccapX0.25_s1_B{B}_*results.json")[0]
    sd = torch.load(rj.replace("_results.json", "_best_model.pt"),
                    map_location="cpu", weights_only=False)["model_state_dict"]
    nbr = sd["node_neighbors"].long()
    D = bfs_distances(nbr)
    for d, horizon in ((3, 8), (4, 8), (5, 8), (6, 8), (7, 9)):
        pairs = [(int(q), int(a)) for q in range(120) for a in range(120) if D[q, a] == d]
        p = uniform_p(nbr, pairs, horizon=horizon)
        gs = "  ".join(f"G{n}={blind(p, n).mean():.3f}" for n in (1, 2, 3, 4))
        print(f"  d={d} (M={horizon}, {len(pairs)} pairs): uniform mean p={p.mean():.5f}  {gs}")
    print("\n  archived transformer-coin on d=7/M=9 (8 seeds, blind at intended n):")
    for tag, n in (("d7m9_g2", 2), ("d7m9_g3", 3)):
        v = tcoin_family(tag, n)
        if len(v):
            print(f"    trained {tag}: heldout blind{n} = {v.mean():.3f}±{v.std():.3f}")


if __name__ == "__main__":
    part_A()
    part_B()
