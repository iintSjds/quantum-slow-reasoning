"""
D3: trained-network (success-path flow) visualization, qstd vs grover.

For selected train QA pairs of one (seed, B), enumerate ALL success paths of
the AR walk (coin input at step s = the pair's unique coin pattern digit) and
draw the flow on the sliding-puzzle graph: edge width/opacity ~ summed success-
path probability through that edge.  Rows = model family, cols = QA pairs.

Pair selection uses the existing p-cache (grover_sweep_analysis) to find the
telling cases: one pair the one-shot model SOLVED (p~1), one it left DEAD
(p~0) that grover parked at p*~0.25, and the pair with median grover p.

Run from repo root:
  python scripts/grover_network_plot.py --seed 1 --B 32
"""
import os, sys, json, argparse
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amplification_scaling as amp
import amplification_step1 as s1
import grover_sweep_analysis as gsa

FAMS = [("qstd", "one-shot trained"), ("grover", "grover-1 trained")]

def success_path_probs(apd, cm, snm, scm, Q, A, N, K, M, thr=1e-9):
    """All success paths with probs (same walk convention as the enumerator)."""
    initial_coins = apd.generate_unique_coin_state(N, K, Q, A, max_length=M)
    coin_probs = (cm.abs() ** 2).numpy()
    snm_np = snm.numpy()
    paths = Counter()
    stack = [(Q, 0, (Q,), 1.0)]
    while stack:
        node, step, path, cp = stack.pop()
        if node == A:
            paths[path] += cp
            continue
        if step >= M:
            continue
        cin = initial_coins[step]
        for k in range(K):
            npb = cp * coin_probs[node, cin, k]
            if npb < thr:
                continue
            stack.append((int(snm_np[node, k]), step + 1,
                          path + (int(snm_np[node, k]),), npb))
    return paths

def edge_flow(paths):
    flow = Counter()
    for path, p in paths.items():
        for u, v in zip(path[:-1], path[1:]):
            flow[(u, v)] += p
    return flow

def pick_pairs(cache_path, seed, B):
    """(qstd-solved, qstd-dead-grover-rescued, median-grover-p) indices."""
    cache = json.load(open(cache_path))
    pq = np.asarray(cache[f"qstd|{seed}|{B}|train"])
    pg = np.asarray(cache[f"grover|{seed}|{B}|train"])
    solved = int(np.argmax(pq))
    dead = np.where(pq < 0.02)[0]
    rescued = int(dead[np.argmax(pg[dead])]) if len(dead) else int(np.argmin(pq))
    med = int(np.argsort(np.abs(pg - np.median(pg)))[0])
    idxs, seen = [], set()
    for i in (solved, rescued, med):
        if i not in seen:
            idxs.append(i); seen.add(i)
    return idxs, pq, pg

def classical_success_paths(tp, nbr, Q, A, M, thr=1e-9):
    """All success paths of the classical softmax walk (same absorbing
    convention as the exact evaluator: absorb on first arrival, <=M steps)."""
    paths = Counter()
    stack = [(Q, 0, (Q,), 1.0)]
    K = nbr.shape[1]
    while stack:
        node, step, path, cp = stack.pop()
        if node == A:
            paths[path] += cp
            continue
        if step >= M:
            continue
        for k in range(K):
            npb = cp * tp[node, k]
            if npb < thr:
                continue
            v = int(nbr[node, k])
            stack.append((v, step + 1, path + (v,), npb))
    return paths


def edge_flow_k(paths, nbr):
    """Flow per edge, keyed (u, v, k) with k the move channel (v = nbr[u,k])."""
    k_of = {}
    for u in range(nbr.shape[0]):
        for k in range(nbr.shape[1]):
            k_of[(u, int(nbr[u, k]))] = k
    flow = Counter()
    for path, p in paths.items():
        for u, v in zip(path[:-1], path[1:]):
            flow[(u, v, k_of[(u, v)])] += p
    return flow


def three_models(args, apd):
    """One typical question, three models: CoNet (collapse), one-shot QuCoNet
    ('semi', also collapse), Grover-trained QuCoNet (ensemble).  Edges colored
    by move channel (RGB)."""
    import torch
    cache = json.load(open(os.path.join(args.out, "p_cache.json")))
    pc = np.asarray(cache[f"cadam|{args.seed}|{args.B}|train"])
    pq = np.asarray(cache[f"qstd|{args.seed}|{args.B}|train"])

    idxq = {f: s1.run_index(args.root, gsa.FAMILIES[f][0])
            for f in ("qstd", "grover", "cadam")}
    tr, _, _ = s1.split_qa(idxq["grover"][(args.seed, args.B)]["results"])

    sd = torch.load(idxq["cadam"][(args.seed, args.B)]["best"],
                    map_location="cpu", weights_only=False)["model_state_dict"]
    tp = torch.softmax(sd["logits"], dim=1).numpy()
    nbr = sd["node_neighbors"].numpy().astype(int)

    models = {}
    for fam in ("qstd", "grover"):
        cm, snm, scm, N, K, cfg = apd.load_quantum_checkpoint(
            idxq[fam][(args.seed, args.B)]["best"])
        models[fam] = (cm, snm, scm, N, K)
    N, K = tp.shape

    # pick the question: both one-shot models saturated, grover holds the
    # largest ensemble among them
    cands = np.where((pc > 0.9) & (pq > 0.9))[0][:24]
    best_i, best_ipr = int(cands[0]), -1.0
    cmg, snmg, scmg, _, _ = models["grover"]
    for i in cands:
        Q, A = tr[int(i)]
        paths = success_path_probs(apd, cmg, snmg, scmg, Q, A, N, K, args.M)
        pt = sum(paths.values())
        if pt <= 0:
            continue
        ipr = 1.0 / sum((v / pt) ** 2 for v in paths.values())
        if ipr > best_ipr:
            best_ipr, best_i = ipr, int(i)
    Q, A = tr[best_i]

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for u in range(N):
        for k in range(K):
            G.add_edge(u, int(nbr[u, k]))
    pos = nx.spring_layout(G, seed=7, iterations=200)
    if args.layout == "layered":
        # x = BFS distance from Q (paths flow left->right); within a shell,
        # order by spring-layout y for continuity
        dist = nx.single_source_shortest_path_length(G, Q)
        shells = {}
        for u in G.nodes:
            shells.setdefault(dist[u], []).append(u)
        spring = pos
        pos = {}
        for d, nodes in shells.items():
            nodes = sorted(nodes, key=lambda u: spring[u][1])
            n = len(nodes)
            for i, u in enumerate(nodes):
                y = 0.0 if n == 1 else (i - (n - 1) / 2) / ((n - 1) / 2)
                pos[u] = (float(d), 1.35 * y)

    KCOL = ["#d62728", "#2ca02c", "#1f77b4"]     # coin channels (quantum only)
    CCOL = "#08306b"                             # classical: no coin register
    panels = []
    panels.append(("CoNet, one-shot trained\n(classical walker)",
                   classical_success_paths(tp, nbr, Q, A, args.M), False))
    cm, snm, scm, _, _ = models["qstd"]
    panels.append(("QuCoNet, one-shot trained\n(measured walk, same architecture)",
                   success_path_probs(apd, cm, snm, scm, Q, A, N, K, args.M), True))
    panels.append(("QuCoNet, Grover-trained\n(one round converts $p^*$ to 1)",
                   success_path_probs(apd, cmg, snmg, scmg, Q, A, N, K, args.M), True))

    # crop all panels to the region the success paths actually use
    used = {Q, A}
    for _, paths, _ in panels:
        for path in paths:
            used.update(path)
    xs = [pos[u][0] for u in used]
    ys = [pos[u][1] for u in used]
    mx = 0.14 * max(max(xs) - min(xs), max(ys) - min(ys))
    xlim = (min(xs) - mx, max(xs) + mx)
    ylim = (min(ys) - mx, max(ys) + mx)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    for ax, (lab, paths, channels) in zip(axes, panels):
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.90", width=0.6)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=11, node_color="0.80")
        pt = sum(paths.values())
        ipr = (1.0 / sum((v / pt) ** 2 for v in paths.values())) if pt > 0 else 0.0
        flow = edge_flow_k(paths, nbr)
        if flow and pt > 0:
            # width/opacity = SHARE of this question's success probability
            # carried by the edge (flow / p_total), so a 3%-mass leftover is
            # faint and a 25%-mass branch is thick -- panel-comparable
            rad = 0.14 if args.layout == "layered" else 0.0
            for (u, v, k), f in sorted(flow.items(), key=lambda kv: kv[1]):
                s = f / pt
                ax.annotate("", xy=pos[v], xytext=pos[u],
                            arrowprops=dict(arrowstyle="-|>",
                                            lw=0.6 + 5.5 * s,
                                            color=KCOL[k] if channels else CCOL,
                                            alpha=min(1.0, 0.25 + 0.75 * s),
                                            connectionstyle=f"arc3,rad={rad}",
                                            shrinkA=3, shrinkB=3))
        for node, c, m in ((Q, "#2ca02c", "o"), (A, "#9467bd", "*")):
            ax.scatter(*pos[node], s=170, c=c, marker=m, zorder=5,
                       edgecolors="k", linewidths=0.7)
        npaths = len(paths)
        ax.set_title(f"{lab}\n$p={pt:.3f}$,  paths$\\,={npaths}$,  "
                     f"IPR$\\,={ipr:.2f}$", fontsize=9.5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_axis_off()
    handles = [plt.Line2D([0], [0], color=c, lw=2.4) for c in KCOL]
    axes[1].legend(handles, [r"$|0\rangle$", r"$|1\rangle$", r"$|2\rangle$"],
                   fontsize=7.5, loc="lower left", framealpha=0.9,
                   title="coin channel", title_fontsize=7.5)
    fig.suptitle(f"Same question (Q={Q} $\\to$ A={A}, seed {args.seed}, "
                 f"B={args.B}): where the success probability flows",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    sfx = "_spring" if args.layout == "spring" else ""
    f = os.path.join(args.out, f"D4_flows3_seed{args.seed}_B{args.B}{sfx}.png")
    fig.savefig(f, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}  (pair idx {best_i}, grover IPR {best_ipr:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("QSR_ROOT", os.path.abspath(os.path.join(HERE, ".."))))
    ap.add_argument("--out", default=os.path.join(HERE, "_sweep_out"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--three-models", action="store_true",
                    help="D4: one question x {CoNet, one-shot QuCoNet, "
                         "Grover QuCoNet}, RGB move channels")
    ap.add_argument("--layout", choices=("spring", "layered"),
                    default="layered",
                    help="layered: x = BFS distance from Q, paths flow "
                         "left to right (three-models mode)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.three_models:
        apd = amp._import_enumerators(args.root)
        three_models(args, apd)
        return

    apd = amp._import_enumerators(args.root)
    idx = {f: s1.run_index(args.root, gsa.FAMILIES[f][0]) for f, _ in FAMS}
    tr, _, _ = s1.split_qa(idx["grover"][(args.seed, args.B)]["results"])
    pair_idx, pq, pg = pick_pairs(os.path.join(args.out, "p_cache.json"),
                                  args.seed, args.B)
    roles = ["one-shot SOLVED", "one-shot DEAD", "median grover p"][:len(pair_idx)]

    models = {}
    for fam, _ in FAMS:
        cm, snm, scm, N, K, cfg = apd.load_quantum_checkpoint(
            idx[fam][(args.seed, args.B)]["best"])
        models[fam] = (cm, snm, scm, N, K)

    # graph + layout straight from the checkpoint's adjacency
    _, snm0, _, N, K = models["qstd"]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for u in range(N):
        for k in range(K):
            G.add_edge(u, int(snm0.numpy()[u, k]))
    pos = nx.spring_layout(G, seed=7, iterations=200)

    fig, axes = plt.subplots(len(FAMS), len(pair_idx),
                             figsize=(4.4 * len(pair_idx), 4.2 * len(FAMS)))
    axes = np.atleast_2d(axes)
    for i, (fam, fam_lab) in enumerate(FAMS):
        cm, snm, scm, N, K = models[fam]
        for j, (pi, role) in enumerate(zip(pair_idx, roles)):
            Q, A = tr[pi]
            ax = axes[i, j]
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.88", width=0.5)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=6, node_color="0.75")
            paths = success_path_probs(apd, cm, snm, scm, Q, A, N, K, args.M)
            ptot = sum(paths.values())
            ipr = (1.0 / sum((v / ptot) ** 2 for v in paths.values())) if ptot > 0 else 0.0
            flow = edge_flow(paths)
            if flow:
                fmax = max(flow.values())
                for (u, v), f in sorted(flow.items(), key=lambda kv: kv[1]):
                    ax.annotate("", xy=pos[v], xytext=pos[u],
                                arrowprops=dict(arrowstyle="-|>", lw=0.6 + 4.0 * f / fmax,
                                                color="#d62728",
                                                alpha=min(1.0, 0.25 + 0.75 * f / fmax),
                                                shrinkA=2, shrinkB=2))
            for node, c, m in ((Q, "#2ca02c", "o"), (A, "#9467bd", "*")):
                ax.scatter(*pos[node], s=130, c=c, marker=m, zorder=5,
                           edgecolors="k", linewidths=0.6)
            ax.set_title(f"{role}\nQ={Q}→A={A}   p={ptot:.3f}  IPR={ipr:.2f}",
                         fontsize=9)
            ax.set_axis_off()
        axes[i, 0].text(-0.06, 0.5, fam_lab, transform=axes[i, 0].transAxes,
                        rotation=90, va="center", fontsize=11, weight="bold")
    fig.suptitle(f"Success-path flow on the trained walk (seed{args.seed}, B={args.B}; "
                 f"green=start, purple star=target; width∝flow)", y=1.0)
    fig.tight_layout()
    f = os.path.join(args.out, f"D3_network_seed{args.seed}_B{args.B}.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}")

if __name__ == "__main__":
    main()
