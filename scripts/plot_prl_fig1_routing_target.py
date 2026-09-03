"""PRL Fig 1: the model, and the behavior that separates it from classical.

Three-figure consolidation (xcai 2026-07-16): the opening figure pairs the
framework with its signature behavior --
  (a) the QuCoNet AR circuit          -- the proposed quantum reasoning model
                                         (drawn by notes/scripts/plot_mapping_schematic)
  (b) CoNet, one-shot trained         -- classical walker collapses to one route
  (c) QuCoNet, Grover-trained         -- amplification keeps a branching ensemble

The success-map panel (draw_maps) moved to Fig 2 panel (a); it stays defined
here and is imported by plot_prl_fig2_collapse_payoff.py.

Panels (b),(c) are the two kept panels of the old three-model flow figure
(D4_flows3; the dropped middle panel -- QuCoNet one-shot, "collapse is the
objective not the substrate" -- lives in the Supplemental Material).  The
question, layout and per-question numbers are selected exactly as in
grover_network_plot.three_models, so they reproduce D4_flows3's Q=79 seed-1
B=32 instance (CoNet p=0.993 IPR=1.10; Grover p=0.250 IPR=3.95).

Run from repo root:
  conda run -n conet python docs/discussion/scripts/plot_prl_fig1_routing_target.py
"""
import os, sys, json, argparse, math
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the SciencePlots styles)
import networkx as nx

plt.style.use(["science", "no-latex"])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amplification_scaling as amp
import amplification_step1 as s1
import grover_sweep_analysis as gsa
import grover_network_plot as gnp

OUTDIR = os.path.abspath(os.path.join(HERE, "..", "figs", "ttamp"))

BLUE = "#1f4e79"
RED = "#c62828"
GRAY = "#8c8c8c"

# ── arrow-width mapping for the flow panels (a),(b) ──────────────────────
# Each drawn arrow carries a share s = (edge flow)/(total success prob), so
# s in [0,1].  Width is  w(s) = WMIN + (WMAX - WMIN) * s**WEXP.
# STRUCT_W (0.6) is the grey structural lattice; it already carries the p->0
# background, so a flow arrow need NOT taper to a hairline.  WMIN is therefore
# set a bit ABOVE the grey width, so a bond with only a modest share still reads
# as a bond rather than a thread.  WEXP<1 (sqrt) lifts the mid-weight bonds -- a
# decent 0.1-0.2 share becomes visibly thicker -- while leaving the endpoints
# fixed.  Collapse stays legible: panel (a)'s residual routes sit at s~0.005
# (near WMIN) and keep a low alpha, while its dominant route is pinned at WMAX;
# panel (b)'s ~0.1-0.25 ensemble branches now carry real weight.  WMAX kept at
# 4.3 so the dominant route reads as a route, not a blob.
STRUCT_W = 0.6
WMIN, WMAX, AFLOOR, WEXP = 1.2, 4.3, 0.25, 0.5


def flow_width(s):
    return WMIN + (WMAX - WMIN) * (s ** WEXP)   # WEXP=1 linear, 0.5 sqrt


def flow_alpha(s):
    return min(1.0, AFLOOR + (1.0 - AFLOOR) * s)


# ── panel (c): analytic success map (ported from plot_quantum_slow_overview) ──
def amplification_map(p, rounds=1):
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    theta = math.asin(math.sqrt(p))
    return math.sin((2 * rounds + 1) * theta) ** 2


def draw_maps(ax):
    p = np.linspace(0.0, 1.0, 800)
    classical = 1.0 - (1.0 - p) ** 3
    quantum = np.array([amplification_map(float(x), 1) for x in p])
    ax.plot(p, p, color=GRAY, lw=1.3, ls=":")
    ax.plot(p, classical, color=BLUE, lw=2.2)
    ax.plot(p, quantum, color=RED, lw=2.6)
    p_star = 0.25
    ax.axvline(p_star, color="0.15", lw=1.0, ls="--")
    ax.annotate("", xy=(0.17, 0.13), xytext=(0.05, 0.13),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.7))
    ax.annotate("", xy=(0.33, 0.13), xytext=(0.49, 0.13),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.7))
    ax.text(0.25, 0.18, "Grover\ntraining target", color=RED, fontsize=11,
            ha="center")
    ax.text(0.70, 0.84, "three classical\nattempts", color=BLUE, fontsize=11,
            ha="center")
    ax.text(0.135, 0.72, "one fixed\nGrover round", color=RED, fontsize=11,
            ha="left")
    ax.text(0.40, 0.34, "one attempt", color=GRAY, fontsize=11,
            ha="center", rotation=33)
    ax.set(xlim=(0, 1), ylim=(0, 1.05),
           xlabel=r"base success probability $p$",
           ylabel="accuracy after inference")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Sampling is monotone; one Grover round\n"
                 "creates an interior optimum", fontsize=10)


# ── panels (a),(b): flow on the trained walk (adapted from three_models) ──
def prepare(args):
    """Load models, pick the question, lay out the graph, enumerate the flows.

    Expensive (checkpoint loads + spring layout + path enumeration); split out
    so the flow panels can be re-rendered under different width mappings
    without recomputing.  Returns a dict consumed by render_flow_panel().
    """
    import torch
    apd = amp._import_enumerators(args.root)
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

    cmg, snmg, scmg, N, K, _ = apd.load_quantum_checkpoint(
        idxq["grover"][(args.seed, args.B)]["best"])
    N, K = tp.shape

    # same question pick as three_models: both one-shot models saturate it,
    # grover holds the largest ensemble among the candidates.
    cands = np.where((pc > 0.9) & (pq > 0.9))[0][:24]
    best_i, best_ipr = int(cands[0]), -1.0
    for i in cands:
        Q, A = tr[int(i)]
        paths = gnp.success_path_probs(apd, cmg, snmg, scmg, Q, A, N, K, args.M)
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
    # layered layout: x = BFS distance from Q so paths flow left -> right
    dist = nx.single_source_shortest_path_length(G, Q)
    shells = {}
    for u in G.nodes:
        shells.setdefault(dist[u], []).append(u)
    spring, pos = pos, {}
    for d, nodes in shells.items():
        nodes = sorted(nodes, key=lambda u: spring[u][1])
        n = len(nodes)
        for i, u in enumerate(nodes):
            y = 0.0 if n == 1 else (i - (n - 1) / 2) / ((n - 1) / 2)
            pos[u] = (float(d), 0.95 * y)

    KCOL = ["#d62728", "#2ca02c", "#1f77b4"]   # coin channels (quantum only)
    CCOL = "#08306b"                           # classical: no coin register
    panels = [
        ("CoNet, single-attempt training",
         gnp.classical_success_paths(tp, nbr, Q, A, args.M), False),
        ("QuCoNet, Grover-trained",
         gnp.success_path_probs(apd, cmg, snmg, scmg, Q, A, N, K, args.M), True),
    ]

    # crop to the region the drawn success paths actually use
    used = {Q, A}
    for _, paths, _ in panels:
        for path in paths:
            used.update(path)
    xs = [pos[u][0] for u in used]
    ys = [pos[u][1] for u in used]
    mx = 0.07 * max(max(xs) - min(xs), max(ys) - min(ys))
    xlim = (min(xs) - mx, max(xs) + mx)
    ylim = (min(ys) - mx, max(ys) + mx)

    return dict(G=G, pos=pos, Q=Q, A=A, panels=panels, nbr=nbr,
                KCOL=KCOL, CCOL=CCOL, xlim=xlim, ylim=ylim, best_ipr=best_ipr)


def render_flow_panel(ax, D, panel, width_fn=flow_width, alpha_fn=flow_alpha):
    """Draw one routing panel: grey lattice + per-edge flow arrows.

    width_fn/alpha_fn map an edge's flow share s in [0,1] to line width / alpha,
    so the same enumerated flows can be re-rendered under different mappings.
    """
    lab, paths, channels = panel
    nx.draw_networkx_edges(D["G"], D["pos"], ax=ax, edge_color="0.90", width=STRUCT_W)
    nx.draw_networkx_nodes(D["G"], D["pos"], ax=ax, node_size=9, node_color="0.80")
    pt = sum(paths.values())
    ipr = (1.0 / sum((v / pt) ** 2 for v in paths.values())) if pt > 0 else 0.0
    flow = gnp.edge_flow_k(paths, D["nbr"])
    if flow and pt > 0:
        for (u, v, k), f in sorted(flow.items(), key=lambda kv: kv[1]):
            s = f / pt
            ax.annotate("", xy=D["pos"][v], xytext=D["pos"][u],
                        arrowprops=dict(arrowstyle="-|>",
                                        lw=width_fn(s),
                                        color=D["KCOL"][k] if channels else D["CCOL"],
                                        alpha=alpha_fn(s),
                                        connectionstyle="arc3,rad=0.11",
                                        shrinkA=3, shrinkB=3))
    for node, c, m in ((D["Q"], "#2ca02c", "o"), (D["A"], "#9467bd", "*")):
        ax.scatter(*D["pos"][node], s=140, c=c, marker=m, zorder=5,
                   edgecolors="k", linewidths=0.7)
    ax.set_title(rf"{lab}: $p={pt:.3f}$, $\mathrm{{IPR}}={ipr:.2f}$",
                 fontsize=10, pad=3)
    ax.set_xlim(*D["xlim"]); ax.set_ylim(*D["ylim"])
    ax.set_axis_off()


def build(args):
    sys.path.insert(0, os.path.join(args.root, "notes", "scripts"))
    import plot_mapping_schematic as pms

    D = prepare(args)
    Q, A, best_ipr = D["Q"], D["A"], D["best_ipr"]

    fig = plt.figure(figsize=(13.6, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.12, 1.0),
                          height_ratios=(1.0, 1.0), wspace=0.06, hspace=0.26)
    ax_arch = fig.add_subplot(gs[:, 0])
    ax_conet = fig.add_subplot(gs[0, 1])
    ax_qu = fig.add_subplot(gs[1, 1])

    pms.draw_circuit(ax_arch, description=False)
    ax_arch.set_xlim(0.9, 10.45)
    ax_arch.set_ylim(0.30, 4.55)
    ax_arch.axis("off")

    render_flow_panel(ax_conet, D, D["panels"][0])
    render_flow_panel(ax_qu, D, D["panels"][1])

    handles = [plt.Line2D([0], [0], color=c, lw=2.4) for c in D["KCOL"]]
    ax_qu.legend(handles, [r"$|0\rangle$", r"$|1\rangle$", r"$|2\rangle$"],
                 fontsize=9, loc="lower left", bbox_to_anchor=(0.0, -0.085),
                 framealpha=0.85, ncol=3,
                 title="path-record channel", title_fontsize=9,
                 columnspacing=0.9, handlelength=1.1,
                 borderpad=0.35, handletextpad=0.5)

    ax_arch.text(0.005, 1.06, "a", transform=ax_arch.transAxes, fontsize=13,
                 fontweight="bold", va="top")
    ax_conet.text(-0.015, 1.16, "b", transform=ax_conet.transAxes, fontsize=13,
                  fontweight="bold", va="top")
    ax_qu.text(-0.015, 1.16, "c", transform=ax_qu.transAxes, fontsize=13,
               fontweight="bold", va="top")

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "Q1_arch_routing.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"fig -> {out}  (Q={Q} A={A}, grover IPR {best_ipr:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    ap.add_argument("--out", default=os.path.join(HERE, "_sweep_out"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--M", type=int, default=8)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
