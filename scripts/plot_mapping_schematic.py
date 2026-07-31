#!/usr/bin/env python3
"""Main-text mapping schematic (fig:mapping): where quantum equals classical
and where it separates.

Two layers, drawn as a 2x2 grid (collab request 2026-07-10: give ML readers
the quantum-classical correspondence in one picture):

  row 1 -- THE MODEL (exact map): a routing-table policy sampling a path
           == the coined walk in AR mode measured at the end; identical
           distribution over paths (which-path registers).
  row 2 -- TEST TIME (the map severs): the same verifier used as a filter
           on complete answers (sample-verify-retry, monotone in p,
           Prop. 1) vs used as a phase mirror inside the circuit
           (rotation, peak at p*). Training targets follow: collapse vs
           interior attractor.

Run from notes/:  python plot_mapping_schematic.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc, FancyArrowPatch

OUT = os.path.join(REPO, "figs", "S14_mapping_schematic.png")
GOOD, BAD, Q, RED, VIO = "#2ca02c", "0.60", "#1f77b4", "#d62728", "#4b0082"


def bezier(p0, p1, p2, n=60):
    t = np.linspace(0, 1, n)[:, None]
    pts = ((1 - t) ** 2) * np.asarray(p0) + 2 * (1 - t) * t * np.asarray(p1) \
        + t ** 2 * np.asarray(p2)
    return pts[:, 0], pts[:, 1]


# ────────────────────────── row 1, left: policy on a graph ─────────────
def draw_policy(ax):
    rng = np.random.default_rng(7)
    nodes = {0: (0.7, 2.6), 1: (2.2, 3.6), 2: (2.3, 1.5), 3: (3.9, 2.8),
             4: (4.0, 0.9), 5: (5.6, 3.5), 6: (5.7, 1.7), 7: (7.2, 2.6)}
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5),
             (4, 6), (5, 6), (5, 7), (6, 7), (2, 3)]
    for a, b in edges:
        ax.plot(*zip(nodes[a], nodes[b]), color="0.8", lw=1.1, zorder=1)
    path = [0, 2, 3, 5, 7]
    for a, b in zip(path, path[1:]):
        ax.annotate("", xy=nodes[b], xytext=nodes[a],
                    arrowprops=dict(arrowstyle="-|>", color=Q, lw=2.4,
                                    shrinkA=8, shrinkB=8), zorder=3)
    for i, (x, y) in nodes.items():
        start, goal = (i == path[0]), (i == path[-1])
        ax.add_patch(Circle((x, y), 0.21,
                            facecolor=Q if start else (GOOD if goal else "white"),
                            edgecolor="k" if (start or goal) else "0.5",
                            lw=1.2 if (start or goal) else 0.9, zorder=4))
    ax.text(*nodes[path[0]], "Q", color="white", ha="center", va="center",
            fontsize=8, fontweight="bold", zorder=5)
    ax.text(*nodes[path[-1]], "A", color="white", ha="center", va="center",
            fontsize=8, fontweight="bold", zorder=5)
    # routing-table icon
    tx, ty = 8.6, 1.1
    ax.add_patch(Rectangle((tx, ty), 2.9, 2.6, facecolor="#f5f5f5",
                           edgecolor="0.4", lw=1.0))
    ax.text(tx + 1.45, ty + 2.25, r"policy $\pi(\,\cdot\,|x)$",
            ha="center", fontsize=8.5)
    for r, row in enumerate((".62  .25  .13", ".08  .81  .11", ".33  .34  .33")):
        ax.text(tx + 1.45, ty + 1.55 - 0.55 * r, row, ha="center",
                fontsize=7.5, family="monospace", color="0.25")
    ax.text(6.0, -0.15, "sample a move per node, step by step;\n"
            "the answer is one full path", ha="center", va="top", fontsize=9)
    ax.set_title("classical walker (CoNet): routing table,\n"
                 "one path per attempt", fontsize=10)


# ────────────────────────── row 1, right: AR quantum circuit ───────────
def draw_circuit(ax):
    x0, x1 = 1.0, 9.6
    wires = [("pos", 3.9, r"$|Q\rangle$")] + \
            [(f"c{s}", 2.9 - 0.75 * (s - 1), rf"$|c_{s}\rangle$") for s in (1, 2, 3)]
    for _, y, lab in wires:
        ax.plot([x0, x1], [y, y], color="0.35", lw=1.1, zorder=1)
        ax.text(x0 - 0.15, y, lab, ha="right", va="center", fontsize=8.5)
    ax.text(x0 + 6.15, 0.62, r"$\cdots\ M$ steps, one fresh coin register each",
            ha="center", fontsize=8)
    for s in (1, 2, 3):
        xc = 1.9 + 2.15 * (s - 1)
        yc = 2.9 - 0.75 * (s - 1)
        # coin conditioned on position (control dot on pos wire)
        ax.plot([xc, xc], [yc, 3.9], color=RED, lw=1.2, zorder=2)
        ax.add_patch(Circle((xc, 3.9), 0.09, facecolor=RED, edgecolor=RED, zorder=3))
        ax.add_patch(Rectangle((xc - 0.42, yc - 0.34), 0.84, 0.68,
                               facecolor="white", edgecolor=RED, lw=1.4, zorder=3))
        ax.text(xc, yc, rf"$C_{{x_{s-1}}}$", ha="center", va="center",
                fontsize=8, color=RED, zorder=4)
        # shift on (pos, c_s)
        xs = xc + 1.05
        ax.add_patch(Rectangle((xs - 0.38, yc - 0.34), 0.76, 3.9 - yc + 0.68,
                               facecolor="white", edgecolor="0.25", lw=1.3, zorder=3))
        ax.text(xs, (3.9 + yc) / 2, r"$S$", ha="center", va="center",
                fontsize=9, zorder=4)
    # meters
    for _, y, _ in wires:
        mx = x1 + 0.35
        ax.add_patch(Rectangle((mx - 0.3, y - 0.28), 0.62, 0.56,
                               facecolor="#f5f5f5", edgecolor="k", lw=1.0, zorder=3))
        ax.add_patch(Arc((mx, y - 0.1), 0.4, 0.4, theta1=20, theta2=160,
                         color="k", lw=1.1, zorder=4))
        ax.plot([mx, mx + 0.13], [y - 0.1, y + 0.16], color="k", lw=1.1, zorder=4)
    ax.text(6.0, -0.15, "the records mark which edge was taken; measured,\n"
            "the walk is a classical Markov chain, its paths orthogonal",
            ha="center", va="top", fontsize=9)
    ax.set_title("quantum walker (QuCoNet), autoregressive mode:\n"
                 r"$\mathcal{A}=\prod_s S\,C_{x}$, then measure", fontsize=10)


# ────────────────────────── row 2, left: verifier as filter ────────────
def draw_filter(ax):
    q = np.array([0.9, 2.8])
    ends = [np.array([7.6, y]) for y in (0.6, 1.8, 4.8, 3.0, 3.9)]
    ctrls = [np.array([4.2, c]) for c in (0.1, 3.5, 5.4, 1.6, 4.7)]
    for i, (e, c) in enumerate(zip(ends, ctrls)):
        x, y = bezier(q, c, e)
        good = (i == 2)
        ax.plot(x, y, "-" if good else "--", color=GOOD if good else BAD,
                lw=2.3 if good else 1.3, alpha=0.95 if good else 0.8)
    for i, e in enumerate(ends):
        good = (i == 2)
        ax.add_patch(Circle(e, 0.20, facecolor="white",
                            edgecolor=GOOD if good else "0.55",
                            lw=1.7 if good else 1.0, zorder=5))
        ax.text(*e, "✓" if good else "✗", ha="center", va="center",
                fontsize=8 if good else 6.5, color=GOOD if good else "0.55",
                zorder=6, fontweight="bold" if good else "normal")
    ax.add_patch(Circle(q, 0.28, facecolor=Q, edgecolor="k", lw=1.0, zorder=5))
    ax.text(*q, "Q", ha="center", va="center", color="white", fontsize=9,
            fontweight="bold", zorder=6)
    # retry loop
    ar = FancyArrowPatch((8.4, 1.2), (1.2, 1.0), connectionstyle="arc3,rad=0.5",
                         arrowstyle="-|>", color="0.3", lw=1.3, ls=":")
    ax.add_patch(ar)
    ax.text(4.7, -0.4, r"retry $\times k$", fontsize=8.5, color="0.3", ha="center")
    ax.text(9.0, 3.4, "verifier =\nfilter on\ncomplete\nanswers",
            fontsize=8.5, color="0.2", ha="left", va="center")
    ax.text(6.0, -1.25,
            r"$\mathrm{cl}_k(p)=1-(1-p)^k$ is monotone in $p$ (Prop.~1)"
            "\ntraining optimum: collapse to $p\\in\\{0,1\\}$",
            ha="center", va="top", fontsize=9)
    ax.set_title("verifier as filter: sample, check, retry", fontsize=10)


# ────────────────────────── row 2, right: verifier as mirror ───────────
def draw_mirror(ax, caption=True):
    q = np.array([0.9, 2.8])
    ends = [np.array([7.6, y]) for y in (0.6, 1.8, 4.8, 3.0, 3.9)]
    ctrls = [np.array([4.2, c]) for c in (0.1, 3.5, 5.4, 1.6, 4.7)]
    for i, (e, c) in enumerate(zip(ends, ctrls)):
        x, y = bezier(q, c, e)
        good = (i == 2)
        ax.plot(x, y, "-", color=RED, lw=2.5 if good else 1.9,
                alpha=0.95 if good else 0.32, solid_capstyle="round")
    ax.add_patch(Circle(q, 0.28, facecolor=Q, edgecolor="k", lw=1.0, zorder=5))
    ax.text(*q, "Q", ha="center", va="center", color="white", fontsize=9,
            fontweight="bold", zorder=6)
    # the mirror: a vertical bar at the endpoints flipping phase of the good one
    ax.plot([8.05, 8.05], [0.2, 5.3], color=VIO, lw=3.0, zorder=4)
    ax.text(8.25, 5.1, r"phase mirror $I-2P_G$", fontsize=8.5, color=VIO,
            ha="left", va="center")
    ax.text(8.25, 4.35, r"$|\mathrm{good}\rangle\!\to\!-|\mathrm{good}\rangle$",
            fontsize=8, color=VIO, ha="left")
    # rotation inset
    ins = ax.inset_axes([0.74, 0.06, 0.25, 0.44])
    th = np.deg2rad(19)
    for ang, col, lw in ((th, "0.4", 1.5), (3 * th, RED, 2.1)):
        ins.annotate("", xy=(np.cos(ang), np.sin(ang)), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=col, lw=lw))
    ins.add_patch(Arc((0, 0), 1.05, 1.05, theta1=np.rad2deg(th),
                      theta2=np.rad2deg(3 * th), color=RED, lw=1.1))
    ins.text(0.72, 0.47, r"$\theta\to3\theta$", fontsize=8, color=RED)
    ins.set_xlim(-0.05, 1.1); ins.set_ylim(-0.05, 1.1)
    ins.set_aspect("equal"); ins.axis("off")
    if caption:
        ax.text(6.0, -1.25,
                r"$A_n(p)=\sin^2[(2n{+}1)\arcsin\sqrt{p}\,]$ peaks at interior $p$"
                "\ntraining optimum: the attractor $p^{*}(n)$",
                ha="center", va="top", fontsize=9)
    ax.set_title("verifier as phase mirror: one coherent try,\n"
                 "rounds rotate weight onto the verified branch", fontsize=10)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.6))
    draw_policy(axes[0, 0]); draw_circuit(axes[0, 1])
    draw_filter(axes[1, 0]); draw_mirror(axes[1, 1])
    for ax in axes.flat:
        ax.set_xlim(-0.4, 12.0); ax.set_ylim(-2.3, 5.9); ax.axis("off")
    axes[0, 0].set_ylim(-1.4, 5.4); axes[0, 1].set_ylim(-1.4, 5.4)

    # between-column verdicts
    fig.text(0.5, 0.76, r"$\equiv$", fontsize=26, ha="center", va="center")
    fig.text(0.5, 0.695, "identical distribution\nover paths\n(which-path registers)",
             fontsize=8.5, ha="center", va="center", color="0.15")
    fig.text(0.5, 0.30, r"$\neq$", fontsize=26, ha="center", va="center",
             color=RED)
    fig.text(0.5, 0.225, "same verifier,\ndifferent access:\nsamples vs. the unitary",
             fontsize=8.5, ha="center", va="center", color="0.15")

    # row banners
    fig.text(0.055, 0.965, "THE MODEL — the map is exact", fontsize=10.5,
             fontweight="bold", ha="left", color="0.1")
    fig.text(0.055, 0.475, "TEST TIME — the map severs", fontsize=10.5,
             fontweight="bold", ha="left", color=RED)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(wspace=0.34, hspace=0.52)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")

    # ── PRL model-only panel: top row (the architecture), collab request #62 2026-07-13.
    # The classical routing-table walker == the AR coined-walk circuit (which-path
    # records), rendered without the test-time row (that overlaps the PRL's Fig 1).
    OUT_MODEL = os.path.join(REPO, "figs", "S14b_model_schematic.png")
    figm, axm = plt.subplots(1, 2, figsize=(12.6, 4.2))
    draw_policy(axm[0]); draw_circuit(axm[1])
    for ax in axm:
        ax.set_xlim(-0.4, 12.0); ax.set_ylim(-1.4, 5.4); ax.axis("off")
    figm.text(0.5, 0.60, r"$\equiv$", fontsize=24, ha="center", va="center", color="0.1")
    figm.text(0.5, 0.49, "identical\ndistribution\nover paths",
              fontsize=8, ha="center", va="center", color="0.2")
    figm.tight_layout(rect=(0, 0, 1, 0.99))
    figm.subplots_adjust(wspace=0.46)
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    figm.savefig(OUT_MODEL, dpi=160, bbox_inches="tight")
    print("wrote", OUT_MODEL)

    # ── full three-panel schematic (was PRL Fig 3, now SM fig:sm-model after the
    # 2026-07-16 three-figure consolidation): the SAME walk in both regimes --
    # measured it is a classical Markov chain (trainable, simulable); run
    # coherently, the verifier acts as a phase mirror and the good/bad subspace
    # rotates -- amplitude amplification, with no classical sampling equivalent.
    OUT_Q3 = os.path.join(REPO, "figs", "Q3_model_quantum.png")
    figq, axq = plt.subplots(1, 3, figsize=(16.5, 4.7))
    draw_policy(axq[0]); draw_circuit(axq[1]); draw_mirror(axq[2], caption=False)
    for ax in axq:
        ax.set_xlim(-0.4, 12.0); ax.axis("off")
    axq[0].set_ylim(-1.4, 5.4); axq[1].set_ylim(-1.4, 5.4)
    axq[2].set_ylim(-2.1, 5.9)
    axq[2].text(5.6, -1.35, "amplitude amplification: certainty in "
                r"$O(1/\sqrt{p})$ coherent" "\nqueries where classical sampling needs "
                r"$O(1/p)$: no classical equivalent",
                ha="center", va="top", fontsize=8.5, color=RED)
    figq.text(0.352, 0.50, r"$\equiv$", fontsize=22, ha="center", va="center", color="0.1")
    figq.text(0.352, 0.40, "measured:\nsame Markov\ndistribution\n(trainable)",
              fontsize=7.6, ha="center", va="center", color="0.2")
    figq.text(0.655, 0.50, r"$\Rightarrow$", fontsize=22, ha="center", va="center", color=RED)
    figq.text(0.655, 0.395, "run\ncoherently:\nverifier\nreflects",
              fontsize=7.6, ha="center", va="center", color=RED)
    figq.text(0.5, 1.02, "The same walk: a classical Markov chain when measured, "
              "an amplifiable superposition when run coherently",
              fontsize=11, ha="center", va="center", fontweight="bold", color="0.1")
    figq.tight_layout(rect=(0, 0, 1, 0.97))
    figq.subplots_adjust(wspace=0.30)
    figq.savefig(OUT_Q3, dpi=160, bbox_inches="tight")
    print("wrote", OUT_Q3)
    print(f"fig -> {OUT}")


if __name__ == "__main__":
    main()
