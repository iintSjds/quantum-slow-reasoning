#!/usr/bin/env python3
"""Fig. 3 (fig:conet): slow thinking as a walk on a concept network.

Vector replacement for the raster illustration: a directed concept network,
one dark single hop (System 1), and one red multi-hop trajectory from the
question node to the answer node (System 2).  Visual language matches
Fig. 1: green circle = question, purple star = answer, light gray lattice,
red trajectory arrows.

Writes notes/prx_path_integral/figs/Q_conet_slow_thinking.pdf.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.dirname(HERE)

GRAY_E = "0.78"      # structural edges
GRAY_N = "0.82"      # structural node fill
GRAY_NE = "0.55"     # structural node edge
GREEN = "#2ca02c"    # question (Fig. 1 convention)
PURPLE = "#9467bd"   # answer (Fig. 1 convention)
RED = "#d62728"      # System-2 trajectory (caption: red arrows)
BLUE = "#08306b"     # System-1 single hop (Fig. 1 classical color)

# ── hand-placed concept nodes ──────────────────────────────────────────────
P = {
    "Q":  (0.06, 0.46),
    "A":  (0.94, 0.60),
    # red System-2 trajectory Q -> t1 -> t2 -> t3 -> A
    "t1": (0.28, 0.58),
    "t2": (0.47, 0.40),
    "t3": (0.70, 0.56),
    # System-1 neighbor of Q
    "s1": (0.20, 0.22),
    # background concepts
    "b1": (0.14, 0.78),
    "b2": (0.36, 0.84),
    "b3": (0.44, 0.13),
    "b4": (0.60, 0.80),
    "b5": (0.63, 0.18),
    "b6": (0.82, 0.30),
    "b7": (0.84, 0.86),
    "b8": (0.32, 0.30),
}

STRUCT_EDGES = [
    ("Q", "b1"), ("b1", "b2"), ("b2", "t1"), ("t1", "b2"),
    ("b1", "t1"), ("Q", "b8"), ("b8", "t2"), ("s1", "b8"),
    ("s1", "b3"), ("b3", "t2"), ("b3", "b5"), ("b5", "t2"),
    ("b5", "b6"), ("b6", "A"), ("b6", "t3"), ("t2", "b5"),
    ("b2", "b4"), ("b4", "t3"), ("b4", "b7"), ("b7", "A"),
    ("t3", "b7"), ("t1", "t2"), ("b8", "s1"), ("t2", "b6"),
]
S2_PATH = ["Q", "t1", "t2", "t3", "A"]
S1_HOP = ("Q", "s1")


def arrow(ax, u, v, color, lw, alpha=1.0, rad=0.12, shrink=7.5, ms=11, z=3):
    ax.add_patch(FancyArrowPatch(
        P[u], P[v], arrowstyle="-|>", mutation_scale=ms,
        connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink,
        lw=lw, color=color, alpha=alpha, zorder=z))


def main():
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # structural lattice (directed, faint)
    for u, v in STRUCT_EDGES:
        arrow(ax, u, v, GRAY_E, 0.9, rad=0.10, ms=6, z=1)

    # background concept nodes
    for name, (x, y) in P.items():
        if name in ("Q", "A"):
            continue
        ax.add_patch(Circle((x, y), 0.028, facecolor=GRAY_N,
                            edgecolor=GRAY_NE, lw=0.8, zorder=2))

    # System-1: one automatic hop
    arrow(ax, *S1_HOP, BLUE, 2.2, rad=0.16, ms=13, z=4)
    ax.text(0.035, 0.175, "System 1:\na single hop", fontsize=7.5,
            color=BLUE, ha="left", va="top", linespacing=1.25)

    # System-2: red trajectory through intermediate states
    for u, v in zip(S2_PATH[:-1], S2_PATH[1:]):
        arrow(ax, u, v, RED, 2.6, rad=-0.16, ms=14, z=5)
    ax.text(0.46, 0.985, "System 2: a trajectory of hops reaches the answer",
            fontsize=7.5, color=RED, ha="center", va="top")

    # question and answer markers (Fig. 1 conventions)
    ax.scatter(*P["Q"], s=210, c=GREEN, marker="o", zorder=6,
               edgecolors="k", linewidths=0.8)
    ax.text(P["Q"][0] - 0.045, P["Q"][1] + 0.035, "$Q$", fontsize=10,
            ha="right", va="bottom")
    ax.scatter(*P["A"], s=300, c=PURPLE, marker="*", zorder=6,
               edgecolors="k", linewidths=0.7)
    ax.text(P["A"][0] + 0.01, P["A"][1] + 0.055, "answer", fontsize=8,
            ha="center", va="bottom")

    out = os.path.join(NOTES_DIR, "figs", "Q_conet_slow_thinking.pdf")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"fig -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
