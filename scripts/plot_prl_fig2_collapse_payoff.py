"""PRL Fig 2: loss map, collapse, and the matched-budget payoff (3 panels).

Main-text figure 2: this figure carries the
whole "different loss -> less collapse -> better performance" arc, in the
shape of the PRX overview figure:

(a) the loss   -- the maps inference returns to training: classical
                  best-of-3 is monotone, one Grover round peaks at the
                  interior target p*(3)=1/4 (draw_maps, imported from
                  plot_prl_fig1_routing_target).
(b) evidence   -- trained p distributions at B=32: one-shot walkers
                  (classical AND quantum) collapse to {0,1}; the
                  G_{<=3}-trained walker concentrates at p*(3)=0.25.
(c) payoff     -- one amplification round on the A_1-trained walker vs the
                  ENVELOPE of the trained control family at the matched
                  3-query budget (per-B max of best-of-3 over the 24
                  classical / same-architecture objectives; the control audit
                  fix -- the old single bo2-trained control understated the
                  strongest classical use of the budget).

Data: _sweep_out/p_cache.json (32 seeds, exact enumeration).  Writes the
figure straight into figs/.

Run from repo root:
  python scripts/plot_prl_fig2_collapse_payoff.py
"""
import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = json.load(open(os.path.join(HERE, "_sweep_out", "p_cache.json")))
OUTDIR = os.path.abspath(os.path.join(HERE, "..", "figs"))
B_LIST = [8, 16, 32, 48, 64, 96, 128]

C_CB, C_Q, C_G = "#08306b", "#ff7f0e", "#d62728"


def A_env(p, n):
    if p <= 0.0: return 0.0
    if p >= 1.0: return 1.0
    th = math.asin(math.sqrt(p))
    k = min(n, max(0, int(round(math.pi / (4 * th) - 0.5))))
    return math.sin((2 * k + 1) * th) ** 2


def pooled(fam, B, split):
    ps = []
    for s in range(1, 33):
        ps += CACHE.get(f"{fam}|{s}|{B}|{split}", [])
    return np.asarray(ps)


def per_seed(fam, B, split, fn):
    v = []
    for s in range(1, 33):
        ps = np.asarray(CACHE.get(f"{fam}|{s}|{B}|{split}", []))
        if len(ps):
            v.append(fn(ps))
    return np.asarray(v)


def curve(fam, split, fn):
    m, lo, hi = [], [], []
    for B in B_LIST:
        v = per_seed(fam, B, split, fn)
        m.append(v.mean() if len(v) else np.nan)
        lo.append(v.min() if len(v) else np.nan)
        hi.append(v.max() if len(v) else np.nan)
    return map(np.asarray, (m, lo, hi))


def main():
    import plot_prl_fig1_routing_target as f1
    plt.rcParams.update({"font.size": 12})

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6), sharey=True)
    ps1 = math.sin(math.pi / 6) ** 2

    # ── (a) the loss: sampling map vs one-round amplification map ───────
    ax = axes[0]
    f1.draw_maps(ax)
    ax.set_title("")
    ax.set_ylabel("accuracy after inference", fontsize=12)
    ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", ""])
    ax.text(0.01, 1.075, "a", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")

    # ── (b) evidence: trained distributions at B=32 ─────────────────────
    # Light bars: histograms (bin width BINW, p*(3) on a bin edge).
    # Curves: moving-window average of the RAW per-question values,
    #   curve(x) = frac{p in [x-d, x+d] ∩ [0,1]} / (in-range length) * BINW,
    # edge-corrected, in fraction-per-bin units so bars and curve share the
    # y-scale.  Line styles are monochrome-safe (dotted reserved for p*).
    ax = axes[1]
    BINW, DWIN = 0.05, 0.03
    bins = np.linspace(0, 1, round(1 / BINW) + 1)
    wgrid = np.linspace(0, 1, 401)
    for fam, col, ls, lab in (
            ("cadam", "#2166ac", (0, (4, 2.2)), "single-attempt, classical"),
            ("qstd", "#1a9850", (0, (5, 1.6, 1, 1.6)),
             "single-attempt, quantum"),
            ("grover", C_G, "solid", "Grover-trained, quantum")):
        ps = pooled(fam, 32, "train")
        weights = np.full(len(ps), 1.0 / len(ps))
        ax.hist(ps, bins=bins, weights=weights, histtype="stepfilled",
                fc=to_rgba(col, 0.18), ec=to_rgba(col, 0.45), lw=0.8)
        fwin = np.empty_like(wgrid)
        for i, x in enumerate(wgrid):
            lo, hi = max(0.0, x - DWIN), min(1.0, x + DWIN)
            fwin[i] = np.mean((ps >= lo) & (ps <= hi)) / (hi - lo) * BINW
        ax.plot(wgrid, fwin, color=col, lw=2.2, ls=ls, label=lab)
    ax.axvline(ps1, ls=":", color="k", lw=0.9)
    ax.text(ps1 - 0.02, 0.97, r"$p^{*}(3)$", fontsize=10.5,
            ha="right", va="top")
    ax.set_xlabel(r"trained $p$ ($B=32$, training questions)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels(["", "0.2", "0.4", "0.6", "0.8", ""])
    ax.text(0.085, 0.50, "fraction of questions", transform=ax.transAxes,
            rotation=90, ha="left", va="center", fontsize=12)
    ax.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.62, 1.0))
    ax.set_title("")
    ax.text(0.01, 1.075, "b", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")

    # ── (c) payoff: one round vs the strongest classical use of 3 queries ──
    CONTROLS = ["cadam", "cstd", "cstd_adam", "cstd_exact", "cbestk",
                "cbestk_exact", "ccap25", "ccap50", "ccap75",
                "centH01", "centH03", "centH10",
                "cbentH01", "cbentH03", "cbentH10",
                "cbestkX4", "cbestkX8", "cbestkX16", "cbestkX32", "cbestkX64",
                "qbkX2", "qbkX4", "qbkX8", "qstd"]
    ax = axes[2]
    m, lo, hi = curve("cadam", "train", np.mean)
    ax.plot(B_LIST, m, ":", color="0.55", lw=1.4,
            label="classical single-attempt")
    env_m, env_lo, env_hi = [], [], []
    for B in B_LIST:
        best = (-1.0, np.nan, np.nan)
        for fam in CONTROLS:
            v = per_seed(fam, B, "train",
                         lambda x: float(np.mean(1 - (1 - x) ** 3)))
            if len(v) and v.mean() > best[0]:
                best = (float(v.mean()), float(v.min()), float(v.max()))
        env_m.append(best[0]); env_lo.append(best[1]); env_hi.append(best[2])
    ax.plot(B_LIST, env_m, "-D", color=C_CB, ms=4, lw=1.8,
            label="strongest classical control\n(best of 24 objectives at $q=3$)")
    ax.fill_between(B_LIST, env_lo, env_hi, color=C_CB, alpha=0.15, lw=0)
    m, lo, hi = curve("grover", "train",
                      lambda x: float(np.mean([A_env(v, 1) for v in x])))
    ax.plot(B_LIST, m, "-s", color=C_G, ms=5, lw=2.2,
            label="Grover-trained + 1 Grover round")
    ax.fill_between(B_LIST, lo, hi, color=C_G, alpha=0.15, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(B_LIST); ax.set_xticklabels(B_LIST)
    ax.set_xlabel(r"training-set size $B$")
    ax.set_ylabel("accuracy on training questions", fontsize=12)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelleft=False, labelright=True)
    ax.legend(fontsize=11.4, loc="lower left", framealpha=0.9)
    ax.set_title("")
    ax.text(0.01, 1.075, "c", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")

    fig.tight_layout(pad=0.25, w_pad=0)
    fig.subplots_adjust(wspace=0)
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "Q2_loss_collapse_payoff.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"fig -> {out}")


if __name__ == "__main__":
    main()
