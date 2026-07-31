"""PRL Fig 2: loss map, collapse, and the matched-budget payoff (3 panels).

Main-text figure 2: this figure carries the
whole "different loss -> less collapse -> better performance" arc, in the
shape of the PRX overview figure:

(a) the loss   -- the maps inference returns to training: classical
                  best-of-3 is monotone, one Grover round peaks at the
                  interior target p*(1)=1/4 (draw_maps, imported from
                  plot_prl_fig1_routing_target).
(b) evidence   -- trained p distributions at B=32: one-shot walkers
                  (classical AND quantum) collapse to {0,1}; the
                  A_1-trained walker concentrates at p*(1)=0.25.
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

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6))
    ps1 = math.sin(math.pi / 6) ** 2

    # ── (a) the loss: sampling map vs one-round amplification map ───────
    ax = axes[0]
    f1.draw_maps(ax)
    ax.text(-0.16, 1.10, "a", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")

    # ── (b) evidence: trained distributions at B=32 ─────────────────────
    ax = axes[1]
    bins = np.linspace(0, 1, 26)
    for fam, col, lab in (("cadam", "#1f77b4", "one-shot trained, classical"),
                          ("qstd", C_Q, "one-shot trained, quantum"),
                          ("grover", C_G, "Grover-trained, quantum")):
        ps = pooled(fam, 32, "train")
        ax.hist(ps, bins=bins, density=True, histtype="step", lw=1.9,
                color=col, label=lab)
    ax.axvline(ps1, ls=":", color="k", lw=0.9)
    ax.annotate(r"$p^*(1)$", (ps1, ax.get_ylim()[1]), ha="center",
                fontsize=9, annotation_clip=False)
    ax.set_yscale("log")
    ax.set_xlabel("trained $p$  (B=32, training questions)")
    ax.set_ylabel("density (log)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="upper center")
    ax.set_title("one-shot training collapses to $\\{0,1\\}$;\n"
                 "amplification training concentrates at $p^*$", fontsize=10)
    ax.text(-0.14, 1.10, "b", transform=ax.transAxes, fontsize=12,
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
    ax.plot(B_LIST, m, ":", color="0.55", lw=1.4, label="classical one-shot")
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
            label="strongest classical control\n(best-of-3 envelope, 24 objectives)")
    ax.fill_between(B_LIST, env_lo, env_hi, color=C_CB, alpha=0.15, lw=0)
    m, lo, hi = curve("grover", "train",
                      lambda x: float(np.mean([A_env(v, 1) for v in x])))
    ax.plot(B_LIST, m, "-s", color=C_G, ms=5, lw=2.2,
            label="Grover-trained + 1 Grover round")
    ax.fill_between(B_LIST, lo, hi, color=C_G, alpha=0.15, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(B_LIST); ax.set_xticklabels(B_LIST)
    ax.set_xlabel("training-set size $B$")
    ax.set_ylabel("accuracy (training questions)")
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=7.6, loc="lower left", framealpha=0.9)
    ax.set_title("the payoff at the matched\nthree-query budget",
                 fontsize=10)
    ax.text(-0.14, 1.10, "c", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "Q2_loss_collapse_payoff.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"fig -> {out}")


if __name__ == "__main__":
    main()
