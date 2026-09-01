#!/usr/bin/env python3
"""Main-text capacity-knee figure with matched classical control.

Reads the cached per-question probabilities produced by
`scripts/frust_analysis.py` (the N=120 sliding-puzzle
large-B Grover-n sweep, n=1..6; n=5,6 extended to B=1024/1216 so the n=6
capacity knee is resolved off the grid edge) and renders the three-panel
figure used in the main text:
  (left)  coherent and classical training accuracy after read-out vs B at
          matched query budgets q=2n+1
  (mid)   held-out accuracy vs B, rising with the intended budget n
  (right) coherent and classical knees B0 vs matched budget q (log-log)

Run (after frust_analysis.py has built the caches):
    python scripts/plot_capacity_knee.py
"""
import os, json, math, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

BS = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 704, 768, 896, 1024, 1216]
NS = [1, 2, 3, 4, 5, 6]
N = 120


def gacc(p, n):
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    th = math.asin(math.sqrt(p)); k = min(n, max(0, round(math.pi / (4 * th) - 0.5)))
    return math.sin((2 * k + 1) * th) ** 2


def pstar(n):
    return math.sin(math.pi / (2 * (2 * n + 1))) ** 2


def knee(bs, acc, thr=0.9):
    ms = np.log2(bs)
    for i in range(len(ms) - 1):
        if acc[i] >= thr > acc[i + 1]:
            t = (acc[i] - thr) / (acc[i] - acc[i + 1]); return 2 ** (ms[i] + t * (ms[i + 1] - ms[i]))
    return 2 ** ms[-1] if acc[-1] >= thr else 2 ** ms[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "_sweep_out", "p_cache_bigB.json"))
    ap.add_argument("--classical", default=os.path.join(HERE, "_sweep_out", "classical_capacity_controlled.json"))
    ap.add_argument("--scaleup", default=os.path.join(HERE, "_sweep_out", "capacity_scaleup_16.json"))
    ap.add_argument("--out", default="figs/S12_capacity_knee.png")
    a = ap.parse_args()
    if not os.path.exists(a.cache):
        raise SystemExit(f"cache {a.cache} not found; run "
                         "scripts/frust_analysis.py first")
    cache = json.load(open(a.cache))

    # per (n,B): aggregate over seeds; per (n,seed): a curve, for knee error bars
    tr = {(n, B): [] for n in NS for B in BS}
    va = {(n, B): [] for n in NS for B in BS}
    per_seed = {n: {s: {} for s in (1, 2, 3, 4)} for n in NS}
    for k, v in cache.items():
        n, s, B = map(int, k.split("|"))
        if B in BS and n in NS:
            tr[(n, B)] += [gacc(p, n) for p in v["tr"]]
            va[(n, B)] += [gacc(p, n) for p in v["va"]]
            per_seed[n][s][B] = float(np.mean([gacc(p, n) for p in v["tr"]]))

    def curve(D, n):
        bs = [B for B in BS if D[(n, B)]]
        return bs, [float(np.mean(D[(n, B)])) for B in bs]

    B0, B0err = [], []
    for n in NS:
        bs, y = curve(tr, n)
        B0.append(knee(bs, y))
        ks = [knee(sorted(per_seed[n][s]), [per_seed[n][s][B] for B in sorted(per_seed[n][s])])
              for s in (1, 2, 3, 4) if len(per_seed[n][s]) >= 2]
        B0err.append(float(np.std(ks)) if ks else 0.0)
    B0 = np.array(B0); B0err = np.array(B0err)
    q = np.array([2 * n + 1 for n in NS], float)
    aq, bq = np.polyfit(np.log(q), np.log(B0), 1)
    ap_, _ = np.polyfit(np.log([1 / pstar(n) for n in NS]), np.log(B0), 1)

    if not os.path.exists(a.classical):
        raise SystemExit(f"controlled classical summary {a.classical} not found")
    cla = json.load(open(a.classical))
    if not math.isclose(float(cla["threshold"]), 0.9):
        raise SystemExit("controlled classical summary must use the B0=0.9 threshold")
    CB0 = np.array([cla["budgets"][str(int(v))]["B0_mean_curve"] for v in q])
    CB0err = np.array([
        np.std(cla["budgets"][str(int(v))]["B0_per_seed"])
        for v in q
    ])
    ac = float(cla["fit"]["exponent"])
    bc = math.log(float(cla["fit"]["prefactor"]))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    cols = ["#E69F00", "#D55E00", "#CC79A7", "#0072B2", "#56B4E9", "#009E73"]
    marks = ["o", "s", "^", "D", "v", "P"]
    for axi, split, D in zip(ax[:2], ("training", "held-out"), (tr, va)):
        for n, c, marker in zip(NS, cols, marks):
            bs, y = curve(D, n)
            axi.plot(bs, y, marker=marker, ls="-", color=c, lw=1.8, ms=3.5,
                     label=rf"$q={2*n+1}$")
            if split == "training":
                classical = cla["budgets"][str(2 * n + 1)]
                axi.plot(classical["B"], classical["mean"], marker=marker, ls="--",
                         color=c, mfc="white", mew=1.0, lw=1.35, ms=3.5)
        axi.set_xscale("log", base=2)
        axi.set_xticks([1, 4, 16, 64, 256, 1024] if split == "training"
                       else [8, 32, 128, 512])
        axi.axhline(0.9, ls=":", color="0.6", lw=1)
        axi.set_xlabel(r"training questions $B$")
        axi.set_title(split); axi.grid(alpha=.3, which="both"); axi.set_ylim(-0.03, 1.05)
    ax[0].set_ylabel("training accuracy after read-out")
    ax[0].legend(fontsize=7.5, ncol=2, loc="lower left")

    axi = ax[2]
    axi.errorbar(q, B0, yerr=B0err, fmt="o", color="#7F3C8D", ms=6, capsize=3, zorder=3)
    axi.errorbar(q, CB0, yerr=CB0err, fmt="s", color="#11A579", ms=5.5, capsize=3, zorder=3)
    qq = np.linspace(q.min(), q.max(), 40)
    axi.plot(qq, np.exp(bq) * qq ** aq, "-", color="#7F3C8D", lw=1.6,
             label=rf"coherent: $B_0\propto q^{{{aq:.2f}}}$")
    axi.plot(qq, np.exp(bc) * qq ** ac, "--", color="#11A579", lw=1.6,
             label=rf"classical: $B_0\propto q^{{{ac:.2f}}}$")
    axi.set_xscale("log"); axi.set_yscale("log"); axi.set_xticks(q)
    axi.set_xticklabels([int(v) for v in q])
    axi.set_xlabel(r"matched query budget $q$"); axi.set_ylabel(r"capacity knee $B_0$")
    axi.set_title("matched-budget knee scaling"); axi.legend(fontsize=8); axi.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(a.out, dpi=150, bbox_inches="tight")

    # ── compact 2-panel capacity for the PRL figure.  Unlike the legacy
    # diagnostic above, this uses the 16-pool paired scale-up, its 0.95
    # 0.95 capacity knee, and bootstrap uncertainty on mean curves and knees.
    if not os.path.exists(a.scaleup):
        raise SystemExit(f"16-pool capacity summary {a.scaleup} not found")
    scaleup = json.load(open(a.scaleup))
    threshold = float(scaleup["metadata"]["threshold"])
    if not math.isclose(threshold, 0.95):
        raise SystemExit("main-text capacity summary must use the B0=0.95 threshold")
    B0 = np.array([
        scaleup["quantum"]["curves"][str(int(v))]["B0_mean_curve"] for v in q
    ])
    CB0 = np.array([
        scaleup["classical"]["curves"][str(int(v))]["B0_mean_curve"] for v in q
    ])

    def bootstrap_yerr(method, centers):
        intervals = scaleup["bootstrap"][f"{method}_B0"]
        low = np.array([intervals[str(int(v))]["68_low"] for v in q])
        high = np.array([intervals[str(int(v))]["68_high"] for v in q])
        return np.vstack((centers - low, high - centers))

    B0err = bootstrap_yerr("quantum", B0)
    CB0err = bootstrap_yerr("classical", CB0)
    aq = float(scaleup["quantum"]["fit"]["exponent"])
    bq = math.log(float(scaleup["quantum"]["fit"]["prefactor"]))
    ac = float(scaleup["classical"]["fit"]["exponent"])
    bc = math.log(float(scaleup["classical"]["fit"]["prefactor"]))
    ap_, _ = np.polyfit(np.log([1 / pstar(n) for n in NS]), np.log(B0), 1)

    # The training plateau+knee and the knee-scaling law.  The held-out middle panel
    # is dropped because the PRL's deep-n figure already carries held-out vs B.
    figc, axc = plt.subplots(1, 2, figsize=(9.2, 3.6),
                            gridspec_kw={"width_ratios": [0.94, 1.06]})
    # Keep only representative low, middle, and high budgets in the curve
    # panel.  All six budgets remain in the scaling panel and its fit.
    shown_ns = [1, 3, 6]
    shown_cols = ["#0072B2", "#D55E00", "#009E73"]
    shown_marks = ["o", "s", "^"]
    for n, c, marker in zip(shown_ns, shown_cols, shown_marks):
        qv = str(2 * n + 1)
        coherent = scaleup["quantum"]["curves"][qv]
        coherent_band = scaleup["bootstrap"]["quantum_curve_mean"][qv]
        bs = np.asarray(coherent["B"])
        y = np.asarray(coherent["mean"])
        qlo = np.asarray(coherent_band["68_low"])
        qhi = np.asarray(coherent_band["68_high"])
        qerr = np.vstack((y - qlo, qhi - y))
        axc[0].errorbar(bs, y, yerr=qerr, fmt="none", ecolor=c,
                        elinewidth=0.8, capsize=1.8, capthick=0.8,
                        alpha=0.8, zorder=1)
        axc[0].plot(bs, y, marker=marker, ls="-", color=c, lw=1.8, ms=4.0,
                    zorder=2)
        classical = scaleup["classical"]["curves"][qv]
        classical_band = scaleup["bootstrap"]["classical_curve_mean"][qv]
        cbs = np.asarray(classical["B"])
        cy = np.asarray(classical["mean"])
        clo = np.asarray(classical_band["68_low"])
        chi = np.asarray(classical_band["68_high"])
        keep = cbs >= 8
        cerr = np.vstack((cy - clo, chi - cy))
        axc[0].errorbar(cbs[keep], cy[keep], yerr=cerr[:, keep], fmt="none",
                        ecolor=c, elinewidth=0.7, capsize=1.8, capthick=0.7,
                        alpha=0.75, zorder=1)
        axc[0].plot(cbs[keep], cy[keep], marker=marker, ls="--",
                    color=c, mfc="white", mew=1.1, lw=1.35, ms=4.0,
                    zorder=2)
    axc[0].set_xscale("log", base=2)
    axc[0].set_xlim(7.2, 1400)
    axc[0].set_xticks([8, 32, 128, 512])
    axc[0].set_xticklabels([8, 32, 128, 512])
    axc[0].axhline(threshold, ls=":", color="0.6", lw=1)
    axc[0].text(8.3, threshold + 0.005, f"{threshold:.2f}", fontsize=8.5,
                color="0.45", va="bottom")
    axc[0].set_xlabel(r"training-set size $B$")
    axc[0].set_ylabel("training accuracy after inference")
    axc[0].grid(alpha=.3, which="both"); axc[0].set_ylim(0, 1)
    budget_handles = [
        Line2D([0], [0], color=c, marker=m, lw=1.5, ms=4.5, label=rf"${int(qv)}$")
        for qv, c, m in zip([2 * n + 1 for n in shown_ns], shown_cols, shown_marks)
    ]
    budget_legend = axc[0].legend(handles=budget_handles, title=r"circuit applications $q$",
                                  fontsize=9.5, title_fontsize=9.5,
                                  handlelength=1.4, columnspacing=1.0,
                                  ncol=3, loc="lower left", frameon=False)
    axc[0].add_artist(budget_legend)
    method_handles = [
        Line2D([0], [0], color="0.3", ls="-", marker="o", ms=4.0, lw=1.8,
               label="QuCoNet"),
        Line2D([0], [0], color="0.3", ls="--", marker="o", mfc="white",
               mew=1.1, ms=4.0, lw=1.35, label=r"CoNet best-of-$q$"),
    ]
    axc[0].legend(handles=method_handles, fontsize=9.5, loc="lower right",
                  frameon=False)
    axc[1].errorbar(q, B0, yerr=B0err, fmt="o", color="#7F3C8D", ms=6,
                    capsize=3, zorder=3)
    axc[1].errorbar(q, CB0, yerr=CB0err, fmt="s", color="0.4", ms=5.5,
                    capsize=3, zorder=3)
    axc[1].plot(qq, np.exp(bq) * qq ** aq, "-", color="#7F3C8D", lw=1.7,
                label=rf"QuCoNet: $B_0\propto q^{{{aq:.2f}}}$")
    axc[1].plot(qq, np.exp(bc) * qq ** ac, "--", color="0.4", lw=1.7,
                label=rf"CoNet: $B_0\propto q^{{{ac:.2f}}}$")
    inverse_target = 1.0 / np.sin(np.pi / (2.0 * q)) ** 2
    target_scale = float(np.exp(np.mean(np.log(B0 / inverse_target))))
    inverse_target_qq = 1.0 / np.sin(np.pi / (2.0 * qq)) ** 2
    axc[1].plot(qq, target_scale * inverse_target_qq, ":", color="0.15", lw=1.4,
                label=r"$1/p^*$ (scaled)")
    axc[1].set_xscale("log"); axc[1].set_yscale("log"); axc[1].set_xticks(q)
    axc[1].set_xticklabels([int(v) for v in q])
    axc[1].xaxis.set_minor_formatter(plt.NullFormatter())
    axc[1].set_xlabel(r"circuit applications $q$"); axc[1].set_ylabel(r"capacity threshold $B_0$")
    axc[1].legend(fontsize=9.5, loc="upper left", frameon=False)
    axc[1].grid(alpha=.3, which="both")
    axc[0].text(-0.10, 1.07, "a", transform=axc[0].transAxes, fontsize=13,
                fontweight="bold", va="top")
    axc[1].text(-0.13, 1.07, "b", transform=axc[1].transAxes, fontsize=13,
                fontweight="bold", va="top")
    figc.tight_layout()
    cout = os.path.join(HERE, "..", "figs", "Q4b_capacity.pdf")
    os.makedirs(os.path.dirname(cout), exist_ok=True)
    figc.savefig(cout, bbox_inches="tight")
    print(f"saved {cout}")
    print(f"B0 = {[round(b, 1) for b in B0]}")
    print(f"classical B0 = {[round(b, 1) for b in CB0]}")
    print(f"exponent (2n+1)={aq:.2f} ; (1/p*)={ap_:.2f} ; "
          f"p*B0={[round(pstar(n) * b, 1) for n, b in zip(NS, B0)]}")
    print(f"classical exponent={ac:.2f} ; capacity ratios="
          f"{[round(x, 2) for x in B0 / CB0]}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
