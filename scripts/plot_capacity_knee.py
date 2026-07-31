#!/usr/bin/env python3
"""Main-text capacity-knee figure (Fig. capacity): B0(n) ~ 1/p*(n).

Reads the cached per-question probabilities produced by
`scripts/frust_analysis.py` (the N=120 sliding-puzzle
large-B Grover-n sweep, n=1..6; n=5,6 extended to B=1024/1216 so the n=6
capacity knee is resolved off the grid edge) and renders the three-panel
figure used in the main text:
  (left)  training accuracy <A_n> vs B, plateau + capacity knee, with the
          classical best-of-3 baseline (no plateau) for contrast
  (mid)   held-out accuracy vs B, rising with the intended budget n
  (right) knee B0 vs budget 2n+1 (log-log), error bars over seeds,
          fit ~ (2n+1)^2 ~ 1/p*(n)

Run from notes/ (after frust_analysis.py has built the caches):
    python plot_capacity_knee.py
"""
import os, json, math, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    ap.add_argument("--classical", default=os.path.join(HERE, "_sweep_out", "classical_bstar.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "figs", "S12_capacity_knee.png"))
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

    cla = json.load(open(a.classical)) if os.path.exists(a.classical) else None

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    cols = ["#f4a582", "#d62728", "#7f0000", "#9467bd", "#6a0dad", "#4b0082"]
    for axi, split, D in zip(ax[:2], ("training", "held-out"), (tr, va)):
        for n, c in zip(NS, cols):
            bs, y = curve(D, n)
            axi.plot(bs, y, "o-", color=c, lw=1.8, ms=3.5, label=f"$n={n}$")
        axi.set_xscale("log", base=2); axi.set_xticks([8, 32, 128, 512])
        axi.set_xticklabels([8, 32, 128, 512])
        axi.axhline(0.9, ls=":", color="0.6", lw=1)
        axi.set_xlabel("training questions $B$")
        axi.set_title(split); axi.grid(alpha=.3, which="both"); axi.set_ylim(-0.03, 1.05)
    # classical best-of-3 baseline on the training panel: no plateau
    if cla:
        cb = sorted((int(B), v[1]) for B, v in cla.items())
        ax[0].plot([b for b, _ in cb], [y for _, y in cb], "s--", color="0.5",
                   lw=1.4, ms=3, label="classical bo3")
    ax[0].set_ylabel(r"accuracy after $n$ rounds  $\langle A_n\rangle$")
    ax[0].legend(fontsize=7.5, ncol=2, loc="lower left")

    axi = ax[2]
    axi.errorbar(q, B0, yerr=B0err, fmt="o", color="#4b0082", ms=6, capsize=3, zorder=3)
    qq = np.linspace(q.min(), q.max(), 40)
    axi.plot(qq, np.exp(bq) * qq ** aq, "-", color="#4b0082", lw=1.6,
             label=r"$B_0\propto(2n{+}1)^{%.1f}$" % aq)
    axi.plot(qq, B0[0] * (qq / q[0]) ** 2, "--", color="0.55", lw=1.1, label="slope 2")
    axi.set_xscale("log"); axi.set_yscale("log"); axi.set_xticks(q)
    axi.set_xticklabels([int(v) for v in q])
    axi.set_xlabel("budget $2n+1$"); axi.set_ylabel("capacity knee $B_0$")
    axi.set_title("knee scaling"); axi.legend(fontsize=8); axi.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(a.out, dpi=150, bbox_inches="tight")

    # ── compact 2-panel capacity figure for the main text: the
    # training plateau+knee and the knee-scaling law.  The held-out middle panel
    # is dropped because the PRL's deep-n figure already carries held-out vs B.
    figc, axc = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for n, c in zip(NS, cols):
        bs, y = curve(tr, n)
        axc[0].plot(bs, y, "o-", color=c, lw=1.8, ms=3.5, label=f"$n={n}$")
    if cla:
        axc[0].plot([b for b, _ in cb], [y for _, y in cb], "s--", color="0.5",
                    lw=1.4, ms=3, label="classical bo3")
    axc[0].set_xscale("log", base=2); axc[0].set_xticks([8, 32, 128, 512])
    axc[0].set_xticklabels([8, 32, 128, 512]); axc[0].axhline(0.9, ls=":", color="0.6", lw=1)
    axc[0].set_xlabel("training questions $B$")
    axc[0].set_ylabel(r"accuracy after $n$ rounds  $\langle A_n\rangle$")
    axc[0].set_title(r"plateau ends at a capacity knee $B_0(n)$")
    axc[0].grid(alpha=.3, which="both"); axc[0].set_ylim(-0.03, 1.05)
    axc[0].legend(fontsize=7.5, ncol=2, loc="lower left")
    axc[1].errorbar(q, B0, yerr=B0err, fmt="o", color="#4b0082", ms=6, capsize=3, zorder=3)
    axc[1].plot(qq, np.exp(bq) * qq ** aq, "-", color="#4b0082", lw=1.6,
                label=r"$B_0\propto(2n{+}1)^{%.1f}$" % aq)
    axc[1].plot(qq, B0[0] * (qq / q[0]) ** 2, "--", color="0.55", lw=1.1, label="slope 2")
    axc[1].set_xscale("log"); axc[1].set_yscale("log"); axc[1].set_xticks(q)
    axc[1].set_xticklabels([int(v) for v in q])
    axc[1].xaxis.set_minor_formatter(plt.NullFormatter())
    axc[1].set_xlabel("budget $2n+1$"); axc[1].set_ylabel("capacity knee $B_0$")
    axc[1].set_title(r"$B_0\propto(2n{+}1)^2\approx1/p^{*}(n)$")
    axc[1].legend(fontsize=8); axc[1].grid(alpha=.3, which="both")
    axc[0].text(-0.10, 1.07, "a", transform=axc[0].transAxes, fontsize=13,
                fontweight="bold", va="top")
    axc[1].text(-0.13, 1.07, "b", transform=axc[1].transAxes, fontsize=13,
                fontweight="bold", va="top")
    figc.tight_layout()
    cout = os.path.join(REPO, "figs", "Q4b_capacity.png")
    os.makedirs(os.path.dirname(cout), exist_ok=True)
    figc.savefig(cout, dpi=170, bbox_inches="tight")
    print(f"saved {cout}")
    print(f"B0 = {[round(b) for b in B0]} +/- {[round(e) for e in B0err]}")
    print(f"exponent (2n+1)={aq:.2f} ; (1/p*)={ap_:.2f} ; "
          f"p*B0={[round(pstar(n) * b, 1) for n, b in zip(NS, B0)]}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
