"""Exchange-rate / budget-scaling analysis (overnight3 ladders).

The question addressed: "if we train with best@2^n, how does it scale with n,
and how does it compare to Grover?"  Two views:

VIEW 1 -- "what a budget buys" (trained-for-budget ladders).  Each model is
trained for its OWN inference budget and evaluated at that budget:

  classical CoNet (cbestkX-b): acc = mean_i [1-(1-p_i)^b]   b in {2,4,8,16,32,64}
  semi-QuCoNet    (qbkX-b):    acc = mean_i [1-(1-p_i)^b]    same arch, AR p
  Grover-QuCoNet  (grover-n):  acc = mean_i [A_n(p_i)]       b=2n+1, n in 1..6

VIEW 2 -- the MECHANISM / exchange rate.  On ONE fixed small-p distribution
(the untrained uniform walk), sweep the inference budget b and transform two
ways: incoherent resampling 1-(1-p)^b vs coherent amplification A_{(b-1)/2}(p).
In the unsaturated regime resampling buys acc ~ b^1 while amplification buys
acc ~ b^2: a quadratic exchange rate for test-time compute.  Fit both exponents
and report the budget ratio at matched accuracy.

Run: python scripts/grover_exchange_rate.py
"""
import os, sys, json, math
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grover_utility_scaling as gus          # reuse aa, A, U, uniform_p, CACHE

CACHE = gus.CACHE
OUT = gus.OUT

# trained-for-budget ladders: budget -> cache family key
CLASS = {2: "cbestk_exact", 4: "cbestkX4", 8: "cbestkX8",
         16: "cbestkX16", 32: "cbestkX32", 64: "cbestkX64"}
SEMI = {2: "qbkX2", 4: "qbkX4", 8: "qbkX8",
        16: "qbkX16", 32: "qbkX32", 64: "qbkX64"}
GROV = {3: ("grover", 1), 5: ("grover2", 2), 7: ("grover3", 3),
        9: ("grover4", 4), 11: ("grover5", 5), 13: ("grover6", 6)}
BS = (8, 32, 128)


def pooled(fam, B, split, smax=8):
    ps = []
    for s in range(1, smax + 1):
        ps += CACHE.get(f"{fam}|{s}|{B}|{split}", [])
    return np.asarray(ps)


def per_seed_resample(fam, b, B, split, smax=8):
    """best-of-b accuracy, one value per seed (for spread)."""
    v = []
    for s in range(1, smax + 1):
        ps = np.asarray(CACHE.get(f"{fam}|{s}|{B}|{split}", []))
        if len(ps):
            v.append(gus.U(ps, b))
    return np.asarray(v)


def per_seed_grover(fam, n, B, split, smax=8):
    v = []
    for s in range(1, smax + 1):
        ps = np.asarray(CACHE.get(f"{fam}|{s}|{B}|{split}", []))
        if len(ps):
            v.append(gus.A(ps, n))
    return np.asarray(v)


def ladder(B, split):
    cl = {b: gus.U(pooled(CLASS[b], B, split), b) for b in CLASS}
    se = {b: gus.U(pooled(SEMI[b], B, split), b) for b in SEMI}
    gr = {b: gus.A(pooled(f, B, split), n) for b, (f, n) in GROV.items()}
    return cl, se, gr


def fit_exponent(bs, acc, lo=0.02, hi=0.85):
    """log-log slope over the unsaturated, above-floor band."""
    bs, acc = np.asarray(bs, float), np.asarray(acc, float)
    m = np.isfinite(acc) & (acc > lo) & (acc < hi)
    if m.sum() < 2:
        return np.nan, np.nan
    a, c = np.polyfit(np.log(bs[m]), np.log(acc[m]), 1)
    return a, m.sum()


def budget_at(acc_by_b, target):
    """Smallest budget reaching >=target, linear-interp in log-budget."""
    bs = sorted(acc_by_b)
    xs = [math.log(b) for b in bs]
    ys = [acc_by_b[b] for b in bs]
    for i in range(1, len(bs)):
        if ys[i] >= target > ys[i - 1] and ys[i] > ys[i - 1]:
            t = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
            return math.exp(xs[i - 1] + t * (xs[i] - xs[i - 1]))
    if ys[0] >= target:
        return bs[0]
    return np.inf


# ─────────────────────── VIEW 1: trained-for-budget ────────────────────
def view1():
    print("\n" + "=" * 74)
    print("VIEW 1  trained-for-budget ladders: acc at the trained budget")
    print("=" * 74)
    for split in ("train", "valid"):
        for B in BS:
            cl, se, gr = ladder(B, split)
            print(f"\n[{split}  B={B}]  budget: acc")
            print("  classical CoNet  " +
                  " ".join(f"{b}:{cl[b]:.3f}" for b in sorted(cl)))
            print("  semi-QuCoNet     " +
                  " ".join(f"{b}:{se[b]:.3f}" for b in sorted(se)))
            print("  Grover-QuCoNet   " +
                  " ".join(f"{b}:{gr[b]:.3f}" for b in sorted(gr)))
            # exponents (trained-for-budget; muddier than fixed-p but honest)
            ac, _ = fit_exponent(list(cl), [cl[b] for b in sorted(cl)])
            ae, _ = fit_exponent(list(se), [se[b] for b in sorted(se)])
            ag, _ = fit_exponent(list(gr), [gr[b] for b in sorted(gr)])
            print(f"  exponent(acc~b^a): classical {ac:+.2f}  semi {ae:+.2f}"
                  f"  Grover {ag:+.2f}")


# ─────────────────────── VIEW 2: fixed-p exchange rate ──────────────────
def view2():
    print("\n" + "=" * 74)
    print("VIEW 2  fixed-p mechanism: untrained-walk p, resample vs amplify")
    print("=" * 74)
    uc = gus.uniform_p()

    def upooled(B, split):
        ps = []
        for s in range(1, 9):
            ps += uc.get(f"uniform|{s}|{B}|{split}", [])
        return np.asarray(ps)

    odd = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]     # quantum budgets 2n+1 (plot)
    cbud = np.unique(np.round(np.logspace(0, 3.5, 80)).astype(int))  # classical sweep
    # the untrained walk is B-independent (no training); use the pooled held-out
    # hardness distribution once.
    ps = np.concatenate([upooled(B, "valid") for B in BS])
    cl = [gus.U(ps, b) for b in odd]
    qm = [gus.A(ps, (b - 1) // 2) for b in odd]
    ac, _ = fit_exponent(odd, cl)
    aq, _ = fit_exponent(odd, qm)
    pbar = ps.mean()
    print(f"\n[held-out hardness]  mean p = {pbar:.3f}  1/sqrt(p_bar) = "
          f"{1/math.sqrt(pbar):.1f}   (n={len(ps)} held-out pairs, B-independent)")
    print("  budget b        " + " ".join(f"{b:>5}" for b in odd))
    print("  resample 1-(1-p)^b " + " ".join(f"{v:5.2f}" for v in cl))
    print("  amplify  A_n(p)    " + " ".join(f"{v:5.2f}" for v in qm))
    print(f"  exponent: resample {ac:+.2f}   amplify {aq:+.2f}")
    cl_ext = dict(zip(cbud.tolist(), [gus.U(ps, int(b)) for b in cbud]))
    qm_ext = dict(zip(odd, qm))
    xr = {}
    for tgt in (0.5, 0.7, 0.9):
        bc = budget_at(cl_ext, tgt)
        bq = budget_at(qm_ext, tgt)
        rr = bc / bq if np.isfinite(bc) and np.isfinite(bq) and bq > 0 else np.inf
        thr = (-math.log(1 - tgt) / math.asin(math.sqrt(tgt))) / math.sqrt(pbar)
        xr[tgt] = (bc, bq, rr, thr)
        print(f"    reach acc {tgt:.1f}: classical b={bc:7.1f}  "
              f"quantum b={bq:5.1f}  exchange rate x{rr:5.1f}  (theory x{thr:.1f})")
    return dict(ps=ps, odd=odd, cl=cl, qm=qm, ac=ac, aq=aq, pbar=pbar,
                cl_ext=cl_ext, qm_ext=qm_ext, xr=xr)


# ─────────────────────── figures ───────────────────────────────────────
def fig_view1():
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), sharex=True, sharey=True)
    for i, split in enumerate(("train", "valid")):
        for j, B in enumerate(BS):
            ax = axes[i][j]
            cl, se, gr = ladder(B, split)
            for d, col, mk, lab in (
                    (cl, "#1f77b4", "x", "classical CoNet, best-of-$b$"),
                    (se, "#17becf", "^", "semi-QuCoNet, best-of-$b$ (same arch)"),
                    (gr, "#d62728", "s", "Grover-QuCoNet, $A_n$ ($b{=}2n{+}1$)")):
                bs = sorted(d)
                ax.plot(bs, [d[b] for b in bs], mk + "-", color=col,
                        ms=6, lw=1.9, label=lab)
            ax.set_xscale("log", base=2)
            ax.set_title(f"{split}  B={B}", fontsize=10)
            ax.grid(alpha=0.3, which="both")
            if i == 1:
                ax.set_xlabel("inference budget $b$ (queries)")
            if j == 0:
                ax.set_ylabel("accuracy at budget")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("What a budget buys: each model trained for its own budget "
                 "$b$, evaluated at $b$ (sliding puzzle, $N{=}120$)")
    fig.tight_layout()
    f = os.path.join(OUT, "S10_budget_ladders.png")
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"fig -> {f}")


def fig_view2(r):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    # (A) fixed-p transform curves, fitted exponents
    axL.plot(r["odd"], r["cl"], "x-", color="#1f77b4", lw=1.9,
             label=f"resample $1-(1-p)^b$  ($\\alpha={r['ac']:.2f}$)")
    axL.plot(r["odd"], r["qm"], "s-", color="#d62728", lw=2.1,
             label=f"amplify $A_n(p)$  ($\\alpha={r['aq']:.2f}$)")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("budget $b$ (queries)")
    axL.set_ylabel("accuracy (untrained-walk $p$)")
    axL.set_title(f"Same $p$, two ways to spend budget ($\\bar p={r['pbar']:.3f}$)",
                  fontsize=10)
    axL.grid(alpha=0.3, which="both")
    axL.legend(fontsize=8.5, loc="lower right")
    # (B) exchange rate vs target accuracy: empirical + closed form 1/sqrt(p)
    As = np.linspace(0.1, 0.97, 40)
    emp, thr = [], []
    for A in As:
        bc = budget_at(r["cl_ext"], A)
        bq = budget_at(r["qm_ext"], A)
        emp.append(bc / bq if np.isfinite(bc) and np.isfinite(bq) and bq > 0 else np.nan)
        thr.append((-math.log(1 - A) / math.asin(math.sqrt(A))) / math.sqrt(r["pbar"]))
    axR.plot(As, thr, "-", color="0.4", lw=1.8,
             label="closed form $\\dfrac{-\\ln(1-A)}{\\arcsin\\!\\sqrt{A}}\\,p^{-1/2}$")
    axR.plot(As, emp, "o", color="#d62728", ms=4, label="measured (held-out)")
    for A, (bc, bq, rr, th) in r["xr"].items():
        axR.annotate(f"$\\times${rr:.0f}", (A, rr), fontsize=8,
                     xytext=(3, 3), textcoords="offset points")
    axR.set_xlabel("target accuracy $A$")
    axR.set_ylabel("exchange rate  $b_{\\rm classical}/b_{\\rm quantum}$")
    axR.set_title(f"Quantum queries buy more ($1/\\sqrt{{\\bar p}}={1/math.sqrt(r['pbar']):.0f}$)",
                  fontsize=10)
    axR.grid(alpha=0.3)
    axR.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Exchange rate: incoherent resampling ($\\alpha{\\approx}1$) vs "
                 "coherent amplification ($\\alpha{\\approx}2$) on hard instances")
    fig.tight_layout()
    f = os.path.join(OUT, "S11_exchange_exponent.png")
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"fig -> {f}")


def fig_main():
    """Compact main-text figure: held-out ladders only (the scaling law)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    for j, B in enumerate(BS):
        ax = axes[j]
        cl, se, gr = ladder(B, "valid")
        ac, _ = fit_exponent(list(cl), [cl[b] for b in sorted(cl)])
        ae, _ = fit_exponent(list(se), [se[b] for b in sorted(se)])
        ag, _ = fit_exponent(list(gr), [gr[b] for b in sorted(gr)])
        for d, col, mk, lab in (
                (cl, "#1f77b4", "x", f"classical CoNet, best-of-$b$ ($\\alpha{{=}}{ac:.1f}$)"),
                (se, "#17becf", "^", f"semi-QuCoNet, best-of-$b$ ($\\alpha{{=}}{ae:.1f}$)"),
                (gr, "#d62728", "s", f"Grover-QuCoNet, $A_n$ ($\\alpha{{=}}{ag:.1f}$)")):
            bs = sorted(d)
            ax.plot(bs, [d[b] for b in bs], mk + "-", color=col, ms=6, lw=2.0, label=lab)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_title(f"held-out, $B={B}$", fontsize=10)
        ax.set_xlabel("inference budget $b$ (queries)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("held-out accuracy at budget")
    fig.tight_layout()
    f = os.path.join(OUT, "S10b_budget_ladders_valid.png")
    fig.savefig(f, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"fig -> {f}")


if __name__ == "__main__":
    view1()
    rows = view2()
    fig_view1()
    fig_view2(rows)
    fig_main()
