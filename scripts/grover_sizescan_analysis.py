"""Problem-size scan: does the trained-for-amplification advantage grow with N?

Reads the per-N p-caches written by grover_sweep_analysis.py
(--qa randreg / randreg240 / randreg480 / randreg960; family names inside
the caches are identical: grover_rr / qstd_rr / cbestk_exact_rr / cstd_exact_rr)
and reports, per (N, B):

  grover 1-shot / +AA1  vs  cbestkX-trained best@3 (matched 3-query budget),
  best@9, eps-ceiling, and the catch-up budget k* = min k such that
  classical best-of-k >= grover+AA1  (inf if not reached by k=10^7).

Data: randreg K=3 M=8, seeds 1-8 (N=960: seeds 1-4, classical only --
quantum runs OOM on the 48 GB card; N=480 quantum exists at B=32 only).

Run AFTER the per-N compute passes:
  python scripts/grover_sizescan_analysis.py
"""
import os, json, math
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_sweep_out")
CACHES = {
    120: "p_cache_rr.json",
    240: "p_cache_rr_N240.json",
    480: "p_cache_rr_N480.json",
    960: "p_cache_rr_N960.json",
}
N_LIST = [120, 240, 480, 960]
B_LIST = [32, 128]
KMAX = 10 ** 7


def pooled(cache, fam, B, split):
    ps = []
    for s in range(1, 9):
        ps += cache.get(f"{fam}|{s}|{B}|{split}", [])
    return np.asarray(ps)


def aa1(ps):
    if not len(ps):
        return np.nan
    out = []
    for p in ps:
        if p <= 0: out.append(0.0); continue
        if p >= 1: out.append(1.0); continue
        th = math.asin(math.sqrt(p))
        k = min(1, max(0, int(round(math.pi / (4 * th) - 0.5))))
        out.append(math.sin((2 * k + 1) * th) ** 2)
    return float(np.mean(out))


def bk(ps, k):
    return float(np.mean(1 - (1 - ps) ** k)) if len(ps) else np.nan


def ceil_(ps, eps=0.02):
    return float(np.mean(ps > eps)) if len(ps) else np.nan


def catchup(ps, target):
    """min k with best-of-k >= target; inf if not reached by KMAX."""
    if not len(ps) or not np.isfinite(target):
        return None
    if bk(ps, KMAX) < target:
        return math.inf
    lo, hi = 1, 1
    while bk(ps, hi) < target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if bk(ps, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def fk(k):
    if k is None: return "--"
    if k == math.inf: return ">1e7"
    return f"{k:,}"


def main():
    caches = {}
    for N, name in CACHES.items():
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            caches[N] = json.load(open(path))
        else:
            print(f"(cache missing, skipped: {name})")

    rows = {}   # (N, B, split) -> dict of metrics
    hdr = (f"{'N':>5} {'B':>4} {'split':>6} | {'grv 1sh':>8} {'grv+AA1':>8} | "
           f"{'cbX 1sh':>8} {'cbX b@3':>8} {'cbX b@9':>8} {'ceil':>6} | "
           f"{'catch-up k*':>12}")
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for split in ("train", "valid"):
        for N in N_LIST:
            if N not in caches:
                continue
            c = caches[N]
            for B in B_LIST:
                g = pooled(c, "grover_rr", B, split)
                cx = pooled(c, "cbestk_exact_rr", B, split)
                if not (len(g) or len(cx)):
                    continue
                t = aa1(g)
                ku = catchup(cx, t) if len(g) else None
                rows[(N, B, split)] = dict(
                    g1=float(np.mean(g)) if len(g) else np.nan, gA=t,
                    c1=float(np.mean(cx)) if len(cx) else np.nan,
                    b3=bk(cx, 3), b9=bk(cx, 9), ceil=ceil_(cx), k=ku)
                r = rows[(N, B, split)]
                print(f"{N:>5} {B:>4} {split:>6} | "
                      f"{r['g1']:>8.3f} {r['gA']:>8.3f} | "
                      f"{r['c1']:>8.3f} {r['b3']:>8.3f} {r['b9']:>8.3f} "
                      f"{r['ceil']:>6.3f} | {fk(r['k']):>12}")
        print("-" * len(hdr))

    # figure: advantage and catch-up vs N at B=32 (quantum exists at all N<=480)
    Ns = [N for N in (120, 240, 480) if N in caches]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, ("train", "valid")):
        gA = [rows[(N, 32, split)]["gA"] for N in Ns]
        b3 = [rows[(N, 32, split)]["b3"] for N in Ns]
        b9 = [rows[(N, 32, split)]["b9"] for N in Ns]
        ax.plot(Ns, gA, "-o", color="#d62728", lw=1.8,
                label="QuCoNet($A_1$) + 1 Grover (3 queries)")
        ax.plot(Ns, b3, "-v", color="#17becf", lw=1.8,
                label="CoNet best-2-trained (exact) + best@3")
        ax.plot(Ns, b9, "--x", color="#1f77b4", lw=1.4,
                label="CoNet best-2-trained (exact) + best@9")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ns); ax.set_xticklabels(Ns)
        ax.set_xlabel("problem size N")
        ax.set_title(f"{split}  (B=32, random-regular)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate")
    axes[1].legend(fontsize=8)
    fig.suptitle("Problem-size scan: matched-budget gap vs N")
    fig.tight_layout()
    f = os.path.join(OUT, "S7_size_scan.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}")

    json.dump({f"{N}|{B}|{s}": {k: (str(v) if v == math.inf else v)
                                for k, v in r.items()}
               for (N, B, s), r in rows.items()},
              open(os.path.join(OUT, "size_scan_rows.json"), "w"), indent=1)
    print("rows -> size_scan_rows.json")


if __name__ == "__main__":
    main()
