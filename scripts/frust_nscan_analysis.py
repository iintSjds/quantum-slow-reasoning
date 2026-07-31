"""N-scaling of the frustration capacity knee B0(n): does B0 ∝ N?

At N=120 (sliding) the knee obeys B0 ~ 1/p*(n), p*.B0 ~ N/15.  This reads the
random-3-regular N-scan (nscan_sweep.sh, N in {120,240,480,960}, distance-6,
M=8) and tests whether B0 ∝ N -- i.e. whether B0/N depends ONLY on n, so that
acc-vs-(B/N) curves collapse across N.  If they do, the frustration law is
size- and topology-independent (the N=120-randreg point also differs in
topology from the N=120 sliding puzzle).

Run: python scripts/frust_nscan_analysis.py
"""
import os, sys, glob, math, json
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DATAROOT = os.environ.get("QSR_ROOT", REPO)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(REPO, "conet"))
import analyze_path_diversity as apd

ROOT = os.path.join(DATAROOT, "from4090", "grover_sweep_nscan")
QADIR = os.path.join(DATAROOT, "from4090", "expr4", "graph_qa")
M, K = 8, 3
SEEDS = (1, 2, 3, 4)
# N -> (poolB, n_list, B_grid)   [same B/N grid across N by construction]
CONFIG = {
    120: (384,  [1, 2, 3], [8, 16, 32, 64, 96, 128, 192, 256, 320]),
    240: (704,  [1, 2, 3], [16, 32, 64, 128, 192, 256, 384, 512, 640]),
    480: (1344, [1, 2, 3], [32, 64, 128, 256, 384, 512, 768, 1024, 1280]),
    960: (1600, [1, 2],    [64, 128, 256, 512, 768, 1024, 1536]),
}
NS = sorted(CONFIG)
ALL_N = [1, 2, 3]


def grover_acc(p, n):
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    th = math.asin(math.sqrt(p)); k = min(n, max(0, round(math.pi / (4 * th) - 0.5)))
    return math.sin((2 * k + 1) * th) ** 2


def pistar(n):
    return math.sin(math.pi / (2 * (2 * n + 1))) ** 2


CACHE = os.path.join(HERE, "_sweep_out", "p_cache_nscan.json")
os.makedirs(os.path.dirname(CACHE), exist_ok=True)
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def eval_ckpt(ck, qa):
    C, snm, scm, N_, K_, _ = apd.load_quantum_checkpoint(ck)
    return [float(apd.compute_quantum_path_diversity(C, snm, scm, int(Q), int(A), N_, K_, M)[3])
            for (Q, A) in qa]


def collect():
    import torch
    atr = {}   # (N,n,B) -> list of A_n over train pairs (pooled across seeds)
    ava = {}
    for N in NS:
        poolB, nlist, bgrid = CONFIG[N]
        for n in nlist:
            for B in bgrid:
                atr[(N, n, B)] = []; ava[(N, n, B)] = []
                for seed in SEEDS:
                    cks = glob.glob(
                        f"{ROOT}/**/grover_n{n}_s{seed}_B{B}_randreg_N{N}_*_D6_*_best_model.pt",
                        recursive=True)
                    if not cks:
                        continue
                    key = f"{N}|{n}|{seed}|{B}"
                    if key in cache:
                        ptr, pva = cache[key]["tr"], cache[key]["va"]
                    else:
                        d = torch.load(f"{QADIR}/randreg_N{N}_K3_M8_B{poolB}_D6_seed{seed}.pt",
                                       weights_only=False)
                        qa = d["qa_pairs"]
                        ptr = eval_ckpt(cks[0], qa[:B])
                        pva = eval_ckpt(cks[0], qa[-64:])
                        cache[key] = {"tr": ptr, "va": pva}
                    atr[(N, n, B)] += [grover_acc(p, n) for p in ptr]
                    ava[(N, n, B)] += [grover_acc(p, n) for p in pva]
        json.dump(cache, open(CACHE, "w"))
        print(f"  N={N} cached")
    return atr, ava


def knee(bs, acc, thr=0.9):
    ms = np.log2(bs)
    for i in range(len(ms) - 1):
        if acc[i] >= thr > acc[i + 1]:
            t = (acc[i] - thr) / (acc[i] - acc[i + 1])
            return ms[i] + t * (ms[i + 1] - ms[i])
    return ms[-1] if acc[-1] >= thr else ms[0]


atr, ava = collect()

# ---- B0(n,N) table ----
B0 = {}
print("\n" + "=" * 78)
print("acc(B, 2n+1) TRAIN + capacity knee B0(n,N)")
print("=" * 78)
for N in NS:
    poolB, nlist, bgrid = CONFIG[N]
    print(f"\nN={N}  (B/N grid: {[round(b/N,2) for b in bgrid]})")
    print(f"{'n':>2} " + " ".join(f"{round(b/N,2):>5}" for b in bgrid) + f" | {'B0':>5} {'B0/N':>5} {'p*B0':>5}")
    for n in nlist:
        tr = [float(np.mean(atr[(N, n, B)])) if atr[(N, n, B)] else np.nan for B in bgrid]
        k = knee(bgrid, tr); b0 = 2 ** k; B0[(N, n)] = b0
        print(f"{n:>2} " + " ".join(f"{v:5.2f}" for v in tr) +
              f" | {b0:5.0f} {b0/N:5.2f} {pistar(n)*b0:5.2f}")

# ---- the test: B0 ∝ N?  (B0/N vs n should coincide across N) ----
print("\n" + "=" * 78)
print("B0/N by (n, N)  -- coincidence across N  <=>  B0 ∝ N")
print("=" * 78)
print(f"{'n':>2} {'2n+1':>4} " + " ".join(f"N{N:<5}" for N in NS) + f" {'slope_B0~N':>10}")
for n in ALL_N:
    row = []
    for N in NS:
        row.append(B0.get((N, n), np.nan))
    # slope of log B0 vs log N (where defined)
    ok = [(N, b) for N, b in zip(NS, row) if np.isfinite(b)]
    if len(ok) >= 2:
        xs = np.log([N for N, _ in ok]); ys = np.log([b for _, b in ok])
        sl = np.polyfit(xs, ys, 1)[0]
    else:
        sl = np.nan
    print(f"{n:>2} {2*n+1:>4} " + " ".join(f"{(b/N):5.2f}" if np.isfinite(b) else "  -  "
          for N, b in zip(NS, row)) + f" {sl:>10.2f}")
# global invariant p*.B0/N
vals = [pistar(n) * B0[(N, n)] / N for N in NS for n in CONFIG[N][1] if (N, n) in B0]
print(f"\np*.B0/N  (should be ~const if the law holds): "
      f"mean={np.mean(vals):.3f}  sd={np.std(vals):.3f}  range=[{min(vals):.3f},{max(vals):.3f}]")

# ---- figure: acc-vs-(B/N) collapse per n + B0/N-vs-n coincidence ----
fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
ncol = {120: "#1f77b4", 240: "#2ca02c", 480: "#ff7f0e", 960: "#d62728"}
for ax, n in zip(axes[:3], ALL_N):
    for N in NS:
        if n not in CONFIG[N][1]:
            continue
        bgrid = CONFIG[N][2]
        y = [float(np.mean(atr[(N, n, B)])) if atr[(N, n, B)] else np.nan for B in bgrid]
        ax.plot([b / N for b in bgrid], y, "o-", color=ncol[N], lw=1.8, ms=4, label=f"N={N}")
    ax.axhline(0.9, ls=":", color="0.6", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("B / N"); ax.set_title(f"$n={n}$ (budget {2*n+1})")
    ax.grid(alpha=0.3, which="both")
    if n == 1:
        ax.set_ylabel("accuracy after $n$ rounds"); ax.legend(fontsize=8)
ax = axes[3]
for N in NS:
    ns = CONFIG[N][1]
    ax.plot([2 * n + 1 for n in ns], [B0[(N, n)] / N for n in ns],
            "o-", color=ncol[N], lw=1.8, ms=5, label=f"N={N}")
ax.set_xscale("log"); ax.set_yscale("log")
qs = [3, 5, 7]; ax.set_xticks(qs); ax.set_xticklabels(qs)
ax.set_xlabel("query budget $2n+1$"); ax.set_ylabel("$B_0/N$")
ax.set_title("$B_0/N$ vs $n$  (coincide $\\Leftrightarrow B_0\\propto N$)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.suptitle("N-scaling of the frustration knee: acc-vs-(B/N) collapse across an "
             "8x N ladder (random 3-regular, distance-6)")
fig.tight_layout()
f = os.path.join(HERE, "_sweep_out", "S13_nscan.png")
fig.savefig(f, dpi=150, bbox_inches="tight")
print(f"\nfig -> {f}")
