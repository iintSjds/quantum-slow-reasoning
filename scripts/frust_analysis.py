"""Frustration-scaling analysis: acc(B, 2n+1) and the capacity knee m0(n).

From the B768 large-B sweep (grover-n=1..4, B in {8..256} on the same N=120
sliding puzzle), compute realized accuracy after n rounds = mean_i A_n(p_i) on
train and held-out, per B, and fit the frustration knee m0(n) where the train
plateau ends -- to see whether training for a deeper amplification budget buys
tolerance to capacity contention (higher m0), and whether the falloff slope is
universal.

Run: python scripts/frust_analysis.py
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

ROOT = os.path.join(DATAROOT, "from4090", "grover_sweep_bigB")
ROOT_EXT = os.path.join(DATAROOT, "from4090", "grover_sweep_ext")        # large-B extension (B1280_D6 pool)
QADIR = os.path.join(DATAROOT, "from4090", "expr4", "graph_qa")
BS_OLD = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 704]
BS_NEW = [768, 896, 1024, 1216]               # extend n=5,6 past the censored edge
BS = BS_OLD + BS_NEW
NS = [1, 2, 3, 4, 5, 6]
N, M = 120, 8


def grover_acc(p, n):
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    th = math.asin(math.sqrt(p)); k = min(n, max(0, round(math.pi / (4 * th) - 0.5)))
    return math.sin((2 * k + 1) * th) ** 2


def pistar(n):
    return math.sin(math.pi / (2 * (2 * n + 1))) ** 2


CACHE = os.path.join(HERE, "_sweep_out", "p_cache_bigB.json")
os.makedirs(os.path.dirname(CACHE), exist_ok=True)
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def eval_ckpt(ck, qa, K):
    """p_i for each (Q,A) via the validated quantum enumerator."""
    C, snm, scm, N_, K_, _ = apd.load_quantum_checkpoint(ck)
    return [float(apd.compute_quantum_path_diversity(C, snm, scm, int(Q), int(A), N_, K_, M)[3])
            for (Q, A) in qa]


def pool_for(B, seed):
    """(search root, QA pool file) for a given B. B<=704 lives on the original
    B768 pool in ROOT; larger B on the extended B1280 pool in ROOT_EXT whose
    first 768 pairs (and last-64 held-out) are byte-identical, so curves stitch."""
    if B in BS_OLD:
        return ROOT, f"{QADIR}/sliding_puzzle_N120_K3_M8_B768_D6_seed{seed}.pt"
    return ROOT_EXT, f"{QADIR}/sliding_puzzle_N120_K3_M8_B1280_D6_seed{seed}.pt"


def collect():
    atr = {(n, B): [] for n in NS for B in BS}
    ava = {(n, B): [] for n in NS for B in BS}
    for n in NS:
        for B in BS:
            for seed in (1, 2, 3, 4):
                root, poolf = pool_for(B, seed)
                cks = glob.glob(f"{root}/**/grover_n{n}_s{seed}_B{B}_*_D6*_best_model.pt",
                                recursive=True)
                if not cks:
                    continue
                key = f"{n}|{seed}|{B}"
                if key in cache:
                    ptr, pva = cache[key]["tr"], cache[key]["va"]
                else:
                    d = __import__("torch").load(poolf, weights_only=False)
                    qa = d["qa_pairs"]
                    ptr = eval_ckpt(cks[0], qa[:B], 3)
                    pva = eval_ckpt(cks[0], qa[-64:], 3)
                    cache[key] = {"tr": ptr, "va": pva}
                atr[(n, B)] += [grover_acc(p, n) for p in ptr]
                ava[(n, B)] += [grover_acc(p, n) for p in pva]
        json.dump(cache, open(CACHE, "w"))
        print(f"  n={n} cached")
    return atr, ava


def knee(bs, acc, thr=0.9):
    """B where the train plateau drops below thr (log-B interp)."""
    ms = np.log2(bs)
    for i in range(len(ms) - 1):
        if acc[i] >= thr > acc[i + 1]:
            t = (acc[i] - thr) / (acc[i] - acc[i + 1])
            return ms[i] + t * (ms[i + 1] - ms[i])
    return ms[-1] if acc[-1] >= thr else ms[0]


def slope(bs, acc, lo=0.05, hi=0.95):
    ms = np.log2(bs); a = np.asarray(acc)
    m = (a > lo) & (a < hi)
    return np.polyfit(ms[m], a[m], 1)[0] if m.sum() >= 2 else np.nan


atr, ava = collect()
print("\n" + "=" * 74)
print("acc(B, 2n+1) = mean A_n after n rounds   [TRAIN]")
print("=" * 74)
print(f"{'n':>2} {'2n+1':>4} " + " ".join(f"B{b:<4}" for b in BS) +
      f" | {'m0(.9)':>6} {'B0':>5} {'B0/N':>5} {'slope':>6} {'p*(n)':>6}")
def mean_curve(D, n):
    """(Bs with data, mean A_n) for budget n -- ragged: n=1..4 stop at B=704."""
    Bs = [B for B in BS if D[(n, B)]]
    return Bs, [float(np.mean(D[(n, B)])) for B in Bs]


def per_seed_knees(n):
    """knee B0 per seed (for error bars), read straight from the cache."""
    ks = []
    for s in (1, 2, 3, 4):
        Bs, ys = [], []
        for B in BS:
            key = f"{n}|{s}|{B}"
            if key in cache:
                Bs.append(B)
                ys.append(float(np.mean([grover_acc(p, n) for p in cache[key]["tr"]])))
        if len(Bs) >= 2:
            ks.append(2 ** knee(Bs, ys))
    return ks


B0s, slopes, B0err = [], [], []
for n in NS:
    Bs, tr = mean_curve(atr, n)
    k = knee(Bs, tr); B0 = 2 ** k; s = slope(Bs, tr)
    ks = per_seed_knees(n); B0err.append(float(np.std(ks)) if ks else 0.0)
    B0s.append(B0); slopes.append(s)
    row = dict(zip(Bs, tr))
    print(f"{n:>2} {2*n+1:>4} " +
          " ".join(f"{row[b]:5.3f}" if b in row else "  -  " for b in BS) +
          f" | {B0:>5.0f}+/-{B0err[-1]:<3.0f} {B0/N:>5.2f} {s:>6.2f} {pistar(n):>6.3f}"
          f"  seeds={[round(x) for x in ks]}")

# ---- B0(n) exponent: is the capacity knee ~ (2n+1)^alpha or ~ 1/p*(n)? ----
q = np.array([2 * n + 1 for n in NS], float)         # query budget 2n+1
B0a = np.array(B0s, float)
a_q, b_q = np.polyfit(np.log(q), np.log(B0a), 1)      # B0 ~ q^a_q
pstar = np.array([pistar(n) for n in NS])
a_p, b_p = np.polyfit(np.log(1 / pstar), np.log(B0a), 1)   # B0 ~ (1/p*)^a_p
print("\n" + "=" * 74)
print("B0(n) capacity-knee scaling   (n=6 RESOLVED off the grid edge)")
print("=" * 74)
print(f"  B0 = {[round(b) for b in B0s]}  +/- {[round(e) for e in B0err]}")
print(f"  B0 vs (2n+1):   exponent = {a_q:.2f}   (1=linear, 2=quadratic)")
print(f"  B0 vs 1/p*(n):  exponent = {a_p:.2f}   (p* = sin^2(pi/(2(2n+1))))")
print(f"  p*(n)*B0(n) = {[round(pistar(n)*b, 1) for n, b in zip(NS, B0s)]}  (invariant?)")
print(f"  B0/N: " + " ".join(f"n{n}={b/N:.2f}" for n, b in zip(NS, B0s)))
print("\n" + "=" * 74)
print("acc(B, 2n+1)   [HELD-OUT]")
print("=" * 74)
print(f"{'n':>2} " + " ".join(f"B{b:<4}" for b in BS))
for n in NS:
    Bs, va = mean_curve(ava, n)
    row = dict(zip(Bs, va))
    print(f"{n:>2} " + " ".join(f"{row[b]:5.3f}" if b in row else "  -  " for b in BS))

# figure
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
cols = ["#f4a582", "#d62728", "#7f0000", "#9467bd", "#6a0dad", "#4b0082"]
TICKS = [8, 32, 128, 512]
for ax, split, D in zip(axes[:2], ("train", "held-out"), (atr, ava)):
    for n, c in zip(NS, cols):
        Bs, y = mean_curve(D, n)
        ax.plot(Bs, y, "o-", color=c, lw=1.9, ms=4, label=f"$n={n}$ (budget {2*n+1})")
    ax.set_xscale("log", base=2); ax.set_xticks(TICKS); ax.set_xticklabels(TICKS)
    ax.axhline(0.9, ls=":", color="0.5", lw=1)
    ax.set_xlabel("B (training questions)"); ax.set_title(f"{split}")
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("accuracy after $n$ rounds"); axes[1].legend(fontsize=8)

# panel 3: capacity knee B0 vs query budget 2n+1 (log-log) with fitted exponent + error bars
ax = axes[2]
ax.errorbar(q, B0a, yerr=B0err, fmt="o", color="#4b0082", ms=7, capsize=3)
qq = np.linspace(q.min(), q.max(), 50)
ax.plot(qq, np.exp(b_q) * qq ** a_q, "-", color="#4b0082", lw=1.6,
        label=f"$B_0\\propto(2n+1)^{{{a_q:.2f}}}$")
ax.plot(qq, B0a[0] * (qq / q[0]) ** 2, "--", color="0.5", lw=1.2, label="slope 2 (quadratic)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(q); ax.set_xticklabels([f"{int(v)}" for v in q])
ax.set_xlabel("query budget $2n+1$"); ax.set_ylabel("capacity knee $B_0$ (train)")
ax.set_title("knee scaling"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

fig.suptitle("Frustration scaling: deeper intended budget $n$ pushes the capacity knee to larger B")
fig.tight_layout()
f = os.path.join(HERE, "_sweep_out", "S12_frustration.png")
fig.savefig(f, dpi=150, bbox_inches="tight")
print(f"\nfig -> {f}")
