"""Control audit, untrained baseline, cross-budget matrix, blind knees.

  A. Controls audit at B=32: best-of-(2n+1) accuracy of EVERY trained
     classical / semi-quantum family, train and held-out, to establish the
     true strongest classical control per budget (is 0.657 really it?).
  B. Untrained (uniform-coin) walker on the same pools: raw p, best-of-k,
     blind and at-most-n Grover accuracies -- the missing baseline.
  C. Cross-budget matrix: Grover-n-trained walkers evaluated at every
     budget n' (held-out, at-most-n' and blind) -- how much of the ladder
     is deeper inference on fixed mass vs deeper training.
  D. Blind-schedule capacity knees from the large-B cache.
  E. OLS slope error + n=1..5 subfit for the capacity exponent.

All parts are cache reads except B's uniform DP (needs only the neighbor
table from any softmax checkpoint).

Run:  python scripts/control_audit_analysis.py
"""
import os, sys, json, math, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "_sweep_out")
DATAROOT = os.environ.get("QSR_ROOT", os.path.abspath(os.path.join(HERE, "..")))
CACHE = json.load(open(os.path.join(OUT, "p_cache.json")))


def aa(p, n):
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    th = math.asin(math.sqrt(p))
    k = min(n, max(0, round(math.pi / (4 * th) - 0.5)))
    return math.sin((2 * k + 1) * th) ** 2


def blind(p, n):
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    return math.sin((2 * n + 1) * math.asin(math.sqrt(p))) ** 2


def pooled(fam, B, split):
    ps = []
    for s in range(1, 33):
        ps += CACHE.get(f"{fam}|{s}|{B}|{split}", [])
    return np.asarray(ps)


# ── A: controls audit at B=32 ───────────────────────────────────────────
def part_A():
    fams = ["cadam", "cstd", "cstd_adam", "cstd_exact", "cbestk", "cbestk_exact",
            "ccap25", "ccap50", "ccap75",
            "centH01", "centH03", "centH10", "cbentH01", "cbentH03", "cbentH10",
            "cbestkX4", "cbestkX8", "cbestkX16", "cbestkX32", "cbestkX64",
            "qbkX2", "qbkX4", "qbkX8", "qstd"]
    print("=" * 100)
    print("A. best-of-(2n+1) of every control at B=32   [train | valid], n=1..4")
    print("=" * 100)
    best = {("tr", n): ("", 0) for n in (1, 2, 3, 4)}
    for n in (1, 2, 3, 4):
        best[("va", n)] = ("", 0)
    for fam in fams:
        row = []
        for split, tag in (("train", "tr"), ("valid", "va")):
            ps = pooled(fam, 32, split)
            if not len(ps):
                row.append("   -   " * 4)
                continue
            cells = []
            for n in (1, 2, 3, 4):
                v = float(np.mean(1 - (1 - ps) ** (2 * n + 1)))
                cells.append(f"{v:.3f}")
                if v > best[(tag, n)][1]:
                    best[(tag, n)] = (fam, v)
            row.append(" ".join(cells))
        print(f"{fam:>14}: tr {row[0]}   va {row[1]}")
    for tag, name in (("tr", "train"), ("va", "valid")):
        print(f"  STRONGEST {name}: " +
              "  ".join(f"bo{2*n+1}={best[(tag,n)][0]}({best[(tag,n)][1]:.3f})"
                        for n in (1, 2, 3, 4)))


# ── B: untrained uniform walker ─────────────────────────────────────────
def uniform_p(nbr, Q, A, M=8):
    """First-arrival probability of the uniform (1/K) walker via absorbing DP."""
    N, K = nbr.shape
    occ = np.zeros(N); occ[Q] = 1.0
    hit = 0.0
    if Q == A:
        return 1.0
    for _ in range(M):
        new = np.zeros(N)
        for k in range(K):
            np.add.at(new, nbr[:, k], occ / K)
        hit += new[A]
        new[A] = 0.0
        occ = new
    return float(hit)


def part_B():
    uni = os.path.join(OUT, "p_uniform.json")
    if os.path.exists(uni):
        d = json.load(open(uni))
        tr_ps = sum((d[k] for k in d if k.endswith("|32|train")), [])
        va_ps = sum((d[k] for k in d if k.endswith("|32|valid")), [])
    else:
        import torch
        import amplification_step1 as s1
        import grover_sweep_analysis as gsa
        idx = s1.run_index(DATAROOT, gsa.FAMILIES["cadam"][0])
        sd = torch.load(idx[(1, 32)]["best"], map_location="cpu",
                        weights_only=False)["model_state_dict"]
        nbr = sd["node_neighbors"].numpy().astype(int)
        gidx = s1.run_index(DATAROOT, gsa.FAMILIES["grover"][0])
        tr_ps, va_ps = [], []
        for seed in range(1, 33):
            if (seed, 32) not in gidx or not gidx[(seed, 32)]["results"]:
                continue
            tr, va, _ = s1.split_qa(gidx[(seed, 32)]["results"])
            tr_ps += [uniform_p(nbr, Q, A) for Q, A in tr]
            va_ps += [uniform_p(nbr, Q, A) for Q, A in va]
    print("\n" + "=" * 100)
    print(f"B. UNTRAINED uniform walker  ({len(tr_ps)} train / {len(va_ps)} valid "
          f"questions pooled over seeds)")
    print("=" * 100)
    for name, ps in (("train", np.asarray(tr_ps)), ("valid", np.asarray(va_ps))):
        print(f"  {name}: mean p = {ps.mean():.4f}  (10-90%: "
              f"{np.percentile(ps,10):.4f}-{np.percentile(ps,90):.4f})")
        bo = [float(np.mean(1 - (1 - ps) ** (2 * n + 1))) for n in (1, 2, 3, 4)]
        bl = [float(np.mean([blind(p, n) for p in ps])) for n in (1, 2, 3, 4)]
        am = [float(np.mean([aa(p, n) for p in ps])) for n in (1, 2, 3, 4)]
        print(f"    classical bo(3,5,7,9): " + " ".join(f"{v:.3f}" for v in bo))
        print(f"    blind Grover-1..4:     " + " ".join(f"{v:.3f}" for v in bl))
        print(f"    at-most-n Grover-1..4: " + " ".join(f"{v:.3f}" for v in am))


# ── C: cross-budget matrix (held-out) ───────────────────────────────────
def part_C():
    print("\n" + "=" * 100)
    print("C. cross matrix: rows = training objective, cols = evaluation budget n'"
          " (held-out, B=32)")
    print("=" * 100)
    for rule, fn in (("at-most-n'", aa), ("blind n'", blind)):
        print(f"  [{rule}]")
        for fam, n in (("grover", 1), ("grover2", 2), ("grover3", 3), ("grover4", 4)):
            ps = pooled(fam, 32, "valid")
            row = [float(np.mean([fn(p, m) for p in ps])) for m in (1, 2, 3, 4)]
            print(f"    trained n={n}: " + " ".join(f"{v:.3f}" for v in row))


# ── D: blind capacity knees ─────────────────────────────────────────────
BS = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 704, 768, 896, 1024, 1216]


def knee(bs, acc, thr=0.9):
    ms = np.log2(bs)
    for i in range(len(ms) - 1):
        if acc[i] >= thr > acc[i + 1]:
            t = (acc[i] - thr) / (acc[i] - acc[i + 1])
            return 2 ** (ms[i] + t * (ms[i + 1] - ms[i]))
    return 2 ** ms[-1] if acc[-1] >= thr else float("nan")


def part_D():
    big = json.load(open(os.path.join(OUT, "p_cache_bigB.json")))
    pool = {}
    for k, d in big.items():
        n, s, B = map(int, k.split("|"))
        pool.setdefault((n, B), []).extend(d["tr"])
    print("\n" + "=" * 100)
    print("D. capacity knees: at-most-n vs blind schedule (thr 0.9)")
    print("=" * 100)
    for n in (1, 2, 3, 4, 5, 6):
        bs = [B for B in BS if (n, B) in pool]
        am = [float(np.mean([aa(p, n) for p in pool[(n, B)]])) for B in bs]
        bl = [float(np.mean([blind(p, n) for p in pool[(n, B)]])) for B in bs]
        print(f"  n={n}: knee(at-most-n) = {knee(bs, am):6.0f}   "
              f"knee(blind) = {knee(bs, bl):6.0f}   "
              f"(blind acc at B={bs[0]}: {bl[0]:.3f})")


# ── E: capacity exponent error decomposition ────────────────────────────
def part_E():
    B0 = np.array([40, 77, 137, 229, 385, 990], float)
    q = np.array([3, 5, 7, 9, 11, 13], float)
    x, y = np.log(q), np.log(B0)
    a, b = np.polyfit(x, y, 1)
    res = y - (a * x + b)
    se = math.sqrt(np.sum(res ** 2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2))
    a5, _ = np.polyfit(x[:5], y[:5], 1)
    print("\n" + "=" * 100)
    print("E. capacity exponent error decomposition")
    print("=" * 100)
    print(f"  full fit: {a:.2f}  OLS slope SE: {se:.2f}")
    print(f"  n=1..5 subfit: {a5:.2f}")
    print(f"  residuals (ln): " + " ".join(f"{r:+.2f}" for r in res))


# ── F: hybrid control -- classical training + blind Grover read-out ─────
def part_F():
    print("\n" + "=" * 78)
    print("F. hybrid: classically trained policy + blind Grover read-out (B=32)")
    print("=" * 78)
    for fam in ("ccap25", "ccap50", "cbestkX32", "cbestkX64"):
        tr, va = pooled(fam, 32, "train"), pooled(fam, 32, "valid")
        if not len(tr):
            continue
        bt = [float(np.mean([blind(p, n) for p in tr])) for n in (1, 2, 3, 4)]
        bv = [float(np.mean([blind(p, n) for p in va])) for n in (1, 2, 3, 4)]
        print(f"  {fam}: mean train p {tr.mean():.3f}, frac>0.3 {float(np.mean(tr > 0.3)):.2f}")
        print(f"    blind train 1..4: " + " ".join(f"{v:.3f}" for v in bt))
        print(f"    blind valid 1..4: " + " ".join(f"{v:.3f}" for v in bv))


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
    part_D()
    part_E()
    part_F()
