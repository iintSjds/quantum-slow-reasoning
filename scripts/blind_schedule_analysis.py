"""Blind-schedule evaluation, capacity-knee error analysis, and deep-n IPR.

All parts except D are pure cache reads (no model loading):
  A: blind fixed-n Grover schedule vs the at-most-n stopping rule, from the
     stored per-question p's of the deep-n runs (p_cache.json).  Also
     per-seed s.d. of training accuracy.
  B: capacity-knee error analysis from p_cache_bigB.json -- per-seed knees,
     question-bootstrap CIs on B0 and on the exponent, threshold
     sensitivity, and the central-fit R^2.
  C: the classical queries-to-threshold exponent is the exact best-of-k
     law's finite-window elasticity, not an anomaly: print the local
     elasticity d ln k / d ln(1/p) and refit the exact law over the same
     interior window and over its small-p tail.
  D: deep-n IPR -- extend ipr_cache.json with grover3/grover4 at B=32
     (checkpoints from the data archive) and print the accuracy-weighted
     <IPR>.

Run:  python scripts/blind_schedule_analysis.py   (--no-ipr for the
      cache-only parts A-C; part D needs the checkpoint archive)
"""
import os, sys, json, math, argparse
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "_sweep_out")
DATAROOT = os.environ.get("QSR_ROOT", os.path.abspath(os.path.join(HERE, "..")))

DEEP = {"grover": 1, "grover2": 2, "grover3": 3, "grover4": 4}


def aa_acc(p, n):
    """At-most-n rule: k = min(n, k*(p)) -- the objective's stopping convention."""
    if p <= 0.0: return 0.0
    if p >= 1.0: return 1.0
    th = math.asin(math.sqrt(p))
    k = min(n, max(0, round(math.pi / (4 * th) - 0.5)))
    return math.sin((2 * k + 1) * th) ** 2


def blind_acc(p, n):
    """Blind deepest schedule: always exactly n rounds, no knowledge of p."""
    if p <= 0.0: return 0.0
    if p >= 1.0: return 1.0
    return math.sin((2 * n + 1) * math.asin(math.sqrt(p))) ** 2


def pistar(n):
    return math.sin(math.pi / (2 * (2 * n + 1))) ** 2


# ───────────────────── A: blind vs oracle (M2) + seed spread (M17) ─────────
def part_A():
    cache = json.load(open(os.path.join(OUT, "p_cache.json")))
    print("=" * 78)
    print("A. blind fixed-n schedule vs at-most-n rule   (p_cache.json, B=32)")
    print("=" * 78)
    print(f"{'n':>2} {'seeds':>5} | {'train amn':>9} {'train blind':>11} "
          f"{'+/-sd(seed)':>11} | {'valid amn':>9} {'valid blind':>11}")
    for fam, n in DEEP.items():
        tr_all, va_all, tr_seed = [], [], []
        for k, ps in cache.items():
            f, seed, B, split = k.split("|")
            if f != fam or B != "32":
                continue
            if split == "train":
                tr_all += ps
                tr_seed.append(np.mean([aa_acc(p, n) for p in ps]))
            else:
                va_all += ps
        tr_amn = np.mean([aa_acc(p, n) for p in tr_all])
        tr_bl = np.mean([blind_acc(p, n) for p in tr_all])
        va_amn = np.mean([aa_acc(p, n) for p in va_all])
        va_bl = np.mean([blind_acc(p, n) for p in va_all])
        print(f"{n:>2} {len(tr_seed):>5} | {tr_amn:>9.3f} {tr_bl:>11.3f} "
              f"{np.std(tr_seed):>11.3f} | {va_amn:>9.3f} {va_bl:>11.3f}")

    # across the full B grid: worst-case |at-most-n - blind| on held-out
    print("\n   held-out |at-most-n - blind| across the full B grid:")
    for fam, n in DEEP.items():
        diffs = {}
        for k, ps in cache.items():
            f, seed, B, split = k.split("|")
            if f != fam or split != "valid":
                continue
            diffs.setdefault(int(B), []).extend(ps)
        row = {B: abs(np.mean([aa_acc(p, n) for p in ps]) -
                      np.mean([blind_acc(p, n) for p in ps]))
               for B, ps in sorted(diffs.items())}
        print(f"   n={n}: " + "  ".join(f"B{B}:{d:.4f}" for B, d in row.items()) +
              f"   (max {max(row.values()):.4f})")

    # sanity: the strongest classical control column of tab:nscan
    print("\n   check ccap25 best-of-(2n+1) held-out at B=32 "
          "(tab:nscan classical column):")
    va = []
    for k, ps in cache.items():
        f, seed, B, split = k.split("|")
        if f == "ccap25" and B == "32" and split == "valid":
            va += ps
    for n in (1, 2, 3, 4):
        bo = np.mean([1 - (1 - p) ** (2 * n + 1) for p in va])
        print(f"   n={n}: bo{2*n+1} = {bo:.3f}")


# ───────────────────── B: capacity-knee errors (M4) ────────────────────────
BS_OLD = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 704]
BS_NEW = [768, 896, 1024, 1216]
BS = BS_OLD + BS_NEW
NS = [1, 2, 3, 4, 5, 6]


def knee(bs, acc, thr=0.9):
    ms = np.log2(bs)
    for i in range(len(ms) - 1):
        if acc[i] >= thr > acc[i + 1]:
            t = (acc[i] - thr) / (acc[i] - acc[i + 1])
            return ms[i] + t * (ms[i + 1] - ms[i])
    return ms[-1] if acc[-1] >= thr else ms[0]


def knees_from(pool, thr, rng=None):
    """pool: {(n,B): [list of A_n]}; optional bootstrap resample."""
    B0s = []
    for n in NS:
        bs, ys = [], []
        for B in BS:
            a = pool.get((n, B))
            if not a:
                continue
            a = np.asarray(a)
            if rng is not None:
                a = a[rng.integers(0, len(a), len(a))]
            bs.append(B)
            ys.append(float(a.mean()))
        B0s.append(2 ** knee(bs, ys, thr))
    return np.array(B0s)


def expfit(B0s):
    q = np.array([2 * n + 1 for n in NS], float)
    a, b = np.polyfit(np.log(q), np.log(B0s), 1)
    pred = a * np.log(q) + b
    ss = np.sum((np.log(B0s) - pred) ** 2)
    r2 = 1 - ss / np.sum((np.log(B0s) - np.log(B0s).mean()) ** 2)
    return a, r2


def part_B(nboot=1000):
    cache = json.load(open(os.path.join(OUT, "p_cache_bigB.json")))
    pool = {}
    seedpool = {}
    for k, d in cache.items():
        n, seed, B = k.split("|")
        n, B = int(n), int(B)
        accs = [aa_acc(p, n) for p in d["tr"]]
        pool.setdefault((n, B), []).extend(accs)
        seedpool.setdefault((n, int(seed)), {})[B] = float(np.mean(accs))
    print("\n" + "=" * 78)
    print("B. capacity-knee error analysis   (p_cache_bigB.json)")
    print("=" * 78)
    for thr in (0.85, 0.90, 0.95):
        B0s = knees_from(pool, thr)
        a, r2 = expfit(B0s)
        print(f"   thr={thr:.2f}: B0 = {[round(b) for b in B0s]}   "
              f"exponent = {a:.2f}  (R^2 = {r2:.3f})")
    # per-seed knees at 0.9
    print("   per-seed knees (thr=0.90):")
    for n in NS:
        ks = []
        for s in (1, 2, 3, 4):
            cur = seedpool.get((n, s), {})
            bs = sorted(cur)
            if len(bs) >= 2:
                ks.append(2 ** knee(bs, [cur[b] for b in bs]))
        print(f"   n={n}: seeds {[round(x) for x in ks]}  "
              f"(mean {np.mean(ks):.0f}, sd {np.std(ks):.0f})")
    # question bootstrap
    rng = np.random.default_rng(0)
    B0bs, exps = [], []
    for _ in range(nboot):
        B0s = knees_from(pool, 0.9, rng)
        B0bs.append(B0s)
        exps.append(expfit(B0s)[0])
    B0bs = np.array(B0bs)
    lo, hi = np.percentile(B0bs, [16, 84], axis=0)
    print(f"   bootstrap ({nboot} reps over questions), thr=0.90:")
    for i, n in enumerate(NS):
        print(f"   n={n}: B0 = {knees_from(pool, 0.9)[i]:.0f}  "
              f"[{lo[i]:.0f}, {hi[i]:.0f}]")
    print(f"   exponent = {np.mean(exps):.2f} +/- {np.std(exps):.2f}  "
          f"(16-84%: [{np.percentile(exps, 16):.2f}, {np.percentile(exps, 84):.2f}])")
    print(f"   p*(n)*B0: {[round(pistar(n) * b, 1) for n, b in zip(NS, knees_from(pool, 0.9))]}")


# ───────────────────── C: query-exponent elasticity (M8) ───────────────────
def part_C():
    print("\n" + "=" * 78)
    print("C. classical queries-to-threshold exponent (M8)")
    print("=" * 78)
    print("   local elasticity d ln k / d ln(1/p) of k = ln(1-tau)/ln(1-p):")
    for p in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
        e = p / ((1 - p) * (-math.log(1 - p)))
        print(f"   p={p:>4}: {e:.2f}")
    cache = json.load(open(os.path.join(OUT, "p_cache.json")))
    ps = []
    for k, v in cache.items():
        f = k.split("|")[0]
        if f in ("grover", "grover2", "qstd", "cstd"):
            ps += v
    ps = np.asarray([p for p in ps if 0.02 < p < 0.999])
    x = np.log(1 / ps)
    for label, kf in (("exact ceil", lambda p: math.ceil(math.log(0.1) / math.log(1 - p))),
                      ("continuous", lambda p: math.log(0.1) / math.log(1 - p))):
        y = np.log([kf(p) for p in ps])
        s_all = np.polyfit(x, y, 1)[0]
        m = ps < 0.1
        s_tail = np.polyfit(x[m], y[m], 1)[0] if m.sum() > 10 else np.nan
        print(f"   fit over pooled window (n={len(ps)}): {label}: "
              f"slope = {s_all:.2f};  small-p tail (p<0.1): {s_tail:.2f}")


# ───────────────────── D: deep-n IPR at B=32 (M19) ─────────────────────────
def part_D():
    import amplification_scaling as amp
    import amplification_step1 as s1
    import grover_sweep_analysis as gsa
    import grover_ipr_analysis as gia
    apd = amp._import_enumerators(".")
    cache_path = os.path.join(OUT, "ipr_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    n_new = 0
    for fam in ("grover3", "grover4"):
        pat, mtype = gsa.FAMILIES[fam]
        idx = s1.run_index(DATAROOT, pat)
        for (seed, B), d in sorted(idx.items()):
            if B != 32 or not d["results"]:
                continue
            tr, va, _ = s1.split_qa(d["results"])
            for split, qa in (("train", tr), ("valid", va)):
                key = f"{fam}|{seed}|{B}|{split}"
                if key in cache:
                    continue
                cache[key] = gia.pipr_list(apd, mtype, d["best"], qa, 8)
                n_new += 1
                if n_new % 8 == 0:
                    json.dump(cache, open(cache_path, "w"))
                    print(f"   ... cached {n_new} (at {key})")
    json.dump(cache, open(cache_path, "w"))
    print("\n" + "=" * 78)
    print(f"D. accuracy-weighted <IPR> at B=32 (ipr_cache.json; {n_new} new)")
    print("=" * 78)
    for fam in ("grover", "grover2", "grover3", "grover4"):
        for split in ("train", "valid"):
            ent = []
            for k, v in cache.items():
                f, seed, B, sp = k.split("|")
                if f == fam and B == "32" and sp == split:
                    ent += v
            if not ent:
                continue
            p = np.array([e[0] for e in ent])
            ipr = np.array([e[1] for e in ent])
            w = float((p * ipr).sum() / p.sum()) if p.sum() > 0 else float("nan")
            print(f"   {fam:>8} {split:>5}: <IPR> = {w:6.2f}   "
                  f"({len(ent)} questions, {len(ent)//(32 if split=='train' else 64)} seeds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ipr", action="store_true")
    args = ap.parse_args()
    part_A()
    part_B()
    part_C()
    if not args.no_ipr:
        part_D()
