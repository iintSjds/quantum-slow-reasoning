#!/usr/bin/env python3
"""Capacity knees under the adaptive G<=q protocol vs blind fixed-depth
G_q, both computed from the SAME per-question caches (p_cache_bigB.json:
legacy 4-pool knee grid, n=1..6, full B range incl. the ext extension)."""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
pc = json.load(open(os.path.join(HERE, '_sweep_out', 'p_cache_bigB.json')))

def G_le(p, q):
    p = np.clip(np.asarray(p, float), 0, 1)
    th = np.arcsin(np.sqrt(p))
    with np.errstate(divide='ignore'):
        rs = np.where(th > 0, np.round(np.pi / (4 * th) - 0.5), 0)
    r = np.minimum(np.maximum(rs, 0), (q - 1) // 2)
    return np.sin((2 * r + 1) * th) ** 2

def G_blind(p, q):
    p = np.clip(np.asarray(p, float), 0, 1)
    return np.sin(q * np.arcsin(np.sqrt(p))) ** 2

# keys are "n|seed|B" -> {"tr": [...], "va": [...]}
from collections import defaultdict
data = defaultdict(list)   # (n,B) -> pooled train p
for k, v in pc.items():
    n, seed, B = k.split('|')
    data[(int(n), int(B))] += v['tr']

def knee(Bs, accs, thr):
    """first downward crossing of thr, interpolated in log B (pipeline rule)."""
    Bs = np.asarray(Bs, float); accs = np.asarray(accs, float)
    order = np.argsort(Bs); Bs, accs = Bs[order], accs[order]
    for i in range(len(Bs) - 1):
        a0, a1 = accs[i], accs[i + 1]
        if a0 >= thr > a1:
            t = (a0 - thr) / (a0 - a1)
            return math.exp(math.log(Bs[i]) + t * (math.log(Bs[i + 1]) - math.log(Bs[i])))
    if accs[-1] >= thr:
        return float('inf')   # censored high
    if accs[0] < thr:
        return float('nan')   # never above
    return float('nan')

for thr in (0.95, 0.90, 0.70):
    print(f"\n=== threshold {thr} ===")
    rows = {}
    for proto, G in (("adaptive G<=q", G_le), ("fixed-depth G_q", G_blind)):
        knees = {}
        for n in range(1, 7):
            q = 2 * n + 1
            pts = sorted((B, float(np.mean(G(np.asarray(ps), q))))
                         for (nn, B), ps in data.items() if nn == n)
            if pts:
                knees[q] = knee([b for b, _ in pts], [a for _, a in pts], thr)
        rows[proto] = knees
        ks = sorted(knees)
        print(f"{proto:>16}: " + " ".join(
            f"q={q}:{('%.0f' % knees[q]) if np.isfinite(knees[q]) else ('cens' if knees[q]==float('inf') else 'none')}"
            for q in ks))
        finite = [(q, v) for q, v in knees.items() if np.isfinite(v) and v > 0]
        if len(finite) >= 3:
            lq = np.log([q for q, _ in finite]); lb = np.log([v for _, v in finite])
            slope = np.polyfit(lq, lb, 1)[0]
            print(f"{'':>16}  exponent (finite points, n={len(finite)}): {slope:.2f}")
