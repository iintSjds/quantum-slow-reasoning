"""
Test-time amplification: best@n (classical) vs amplitude-amplification (quantum)
on TRAINED CoNet / QuCoNet (AR) models, computed from exact per-QA success
probability p.  No retraining, no circuit simulation.

For a QA pair the trained policy solves with probability p:
  classical best@n : acc = 1 - (1-p)^n            (rescue at n ~ 1/p)
  quantum AA       : acc = sin^2((2n+1) arcsin sqrt(p))  (rescue at n ~ 1/sqrt(p))

p is computed by EXACT path enumeration, reusing
conet/analyze_path_diversity.py (compute_{classical,quantum}_path_diversity).

Outputs:
  F1  p-histogram, CoNet vs QuCoNet, at a given B   (the make-or-break)
  F2  aggregate acc(n), four curves {model}x{method}
  F3  queries-to-threshold vs 1/p, pooled (slopes 1.0 vs 0.5)

Run from repo root:
  python scripts/amplification_scaling.py \
      --Bs 1,6,16,64,128 --seeds 8
"""
import os, sys, glob, re, json, argparse
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# ── reuse the exact-p enumerators from conet/analyze_path_diversity.py ──
def _import_enumerators(root=None):
    sys.path.insert(0, os.path.join(REPO, "conet"))
    import analyze_path_diversity as apd
    return apd

# ───────────────────────── analytic inference curves ──────────────────
def cl_acc(p, n):
    """Classical best@n: prob at least one of n rollouts hits the target."""
    return 1.0 - (1.0 - p) ** n

def aa_acc(p, n):
    """Amplitude amplification with budget n, known p, optimal stop (envelope)."""
    p = float(p)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    theta = np.arcsin(np.sqrt(p))
    kstar = max(0, int(round(np.pi / (4 * theta) - 0.5)))   # iters to first peak
    k = min(n, kstar)
    return float(np.sin((2 * k + 1) * theta) ** 2)

def n_to_threshold_classical(p, tau=0.9):
    if p <= 0.0:
        return np.inf
    if p >= 1.0:
        return 1.0
    return float(np.ceil(np.log(1.0 - tau) / np.log(1.0 - p)))

def n_to_threshold_quantum(p, tau=0.9):
    """Calls (2j+1) needed to cross ``tau`` with a phase-matched final step."""
    if p <= 0.0:
        return np.inf
    if p >= 1.0:
        return 1.0
    theta = np.arcsin(np.sqrt(p))
    kth = (np.arcsin(np.sqrt(tau)) / theta - 1.0) / 2.0      # first crossing
    iterations = np.ceil(max(0.0, kth))
    return float(2.0 * iterations + 1.0)

# ───────────────────────── checkpoint indexing ────────────────────────
SEED_RE = re.compile(r"seed(\d+)")
BQA_RE  = re.compile(r"_B(\d+)_Q(\d+)_A(\d+)")

def index_checkpoints(paths):
    """Map (seed, B) -> checkpoint path."""
    idx = {}
    for p in paths:
        s = SEED_RE.search(p)
        bqa = BQA_RE.search(os.path.basename(p))
        if not (s and bqa):
            continue
        seed, B = int(s.group(1)), int(bqa.group(1))
        idx[(seed, B)] = p          # one best_model per (seed,B)
    return idx

def load_qa_pairs(root, seed):
    f = os.path.join(root, "archive/expr4/graph_qa",
                     f"sliding_puzzle_N120_K3_M8_B192_D6_seed{seed}.pt")
    d = torch.load(f, map_location="cpu", weights_only=False)
    return [(int(q), int(a)) for (q, a) in d["qa_pairs"]]

# ───────────────────────── per-QA p extraction ────────────────────────
def p_classical(apd, ckpt, qa_pairs, M):
    ei, tp, nptr, N, K = apd.load_classical_checkpoint(ckpt)
    out = []
    for (Q, A) in qa_pairs:
        _, _, _, ts = apd.compute_classical_path_diversity(ei, tp, nptr, Q, A, M)
        out.append(float(ts))
    return out

def p_quantum(apd, ckpt, qa_pairs, M):
    cm, snm, scm, N, K, cfg = apd.load_quantum_checkpoint(ckpt)
    out = []
    for (Q, A) in qa_pairs:
        _, _, _, ts = apd.compute_quantum_path_diversity(cm, snm, scm, Q, A, N, K, M)
        out.append(float(ts))
    return out

# ───────────────────────── collection ─────────────────────────────────
def collect(root, apd, Bs, seeds, M=8):
    """Return rows: dict(model, seed, B, p) with p a list over qa_pairs[:B]."""
    qroot = os.path.join(root, "archive/expr4/quantum")
    croot = os.path.join(root, "archive/expr4/classical")
    qidx = index_checkpoints(glob.glob(os.path.join(qroot, "**/*best_model*.pt"), recursive=True))
    cidx = index_checkpoints(glob.glob(os.path.join(croot, "**/*best_model*.pt"), recursive=True))
    avail_seeds = sorted({s for (s, B) in qidx} & {s for (s, B) in cidx})
    use_seeds = avail_seeds[:seeds] if isinstance(seeds, int) else seeds
    print(f"available paired seeds: {len(avail_seeds)}; using {len(use_seeds)}: {use_seeds[:12]}")

    rows = []
    for seed in use_seeds:
        qa_all = load_qa_pairs(root, seed)
        for B in Bs:
            if (seed, B) not in qidx or (seed, B) not in cidx:
                print(f"  skip seed{seed} B{B} (missing ckpt)"); continue
            qa = qa_all[:B]
            pc = p_classical(apd, cidx[(seed, B)], qa, M)
            pq = p_quantum(apd, qidx[(seed, B)], qa, M)
            rows.append(dict(model="classical", seed=seed, B=B, p=pc))
            rows.append(dict(model="quantum",   seed=seed, B=B, p=pq))
            print(f"  seed{seed:>4} B{B:>4}: "
                  f"cl mean_p={np.mean(pc):.3f} interior={frac_interior(pc):.2f} | "
                  f"qu mean_p={np.mean(pq):.3f} interior={frac_interior(pq):.2f}")
    return rows

def frac_interior(ps, lo=0.02, hi=0.98):
    ps = np.asarray(ps)
    return float(np.mean((ps > lo) & (ps < hi))) if len(ps) else 0.0

# ───────────────────────── summary + figures ──────────────────────────
def pool_p(rows, model, B):
    ps = []
    for r in rows:
        if r["model"] == model and r["B"] == B:
            ps.extend(r["p"])
    return np.asarray(ps)

def summary(rows, Bs):
    print("\n" + "=" * 74)
    print(f"{'B':>5} {'model':>10} {'n':>6} {'mean_p':>8} {'dead<.02':>9} "
          f"{'interior':>9} {'solved>.98':>11}")
    print("-" * 74)
    for B in Bs:
        for model in ("classical", "quantum"):
            ps = pool_p(rows, model, B)
            if not len(ps):
                continue
            dead = np.mean(ps <= 0.02); inter = frac_interior(ps); solved = np.mean(ps >= 0.98)
            print(f"{B:>5} {model:>10} {len(ps):>6} {ps.mean():>8.3f} "
                  f"{dead:>9.2f} {inter:>9.2f} {solved:>11.2f}")

def fig_hist(rows, B, out):
    pc, pq = pool_p(rows, "classical", B), pool_p(rows, "quantum", B)
    if not len(pc) or not len(pq):
        print(f"[F1] no data at B={B}"); return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 26)
    ax.hist(pc, bins=bins, alpha=0.55, label=f"CoNet (interior={frac_interior(pc):.2f})", color="#1f77b4")
    ax.hist(pq, bins=bins, alpha=0.55, label=f"QuCoNet (interior={frac_interior(pq):.2f})", color="#d62728")
    ax.axvspan(0.02, 0.98, color="gray", alpha=0.08)
    ax.set_xlabel("per-QA success probability  p"); ax.set_ylabel("count")
    ax.set_title(f"F1: per-QA p distribution, B={B}  (interior 0.02<p<0.98 = amplifiable)")
    ax.legend(); fig.tight_layout()
    f = os.path.join(out, f"F1_p_hist_B{B}.png"); fig.savefig(f, dpi=150); plt.close(fig)
    print(f"[F1] saved {f}")

def fig_acc_vs_n(rows, Bs, out, nmax=128):
    ns = np.unique(np.round(np.geomspace(1, nmax, 40)).astype(int))
    for B in Bs:
        pc, pq = pool_p(rows, "classical", B), pool_p(rows, "quantum", B)
        if not len(pc) or not len(pq):
            continue
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(ns, [np.mean([cl_acc(p, n) for p in pc]) for n in ns], "o-", ms=3, label="CoNet  best@n", color="#1f77b4")
        ax.plot(ns, [np.mean([aa_acc(p, n) for p in pc]) for n in ns], "s--", ms=3, label="CoNet  AA",     color="#7fb3e0")
        ax.plot(ns, [np.mean([cl_acc(p, n) for p in pq]) for n in ns], "o-", ms=3, label="QuCoNet best@n", color="#d62728")
        ax.plot(ns, [np.mean([aa_acc(p, n) for p in pq]) for n in ns], "s--", ms=3, label="QuCoNet AA",    color="#f0a0a0")
        ax.set_xscale("log"); ax.set_xlabel("test-time budget  n"); ax.set_ylabel("batch accuracy")
        ax.set_title(f"F2: accuracy vs test-time compute, B={B}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        f = os.path.join(out, f"F2_acc_vs_n_B{B}.png"); fig.savefig(f, dpi=150); plt.close(fig)
        print(f"[F2] saved {f}")

def fig_queries_scatter(rows, out, tau=0.9):
    ps = []
    for r in rows:
        ps.extend(r["p"])
    ps = np.asarray(ps)
    m = (ps > 0.02) & (ps < 0.999)            # amplifiable, finite queries
    ps = ps[m]
    if not len(ps):
        print("[F3] no interior p"); return
    x = 1.0 / ps
    nc = np.array([n_to_threshold_classical(p, tau) for p in ps])
    nq = np.array([n_to_threshold_quantum(p, tau) for p in ps])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(x, nc, s=8, alpha=0.35, label=r"classical sampling, $O(1/p)$", color="#1f77b4")
    ax.scatter(x, nq, s=8, alpha=0.35, label=r"phase-matched amplification, $O(1/\sqrt{p})$", color="#d62728")
    # log-log slope fits
    sc = np.polyfit(np.log(x), np.log(nc), 1)[0]
    qm = nq > 0
    sq = np.polyfit(np.log(x[qm]), np.log(nq[qm]), 1)[0]
    xr = np.array([x.min(), x.max()])
    cc = np.polyfit(np.log(x), np.log(nc), 1)
    cq = np.polyfit(np.log(x[qm]), np.log(nq[qm]), 1)
    ax.plot(xr, np.exp(np.polyval(cc, np.log(xr))), "-", color="#1f77b4", lw=1.5)
    ax.plot(xr, np.exp(np.polyval(cq, np.log(xr))), "-", color="#d62728", lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"inverse base probability $1/p$")
    ax.set_ylabel(rf"reasoning-circuit applications to reach success ${tau}$")
    ax.legend(); ax.grid(alpha=0.3, which="both"); fig.tight_layout()
    f = os.path.join(out, "F3_queries_vs_invp.pdf"); fig.savefig(f); plt.close(fig)
    print(f"[F3] saved {f}  (slopes best@n={sc:.2f}, AA={sq:.2f})")

# ───────────────────────── main ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("QSR_ROOT", os.path.abspath(os.path.join(HERE, ".."))))
    ap.add_argument("--Bs", default="1,6,16,64,128")
    ap.add_argument("--seeds", default="8", help="int N (first N paired seeds) or comma list")
    ap.add_argument("--out", default=os.path.join(HERE, "_amp_out"))
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--figs", default="F1", help="comma subset of F1,F2,F3 or 'all'")
    ap.add_argument("--cache", action="store_true", help="reuse per_qa_p.json if present")
    args = ap.parse_args()

    Bs = [int(b) for b in args.Bs.split(",")]
    seeds = int(args.seeds) if args.seeds.isdigit() else [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "per_qa_p.json")

    if args.cache and os.path.exists(cache):
        rows = json.load(open(cache))
        print(f"loaded cache: {len(rows)} rows from {cache}")
    else:
        apd = _import_enumerators(args.root)
        rows = collect(args.root, apd, Bs, seeds, M=args.M)
        json.dump([{**r} for r in rows], open(cache, "w"))
    summary(rows, Bs)

    figs = ["F1", "F2", "F3"] if args.figs == "all" else args.figs.split(",")
    if "F1" in figs:
        fig_hist(rows, max(Bs), args.out)
    if "F2" in figs:
        fig_acc_vs_n(rows, [b for b in (6, max(Bs)) if b in Bs], args.out)
    if "F3" in figs:
        fig_queries_scatter(rows, args.out)

if __name__ == "__main__":
    main()
