"""
IPR / training-dynamics analysis of the grover-n=1 sweep vs one-shot baselines.

Question: when the grover objective parks a train pair at p*=0.25, is that a
DIVERSE ensemble of paths summing to 0.25 (IPR >> 1, the implicit path
ensemble survives) or a single attenuated path (IPR ~ 1)?  And how do SR/IPR
evolve during training under the two objectives?

Parts:
  D1 (histories, free):   train SR + train IPR vs epoch, grover vs qstd,
                          per-seed thin lines + seed-mean bold, B in {8,32,128}.
  D2 (checkpoints):       exact per-QA (p, IPR) from best models -> accuracy-
                          weighted <IPR> vs B for grover/qstd/cadam(+cbestk
                          when pulled); cached like the p-cache.
  D2b:                    per-QA scatter IPR vs p at B=32 (where does the
                          diversity live?).

Run from repo root:
  python scripts/grover_ipr_analysis.py
  (--parts D1  for the free part only)
"""
import os, sys, json, argparse
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amplification_scaling as amp
import amplification_step1 as s1
import grover_sweep_analysis as gsa          # FAMILIES, B_LIST, p_list_softmax

P_STAR1 = gsa.P_STAR1
HIST_FAMS = {"grover": "#d62728", "qstd": "#ff7f0e"}
IPR_FAMS = {"grover": "#d62728", "qstd": "#ff7f0e", "cadam": "#1f77b4",
            "cbestk": "#08306b", "cstd": "#6baed6",
            "grover2": "#7f0000", "cbestk_exact": "#17becf"}

# ───────────────────────── D1: training dynamics from histories ───────
def load_histories(root, fams=("grover", "qstd")):
    """(fam, seed, B) -> list of eval entries (dicts with epoch/sr/ipr)."""
    out = {}
    for fam in fams:
        pat, _ = gsa.FAMILIES[fam]
        idx = s1.run_index(root, pat)
        for (seed, B), d in idx.items():
            if not d["results"]:
                continue
            h = json.load(open(d["results"])).get("history", [])
            ev = [e for e in h if "train_ipr" in e]
            if ev:
                out[(fam, seed, B)] = ev
    return out

def fig_dynamics(hist, out, Bs=(8, 32, 128), max_seed=8):
    fig, axes = plt.subplots(2, len(Bs), figsize=(3.8 * len(Bs), 6.4), sharex="col")
    for j, B in enumerate(Bs):
        for fam, col in HIST_FAMS.items():
            series = [(s, hist[(fam, s, B)]) for s in range(1, max_seed + 1)
                      if (fam, s, B) in hist]
            for row, key in ((0, "train_sr"), (1, "train_ipr")):
                ax = axes[row, j]
                grid = {}
                for s, ev in series:
                    ep = [e["epoch"] for e in ev]
                    v = [e[key] for e in ev]
                    ax.plot(ep, v, color=col, lw=0.7, alpha=0.30)
                    for e_, v_ in zip(ep, v):
                        grid.setdefault(e_, []).append(v_)
                if grid:
                    eps = sorted(e for e in grid if len(grid[e]) >= 2)
                    ax.plot(eps, [np.mean(grid[e]) for e in eps], color=col,
                            lw=2.2, label=f"{fam} (n={len(series)} seeds)")
        axes[0, j].axhline(P_STAR1, ls=":", color="gray", lw=1)
        axes[0, j].set_title(f"B={B}")
        axes[1, j].set_xlabel("epoch")
    axes[0, 0].set_ylabel("train SR (one-shot)")
    axes[1, 0].set_ylabel("train ⟨IPR⟩ (acc-weighted)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Training dynamics: grover-1 objective (red) vs one-shot (orange). "
                 "Dotted = attractor p*=0.25; one-shot SR keeps climbing while IPR collapses to 1")
    fig.tight_layout()
    f = os.path.join(out, "D1_dynamics.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}")

# ───────────────────────── D2: exact (p, IPR) from checkpoints ────────
def pipr_list(apd, mtype, ckpt, qa, M):
    """[(p, ipr)] per QA.  IPR of the success-path distribution (1 if <=1 path)."""
    if mtype == "quantum":
        cm, snm, scm, N, K, cfg = apd.load_quantum_checkpoint(ckpt)
        rr = [apd.compute_quantum_path_diversity(cm, snm, scm, Q, A, N, K, M)
              for (Q, A) in qa]
    elif mtype == "classical":
        ei, tp, nptr, N, K = apd.load_classical_checkpoint(ckpt)
        rr = [apd.compute_classical_path_diversity(ei, tp, nptr, Q, A, M)
              for (Q, A) in qa]
    else:  # softmax state dict (conet_adam_training.py)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        N, K = sd["logits"].shape
        tp = torch.softmax(sd["logits"], dim=1).flatten().numpy()
        ei = torch.stack([torch.arange(N).repeat_interleave(K),
                          sd["node_neighbors"].flatten().long()])
        nptr = np.arange(0, (N + 1) * K, K)
        rr = [apd.compute_classical_path_diversity(ei, tp, nptr, Q, A, M)
              for (Q, A) in qa]
    return [[float(r[3]), float(r[0])] for r in rr]   # (total_success, IPR)

def compute(args):
    apd = amp._import_enumerators(args.root)
    cache_path = os.path.join(args.out, "ipr_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    idx = {f: s1.run_index(args.root, gsa.FAMILIES[f][0]) for f in IPR_FAMS}
    absent = [f for f, ix in idx.items() if not ix]
    if absent:
        print(f"families with no runs yet (skipped): {absent}")
    n_new = 0
    for B in gsa.B_LIST:
        seeds = sorted(s for (s, b) in idx["grover"] if b == B and s <= args.max_seed)
        for seed in seeds:
            tr, va, _ = s1.split_qa(idx["grover"][(seed, B)]["results"])
            for fam in IPR_FAMS:
                if fam in absent or (seed, B) not in idx[fam]:
                    continue
                mtype = gsa.FAMILIES[fam][1]
                ck = idx[fam][(seed, B)]["best"]
                for split, qa in (("train", tr), ("valid", va)):
                    key = f"{fam}|{seed}|{B}|{split}"
                    if key in cache:
                        continue
                    cache[key] = pipr_list(apd, mtype, ck, qa, args.M)
                    n_new += 1
                    if n_new % 10 == 0:
                        json.dump(cache, open(cache_path, "w"))
                        print(f"  ... cached {n_new} (at {key})")
    json.dump(cache, open(cache_path, "w"))
    print(f"compute done: {n_new} new, cache = {len(cache)}")
    return cache

def wipr(rows):
    """Accuracy-weighted <IPR> = sum(p_j * ipr_j) / sum(p_j)  (project convention)."""
    a = np.asarray(rows, dtype=float)
    if not len(a) or a[:, 0].sum() <= 0:
        return np.nan
    return float((a[:, 0] * a[:, 1]).sum() / a[:, 0].sum())

def pooled(cache, fam, B, split):
    rows = []
    for s in range(1, 99):
        rows.extend(cache.get(f"{fam}|{s}|{B}|{split}", []))
    return rows

def table(cache):
    fams = [f for f in IPR_FAMS if any(k.startswith(f + "|") for k in cache)]
    hdr = f"{'B':>4} {'split':>6} | " + " ".join(f"{f+' <IPR>':>12}" for f in fams)
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for split in ("train", "valid"):
        for B in gsa.B_LIST:
            vals = [wipr(pooled(cache, f, B, split)) for f in fams]
            if all(np.isnan(v) for v in vals):
                continue
            cells = " ".join(f"{v:>12.3f}" if np.isfinite(v) else f"{'-':>12}"
                             for v in vals)
            print(f"{B:>4} {split:>6} | {cells}")
        print("-" * len(hdr))

def fig_ipr_vs_B(cache, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, ("train", "valid")):
        for fam, col in IPR_FAMS.items():
            m = []
            for B in gsa.B_LIST:
                per_seed = [wipr(cache.get(f"{fam}|{s}|{B}|{split}", []))
                            for s in range(1, 99)
                            if f"{fam}|{s}|{B}|{split}" in cache]
                per_seed = [v for v in per_seed if np.isfinite(v)]
                m.append(np.mean(per_seed) if per_seed else np.nan)
            if not np.any(np.isfinite(m)):
                continue
            ax.plot(gsa.B_LIST, m, "-o", color=col, ms=4, lw=1.8, label=fam)
        ax.axhline(1.0, ls=":", color="gray", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(gsa.B_LIST); ax.set_xticklabels(gsa.B_LIST)
        ax.set_xlabel("B"); ax.set_title(split); ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy-weighted ⟨IPR⟩ (effective # paths)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Path diversity of the trained models: does inference-aware training "
                 "keep an ensemble (IPR>1) or one attenuated path (IPR≈1)?")
    fig.tight_layout()
    f = os.path.join(out, "D2_ipr_vs_B.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}")

def fig_scatter(cache, out, B=32):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, split in zip(axes, ("train", "valid")):
        for fam, col in (("qstd", "#ff7f0e"), ("cadam", "#1f77b4"), ("grover", "#d62728")):
            rows = np.asarray(pooled(cache, fam, B, split), dtype=float)
            if not len(rows):
                continue
            ax.scatter(rows[:, 0], rows[:, 1], s=10, alpha=0.45, color=col,
                       label=fam, edgecolors="none")
        ax.axvline(P_STAR1, ls=":", color="gray", lw=1)
        ax.set_xlabel("per-QA success prob p")
        ax.set_title(f"{split} (B={B})")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("per-QA IPR (# effective paths)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Where the diversity lives: per-QA (p, IPR)")
    fig.tight_layout()
    f = os.path.join(out, f"D2b_ipr_scatter_B{B}.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("QSR_ROOT", os.path.abspath(os.path.join(HERE, ".."))))
    ap.add_argument("--out", default=os.path.join(HERE, "_sweep_out"))
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--max-seed", type=int, default=8)
    ap.add_argument("--parts", default="D1,D2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    parts = args.parts.split(",")

    if "D1" in parts:
        hist = load_histories(args.root)
        fig_dynamics(hist, args.out)
    if "D2" in parts:
        cache = compute(args)
        table(cache)
        fig_ipr_vs_B(cache, args.out)
        fig_scatter(cache, args.out)

if __name__ == "__main__":
    main()
