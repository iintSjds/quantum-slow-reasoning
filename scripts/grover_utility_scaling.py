"""Utility-vs-budget preview: U(k) = accuracy at verified-sample budget k.

Motivation: an UNTRAINED (or capped) classical model
keeps p>0 everywhere, so "unlimited resampling" eventually solves everything;
the honest quantity is accuracy as a function of the spent budget.  This
script plots U(k) for every training regime we have, plus the quantum ladder
(grover-n trained, evaluated at its own budget 2n+1), from the existing
p-caches.  The classical curves here are trained-for-2 (or capped) models
swept over k; the trained-for-k ladder (cbX-k, qbk-k) is being added by
overnight3.

  U_cl(k)  = mean_i [1 - (1-p_i)^k]          (one model, budget swept)
  U_q(2n+1)= mean_i [A_n(p_i^{(n)})]         (one trained model PER budget)

Also computes the untrained (uniform-walk) curve directly from the graph.

Run: python scripts/grover_utility_scaling.py
"""
import os, sys, json, math
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("QSR_ROOT", os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import amplification_scaling as amp
import amplification_step1 as s1

OUT = os.path.join(HERE, "_sweep_out")
CACHE = json.load(open(os.path.join(OUT, "p_cache.json")))
UCACHE_PATH = os.path.join(OUT, "p_uniform.json")


def pooled(fam, B, split):
    ps = []
    for s in range(1, 33):
        ps += CACHE.get(f"{fam}|{s}|{B}|{split}", [])
    return np.asarray(ps)


def aa(p, n):
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    th = math.asin(math.sqrt(p))
    k = min(n, max(0, int(round(math.pi / (4 * th) - 0.5))))
    return math.sin((2 * k + 1) * th) ** 2


def A(ps, n):
    return float(np.mean([aa(p, n) for p in ps])) if len(ps) else np.nan


def U(ps, k):
    return float(np.mean(1.0 - (1.0 - np.asarray(ps)) ** k)) if len(ps) else np.nan


def uniform_p():
    """Exact per-question p of the UNTRAINED (uniform) walk, per (seed,B,split).

    Uses each cbestk_exact run's results.json for the split and its checkpoint
    only for node_neighbors; transition probs are uniform 1/K.
    """
    if os.path.exists(UCACHE_PATH):
        return json.load(open(UCACHE_PATH))
    apd = amp._import_enumerators(ROOT)
    idx = s1.run_index(ROOT, "from4090/grover_sweep/**/cbestkX2_*_best_model.pt")
    out = {}
    for (seed, B), rec in sorted(idx.items()):
        if seed > 8:
            continue
        tr, va, _ = s1.split_qa(rec["results"])
        sd = torch.load(rec["best"], map_location="cpu",
                        weights_only=False)["model_state_dict"]
        N, K = sd["logits"].shape
        tp = np.full(N * K, 1.0 / K)
        ei = torch.stack([torch.arange(N).repeat_interleave(K),
                          sd["node_neighbors"].flatten().long()])
        nptr = np.arange(0, (N + 1) * K, K)
        for split, qa in (("train", tr), ("valid", va)):
            out[f"uniform|{seed}|{B}|{split}"] = [
                float(apd.compute_classical_path_diversity(ei, tp, nptr, Q, A_, 8)[3])
                for (Q, A_) in qa]
    json.dump(out, open(UCACHE_PATH, "w"))
    return out


def main():
    uc = uniform_p()

    def upooled(B, split):
        ps = []
        for s in range(1, 9):
            ps += uc.get(f"uniform|{s}|{B}|{split}", [])
        return np.asarray(ps)

    ks = np.unique(np.round(np.logspace(0, 6, 40)).astype(int))
    Bs = (8, 32, 128)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    print(f"{'B':>4} {'split':>6} | {'family':<12} " +
          " ".join(f"U@{k:<6}" for k in (1, 3, 9, 100, 10**4, 10**6)))
    for j, B in enumerate(Bs):
        for i, split in enumerate(("train", "valid")):
            ax = axes[i][j]
            for fam, ps, col, lab in (
                    ("uniform", upooled(B, split), "0.6", "untrained (uniform walk)"),
                    ("cadam", pooled("cadam", B, split), "#1f77b4", "trained one-shot"),
                    ("cbestk_exact", pooled("cbestk_exact", B, split), "#17becf",
                     "trained best-of-2 (exact)"),
                    ("ccap25", pooled("ccap25", B, split), "#2ca02c",
                     "trained capped 0.25 (exact)")):
                if not len(ps):
                    continue
                ax.plot(ks, [U(ps, k) for k in ks], "-", color=col, lw=1.7, label=lab)
                row = " ".join(f"{U(ps,k):7.3f}" for k in (1, 3, 9, 100, 10**4, 10**6))
                print(f"{B:>4} {split:>6} | {fam:<12} {row}")
            # quantum ladder: budget 2n+1, model trained for n
            bud = [2 * n + 1 for n in (1, 2, 3, 4)]
            qa_ = [A(pooled(f, B, split), n) for f, n in
                   (("grover", 1), ("grover2", 2), ("grover3", 3), ("grover4", 4))]
            ax.plot(bud, qa_, "s-", color="#d62728", ms=6, lw=2.2, zorder=5,
                    label="Grover-$n$ trained, $n$ rounds\n(budget $2n{+}1$)")
            print(f"{B:>4} {split:>6} | {'grover(n)':<12} " +
                  " ".join(f"{v:7.3f}" for v in qa_) + "   @budgets 3/5/7/9")
            ax.set_xscale("log")
            ax.set_title(f"B={B}, {split}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("verified-sample budget $k$ (log)")
            if j == 0:
                ax.set_ylabel("accuracy at budget")
    axes[0][0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Utility vs budget: every classical training regime, swept over k; "
                 "quantum trained-per-budget ladder (red)")
    fig.tight_layout()
    f = os.path.join(OUT, "S8_utility_vs_budget.png")
    fig.savefig(f, dpi=150, bbox_inches="tight")
    print(f"fig -> {f}")


if __name__ == "__main__":
    main()
