"""
B-sweep analysis for the route-2 grover experiment (GPU sweep).

Compares three trained-model families on the SAME QA files / splits
(train_qa = first B pairs, valid_qa = fixed 64 held-out, verified identical
across families for matched (seed, B)):

  grover : QuCoNet-AR trained with the Grover-n=1 objective
           (adam lr=0.05, <=200 epochs + early stop; GPU sweep,
            pulled to from4090/grover_sweep/)
  qstd   : QuCoNet-AR trained one-shot (expr4 archive)   [protocol differs:
           optimizer/lr not matched -- reference, not a strict control]
  cstd   : CoNet classical trained one-shot (expr4, lr=3.0 REINFORCE)

Test-time systems at MATCHED budget (1 Grover iteration ~ 2 base queries):
  cstd + best@2, grover + 1 Grover, qstd + 1 Grover (post-hoc AA on the
  collapsed one-shot model -- isolates "training for the budget" from
  "just adding AA at test time").  Plus cstd best@inf ceiling (= non-dead
  fraction; a collapsed classical model can never exceed it).

Baselines are restricted per-B to the seeds whose grover run exists
(paired comparison); as of 2026-07-02 that is seeds 1-8 at B<=64 and
odd seeds at B in {96,128} (even-seed big-B runs still training).

Run from repo root (compute is cached; safe to interrupt/rerun):
  python scripts/grover_sweep_analysis.py
"""
import os, sys, glob, json, math, argparse
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amplification_scaling as amp          # aa_acc, frac_interior, enumerator import
import amplification_step1 as s1             # run_index, split_qa, p_list

FAMILIES = {   # name -> (glob under root, enumerator model type)
    "grover": ("from4090/grover_sweep/**/grover_n1_*_best_model.pt", "quantum"),
    "qstd":   ("from4090/expr4/quantum/**/*_best_model.pt",          "quantum"),
    "cstd":   ("from4090/expr4/classical/N_120_k_3_M_8/**/*_best_model.pt", "classical"),
    # matched-protocol classical (adam lr=0.05, same as the grover runs);
    # conet_adam_training.py saves a SoftmaxRouter state dict, not CSR tensors:
    "cadam":  ("from4090/expr4/conet_adam/**/*_best_model.pt",       "softmax"),
    # classical inference-aware control (trained for best-of-2):
    "cbestk": ("from4090/grover_sweep/**/cbestk2_*_best_model.pt",   "softmax"),
    # protocol-identical one-shot classical (same 200ep+ES run as cbestk;
    # sanity check that cadam's 120ep archive is converged):
    "cstd_adam": ("from4090/grover_sweep/**/cstd_adam_*_best_model.pt", "softmax"),
    # exact-gradient classical controls (differentiable DP; no REINFORCE
    # dead-pair blindness -- isolate pure objective shape):
    "cbestk_exact": ("from4090/grover_sweep/**/cbestkX2_*_best_model.pt", "softmax"),
    "cstd_exact":   ("from4090/grover_sweep/**/cstdX1_*_best_model.pt",   "softmax"),
    # grover-n=2 (attractor p*=0.095):
    "grover2": ("from4090/grover_sweep/**/grover_n2_*_best_model.pt", "quantum"),
    # n-scan (B=32, seeds 1-8 only; attractors p*=0.049 / 0.030):
    "grover3": ("from4090/grover_sweep/**/grover_n3_*_best_model.pt", "quantum"),
    "grover4": ("from4090/grover_sweep/**/grover_n4_*_best_model.pt", "quantum"),
    # capped exact-gradient controls (imported confidence target; seeds 1-8):
    "ccap25": ("from4090/grover_sweep/**/ccapX0.25_*_best_model.pt", "softmax"),
    "ccap50": ("from4090/grover_sweep/**/ccapX0.5_*_best_model.pt",  "softmax"),
    "ccap75": ("from4090/grover_sweep/**/ccapX0.75_*_best_model.pt", "softmax"),
    # max-entropy controls ("just add an entropy bonus"; B in {8,32,128},
    # seeds 1-8, coef 0.01/0.03/0.1; cent = one-shot+H, cbent = best-of-2+H):
    "centH01":  ("from4090/grover_sweep/**/centH0.01_*_best_model.pt",  "softmax"),
    "centH03":  ("from4090/grover_sweep/**/centH0.03_*_best_model.pt",  "softmax"),
    "centH10":  ("from4090/grover_sweep/**/centH0.1_*_best_model.pt",   "softmax"),
    "cbentH01": ("from4090/grover_sweep/**/cbentH0.01_*_best_model.pt", "softmax"),
    "cbentH03": ("from4090/grover_sweep/**/cbentH0.03_*_best_model.pt", "softmax"),
    "cbentH10": ("from4090/grover_sweep/**/cbentH0.1_*_best_model.pt",  "softmax"),
    # ── budget ladders (overnight3): each model trained for its OWN budget,
    #    evaluated at that budget.  qbkX-k = same-architecture QuCoNet-AR
    #    best-of-k ("semi" control, 720 params; AR p == classical Markov p by
    #    the which-path theorem).  cbestkX-k = classical CoNet exact best-of-k.
    #    grover5/6 extend the quantum ladder to budgets 11/13.  qbk + grover5/6
    #    are B in {8,32,128}; cbestkX-k is the full B grid.
    "qbkX2":  ("from4090/grover_sweep/**/qbkX2_*_best_model.pt",  "quantum"),
    "qbkX4":  ("from4090/grover_sweep/**/qbkX4_*_best_model.pt",  "quantum"),
    "qbkX8":  ("from4090/grover_sweep/**/qbkX8_*_best_model.pt",  "quantum"),
    "qbkX16": ("from4090/grover_sweep/**/qbkX16_*_best_model.pt", "quantum"),
    "qbkX32": ("from4090/grover_sweep/**/qbkX32_*_best_model.pt", "quantum"),
    "qbkX64": ("from4090/grover_sweep/**/qbkX64_*_best_model.pt", "quantum"),
    "cbestkX4":  ("from4090/grover_sweep/**/cbestkX4_*_best_model.pt",  "softmax"),
    "cbestkX8":  ("from4090/grover_sweep/**/cbestkX8_*_best_model.pt",  "softmax"),
    "cbestkX16": ("from4090/grover_sweep/**/cbestkX16_*_best_model.pt", "softmax"),
    "cbestkX32": ("from4090/grover_sweep/**/cbestkX32_*_best_model.pt", "softmax"),
    "cbestkX64": ("from4090/grover_sweep/**/cbestkX64_*_best_model.pt", "softmax"),
    "grover5": ("from4090/grover_sweep/**/grover_n5_*_best_model.pt", "quantum"),
    "grover6": ("from4090/grover_sweep/**/grover_n6_*_best_model.pt", "quantum"),
}
SPARSE = {"grover3", "grover4", "ccap25", "ccap50", "ccap75",
          "centH01", "centH03", "centH10", "cbentH01", "cbentH03", "cbentH10",
          "qbkX2", "qbkX4", "qbkX8", "qbkX16", "qbkX32", "qbkX64",
          "grover5", "grover6"}

# random-regular graph family (replication; own root dir so (seed,B) keys
# don't collide with the sliding-puzzle runs; std_* = one-shot QuCoNet):
FAMILIES_RR = {
    "grover_rr":       ("from4090/grover_sweep_rr/**/grover_n1_*_best_model.pt", "quantum"),
    "qstd_rr":         ("from4090/grover_sweep_rr/**/std_*_best_model.pt",       "quantum"),
    "cstd_rr":         ("from4090/grover_sweep_rr/**/cstd_adam_*_best_model.pt", "softmax"),
    "cbestk_exact_rr": ("from4090/grover_sweep_rr/**/cbestkX2_*_best_model.pt",  "softmax"),
    "cstd_exact_rr":   ("from4090/grover_sweep_rr/**/cstdX1_*_best_model.pt",    "softmax"),
    "ccap25_rr":       ("from4090/grover_sweep_rr/**/ccapX0.25_*_best_model.pt", "softmax"),
    # deep-n on rr (B in {8,32,128}, seeds 1-8):
    "grover2_rr":      ("from4090/grover_sweep_rr/**/grover_n2_*_best_model.pt", "quantum"),
    "grover3_rr":      ("from4090/grover_sweep_rr/**/grover_n3_*_best_model.pt", "quantum"),
    "grover4_rr":      ("from4090/grover_sweep_rr/**/grover_n4_*_best_model.pt", "quantum"),
}
B_LIST_RR = [8, 32, 128]

# problem-size scan (randreg N=240/480/960; SEPARATE root dirs + caches --
# (seed,B) filename keys collide across N).  N=480: quantum B=32 only
# (B=128 OOM on the 48 GB card); N=960: classical control only (GPU OOM).
def families_scanN(n):
    root = f"from4090/grover_sweep_rr_N{n}"
    return {
        "grover_rr":       (f"{root}/**/grover_n1_*_best_model.pt", "quantum"),
        "qstd_rr":         (f"{root}/**/std_*_best_model.pt",       "quantum"),
        "cbestk_exact_rr": (f"{root}/**/cbestkX2_*_best_model.pt",  "softmax"),
        "cstd_exact_rr":   (f"{root}/**/cstdX1_*_best_model.pt",    "softmax"),
    }
REF = "grover"           # family whose results.json defines the QA split
CACHE_NAME = "p_cache.json"

def p_list_softmax(apd, ckpt, qa, M):
    """Exact per-QA p for conet_adam_training.py checkpoints
    (state dict = logits [N,K] + node_neighbors [N,K])."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    N, K = sd["logits"].shape
    tp = torch.softmax(sd["logits"], dim=1).flatten().numpy()
    ei = torch.stack([torch.arange(N).repeat_interleave(K),
                      sd["node_neighbors"].flatten().long()])
    nptr = np.arange(0, (N + 1) * K, K)
    return [float(apd.compute_classical_path_diversity(ei, tp, nptr, Q, A, M)[3])
            for (Q, A) in qa]
B_LIST = [8, 16, 32, 48, 64, 96, 128]
P_STAR1 = math.sin(math.pi / 6) ** 2          # attractor of the n=1 objective = 0.25

def aa_sr(ps, n=1):
    return float(np.mean([amp.aa_acc(p, n) for p in ps])) if len(ps) else np.nan

def bestk_sr(ps, k=2):
    ps = np.asarray(ps)
    return float(np.mean(1.0 - (1.0 - ps) ** k)) if len(ps) else np.nan

def ceiling(ps):
    return float(np.mean(np.asarray(ps) > 0.02)) if len(ps) else np.nan

def dead_frac(ps):
    return float(np.mean(np.asarray(ps) <= 0.02)) if len(ps) else np.nan

# ───────────────────────── compute stage (cached) ─────────────────────
def compute(args):
    global REF
    apd = amp._import_enumerators(args.root)
    cache_path = os.path.join(args.out, CACHE_NAME)
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    idx = {f: s1.run_index(args.root, pat) for f, (pat, _) in FAMILIES.items()}
    absent = [f for f, ix in idx.items() if not ix]
    if absent:
        print(f"families with no runs yet (skipped): {absent}")
    if REF in absent:   # e.g. N=960: classical control only (GPU OOM)
        REF = max(idx, key=lambda f: len(idx[f]))
        print(f"REF family absent; falling back to REF={REF}")

    n_new = 0
    for B in B_LIST:
        seeds = sorted(s for (s, b) in idx[REF] if b == B and s <= args.max_seed)
        for seed in seeds:
            rj = idx[REF][(seed, B)]["results"]
            tr, va, _ = s1.split_qa(rj)
            # sanity: split must match the archived one-shot run's split
            rjq = idx.get("qstd", {}).get((seed, B), {}).get("results")
            if rjq:
                trq, vaq, _ = s1.split_qa(rjq)
                assert tr == trq and va == vaq, f"split mismatch seed{seed} B{B}"
            for fam, (_, mtype) in FAMILIES.items():
                if fam in absent:
                    continue
                if (seed, B) not in idx[fam]:
                    if fam not in SPARSE:
                        print(f"  MISSING {fam} seed{seed} B{B}")
                    continue
                ck = idx[fam][(seed, B)]["best"]
                for split, qa in (("train", tr), ("valid", va)):
                    key = f"{fam}|{seed}|{B}|{split}"
                    if key in cache:
                        continue
                    if mtype == "softmax":
                        cache[key] = p_list_softmax(apd, ck, qa, args.M)
                    else:
                        cache[key] = s1.p_list(apd, mtype, ck, qa, args.M)
                    n_new += 1
                    if n_new % 10 == 0:
                        json.dump(cache, open(cache_path, "w"))
                        print(f"  ... cached {n_new} new evals (at {key})")
    json.dump(cache, open(cache_path, "w"))
    print(f"compute done: {n_new} new evals, cache = {len(cache)} entries")
    return cache

# ───────────────────────── aggregation ────────────────────────────────
def seeds_at(cache, B):
    return sorted({int(k.split("|")[1]) for k in cache
                   if k.split("|")[0] == REF and int(k.split("|")[2]) == B})

def get_p(cache, fam, seed, B, split):
    return np.asarray(cache.get(f"{fam}|{seed}|{B}|{split}", []))

def per_seed(cache, fam, B, split, fn):
    vals = []
    for s in seeds_at(cache, B):
        ps = get_p(cache, fam, s, B, split)
        if len(ps):
            vals.append(fn(ps))
    return np.asarray(vals)

def curve(cache, fam, split, fn):
    """mean / min / max across paired seeds, per B."""
    m, lo, hi = [], [], []
    for B in B_LIST:
        v = per_seed(cache, fam, B, split, fn)
        m.append(np.mean(v) if len(v) else np.nan)
        lo.append(np.min(v) if len(v) else np.nan)
        hi.append(np.max(v) if len(v) else np.nan)
    return np.array(m), np.array(lo), np.array(hi)

def pooled(cache, fam, B, split):
    ps = []
    for s in seeds_at(cache, B):
        ps.extend(get_p(cache, fam, s, B, split))
    return np.asarray(ps)

# ───────────────────────── table ──────────────────────────────────────
def _fmt(x, w=9):
    return f"{x:>{w}.3f}" if np.isfinite(x) else " " * (w - 1) + "-"

def table(cache):
    hdr = (f"{'B':>4} {'split':>6} {'#sd':>4} | "
           f"{'cadam 1sh':>9} {'cadam b@2':>9} {'ceil':>6} | "
           f"{'cbestk b@2':>10} {'cb ceil':>7} {'cb int':>6} | "
           f"{'cbX b@2':>8} {'cbX ceil':>8} | "
           f"{'cstdA b@2':>9} | {'qstd+AA1':>9} | "
           f"{'grv 1sh':>8} {'grv+AA1':>8} {'grv int':>8} | "
           f"{'g2+AA2':>7} {'g3+AA3':>7} {'g4+AA4':>7}")
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for split in ("train", "valid"):
        for B in B_LIST:
            ns = len(seeds_at(cache, B))
            ca = pooled(cache, "cadam", B, split)
            cb = pooled(cache, "cbestk", B, split)
            cx = pooled(cache, "cbestk_exact", B, split)
            cs = pooled(cache, "cstd_adam", B, split)
            q = pooled(cache, "qstd", B, split)
            g = pooled(cache, "grover", B, split)
            g2 = pooled(cache, "grover2", B, split)
            g3 = pooled(cache, "grover3", B, split)
            g4 = pooled(cache, "grover4", B, split)
            if not len(g):
                continue
            ca_1 = np.mean(ca) if len(ca) else np.nan
            g_1 = np.mean(g) if len(g) else np.nan
            g_int = amp.frac_interior(g) if len(g) else np.nan
            cb_int = amp.frac_interior(cb) if len(cb) else np.nan
            print(f"{B:>4} {split:>6} {ns:>4} | "
                  f"{_fmt(ca_1)} {_fmt(bestk_sr(ca,2))} {_fmt(ceiling(ca),6)} | "
                  f"{_fmt(bestk_sr(cb,2),10)} {_fmt(ceiling(cb),7)} {_fmt(cb_int,6)} | "
                  f"{_fmt(bestk_sr(cx,2),8)} {_fmt(ceiling(cx),8)} | "
                  f"{_fmt(bestk_sr(cs,2))} | {_fmt(aa_sr(q,1))} | "
                  f"{_fmt(g_1,8)} {_fmt(aa_sr(g,1),8)} {_fmt(g_int,8)} | "
                  f"{_fmt(aa_sr(g2,2),7)} {_fmt(aa_sr(g3,3),7)} {_fmt(aa_sr(g4,4),7)}")
        print("-" * len(hdr))

def rr_table(cache):
    hdr = (f"{'B':>4} {'split':>6} {'#sd':>4} | "
           f"{'cstd 1sh':>8} {'cstd b@2':>8} {'ceil':>6} | "
           f"{'cbX b@2':>8} {'cbX ceil':>8} | {'cstdX b@2':>9} | "
           f"{'qstd+AA1':>9} | {'grv 1sh':>8} {'grv+AA1':>8} {'grv int':>8}")
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for split in ("train", "valid"):
        for B in B_LIST:
            ns = len(seeds_at(cache, B))
            cs = pooled(cache, "cstd_rr", B, split)
            cx = pooled(cache, "cbestk_exact_rr", B, split)
            sx = pooled(cache, "cstd_exact_rr", B, split)
            q  = pooled(cache, "qstd_rr", B, split)
            g  = pooled(cache, "grover_rr", B, split)
            if not (len(g) or len(cx)):   # N=960 rows are classical-only
                continue
            print(f"{B:>4} {split:>6} {ns:>4} | "
                  f"{_fmt(np.mean(cs) if len(cs) else np.nan,8)} "
                  f"{_fmt(bestk_sr(cs,2),8)} {_fmt(ceiling(cs),6)} | "
                  f"{_fmt(bestk_sr(cx,2),8)} {_fmt(ceiling(cx),8)} | "
                  f"{_fmt(bestk_sr(sx,2),9)} | {_fmt(aa_sr(q,1),9)} | "
                  f"{_fmt(np.mean(g) if len(g) else np.nan,8)} {_fmt(aa_sr(g,1),8)} "
                  f"{_fmt(amp.frac_interior(g) if len(g) else np.nan,8)}")
        print("-" * len(hdr))

# ───────────────────────── figures ────────────────────────────────────
STYLE = {  # fam+system -> (label, color, ls, marker)
    "cadam_b2":   ("CoNet(1-shot) + best@2",          "#1f77b4", "-",  "o"),
    "cbestk_b2":  ("CoNet(best-2-trained) + best@2",  "#08306b", "-",  "D"),
    "cbestkx_b2": ("CoNet(best-2, exact grad) + best@2", "#17becf", "-", "v"),
    "grover_aa":  ("QuCoNet(grover-1) + 1 Grover",    "#d62728", "-",  "s"),
    "grover2_aa": ("QuCoNet(grover-2) + 2 Grover",    "#7f0000", "--", "P"),
    "qstd_aa":    ("QuCoNet(1-shot) + 1 Grover",      "#ff7f0e", "--", "^"),
    "cadam_ceil": ("CoNet(1-shot) best@$\\infty$ ceiling", "0.4", ":", ""),
    "cadam_1":    ("CoNet 1-shot",                    "#9ecae1", ":",  ""),
    "grover_1":   ("QuCoNet(grover-1) 1-shot",        "#ff9896", ":",  ""),
}

def fig_headline(cache, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))   # own y-scale per split
    for ax, split in zip(axes, ("train", "valid")):
        for key, fam, fn in (
                ("cadam_1",    "cadam",  np.mean),
                ("grover_1",   "grover", np.mean),
                ("cadam_ceil", "cadam",  ceiling),
                ("cadam_b2",   "cadam",  lambda p: bestk_sr(p, 2)),
                ("cbestk_b2",  "cbestk", lambda p: bestk_sr(p, 2)),
                ("cbestkx_b2", "cbestk_exact", lambda p: bestk_sr(p, 2)),
                ("qstd_aa",    "qstd",   lambda p: aa_sr(p, 1)),
                ("grover2_aa", "grover2", lambda p: aa_sr(p, 2)),
                ("grover_aa",  "grover", lambda p: aa_sr(p, 1))):
            lab, col, ls, mk = STYLE[key]
            m, lo, hi = curve(cache, fam, split, fn)
            if not np.any(np.isfinite(m)):
                continue
            ax.plot(B_LIST, m, ls, color=col, marker=mk, ms=4, lw=1.8, label=lab)
            if key in ("cadam_b2", "cbestk_b2", "grover_aa"):
                ax.fill_between(B_LIST, lo, hi, color=col, alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(B_LIST); ax.set_xticklabels(B_LIST)
        ax.set_xlabel("B (training pairs)")
        ax.set_title(f"{split} — matched budget (1 Grover ≈ 2 queries)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Train for the budget you'll spend: inference-aware QuCoNet vs one-shot-trained systems", y=1.02)
    fig.tight_layout()
    f = os.path.join(out, "S1_headline_vs_B.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}")

def fig_hist(cache, out, Bs=(8, 32, 128)):
    fig, axes = plt.subplots(2, len(Bs), figsize=(3.6 * len(Bs), 6), sharex=True)
    for j, B in enumerate(Bs):
        for i, split in enumerate(("train", "valid")):
            ax = axes[i, j]
            for fam, col in (("cadam", "#1f77b4"), ("cbestk", "#08306b"),
                             ("qstd", "#ff7f0e"), ("grover", "#d62728")):
                ps = pooled(cache, fam, B, split)
                if len(ps):
                    ax.hist(ps, bins=np.linspace(0, 1, 41), density=True,
                            histtype="step", lw=1.6, color=col, label=fam)
            ax.axvline(P_STAR1, ls=":", color="gray", lw=1)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"B={B}")
            if j == 0:
                ax.set_ylabel(f"{split}\ndensity (log)")
            if i == 1:
                ax.set_xlabel("per-QA success prob p")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("p distributions: one-shot collapses to {0,1}; grover-1 parks mass at p*≈0.25 (dotted)")
    fig.tight_layout()
    f = os.path.join(out, "S2_p_hist.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}")

def fig_interior(cache, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, split in zip(axes, ("train", "valid")):
        for fam, col in (("cadam", "#1f77b4"), ("cbestk", "#08306b"),
                         ("qstd", "#ff7f0e"), ("grover", "#d62728")):
            m, lo, hi = curve(cache, fam, split, amp.frac_interior)
            if not np.any(np.isfinite(m)):
                continue
            ax.plot(B_LIST, m, "-o", color=col, ms=4, lw=1.8, label=fam)
            ax.fill_between(B_LIST, lo, hi, color=col, alpha=0.15, lw=0)
        ax.set_xscale("log", base=2); ax.set_xticks(B_LIST); ax.set_xticklabels(B_LIST)
        ax.set_xlabel("B"); ax.set_title(split); ax.grid(alpha=0.3)
    axes[0].set_ylabel("interior mass  frac(0.02 < p < 0.98)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Amplifiable headroom vs B")
    fig.tight_layout()
    f = os.path.join(out, "S3_interior_vs_B.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}")

def fig_nscan(cache, out, B=32):
    """Attractor scan at call budgets q=3,5,7,9.
    Each parks its mass at p*(q) = sin^2(pi/(2q))."""
    fams = [("grover", 1, "#d62728"), ("grover2", 2, "#7f0000"),
            ("grover3", 3, "#9467bd"), ("grover4", 4, "#4b0082")]
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True,
                                  sharey=True)
    axes = axes_grid.ravel()
    for ax, (fam, n, col) in zip(axes, fams):
        ps = pooled(cache, fam, B, "train")
        if not len(ps):
            ax.set_visible(False); continue
        pstar = math.sin(math.pi / (2 * (2 * n + 1))) ** 2
        ax.hist(ps, bins=np.linspace(0, 1, 81), density=True,
                histtype="stepfilled", color=col, alpha=0.75)
        ax.axvline(pstar, ls="--", color="k", lw=1.2)
        ax.set_yscale("log")
        q = 2 * n + 1
        ax.set_title(rf"$q={q}$, $p^*({q})={pstar:.3f}$", fontsize=10)
        ax.set_xlabel(r"training probability $p$")
        ax.set_xlim(0, 1)
    axes_grid[0, 0].set_ylabel("probability density")
    axes_grid[1, 0].set_ylabel("probability density")
    fig.tight_layout()
    f = os.path.join(out, "S4_nscan_attractor.pdf")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}")

def fig_headline_rr(cache, out):
    """Replication on random-regular graphs (seeds 1-8, B in {8,32,128})."""
    series = [
        ("cstd_rr 1-shot",              "cstd_rr",         np.mean,                  "#9ecae1", ":",  ""),
        ("cstd_rr best@$\\infty$ ceil", "cstd_rr",         ceiling,                  "0.4",     ":",  ""),
        ("cstd_rr + best@2",            "cstd_rr",         lambda p: bestk_sr(p, 2), "#1f77b4", "-",  "o"),
        ("cbestk-exact_rr + best@2",    "cbestk_exact_rr", lambda p: bestk_sr(p, 2), "#17becf", "-",  "v"),
        ("QuCoNet 1-shot + 1 Grover",   "qstd_rr",         lambda p: aa_sr(p, 1),    "#ff7f0e", "--", "^"),
        ("grover-1 1-shot",             "grover_rr",       np.mean,                  "#ff9896", ":",  ""),
        ("grover-1 + 1 Grover",         "grover_rr",       lambda p: aa_sr(p, 1),    "#d62728", "-",  "s"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, split in zip(axes, ("train", "valid")):
        for lab, fam, fn, col, ls, mk in series:
            m, lo, hi = curve(cache, fam, split, fn)
            if not np.any(np.isfinite(m)):
                continue
            ax.plot(B_LIST, m, ls, color=col, marker=mk, ms=4, lw=1.8, label=lab)
            if mk in ("o", "s", "v"):
                ax.fill_between(B_LIST, lo, hi, color=col, alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(B_LIST); ax.set_xticklabels(B_LIST)
        ax.set_xlabel("B (training pairs)")
        ax.set_title(f"{split} — random-regular graphs (N=120, K=3)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Replication on a second graph family: matched-budget comparison")
    fig.tight_layout()
    f = os.path.join(out, "S5_headline_rr.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}")

def entropy_table(cache):
    """Max-entropy controls vs the exact-gradient best-of-2 reference.
    Question: does an entropy bonus recover test-time improvability?"""
    ent_B = [8, 32, 128]
    hdr = (f"{'B':>4} {'split':>6} | {'cbX b@2':>8} {'int':>6} | "
           f"{'cH.01':>7} {'cH.03':>7} {'cH.10':>7} | "
           f"{'cbH.01':>7} {'cbH.03':>7} {'cbH.10':>7} | "
           f"{'cbH.03 int':>10} {'ceil':>6} | {'grv+AA1':>8}")
    print("\n=== max-entropy controls (b@2 unless noted; pooled seeds 1-8) ===")
    print(hdr); print("-" * len(hdr))
    for split in ("train", "valid"):
        for B in ent_B:
            cx = pooled(cache, "cbestk_exact", B, split)
            g = pooled(cache, "grover", B, split)
            ce = [pooled(cache, f, B, split) for f in ("centH01", "centH03", "centH10")]
            cb = [pooled(cache, f, B, split) for f in ("cbentH01", "cbentH03", "cbentH10")]
            if not any(len(x) for x in ce + cb):
                continue
            print(f"{B:>4} {split:>6} | {_fmt(bestk_sr(cx,2),8)} "
                  f"{_fmt(amp.frac_interior(cx) if len(cx) else np.nan,6)} | "
                  + " ".join(_fmt(bestk_sr(p, 2), 7) for p in ce) + " | "
                  + " ".join(_fmt(bestk_sr(p, 2), 7) for p in cb) + " | "
                  f"{_fmt(amp.frac_interior(cb[1]) if len(cb[1]) else np.nan,10)} "
                  f"{_fmt(ceiling(cb[1]),6)} | {_fmt(aa_sr(g,1),8)}")
        print("-" * len(hdr))

def fig_deepn_valid(cache, out, rr=False):
    """Deep-n ladder vs B: A_n-trained + AA_n, against the strongest
    classical controls at the most generous matched budget (best-of-9,
    the budget of the deepest quantum system shown)."""
    sfx = "_rr" if rr else ""
    fams = [(f"grover{sfx}", 1, "#d62728"), (f"grover2{sfx}", 2, "#7f0000"),
            (f"grover3{sfx}", 3, "#9467bd"), (f"grover4{sfx}", 4, "#4b0082")]
    cls = [(f"cbestk_exact{sfx}", "best-of-two, nine-attempt reference", "#17becf"),
           (f"ccap25{sfx}", r"capped at $0.25$, nine-attempt reference", "#1f77b4")]
    Bs = B_LIST_RR if rr else B_LIST
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, letter in zip(axes, ("a", "b")):
        ax.text(-0.07, 1.09, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top")
    for ax, split in zip(axes, ("train", "valid")):
        for fam, lab, col in cls:
            m = [bestk_sr(pooled(cache, fam, B, split), 9) for B in Bs]
            if np.any(np.isfinite(m)):
                ax.plot(Bs, m, "--", color=col, marker="x", ms=5, lw=1.6, label=lab)
        for fam, n, col in fams:
            m = [aa_sr(pooled(cache, fam, B, split), n) for B in Bs]
            if np.any(np.isfinite(m)):
                ax.plot(Bs, m, "-", color=col, marker="o", ms=4, lw=1.8,
                        label=rf"Grover-trained, $q={2 * n + 1}$")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Bs); ax.set_xticklabels(Bs)
        ax.set_xlabel(r"training-set size $B$")
        ax.set_title("training" if split == "train" else "held-out")
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    f = os.path.join(out, f"S6_deepn_valid{sfx}.pdf")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    print(f"fig -> {f}")

def main():
    global FAMILIES, B_LIST, REF, CACHE_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("QSR_ROOT", os.path.abspath(os.path.join(HERE, ".."))))
    ap.add_argument("--out", default=os.path.join(HERE, "_sweep_out"))
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--max-seed", type=int, default=8)
    ap.add_argument("--figs-only", action="store_true")
    ap.add_argument("--qa", default="sliding",
                    choices=("sliding", "randreg",
                             "randreg240", "randreg480", "randreg960"),
                    help="graph family: sliding puzzle (default), random-"
                         "regular N=120, or the problem-size scan dirs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.qa == "randreg":
        FAMILIES, B_LIST, REF = FAMILIES_RR, B_LIST_RR, "grover_rr"
        CACHE_NAME = "p_cache_rr.json"
    elif args.qa.startswith("randreg"):          # randreg240/480/960
        n = args.qa[len("randreg"):]
        FAMILIES, B_LIST, REF = families_scanN(n), [32, 128], "grover_rr"
        CACHE_NAME = f"p_cache_rr_N{n}.json"

    if args.figs_only:
        cache = json.load(open(os.path.join(args.out, CACHE_NAME)))
        if not any(k.startswith(REF + "|") for k in cache):
            REF = max({k.split("|")[0] for k in cache},
                      key=lambda f: sum(k.startswith(f + "|") for k in cache))
    else:
        cache = compute(args)
    if args.qa == "randreg":
        rr_table(cache)
        fig_headline_rr(cache, args.out)
        fig_deepn_valid(cache, args.out, rr=True)
    elif args.qa.startswith("randreg"):          # size scan: table only
        rr_table(cache)
    else:
        table(cache)
        entropy_table(cache)
        fig_headline(cache, args.out)
        fig_hist(cache, args.out)
        fig_interior(cache, args.out)
        fig_nscan(cache, args.out)
        fig_deepn_valid(cache, args.out, rr=False)

if __name__ == "__main__":
    main()
