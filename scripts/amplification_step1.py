"""
Step-1 cheap post-process checks for the test-time-amplification idea.
No retraining, no circuit simulation -- reuses exact per-QA p enumeration
from conet/analyze_path_diversity.py and the analytic curves in
amplification_scaling.py.

Two questions:

  Part A (held-out):  the make-or-break F1 was computed on the TRAINING QA
      (qa[:B]).  Does "QuCoNet is more bimodal / less amplifiable" survive on
      the fixed 64-pair held-out set (valid_qa, identical across B, no train
      overlap)?  If on held-out QuCoNet has MORE interior mass, the inversion
      is a training-set artifact.

  Part B (trajectory): the only run with per-epoch checkpoints is seed2_B3
      (24 quantum epoch dumps + 24 classical step dumps).  Trace interior mass
      (amplifiable headroom) and mean_p (SR proxy) vs training step for BOTH
      models.  Route-1 hypothesis: QuCoNet headroom peaks at intermediate SR
      and then collapses to the bimodal endpoint -> stop early = more headroom.

Run from repo root:
  python scripts/amplification_step1.py \
      --parts A,B --Bs 16,64,128 --seeds 8
"""
import os, sys, glob, re, json, argparse
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amplification_scaling as amp          # cl_acc, aa_acc, frac_interior, enumerator import

SEED_RE = amp.SEED_RE
BQA_RE  = amp.BQA_RE
frac_interior = amp.frac_interior

QROOT_GLOB = "archive/expr4/quantum/**/*_best_model.pt"
CROOT_GLOB = "archive/expr4/classical/N_120_k_3_M_8/**/*_best_model.pt"   # lr=3.0, NOT legacy
QTRAJ_DIR  = "archive/expr4/quantum_ckpts/seed2_B3"
CTRAJ_DIR  = ("archive/expr4/classical/ckpts/N_120_k_3_M_8/"
              "conet_rl_sliding_puzzle_N120_K3_M8_B192_D6_seed2_B3_s42")

# ───────────────────────── indexing / loading ─────────────────────────
def run_index(root, pat):
    """Map (seed,B) -> dict(best=..., results=...) for each run directory."""
    idx = {}
    for bm in glob.glob(os.path.join(root, pat), recursive=True):
        s = SEED_RE.search(bm)
        b = BQA_RE.search(os.path.basename(bm))
        if not (s and b):
            continue
        seed, B = int(s.group(1)), int(b.group(1))
        rj = bm.replace("_best_model.pt", "_results.json")
        idx[(seed, B)] = dict(best=bm, results=rj if os.path.exists(rj) else None)
    return idx

def split_qa(results_path):
    d = json.load(open(results_path))
    tr = [(int(q), int(a)) for q, a in d["train_qa"]]
    va = [(int(q), int(a)) for q, a in d["valid_qa"]]
    return tr, va, d

def p_list(apd, model, ckpt, qa, M):
    """Exact per-QA success probability for a list of (Q,A)."""
    if model == "classical":
        ei, tp, nptr, N, K = apd.load_classical_checkpoint(ckpt)
        return [float(apd.compute_classical_path_diversity(ei, tp, nptr, Q, A, M)[3])
                for (Q, A) in qa]
    else:
        cm, snm, scm, N, K, cfg = apd.load_quantum_checkpoint(ckpt)
        return [float(apd.compute_quantum_path_diversity(cm, snm, scm, Q, A, N, K, M)[3])
                for (Q, A) in qa]

# ───────────────────────── Part A: held-out ───────────────────────────
def collect_heldout(root, apd, Bs, seeds, M):
    qidx = run_index(root, QROOT_GLOB)
    cidx = run_index(root, CROOT_GLOB)
    paired = sorted({s for (s, b) in qidx} & {s for (s, b) in cidx})
    use = paired[:seeds] if isinstance(seeds, int) else seeds
    print(f"paired seeds: {len(paired)}; using {len(use)}: {use[:12]}")
    rows = []
    for seed in use:
        for B in Bs:
            if (seed, B) not in qidx or (seed, B) not in cidx:
                print(f"  skip seed{seed} B{B}"); continue
            rjq = qidx[(seed, B)]["results"] or cidx[(seed, B)]["results"]
            tr, va, _ = split_qa(rjq)
            rec = {}
            for model, ix in (("classical", cidx), ("quantum", qidx)):
                ck = ix[(seed, B)]["best"]
                for split, qa in (("train", tr), ("valid", va)):
                    ps = p_list(apd, model, ck, qa, M)
                    rows.append(dict(model=model, seed=seed, B=B, split=split, p=ps))
                    rec[(model, split)] = ps
            print(f"  seed{seed:>3} B{B:>3} | "
                  f"cl tr(mp={np.mean(rec[('classical','train')]):.2f},"
                  f"in={frac_interior(rec[('classical','train')]):.2f}) "
                  f"va(mp={np.mean(rec[('classical','valid')]):.2f},"
                  f"in={frac_interior(rec[('classical','valid')]):.2f}) | "
                  f"qu tr(mp={np.mean(rec[('quantum','train')]):.2f},"
                  f"in={frac_interior(rec[('quantum','train')]):.2f}) "
                  f"va(mp={np.mean(rec[('quantum','valid')]):.2f},"
                  f"in={frac_interior(rec[('quantum','valid')]):.2f})")
    return rows

def pool(rows, model, B, split):
    ps = []
    for r in rows:
        if r["model"] == model and r["B"] == B and r["split"] == split:
            ps.extend(r["p"])
    return np.asarray(ps)

def summary_heldout(rows, Bs):
    print("\n" + "=" * 86)
    print(f"{'B':>4} {'split':>6} {'model':>10} {'n':>5} {'mean_p':>8} "
          f"{'dead<.02':>9} {'interior':>9} {'solved>.98':>11}")
    print("-" * 86)
    for B in Bs:
        for split in ("train", "valid"):
            for model in ("classical", "quantum"):
                ps = pool(rows, model, B, split)
                if not len(ps):
                    continue
                print(f"{B:>4} {split:>6} {model:>10} {len(ps):>5} {ps.mean():>8.3f} "
                      f"{np.mean(ps<=0.02):>9.2f} {frac_interior(ps):>9.2f} "
                      f"{np.mean(ps>=0.98):>11.2f}")
        print("-" * 86)

def fig_heldout_hist(rows, B, split, out):
    pc, pq = pool(rows, "classical", B, split), pool(rows, "quantum", B, split)
    if not len(pc) or not len(pq):
        print(f"[F1b] no data B={B} split={split}"); return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 26)
    ax.hist(pc, bins=bins, alpha=0.55, color="#1f77b4",
            label=f"CoNet   (interior={frac_interior(pc):.2f}, mp={pc.mean():.2f})")
    ax.hist(pq, bins=bins, alpha=0.55, color="#d62728",
            label=f"QuCoNet (interior={frac_interior(pq):.2f}, mp={pq.mean():.2f})")
    ax.axvspan(0.02, 0.98, color="gray", alpha=0.08)
    ax.set_xlabel("per-QA success probability  p"); ax.set_ylabel("count")
    ax.set_title(f"F1b: per-QA p, {split} split, B={B}  (interior = amplifiable)")
    ax.legend(); fig.tight_layout()
    f = os.path.join(out, f"F1b_{split}_hist_B{B}.png"); fig.savefig(f, dpi=150); plt.close(fig)
    print(f"[F1b] saved {f}")

def fig_interior_vs_B(rows, Bs, out):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    styles = {("classical","train"):("#1f77b4","o-"), ("classical","valid"):("#7fb3e0","o--"),
              ("quantum","train"):("#d62728","s-"),   ("quantum","valid"):("#f0a0a0","s--")}
    for (model, split), (c, ls) in styles.items():
        ys = [frac_interior(pool(rows, model, B, split)) for B in Bs]
        ax.plot(Bs, ys, ls, color=c, label=f"{model} {split}")
    ax.set_xscale("log"); ax.set_xlabel("training batch B"); ax.set_ylabel("interior mass (0.02<p<0.98)")
    ax.set_title("F5: amplifiable (interior) mass vs B, train vs held-out")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    f = os.path.join(out, "F5_interior_vs_B.png"); fig.savefig(f, dpi=150); plt.close(fig)
    print(f"[F5] saved {f}")

# ───────────────────────── Part B: seed2_B3 trajectory ────────────────
def _step_num(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1

def verify_seed2(apd, root):
    """Confirm the epoch dump == graph seed 2 by the identity test: the converged
    model solves ONLY seed2's 3 train QA (train_mp=1) and ~0 on other seeds' QA.
    (Param-matching to the archived best_model FAILS -- the per-epoch dump is a
    separate init of the same seed2/B3 config; both reach SR~1 via different coins.)"""
    cm, snm, scm, N, K, _ = apd.load_quantum_checkpoint(
        os.path.join(root, QTRAJ_DIR, "epoch_0120.pt"))
    qidx = run_index(root, QROOT_GLOB)
    def train3(seed):
        tr, _, _ = split_qa(qidx[(seed, 3)]["results"]); return tr[:3]
    hits = {}
    for seed in (1, 2, 3, 5):
        mp = np.mean([apd.compute_quantum_path_diversity(cm, snm, scm, Q, A, N, K, 8)[3]
                      for (Q, A) in train3(seed)])
        hits[seed] = float(mp)
    ok = hits[2] > 0.95 and max(v for s, v in hits.items() if s != 2) < 0.1
    print(f"[verify] epoch_0120 train_mp by seed: {hits}  -> "
          f"{'seed2 CONFIRMED' if ok else 'AMBIGUOUS'}")
    return ok

def collect_trajectory(root, apd, M):
    qidx = run_index(root, QROOT_GLOB)
    tr, va, dq = split_qa(qidx[(2, 3)]["results"])     # seed2 B3 QA split (3 train, 64 valid)
    hist = dq.get("history", [])
    qfiles = sorted(glob.glob(os.path.join(root, QTRAJ_DIR, "epoch_*.pt")), key=_step_num)
    cfiles = sorted(glob.glob(os.path.join(root, CTRAJ_DIR, "step_*.pt")),  key=_step_num)
    print(f"trajectory: {len(qfiles)} quantum epoch ckpts, {len(cfiles)} classical step ckpts")

    def series(files, model):
        rows = []
        for f in files:
            ptr = p_list(apd, model, f, tr, M)
            pva = p_list(apd, model, f, va, M)
            rows.append(dict(step=_step_num(f),
                             train_mp=float(np.mean(ptr)),
                             valid_mp=float(np.mean(pva)),
                             valid_interior=frac_interior(pva),
                             valid_dead=float(np.mean(np.asarray(pva) <= 0.02)),
                             valid_solved=float(np.mean(np.asarray(pva) >= 0.98))))
            print(f"  {model:>9} step{rows[-1]['step']:>4}: "
                  f"train_mp={rows[-1]['train_mp']:.3f} valid_mp={rows[-1]['valid_mp']:.3f} "
                  f"valid_interior={rows[-1]['valid_interior']:.3f}")
        return rows

    return dict(quantum=series(qfiles, "quantum"),
                classical=series(cfiles, "classical"),
                history=hist)

def fig_trajectory(traj, out):
    q, c, hist = traj["quantum"], traj["classical"], traj["history"]
    qs = [r["step"] for r in q]; cs = [r["step"] for r in c]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.6), sharex=True)

    # panel 1: SR proxy (mean_p) train + valid
    ax1.plot(qs, [r["train_mp"] for r in q], "s-", color="#d62728", ms=4, label="QuCoNet train mp")
    ax1.plot(qs, [r["valid_mp"] for r in q], "s--", color="#f0a0a0", ms=4, label="QuCoNet valid mp")
    ax1.plot(cs, [r["train_mp"] for r in c], "o-", color="#1f77b4", ms=4, label="CoNet train mp")
    ax1.plot(cs, [r["valid_mp"] for r in c], "o--", color="#7fb3e0", ms=4, label="CoNet valid mp")
    if hist:
        he = [h["epoch"] for h in hist if "valid_sr" in h]
        hv = [h["valid_sr"] for h in hist if "valid_sr" in h]
        ax1.plot(he, hv, ":", color="k", lw=1, alpha=0.6, label="qu valid_sr (results.json)")
    ax1.set_ylabel("mean_p  (success-prob proxy)")
    ax1.set_title("F4: seed2_B3 trajectory — SR proxy (top) & held-out amplifiable mass (bottom)")
    ax1.legend(fontsize=7.5); ax1.grid(alpha=0.3)

    # panel 2: held-out interior mass (the headroom that AA/best@n can exploit)
    ax2.plot(qs, [r["valid_interior"] for r in q], "s-", color="#d62728", ms=4, label="QuCoNet valid interior")
    ax2.plot(cs, [r["valid_interior"] for r in c], "o-", color="#1f77b4", ms=4, label="CoNet valid interior")
    ax2.plot(qs, [r["valid_dead"] for r in q], "s:", color="#f0a0a0", ms=3, label="QuCoNet valid dead(≤.02)")
    ax2.plot(cs, [r["valid_dead"] for r in c], "o:", color="#7fb3e0", ms=3, label="CoNet valid dead(≤.02)")
    ax2.set_xlabel("training checkpoint  (quantum epoch ≈ classical step; B=3)")
    ax2.set_ylabel("held-out fraction")
    ax2.legend(fontsize=7.5); ax2.grid(alpha=0.3)
    fig.tight_layout()
    f = os.path.join(out, "F4_trajectory_seed2_B3.png"); fig.savefig(f, dpi=150); plt.close(fig)
    print(f"[F4] saved {f}")

# ───────────────────────── main ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("QSR_ROOT", os.path.abspath(os.path.join(HERE, ".."))))
    ap.add_argument("--parts", default="A,B")
    ap.add_argument("--Bs", default="16,64,128")
    ap.add_argument("--seeds", default="8")
    ap.add_argument("--out", default=os.path.join(HERE, "_step1_out"))
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--cache", action="store_true")
    args = ap.parse_args()

    Bs = [int(b) for b in args.Bs.split(",")]
    seeds = int(args.seeds) if args.seeds.isdigit() else [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    parts = args.parts.split(",")
    apd = amp._import_enumerators(args.root)

    if "A" in parts:
        cacheA = os.path.join(args.out, "heldout_p.json")
        if args.cache and os.path.exists(cacheA):
            rows = json.load(open(cacheA)); print(f"loaded {cacheA}: {len(rows)} rows")
        else:
            rows = collect_heldout(args.root, apd, Bs, seeds, args.M)
            json.dump(rows, open(cacheA, "w"))
        summary_heldout(rows, Bs)
        for B in Bs:
            fig_heldout_hist(rows, B, "valid", args.out)
        fig_heldout_hist(rows, max(Bs), "train", args.out)     # sanity vs original F1
        fig_interior_vs_B(rows, Bs, args.out)

    if "B" in parts:
        cacheB = os.path.join(args.out, "traj_seed2_B3.json")
        verify_seed2(apd, args.root)
        if args.cache and os.path.exists(cacheB):
            traj = json.load(open(cacheB)); print(f"loaded {cacheB}")
        else:
            traj = collect_trajectory(args.root, apd, args.M)
            json.dump(traj, open(cacheB, "w"))
        fig_trajectory(traj, args.out)

if __name__ == "__main__":
    main()
