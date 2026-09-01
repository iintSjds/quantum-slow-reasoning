#!/usr/bin/env python3
"""PRL SM figure for the parameter-sharing neural coin (fig:sm-neural).

Built from the merged 32-seed transformer-coin runs
(from4090/tcoin_merged/results/, exact enumeration) so every number matches
Table~\\ref{tab:neural}.  Three panels:
  (a) attractor vs collapse: trained per-question p at B=32 -- the
      amplification objective parks the 29k-parameter transformer at the
      ridges p*(3)=0.250 and p*(5)=0.095 (measured means 0.249 and 0.108),
      while the same network under the one-shot objective collapses to {0,1};
  (b) capacity recast in model size: <G_{<=3}> vs training load B at fixed 29k
      parameters -- the plateau runs past B=256 (6.4x the 720-parameter
      table's knee B0=40), and held-out accuracy rises with B;
  (c) held-out ladder at matched answer-circuit budget q: the shared transformer
      beats the per-node table and the strongest same-architecture classical
      control, already at q=3.

Run: python scripts/plot_tcoin_prl_sm.py
"""
import os, re, json, glob, math
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DATAROOT = os.environ.get("QSR_ROOT", REPO)
ROOT = os.path.join(DATAROOT, "from4090", "tcoin_merged", "results")
OUT = os.path.join(REPO, "figs", "Q5_neural_coin.pdf")
# Per-node angle-table walker, held-out fixed-budget accuracy at B=32.
TAB_VALID = {1: 0.042, 2: 0.147, 3: 0.272, 4: 0.407}


def pstar(n):
    return math.sin(math.pi / (2 * (2 * n + 1))) ** 2


def gacc(p, n):
    """At-most-n rule (the training objective's convention; used for train-side accuracy)."""
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    th = math.asin(math.sqrt(p)); k = min(n, max(0, round(math.pi / (4 * th) - 0.5)))
    return math.sin((2 * k + 1) * th) ** 2


def bacc(p, n):
    """Blind schedule: always exactly n rounds, never consults p (held-out convention)."""
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    return math.sin((2 * n + 1) * math.asin(math.sqrt(p))) ** 2


def load():
    recs = []
    for f in glob.glob(os.path.join(ROOT, "*_history.json")):
        if os.path.basename(f).startswith("._"):
            continue
        d = json.load(open(f))
        m = re.match(r"tcoin_(g1|g2|g3|g4|std|bk2|cap25)_s(\d+)_B(\d+)$", d["args"]["label"])
        if not m:
            continue
        fin = d["history"][-1]
        if fin["epoch"] == 0:
            continue
        recs.append(dict(fam=m.group(1), s=int(m.group(2)), B=int(m.group(3)),
                         tr=np.asarray(fin["train"]["p"]),
                         va=np.asarray(fin["valid"]["p"]) if fin["valid"] else None))
    return recs


def sel(R, fam, B):
    return [r for r in R if r["fam"] == fam and r["B"] == B]


R = load()
print(f"loaded {len(R)} merged runs")

fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.7))

# ── (a) attractor vs collapse ──────────────────────────────────────────
a = ax[0]
for fam, col, lab in (("g1", "#d62728", r"Grover-based, $q=3$"),
                      ("g2", "#7f0000", r"Grover-based, $q=5$"),
                      ("std", "0.35", "single-attempt")):
    ps = np.concatenate([r["tr"] for r in sel(R, fam, 32)])
    a.hist(ps, bins=np.linspace(0, 1, 26), density=True, histtype="step",
           lw=1.9, color=col, label=lab)
for n in (1, 2):
    q = 2 * n + 1
    a.axvline(pstar(n), ls="--", color="k", lw=0.9, alpha=0.7)
    a.annotate(rf"$p^*({q})$", (pstar(n), a.get_ylim()[1]), ha="center",
               fontsize=8.5, annotation_clip=False)
a.set_yscale("log"); a.set_xlim(0, 1)
a.set_xlabel(r"trained $p$  ($B=32$, $32$ seeds)"); a.set_ylabel("density (log)")
a.legend(fontsize=8, loc="upper center")
a.set_title("(a) Training distribution", fontsize=10)

# ── (b) capacity recast in model size ──────────────────────────────────
b = ax[1]
Bs = [32, 64, 96, 128, 192, 256]
trA = [np.mean([np.mean([gacc(p, 1) for p in r["tr"]]) for r in sel(R, "g1", B)]) for B in Bs]
vaA = [np.mean([np.mean([bacc(p, 1) for p in r["va"]]) for r in sel(R, "g1", B)]) for B in Bs]
b.plot(Bs, trA, "o-", color="#4b0082", lw=2, ms=5, label=r"training $\langle G_{\leq 3}\rangle$")
b.axvline(40, ls=":", color="0.5", lw=1.3)
b.text(41, 0.905, r"$720$-parameter table: $B_0=40$", fontsize=7.5, color="0.35", rotation=90, va="bottom")
b.axhline(0.9, ls=":", color="0.7", lw=1)
b.set_xscale("log", base=2); b.set_xticks(Bs); b.set_xticklabels(Bs)
b.set_ylim(0.0, 1.03); b.set_xlabel(r"training-set size $B$ (fixed $29$k parameters)")
b.set_ylabel(r"training $\langle G_{\leq 3}\rangle$", color="#4b0082")
b.tick_params(axis="y", labelcolor="#4b0082")
b2 = b.twinx()
b2.plot(Bs, vaA, "s--", color="#d62728", lw=1.8, ms=5, label=r"held-out $\langle G_3\rangle$")
b2.set_ylim(0.0, 0.16); b2.set_ylabel(r"held-out $\langle G_3\rangle$", color="#d62728")
b2.tick_params(axis="y", labelcolor="#d62728")
b.set_title("(b) Dependence on training-set size", fontsize=10)

# ── (c) held-out ladder at matched budget ──────────────────────────────
c = ax[2]
ns = [1, 2, 3, 4]
qs = [2 * n + 1 for n in ns]
tva = [np.mean([np.mean([bacc(p, n) for p in r["va"]]) for r in sel(R, f"g{n}", 32)]) for n in ns]
ctrl = []
for n in ns:
    best = 0.0
    for fam in ("cap25", "bk2"):
        va = np.concatenate([r["va"] for r in sel(R, fam, 32)])
        best = max(best, float(np.mean(1 - (1 - va) ** (2 * n + 1))))
    ctrl.append(best)
c.plot(qs, tva, "o-", color="#d62728", lw=2.2, ms=7, label="neural quantum AI, Grover-trained")
c.plot(qs, [TAB_VALID[n] for n in ns], "s--", mfc="none", color="#1f77b4",
       lw=1.8, ms=7, label="QuCoNet, Grover-trained")
c.plot(qs, ctrl, "^:", color="0.4", lw=1.8, ms=7,
       label="strongest same-arch. control,\nbest-of-$q$")
for x, y, cy in zip(qs, tva, ctrl):
    # Ratio of the unrounded values, so the annotation matches tab:neural
    # and the SM text (0.496/0.0604 -> 8.2x).
    ratio = y / cy
    c.annotate(rf"${ratio:.1f}\times$", (x, y), textcoords="offset points", xytext=(7, 4),
               fontsize=8.5, color="#d62728")
c.set_xticks(qs); c.set_xlim(2.6, 9.4); c.set_ylim(0, 0.58)
c.set_xlabel(r"reasoning-circuit applications $q$")
c.set_ylabel("held-out accuracy")
c.legend(fontsize=7.8, loc="upper left")
c.set_title("(c) Held-out conversion", fontsize=10)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}")
print("transformer held-out ladder:", [round(v, 3) for v in tva])
print("strongest control best-of-q:", [round(v, 3) for v in ctrl])
print("capacity trainA1:", [round(v, 3) for v in trA], "| held-out:", [round(v, 3) for v in vaA])
