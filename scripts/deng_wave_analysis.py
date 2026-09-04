#!/usr/bin/env python3
"""P1-5b / P1-7b analysis of the Deng-response experiment wave.

Reads the harvested runs under from4090/deng_wave/results/ (36 seedaudit
runs: {grover n=1, grover n=4, best-of-4} x pools 1-4 x torch seeds
{7,123,2026}; 16 relabel runs: grover n=1 on pools 1-8 x 2 node
permutations, torch seed 42), computes per-question (p, IPR) on the train
and held-out splits with the standard enumerator, and caches them to
_sweep_out/deng_wave_cache.json (keys "tag|split" -> [[p, ipr], ...]).

Report mode (--report) prints
  A. P1-5b: pool x torch-seed matrices (archived seed-42 column from
     p_cache.json), variance decomposition (pool sd vs optimizer-seed sd),
     and per-run boundary fractions (p_i > 0.98);
  B. P1-7b: relabeled-pool accuracies against the native seed-42 runs.

Run from the repo root; compute is resumable.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("QSR_REPO_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
os.environ["QSR_REPO_ROOT"] = ROOT
sys.path.insert(0, os.path.join(ROOT, "quconet"))
sys.path.insert(0, HERE)

import amplification_scaling as amp            # noqa: E402

OUT = os.path.join(HERE, "_sweep_out")
CACHE = os.path.join(OUT, "deng_wave_cache.json")
RESULTS = os.path.join(ROOT, "from4090", "deng_wave", "results")
M = 8


def G_le(p, q):
    p = np.clip(np.asarray(p, float), 0, 1)
    th = np.arcsin(np.sqrt(p))
    rs = np.where(th > 0, np.round(np.pi / (4 * th) - 0.5), 0)
    r = np.minimum(np.maximum(rs, 0), (q - 1) // 2)
    return np.sin((2 * r + 1) * th) ** 2


def G_blind(p, q):
    p = np.clip(np.asarray(p, float), 0, 1)
    return np.sin(q * np.arcsin(np.sqrt(p))) ** 2


def bok(p, k):
    return 1.0 - (1.0 - np.clip(np.asarray(p, float), 0, 1)) ** k


def run_tag(d):
    """Parse a harvested run directory name into a tag."""
    base = os.path.basename(d.rstrip("/"))
    rj = glob.glob(os.path.join(d, "*results.json"))
    if not rj:
        return None, None
    cfg = json.load(open(rj[0]))["config"]
    import re
    m = re.search(r"_seed(\d+)(?:_perm(\d+))?_s(\d+)_", base)
    pool, perm, ts = m.group(1), m.group(2), m.group(3)
    if cfg["loss_type"] == "grover":
        fam = f"g{cfg['grover_n']}"
    else:
        fam = f"qbk{cfg['best_k']}"
    tag = f"{fam}|{pool}|{ts}" if perm is None else f"relabel_{fam}|{pool}p{perm}|{ts}"
    return tag, rj[0]


def compute():
    apd = amp._import_enumerators(ROOT)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    dirs = sorted(glob.glob(os.path.join(RESULTS, "*")))
    for d in dirs:
        tag, rj = run_tag(d)
        if tag is None or f"{tag}|train" in cache:
            continue
        meta = json.load(open(rj))
        best = glob.glob(os.path.join(d, "*best_model.pt"))[0]
        cm, snm, scm, N, K, _ = apd.load_quantum_checkpoint(best)
        for split, qa in (("train", meta["train_qa"]), ("valid", meta["valid_qa"])):
            rr = [apd.compute_quantum_path_diversity(cm, snm, scm, Q, A, N, K, M)
                  for (Q, A) in qa]
            cache[f"{tag}|{split}"] = [[float(r[3]), float(r[0])] for r in rr]
        json.dump(cache, open(CACHE, "w"))
        pm = np.mean([x[0] for x in cache[f"{tag}|train"]])
        print(f"[{tag}] cached (train p mean {pm:.3f})", flush=True)
    print(f"compute done: {len(cache)} entries")


def ps(cache, tag, split):
    return np.asarray([x[0] for x in cache[f"{tag}|{split}"]], float)


def report():
    cache = json.load(open(CACHE))
    pc = json.load(open(os.path.join(OUT, "p_cache.json")))

    # ── A. P1-5b seed cross-audit ────────────────────────────────────────
    fams = {
        "g1":   ("grover",  lambda p: float(np.mean(G_le(p, 3))),
                 lambda p: float(np.mean(G_blind(p, 3)))),
        "g4":   ("grover4", lambda p: float(np.mean(G_le(p, 9))),
                 lambda p: float(np.mean(G_blind(p, 9)))),
        "qbk4": ("qbkX4",   lambda p: float(np.mean(bok(p, 3))),
                 lambda p: float(np.mean(bok(p, 9)))),
    }
    TS = ["42", "7", "123", "2026"]
    print("=" * 72)
    print("A. P1-5b pool x optimizer-seed cross-audit "
          "(train metric / held-out metric)")
    for fam, (arch, f_tr, f_va) in fams.items():
        print(f"\n--- {fam}  (train: G<=q or bo-3; held-out: G_q or bo-9) ---")
        tr = np.full((4, 4), np.nan)
        va = np.full((4, 4), np.nan)
        bfrac = np.full((4, 4), np.nan)
        for i, pool in enumerate("1234"):
            for j, ts in enumerate(TS):
                if ts == "42":
                    ptr = np.asarray(pc[f"{arch}|{pool}|32|train"], float)
                    pva = np.asarray(pc[f"{arch}|{pool}|32|valid"], float)
                else:
                    key = f"{fam}|{pool}|{ts}"
                    if f"{key}|train" not in cache:
                        continue
                    ptr, pva = ps(cache, key, "train"), ps(cache, key, "valid")
                tr[i, j] = f_tr(ptr)
                va[i, j] = f_va(pva)
                bfrac[i, j] = float(np.mean(ptr > 0.98))
        for name, mat in (("train", tr), ("held-out", va)):
            print(f"  {name}: rows=pools 1-4, cols=torch seeds {TS}")
            for i in range(4):
                print("    " + " ".join(f"{mat[i, j]:.3f}" for j in range(4)))
            pool_means = np.nanmean(mat, axis=1)
            sd_pool = float(np.nanstd(pool_means, ddof=1))
            sd_seed = float(np.sqrt(np.nanmean(np.nanvar(mat, axis=1, ddof=1))))
            print(f"    sd_pool {sd_pool:.4f}   sd_seed(within pool) {sd_seed:.4f}")
        print(f"  boundary fraction (train p_i > 0.98), same layout:")
        for i in range(4):
            print("    " + " ".join(f"{bfrac[i, j]:.3f}" for j in range(4)))

    # ── B. P1-7b relabeling invariance ───────────────────────────────────
    print("\n" + "=" * 72)
    print("B. P1-7b node-relabeling invariance (grover n=1, torch seed 42)")
    print("  pool | native tr / va | perm1 tr / va | perm2 tr / va")
    dtr, dva = [], []
    for pool in "12345678":
        nat_tr = float(np.mean(G_le(np.asarray(pc[f"grover|{pool}|32|train"]), 3)))
        nat_va = float(np.mean(G_blind(np.asarray(pc[f"grover|{pool}|32|valid"]), 3)))
        row = f"  {pool:>4} | {nat_tr:.3f} / {nat_va:.3f}"
        for perm in "12":
            key = f"relabel_g1|{pool}p{perm}|42"
            t = float(np.mean(G_le(ps(cache, key, "train"), 3)))
            v = float(np.mean(G_blind(ps(cache, key, "valid"), 3)))
            row += f" | {t:.3f} / {v:.3f}"
            dtr.append(t - nat_tr)
            dva.append(v - nat_va)
        print(row)
    for name, dd in (("train", dtr), ("held-out", dva)):
        dd = np.asarray(dd)
        print(f"  relabeled - native {name}: mean {dd.mean():+.4f}  "
              f"sd {dd.std(ddof=1):.4f}  range [{dd.min():+.4f}, {dd.max():+.4f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else compute()
