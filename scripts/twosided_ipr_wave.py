#!/usr/bin/env python3
"""Path-diversity comparison: IPR of the two-sided imported-target COIN control.

Retrains the tscoin family (L = sum_i (p_i - c)^2 on the coin architecture,
protocol identical to twosided_coin_control.py) because the original runs
saved no checkpoints, then measures the per-question success-path IPR of the
trained walkers with the same enumerator used for ipr_cache.json.  Output:
_sweep_out/ipr_twosided_coin.json with keys "tag|seed|32|split" ->
[[p, ipr], ...], plus a printed accuracy-weighted comparison against the
native Grover-trained IPR ladder.

Run from repo root (set QSR_REPO_ROOT if the repo is elsewhere):
    python docs/discussion/scripts/twosided_ipr_wave.py
"""
import os, sys, json, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("QSR_REPO_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
os.environ["QSR_REPO_ROOT"] = ROOT
sys.path.insert(0, os.path.join(ROOT, "quconet"))
sys.path.insert(0, HERE)

import twosided_coin_control as tcc            # noqa: E402
import amplification_scaling as amp            # noqa: E402

OUT = os.path.join(ROOT, "docs", "discussion", "scripts", "_sweep_out")
CACHE = os.path.join(OUT, "ipr_twosided_coin.json")


def main():
    apd = amp._import_enumerators(ROOT)
    tcc.validate()                                             # replica check
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    ipr_native = json.load(open(os.path.join(OUT, "ipr_cache.json")))

    for tag, c in tcc.CAPS.items():
        for seed in tcc.SEEDS:
            if f"{tag}|{seed}|{tcc.B}|train" in cache:
                continue
            shift_map, tq, vq, _ = tcc.seed_assets(seed)
            coin = tcc.train(shift_map, tq, c)

            # wrap the trained coin in an archived-format checkpoint so the
            # standard quantum enumerator loads it unchanged
            import glob as _g
            tpl = _g.glob(f"{tcc.SWEEP}/quconet_adam_sliding_puzzle_N120*_seed{seed}_s42_"
                          f"N120_K3_M8_B{tcc.B}_*/grover_n1_s{seed}_B{tcc.B}_*best_model.pt")[0]
            ck = torch.load(tpl, map_location="cpu", weights_only=False)
            ck["model_state_dict"]["coin.hamiltonian_params"] = \
                coin.hamiltonian_params.detach().clone()
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                torch.save(ck, f.name)
                tmp = f.name
            try:
                cm, snm, scm, N, K, cfg = apd.load_quantum_checkpoint(tmp)
                for split, qa in (("train", tq), ("valid", vq)):
                    rr = [apd.compute_quantum_path_diversity(cm, snm, scm, Q, A, N, K, tcc.M)
                          for (Q, A) in qa]
                    cache[f"{tag}|{seed}|{tcc.B}|{split}"] = \
                        [[float(r[3]), float(r[0])] for r in rr]
            finally:
                os.unlink(tmp)
            json.dump(cache, open(CACHE, "w"))
            print(f"[{tag} seed {seed}] cached "
                  f"(train p mean {np.mean([x[0] for x in cache[f'{tag}|{seed}|{tcc.B}|train']]):.3f})",
                  flush=True)

    def wipr(rows):
        a = np.asarray(rows, float)
        return float((a[:, 0] * a[:, 1]).sum() / a[:, 0].sum())

    natives = {1: "grover", 2: "grover2", 3: "grover3", 4: "grover4"}
    tags = {1: "tscoin25", 2: "tscoin095", 3: "tscoin0495", 4: "tscoin0302"}
    print("\naccuracy-weighted IPR (pooled seeds 1-8, train split):")
    print(f"{'n':>3} {'native Grover':>14} {'two-sided coin':>15}")
    for n in (1, 2, 3, 4):
        nat = []
        for s in tcc.SEEDS:
            nat += ipr_native.get(f"{natives[n]}|{s}|{tcc.B}|train", [])
        two = []
        for s in tcc.SEEDS:
            two += cache.get(f"{tags[n]}|{s}|{tcc.B}|train", [])
        print(f"{n:>3} {wipr(nat):>14.3f} {wipr(two):>15.3f}")


if __name__ == "__main__":
    main()
