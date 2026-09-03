#!/usr/bin/env python3
"""P1-2 cost-model sensitivity: matched classical attempts under variable
verification pricing.

With a coherent reflection priced at lambda_v circuit-application units and a
classical verification at lambda_c, the budget that funds r Grover rounds
(q = 2r+1 applications) funds k = [(2r+1) + lambda_v*r] / (1 + lambda_c)
classical attempts.  The adverse weighting (lambda_v=1, lambda_c=0) gives
k = 3r+1 = 13 at r = 4.  This script recomputes the best-of-k envelope of the
27 terminal-outcome controls at k = 13 from the shared per-question cache and
reports the revised held-out ratios quoted in the Supplemental Material.

Run from the repository root.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = json.load(open(os.path.join(HERE, "_sweep_out", "p_cache.json")))

CONTROLS = [
    "cadam", "cbentH01", "cbentH03", "cbentH10", "cbestk", "cbestkX16",
    "cbestkX32", "cbestkX4", "cbestkX64", "cbestkX8", "cbestk_exact",
    "ccap25", "ccap50", "ccap75", "centH01", "centH03", "centH10",
    "cstd", "cstd_adam", "cstd_exact",
    "qbkX16", "qbkX2", "qbkX32", "qbkX4", "qbkX64", "qbkX8", "qstd",
]
QUANTUM_Q9 = {"QuCoNet": 0.407, "neural quantum AI": 0.496}


def best_envelope(k, split="valid", B="32"):
    best = None
    for fam in CONTROLS:
        seed_means = [
            float(np.mean(1.0 - (1.0 - np.asarray(v, float)) ** k))
            for key, v in CACHE.items()
            for f, seed, b, s in [key.split("|")]
            if f == fam and b == B and s == split
        ]
        if len(seed_means) >= 8:
            m = float(np.mean(seed_means))
            if best is None or m > best[1]:
                best = (fam, m, float(np.std(seed_means, ddof=1)), len(seed_means))
    return best


def main():
    for k in (3, 5, 7, 9, 13):
        fam, m, sd, n = best_envelope(k)
        print(f"k={k:>2}: best {fam:>12}  held-out {m:.4f}  sd {sd:.3f}  ({n} seeds)")
    b13 = best_envelope(13)[1]
    for name, acc in QUANTUM_Q9.items():
        print(f"lambda_v=1, lambda_c=0 (k=13 vs q=9): {name} ratio "
              f"{acc:.3f}/{b13:.4f} = {acc / b13:.2f}x")


if __name__ == "__main__":
    main()
