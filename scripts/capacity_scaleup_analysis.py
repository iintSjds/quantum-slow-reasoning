#!/usr/bin/env python3
"""Analyze the 16-pool matched QuCoNet/CoNet capacity scale-up.

The four archived QuCoNet pools are read from ``p_cache_bigB.json``.  New
QuCoNet runs are read directly from their best-checkpoint epoch: for Grover-n
training, ``-train_loss / B`` is exactly the mean post-read-out accuracy
``mean_i A_n(p_i)``.  Controlled CoNet runs are loaded through
``classical_capacity_controlled.load_runs``.

Uncertainty is evaluated at the independent question-pool level.  Curves store
sample SD and SEM across pools.  Knee and exponent intervals use a paired seed
bootstrap: one resampled seed list is shared by every q and both methods.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import re
from pathlib import Path

import numpy as np


QUANTUM_RE = re.compile(
    r"grover_n(?P<n>\d+)_s(?P<seed>\d+)_B(?P<B>\d+)_.*_results\.json$"
)
CLASSICAL_RE = re.compile(
    r"qbkX(?P<q>\d+)_s(?P<seed>\d+)_B(?P<B>\d+)_.*_results\.json$"
)
Q_VALUES = (3, 5, 7, 9, 11, 13)
BASE_B = (8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 704)
QUANTUM_B = {
    3: BASE_B,
    5: BASE_B,
    7: BASE_B,
    9: BASE_B,
    11: BASE_B + (768, 896, 1024),
    13: BASE_B + (768, 896, 1024, 1216),
}
CLASSICAL_B = (1, 2, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 96, 128)


def grover_accuracy(probability: float, rounds: int) -> float:
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return 1.0
    theta = math.asin(math.sqrt(probability))
    optimum = round(math.pi / (4 * theta) - 0.5)
    used = min(rounds, max(0, optimum))
    return math.sin((2 * used + 1) * theta) ** 2


def knee(B_values: list[int] | tuple[int, ...], values, threshold: float) -> float:
    """First downward crossing after the curve has entered the plateau.

    A small-B optimization miss can put the first sampled point just below the
    threshold even though the following points form the high-accuracy plateau.
    Such a point is not a capacity knee, so only a high-to-low crossing counts.
    """
    x = np.log(np.asarray(B_values, dtype=float))
    y = np.asarray(values, dtype=float)
    for index in range(len(x) - 1):
        if y[index] >= threshold > y[index + 1]:
            fraction = (y[index] - threshold) / (y[index] - y[index + 1])
            return float(math.exp(x[index] + fraction * (x[index + 1] - x[index])))
    return math.nan


def load_archived_quantum(cache_file: str, seeds: set[int]) -> dict:
    cache = json.load(open(cache_file, encoding="utf-8"))
    runs = {}
    for key, values in cache.items():
        rounds, seed, B = map(int, key.split("|"))
        q = 2 * rounds + 1
        if seed not in seeds or q not in Q_VALUES:
            continue
        runs[(q, seed, B)] = float(
            np.mean([grover_accuracy(p, rounds) for p in values["tr"]])
        )
    return runs


def pick_record(data: dict, record_rule: str) -> dict:
    """History row whose ``-train_loss/B`` the curves use.

    ``final`` is the converged state.  ``best`` reproduces the archived
    selection (row nearest ``metrics.best_epoch``); the sweep driver selected
    that epoch by max raw SR, which for the interior-target grover objective
    archives the early overshoot transient at low B/N (the low-B dip).
    """
    if record_rule == "best":
        best_epoch = int(data["metrics"]["best_epoch"])
        return min(data["history"], key=lambda row: abs(int(row["epoch"]) - best_epoch))
    rows = [row for row in data["history"] if row.get("train_loss") is not None]
    if not rows:
        raise ValueError("history has no rows with a loss")
    return rows[-1]


def load_new_quantum(results_root: str, seeds: set[int], record_rule: str) -> dict:
    indexed = {}
    pattern = os.path.join(results_root, "**", "*_results.json")
    for filename in glob.glob(pattern, recursive=True):
        match = QUANTUM_RE.search(os.path.basename(filename))
        if not match:
            continue
        rounds = int(match.group("n"))
        seed = int(match.group("seed"))
        B = int(match.group("B"))
        q = 2 * rounds + 1
        if seed not in seeds or q not in Q_VALUES or B not in QUANTUM_B[q]:
            continue
        key = (q, seed, B)
        mtime = os.path.getmtime(filename)
        if key in indexed and indexed[key][0] >= mtime:
            continue
        data = json.load(open(filename, encoding="utf-8"))
        record = pick_record(data, record_rule)
        if record["train_loss"] is None:
            raise ValueError(f"selected epoch has no loss in {filename}")
        indexed[key] = (mtime, float(-record["train_loss"] / B), filename)
    return {key: item[1] for key, item in indexed.items()}


def load_classical(results_root: str, seeds: set[int], record_rule: str) -> dict:
    indexed = {}
    pattern = os.path.join(results_root, "**", "*_results.json")
    for filename in glob.glob(pattern, recursive=True):
        match = CLASSICAL_RE.search(os.path.basename(filename))
        if not match:
            continue
        q = int(match.group("q"))
        seed = int(match.group("seed"))
        B = int(match.group("B"))
        if q not in Q_VALUES or seed not in seeds or B not in CLASSICAL_B:
            continue
        key = (q, seed, B)
        mtime = os.path.getmtime(filename)
        if key in indexed and indexed[key][0] >= mtime:
            continue
        data = json.load(open(filename, encoding="utf-8"))
        record = pick_record(data, record_rule)
        indexed[key] = (mtime, float(-record["train_loss"] / B), filename)
    return {key: item[1] for key, item in indexed.items()}


def validate_cells(name: str, runs: dict, B_grid: dict, seeds: tuple[int, ...]) -> None:
    missing = [
        (q, seed, B)
        for q in Q_VALUES
        for seed in seeds
        for B in B_grid[q]
        if (q, seed, B) not in runs
    ]
    if missing:
        preview = ", ".join(map(str, missing[:12]))
        raise SystemExit(f"{name}: {len(missing)} cells missing; first: {preview}")


def summarize_method(runs: dict, B_grid: dict, seeds: tuple[int, ...], threshold: float):
    output = {}
    arrays = {}
    for q in Q_VALUES:
        B = list(B_grid[q])
        values = np.asarray([[runs[(q, seed, b)] for b in B] for seed in seeds])
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        seed_knees = [knee(B, row, threshold) for row in values]
        output[str(q)] = {
            "B": B,
            "mean": mean.tolist(),
            "sd": sd.tolist(),
            "sem": (sd / math.sqrt(len(seeds))).tolist(),
            "per_seed": {
                str(seed): row.tolist() for seed, row in zip(seeds, values)
            },
            "B0_mean_curve": knee(B, mean, threshold),
            "B0_per_seed": seed_knees,
        }
        arrays[q] = values
    return output, arrays


def fit_exponent(B0_values) -> dict:
    x = np.log(np.asarray(Q_VALUES, dtype=float))
    y = np.log(np.asarray(B0_values, dtype=float))
    exponent, intercept = np.polyfit(x, y, 1)
    residual = y - (exponent * x + intercept)
    slope_se = math.sqrt(np.sum(residual**2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2))
    r2 = 1.0 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)
    return {
        "exponent": float(exponent),
        "prefactor": float(math.exp(intercept)),
        "ols_slope_se": float(slope_se),
        "r2": float(r2),
    }


def percentiles(values) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no finite bootstrap values")
    p = np.percentile(finite, [2.5, 16.0, 50.0, 84.0, 97.5])
    return {
        "num_finite": int(finite.size),
        "95_low": float(p[0]),
        "68_low": float(p[1]),
        "median": float(p[2]),
        "68_high": float(p[3]),
        "95_high": float(p[4]),
    }


def paired_bootstrap(quantum_arrays, classical_arrays, threshold, draws, seed=20260817):
    rng = np.random.default_rng(seed)
    num_seeds = next(iter(quantum_arrays.values())).shape[0]
    quantum_knees = {q: [] for q in Q_VALUES}
    classical_knees = {q: [] for q in Q_VALUES}
    quantum_curves = {q: [] for q in Q_VALUES}
    classical_curves = {q: [] for q in Q_VALUES}
    quantum_exponents = []
    classical_exponents = []
    differences = []
    for _ in range(draws):
        selected = rng.integers(0, num_seeds, size=num_seeds)
        q_B0 = []
        c_B0 = []
        for q in Q_VALUES:
            q_curve = quantum_arrays[q][selected].mean(axis=0)
            c_curve = classical_arrays[q][selected].mean(axis=0)
            quantum_curves[q].append(q_curve)
            classical_curves[q].append(c_curve)
            qk = knee(QUANTUM_B[q], q_curve, threshold)
            ck = knee(CLASSICAL_B, c_curve, threshold)
            q_B0.append(qk)
            c_B0.append(ck)
            quantum_knees[q].append(qk)
            classical_knees[q].append(ck)
        if np.all(np.isfinite(q_B0)) and np.all(np.isfinite(c_B0)):
            qe = fit_exponent(q_B0)["exponent"]
            ce = fit_exponent(c_B0)["exponent"]
            quantum_exponents.append(qe)
            classical_exponents.append(ce)
            differences.append(qe - ce)

    def curve_bands(draws_by_q):
        output = {}
        for q in Q_VALUES:
            values = np.asarray(draws_by_q[q])
            low, high = np.percentile(values, [16.0, 84.0], axis=0)
            output[str(q)] = {"68_low": low.tolist(), "68_high": high.tolist()}
        return output

    return {
        "draws_requested": draws,
        "draws_with_resolved_knees": len(differences),
        "quantum_B0": {str(q): percentiles(quantum_knees[q]) for q in Q_VALUES},
        "classical_B0": {str(q): percentiles(classical_knees[q]) for q in Q_VALUES},
        "quantum_curve_mean": curve_bands(quantum_curves),
        "classical_curve_mean": curve_bands(classical_curves),
        "quantum_exponent": percentiles(quantum_exponents),
        "classical_exponent": percentiles(classical_exponents),
        "exponent_difference": percentiles(differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantum-results", action="append", required=True,
                        help="run-record root(s); repeatable")
    parser.add_argument("--classical-results", action="append", required=True,
                        help="run-record root(s); repeatable")
    parser.add_argument("--legacy-quantum-cache", default=None,
                        help="per-question cache of archived BEST checkpoints "
                             "(seeds 1-4).  Under --record final, pass those "
                             "seeds' record roots via --quantum-results "
                             "instead: the cache only holds best-epoch states.")
    parser.add_argument("--record", choices=("final", "best"), default="final",
                        help="history row for each cell: converged final epoch "
                             "(default), or the archived max-raw-SR epoch that "
                             "produced the published low-B dip")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument(
        "--out",
        default=os.path.join(HERE, "_sweep_out", "capacity_scaleup_16.json"),
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    seed_set = set(seeds)

    quantum = {}
    if args.legacy_quantum_cache:
        quantum.update(load_archived_quantum(args.legacy_quantum_cache, seed_set))
    for root in args.quantum_results:
        quantum.update(load_new_quantum(root, seed_set, args.record))
    classical = {}
    for root in args.classical_results:
        classical.update(load_classical(root, seed_set, args.record))
    quantum_grid = {q: QUANTUM_B[q] for q in Q_VALUES}
    classical_grid = {q: CLASSICAL_B for q in Q_VALUES}
    validate_cells("QuCoNet", quantum, quantum_grid, seeds)
    validate_cells("CoNet", classical, classical_grid, seeds)

    q_summary, q_arrays = summarize_method(quantum, quantum_grid, seeds, args.threshold)
    c_summary, c_arrays = summarize_method(classical, classical_grid, seeds, args.threshold)
    q_fit = fit_exponent([q_summary[str(q)]["B0_mean_curve"] for q in Q_VALUES])
    c_fit = fit_exponent([c_summary[str(q)]["B0_mean_curve"] for q in Q_VALUES])
    bootstrap = paired_bootstrap(q_arrays, c_arrays, args.threshold, args.bootstrap)

    output = {
        "metadata": {
            "threshold": args.threshold,
            "record_rule": args.record,
            "legacy_cache_used": bool(args.legacy_quantum_cache),
            "seeds": list(seeds),
            "num_paired_pools": len(seeds),
            "uncertainty_unit": "question-pool seed",
            "curve_uncertainty": "sample SD and SEM plus paired-bootstrap 68% intervals",
            "intervals": "paired question-pool bootstrap percentiles",
        },
        "quantum": {"curves": q_summary, "fit": q_fit},
        "classical": {"curves": c_summary, "fit": c_fit},
        "bootstrap": bootstrap,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    print(f"QuCoNet exponent: {q_fit['exponent']:.3f} {bootstrap['quantum_exponent']}")
    print(f"CoNet exponent: {c_fit['exponent']:.3f} {bootstrap['classical_exponent']}")
    print(f"difference: {bootstrap['exponent_difference']}")


if __name__ == "__main__":
    main()
