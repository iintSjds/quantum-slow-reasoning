"""Analyze the controlled same-architecture classical capacity sweep.

For each run, the reported training accuracy is the objective actually optimized:

    mean_i [1 - (1 - p_i)**q] = -train_loss / B.

The script selects the checkpoint epoch recorded by the trainer, resolves the
capacity knee by interpolation in log(B), reports seed dispersion and threshold
sensitivity, and writes both a compact JSON cache and a vector PDF diagnostic.

Example (from the repository root):
    python scripts/classical_capacity_controlled.py \
        --results results --budgets 3 7 13
"""

import argparse
import glob
import json
import math
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401

    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False


RUN_RE = re.compile(
    r"qbkX(?P<q>\d+)_s(?P<seed>\d+)_B(?P<B>\d+)_"
    r".*B768_D6_seed\d+.*_results\.json$"
)


def load_runs(results_root: str, budgets: set[int]) -> dict[tuple[int, int, int], dict]:
    """Load the newest result for each (q, seed, B), ignoring older reruns."""
    indexed: dict[tuple[int, int, int], tuple[float, dict]] = {}
    pattern = os.path.join(results_root, "**", "*_results.json")
    for filename in glob.glob(pattern, recursive=True):
        match = RUN_RE.search(os.path.basename(filename))
        if not match:
            continue
        q = int(match.group("q"))
        if q not in budgets:
            continue
        key = (q, int(match.group("seed")), int(match.group("B")))
        mtime = os.path.getmtime(filename)
        if key in indexed and indexed[key][0] >= mtime:
            continue
        with open(filename, encoding="utf-8") as handle:
            data = json.load(handle)
        B = key[2]
        best_epoch = int(data["metrics"]["best_epoch"])
        history = data["history"]
        record = min(history, key=lambda item: abs(int(item["epoch"]) - best_epoch))
        indexed[key] = (
            mtime,
            {
                "q": q,
                "seed": key[1],
                "B": B,
                "epoch": int(record["epoch"]),
                "one_shot": float(record["train_sr"]),
                "post_readout": float(-record["train_loss"] / B),
                "elapsed_s": float(data["metrics"]["elapsed_s"]),
                "source": filename,
            },
        )
    return {key: value[1] for key, value in indexed.items()}


def knee(B_values: list[int], accuracies: list[float], threshold: float) -> float:
    """First downward threshold crossing, interpolated linearly in log(B)."""
    if not B_values:
        return math.nan
    if accuracies[0] < threshold:
        return float(B_values[0])
    for index in range(len(B_values) - 1):
        y_left, y_right = accuracies[index], accuracies[index + 1]
        if y_left >= threshold > y_right:
            fraction = (y_left - threshold) / (y_left - y_right)
            x_left, x_right = math.log(B_values[index]), math.log(B_values[index + 1])
            return math.exp(x_left + fraction * (x_right - x_left))
    return float(B_values[-1]) if accuracies[-1] >= threshold else math.nan


def summarize(runs: dict, budgets: list[int], threshold: float) -> dict:
    summary = {"threshold": threshold, "budgets": {}, "fit": {}}
    for q in budgets:
        seeds = sorted({seed for qq, seed, _ in runs if qq == q})
        B_values = sorted({B for qq, _, B in runs if qq == q})
        curves = {}
        seed_knees = []
        for seed in seeds:
            seed_B = [B for B in B_values if (q, seed, B) in runs]
            seed_y = [runs[(q, seed, B)]["post_readout"] for B in seed_B]
            curves[str(seed)] = {str(B): y for B, y in zip(seed_B, seed_y)}
            if len(seed_B) == len(B_values):
                seed_knees.append(knee(seed_B, seed_y, threshold))
        means = []
        stds = []
        for B in B_values:
            values = [
                runs[(q, seed, B)]["post_readout"]
                for seed in seeds
                if (q, seed, B) in runs
            ]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        summary["budgets"][str(q)] = {
            "B": B_values,
            "mean": means,
            "std": stds,
            "curves": curves,
            "B0_mean_curve": knee(B_values, means, threshold),
            "B0_per_seed": seed_knees,
            "B0_seed_mean": float(np.mean(seed_knees)) if seed_knees else math.nan,
            "B0_seed_std": (
                float(np.std(seed_knees, ddof=1)) if len(seed_knees) > 1 else 0.0
            ),
        }

    q_values = []
    B0_values = []
    for q in budgets:
        B0 = summary["budgets"][str(q)]["B0_mean_curve"]
        if math.isfinite(B0):
            q_values.append(q)
            B0_values.append(B0)
    if len(q_values) >= 2:
        exponent, intercept = np.polyfit(np.log(q_values), np.log(B0_values), 1)
        summary["fit"] = {
            "q": q_values,
            "B0": B0_values,
            "exponent": float(exponent),
            "prefactor": float(math.exp(intercept)),
        }
    return summary


def bootstrap(runs: dict, budgets: list[int], threshold: float, draws: int) -> dict:
    """Paired seed bootstrap for knee and scaling-exponent uncertainty."""
    common_seeds = sorted(
        set.intersection(
            *[
                {seed for qq, seed, _ in runs if qq == q}
                for q in budgets
            ]
        )
    )
    if len(common_seeds) < 2 or draws <= 0:
        return {}
    rng = np.random.default_rng(20260817)
    knee_draws = {q: [] for q in budgets}
    exponent_draws = []
    for _ in range(draws):
        sampled = rng.choice(common_seeds, size=len(common_seeds), replace=True)
        fitted_q = []
        fitted_B0 = []
        for q in budgets:
            B_values = sorted({B for qq, _, B in runs if qq == q})
            means = [
                float(np.mean([runs[(q, int(seed), B)]["post_readout"] for seed in sampled]))
                for B in B_values
            ]
            B0 = knee(B_values, means, threshold)
            knee_draws[q].append(B0)
            if math.isfinite(B0):
                fitted_q.append(q)
                fitted_B0.append(B0)
        if len(fitted_q) == len(budgets):
            exponent_draws.append(float(np.polyfit(np.log(fitted_q), np.log(fitted_B0), 1)[0]))

    def interval(values: list[float]) -> list[float]:
        return [float(value) for value in np.percentile(values, [2.5, 50.0, 97.5])]

    return {
        "draws": draws,
        "paired_seeds": common_seeds,
        "B0_95pct": {str(q): interval(knee_draws[q]) for q in budgets},
        "exponent_95pct": interval(exponent_draws),
    }


def plot_summary(summary: dict, output: str) -> None:
    plt.style.use(["science", "no-latex"] if HAS_SCIENCEPLOTS else ["default"])
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    markers = ["o", "s", "^", "D", "v", "P"]
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))
    threshold = summary["threshold"]

    for index, (q_text, values) in enumerate(summary["budgets"].items()):
        B = np.asarray(values["B"], dtype=float)
        mean = np.asarray(values["mean"])
        std = np.asarray(values["std"])
        color = colors[index % len(colors)]
        axes[0].plot(
            B,
            mean,
            marker=markers[index % len(markers)],
            color=color,
            linewidth=1.5,
            markersize=4,
            label=rf"best-of-${q_text}$",
        )
        axes[0].fill_between(B, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)

    axes[0].axhline(threshold, color="0.45", linestyle=":", linewidth=1)
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(0, 1.01)
    axes[0].set_xlabel(r"training questions $B$")
    axes[0].set_ylabel(r"best-of-$q$ training accuracy")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2, which="both")

    q_values = []
    B0_values = []
    B0_errors = []
    for q_text, values in summary["budgets"].items():
        q_values.append(float(q_text))
        B0_values.append(values["B0_mean_curve"])
        B0_errors.append(values["B0_seed_std"])
    q_values = np.asarray(q_values)
    B0_values = np.asarray(B0_values)
    B0_errors = np.asarray(B0_errors)
    axes[1].errorbar(
        q_values,
        B0_values,
        yerr=B0_errors,
        fmt="o",
        color="#0072B2",
        capsize=2.5,
        markersize=4.5,
    )
    fit = summary.get("fit", {})
    if fit:
        q_line = np.geomspace(q_values.min(), q_values.max(), 100)
        exponent = fit["exponent"]
        axes[1].plot(
            q_line,
            fit["prefactor"] * q_line**exponent,
            color="#0072B2",
            linewidth=1.5,
            label=rf"pilot fit: $B_0\propto q^{{{exponent:.2f}}}$",
        )
        axes[1].legend(frameon=False)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(q_values)
    axes[1].set_xticklabels([str(int(q)) for q in q_values])
    axes[1].set_xlabel(r"classical query budget $q$")
    axes[1].set_ylabel(r"capacity knee $B_0$")
    axes[1].grid(alpha=0.2, which="both")

    for label, axis in zip(("a", "b"), axes):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold", va="top")
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--budgets", type=int, nargs="+", default=[3, 7, 13])
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--json-out",
        default=os.path.join(HERE, "_sweep_out", "classical_capacity_controlled.json"),
    )
    parser.add_argument(
        "--figure-out",
        default=os.path.join(HERE, "_sweep_out", "classical_capacity_controlled.pdf"),
    )
    args = parser.parse_args()

    budgets = sorted(set(args.budgets))
    runs = load_runs(args.results, set(budgets))
    expected = len(budgets) * 4 * 14
    print(f"loaded {len(runs)} cells (pilot target: {expected})")
    summary = summarize(runs, budgets, args.threshold)
    summary["bootstrap"] = bootstrap(runs, budgets, args.threshold, args.bootstrap)
    summary["threshold_sensitivity"] = {}
    for threshold in (0.85, 0.90, 0.95):
        alternate = summarize(runs, budgets, threshold)
        summary["threshold_sensitivity"][str(threshold)] = {
            "B0": {
                str(q): alternate["budgets"][str(q)]["B0_mean_curve"]
                for q in budgets
            },
            "exponent": alternate.get("fit", {}).get("exponent", math.nan),
        }
    summary["metadata"] = {
        "results_root": os.path.abspath(args.results),
        "num_cells": len(runs),
        "budgets": budgets,
        "pool": "sliding_puzzle_N120_K3_M8_B768_D6, seeds 1--4",
        "capacity_definition": "first downward crossing of post-readout train accuracy",
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_summary(summary, args.figure_out)

    for q in budgets:
        values = summary["budgets"][str(q)]
        print(
            f"q={q:2d}: B0(mean curve)={values['B0_mean_curve']:.2f}; "
            f"per-seed mean={values['B0_seed_mean']:.2f} "
            f"+/- {values['B0_seed_std']:.2f}"
        )
    if summary.get("fit"):
        print(f"pilot exponent = {summary['fit']['exponent']:.3f}")
    if summary.get("bootstrap"):
        lo, median, hi = summary["bootstrap"]["exponent_95pct"]
        print(f"paired-seed bootstrap exponent = {median:.3f} [{lo:.3f}, {hi:.3f}]")
    print("threshold sensitivity:")
    for threshold, values in summary["threshold_sensitivity"].items():
        print(f"  threshold={threshold}: exponent={values['exponent']:.3f}")
    print(f"JSON -> {args.json_out}")
    print(f"PDF  -> {args.figure_out}")


if __name__ == "__main__":
    main()
