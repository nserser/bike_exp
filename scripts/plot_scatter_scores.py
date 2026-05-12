#!/usr/bin/env python3
"""Generate key-score scatter and looseness plots."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run)
    rows = read_rows(run_dir / "csv" / "key_scores.csv")
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["bound_type"])
        grouped[key].append((float(row["score"]), float(row["exact_p_h"])))

    models = sorted({model for model, _ in grouped})
    fig, axes = plt.subplots(len(models), 1, figsize=(8, 3.2 * max(len(models), 1)), squeeze=False)
    for idx, model in enumerate(models):
        ax = axes[idx][0]
        for bound_type in ["direct_bin", "finite_state"]:
            points = grouped.get((model, bound_type), [])
            if not points:
                continue
            xs = [score for score, _ in points]
            ys = [value if value > 0 else 1e-12 for _, value in points]
            ax.scatter(xs, ys, s=12, alpha=0.6, label=bound_type)
        ax.set_yscale("log")
        ax.set_xlabel("Model score")
        ax.set_ylabel("exact p_H")
        ax.set_title(model)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "scatter_scores.pdf")

    looseness = defaultdict(list)
    for row in rows:
        exact = float(row["exact_p_h"])
        if exact > 0:
            looseness[(row["model"], row["bound_type"])].append(float(row["score"]) / exact)

    labels = []
    values = []
    for key in sorted(looseness):
        labels.append(f"{key[0]} {key[1]}")
        values.append(looseness[key])

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    if values:
        ax2.boxplot(values, labels=labels, vert=True)
    ax2.set_ylabel("Looseness ratio")
    ax2.set_title("Looseness by Model")
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    ax2.grid(True, axis="y", alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(run_dir / "plots" / "looseness_boxplot.pdf")


if __name__ == "__main__":
    main()
