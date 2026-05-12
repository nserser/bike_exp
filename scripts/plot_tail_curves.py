#!/usr/bin/env python3
"""Generate tail curve plot from artifact CSVs."""

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
    rows = read_rows(run_dir / "csv" / "tail_curves.csv")
    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["bin_budget"], row["bound_type"])
        grouped[key].append((float(row["tau"]), float(row["tail_value"])))

    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, budget, bound_type), points in sorted(grouped.items()):
        points.sort()
        xs = [tau if tau > 0 else 1e-12 for tau, _ in points]
        ys = [value for _, value in points]
        label = f"{model} {bound_type} B={budget}"
        linestyle = "-" if bound_type == "exact" else "--" if bound_type == "direct_bin" else ":"
        ax.plot(xs, ys, label=label, linewidth=1.8, linestyle=linestyle)

    ax.set_xscale("log")
    ax.set_xlabel("tau")
    ax.set_ylabel("Pr[p_H > tau]")
    ax.set_title("Tail Curves")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "tail_curves.pdf")


if __name__ == "__main__":
    main()
