#!/usr/bin/env python3
"""Create markdown summary tables from artifact CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(row.get(col, "") for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run)
    metrics = read_rows(run_dir / "csv" / "metrics.csv")
    top_tail = read_rows(run_dir / "csv" / "top_tail_recall.csv")

    metrics_cols = [
        "model",
        "bin_budget",
        "bound_type",
        "coverage",
        "tail_l1_error",
        "delta_bound",
        "delta_exact",
    ]
    top_cols = [
        "model",
        "bin_budget",
        "bound_type",
        "tail_fraction",
        "precision",
        "recall",
    ]

    output = []
    output.append("## Coverage and Tail Metrics")
    output.append(markdown_table(metrics, metrics_cols))
    output.append("")
    output.append("## Top-Tail Recall")
    output.append(markdown_table(top_tail, top_cols))
    print("\n".join(output))


if __name__ == "__main__":
    main()
