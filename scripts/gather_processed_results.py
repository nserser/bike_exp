#!/usr/bin/env python3
"""Aggregate completed experiment artifacts into cross-run CSV and markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tomllib
from collections import defaultdict
from pathlib import Path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def fmt_float(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.{digits}g}"


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(row.get(col, "") for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def pick_metric(metrics: list[dict[str, str]], bound_type: str) -> dict[str, str]:
    matches = [row for row in metrics if row["bound_type"] == bound_type]
    if not matches:
        return {}
    matches.sort(
        key=lambda row: (
            int(row["bin_budget"]),
            -float(row["coverage"]),
            -float(row["spearman"]),
        )
    )
    return matches[-1]


def classify_run(run_id: str) -> str:
    if run_id.startswith("exact_toy_"):
        return "exact_toy_split"
    if run_id.startswith("exact_sampled_"):
        return "exact_sampled_split"
    return "other"


def summarize_standard_run(run_dir: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    config = read_config(run_dir / "configs" / "config.resolved.toml")
    ph_exact = read_csv_rows(run_dir / "csv" / "ph_exact.csv")
    metrics = read_csv_rows(run_dir / "csv" / "metrics.csv")

    run_id = config["run"]["run_id"]
    models = config["state"]["models"]
    model = ",".join(models)
    p_vals = [float(row["p_h"]) for row in ph_exact]
    total_failures = sum(int(row["failures"]) for row in ph_exact)
    errors_per_key = sorted({int(row["total_errors"]) for row in ph_exact})

    direct = pick_metric(metrics, "direct_bin")
    finite_state = pick_metric(metrics, "finite_state")

    summary = {
        "family": classify_run(run_id),
        "run_id": run_id,
        "artifact_dir": str(run_dir),
        "model": model,
        "r": str(config["parameters"]["r"]),
        "d": str(config["parameters"]["d"]),
        "t": str(config["parameters"]["t"]),
        "configured_key_count": str(config["sampling"]["key_count"]),
        "observed_keys": str(len(ph_exact)),
        "errors_per_key": ";".join(str(value) for value in errors_per_key),
        "p_h_min": fmt_float(min(p_vals)),
        "p_h_mean": fmt_float(safe_mean(p_vals)),
        "p_h_max": fmt_float(max(p_vals)),
        "total_failures": str(total_failures),
        "direct_budget": direct.get("bin_budget", ""),
        "direct_coverage": direct.get("coverage", ""),
        "direct_tail_l1_error": direct.get("tail_l1_error", ""),
        "direct_spearman": direct.get("spearman", ""),
        "direct_vacuous_bin_fraction": direct.get("vacuous_bin_fraction", ""),
        "direct_delta_bound": direct.get("delta_bound", ""),
        "finite_state_budget": finite_state.get("bin_budget", ""),
        "finite_state_coverage": finite_state.get("coverage", ""),
        "finite_state_tail_l1_error": finite_state.get("tail_l1_error", ""),
        "finite_state_spearman": finite_state.get("spearman", ""),
        "finite_state_vacuous_bin_fraction": finite_state.get("vacuous_bin_fraction", ""),
        "finite_state_delta_bound": finite_state.get("delta_bound", ""),
    }

    combined_metrics = []
    for row in metrics:
        merged = dict(row)
        merged["family"] = summary["family"]
        merged["artifact_dir"] = str(run_dir)
        combined_metrics.append(merged)

    return summary, combined_metrics


def summarize_rare_event_run(run_dir: Path) -> list[dict[str, str]]:
    config = read_config(run_dir / "configs" / "config.resolved.toml")
    rows = read_jsonl_rows(run_dir / "json" / "rare_event.jsonl")
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["run_id"]), int(row["u"]))].append(row)

    model = ",".join(config["state"]["models"])
    summary_rows: list[dict[str, str]] = []
    for (source_run_id, u_value), bucket in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        conditional_rates = [float(row["conditional_failure_rate"]) for row in bucket]
        contributions = [float(row["contribution_proxy"]) for row in bucket]
        tail_masses = [float(row["tail_mass_A_union"]) for row in bucket]
        summary_rows.append(
            {
                "family": "rare_event",
                "rare_artifact_dir": str(run_dir),
                "source_run_id": source_run_id,
                "model": model,
                "u": str(u_value),
                "keys": str(len(bucket)),
                "mean_conditional_failure_rate": fmt_float(safe_mean(conditional_rates)),
                "min_conditional_failure_rate": fmt_float(min(conditional_rates)),
                "max_conditional_failure_rate": fmt_float(max(conditional_rates)),
                "mean_contribution_proxy": fmt_float(safe_mean(contributions)),
                "max_contribution_proxy": fmt_float(max(contributions)),
                "mean_tail_mass_A_union": fmt_float(safe_mean(tail_masses)),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--out-dir", default="artifacts/processed_results")
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    out_dir = Path(args.out_dir)

    standard_patterns = [
        "exact_toy_r13_d3_t3_m*",
        "exact_sampled_r31_d5_t4_m*",
    ]
    rare_patterns = [
        "rare_event_exact_sampled_r31_d5_t4_m*",
    ]

    standard_dirs = sorted(
        {
            path
            for pattern in standard_patterns
            for path in artifacts_root.glob(pattern)
            if (path / "csv" / "metrics.csv").exists() and (path / "csv" / "ph_exact.csv").exists()
        }
    )
    rare_dirs = sorted(
        {
            path
            for pattern in rare_patterns
            for path in artifacts_root.glob(pattern)
            if (path / "json" / "rare_event.jsonl").exists()
        }
    )

    run_summaries: list[dict[str, str]] = []
    combined_metrics: list[dict[str, str]] = []
    for run_dir in standard_dirs:
        summary, metric_rows = summarize_standard_run(run_dir)
        run_summaries.append(summary)
        combined_metrics.extend(metric_rows)

    rare_summaries: list[dict[str, str]] = []
    for run_dir in rare_dirs:
        rare_summaries.extend(summarize_rare_event_run(run_dir))

    run_summary_columns = [
        "family",
        "run_id",
        "artifact_dir",
        "model",
        "r",
        "d",
        "t",
        "configured_key_count",
        "observed_keys",
        "errors_per_key",
        "p_h_min",
        "p_h_mean",
        "p_h_max",
        "total_failures",
        "direct_budget",
        "direct_coverage",
        "direct_tail_l1_error",
        "direct_spearman",
        "direct_vacuous_bin_fraction",
        "direct_delta_bound",
        "finite_state_budget",
        "finite_state_coverage",
        "finite_state_tail_l1_error",
        "finite_state_spearman",
        "finite_state_vacuous_bin_fraction",
        "finite_state_delta_bound",
    ]
    metric_columns = [
        "family",
        "artifact_dir",
        "run_id",
        "model",
        "bin_budget",
        "bound_type",
        "coverage",
        "median_looseness",
        "p95_looseness",
        "max_looseness",
        "vacuous_bin_fraction",
        "spearman",
        "tail_l1_error",
        "delta_bound",
        "delta_exact",
    ]
    rare_columns = [
        "family",
        "rare_artifact_dir",
        "source_run_id",
        "model",
        "u",
        "keys",
        "mean_conditional_failure_rate",
        "min_conditional_failure_rate",
        "max_conditional_failure_rate",
        "mean_contribution_proxy",
        "max_contribution_proxy",
        "mean_tail_mass_A_union",
    ]

    write_csv(out_dir / "run_summary.csv", run_summaries, run_summary_columns)
    write_csv(out_dir / "combined_metrics.csv", combined_metrics, metric_columns)
    write_csv(out_dir / "rare_event_summary.csv", rare_summaries, rare_columns)

    toy_rows = [row for row in run_summaries if row["family"] == "exact_toy_split"]
    sampled_rows = [row for row in run_summaries if row["family"] == "exact_sampled_split"]

    md_lines = []
    md_lines.append("# Processed Experiment Results")
    md_lines.append("")
    md_lines.append("This summary gathers the completed split-model experiment runs currently present under `artifacts/`.")
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("- The sampled `m0..m3` runs are low-memory split runs, not one unified all-model run.")
    md_lines.append("- `configured_key_count` differs across sampled split configs, so cross-model comparisons should be interpreted with that in mind.")
    md_lines.append("")
    if toy_rows:
        md_lines.append("## Exact Toy Split Summary")
        md_lines.append(
            markdown_table(
                toy_rows,
                [
                    "run_id",
                    "model",
                    "observed_keys",
                    "p_h_mean",
                    "direct_budget",
                    "direct_tail_l1_error",
                    "finite_state_budget",
                    "finite_state_tail_l1_error",
                ],
            )
        )
        md_lines.append("")
    if sampled_rows:
        md_lines.append("## Exact Sampled Split Summary")
        md_lines.append(
            markdown_table(
                sampled_rows,
                [
                    "run_id",
                    "model",
                    "configured_key_count",
                    "observed_keys",
                    "p_h_mean",
                    "direct_budget",
                    "direct_tail_l1_error",
                    "finite_state_budget",
                    "finite_state_tail_l1_error",
                ],
            )
        )
        md_lines.append("")
    if rare_summaries:
        md_lines.append("## Rare-Event Summary")
        md_lines.append(
            markdown_table(
                rare_summaries,
                [
                    "source_run_id",
                    "model",
                    "u",
                    "keys",
                    "mean_conditional_failure_rate",
                    "mean_contribution_proxy",
                    "mean_tail_mass_A_union",
                ],
            )
        )
        md_lines.append("")
    md_lines.append("## Generated Files")
    md_lines.append("- `run_summary.csv`")
    md_lines.append("- `combined_metrics.csv`")
    md_lines.append("- `rare_event_summary.csv`")

    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
