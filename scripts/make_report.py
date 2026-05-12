#!/usr/bin/env python3
"""Generate report.md and plots from an artifact directory."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_script(script: Path, run_dir: Path) -> None:
    subprocess.run(["python3", str(script), "--run", str(run_dir)], check=True)


def summarize_ph(rows: list[dict[str, str]]) -> dict[str, str]:
    p_vals = [float(row["p_h"]) for row in rows]
    failures = [int(row["failures"]) for row in rows]
    totals = [int(row["total_errors"]) for row in rows]
    return {
        "keys": str(len(rows)),
        "p_min": f"{min(p_vals):.6g}" if p_vals else "0",
        "p_max": f"{max(p_vals):.6g}" if p_vals else "0",
        "avg_p": f"{sum(p_vals) / len(p_vals):.6g}" if p_vals else "0",
        "errors_per_key": str(totals[0]) if totals else "0",
        "total_failures": str(sum(failures)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run)
    script_dir = Path(__file__).resolve().parent
    run_script(script_dir / "plot_tail_curves.py", run_dir)
    run_script(script_dir / "plot_scatter_scores.py", run_dir)

    metrics = read_rows(run_dir / "csv" / "metrics.csv")
    ph_exact = read_rows(run_dir / "csv" / "ph_exact.csv")
    fs_bounds = read_rows(run_dir / "csv" / "fs_bounds.csv")
    manifest = read_text(run_dir / "manifest.json")
    config = read_text(run_dir / "configs" / "config.resolved.toml")
    summary_tables = subprocess.run(
        ["python3", str(script_dir / "make_summary_tables.py"), "--run", str(run_dir)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    ph_summary = summarize_ph(ph_exact)
    finite_state_rows = [row for row in metrics if row["bound_type"] == "finite_state"]
    direct_rows = [row for row in metrics if row["bound_type"] == "direct_bin"]
    vacuous_fs = sum(1 for row in fs_bounds if row["vacuous_flag"] == "true")

    lines = []
    lines.append("# Experiment Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- Keys processed: {ph_summary['keys']}")
    lines.append(f"- Errors per key: {ph_summary['errors_per_key']}")
    lines.append(f"- Observed p_H range: {ph_summary['p_min']} to {ph_summary['p_max']}")
    lines.append(f"- Average DFR: {ph_summary['avg_p']}")
    lines.append(f"- Total failures across enumerated errors: {ph_summary['total_failures']}")
    lines.append(f"- Finite-state vacuous bins: {vacuous_fs}")
    lines.append("")
    lines.append("## Claim Boundary")
    lines.append("This artifact certifies reduced exact domains only. It does not imply any production-parameter BIKE DFR claim.")
    lines.append("")
    lines.append(summary_tables)
    lines.append("")
    lines.append("## Finite-State Notes")
    lines.append(f"- Direct metric rows: {len(direct_rows)}")
    lines.append(f"- Finite-state metric rows: {len(finite_state_rows)}")
    lines.append("- Transition envelopes now include conditioned observed successor sets per (m,l) pair.")
    lines.append("")
    lines.append("## Plots")
    lines.append("- `plots/tail_curves.pdf`")
    lines.append("- `plots/scatter_scores.pdf`")
    lines.append("- `plots/looseness_boxplot.pdf`")
    lines.append("")
    lines.append("## Resolved Config")
    lines.append("```toml")
    lines.append(config.strip())
    lines.append("```")
    lines.append("")
    lines.append("## Manifest")
    lines.append("```json")
    lines.append(manifest.strip())
    lines.append("```")

    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
