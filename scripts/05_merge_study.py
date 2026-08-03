#!/usr/bin/env python3
"""Merge per-scenario study outputs into one report and summary figure.

``scripts/03_run_study.py`` can be run once per scenario -- which is what you
want on a small machine, since the scenarios are independent and running them
as two processes uses the cores better than one process running them in series.
This recombines the pieces.

Example
-------
    python scripts/03_run_study.py --scenarios barrier --outdir runs/study_barrier &
    python scripts/03_run_study.py --scenarios conduit --outdir runs/study_conduit &
    wait
    python scripts/05_merge_study.py runs/study_barrier runs/study_conduit \
        --outdir runs/study
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gwpinns.benchmark import load_benchmark
from gwpinns.benchmark.generate import head_jump
from gwpinns.evaluation import plots

COLUMNS = [
    ("architecture", "architecture"),
    ("head_rmse_m", "head RMSE (m)"),
    ("head_rmse_unobserved_m", "head RMSE unobs. (m)"),
    ("logk_rmse_background", "log₁₀K RMSE (bulk)"),
    ("k_background_pred_m_per_d", "K bulk (m/d)"),
    ("k_fault_pred_m_per_d", "K fault (m/d)"),
    ("contrast_log10_pred", "contrast log₁₀"),
    ("head_jump_pred_m", "head jump (m)"),
    ("predicted_label", "verdict"),
    ("correct_classification", "correct"),
    ("wall_time_s", "time (s)"),
]


def markdown_table(rows: list[dict], columns=COLUMNS) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, bool):
                cells.append("yes" if value else "**no**")
            elif isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="per-scenario study directories")
    parser.add_argument("--outdir", default="runs/study")
    parser.add_argument("--datadir", default="data/benchmark",
                        help="benchmark root, used to backfill missing metrics")
    return parser.parse_args(argv)


def backfill_head_jump(source: Path, scenario: str, entries: dict, datadir: Path) -> None:
    """Recompute the fault head jump from saved predictions where it is missing.

    Runs started before the metric existed still have their predicted head grids
    on disk, so the diagnostic can be recovered without retraining.
    """
    if all(e.get("head_jump_pred_m") is not None for e in entries.values()):
        return
    scenario_dir = datadir / scenario
    if not (scenario_dir / "manifest.json").exists():
        return
    data = load_benchmark(scenario_dir)
    truth = head_jump(data.cfg, data.heads[-1])

    for architecture, metrics in entries.items():
        predictions = source / scenario / architecture / "predictions.npz"
        if metrics.get("head_jump_pred_m") is not None or not predictions.exists():
            continue
        with np.load(predictions) as payload:
            metrics["head_jump_pred_m"] = head_jump(data.cfg, payload["h_pred"][-1])
        metrics["head_jump_true_m"] = truth


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, dict] = {}
    for source in args.inputs:
        source = Path(source)
        payload = json.loads((source / "results.json").read_text())
        for scenario, entries in payload.items():
            backfill_head_jump(source, scenario, entries, Path(args.datadir))
        merged.update(payload)
        for scenario in payload:
            figures = source / scenario / "figures"
            if figures.is_dir():
                target = outdir / scenario / "figures"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(figures, target, dirs_exist_ok=True)

    # Keep a stable scenario order regardless of how the runs finished.
    ordered = {k: merged[k] for k in ("barrier", "conduit") if k in merged}
    ordered.update({k: v for k, v in merged.items() if k not in ordered})

    (outdir / "results.json").write_text(json.dumps(ordered, indent=2, default=str))
    plots.plot_fault_summary(ordered, outdir / "figures" / "fault_summary.png")

    lines = ["# Inverse PINN architecture comparison", ""]
    for scenario, entries in ordered.items():
        first = next(iter(entries.values()))
        lines += [
            f"## Scenario: {scenario}",
            "",
            f"True K_fault = {first['k_fault_true_m_per_d']:g} m/d, "
            f"true bulk K = {first['k_background_true_m_per_d']:.3g} m/d, "
            f"true contrast = {first['contrast_log10_true']:.2f} log₁₀ units.",
            "",
            markdown_table(list(entries.values())),
            "",
        ]
    (outdir / "REPORT.md").write_text("\n".join(lines))

    print((outdir / "REPORT.md").read_text())
    print(f"merged -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
