#!/usr/bin/env python3
"""Run the full comparative study and write the report.

Trains every architecture on every scenario (see also 05_merge_study.py for
recombining per-scenario runs), scores each against the benchmark
truth, and emits figures, a JSON results file and a Markdown summary.

Examples
--------
    python scripts/03_run_study.py --quick                  # smoke run, minutes
    python scripts/03_run_study.py                          # full study
    python scripts/03_run_study.py --scenarios barrier \
        --architectures cpinn --adam 20000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gwpinns.benchmark import generate_benchmark, load_benchmark
from gwpinns.config import BenchmarkConfig
from gwpinns.evaluation import plots
from gwpinns.evaluation.report import evaluate_model
from gwpinns.pinn import Scaling, TrainConfig, train
from gwpinns.pinn.base import LossWeights

ARCHITECTURES = ("baseline", "mixed", "cpinn")
SCENARIOS = ("barrier", "conduit")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datadir", default="data/benchmark")
    parser.add_argument("--outdir", default="runs/study")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES), choices=list(ARCHITECTURES))
    parser.add_argument("--engine", choices=["auto", "mf6", "fd"], default="auto")
    parser.add_argument("--adam", type=int, default=12000)
    parser.add_argument("--lbfgs", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--ramp", type=int, default=2000)
    parser.add_argument("--collocation", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="tiny budget, for smoke testing")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def build_train_config(args: argparse.Namespace, architecture: str) -> TrainConfig:
    return TrainConfig(
        architecture=architecture,
        adam_iterations=args.adam,
        lbfgs_iterations=args.lbfgs,
        warmup_iterations=args.warmup,
        ramp_iterations=args.ramp,
        collocation_points=args.collocation,
        seed=args.seed,
        weights=LossWeights(),
    )


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            elif isinstance(value, bool):
                cells.append("yes" if value else "**no**")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.adam, args.lbfgs, args.warmup = 400, 0, 100
        args.ramp, args.collocation = 150, 600

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, dict]] = {}
    histories: dict[str, dict[str, dict]] = {}
    started = time.perf_counter()

    for scenario in args.scenarios:
        scenario_dir = datadir / scenario
        if not (scenario_dir / "manifest.json").exists():
            print(f"[{scenario}] benchmark missing -- generating it")
            generate_benchmark(
                BenchmarkConfig(scenario=scenario), datadir, engine=args.engine
            )
        data = load_benchmark(scenario_dir)
        scaling = Scaling.from_observations(data.cfg, data.obs.head_obs)

        results[scenario] = {}
        histories[scenario] = {}
        predictions: dict[str, np.ndarray] = {}
        head_predictions: dict[str, np.ndarray] = {}

        for architecture in args.architectures:
            print(f"\n=== {scenario} / {architecture} ===")
            run_dir = outdir / scenario / architecture
            train_cfg = build_train_config(args, architecture)

            model, history = train(
                data, train_cfg, scaling=scaling, output_dir=run_dir,
                verbose=not args.quiet,
            )
            evaluation = evaluate_model(model, data)
            metrics = evaluation["metrics"]
            metrics["wall_time_s"] = history.wall_time_s
            metrics["architecture"] = architecture

            np.savez_compressed(
                run_dir / "predictions.npz",
                k_pred=evaluation["k_pred"], h_pred=evaluation["h_pred"],
            )
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

            results[scenario][architecture] = metrics
            histories[scenario][architecture] = history.to_dict()
            predictions[architecture] = evaluation["k_pred"]
            head_predictions[architecture] = evaluation["h_pred"]

            print(
                f"  -> head RMSE {metrics['head_rmse_m']:.3f} m | "
                f"log10K RMSE {metrics['logk_rmse_background']:.3f} | "
                f"K_fault {metrics['k_fault_pred_m_per_d']:.4g} "
                f"(true {metrics['k_fault_true_m_per_d']:.4g}) | "
                f"verdict {metrics['predicted_label']} "
                f"{'OK' if metrics['correct_classification'] else 'WRONG'}"
            )

        figdir = outdir / scenario / "figures"
        plots.plot_conductivity_comparison(
            data.cfg, data.k_true, predictions, figdir / "conductivity_layer0.png", layer=0
        )
        plots.plot_fault_transect(
            data.cfg, data.k_true, predictions, figdir / "fault_transect.png", layer=0
        )
        plots.plot_training_history(histories[scenario], figdir / "training_history.png")
        plots.plot_head_timeseries(
            data.cfg, data.obs, data.times, data.heads, head_predictions,
            figdir / "head_timeseries.png",
        )
        plots.plot_benchmark_overview(
            data.cfg, data.k_true, data.heads, data.times, data.obs,
            figdir / "benchmark_overview.png",
        )

    plots.plot_fault_summary(results, outdir / "figures" / "fault_summary.png")
    (outdir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # --- Markdown report ------------------------------------------------------
    elapsed = time.perf_counter() - started
    lines = [
        "# Inverse PINN architecture comparison",
        "",
        f"Total wall time: {elapsed / 60:.1f} min. "
        f"Adam iterations: {args.adam}; collocation points: {args.collocation}.",
        "",
    ]
    columns = [
        ("architecture", "architecture"),
        ("head_rmse_m", "head RMSE (m)"),
        ("head_rmse_unobserved_m", "head RMSE unobs. (m)"),
        ("logk_rmse_background", "log₁₀K RMSE (bulk)"),
        ("k_background_pred_m_per_d", "K bulk (m/d)"),
        ("k_fault_pred_m_per_d", "K fault (m/d)"),
        ("contrast_log10_pred", "contrast log₁₀"),
        ("predicted_label", "verdict"),
        ("correct_classification", "correct"),
        ("wall_time_s", "time (s)"),
    ]
    for scenario, entries in results.items():
        first = next(iter(entries.values()))
        lines += [
            f"## Scenario: {scenario}",
            "",
            f"True K_fault = {first['k_fault_true_m_per_d']:g} m/d, "
            f"true bulk K = {first['k_background_true_m_per_d']:.3g} m/d, "
            f"true contrast = {first['contrast_log10_true']:.2f} log₁₀ units.",
            "",
            markdown_table(list(entries.values()), columns),
            "",
        ]
    (outdir / "REPORT.md").write_text("\n".join(lines))

    print(f"\nStudy complete in {elapsed / 60:.1f} min -> {outdir}")
    print((outdir / "REPORT.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
