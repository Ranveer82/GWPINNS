#!/usr/bin/env python3
"""Phase 1 -- generate the synthetic fault benchmark.

Runs the forward groundwater model for both scenarios and writes a
self-contained dataset per scenario.

Examples
--------
    python scripts/01_generate_benchmark.py
    python scripts/01_generate_benchmark.py --engine mf6 --outdir data/benchmark
    python scripts/01_generate_benchmark.py --scenario barrier --no-figures

By default the script prefers a MODFLOW 6 binary on PATH and falls back to the
bundled finite-difference reference solver, which implements the same
discretisation.  Install MODFLOW with::

    python -m flopy.utils.get_modflow ~/.local/bin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gwpinns.config import SCENARIOS, BenchmarkConfig
from gwpinns.benchmark import find_mf6_executable, generate_benchmark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default="data/benchmark", help="output directory")
    parser.add_argument(
        "--scenario", choices=[*SCENARIOS, "both"], default="both",
        help="which fault behaviour to simulate",
    )
    parser.add_argument(
        "--engine", choices=["auto", "mf6", "fd"], default="auto",
        help="forward solver: MODFLOW 6, the reference FD solver, or auto",
    )
    parser.add_argument("--mf6-exe", default="mf6", help="name or path of the MODFLOW 6 binary")
    parser.add_argument("--noise-std", type=float, default=None, help="override observation noise (m)")
    parser.add_argument("--no-figures", action="store_true", help="skip the overview figures")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    scenarios = list(SCENARIOS) if args.scenario == "both" else [args.scenario]

    exe = find_mf6_executable(args.mf6_exe)
    print(f"MODFLOW 6 executable: {exe or 'not found -- using the reference FD solver'}")

    summary = {}
    for scenario in scenarios:
        cfg = BenchmarkConfig(scenario=scenario)
        if args.noise_std is not None:
            from dataclasses import replace
            cfg = replace(cfg, observations=replace(cfg.observations, noise_std=args.noise_std))

        data = generate_benchmark(
            cfg, outdir, engine=args.engine, exe_name=args.mf6_exe, verbose=True
        )
        summary[scenario] = json.loads((outdir / scenario / "manifest.json").read_text())

        if not args.no_figures:
            from gwpinns.evaluation.plots import plot_benchmark_overview

            figure = plot_benchmark_overview(
                cfg, data.k_true, data.heads, data.times, data.obs,
                outdir / scenario / "figures" / "benchmark_overview.png",
            )
            print(f"  figure -> {figure}")

    print("\nSummary")
    for scenario, manifest in summary.items():
        sig = manifest["fault_signature"]
        print(
            f"  {scenario:8s} solver={manifest['solver']:4s} "
            f"obs={manifest['n_observation_values']:5d} "
            f"({100 * manifest['data_coverage_fraction']:.2f}% coverage)  "
            f"head jump steady {sig['head_jump_steady_m']:+.3f} m / "
            f"final {sig['head_jump_final_m']:+.3f} m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
