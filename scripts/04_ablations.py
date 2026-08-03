#!/usr/bin/env python3
"""Ablations that test whether the study's design choices actually matter.

Three questions, each answered by re-running one architecture with one thing
changed:

``interface``
    Does the cPINN's leaky-wall condition matter, or would textbook head
    continuity have done?  Continuity cannot sustain a head jump, so it should
    fail on the barrier scenario and be harmless on the conduit one.

``prior``
    How much of the recovered conductivity is data and how much is the weak
    Tikhonov prior?  Sweeps the prior's bulk value well away from the truth; if
    the answer tracks the prior, the data are not constraining K.

``weighting``
    Is the ``1 + |beta W|`` residual normalisation load-bearing, or cosmetic?

Examples
--------
    python scripts/04_ablations.py --which interface
    python scripts/04_ablations.py --which prior --adam 6000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gwpinns.benchmark import load_benchmark
from gwpinns.evaluation.report import evaluate_model
from gwpinns.pinn import Scaling, TrainConfig, train
from gwpinns.pinn.base import LossWeights


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--which", nargs="+", default=["interface", "prior", "weighting"],
                        choices=["interface", "prior", "weighting"])
    parser.add_argument("--scenarios", nargs="+", default=["barrier", "conduit"],
                        choices=["barrier", "conduit"],
                        help="scenarios for the interface ablation")
    parser.add_argument("--datadir", default="data/benchmark")
    parser.add_argument("--outdir", default="runs/ablations")
    parser.add_argument("--adam", type=int, default=8000)
    parser.add_argument("--lbfgs", type=int, default=200)
    parser.add_argument("--collocation", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def run_case(args, scenario: str, name: str, **overrides) -> dict:
    data = load_benchmark(Path(args.datadir) / scenario)
    scaling = Scaling.from_observations(data.cfg, data.obs.head_obs)

    weights = LossWeights(**overrides.pop("weights", {}))
    train_cfg = TrainConfig(
        adam_iterations=args.adam,
        lbfgs_iterations=args.lbfgs,
        collocation_points=args.collocation,
        seed=args.seed,
        weights=weights,
        **overrides,
    )
    outdir = Path(args.outdir) / scenario / name
    print(f"\n=== ablation {name} / {scenario} ===")
    model, history = train(data, train_cfg, scaling=scaling, output_dir=outdir,
                           verbose=not args.quiet)

    metrics = evaluate_model(model, data)["metrics"]
    metrics["case"] = name
    metrics["wall_time_s"] = history.wall_time_s
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(
        f"  -> K_fault {metrics['k_fault_pred_m_per_d']:.4g} "
        f"(true {metrics['k_fault_true_m_per_d']:.4g}) | "
        f"bulk K {metrics['k_background_pred_m_per_d']:.3g} | "
        f"verdict {metrics['predicted_label']} "
        f"{'OK' if metrics['correct_classification'] else 'WRONG'}"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results: dict[str, list[dict]] = {}

    if "interface" in args.which:
        rows = []
        for scenario in args.scenarios:
            for mode in ("conductance", "continuity"):
                rows.append(
                    run_case(args, scenario, f"cpinn-{mode}",
                             architecture="cpinn", interface_mode=mode)
                )
        results["interface"] = rows

    if "prior" in args.which:
        rows = []
        for log_k in (-1.0, 0.0, 1.0):
            rows.append(
                run_case(args, "barrier", f"prior-logk{log_k:+.0f}",
                         architecture="mixed", log_k_init=log_k)
            )
        rows.append(
            run_case(args, "barrier", "prior-off",
                     architecture="mixed", weights={"k_prior": 0.0})
        )
        results["prior"] = rows

    if "weighting" in args.which:
        rows = []
        for mode in ("source", "none"):
            rows.append(
                run_case(args, "barrier", f"residual-{mode}",
                         architecture="mixed", residual_weighting=mode)
            )
        results["weighting"] = rows

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ablations.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nAblations written to {outdir / 'ablations.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
