#!/usr/bin/env python3
"""Phase 2 -- train one inverse PINN architecture on one benchmark scenario.

Examples
--------
    python scripts/02_train.py --scenario barrier --architecture cpinn
    python scripts/02_train.py --scenario conduit --architecture mixed --adam 20000
    python scripts/02_train.py --scenario barrier --architecture cpinn \
        --interface-mode continuity --tag continuity-ablation

The benchmark must already exist -- run ``scripts/01_generate_benchmark.py``
first.  Training writes the model weights, the loss history, the metric report
and the recovered fields to ``runs/<scenario>/<architecture><tag>/``.
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
    parser.add_argument("--scenario", choices=["barrier", "conduit"], required=True)
    parser.add_argument("--architecture", choices=["baseline", "mixed", "cpinn"], required=True)
    parser.add_argument("--datadir", default="data/benchmark")
    parser.add_argument("--outdir", default="runs")
    parser.add_argument("--tag", default="", help="suffix for the run directory")

    parser.add_argument("--adam", type=int, default=12000, help="Adam iterations")
    parser.add_argument("--lbfgs", type=int, default=250, help="L-BFGS polish iterations")
    parser.add_argument("--warmup", type=int, default=600, help="data-only warm-up iterations")
    parser.add_argument("--ramp", type=int, default=2000, help="physics curriculum ramp length")
    parser.add_argument("--collocation", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")

    parser.add_argument("--data-weight", type=float, default=10.0)
    parser.add_argument("--k-prior-weight", type=float, default=0.25)
    parser.add_argument("--log-k-init", type=float, default=0.0,
                        help="prior/initial bulk log10 K in m/d")
    parser.add_argument("--no-adaptive-weights", action="store_true")
    parser.add_argument("--interface-mode", choices=["conductance", "continuity"],
                        default="conductance", help="cPINN only")
    parser.add_argument("--gamma-mode", choices=["field", "scalar"], default="field",
                        help="cPINN only: fault leakance as a plane field or one scalar")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data = load_benchmark(Path(args.datadir) / args.scenario)
    scaling = Scaling.from_observations(data.cfg, data.obs.head_obs)

    weights = LossWeights(data=args.data_weight, k_prior=args.k_prior_weight)
    train_cfg = TrainConfig(
        architecture=args.architecture,
        adam_iterations=args.adam,
        lbfgs_iterations=args.lbfgs,
        warmup_iterations=args.warmup,
        ramp_iterations=args.ramp,
        collocation_points=args.collocation,
        learning_rate=args.lr,
        seed=args.seed,
        width=args.width,
        depth=args.depth,
        dtype=args.dtype,
        weights=weights,
        adaptive_weights=not args.no_adaptive_weights,
        interface_mode=args.interface_mode,
        gamma_mode=args.gamma_mode,
        log_k_init=args.log_k_init,
    )

    name = f"{args.architecture}{args.tag}"
    outdir = Path(args.outdir) / args.scenario / name
    print(f"Training {name} on the {args.scenario} scenario "
          f"({len(data.obs)} observations, solver={data.solver})")

    model, history = train(
        data, train_cfg, scaling=scaling, output_dir=outdir, verbose=not args.quiet
    )

    result = evaluate_model(model, data)
    metrics = result["metrics"]
    np.savez_compressed(
        outdir / "predictions.npz", k_pred=result["k_pred"], h_pred=result["h_pred"]
    )
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print(f"\nResults for {name} / {args.scenario}  ({history.wall_time_s:.0f} s)")
    print(f"  head RMSE              {metrics['head_rmse_m']:.3f} m "
          f"(unobserved cells {metrics['head_rmse_unobserved_m']:.3f} m)")
    print(f"  log10 K RMSE (bulk)    {metrics['logk_rmse_background']:.3f}")
    print(f"  K background           {metrics['k_background_pred_m_per_d']:.3g} m/d "
          f"(true {metrics['k_background_true_m_per_d']:.3g})")
    print(f"  K fault                {metrics['k_fault_pred_m_per_d']:.4g} m/d "
          f"(true {metrics['k_fault_true_m_per_d']:.4g})")
    print(f"  fault verdict          {metrics['predicted_label']} "
          f"(true {metrics['true_scenario']}) -> "
          f"{'CORRECT' if metrics['correct_classification'] else 'WRONG'}")
    print(f"  written to             {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
