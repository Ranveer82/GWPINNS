#!/usr/bin/env python3
"""Train the PINN, export rasters, and write the accuracy report and plots.

    python scripts/train.py sample_data/config.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gwpinn.config import load_config
from gwpinn.dataset import build_dataset
from gwpinn.postproc.predict import export_rasters, predict_grid
from gwpinn.postproc.report import build_report, format_report, load_truth, save_report
from gwpinn.train.trainer import train_ensemble


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("-o", "--workdir", default=None, help="override paths.workdir")
    ap.add_argument("--arch", default=None, help="override model.arch")
    ap.add_argument("--fault-coords", dest="fault_coords", action="store_true",
                    default=None,
                    help="map coordinates through tanh(signed distance) per fault")
    ap.add_argument("--no-fault-coords", dest="fault_coords", action="store_false",
                    help="append the fault indicators instead of mapping coordinates")
    ap.add_argument("--iters", type=int, default=None, help="override train.adam_iters")
    ap.add_argument("--lbfgs", type=int, default=None, help="override train.lbfgs_iters")
    ap.add_argument("--ensemble", type=int, default=None, help="override train.n_ensemble")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--truth", default=None, help="directory of reference rasters")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))

    cfg = load_config(args.config)
    if args.workdir:
        cfg.paths.workdir = args.workdir
    if args.arch:
        cfg.model.arch = args.arch
    if args.fault_coords is not None:
        cfg.model.fault_coords = args.fault_coords
    if args.iters is not None:
        cfg.train.adam_iters = args.iters
    if args.lbfgs is not None:
        cfg.train.lbfgs_iters = args.lbfgs
    if args.ensemble is not None:
        cfg.train.n_ensemble = args.ensemble
    if args.seed is not None:
        cfg.train.seed = args.seed

    workdir = pathlib.Path(cfg.paths.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg.save(workdir / "config_used.yaml")

    dtype = torch.float64 if cfg.train.dtype == "float64" else torch.float32

    print("=" * 78)
    print("1/5  Reading inputs")
    print("=" * 78)
    t0 = time.time()
    ds = build_dataset(cfg, device=cfg.train.device, dtype=dtype)

    print()
    print("=" * 78)
    print(f"2/5  Training ({cfg.model.arch}, {cfg.train.adam_iters} Adam "
          f"+ {cfg.train.lbfgs_iters} L-BFGS iterations)")
    print("=" * 78)
    models, histories = train_ensemble(ds, cfg)
    train_time = time.time() - t0
    with open(workdir / "history.json", "w") as fh:
        json.dump(histories, fh, indent=1)

    print()
    print("=" * 78)
    print("3/5  Predicting on the output grid")
    print("=" * 78)
    pred = predict_grid(models, ds, downsample=cfg.domain.plot_downsample)
    if cfg.output.write_rasters:
        export_rasters(
            pred, ds, workdir / "rasters",
            nodata=cfg.output.nodata, dtype=cfg.output.raster_dtype,
        )

    print()
    print("=" * 78)
    print("4/5  Accuracy assessment")
    print("=" * 78)
    truth_dir = args.truth or (pathlib.Path(args.config).parent / "truth")
    truth = load_truth(truth_dir, ds)
    if truth:
        print(f"  scoring against reference rasters in {truth_dir}")
    report = build_report(models, ds, pred, histories, truth)
    report["config"]["train_seconds"] = train_time
    save_report(report, workdir / "report.json")
    print()
    print(format_report(report))

    print()
    print("=" * 78)
    print("5/5  Plots")
    print("=" * 78)
    if args.no_plots or not cfg.output.write_plots:
        print("  skipped")
    else:
        from gwpinn.postproc.plots import make_all_plots

        paths = make_all_plots(models, ds, pred, histories, report, truth,
                               workdir / "plots")
        print(f"  wrote {len(paths)} figures to {workdir / 'plots'}")

    torch.save(
        {"state_dicts": [m.state_dict() for m in models], "config": cfg.to_dict()},
        workdir / "model.pt",
    )
    print(f"\nAll outputs in {workdir}   (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
