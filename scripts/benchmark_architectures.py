#!/usr/bin/env python3
"""Compare backbone architectures on identical data, seeds and budget.

Every architecture is trained on the same collocation schedule with the same
loss terms, so the only thing that varies is the field representation. Scores
are the held-out well RMSE and, where a reference solution is available, the
cell-by-cell error of the head and transmissivity fields.
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
from gwpinn.postproc.predict import predict_grid
from gwpinn.postproc.report import build_report, load_truth
from gwpinn.train.trainer import Trainer

ARCHS = ("mlp", "resnet", "modified_mlp", "cnn")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("-o", "--out", default="runs/benchmark")
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--archs", nargs="*", default=list(ARCHS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg0 = load_config(args.config)
    ds = build_dataset(cfg0, verbose=True)
    truth = load_truth(pathlib.Path(args.config).parent / "truth", ds)

    results = []
    for arch in args.archs:
        for seed in range(args.seeds):
            cfg = load_config(args.config)
            cfg.model.arch = arch
            cfg.train.adam_iters = args.iters
            cfg.train.lbfgs_iters = 0
            cfg.train.seed = seed
            cfg.train.log_every = max(args.iters // 4, 1)

            print("\n" + "=" * 78)
            print(f"architecture = {arch}   seed = {seed}")
            print("=" * 78)
            t0 = time.time()
            tr = Trainer(ds, cfg, seed=seed)
            hist = tr.fit()
            elapsed = time.time() - t0

            pred = predict_grid([tr.model], ds, downsample=2, verbose=False)
            rep = build_report([tr.model], ds, pred, [hist], truth)

            row = {
                "arch": arch,
                "seed": seed,
                "seconds": elapsed,
                "params": int(sum(p.numel() for p in tr.model.parameters()
                                  if p.requires_grad)),
                "val_rmse_m": rep["head"]["validation"]["rmse"],
                "train_rmse_m": rep["head"]["train"]["rmse"],
                "val_r2": rep["head"]["validation"]["r2"],
                "logT_rmse": rep["properties"].get("log10T", {}).get("rmse", float("nan")),
                "final_pde_loss": hist[-1]["loss_pde"],
            }
            for k, v in rep.get("grid", {}).items():
                row[f"grid_{k}_rmse"] = v["rmse"]
                row[f"grid_{k}_r2"] = v["r2"]
            results.append(row)
            print(f"  -> val RMSE {row['val_rmse_m']:.3f} m, {elapsed:.0f}s")

            with open(out / "benchmark.json", "w") as fh:
                json.dump(results, fh, indent=2, default=float)

    print("\n" + "=" * 96)
    print("ARCHITECTURE COMPARISON")
    print("=" * 96)
    cols = ["arch", "params", "seconds", "val_rmse_m", "grid_head_L0_rmse",
            "grid_log10T_L0_rmse", "grid_log10T_L0_r2"]
    header = f"{'arch':<14s}{'params':>9s}{'sec':>7s}{'valRMSE':>10s}" \
             f"{'headRMSE':>10s}{'log10T':>9s}{'log10T R2':>11s}"
    print(header)
    print("-" * 96)
    for r in results:
        print(
            f"{r['arch']:<14s}{r['params']:>9d}{r['seconds']:>7.0f}"
            f"{r['val_rmse_m']:>10.3f}"
            f"{r.get('grid_head_L0_rmse', float('nan')):>10.3f}"
            f"{r.get('grid_log10T_L0_rmse', float('nan')):>9.3f}"
            f"{r.get('grid_log10T_L0_r2', float('nan')):>11.3f}"
        )
    print("=" * 96)
    print(f"\nwritten to {out / 'benchmark.json'}")


if __name__ == "__main__":
    main()
