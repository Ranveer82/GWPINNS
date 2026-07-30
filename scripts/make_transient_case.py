#!/usr/bin/env python3
"""Build a transient version of the synthetic case.

Storage coefficients do not appear in the steady-state flow equation, so no
amount of steady head data can identify them. This script drives the same
aquifer with a falling river stage and writes time-stamped observations, which
makes ``S`` recoverable from the flow field itself.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gwpinn.config import load_config
from gwpinn.data.fdsolver import fault_face_multipliers, solve_steady, solve_transient
from gwpinn.data.synthetic import CRS, make_synthetic_case
from gwpinn.io.raster import read_raster, write_raster


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--outdir", default="sample_data_transient")
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--ny", type=int, default=96)
    ap.add_argument("--cellsize", type=float, default=80.0)
    ap.add_argument("--n-times", type=int, default=6)
    ap.add_argument("--duration", type=float, default=180.0, help="days")
    ap.add_argument("--drawdown", type=float, default=2.5, help="river stage fall (m)")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    print("Building the steady base case ...")
    case = make_synthetic_case(
        outdir, nx=args.nx, ny=args.ny, cellsize=args.cellsize,
        n_wells=70, n_gauges=6, n_pumping_tests=30, seed=args.seed,
    )
    rng = np.random.default_rng(args.seed + 1)

    xs, ys = case.xs, case.ys
    xx, yy = np.meshgrid(xs, ys)
    active = case.active

    # Re-derive the river forcing from what was written out.
    river_poly, _ = _read_polygon(outdir / "river_polygon.shp")
    in_river = shapely.contains_xy(river_poly, xx, yy) & active
    river_cond = np.where(in_river, 5.0e-2, 0.0)

    stage0 = np.zeros_like(xx)
    head0 = case.head
    stage0[in_river] = head0[0][in_river]

    faults = gpd.read_file(outdir / "faults.shp")
    lines = [np.asarray(g.coords) for g in faults.geometry]
    mx, my = fault_face_multipliers(lines, case.fault_alpha, xs, ys)

    # A linear drawdown of the river, which propagates into the aquifer at a
    # rate set by the diffusivity T/S - that is what makes S observable.
    times = list(np.linspace(0.0, args.duration, args.n_times))
    stage_series = [stage0 - args.drawdown * (t / args.duration) for t in times]

    print(f"Solving {len(times)} transient steps ...")
    series = solve_transient(
        K=case.K, S=case.S, top=case.top, bottoms=case.bottoms,
        cellsize=case.cellsize, active=active, times=times, h_init=case.head,
        leakance=[1.0e-3],
        recharge_series=[np.full(xx.shape, 1.2e-4)] * len(times),
        river_cond=river_cond, river_stage_series=stage_series,
        unconfined_top=True, min_thickness=2.0, verbose=True,
    )

    truth = outdir / "truth"
    for i, t in enumerate(times):
        for l in range(2):
            write_raster(truth / f"head_t{i}_L{l}.tif", series[i, l],
                         case.transform, CRS)
    # The report scores against ``head_L*.tif``; point those at the final state.
    for l in range(2):
        write_raster(truth / f"head_L{l}.tif", series[-1, l], case.transform, CRS)

    # ---- time-stamped observations --------------------------------------
    wells = gpd.read_file(outdir / "head_obs.shp")
    wx = wells.geometry.x.to_numpy()
    wy = wells.geometry.y.to_numpy()
    wl = wells["layer"].to_numpy().astype(int)

    rows = []
    for i, t in enumerate(times):
        for l in (0, 1):
            m = wl == l
            if not m.any():
                continue
            v = _bilinear(series[i, l], xs, ys, wx[m], wy[m])
            for xi, yi, vi in zip(wx[m], wy[m], v):
                if np.isfinite(vi):
                    rows.append((xi, yi, float(vi + rng.normal(0, 0.05)), l, float(t)))

    gpd.GeoDataFrame(
        {
            "head": [r[2] for r in rows],
            "layer": [r[3] for r in rows],
            "time": [r[4] for r in rows],
        },
        geometry=[Point(r[0], r[1]) for r in rows],
        crs=CRS,
    ).to_file(outdir / "head_obs.shp")

    gauges = gpd.read_file(outdir / "gauges.shp")
    gx = gauges.geometry.x.to_numpy()
    gy = gauges.geometry.y.to_numpy()
    grows = []
    for i, t in enumerate(times):
        v = _bilinear(stage_series[i], xs, ys, gx, gy)
        base = gauges["stage"].to_numpy()
        for xi, yi, b in zip(gx, gy, base):
            grows.append((xi, yi, float(b - args.drawdown * (t / args.duration)), float(t)))
    gpd.GeoDataFrame(
        {"stage": [r[2] for r in grows], "time": [r[3] for r in grows]},
        geometry=[Point(r[0], r[1]) for r in grows],
        crs=CRS,
    ).to_file(outdir / "gauges.shp")

    # ---- config ----------------------------------------------------------
    cfg = load_config(outdir / "config.yaml")
    cfg.physics.regime = "transient"
    cfg.physics.times = [float(t) for t in times]
    cfg.paths.workdir = "../runs/demo_transient"
    for name in ("head_obs", "gauge_obs", "river_polygon", "river_centerline",
                 "faults", "dtm", "prop_obs", "domain"):
        v = getattr(cfg.paths, name)
        if v:
            setattr(cfg.paths, name, pathlib.Path(v).name)
    cfg.paths.layer_bottoms = [pathlib.Path(v).name for v in cfg.paths.layer_bottoms]
    cfg.save(outdir / "config.yaml")

    with open(truth / "transient_meta.json", "w") as fh:
        json.dump({"times": times, "drawdown_m": args.drawdown,
                   "n_head_obs": len(rows)}, fh, indent=2)

    print(f"\n{len(rows)} time-stamped head observations over {len(times)} times.")
    print(f"Run with:\n  python scripts/train.py {outdir}/config.yaml")


def _read_polygon(path):
    from gwpinn.io.vector import read_polygon

    return read_polygon(path)


def _bilinear(grid, xs, ys, x, y):
    fx = np.interp(x, xs, np.arange(len(xs)))
    fy = np.interp(y, ys[::-1], np.arange(len(ys))[::-1])
    i0 = np.clip(np.floor(fy).astype(int), 0, len(ys) - 2)
    j0 = np.clip(np.floor(fx).astype(int), 0, len(xs) - 2)
    ty, tx = fy - i0, fx - j0
    return (
        grid[i0, j0] * (1 - ty) * (1 - tx) + grid[i0, j0 + 1] * (1 - ty) * tx
        + grid[i0 + 1, j0] * ty * (1 - tx) + grid[i0 + 1, j0 + 1] * ty * tx
    )


if __name__ == "__main__":
    main()
