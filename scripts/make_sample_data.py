#!/usr/bin/env python3
"""Generate the synthetic test case (shapefiles, rasters, config, ground truth)."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gwpinn.data.synthetic import make_synthetic_case


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--outdir", default="sample_data")
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--ny", type=int, default=160)
    ap.add_argument("--cellsize", type=float, default=50.0)
    ap.add_argument("--wells", type=int, default=60)
    ap.add_argument("--gauges", type=int, default=6)
    ap.add_argument("--pumping-tests", type=int, default=28)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    print(f"Generating synthetic case in {args.outdir} ...")
    case = make_synthetic_case(
        args.outdir,
        nx=args.nx,
        ny=args.ny,
        cellsize=args.cellsize,
        n_wells=args.wells,
        n_gauges=args.gauges,
        n_pumping_tests=args.pumping_tests,
        seed=args.seed,
    )
    print(f"\nDone. Run the model with:\n  python scripts/train.py {case.outdir}/config.yaml")


if __name__ == "__main__":
    main()
