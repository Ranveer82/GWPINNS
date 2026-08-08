#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much of the "head error" is MODFLOW's own time-discretisation error?

Motivation
----------
The control-volume surrogate got *worse* on head when given four times the
compute (0.772 -> 0.826 m) while its mass balance improved sharply
(0.672 -> 0.503).  That combination is the signature of a surrogate converging
to a different solution from its reference, not of a surrogate failing to
converge.

There is an obvious candidate.  The reference takes one backward-Euler step per
2-hour stress period, so the 12.4 h semi-diurnal tide is resolved with about six
steps per cycle.  Backward Euler is first-order and strongly damping, so at that
resolution it noticeably attenuates the tidal signal.  The surrogates, by
contrast, differentiate ``h`` with respect to ``t`` analytically and so solve the
*continuous-time* problem.  As a surrogate drives its residual down it approaches
the exact solution - which is not the reference.

If that is what is happening, the head criterion has a floor that this study
never established, and "0.77 m" is partly a measurement of MODFLOW's time step.

The test
--------
Re-solve exactly the same case with ``substeps`` backward-Euler steps per stress
period and compare against the 1-step reference at the same output times.  Their
difference is the time-discretisation error of the reference, and therefore the
floor of the head metric for any continuous-time surrogate.

    python scripts/check_reference_timestep.py --substeps 16
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Optional, Sequence

import numpy as np

from gwpinn.benchmark import build_reduced_case, load_reduced_case


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo = pathlib.Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default=str(repo / "runs" / "mf6_hetero_benchmark"))
    ap.add_argument("--substeps", type=int, default=16)
    ap.add_argument("--out", default=str(repo / "runs" / "pinn_formulation_study"
                                         / "timestep_floor.json"))
    args = ap.parse_args(argv)

    bench = pathlib.Path(args.bench)
    coarse = load_reduced_case(bench / "reduced_case" / "case.npz")
    print(f"[coarse] {coarse.nper} stress periods, 1 step each "
          f"({coarse.dt * 24:g} h per step)")

    fine = build_reduced_case(
        args.bench, substeps=args.substeps,
        workdir=bench / "reduced_case" / f"mf6_sub{args.substeps}", quiet=True,
    )
    print(f"[fine]   {fine.nper} stress periods, {args.substeps} steps each "
          f"({coarse.dt * 24 / args.substeps:g} h per step)")

    a, b = coarse.head, fine.head
    diff = a - b
    amp_a = (a.max(0) - a.min(0))
    amp_b = (b.max(0) - b.min(0))

    ff = coarse.fault_faces.astype(int)
    ja = a[:, ff[:, 0], ff[:, 1]] - a[:, ff[:, 2], ff[:, 3]]
    jb = b[:, ff[:, 0], ff[:, 1]] - b[:, ff[:, 2], ff[:, 3]]

    out = {
        "substeps": args.substeps,
        "head_rmse_coarse_vs_fine_m": float(np.sqrt((diff ** 2).mean())),
        "head_max_diff_m": float(np.abs(diff).max()),
        "amplitude_coarse_m": float(amp_a.mean()),
        "amplitude_fine_m": float(amp_b.mean()),
        "amplitude_damping_frac": float(1.0 - amp_a.mean() / max(amp_b.mean(), 1e-9)),
        "fault_jump_rmse_coarse_vs_fine_m": float(np.sqrt(((ja - jb) ** 2).mean())),
    }

    print("\n=== time-discretisation error of the reference ===")
    for k, v in out.items():
        print(f"  {k:36s} {v:.5g}")
    print("\nInterpretation: the head RMSE above is a *floor* for any "
          "continuous-time surrogate\nscored against the 1-step reference - "
          "error it cannot remove by converging.")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
