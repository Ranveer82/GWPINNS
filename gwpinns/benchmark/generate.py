"""Orchestration and on-disk format for the benchmark datasets.

``generate_benchmark`` runs the forward model (MODFLOW 6 when the ``mf6``
binary is available, otherwise the bundled reference finite-difference solver)
and writes a self-contained directory:

    <outdir>/<scenario>/
        config.json        full BenchmarkConfig, round-trippable
        truth.npz          head grid, true K, stress fields, coordinates
        observations.csv   the sparse monitoring dataset (PINN training input)
        sources.csv        well coordinates and pumping schedules
        manifest.json      provenance + summary diagnostics
"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import BenchmarkConfig
from .fields import fault_mask, well_cells
from .modflow6 import ForwardSolution, find_mf6_executable, run_modflow
from .observations import ObservationSet, sample_observations

__all__ = [
    "BenchmarkData",
    "generate_benchmark",
    "load_benchmark",
    "fault_signature",
    "head_jump",
]


@dataclass
class BenchmarkData:
    """Everything a training run needs, plus the truth used only for scoring."""

    cfg: BenchmarkConfig
    times: np.ndarray
    heads: np.ndarray
    k_true: np.ndarray
    recharge: np.ndarray
    et_max_rate: np.ndarray
    obs: ObservationSet
    solver: str


def head_jump(
    cfg: BenchmarkConfig, head: np.ndarray, band: float = 600.0
) -> float:
    """Head discontinuity across the fault, in metres.

    Estimated by regressing head against signed fault distance in a band on
    each side and extrapolating both fits to the respective fault-zone walls
    (``d = -w/2`` and ``d = +w/2``).  Extrapolating rather than averaging
    removes the regional gradient, so the number isolates the jump the fault
    itself causes: large and positive for a barrier, ~0 for a conduit.
    """
    x, y, _ = cfg.grid.cell_center_arrays()
    dist = cfg.fault.signed_distance(x, y)
    half = 0.5 * cfg.fault.width

    walls = []
    for sign in (-1.0, 1.0):
        mask = (sign * dist > half) & (sign * dist < half + band)
        if mask.sum() < 4:
            return float("nan")
        design = np.column_stack([dist[mask], np.ones(int(mask.sum()))])
        slope, intercept = np.linalg.lstsq(design, head[mask], rcond=None)[0]
        walls.append(slope * sign * half + intercept)
    return float(walls[0] - walls[1])


def fault_signature(cfg: BenchmarkConfig, solution: ForwardSolution) -> dict[str, float]:
    """Quantify how strongly the fault imprints on the head field."""
    return {
        "head_jump_steady_m": head_jump(cfg, solution.heads[0]),
        "head_jump_final_m": head_jump(cfg, solution.heads[-1]),
        "fault_k_m_per_d": float(cfg.fault_k),
        "fault_conductance_per_d": float(cfg.fault.conductance(cfg.scenario)),
        "background_k_gmean_m_per_d": float(
            np.exp(np.log(solution.conductivity[~fault_mask(cfg)]).mean())
        ),
    }


def _write_observations(path: Path, obs: ObservationSet) -> None:
    cols = obs.as_columns()
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols.keys())
        for row in zip(*cols.values()):
            writer.writerow(
                [f"{v:.6f}" if isinstance(v, (float, np.floating)) else v for v in row]
            )


def _write_sources(path: Path, cfg: BenchmarkConfig) -> None:
    z_centers = cfg.grid.z_centers()
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["name", "x", "y", "z", "layer", "row", "col", "cell_volume_m3"]
        header += [f"rate_sp{p + 1}_m3_per_d" for p in range(cfg.time.n_periods)]
        writer.writerow(header)
        for well, (k, i, j) in zip(cfg.wells, well_cells(cfg)):
            writer.writerow(
                [
                    well.name,
                    f"{well.x:.2f}",
                    f"{well.y:.2f}",
                    f"{z_centers[k]:.2f}",
                    k,
                    i,
                    j,
                    f"{cfg.grid.cell_volume[k]:.2f}",
                ]
                + [f"{r:.2f}" for r in well.rates]
            )


def generate_benchmark(
    cfg: BenchmarkConfig,
    outdir: str | Path,
    engine: str = "auto",
    exe_name: str = "mf6",
    verbose: bool = True,
) -> BenchmarkData:
    """Run the forward model for one scenario and write the dataset to disk.

    Parameters
    ----------
    engine:
        ``"mf6"`` forces MODFLOW 6 (error if unavailable), ``"fd"`` forces the
        reference solver, ``"auto"`` prefers MODFLOW and falls back with a
        warning.
    """
    from .fdsolver import solve_forward  # local import avoids a cycle

    outdir = Path(outdir) / cfg.scenario
    outdir.mkdir(parents=True, exist_ok=True)

    if engine not in {"auto", "mf6", "fd"}:
        raise ValueError(f"engine must be 'auto', 'mf6' or 'fd', got {engine!r}")

    solution: ForwardSolution | None = None
    if engine in {"auto", "mf6"}:
        have_mf6 = find_mf6_executable(exe_name) is not None
        if have_mf6:
            if verbose:
                print(f"[{cfg.scenario}] running MODFLOW 6 ...")
            solution = run_modflow(cfg, outdir / "mf6", exe_name=exe_name)
        elif engine == "mf6":
            raise FileNotFoundError(
                f"engine='mf6' requested but {exe_name!r} is not on PATH"
            )
        else:
            warnings.warn(
                f"MODFLOW 6 executable {exe_name!r} not found; falling back to the "
                "bundled finite-difference reference solver (same discretisation).",
                RuntimeWarning,
                stacklevel=2,
            )

    if solution is None:
        if verbose:
            print(f"[{cfg.scenario}] running reference finite-difference solver ...")
        solution = solve_forward(cfg, verbose=verbose)

    obs = sample_observations(cfg, solution)

    # --- write ---------------------------------------------------------------
    cfg.to_json(outdir / "config.json")
    np.savez_compressed(
        outdir / "truth.npz",
        times=solution.times,
        heads=solution.heads,
        k_true=solution.conductivity,
        recharge=solution.recharge,
        et_max_rate=solution.et_max_rate,
        x_centers=cfg.grid.x_centers(),
        y_centers=cfg.grid.y_centers(),
        z_centers=cfg.grid.z_centers(),
        fault_mask=fault_mask(cfg),
    )
    _write_observations(outdir / "observations.csv", obs)
    _write_sources(outdir / "sources.csv", cfg)
    np.savez_compressed(
        outdir / "observations.npz",
        **{k: v for k, v in obs.as_columns().items()},
    )

    signature = fault_signature(cfg, solution)
    manifest = {
        "scenario": cfg.scenario,
        "solver": solution.solver,
        "n_cells": int(cfg.grid.nlay * cfg.grid.nrow * cfg.grid.ncol),
        "n_output_times": int(solution.times.size),
        "n_observation_points": obs.n_points,
        "n_observation_values": len(obs),
        "observation_noise_std_m": cfg.observations.noise_std,
        "data_coverage_fraction": float(
            len(obs) / (cfg.grid.nlay * cfg.grid.nrow * cfg.grid.ncol * solution.times.size)
        ),
        "head_range_m": [float(solution.heads.min()), float(solution.heads.max())],
        "k_range_m_per_d": [
            float(solution.conductivity.min()),
            float(solution.conductivity.max()),
        ],
        "fault_signature": signature,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if verbose:
        print(
            f"[{cfg.scenario}] solver={solution.solver} "
            f"heads {manifest['head_range_m'][0]:.2f}..{manifest['head_range_m'][1]:.2f} m, "
            f"{manifest['n_observation_values']} observations "
            f"({100 * manifest['data_coverage_fraction']:.2f}% coverage), "
            f"fault head jump {signature['head_jump_final_m']:+.3f} m"
        )

    return BenchmarkData(
        cfg=cfg,
        times=solution.times,
        heads=solution.heads,
        k_true=solution.conductivity,
        recharge=solution.recharge,
        et_max_rate=solution.et_max_rate,
        obs=obs,
        solver=solution.solver,
    )


def load_benchmark(directory: str | Path) -> BenchmarkData:
    """Read back a dataset written by :func:`generate_benchmark`."""
    directory = Path(directory)
    cfg = BenchmarkConfig.from_json(directory / "config.json")
    truth = np.load(directory / "truth.npz", allow_pickle=False)
    raw = np.load(directory / "observations.npz", allow_pickle=False)
    manifest = json.loads((directory / "manifest.json").read_text())

    obs = ObservationSet(
        name=raw["name"],
        x=raw["x"],
        y=raw["y"],
        z=raw["z"],
        layer=raw["layer"],
        t=raw["time"],
        head_true=raw["head_true"],
        head_obs=raw["head_obs"],
        signed_distance=raw["fault_distance"],
    )
    return BenchmarkData(
        cfg=cfg,
        times=truth["times"],
        heads=truth["heads"],
        k_true=truth["k_true"],
        recharge=truth["recharge"],
        et_max_rate=truth["et_max_rate"],
        obs=obs,
        solver=manifest.get("solver", "unknown"),
    )
