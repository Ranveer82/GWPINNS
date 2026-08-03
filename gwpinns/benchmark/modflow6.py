"""FloPy / MODFLOW 6 forward model for the fault benchmark.

Builds a three-layer confined GWF model bisected by a fault zone, driven by

* constant head on the west and east faces (regional gradient across the fault),
* spatially variable areal recharge (RCHA),
* head-dependent evapotranspiration (EVTA, linear ramp),
* three transient abstraction wells (WEL),

and runs it for one steady-state spin-up period followed by ``n_periods``
transient stress periods.

The aquifer is kept **confined** (``icelltype=0``) on purpose: it makes the
governing equation exactly

    Ss * dh/dt = div(K grad h) + W

which is the equation the PINN residuals implement.  An unconfined formulation
would introduce an ``h``-dependent transmissivity and the comparison between
forward model and inverse model would no longer be exact.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import BenchmarkConfig
from .fields import build_conductivity, build_et_max_rate, build_recharge, well_cells

__all__ = ["ForwardSolution", "find_mf6_executable", "build_simulation", "run_modflow"]


@dataclass
class ForwardSolution:
    """Head history produced by a forward run.

    Attributes
    ----------
    times:
        Output times in days, ``times[0] == 0`` being the steady-state spin-up.
    heads:
        Head array of shape ``(ntimes, nlay, nrow, ncol)``.
    conductivity:
        The true K field, shape ``(nlay, nrow, ncol)`` -- for evaluation only.
    recharge, et_max_rate:
        Stress fields of shape ``(nrow, ncol)``.
    solver:
        ``"mf6"`` or ``"fd"`` -- which engine produced the solution.
    """

    times: np.ndarray
    heads: np.ndarray
    conductivity: np.ndarray
    recharge: np.ndarray
    et_max_rate: np.ndarray
    solver: str

    def __post_init__(self) -> None:
        if self.heads.shape[0] != self.times.size:
            raise ValueError("heads and times disagree on the number of output steps")


def find_mf6_executable(explicit: str | None = None) -> str | None:
    """Locate an ``mf6`` binary on PATH (or validate an explicit path)."""
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        found = shutil.which(explicit)
        return found
    return shutil.which("mf6")


def _stress_period_data(cfg: BenchmarkConfig) -> dict[int, list]:
    """WEL stress-period data. Period 0 is the steady-state spin-up (no pumping)."""
    cells = well_cells(cfg)
    spd: dict[int, list] = {0: []}
    for period in range(cfg.time.n_periods):
        entries = []
        for well, (k, i, j) in zip(cfg.wells, cells):
            rate = well.rate_at(period)
            if rate != 0.0:
                entries.append([(k, i, j), rate, well.name])
        spd[period + 1] = entries
    return spd


def _chd_data(cfg: BenchmarkConfig) -> list:
    """Constant head on the west (col 0) and east (col ncol-1) faces, all layers."""
    grid, bc = cfg.grid, cfg.boundary
    data = []
    for k in range(grid.nlay):
        for i in range(grid.nrow):
            data.append([(k, i, 0), bc.head_west])
            data.append([(k, i, grid.ncol - 1), bc.head_east])
    return data


def build_simulation(
    cfg: BenchmarkConfig,
    workspace: str | Path,
    exe_name: str = "mf6",
):
    """Assemble the MODFLOW 6 simulation. Requires ``flopy``."""
    import flopy  # imported lazily so the PINN code does not depend on flopy

    grid, tim = cfg.grid, cfg.time
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    name = f"fault_{cfg.scenario}"
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        version="mf6",
        exe_name=exe_name,
        sim_ws=str(workspace),
        continue_=False,
    )

    # --- time discretisation ------------------------------------------------
    # Period 0 is a unit-length steady-state spin-up that establishes the
    # pre-pumping head distribution; its output time is shifted to t=0 later.
    perioddata = [(1.0, 1, 1.0)]
    perioddata += [
        (tim.period_length, tim.steps_per_period, 1.0) for _ in range(tim.n_periods)
    ]
    flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=len(perioddata), perioddata=perioddata
    )

    flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="MODERATE",
        outer_maximum=200,
        inner_maximum=300,
        outer_dvclose=1.0e-7,
        inner_dvclose=1.0e-8,
        linear_acceleration="BICGSTAB",
        relaxation_factor=0.0,
    )

    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, save_flows=True, newtonoptions=None
    )

    # --- space --------------------------------------------------------------
    flopy.mf6.ModflowGwfdis(
        gwf,
        length_units="METERS",
        nlay=grid.nlay,
        nrow=grid.nrow,
        ncol=grid.ncol,
        delr=grid.dx,
        delc=grid.dy,
        top=grid.top,
        botm=list(grid.botm),
    )

    # --- properties ---------------------------------------------------------
    k3d = build_conductivity(cfg)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,            # confined -> linear, matches the PINN's PDE
        k=k3d,
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        save_flows=True,
        iconvert=0,
        ss=cfg.aquifer.specific_storage,
        sy=0.0,
        steady_state={0: True},
        transient={1: True},
    )

    # --- initial condition --------------------------------------------------
    # Linear west->east ramp; the steady-state period relaxes it to equilibrium.
    x = grid.x_centers()
    ramp = cfg.boundary.head_west + (
        cfg.boundary.head_east - cfg.boundary.head_west
    ) * (x / grid.Lx)
    strt = np.broadcast_to(ramp, (grid.nlay, grid.nrow, grid.ncol)).copy()
    flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # --- boundary conditions and stresses -----------------------------------
    flopy.mf6.ModflowGwfchd(
        gwf, save_flows=True, maxbound=len(_chd_data(cfg)),
        stress_period_data={0: _chd_data(cfg)},
    )
    flopy.mf6.ModflowGwfrcha(
        gwf, save_flows=True, recharge={0: build_recharge(cfg)}
    )
    flopy.mf6.ModflowGwfevta(
        gwf,
        save_flows=True,
        surface={0: cfg.et.surface},
        rate={0: build_et_max_rate(cfg)},
        depth={0: cfg.et.extinction_depth},
    )
    flopy.mf6.ModflowGwfwel(
        gwf,
        save_flows=True,
        boundnames=True,
        maxbound=len(cfg.wells),
        stress_period_data=_stress_period_data(cfg),
    )

    # --- output control -----------------------------------------------------
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim, gwf, k3d


def run_modflow(
    cfg: BenchmarkConfig,
    workspace: str | Path,
    exe_name: str = "mf6",
    silent: bool = True,
) -> ForwardSolution:
    """Build, run and post-process the MODFLOW 6 model.

    Raises
    ------
    FileNotFoundError
        If no ``mf6`` executable can be located.
    RuntimeError
        If MODFLOW terminates without a normal convergence message.
    """
    import flopy

    exe = find_mf6_executable(exe_name)
    if exe is None:
        raise FileNotFoundError(
            f"MODFLOW 6 executable {exe_name!r} not found on PATH. "
            "Install it with `python -m flopy.utils.get_modflow <bindir>` or use "
            "the bundled finite-difference reference solver instead."
        )

    sim, gwf, k3d = build_simulation(cfg, workspace, exe_name=exe)
    sim.write_simulation(silent=silent)
    success, buff = sim.run_simulation(silent=silent)
    if not success:
        raise RuntimeError("MODFLOW 6 did not terminate normally:\n" + "\n".join(buff))

    head_file = Path(workspace) / f"fault_{cfg.scenario}.hds"
    hds = flopy.utils.HeadFile(str(head_file))
    times = np.asarray(hds.get_times(), dtype=float)
    heads = np.stack([hds.get_data(totim=t) for t in times], axis=0)
    hds.close()

    # Shift the clock so the steady-state spin-up sits at t = 0.
    times = times - times[0]

    return ForwardSolution(
        times=times,
        heads=heads,
        conductivity=k3d,
        recharge=build_recharge(cfg),
        et_max_rate=build_et_max_rate(cfg),
        solver="mf6",
    )
