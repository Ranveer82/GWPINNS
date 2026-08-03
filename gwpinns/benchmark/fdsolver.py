"""Reference finite-difference groundwater solver (MODFLOW 6 equivalent).

This is not an approximation of MODFLOW -- it is the same discretisation:

* cell-centred finite volume on the structured grid,
* **harmonic-mean** inter-cell conductances weighted by half cell widths,
  identical to MODFLOW's ``CR``/``CC``/``CV`` formulation for confined flow,
* fully implicit (backward Euler) time stepping,
* constant-head cells imposed by row replacement,
* recharge as a specified cell inflow,
* evapotranspiration linearised implicitly on the MODFLOW ramp, with Picard
  outer iterations to resolve which ramp segment each cell sits on.

It exists so the benchmark is reproducible in environments without the compiled
``mf6`` binary, and so the test-suite can check the forward model against
analytical solutions (Theis, 1-D steady flow with a low-K barrier).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..config import BenchmarkConfig
from .fields import build_conductivity, build_et_max_rate, build_recharge, well_cells
from .modflow6 import ForwardSolution

__all__ = ["build_conductance_matrix", "solve_forward"]


def build_conductance_matrix(cfg: BenchmarkConfig, k3d: np.ndarray) -> sp.csr_matrix:
    """Assemble the symmetric conductance operator ``L``.

    ``(L h)_n = sum_m C_nm (h_n - h_m)`` -- i.e. positive diagonal ``sum C``,
    negative off-diagonals ``-C``.  Units are [L^2/T].
    """
    grid = cfg.grid
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    idx = np.arange(nlay * nrow * ncol).reshape(nlay, nrow, ncol)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    def _connect(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        a, b, c = a.ravel(), b.ravel(), c.ravel()
        rows.extend([a, b, a, b])
        cols.extend([b, a, a, b])
        vals.extend([-c, -c, c, c])

    # x-direction (between columns): harmonic mean over the two half-widths.
    if ncol > 1:
        cx = (2.0 * dy * dz[:, None, None]) / (
            dx / k3d[:, :, :-1] + dx / k3d[:, :, 1:]
        )
        _connect(idx[:, :, :-1], idx[:, :, 1:], cx)

    # y-direction (between rows).
    if nrow > 1:
        cy = (2.0 * dx * dz[:, None, None]) / (
            dy / k3d[:, :-1, :] + dy / k3d[:, 1:, :]
        )
        _connect(idx[:, :-1, :], idx[:, 1:, :], cy)

    # z-direction (between layers): half-thicknesses differ per layer.
    if nlay > 1:
        cz = (2.0 * dx * dy) / (
            dz[:-1, None, None] / k3d[:-1] + dz[1:, None, None] / k3d[1:]
        )
        _connect(idx[:-1], idx[1:], cz)

    n = nlay * nrow * ncol
    mat = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return mat.tocsr()


def _constant_head(cfg: BenchmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(flat_indices, heads)`` for the west and east constant-head faces."""
    grid, bc = cfg.grid, cfg.boundary
    idx = np.arange(grid.nlay * grid.nrow * grid.ncol).reshape(
        grid.nlay, grid.nrow, grid.ncol
    )
    west = idx[:, :, 0].ravel()
    east = idx[:, :, -1].ravel()
    indices = np.concatenate([west, east])
    heads = np.concatenate(
        [np.full(west.size, bc.head_west), np.full(east.size, bc.head_east)]
    )
    return indices, heads


def _well_vector(cfg: BenchmarkConfig, period: int) -> np.ndarray:
    """Volumetric well rates [L^3/T] as a flat cell vector for a stress period."""
    grid = cfg.grid
    q = np.zeros(grid.nlay * grid.nrow * grid.ncol)
    if period < 0:
        return q  # steady-state spin-up: wells are off
    for well, (k, i, j) in zip(cfg.wells, well_cells(cfg)):
        flat = (k * grid.nrow + i) * grid.ncol + j
        q[flat] += well.rate_at(period)
    return q


def _recharge_vector(cfg: BenchmarkConfig, recharge: np.ndarray) -> np.ndarray:
    """Recharge inflow [L^3/T] applied to layer 0."""
    grid = cfg.grid
    q = np.zeros((grid.nlay, grid.nrow, grid.ncol))
    q[0] = recharge * grid.dx * grid.dy
    return q.ravel()


def _et_terms(
    cfg: BenchmarkConfig, head: np.ndarray, et_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Implicit linearisation of the ET ramp.

    Returns ``(slope, offset)`` such that the outflow is
    ``ET(h) = slope * h + offset`` for the segment each cell currently occupies.
    Only layer 0 is affected.
    """
    grid, et = cfg.grid, cfg.et
    shape = (grid.nlay, grid.nrow, grid.ncol)
    slope = np.zeros(shape)
    offset = np.zeros(shape)

    area = grid.dx * grid.dy
    qmax = et_max * area                       # [L^3/T] at full rate
    h_top = head.reshape(shape)[0]
    h_bot = et.surface - et.extinction_depth

    above = h_top >= et.surface
    ramp = (h_top > h_bot) & ~above

    # Saturated segment: constant outflow.
    offset[0][above] = qmax[above]
    # Ramp segment: outflow grows linearly with head.
    slope[0][ramp] = qmax[ramp] / et.extinction_depth
    offset[0][ramp] = -qmax[ramp] * h_bot / et.extinction_depth
    # Below extinction depth: both terms stay zero.

    return slope.ravel(), offset.ravel()


def _solve_step(
    cfg: BenchmarkConfig,
    lap: sp.csr_matrix,
    storage: np.ndarray,
    head_old: np.ndarray | None,
    dt: float | None,
    source: np.ndarray,
    et_max: np.ndarray,
    chd_idx: np.ndarray,
    chd_head: np.ndarray,
    picard_max: int = 25,
    picard_tol: float = 1.0e-8,
) -> np.ndarray:
    """One implicit solve (steady state if ``dt`` is None), with ET Picard loop."""
    n = lap.shape[0]
    steady = dt is None
    stor_coeff = np.zeros(n) if steady else storage / dt

    head = (
        head_old.copy()
        if head_old is not None
        else np.full(n, float(np.mean(chd_head)))
    )
    head[chd_idx] = chd_head

    keep = np.ones(n, dtype=bool)
    keep[chd_idx] = False

    for _ in range(picard_max):
        et_slope, et_offset = _et_terms(cfg, head, et_max)

        diag = stor_coeff + et_slope
        rhs = source - et_offset
        if not steady:
            rhs = rhs + stor_coeff * head_old

        mat = (lap + sp.diags(diag)).tolil()
        # Impose constant heads by row replacement.
        mat[chd_idx, :] = 0.0
        mat = mat.tocsr()
        mat = mat + sp.coo_matrix(
            (np.ones(chd_idx.size), (chd_idx, chd_idx)), shape=(n, n)
        ).tocsr()
        rhs = rhs.copy()
        rhs[chd_idx] = chd_head

        new_head = spla.spsolve(mat.tocsc(), rhs)
        if not np.all(np.isfinite(new_head)):
            raise RuntimeError("finite-difference solve produced non-finite heads")

        delta = np.max(np.abs(new_head - head)) if head is not None else np.inf
        head = new_head
        if delta < picard_tol:
            break

    return head


def solve_forward(cfg: BenchmarkConfig, verbose: bool = False) -> ForwardSolution:
    """Run the full steady-state + transient simulation."""
    grid, tim = cfg.grid, cfg.time

    k3d = build_conductivity(cfg)
    recharge = build_recharge(cfg)
    et_max = build_et_max_rate(cfg)

    lap = build_conductance_matrix(cfg, k3d)
    storage = (
        cfg.aquifer.specific_storage
        * np.broadcast_to(
            grid.cell_volume[:, None, None], (grid.nlay, grid.nrow, grid.ncol)
        )
    ).ravel()

    chd_idx, chd_head = _constant_head(cfg)
    rch_vec = _recharge_vector(cfg, recharge)

    # --- steady-state spin-up ------------------------------------------------
    head = _solve_step(
        cfg, lap, storage, None, None, rch_vec + _well_vector(cfg, -1),
        et_max, chd_idx, chd_head,
    )
    heads = [head.reshape(grid.nlay, grid.nrow, grid.ncol).copy()]
    times = [0.0]

    # --- transient stress periods -------------------------------------------
    t = 0.0
    for period in range(tim.n_periods):
        source = rch_vec + _well_vector(cfg, period)
        for _ in range(tim.steps_per_period):
            head = _solve_step(
                cfg, lap, storage, head, tim.dt, source, et_max, chd_idx, chd_head
            )
            t += tim.dt
            heads.append(head.reshape(grid.nlay, grid.nrow, grid.ncol).copy())
            times.append(t)
        if verbose:
            print(
                f"  stress period {period + 1}/{tim.n_periods} done, "
                f"t={t:.0f} d, head range "
                f"[{head.min():.2f}, {head.max():.2f}] m"
            )

    return ForwardSolution(
        times=np.asarray(times),
        heads=np.stack(heads, axis=0),
        conductivity=k3d,
        recharge=recharge,
        et_max_rate=et_max,
        solver="fd",
    )
