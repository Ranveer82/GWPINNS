"""Reference finite-difference solver for the same equation the PINN solves.

This is the ground truth the synthetic experiment is scored against. It is a
conventional cell-centred, harmonic-mean finite-volume discretisation of the
quasi-3D multilayer flow equation - the same scheme MODFLOW uses - so agreement
between it and the PINN is evidence about the PINN, not about a shared
formulation error: one is a mesh-based linear solve, the other a mesh-free
optimisation, and they have no code in common beyond the physics they represent.

Faults enter as multipliers on individual cell faces, which is the discrete form
of the anisotropic barrier the PINN uses.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import shapely
from shapely.geometry import LineString


# --------------------------------------------------------------------------- #
# Fault -> face multipliers
# --------------------------------------------------------------------------- #


def fault_face_multipliers(
    lines: Sequence[np.ndarray],
    alphas: Sequence[float],
    xs: np.ndarray,
    ys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Conductance multipliers on the x- and y-faces of a cell-centred grid.

    A face is "crossed" when the straight segment joining the two cell centres
    intersects the fault trace; its conductance is then multiplied by that
    fault's permeability. Where several faults cross the same face the smallest
    multiplier wins.

    Returns ``(mx, my)`` with shapes ``(ny, nx-1)`` and ``(ny-1, nx)``.
    """
    ny, nx = len(ys), len(xs)
    mx = np.ones((ny, nx - 1))
    my = np.ones((ny - 1, nx))
    if not lines:
        return mx, my

    xx, yy = np.meshgrid(xs, ys)

    # Segments joining horizontally / vertically adjacent cell centres.
    xseg = np.stack(
        [xx[:, :-1], yy[:, :-1], xx[:, 1:], yy[:, 1:]], axis=-1
    ).reshape(-1, 4)
    yseg = np.stack(
        [xx[:-1, :], yy[:-1, :], xx[1:, :], yy[1:, :]], axis=-1
    ).reshape(-1, 4)

    xlines = shapely.linestrings(
        np.stack([xseg[:, [0, 2]], xseg[:, [1, 3]]], axis=-1).reshape(-1, 2, 2)
    )
    ylines = shapely.linestrings(
        np.stack([yseg[:, [0, 2]], yseg[:, [1, 3]]], axis=-1).reshape(-1, 2, 2)
    )

    for line, alpha in zip(lines, alphas):
        arr = np.asarray(line, dtype=float)[:, :2]
        if len(arr) < 2:
            continue
        geom = LineString(arr)
        hit_x = shapely.intersects(xlines, geom).reshape(ny, nx - 1)
        hit_y = shapely.intersects(ylines, geom).reshape(ny - 1, nx)
        mx[hit_x] = np.minimum(mx[hit_x], float(alpha))
        my[hit_y] = np.minimum(my[hit_y], float(alpha))

    return mx, my


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * a * b / np.maximum(a + b, 1e-30)


def _transmissivity(
    K: List[np.ndarray],
    h: np.ndarray,
    top: np.ndarray,
    bottoms: List[np.ndarray],
    unconfined_top: bool,
    min_thickness: float,
) -> List[np.ndarray]:
    T = []
    for l, k in enumerate(K):
        upper = top if l == 0 else bottoms[l - 1]
        full = np.maximum(upper - bottoms[l], min_thickness)
        if l == 0 and unconfined_top:
            b = np.clip(h[0] - bottoms[0], min_thickness, full)
        else:
            b = full
        T.append(k * b)
    return T


def _assemble(
    T: List[np.ndarray],
    active: np.ndarray,
    cellsize: float,
    mx: np.ndarray,
    my: np.ndarray,
    leakance: Sequence[float],
    recharge: np.ndarray,
    river_cond: np.ndarray,
    river_stage: np.ndarray,
    fixed_mask: Optional[np.ndarray],
    fixed_head: Optional[np.ndarray],
    storage_diag: Optional[np.ndarray] = None,
    storage_rhs: Optional[np.ndarray] = None,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    n_layers = len(T)
    ny, nx = active.shape
    ncell = ny * nx
    area = cellsize * cellsize

    def gid(l, i, j):
        return l * ncell + i * nx + j

    rows, cols, vals = [], [], []
    rhs = np.zeros(n_layers * ncell)

    ii, jj = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    for l in range(n_layers):
        diag = np.zeros((ny, nx))

        # ---- horizontal conductances (square cells: C = T_harm) ----------
        cx = _harmonic(T[l][:, :-1], T[l][:, 1:]) * mx
        cy = _harmonic(T[l][:-1, :], T[l][1:, :]) * my
        cx = np.where(active[:, :-1] & active[:, 1:], cx, 0.0)
        cy = np.where(active[:-1, :] & active[1:, :], cy, 0.0)

        for (c, a_i, a_j, b_i, b_j) in (
            (cx, ii[:, :-1], jj[:, :-1], ii[:, 1:], jj[:, 1:]),
            (cy, ii[:-1, :], jj[:-1, :], ii[1:, :], jj[1:, :]),
        ):
            ga = gid(l, a_i.ravel(), a_j.ravel())
            gb = gid(l, b_i.ravel(), b_j.ravel())
            cv = c.ravel()
            rows.extend([ga, gb])
            cols.extend([gb, ga])
            vals.extend([cv, cv])
            np.add.at(diag, (a_i.ravel(), a_j.ravel()), -cv)
            np.add.at(diag, (b_i.ravel(), b_j.ravel()), -cv)

        # ---- vertical leakage --------------------------------------------
        for other, k in ((l - 1, l - 1), (l + 1, l)):
            if 0 <= other < n_layers and 0 <= k < len(leakance):
                cv = np.where(active, float(leakance[k]) * area, 0.0)
                g_self = gid(l, ii.ravel(), jj.ravel())
                g_other = gid(other, ii.ravel(), jj.ravel())
                rows.append(g_self)
                cols.append(g_other)
                vals.append(cv.ravel())
                diag -= cv

        # ---- sources ------------------------------------------------------
        if l == 0:
            rhs[gid(0, ii.ravel(), jj.ravel())] -= (recharge * area * active).ravel()
            rc = river_cond * area * active
            diag -= rc
            rhs[gid(0, ii.ravel(), jj.ravel())] -= (rc * river_stage).ravel()

        if storage_diag is not None:
            # np.where, not multiplication by the mask: the previous head is NaN
            # outside the active area and NaN * 0 is still NaN, which would put
            # NaN into the right-hand side and make the solve return garbage.
            diag -= np.where(active, storage_diag[l], 0.0)
            rhs[gid(l, ii.ravel(), jj.ravel())] -= np.where(
                active, storage_rhs[l], 0.0
            ).ravel()

        # Inactive cells get a trivial identity row.
        diag = np.where(active, diag, 1.0)
        # Guard against a fully isolated active cell.
        diag = np.where(np.abs(diag) < 1e-30, -1.0, diag)

        rows.append(gid(l, ii.ravel(), jj.ravel()))
        cols.append(gid(l, ii.ravel(), jj.ravel()))
        vals.append(diag.ravel())

    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_layers * ncell, n_layers * ncell),
    ).tocsr()

    # ---- prescribed heads -------------------------------------------------
    if fixed_mask is not None and fixed_mask.any():
        A = A.tolil()
        for l in range(n_layers):
            idx = np.nonzero(fixed_mask.ravel())[0] + l * ncell
            for g in idx:
                A.rows[g] = [g]
                A.data[g] = [1.0]
            rhs[idx] = fixed_head.ravel()[np.nonzero(fixed_mask.ravel())[0]]
        A = A.tocsr()

    return A, rhs


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #


def solve_steady(
    K: List[np.ndarray],
    top: np.ndarray,
    bottoms: List[np.ndarray],
    cellsize: float,
    active: np.ndarray,
    leakance: Sequence[float] = (),
    recharge: Optional[np.ndarray] = None,
    river_cond: Optional[np.ndarray] = None,
    river_stage: Optional[np.ndarray] = None,
    fault_mx: Optional[np.ndarray] = None,
    fault_my: Optional[np.ndarray] = None,
    fixed_mask: Optional[np.ndarray] = None,
    fixed_head: Optional[np.ndarray] = None,
    unconfined_top: bool = True,
    min_thickness: float = 1.0,
    n_picard: int = 40,
    tol: float = 1e-5,
    relax: float = 0.7,
    verbose: bool = False,
) -> np.ndarray:
    """Steady-state heads, shape ``(n_layers, ny, nx)``.

    The unconfined top layer makes the system nonlinear, so transmissivity is
    lagged and the solve is repeated (Picard) with under-relaxation until the
    head change falls below ``tol``.
    """
    ny, nx = active.shape
    n_layers = len(K)
    recharge = np.zeros((ny, nx)) if recharge is None else recharge
    river_cond = np.zeros((ny, nx)) if river_cond is None else river_cond
    river_stage = np.zeros((ny, nx)) if river_stage is None else river_stage
    mx = np.ones((ny, nx - 1)) if fault_mx is None else fault_mx
    my = np.ones((ny - 1, nx)) if fault_my is None else fault_my

    # Start from the river stage where known, otherwise mid-aquifer.
    h0 = np.where(river_cond > 0, river_stage, np.nan)
    fill = np.nanmean(h0) if np.isfinite(h0).any() else float(np.nanmean(top) - 5.0)
    h = np.repeat(np.where(np.isfinite(h0), h0, fill)[None], n_layers, axis=0)

    for it in range(n_picard):
        T = _transmissivity(K, h, top, bottoms, unconfined_top, min_thickness)
        A, rhs = _assemble(
            T, active, cellsize, mx, my, leakance, recharge,
            river_cond, river_stage, fixed_mask, fixed_head,
        )
        new = spla.spsolve(A, rhs).reshape(n_layers, ny, nx)
        new = np.where(active[None], new, np.nan)

        delta = np.nanmax(np.abs(new - h)) if np.isfinite(new).any() else np.inf
        # Take the first step in full: the starting guess carries no information
        # worth blending with, and for a confined (linear) system this single
        # solve is already the exact answer. Under-relax only afterwards, where
        # it is damping the unconfined nonlinearity.
        h = new if it == 0 else h + relax * (new - h)
        if verbose:
            print(f"    picard {it:3d}: max dh = {delta:.4e} m")
        if delta < tol:
            break
    else:
        if n_picard > 1 and delta > tol:
            import warnings

            warnings.warn(
                f"steady solve did not converge: max head change {delta:.3e} m "
                f"after {n_picard} Picard iterations (tol={tol:.1e})",
                RuntimeWarning,
                stacklevel=2,
            )

    return np.where(active[None], h, np.nan)


def solve_transient(
    K: List[np.ndarray],
    S: List[np.ndarray],
    top: np.ndarray,
    bottoms: List[np.ndarray],
    cellsize: float,
    active: np.ndarray,
    times: Sequence[float],
    h_init: np.ndarray,
    leakance: Sequence[float] = (),
    recharge_series: Optional[Sequence[np.ndarray]] = None,
    river_cond: Optional[np.ndarray] = None,
    river_stage_series: Optional[Sequence[np.ndarray]] = None,
    fault_mx: Optional[np.ndarray] = None,
    fault_my: Optional[np.ndarray] = None,
    unconfined_top: bool = True,
    min_thickness: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """Backward-Euler transient solution, shape ``(n_times, n_layers, ny, nx)``.

    Storage only enters the equation through the time derivative, which is why a
    transient run is the only setting in which ``S`` is identifiable from heads.
    """
    ny, nx = active.shape
    n_layers = len(K)
    area = cellsize * cellsize
    mx = np.ones((ny, nx - 1)) if fault_mx is None else fault_mx
    my = np.ones((ny - 1, nx)) if fault_my is None else fault_my
    river_cond = np.zeros((ny, nx)) if river_cond is None else river_cond

    times = list(times)
    out = np.zeros((len(times), n_layers, ny, nx))
    # Work with a finite array; masking is applied to the output only.
    h = np.where(np.isfinite(h_init), h_init, np.nanmean(h_init))
    out[0] = np.where(active[None], h, np.nan)

    for step in range(1, len(times)):
        dt = float(times[step] - times[step - 1])
        rch = (
            recharge_series[step] if recharge_series is not None
            else np.zeros((ny, nx))
        )
        stage = (
            river_stage_series[step] if river_stage_series is not None
            else np.zeros((ny, nx))
        )

        h_prev = h.copy()
        for _ in range(6):  # Picard on the unconfined layer
            T = _transmissivity(K, h, top, bottoms, unconfined_top, min_thickness)
            sd = np.stack([S[l] * area / dt for l in range(n_layers)])
            sr = np.stack([S[l] * area / dt * h_prev[l] for l in range(n_layers)])
            A, rhs = _assemble(
                T, active, cellsize, mx, my, leakance, rch,
                river_cond, stage, None, None,
                storage_diag=sd, storage_rhs=sr,
            )
            new = spla.spsolve(A, rhs).reshape(n_layers, ny, nx)
            if np.nanmax(np.abs(new - h)) < 1e-6:
                h = new
                break
            h = new

        out[step] = np.where(active[None], h, np.nan)
        if verbose:
            print(f"    t = {times[step]:8.2f} d, mean head = {np.nanmean(h):.3f} m")

    return out


__all__ = [
    "solve_steady",
    "solve_transient",
    "fault_face_multipliers",
]
