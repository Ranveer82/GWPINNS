"""Construction of the benchmark's material-property and stress fields.

The hydraulic conductivity is built in two stages:

1. A layered, log-normal *background* field generated with an FFT circulant
   embedding of an anisotropic exponential covariance.  Layers share a common
   component so that the field is vertically correlated rather than three
   independent slices.
2. A **sharp** overprint of the fault zone -- every cell whose centre falls
   within ``|d| <= w/2`` of the fault plane is overwritten with the scenario's
   fault conductivity.  The result is a genuine discontinuity spanning up to
   five orders of magnitude, which is exactly the feature the PINN
   architectures are being compared on.
"""

from __future__ import annotations

import numpy as np

from ..config import BenchmarkConfig, GridConfig

__all__ = [
    "gaussian_random_field_2d",
    "build_background_conductivity",
    "build_conductivity",
    "fault_mask",
    "build_recharge",
    "build_et_max_rate",
    "well_cells",
]


def gaussian_random_field_2d(
    nrow: int,
    ncol: int,
    dy: float,
    dx: float,
    corr_y: float,
    corr_x: float,
    rng: np.random.Generator,
    pad: int = 2,
) -> np.ndarray:
    """Unit-variance, zero-mean Gaussian random field with exponential covariance.

    Uses circulant embedding on a padded grid so the returned field is not
    artificially periodic across the domain.  An exponential covariance is used
    (rather than Gaussian) because it is unconditionally positive-definite and
    produces the rough, realistic texture expected of a conductivity field.
    """
    NR, NC = nrow * pad, ncol * pad

    # Periodic lag distances on the padded grid.
    iy = np.arange(NR)
    ix = np.arange(NC)
    hy = np.minimum(iy, NR - iy) * dy
    hx = np.minimum(ix, NC - ix) * dx
    lag = np.sqrt((hy[:, None] / corr_y) ** 2 + (hx[None, :] / corr_x) ** 2)

    cov = np.exp(-lag)
    spectrum = np.fft.fft2(cov).real
    spectrum = np.clip(spectrum, 0.0, None)

    noise = rng.standard_normal((NR, NC))
    field = np.fft.ifft2(np.sqrt(spectrum) * np.fft.fft2(noise)).real
    field = field[:nrow, :ncol]

    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= std
    return field


def build_background_conductivity(cfg: BenchmarkConfig) -> np.ndarray:
    """Heterogeneous log-normal K without the fault, shape ``(nlay, nrow, ncol)``."""
    grid, aq = cfg.grid, cfg.aquifer
    rng = np.random.default_rng(aq.seed)

    corr_x, corr_y, _ = aq.corr_len
    shared = gaussian_random_field_2d(
        grid.nrow, grid.ncol, grid.dy, grid.dx, corr_y, corr_x, rng
    )

    # Vertical correlation: each layer mixes a shared component with its own.
    # rho is derived from the vertical correlation length relative to the mean
    # layer thickness, so the config's corr_len[2] stays meaningful.
    mean_dz = float(np.mean(grid.dz))
    rho = float(np.exp(-mean_dz / max(aq.corr_len[2], 1e-9)))
    rho = float(np.clip(rho, 0.0, 0.99))

    k = np.empty((grid.nlay, grid.nrow, grid.ncol), dtype=float)
    for layer in range(grid.nlay):
        own = gaussian_random_field_2d(
            grid.nrow, grid.ncol, grid.dy, grid.dx, corr_y, corr_x, rng
        )
        mixed = rho * shared + np.sqrt(1.0 - rho**2) * own
        ln_k = np.log(aq.k_layer_gmean[layer]) + aq.sigma_lnk * mixed
        k[layer] = np.exp(ln_k)
    return k


def fault_mask(cfg: BenchmarkConfig) -> np.ndarray:
    """Boolean mask of fault-zone cells, shape ``(nlay, nrow, ncol)``."""
    x, y, _ = cfg.grid.cell_center_arrays()
    return cfg.fault.in_fault_zone(x, y)


def build_conductivity(cfg: BenchmarkConfig) -> np.ndarray:
    """Background K with the scenario's fault zone burned in."""
    k = build_background_conductivity(cfg)
    mask = fault_mask(cfg)
    k[mask] = cfg.fault_k
    return k


def build_recharge(cfg: BenchmarkConfig) -> np.ndarray:
    """Areal recharge rate [m/d] on the model grid, shape ``(nrow, ncol)``."""
    grid = cfg.grid
    x, y = np.meshgrid(grid.x_centers(), grid.y_centers(), indexing="xy")
    return cfg.recharge.rate_field(x, y, grid.Lx, grid.Ly)


def build_et_max_rate(cfg: BenchmarkConfig) -> np.ndarray:
    """Maximum ET rate [m/d] on the model grid, shape ``(nrow, ncol)``."""
    grid = cfg.grid
    x, y = np.meshgrid(grid.x_centers(), grid.y_centers(), indexing="xy")
    return cfg.et.max_rate_field(x, y, grid.Lx, grid.Ly)


def well_cells(cfg: BenchmarkConfig) -> list[tuple[int, int, int]]:
    """Resolve each configured well to its ``(layer, row, col)`` cell index."""
    cells = []
    for well in cfg.wells:
        row, col = cfg.grid.locate(well.x, well.y)
        cells.append((well.layer, row, col))
    return cells
