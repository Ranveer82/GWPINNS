"""Differentiable source/sink term ``W(x, y, z, t, h)`` in 1/day.

``W`` is the volumetric source per unit aquifer volume appearing in

    Ss dh/dt = div(K grad h) + W

and it is **known** to the inverse problem: well locations and pumping
schedules are metered, and recharge/ET are prescribed constitutive laws.  What
is unknown is the conductivity field.

Three contributions, each reproducing the discrete form the forward model
applied so that the residual is consistent with the data:

* **Wells** -- rate spread over the volume of the cell containing the screen,
  ``Q / V_cell``, active only inside that cell and only during the stress
  periods when the pump runs.
* **Recharge** -- ``R(x, y) / dz_0`` inside the top layer, zero elsewhere
  (MODFLOW applies RCH as a cell source; the aquifer top face is no-flow).
* **Evapotranspiration** -- ``-ET(h)/dz_0`` inside the top layer, following the
  MODFLOW linear ramp.  This one depends on the *predicted* head, so it stays
  in the autograd graph and the PINN solves a genuinely head-dependent sink.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import BenchmarkConfig
from ..benchmark.fields import build_et_max_rate, build_recharge, well_cells

__all__ = ["ForcingTerm"]


class ForcingTerm(nn.Module):
    """Evaluates ``W`` at arbitrary points, differentiably in ``h``."""

    def __init__(
        self,
        cfg: BenchmarkConfig,
        recharge: np.ndarray | None = None,
        et_max_rate: np.ndarray | None = None,
    ):
        super().__init__()
        grid, tim, et = cfg.grid, cfg.time, cfg.et
        self.nrow, self.ncol, self.nlay = grid.nrow, grid.ncol, grid.nlay
        self.dx, self.dy, self.Ly = grid.dx, grid.dy, grid.Ly
        self.period_length = tim.period_length
        self.n_periods = tim.n_periods
        self.et_surface = et.surface
        self.et_extinction = et.extinction_depth

        if recharge is None:
            recharge = build_recharge(cfg)
        if et_max_rate is None:
            et_max_rate = build_et_max_rate(cfg)

        self.register_buffer("recharge", torch.as_tensor(recharge.ravel(), dtype=torch.float32))
        self.register_buffer("et_max", torch.as_tensor(et_max_rate.ravel(), dtype=torch.float32))
        self.register_buffer("botm", torch.as_tensor(np.asarray(grid.botm), dtype=torch.float32))
        self.register_buffer("dz", torch.as_tensor(grid.dz, dtype=torch.float32))

        cells = well_cells(cfg)
        self.n_wells = len(cells)
        self.register_buffer(
            "well_cell",
            torch.as_tensor(np.asarray(cells, dtype=np.int64), dtype=torch.long),
        )
        self.register_buffer(
            "well_rates",
            torch.as_tensor(
                np.asarray([w.rates for w in cfg.wells], dtype=np.float64),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "well_volume",
            torch.as_tensor(
                np.asarray([grid.cell_volume[k] for k, _, _ in cells]), dtype=torch.float32
            ),
        )

    # -- discrete cell lookup ------------------------------------------------ #
    def _cell_indices(self, x, y, z):
        """Nearest-cell ``(layer, row, col)`` for physical coordinates."""
        col = torch.clamp(torch.floor(x / self.dx), 0, self.ncol - 1).long()
        row = torch.clamp(torch.floor((self.Ly - y) / self.dy), 0, self.nrow - 1).long()
        # botm is descending, so counting the bottoms above z gives the layer.
        layer = torch.clamp(
            (z.unsqueeze(-1) < self.botm).sum(dim=-1), 0, self.nlay - 1
        ).long()
        return layer, row, col

    def _stress_period(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transient stress-period index and an ``is-pumping`` mask."""
        active = (t > 0.0).to(t.dtype)
        period = torch.clamp(
            torch.ceil(t / self.period_length) - 1.0, 0, self.n_periods - 1
        ).long()
        return period, active

    # -- forcing components -------------------------------------------------- #
    def evapotranspiration_rate(self, h: torch.Tensor, et_max: torch.Tensor) -> torch.Tensor:
        """MODFLOW linear ET ramp, in m/d. Differentiable in ``h``."""
        frac = (h - (self.et_surface - self.et_extinction)) / self.et_extinction
        return et_max * torch.clamp(frac, 0.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``W`` in 1/day with shape ``(N, 1)``.

        All inputs are physical-unit column tensors of shape ``(N, 1)``.
        """
        xf, yf, zf, tf = (a.reshape(-1) for a in (x, y, z, t))
        hf = h.reshape(-1)

        layer, row, col = self._cell_indices(xf, yf, zf)
        flat2d = row * self.ncol + col
        top = (layer == 0).to(xf.dtype)
        dz_top = self.dz[0]

        # Recharge (inflow) and ET (head-dependent outflow), top layer only.
        w = top * self.recharge[flat2d] / dz_top
        et = self.evapotranspiration_rate(hf, self.et_max[flat2d])
        w = w - top * et / dz_top

        # Wells: active only inside the screened cell during a pumping period.
        period, active = self._stress_period(tf)
        for idx in range(self.n_wells):
            k, i, j = (int(v) for v in self.well_cell[idx])
            inside = ((layer == k) & (row == i) & (col == j)).to(xf.dtype)
            rate = self.well_rates[idx][period] * active
            w = w + inside * rate / self.well_volume[idx]

        return w.reshape(-1, 1)

    # -- convenience --------------------------------------------------------- #
    @torch.no_grad()
    def on_grid(self, cfg: BenchmarkConfig, heads: np.ndarray, time: float) -> np.ndarray:
        """Evaluate ``W`` on the model grid, for cross-checking against the solver."""
        x, y, z = cfg.grid.cell_center_arrays()
        device = self.recharge.device
        args = [
            torch.as_tensor(a.ravel(), dtype=torch.float32, device=device).reshape(-1, 1)
            for a in (x, y, z)
        ]
        t = torch.full_like(args[0], float(time))
        h = torch.as_tensor(
            heads.ravel(), dtype=torch.float32, device=device
        ).reshape(-1, 1)
        return self(args[0], args[1], args[2], t, h).cpu().numpy().reshape(x.shape)
