"""Collocation, boundary and interface point sampling.

Uniform sampling alone will not resolve this problem: the residual is dominated
by two thin features -- the pumping cells and the fault zone -- that together
occupy well under 1% of the domain volume.  The sampler therefore draws a
configurable fraction of points *directly* inside those features, using exact
inverse geometry (sample the signed fault distance and solve for ``x``) rather
than rejection sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import BenchmarkConfig
from ..benchmark.fields import well_cells

__all__ = ["CollocationBatch", "BoundaryBatch", "InterfaceBatch", "DomainSampler"]


@dataclass
class CollocationBatch:
    """Physical-unit column tensors of shape ``(N, 1)``."""

    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    t: torch.Tensor

    def __len__(self) -> int:
        return int(self.x.shape[0])


@dataclass
class BoundaryBatch:
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    t: torch.Tensor
    value: torch.Tensor | None = None   # prescribed head for Dirichlet faces
    axis: int | None = None             # 0/1/2 -- normal direction for no-flow faces


@dataclass
class InterfaceBatch:
    """Points on the fault mid-plane, with the two wall positions either side."""

    y: torch.Tensor
    z: torch.Tensor
    t: torch.Tensor
    x_west: torch.Tensor
    y_west: torch.Tensor
    x_east: torch.Tensor
    y_east: torch.Tensor


class DomainSampler:
    """Draws training points for one benchmark configuration."""

    def __init__(
        self,
        cfg: BenchmarkConfig,
        device: torch.device | str = "cpu",
        seed: int = 0,
        well_pad_cells: float = 2.0,
        dtype: torch.dtype | None = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.rng = np.random.default_rng(seed)
        self.well_pad_cells = well_pad_cells

        grid = cfg.grid
        self.t_max = cfg.time.total_time
        self.nx, self.ny = cfg.fault.normal
        self.half_width = 0.5 * cfg.fault.width

        z_edges = np.concatenate([[grid.top], np.asarray(grid.botm, dtype=float)])
        self._z_edges = z_edges
        self._well_boxes = []
        for well, (k, i, j) in zip(cfg.wells, well_cells(cfg)):
            pad_x = self.well_pad_cells * grid.dx
            pad_y = self.well_pad_cells * grid.dy
            self._well_boxes.append(
                (
                    max(well.x - pad_x, 0.0),
                    min(well.x + pad_x, grid.Lx),
                    max(well.y - pad_y, 0.0),
                    min(well.y + pad_y, grid.Ly),
                    z_edges[k + 1],
                    z_edges[k],
                )
            )

    # -- helpers ------------------------------------------------------------- #
    def _col(self, values: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        tensor = torch.as_tensor(
            np.asarray(values, dtype=np.float64).reshape(-1, 1),
            device=self.device,
            dtype=self.dtype,
        )
        return tensor.requires_grad_(requires_grad)

    def _uniform_time(self, n: int) -> np.ndarray:
        """Times strictly inside the transient window."""
        return self.rng.uniform(1e-6, self.t_max, size=n)

    def _x_from_distance(self, dist: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve ``d = (x-x0)nx + (y-y0)ny`` for ``x``."""
        fault = self.cfg.fault
        return fault.x0 + (dist - (y - fault.y0) * self.ny) / self.nx

    def _bulk_points(self, n: int, side: int | None) -> tuple[np.ndarray, ...]:
        """Uniform points in the domain (or in one fault block)."""
        grid = self.cfg.grid
        xs, ys, zs = [], [], []
        remaining = n
        for _ in range(64):
            if remaining <= 0:
                break
            draw = max(remaining * 2, 64)
            x = self.rng.uniform(0.0, grid.Lx, size=draw)
            y = self.rng.uniform(0.0, grid.Ly, size=draw)
            z = self.rng.uniform(grid.zbot, grid.top, size=draw)
            if side is not None:
                dist = self.cfg.fault.signed_distance(x, y)
                keep = (side * dist) > self.half_width
                x, y, z = x[keep], y[keep], z[keep]
            take = min(remaining, x.size)
            xs.append(x[:take])
            ys.append(y[:take])
            zs.append(z[:take])
            remaining -= take
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)

    def _fault_points(self, n: int, side: int | None, band: float) -> tuple[np.ndarray, ...]:
        """Points concentrated within ``band`` metres of the fault."""
        grid = self.cfg.grid
        if side is None:
            low, high = -self.half_width - band, self.half_width + band
        else:
            low, high = self.half_width, self.half_width + band
        xs, ys, zs = [], [], []
        remaining = n
        for _ in range(64):
            if remaining <= 0:
                break
            draw = max(remaining * 2, 64)
            dist = self.rng.uniform(low, high, size=draw)
            if side is not None:
                dist = side * dist
            y = self.rng.uniform(0.0, grid.Ly, size=draw)
            z = self.rng.uniform(grid.zbot, grid.top, size=draw)
            x = self._x_from_distance(dist, y)
            keep = (x >= 0.0) & (x <= grid.Lx)
            x, y, z = x[keep], y[keep], z[keep]
            take = min(remaining, x.size)
            xs.append(x[:take])
            ys.append(y[:take])
            zs.append(z[:take])
            remaining -= take
        if not xs:
            return self._bulk_points(n, side)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)

    def _well_points(self, n: int, side: int | None) -> tuple[np.ndarray, ...]:
        """Points in and around the pumping cells."""
        boxes = self._well_boxes
        if not boxes:
            return self._bulk_points(n, side)
        xs, ys, zs = [], [], []
        remaining = n
        for _ in range(64):
            if remaining <= 0:
                break
            draw = max(remaining * 2, 64)
            pick = self.rng.integers(0, len(boxes), size=draw)
            box = np.asarray(boxes)[pick]
            x = self.rng.uniform(box[:, 0], box[:, 1])
            y = self.rng.uniform(box[:, 2], box[:, 3])
            z = self.rng.uniform(box[:, 4], box[:, 5])
            if side is not None:
                dist = self.cfg.fault.signed_distance(x, y)
                keep = (side * dist) > self.half_width
                x, y, z = x[keep], y[keep], z[keep]
            take = min(remaining, x.size)
            xs.append(x[:take])
            ys.append(y[:take])
            zs.append(z[:take])
            remaining -= take
        if not xs or np.concatenate(xs).size == 0:
            return self._bulk_points(n, side)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)

    # -- public API ---------------------------------------------------------- #
    def collocation(
        self,
        n: int,
        well_fraction: float = 0.15,
        fault_fraction: float = 0.25,
        fault_band: float = 400.0,
        side: int | None = None,
        requires_grad: bool = True,
    ) -> CollocationBatch:
        """Draw ``n`` PDE collocation points.

        ``side`` restricts sampling to one fault block (``-1`` west, ``+1``
        east) for the cPINN; ``None`` covers the whole domain including the
        fault zone itself, which is what the baseline and mixed models see.
        """
        n_well = int(round(n * well_fraction))
        n_fault = int(round(n * fault_fraction))
        n_bulk = max(n - n_well - n_fault, 0)

        parts = [self._bulk_points(n_bulk, side)]
        if n_fault:
            parts.append(self._fault_points(n_fault, side, fault_band))
        if n_well:
            parts.append(self._well_points(n_well, side))

        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        z = np.concatenate([p[2] for p in parts])
        t = self._uniform_time(x.size)

        return CollocationBatch(
            x=self._col(x, requires_grad),
            y=self._col(y, requires_grad),
            z=self._col(z, requires_grad),
            t=self._col(t, requires_grad),
        )

    def dirichlet(self, n: int, requires_grad: bool = False) -> BoundaryBatch:
        """Points on the constant-head west and east faces."""
        grid, bc = self.cfg.grid, self.cfg.boundary
        half = n // 2
        x = np.concatenate([np.zeros(half), np.full(n - half, grid.Lx)])
        value = np.concatenate(
            [np.full(half, bc.head_west), np.full(n - half, bc.head_east)]
        )
        y = self.rng.uniform(0.0, grid.Ly, size=n)
        z = self.rng.uniform(grid.zbot, grid.top, size=n)
        t = self._uniform_time(n)
        return BoundaryBatch(
            x=self._col(x, requires_grad),
            y=self._col(y, requires_grad),
            z=self._col(z, requires_grad),
            t=self._col(t, requires_grad),
            value=self._col(value),
        )

    def neumann(self, n: int, side: int | None = None) -> list[BoundaryBatch]:
        """Points on the four no-flow faces: north/south (y) and top/bottom (z).

        Returned batches carry ``axis`` = 1 for the ``y`` faces and 2 for the
        ``z`` faces, i.e. the direction whose flux must vanish.
        """
        grid = self.cfg.grid
        per_face = max(n // 4, 1)
        batches: list[BoundaryBatch] = []

        for axis, fixed_values in ((1, (0.0, grid.Ly)), (2, (grid.zbot, grid.top))):
            for fixed in fixed_values:
                xs, ys, zs = [], [], []
                remaining = per_face
                for _ in range(64):
                    if remaining <= 0:
                        break
                    draw = max(remaining * 2, 64)
                    x = self.rng.uniform(0.0, grid.Lx, size=draw)
                    y = (
                        np.full(draw, fixed)
                        if axis == 1
                        else self.rng.uniform(0.0, grid.Ly, size=draw)
                    )
                    z = (
                        np.full(draw, fixed)
                        if axis == 2
                        else self.rng.uniform(grid.zbot, grid.top, size=draw)
                    )
                    if side is not None:
                        dist = self.cfg.fault.signed_distance(x, y)
                        keep = (side * dist) > self.half_width
                        x, y, z = x[keep], y[keep], z[keep]
                    take = min(remaining, x.size)
                    xs.append(x[:take])
                    ys.append(y[:take])
                    zs.append(z[:take])
                    remaining -= take
                if not xs or np.concatenate(xs).size == 0:
                    continue
                x, y, z = (np.concatenate(a) for a in (xs, ys, zs))
                batches.append(
                    BoundaryBatch(
                        x=self._col(x, True),
                        y=self._col(y, True),
                        z=self._col(z, True),
                        t=self._col(self._uniform_time(x.size), True),
                        axis=axis,
                    )
                )
        return batches

    def interface(self, n: int, requires_grad: bool = True) -> InterfaceBatch:
        """Points on the fault mid-plane plus the matching wall positions.

        The two wall points sit at signed distance ``-w/2`` and ``+w/2``, i.e.
        exactly on the faces of the fault zone, so the cPINN's two networks are
        never evaluated inside the zone they do not model.
        """
        grid = self.cfg.grid
        offset_x = self.half_width * self.nx
        offset_y = self.half_width * self.ny

        ys, zs, xms = [], [], []
        remaining = n
        for _ in range(64):
            if remaining <= 0:
                break
            draw = max(remaining * 2, 64)
            # Keep the y-margin so that the normal offset stays inside the grid.
            y = self.rng.uniform(abs(offset_y), grid.Ly - abs(offset_y), size=draw)
            z = self.rng.uniform(grid.zbot, grid.top, size=draw)
            x_mid = self._x_from_distance(np.zeros(draw), y)
            keep = (x_mid - abs(offset_x) >= 0.0) & (x_mid + abs(offset_x) <= grid.Lx)
            take = min(remaining, int(keep.sum()))
            ys.append(y[keep][:take])
            zs.append(z[keep][:take])
            xms.append(x_mid[keep][:take])
            remaining -= take

        y = np.concatenate(ys)
        z = np.concatenate(zs)
        x_mid = np.concatenate(xms)

        # Step along the plane normal, which moves in x *and* y.
        return InterfaceBatch(
            y=self._col(y, requires_grad),
            z=self._col(z, requires_grad),
            t=self._col(self._uniform_time(y.size), requires_grad),
            x_west=self._col(x_mid - offset_x, requires_grad),
            y_west=self._col(y - offset_y, requires_grad),
            x_east=self._col(x_mid + offset_x, requires_grad),
            y_east=self._col(y + offset_y, requires_grad),
        )
