"""Raster input/output and point sampling."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine, rowcol, xy


@dataclass
class Raster:
    """A single-band raster held in memory.

    ``values`` is a float array with NaN marking no-data, which keeps the
    downstream arithmetic simple.
    """

    values: np.ndarray  # (ny, nx), float, NaN = nodata
    transform: Affine
    crs: Optional[object] = None

    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape

    @property
    def cellsize(self) -> float:
        return float(abs(self.transform.a))

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax)."""
        ny, nx = self.shape
        x0, y0 = self.transform * (0, 0)
        x1, y1 = self.transform * (nx, ny)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """1-D coordinate vectors of the cell centres (x ascending, y descending)."""
        ny, nx = self.shape
        xs, _ = xy(self.transform, np.zeros(nx, dtype=int), np.arange(nx))
        _, ys = xy(self.transform, np.arange(ny), np.zeros(ny, dtype=int))
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = self.cell_centers()
        return np.meshgrid(xs, ys)

    # ------------------------------------------------------------------ #

    def sample(self, x: np.ndarray, y: np.ndarray, method: str = "bilinear") -> np.ndarray:
        """Sample the raster at arbitrary coordinates.

        Points outside the raster, or landing on no-data, come back as NaN.
        Bilinear weights are renormalised over the valid neighbours so that a
        point next to a no-data cell still gets a sensible value instead of
        being poisoned by the NaN.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        ny, nx = self.shape

        # Fractional pixel coordinates of the cell centres.
        inv = ~self.transform
        col, row = inv * (x, y)
        col = np.asarray(col, dtype=float) - 0.5
        row = np.asarray(row, dtype=float) - 0.5

        if method == "nearest":
            c = np.rint(col).astype(int)
            r = np.rint(row).astype(int)
            ok = (c >= 0) & (c < nx) & (r >= 0) & (r < ny)
            out = np.full(x.shape, np.nan)
            out[ok] = self.values[r[ok], c[ok]]
            return out

        c0 = np.floor(col).astype(int)
        r0 = np.floor(row).astype(int)
        fx = col - c0
        fy = row - r0

        out = np.full(x.shape, np.nan)
        acc = np.zeros(x.shape)
        wsum = np.zeros(x.shape)
        for dr, dc, wgt in (
            (0, 0, (1 - fx) * (1 - fy)),
            (0, 1, fx * (1 - fy)),
            (1, 0, (1 - fx) * fy),
            (1, 1, fx * fy),
        ):
            rr = np.clip(r0 + dr, 0, ny - 1)
            cc = np.clip(c0 + dc, 0, nx - 1)
            inside = (r0 + dr >= 0) & (r0 + dr < ny) & (c0 + dc >= 0) & (c0 + dc < nx)
            v = self.values[rr, cc]
            good = inside & np.isfinite(v) & (wgt > 0)
            acc[good] += wgt[good] * v[good]
            wsum[good] += wgt[good]

        ok = wsum > 1e-12
        out[ok] = acc[ok] / wsum[ok]
        return out

    def resample_to(self, other: "Raster", method: str = "bilinear") -> "Raster":
        """Resample onto the grid of ``other`` (same CRS assumed)."""
        xx, yy = other.meshgrid()
        vals = self.sample(xx.ravel(), yy.ravel(), method=method).reshape(other.shape)
        return Raster(vals, other.transform, other.crs)

    def filled(self, value: float) -> np.ndarray:
        out = self.values.copy()
        out[~np.isfinite(out)] = value
        return out

    def copy_like(self, values: np.ndarray) -> "Raster":
        return Raster(np.asarray(values, dtype=float), self.transform, self.crs)


# --------------------------------------------------------------------------- #


def read_raster(path: str | pathlib.Path, band: int = 1) -> Raster:
    """Read a single band into a :class:`Raster` with NaN no-data."""
    with rasterio.open(str(path)) as src:
        arr = src.read(band, masked=True).astype("float64")
        values = arr.filled(np.nan)
        return Raster(values, src.transform, src.crs)


def write_raster(
    path: str | pathlib.Path,
    values: np.ndarray,
    transform: Affine,
    crs: Optional[object] = None,
    nodata: float = -9999.0,
    dtype: str = "float32",
) -> None:
    """Write a single-band GeoTIFF, mapping NaN to ``nodata``."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype="float64")
    out = np.where(np.isfinite(values), values, nodata).astype(dtype)
    ny, nx = out.shape
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        tiled=False,
    ) as dst:
        dst.write(out, 1)


def make_transform(xmin: float, ymax: float, cellsize: float) -> Affine:
    """North-up affine transform for a regular grid."""
    return Affine(cellsize, 0.0, xmin, 0.0, -cellsize, ymax)


def grid_from_bounds(
    bounds: Tuple[float, float, float, float], cellsize: float, crs=None
) -> Raster:
    """Empty (all-NaN) raster covering ``bounds`` at ``cellsize`` resolution."""
    xmin, ymin, xmax, ymax = bounds
    nx = max(1, int(np.ceil((xmax - xmin) / cellsize)))
    ny = max(1, int(np.ceil((ymax - ymin) / cellsize)))
    transform = make_transform(xmin, ymin + ny * cellsize, cellsize)
    return Raster(np.full((ny, nx), np.nan), transform, crs)


__all__ = [
    "Raster",
    "read_raster",
    "write_raster",
    "make_transform",
    "grid_from_bounds",
    "rowcol",
]
