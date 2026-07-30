"""Model domain: active-cell mask, interior sampling, boundary sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import shapely
from rasterio.features import rasterize
from shapely.geometry import MultiPolygon, Polygon, box

from gwpinn.io.raster import Raster, grid_from_bounds


@dataclass
class ModelDomain:
    """The active model area and the raster template that outputs are written on."""

    polygon: object              # shapely Polygon / MultiPolygon
    template: Raster             # grid definition (values unused)
    active: np.ndarray           # bool (ny, nx)

    # ------------------------------------------------------------------ #

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return tuple(self.polygon.bounds)  # type: ignore[return-value]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.template.shape

    @property
    def diagonal(self) -> float:
        xmin, ymin, xmax, ymax = self.bounds
        return float(np.hypot(xmax - xmin, ymax - ymin))

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return shapely.contains_xy(self.polygon, np.asarray(x), np.asarray(y))

    def cell_centers_active(self) -> Tuple[np.ndarray, np.ndarray]:
        xx, yy = self.template.meshgrid()
        return xx[self.active], yy[self.active]

    # ------------------------------------------------------------------ #

    def sample_interior(
        self, n: int, rng: Optional[np.random.Generator] = None, oversample: float = 3.0
    ) -> np.ndarray:
        """``n`` uniform random points inside the domain, shape (n, 2)."""
        rng = rng or np.random.default_rng()
        xmin, ymin, xmax, ymax = self.bounds
        out = np.zeros((0, 2))
        # Rejection sampling; the loop guards against thin/complex domains.
        for _ in range(24):
            m = int(max(n * oversample, 64))
            cand = np.column_stack(
                [rng.uniform(xmin, xmax, m), rng.uniform(ymin, ymax, m)]
            )
            keep = self.contains(cand[:, 0], cand[:, 1])
            out = np.vstack([out, cand[keep]])
            if len(out) >= n:
                break
        if len(out) < n:  # pathological domain - pad from active cell centres
            cx, cy = self.cell_centers_active()
            if cx.size:
                idx = rng.integers(0, cx.size, n - len(out))
                out = np.vstack([out, np.column_stack([cx[idx], cy[idx]])])
        return out[:n]

    def sample_boundary(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the domain outline.

        Returns ``(xy, normals)`` where ``normals`` are unit outward normals -
        what the no-flow condition ``T grad(h) . n = 0`` needs.
        """
        rng = rng or np.random.default_rng()
        polys = (
            list(self.polygon.geoms)
            if isinstance(self.polygon, MultiPolygon)
            else [self.polygon]
        )
        rings = [np.asarray(p.exterior.coords, dtype=float)[:, :2] for p in polys]
        lengths = np.array(
            [np.hypot(*np.diff(r, axis=0).T).sum() for r in rings], dtype=float
        )
        probs = lengths / lengths.sum()
        counts = rng.multinomial(n, probs)

        pts, nrm = [], []
        for ring, cnt in zip(rings, counts):
            if cnt == 0:
                continue
            seg = np.diff(ring, axis=0)
            seg_len = np.hypot(seg[:, 0], seg[:, 1])
            cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            s = rng.uniform(0.0, cum[-1], cnt)
            k = np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1)
            t = (s - cum[k]) / np.maximum(seg_len[k], 1e-12)
            p = ring[k] + seg[k] * t[:, None]
            # Right-hand normal of the segment; orient outward below.
            tang = seg[k] / np.maximum(seg_len[k], 1e-12)[:, None]
            nn = np.column_stack([tang[:, 1], -tang[:, 0]])
            eps = 1e-3 * max(self.diagonal, 1.0)
            outside = ~self.contains(p[:, 0] + eps * nn[:, 0], p[:, 1] + eps * nn[:, 1])
            nn[~outside] *= -1.0
            pts.append(p)
            nrm.append(nn)

        if not pts:
            return np.zeros((0, 2)), np.zeros((0, 2))
        return np.vstack(pts), np.vstack(nrm)


# --------------------------------------------------------------------------- #


def build_domain(
    dtm: Optional[Raster] = None,
    domain_polygon: Optional[object] = None,
    cellsize: Optional[float] = None,
    use_dtm_footprint: bool = True,
    crs: Optional[object] = None,
) -> ModelDomain:
    """Assemble the model domain from a DTM and/or an explicit domain polygon.

    The active area is the intersection of the supplied polygon (if any) with
    the DTM's valid-data footprint (if requested).
    """
    if dtm is None and domain_polygon is None:
        raise ValueError("need a DTM or a domain polygon")

    # ---- raster template -------------------------------------------------
    if dtm is not None and cellsize in (None, dtm.cellsize):
        template = Raster(np.full(dtm.shape, np.nan), dtm.transform, dtm.crs or crs)
    else:
        if cellsize is None:
            raise ValueError("cellsize is required when no DTM is given")
        bounds = dtm.bounds if dtm is not None else domain_polygon.bounds
        template = grid_from_bounds(bounds, cellsize, crs=(dtm.crs if dtm else crs))

    # ---- active mask -----------------------------------------------------
    active = np.ones(template.shape, dtype=bool)

    if dtm is not None and use_dtm_footprint:
        dtm_on_grid = dtm if dtm.shape == template.shape else dtm.resample_to(template)
        active &= np.isfinite(dtm_on_grid.values)

    if domain_polygon is not None:
        burned = rasterize(
            [(domain_polygon, 1)],
            out_shape=template.shape,
            transform=template.transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        ).astype(bool)
        active &= burned

    if not active.any():
        raise ValueError("model domain is empty after masking")

    # ---- polygon describing the active area ------------------------------
    if domain_polygon is not None:
        polygon = domain_polygon
        if dtm is not None and use_dtm_footprint:
            polygon = polygon.intersection(box(*_mask_bounds(template, active)))
    else:
        polygon = box(*_mask_bounds(template, active))

    return ModelDomain(polygon=polygon, template=template, active=active)


def _mask_bounds(template: Raster, active: np.ndarray) -> Tuple[float, float, float, float]:
    """Bounding box (in CRS units) of the True cells of ``active``."""
    rows, cols = np.nonzero(active)
    cs = template.cellsize
    xs, ys = template.cell_centers()
    return (
        float(xs[cols.min()] - 0.5 * cs),
        float(ys[rows.max()] - 0.5 * cs),
        float(xs[cols.max()] + 0.5 * cs),
        float(ys[rows.min()] + 0.5 * cs),
    )


__all__ = ["ModelDomain", "build_domain"]
