"""River stage interpolation.

Gauge stations give the water-surface elevation at a handful of points; the
river polygon gives the wetted area over which that stage has to be known. The
interpolation is done in *along-stream* coordinates rather than in the plane:
stage varies almost entirely with distance downstream, so a 2-D interpolator
would leak the downstream gradient across meander bends and produce a stage
surface that is not monotone along the channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import shapely
from scipy.interpolate import PchipInterpolator
from shapely.geometry import MultiPolygon, Polygon

# --------------------------------------------------------------------------- #
# Polyline helpers
# --------------------------------------------------------------------------- #


def polyline_arclength(nodes: np.ndarray) -> np.ndarray:
    """Cumulative arc length at each node."""
    d = np.hypot(*np.diff(nodes, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_polyline(nodes: np.ndarray, n: int) -> np.ndarray:
    """Re-sample a polyline to ``n`` equally spaced nodes."""
    s = polyline_arclength(nodes)
    if s[-1] <= 0:
        return np.repeat(nodes[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    return np.column_stack(
        [np.interp(target, s, nodes[:, 0]), np.interp(target, s, nodes[:, 1])]
    )


def smooth_polyline(nodes: np.ndarray, passes: int = 2) -> np.ndarray:
    """Simple 1-2-1 smoothing with fixed endpoints."""
    out = nodes.copy()
    for _ in range(passes):
        inner = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = np.vstack([out[:1], inner, out[-1:]])
    return out


def project_to_polyline(
    nodes: np.ndarray, pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points onto a polyline.

    Returns ``(s, dist, foot)``: arc-length position of the closest point on the
    polyline, the perpendicular distance to it, and the foot point itself.
    """
    a = nodes[:-1]                      # (S, 2)
    b = nodes[1:]
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    L2 = np.maximum(L2, 1e-12)
    s0 = polyline_arclength(nodes)[:-1]
    seglen = np.sqrt(L2)

    ap = pts[:, None, :] - a[None, :, :]             # (N, S, 2)
    t = np.einsum("nsi,si->ns", ap, ab) / L2[None, :]
    t = np.clip(t, 0.0, 1.0)
    foot = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(pts[:, None, :] - foot, axis=2)

    k = np.argmin(d, axis=1)
    idx = np.arange(len(pts))
    return (
        s0[k] + t[idx, k] * seglen[k],
        d[idx, k],
        foot[idx, k],
    )


def sample_points_in_polygon(
    poly, n: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    xmin, ymin, xmax, ymax = poly.bounds
    out = np.zeros((0, 2))
    for _ in range(32):
        cand = np.column_stack(
            [rng.uniform(xmin, xmax, 4 * n), rng.uniform(ymin, ymax, 4 * n)]
        )
        keep = shapely.contains_xy(poly, cand[:, 0], cand[:, 1])
        out = np.vstack([out, cand[keep]])
        if len(out) >= n:
            break
    return out[:n]


# --------------------------------------------------------------------------- #
# Centerline
# --------------------------------------------------------------------------- #


def polygon_centerline(
    poly,
    n_nodes: int = 60,
    n_samples: int = 6000,
    n_iter: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Extract a channel centerline from a river-surface polygon.

    A principal curve is fitted to points sampled inside the polygon: start from
    the first principal axis, bin the points by their position along the current
    curve, take the cross-sectional centroid of each bin, then re-project and
    repeat. For an elongated polygon this converges to the medial axis in a few
    iterations and, unlike a raster skeleton, needs no extra dependency and
    produces no spurious branches.
    """
    rng = rng or np.random.default_rng(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)

    pts = sample_points_in_polygon(poly, n_samples, rng)
    if len(pts) < 10:
        return np.asarray(poly.exterior.coords, dtype=float)[:, :2]

    # Initialise with the first principal axis.
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    t = (pts - c) @ vt[0]

    nodes = _bin_centroids(pts, t, n_nodes)
    for _ in range(n_iter):
        nodes = smooth_polyline(resample_polyline(nodes, n_nodes), passes=2)
        t, _, _ = project_to_polyline(nodes, pts)
        nodes = _bin_centroids(pts, t, n_nodes)

    nodes = smooth_polyline(resample_polyline(nodes, n_nodes), passes=2)
    return _extend_to_polygon_ends(nodes, poly)


def _bin_centroids(pts: np.ndarray, t: np.ndarray, n_nodes: int) -> np.ndarray:
    """Mean position of the points in each of ``n_nodes`` bins of ``t``."""
    edges = np.linspace(t.min(), t.max(), n_nodes + 1)
    idx = np.clip(np.searchsorted(edges, t) - 1, 0, n_nodes - 1)
    nodes, keep = [], []
    for k in range(n_nodes):
        m = idx == k
        if m.sum() >= 3:
            nodes.append(pts[m].mean(axis=0))
            keep.append(k)
    nodes = np.asarray(nodes)
    if len(nodes) < 2:  # degenerate - fall back to the overall extent
        order = np.argsort(t)
        return pts[order][[0, -1]]
    return nodes


def _extend_to_polygon_ends(nodes: np.ndarray, poly) -> np.ndarray:
    """Push the first/last node out to the polygon edge.

    Binning always loses half a bin at each end, which would leave the upstream
    and downstream extremes of the river without a centerline to project onto.
    """
    out = nodes.copy()
    for end, nxt in ((0, 1), (-1, -2)):
        d = out[end] - out[nxt]
        n = np.hypot(*d)
        if n < 1e-9:
            continue
        d = d / n
        step = n * 0.05
        p = out[end].copy()
        for _ in range(60):
            q = p + d * step
            if not shapely.contains_xy(poly, q[0], q[1]):
                break
            p = q
        out[end] = p
    return out


# --------------------------------------------------------------------------- #
# Stage interpolation
# --------------------------------------------------------------------------- #


@dataclass
class RiverStage:
    """Water-surface elevation over the river polygon.

    ``centerline`` is oriented so that arc length increases downstream, i.e. so
    that stage decreases with arc length.
    """

    centerline: np.ndarray          # (M, 2)
    s_gauge: np.ndarray             # (G,) arc length of each gauge
    stage_gauge: np.ndarray         # (G,) observed stage
    interpolator: object            # callable s -> stage
    polygon: Optional[object] = None

    # ------------------------------------------------------------------ #

    def stage_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Interpolated stage at arbitrary points (extrapolated by the end values)."""
        pts = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        if len(pts) == 0:
            return np.zeros(0)
        s, _, _ = project_to_polyline(self.centerline, pts)
        s = np.clip(s, self.s_gauge.min(), self.s_gauge.max())
        return np.asarray(self.interpolator(s), dtype=float)

    def distance_to_center(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pts = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        _, d, _ = project_to_polyline(self.centerline, pts)
        return d

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.polygon is None:
            return np.zeros(np.size(x), dtype=bool)
        return shapely.contains_xy(self.polygon, np.asarray(x), np.asarray(y))

    def sample_river(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random points inside the river polygon and their stage."""
        rng = rng or np.random.default_rng()
        if self.polygon is None:
            return np.zeros((0, 2)), np.zeros(0)
        pts = sample_points_in_polygon(self.polygon, n, rng)
        return pts, self.stage_at(pts[:, 0], pts[:, 1])


def build_river_stage(
    polygon,
    gauge_xy: np.ndarray,
    gauge_stage: np.ndarray,
    centerline: Optional[np.ndarray] = None,
    monotonic: bool = True,
    n_nodes: int = 60,
    rng: Optional[np.random.Generator] = None,
) -> RiverStage:
    """Fit the along-stream stage profile.

    With ``monotonic=True`` the gauge values are first passed through a
    decreasing isotonic regression, so the interpolated water surface never
    slopes uphill in the downstream direction even if the gauge readings are
    noisy or were taken at slightly different times.
    """
    gauge_xy = np.asarray(gauge_xy, dtype=float).reshape(-1, 2)
    gauge_stage = np.asarray(gauge_stage, dtype=float).ravel()
    if len(gauge_xy) == 0:
        raise ValueError("no river gauges supplied")

    if centerline is None:
        centerline = polygon_centerline(polygon, n_nodes=n_nodes, rng=rng)
    centerline = np.asarray(centerline, dtype=float)

    s, _, _ = project_to_polyline(centerline, gauge_xy)

    # Orient the centerline downstream (stage must fall as s grows).
    if len(s) > 1 and np.polyfit(s, gauge_stage, 1)[0] > 0:
        centerline = centerline[::-1].copy()
        s, _, _ = project_to_polyline(centerline, gauge_xy)

    order = np.argsort(s)
    s, z = s[order], gauge_stage[order]

    # Average duplicate stations at the same chainage.
    s, inv = np.unique(np.round(s, 6), return_inverse=True)
    z = np.bincount(inv, weights=z) / np.bincount(inv)

    if monotonic and len(s) >= 3:
        from sklearn.isotonic import IsotonicRegression

        z = IsotonicRegression(increasing=False).fit_transform(s, z)

    if len(s) == 1:
        const = float(z[0])
        interp = lambda q: np.full(np.shape(q), const)  # noqa: E731
    elif len(s) == 2:
        interp = lambda q: np.interp(q, s, z)  # noqa: E731
    else:
        interp = PchipInterpolator(s, z, extrapolate=True)

    return RiverStage(
        centerline=centerline,
        s_gauge=s,
        stage_gauge=z,
        interpolator=interp,
        polygon=polygon,
    )


__all__ = [
    "RiverStage",
    "build_river_stage",
    "polygon_centerline",
    "project_to_polyline",
    "resample_polyline",
    "polyline_arclength",
    "sample_points_in_polygon",
]
