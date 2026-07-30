"""Vector (shapefile) input.

Field names in real-world shapefiles are inconsistent and DBF truncates them to
ten characters, so attributes are resolved through a case-insensitive alias
lookup rather than by exact name.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union

# --------------------------------------------------------------------------- #
# Attribute resolution
# --------------------------------------------------------------------------- #

#: Canonical name -> accepted spellings (matched case-insensitively).
ALIASES: Dict[str, Sequence[str]] = {
    "head": ("head", "gwhead", "gw_head", "h", "wl", "waterlevel", "water_leve",
             "level", "obs_head", "head_m", "gwl"),
    "stage": ("stage", "wl", "waterlevel", "water_leve", "level", "riverstage",
              "river_stag", "h", "stage_m"),
    "layer": ("layer", "lay", "lyr", "aquifer", "layer_id", "layerid"),
    "time": ("time", "t", "day", "days", "date", "tstep"),
    "weight": ("weight", "wt", "w", "sigma", "std"),
    "perm": ("perm", "permeabili", "permeability", "k", "kfault", "barrier",
             "perm_flag", "trainable"),
    "T": ("t", "transmissi", "transmissivity", "trans", "t_m2d", "tval"),
    "S": ("s", "storativit", "storativity", "storage", "storagecoe", "sval",
          "stor_coef"),
    "bctype": ("bctype", "bc_type", "type", "bc"),
    "value": ("value", "val", "head", "h"),
    "cond": ("cond", "conductanc", "conductance", "c"),
    "name": ("name", "id", "station", "site", "label"),
}


def resolve_field(columns: Iterable[str], canonical: str) -> Optional[str]:
    """Find the column in ``columns`` matching ``canonical``, or ``None``."""
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    # Exact-ish match first, then alias list, then prefix match.
    if canonical.lower() in lower:
        return lower[canonical.lower()]
    for alias in ALIASES.get(canonical, ()):  # ordered by preference
        if alias in lower:
            return lower[alias]
    for c in cols:
        if c.lower().startswith(canonical.lower()):
            return c
    return None


def _column(gdf: gpd.GeoDataFrame, canonical: str) -> Optional[np.ndarray]:
    name = resolve_field([c for c in gdf.columns if c != "geometry"], canonical)
    if name is None:
        return None
    return gdf[name].to_numpy()


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #


@dataclass
class PointSet:
    """Point observations with an arbitrary set of named attributes."""

    x: np.ndarray
    y: np.ndarray
    attrs: Dict[str, np.ndarray] = field(default_factory=dict)
    crs: Optional[object] = None

    def __len__(self) -> int:
        return int(self.x.size)

    def get(self, name: str, default: Optional[float] = None) -> Optional[np.ndarray]:
        if name in self.attrs:
            return self.attrs[name]
        if default is None:
            return None
        return np.full(len(self), float(default))

    def subset(self, mask: np.ndarray) -> "PointSet":
        mask = np.asarray(mask, dtype=bool)
        return PointSet(
            self.x[mask],
            self.y[mask],
            {k: v[mask] for k, v in self.attrs.items()},
            self.crs,
        )

    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])


@dataclass
class PolylineSet:
    """Polylines (e.g. faults) with per-feature attributes."""

    lines: List[np.ndarray]  # each (n_i, 2)
    attrs: Dict[str, np.ndarray] = field(default_factory=dict)
    crs: Optional[object] = None

    def __len__(self) -> int:
        return len(self.lines)

    def segments(self) -> np.ndarray:
        """All segments as an (n_seg, 4) array [x0, y0, x1, y1]."""
        segs = []
        for ln in self.lines:
            if len(ln) >= 2:
                segs.append(np.column_stack([ln[:-1], ln[1:]]))
        if not segs:
            return np.zeros((0, 4))
        return np.vstack(segs)

    def segment_owner(self) -> np.ndarray:
        """Index of the parent polyline for each segment returned by ``segments``."""
        owner = []
        for i, ln in enumerate(self.lines):
            if len(ln) >= 2:
                owner.append(np.full(len(ln) - 1, i, dtype=int))
        if not owner:
            return np.zeros(0, dtype=int)
        return np.concatenate(owner)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #


def read_points(
    path: str | pathlib.Path, fields: Sequence[str] = ()
) -> PointSet:
    """Read a point shapefile, pulling out the requested canonical fields."""
    gdf = gpd.read_file(str(path))
    geom = gdf.geometry
    # Accept multipoints by taking the representative point.
    pts = geom.representative_point()
    x = pts.x.to_numpy(dtype=float)
    y = pts.y.to_numpy(dtype=float)

    attrs: Dict[str, np.ndarray] = {}
    for name in fields:
        col = _column(gdf, name)
        if col is None:
            continue
        try:
            attrs[name] = np.asarray(col, dtype=float)
        except (TypeError, ValueError):
            attrs[name] = np.asarray(col)

    return PointSet(x, y, attrs, gdf.crs)


def read_polylines(
    path: str | pathlib.Path, fields: Sequence[str] = ()
) -> PolylineSet:
    """Read a line shapefile. MultiLineStrings are exploded into parts."""
    gdf = gpd.read_file(str(path))

    lines: List[np.ndarray] = []
    keep_idx: List[int] = []
    for i, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
        for part in parts:
            if isinstance(part, LineString) and len(part.coords) >= 2:
                lines.append(np.asarray(part.coords, dtype=float)[:, :2])
                keep_idx.append(i)

    idx = np.asarray(keep_idx, dtype=int)
    attrs: Dict[str, np.ndarray] = {}
    for name in fields:
        col = _column(gdf, name)
        if col is None:
            continue
        col = np.asarray(col)
        try:
            attrs[name] = np.asarray(col[idx], dtype=float)
        except (TypeError, ValueError):
            attrs[name] = col[idx]

    return PolylineSet(lines, attrs, gdf.crs)


def read_polygon(path: str | pathlib.Path):
    """Read a polygon shapefile and return the union of all features."""
    gdf = gpd.read_file(str(path))
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        raise ValueError(f"no polygon geometry in {path}")
    merged = unary_union(geoms)
    if not isinstance(merged, (Polygon, MultiPolygon)):
        raise ValueError(f"{path} does not contain polygons")
    return merged, gdf.crs


def densify(line: np.ndarray, spacing: float) -> np.ndarray:
    """Insert vertices so no segment of ``line`` is longer than ``spacing``."""
    if len(line) < 2 or spacing <= 0:
        return line
    out = [line[0]]
    for a, b in zip(line[:-1], line[1:]):
        d = float(np.hypot(*(b - a)))
        n = max(1, int(np.ceil(d / spacing)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out)


def simplify_polylines(pls: PolylineSet, tolerance: float) -> PolylineSet:
    """Douglas-Peucker simplification, keeping attributes.

    Fault distances are evaluated against every segment on every collocation
    point, so keeping the segment count down matters for runtime.
    """
    if tolerance <= 0:
        return pls
    lines = []
    for ln in pls.lines:
        simp = np.asarray(LineString(ln).simplify(tolerance).coords, dtype=float)
        lines.append(simp if len(simp) >= 2 else ln)
    return PolylineSet(lines, dict(pls.attrs), pls.crs)


__all__ = [
    "PointSet",
    "PolylineSet",
    "read_points",
    "read_polylines",
    "read_polygon",
    "resolve_field",
    "densify",
    "simplify_polylines",
]
