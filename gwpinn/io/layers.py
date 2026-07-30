"""Aquifer layer geometry: top surface, bottom surfaces, overlap correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from gwpinn.io.raster import Raster


def correct_layer_overlaps(
    top: np.ndarray,
    bottoms: Sequence[np.ndarray],
    min_thickness: float = 1.0,
) -> Tuple[np.ndarray, List[np.ndarray], dict]:
    """Enforce a strictly decreasing stack of elevations.

    Input surfaces come from independent interpolations and routinely cross each
    other, which would give negative layer thickness. Each bottom is pushed down
    so that every layer is at least ``min_thickness`` thick::

        z_bot[l] = min(z_bot[l], upper[l] - min_thickness)

    where ``upper[0] = top`` and ``upper[l] = z_bot[l-1]``. Corrections
    therefore cascade downwards and the top surface is never modified.

    Returns the (unchanged) top, the corrected bottoms, and a report giving the
    number and magnitude of corrections per layer.
    """
    top = np.asarray(top, dtype=float)
    corrected: List[np.ndarray] = []
    report = {"min_thickness": float(min_thickness), "layers": []}

    upper = top
    for l, raw in enumerate(bottoms):
        b = np.asarray(raw, dtype=float).copy()

        # Only compare where both surfaces are defined.
        valid = np.isfinite(upper) & np.isfinite(b)
        limit = upper - min_thickness
        violates = valid & (b > limit)

        shift = np.zeros_like(b)
        shift[violates] = b[violates] - limit[violates]
        b[violates] = limit[violates]

        # A bottom with no data inherits a nominal thickness from the surface
        # above so the stack stays well defined everywhere the top is known.
        gap = np.isfinite(upper) & ~np.isfinite(b)
        b[gap] = upper[gap] - max(min_thickness, 1.0)

        report["layers"].append(
            {
                "layer": l,
                "n_cells_corrected": int(violates.sum()),
                "fraction_corrected": float(violates.mean()) if violates.size else 0.0,
                "max_shift_m": float(shift.max()) if violates.any() else 0.0,
                "mean_shift_m": float(shift[violates].mean()) if violates.any() else 0.0,
                "n_cells_filled": int(gap.sum()),
            }
        )

        corrected.append(b)
        upper = b

    return top, corrected, report


@dataclass
class LayerGeometry:
    """Layer top/bottom elevations resampled onto a single common grid."""

    top: Raster                 # DTM, top of layer 0
    bottoms: List[Raster]       # one per layer, top -> bottom
    report: dict                # overlap-correction diagnostics

    @property
    def n_layers(self) -> int:
        return len(self.bottoms)

    def thickness(self, layer: int) -> np.ndarray:
        """Geometric thickness of ``layer`` (m)."""
        upper = self.top.values if layer == 0 else self.bottoms[layer - 1].values
        return upper - self.bottoms[layer].values

    def upper_surface(self, layer: int) -> np.ndarray:
        return self.top.values if layer == 0 else self.bottoms[layer - 1].values

    def sample_top(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.top.sample(x, y)

    def sample_bottom(self, layer: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.bottoms[layer].sample(x, y)

    def sample_upper(self, layer: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if layer == 0:
            return self.sample_top(x, y)
        return self.sample_bottom(layer - 1, x, y)

    def sample_thickness(self, layer: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.sample_upper(layer, x, y) - self.sample_bottom(layer, x, y)

    def summary(self) -> str:
        lines = ["Layer geometry:"]
        for l in range(self.n_layers):
            th = self.thickness(l)
            th = th[np.isfinite(th)]
            rep = self.report["layers"][l]
            lines.append(
                f"  layer {l}: thickness min={th.min():8.2f} "
                f"mean={th.mean():8.2f} max={th.max():8.2f} m | "
                f"overlap-corrected cells={rep['n_cells_corrected']} "
                f"({100 * rep['fraction_corrected']:.2f}%), "
                f"max shift={rep['max_shift_m']:.2f} m"
            )
        return "\n".join(lines)


def build_layer_geometry(
    dtm: Raster,
    bottom_rasters: Sequence[Raster],
    grid: Optional[Raster] = None,
    min_thickness: float = 1.0,
) -> LayerGeometry:
    """Resample all elevation surfaces onto ``grid`` and fix overlaps."""
    target = grid if grid is not None else dtm

    top = dtm.resample_to(target) if dtm is not target else dtm
    bots = [b.resample_to(target) for b in bottom_rasters]

    top_vals, bot_vals, report = correct_layer_overlaps(
        top.values, [b.values for b in bots], min_thickness=min_thickness
    )

    return LayerGeometry(
        top=target.copy_like(top_vals),
        bottoms=[target.copy_like(v) for v in bot_vals],
        report=report,
    )


__all__ = ["LayerGeometry", "build_layer_geometry", "correct_layer_overlaps"]
