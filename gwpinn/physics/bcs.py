"""Outer-boundary conditions.

Three kinds are supported on the domain outline:

``noflow``
    ``T A grad(h) . n = 0``. The default everywhere no other condition is given -
    a groundwater divide or an impermeable contact.
``head``
    ``h = value`` (Dirichlet / prescribed head).
``ghb``
    ``T A grad(h) . n = C (value - h)`` (general head boundary): a Robin
    condition standing in for whatever lies beyond the modelled area.

Boundary segments are matched to the sampled outline points by nearest-neighbour
lookup, so a boundary shapefile need not share vertices with the domain polygon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gwpinn.physics.operators import grad

BC_TYPES = ("noflow", "head", "ghb")


@dataclass
class BoundaryConditions:
    """Boundary type / value / conductance resolved at sampled outline points."""

    kind: np.ndarray            # (M,) int codes: 0 noflow, 1 head, 2 ghb
    value: np.ndarray           # (M,) prescribed head (NaN where unused)
    conductance: np.ndarray     # (M,) GHB conductance (m^2/d per m of boundary)
    layers: Optional[np.ndarray] = None   # (M,) target layer, -1 = all

    @classmethod
    def all_noflow(cls, n: int) -> "BoundaryConditions":
        return cls(
            kind=np.zeros(n, dtype=int),
            value=np.full(n, np.nan),
            conductance=np.zeros(n),
            layers=np.full(n, -1, dtype=int),
        )

    def subset(self, mask: np.ndarray) -> "BoundaryConditions":
        mask = np.asarray(mask, dtype=bool)
        return BoundaryConditions(
            self.kind[mask],
            self.value[mask],
            self.conductance[mask],
            None if self.layers is None else self.layers[mask],
        )


def assign_boundary_conditions(
    boundary_xy: np.ndarray,
    bc_lines: Optional[Sequence[np.ndarray]] = None,
    bc_types: Optional[Sequence[str]] = None,
    bc_values: Optional[Sequence[float]] = None,
    bc_cond: Optional[Sequence[float]] = None,
    snap_distance: float = np.inf,
) -> BoundaryConditions:
    """Attach conditions from a boundary shapefile to sampled outline points."""
    n = len(boundary_xy)
    bc = BoundaryConditions.all_noflow(n)
    if not bc_lines:
        return bc

    from gwpinn.geo.river import project_to_polyline

    best = np.full(n, np.inf)
    for i, line in enumerate(bc_lines):
        line = np.asarray(line, dtype=float)[:, :2]
        if len(line) < 2:
            continue
        _, d, _ = project_to_polyline(line, boundary_xy)
        take = (d < best) & (d <= snap_distance)
        if not take.any():
            continue
        best[take] = d[take]

        kind = "noflow"
        if bc_types is not None and i < len(bc_types):
            raw = bc_types[i]
            kind = str(raw).lower().strip() if raw is not None else "noflow"
            kind = kind if kind in BC_TYPES else "noflow"
        bc.kind[take] = BC_TYPES.index(kind)

        if bc_values is not None and i < len(bc_values):
            bc.value[take] = float(bc_values[i])
        if bc_cond is not None and i < len(bc_cond):
            bc.conductance[take] = float(bc_cond[i])

    return bc


# --------------------------------------------------------------------------- #


def boundary_residual(
    xy: torch.Tensor,
    normals: torch.Tensor,
    h: torch.Tensor,
    T: torch.Tensor,
    bc: BoundaryConditions,
    anis: Optional[torch.Tensor] = None,
    flux_scale: float = 1.0,
    head_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Residuals of every boundary condition present.

    Returns a dict with the entries that actually occur among ``"noflow"``,
    ``"head"`` and ``"ghb"``; each is a flat tensor of residuals already scaled
    to O(1).
    """
    device, dtype = xy.device, xy.dtype
    kind = torch.as_tensor(bc.kind, device=device)
    out: Dict[str, torch.Tensor] = {}

    n_layers = h.shape[1]
    normal_flux: List[torch.Tensor] = []
    for l in range(n_layers):
        gh = grad(h[:, l], xy)
        if anis is not None:
            gh = torch.einsum("nij,nj->ni", anis, gh)
        q = -T[:, l : l + 1] * gh
        normal_flux.append((q * normals).sum(dim=1))
    qn = torch.stack(normal_flux, dim=1)              # (M, L)

    m_noflow = kind == 0
    if bool(m_noflow.any()):
        out["noflow"] = (qn[m_noflow] / flux_scale).reshape(-1)

    m_head = kind == 1
    if bool(m_head.any()):
        target = torch.as_tensor(bc.value, device=device, dtype=dtype)[m_head]
        ok = torch.isfinite(target)
        if bool(ok.any()):
            diff = h[m_head][ok] - target[ok][:, None]
            out["head"] = (diff / head_scale).reshape(-1)

    m_ghb = kind == 2
    if bool(m_ghb.any()):
        target = torch.as_tensor(bc.value, device=device, dtype=dtype)[m_ghb]
        cond = torch.as_tensor(bc.conductance, device=device, dtype=dtype)[m_ghb]
        ok = torch.isfinite(target)
        if bool(ok.any()):
            expected = cond[ok][:, None] * (target[ok][:, None] - h[m_ghb][ok])
            out["ghb"] = ((qn[m_ghb][ok] - expected) / flux_scale).reshape(-1)

    return out


__all__ = [
    "BoundaryConditions",
    "assign_boundary_conditions",
    "boundary_residual",
    "BC_TYPES",
]
