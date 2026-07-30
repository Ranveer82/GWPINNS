"""Quasi-3D multilayer groundwater flow.

For every aquifer layer ``l``:

.. math::

    S_l \\frac{\\partial h_l}{\\partial t}
      = \\nabla\\!\\cdot\\!\\big(T_l A \\nabla h_l\\big)
      + C_{l-1,l}\\,(h_{l-1}-h_l)
      + C_{l,l+1}\\,(h_{l+1}-h_l)
      + R_l + \\Gamma^{riv}_l

This is the same conceptual model MODFLOW uses: each aquifer is depth-integrated
into a 2-D transmissivity ``T_l``, and the aquitards between them are collapsed
into vertical leakance coefficients ``C``. ``A`` is the fault anisotropy tensor
(:mod:`gwpinn.geo.faults`), and the river exchange :math:`\\Gamma^{riv}` is a
Robin term, i.e. the RIV package.

The top layer is unconfined by default, so :math:`T_0 = K_0\\,(h_0 - z^{bot}_0)`
and the equation is nonlinear in the head - which is exactly the case where a
mesh-free PINN is convenient, since no outer Picard/Newton iteration is needed.

**Identifiability.** In steady state the equation contains no ``S`` at all, so a
storage coefficient cannot be recovered from steady heads by any method. In that
regime ``S`` is constrained only by the pumping-test points and their variogram;
run in ``transient`` mode if ``S`` is to be inferred from the flow field itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from gwpinn.geo.faults import FaultField
from gwpinn.physics.operators import divergence, grad, smooth_clamp


@dataclass
class LayerElevations:
    """Layer geometry sampled at the collocation points."""

    top: torch.Tensor          # (N,)   ground surface / top of layer 0
    bottoms: torch.Tensor      # (N, L) bottom of each layer, decreasing

    def upper(self, layer: int) -> torch.Tensor:
        return self.top if layer == 0 else self.bottoms[:, layer - 1]

    def bottom(self, layer: int) -> torch.Tensor:
        return self.bottoms[:, layer]

    def full_thickness(self, layer: int) -> torch.Tensor:
        return self.upper(layer) - self.bottoms[:, layer]


@dataclass
class Sources:
    """Sink/source terms sampled at the collocation points."""

    recharge: Optional[torch.Tensor] = None     # (N,)   m/d, into layer 0
    river_stage: Optional[torch.Tensor] = None  # (N,)   m a.s.l.
    river_mask: Optional[torch.Tensor] = None   # (N,)   1 inside the river, else 0
    wells: Optional[torch.Tensor] = None        # (N, L) m/d (negative = abstraction)


class GroundwaterFlow(nn.Module):
    """PDE residual, plus the trainable coefficients that live in the equation."""

    def __init__(
        self,
        n_layers: int,
        faults: Optional[FaultField] = None,
        leakance: Optional[Sequence[float]] = None,
        leakance_default: float = 1e-3,
        train_leakance: bool = True,
        river_conductance: float = 5e-2,
        train_river_conductance: bool = True,
        unconfined_top: bool = True,
        min_thickness: float = 1.0,
        residual_scale: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.faults = faults
        self.unconfined_top = bool(unconfined_top)
        self.min_thickness = float(min_thickness)
        self.residual_scale = float(residual_scale)

        # Vertical leakance of each aquitard, trained in log space so it stays
        # positive and can move across orders of magnitude.
        n_iface = max(self.n_layers - 1, 0)
        vals = list(leakance) if leakance else []
        if len(vals) < n_iface:
            vals += [leakance_default] * (n_iface - len(vals))
        log_c = torch.log(torch.as_tensor(vals[:n_iface], dtype=dtype).clamp_min(1e-12))
        if n_iface == 0:
            log_c = torch.zeros(0, dtype=dtype)
        self.log_leakance = nn.Parameter(log_c, requires_grad=train_leakance)

        log_riv = torch.log(torch.tensor(max(river_conductance, 1e-12), dtype=dtype))
        self.log_river_cond = nn.Parameter(log_riv, requires_grad=train_river_conductance)

    # ------------------------------------------------------------------ #

    @property
    def leakance(self) -> torch.Tensor:
        return torch.exp(self.log_leakance)

    @property
    def river_conductance(self) -> torch.Tensor:
        return torch.exp(self.log_river_cond)

    # ------------------------------------------------------------------ #

    def saturated_thickness(
        self, h: torch.Tensor, elev: LayerElevations
    ) -> torch.Tensor:
        """Thickness of every layer, ``(N, L)``.

        The unconfined top layer follows the water table and is smoothly clamped
        between ``min_thickness`` and the full geometric thickness; deeper layers
        are confined and keep their geometric thickness.
        """
        cols = []
        for l in range(self.n_layers):
            full = elev.full_thickness(l).clamp_min(self.min_thickness)
            if l == 0 and self.unconfined_top:
                b = smooth_clamp(
                    h[:, 0] - elev.bottom(0), self.min_thickness, full, beta=1.0
                )
            else:
                b = full
            cols.append(b)
        return torch.stack(cols, dim=1)

    def transmissivity(
        self, K: torch.Tensor, h: torch.Tensor, elev: LayerElevations
    ) -> torch.Tensor:
        """``T = K * b``, shape ``(N, L)``."""
        return K * self.saturated_thickness(h, elev)

    # ------------------------------------------------------------------ #

    def residual(
        self,
        xy: torch.Tensor,
        h: torch.Tensor,
        K: torch.Tensor,
        elev: LayerElevations,
        sources: Optional[Sources] = None,
        S: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        anis: Optional[torch.Tensor] = None,
        scaled: bool = True,
    ) -> torch.Tensor:
        """Residual of the flow equation at every point and layer, ``(N, L)``.

        ``xy`` (and ``t``, when transient) must carry ``requires_grad``.
        """
        T = self.transmissivity(K, h, elev)

        if anis is None:
            anis = (
                self.faults.anisotropy(xy)
                if (self.faults is not None and self.faults.has_faults)
                else None
            )

        res_cols: List[torch.Tensor] = []
        for l in range(self.n_layers):
            gh = grad(h[:, l], xy)                                  # (N, 2)
            if anis is not None:
                gh = torch.einsum("nij,nj->ni", anis, gh)
            flux = T[:, l : l + 1] * gh
            r = divergence(flux, xy)

            # Vertical leakage with the layers above and below.
            if l > 0:
                r = r + self.leakance[l - 1] * (h[:, l - 1] - h[:, l])
            if l < self.n_layers - 1:
                r = r + self.leakance[l] * (h[:, l + 1] - h[:, l])

            if sources is not None:
                if l == 0 and sources.recharge is not None:
                    r = r + sources.recharge
                if l == 0 and sources.river_mask is not None and sources.river_stage is not None:
                    r = r + self.river_conductance * sources.river_mask * (
                        sources.river_stage - h[:, 0]
                    )
                if sources.wells is not None:
                    r = r + sources.wells[:, l]

            if t is not None and S is not None:
                r = r - S[:, l] * grad(h[:, l], t)[:, 0]

            res_cols.append(r)

        res = torch.stack(res_cols, dim=1)
        return res / self.residual_scale if scaled else res

    # ------------------------------------------------------------------ #

    def darcy_flux(
        self,
        xy: torch.Tensor,
        h: torch.Tensor,
        K: torch.Tensor,
        elev: LayerElevations,
        anis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Depth-integrated Darcy flux ``q = -T A grad(h)``, shape ``(N, L, 2)``."""
        T = self.transmissivity(K, h, elev)
        if anis is None and self.faults is not None and self.faults.has_faults:
            anis = self.faults.anisotropy(xy)
        out = []
        for l in range(self.n_layers):
            gh = grad(h[:, l], xy)
            if anis is not None:
                gh = torch.einsum("nij,nj->ni", anis, gh)
            out.append(-T[:, l : l + 1] * gh)
        return torch.stack(out, dim=1)

    def extra_repr(self) -> str:
        return (
            f"n_layers={self.n_layers}, unconfined_top={self.unconfined_top}, "
            f"residual_scale={self.residual_scale:.3g}"
        )


def estimate_residual_scale(
    t_ref: float, h_scale: float, length_scale: float, recharge: float = 0.0
) -> float:
    """Characteristic magnitude of the flow equation, used to non-dimensionalise.

    Without this the residual for a realistic aquifer sits many orders of
    magnitude away from the data losses and no fixed set of loss weights works.
    Recharge is included because in a shallow aquifer it, not the lateral flux
    divergence, sets the size of the terms being balanced.
    """
    lateral = t_ref * h_scale / max(length_scale**2, 1e-12)
    return float(max(lateral + abs(recharge), 1e-12))


__all__ = [
    "GroundwaterFlow",
    "LayerElevations",
    "Sources",
    "estimate_residual_scale",
]
