"""Architecture 2 -- mixed-variable (velocity-head) inverse PINN.

The network predicts the Darcy flux as an *independent output* alongside the
head,

    (H, U_x, U_y, U_z) = N(X, Y, Z, T),     U_i = q_i / q0,   q0 = K0 dh / L0
    K                  = N_k(X, Y, Z)

and the second-order equation is split into two first-order statements:

    Darcy       :  U_i + (K/K0) mu_i d_i H            = 0
    continuity  :  d_T H + alpha K0 sum_i mu_i d_i U_i - beta W = 0

Why this matters at a fault
---------------------------
The head is continuous across the fault but its *gradient* is not, and ``K``
itself jumps.  The baseline model has to differentiate that jump twice.  Here
each residual contains only first derivatives, and -- crucially -- the quantity
that is physically smooth across the interface, the normal flux ``q_n``, is
represented directly by a network output rather than reconstructed from a
product of two discontinuous factors.  The no-flow boundaries also become
algebraic (``U_n = 0``) instead of differential.
"""

from __future__ import annotations

import torch

from ..config import BenchmarkConfig
from .base import BasePINN
from .derivatives import grad
from .forcing import ForcingTerm
from .networks import MLP, LogConductivityNet
from .scaling import Scaling

__all__ = ["MixedPINN"]


class MixedPINN(BasePINN):
    """Head + Darcy-velocity network with first-order physics residuals."""

    def __init__(
        self,
        cfg: BenchmarkConfig,
        scaling: Scaling,
        forcing: ForcingTerm | None = None,
        width: int = 96,
        depth: int = 5,
        fourier: int = 64,
        fourier_sigma: float = 2.0,
        k_width: int = 96,
        k_depth: int = 5,
        k_fourier: int = 64,
        k_fourier_sigma: float = 3.0,
        log_k_min: float = -4.0,
        log_k_max: float = 3.0,
        log_k_init: float = 0.0,
        residual_weighting: str = "source",
    ):
        super().__init__(cfg, scaling, forcing, residual_weighting, log_k_prior=log_k_init)
        # One trunk with four outputs: head and the three flux components.
        self.field_net = MLP(
            4, 4, width=width, depth=depth,
            fourier_features=fourier, fourier_sigma=fourier_sigma,
        )
        self.k_net = LogConductivityNet(
            3, width=k_width, depth=k_depth,
            fourier_features=k_fourier, fourier_sigma=k_fourier_sigma,
            log_k_min=log_k_min, log_k_max=log_k_max, log_k_init=log_k_init,
        )
        self.k_ref = float(scaling.k_ref)

    # -- fields ---------------------------------------------------------------- #
    def fields(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(H, U)`` with ``U`` of shape ``(N, 3)``."""
        out = self.field_net(u)
        return out[:, 0:1], out[:, 1:4]

    def head_scaled(self, u: torch.Tensor) -> torch.Tensor:
        return self.field_net(u)[:, 0:1]

    def flux_scaled(self, u: torch.Tensor) -> torch.Tensor:
        return self.field_net(u)[:, 1:4]

    def conductivity(self, u3: torch.Tensor) -> torch.Tensor:
        return self.k_net(u3)

    @torch.no_grad()
    def darcy_flux(self, u: torch.Tensor) -> torch.Tensor:
        """Darcy flux in m/d, shape ``(N, 3)``."""
        return self.flux_scaled(u) * self.scaling.q_ref

    # -- physics --------------------------------------------------------------- #
    def darcy_residuals(self, u: torch.Tensor) -> torch.Tensor:
        """Residual of ``U_i + (K/K0) mu_i d_i H`` for each axis, shape ``(N, 3)``."""
        head, flux = self.fields(u)
        gradient = grad(head, u)
        k = self.conductivity(u[:, :3])
        terms = [
            flux[:, axis : axis + 1]
            + (k / self.k_ref) * self.mu[axis] * gradient[:, axis : axis + 1]
            for axis in range(3)
        ]
        return torch.cat(terms, dim=1)

    def continuity_residual(self, u: torch.Tensor) -> torch.Tensor:
        """Residual of ``d_T H + alpha K0 sum_i mu_i d_i U_i - beta W``."""
        head, flux = self.fields(u)
        gradient = grad(head, u)

        divergence = None
        for axis in range(3):
            term = self.mu[axis] * grad(flux[:, axis : axis + 1], u)[:, axis : axis + 1]
            divergence = term if divergence is None else divergence + term

        source = self.source(u, head)
        residual = gradient[:, 3:4] + self.alpha * self.k_ref * divergence - source
        return residual / self.residual_scale(source)

    def pde_losses(self, batch) -> dict[str, torch.Tensor]:
        u = self.unit_input(batch.x, batch.y, batch.z, batch.t)
        return {
            "pde": torch.mean(self.continuity_residual(u) ** 2),
            "darcy": torch.mean(self.darcy_residuals(u) ** 2),
        }

    # -- boundaries ------------------------------------------------------------ #
    def boundary_losses(self, dirichlet=None, neumann=None) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        if dirichlet is not None:
            u = self.unit_input(dirichlet.x, dirichlet.y, dirichlet.z, dirichlet.t)
            pred = self.head_scaled(u)
            target = self.scaling.encode_head(dirichlet.value)
            losses["dirichlet"] = torch.mean((pred - target) ** 2)

        if neumann:
            total = None
            count = 0
            for face in neumann:
                u = self.unit_input(face.x, face.y, face.z, face.t)
                # Algebraic, not differential: the flux component *is* an output.
                normal_flux = self.flux_scaled(u)[:, face.axis : face.axis + 1]
                term = torch.mean(normal_flux**2)
                total = term if total is None else total + term
                count += 1
            if count:
                losses["neumann"] = total / count

        return losses
