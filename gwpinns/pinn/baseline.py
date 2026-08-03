"""Architecture 1 -- baseline inverse PINN.

Two fully connected networks,

    H = N_h(X, Y, Z, T)        scaled hydraulic head
    K = N_k(X, Y, Z)           hydraulic conductivity, m/d

trained on data misfit plus the **second-order** transient flow residual

    r = d_T H - alpha * sum_i mu_i^2 d_i( K d_i H ) - beta * W .

The divergence is formed by nested autograd -- first ``d_i H``, then
``d_i (K d_i H)`` -- so the ``grad K . grad h`` cross term is captured exactly
without ever writing it out.  That is also this architecture's weakness at a
fault: ``K`` jumps by orders of magnitude across the zone, so ``d_i K`` is
near-singular there and the residual it produces is enormous.  Quantifying that
failure is the point of comparing it against the other two.
"""

from __future__ import annotations

import torch

from ..config import BenchmarkConfig
from .base import BasePINN
from .derivatives import grad
from .forcing import ForcingTerm
from .networks import MLP, LogConductivityNet
from .scaling import Scaling

__all__ = ["BaselinePINN"]


class BaselinePINN(BasePINN):
    """Standard fully connected inverse PINN predicting ``h`` and ``K``."""

    def __init__(
        self,
        cfg: BenchmarkConfig,
        scaling: Scaling,
        forcing: ForcingTerm | None = None,
        head_width: int = 96,
        head_depth: int = 5,
        head_fourier: int = 64,
        head_fourier_sigma: float = 2.0,
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
        self.head_net = MLP(
            4, 1, width=head_width, depth=head_depth,
            fourier_features=head_fourier, fourier_sigma=head_fourier_sigma,
        )
        self.k_net = LogConductivityNet(
            3, width=k_width, depth=k_depth,
            fourier_features=k_fourier, fourier_sigma=k_fourier_sigma,
            log_k_min=log_k_min, log_k_max=log_k_max, log_k_init=log_k_init,
        )

    # -- fields ---------------------------------------------------------------- #
    def head_scaled(self, u: torch.Tensor) -> torch.Tensor:
        return self.head_net(u)

    def conductivity(self, u3: torch.Tensor) -> torch.Tensor:
        return self.k_net(u3)

    # -- physics --------------------------------------------------------------- #
    def pde_residual(self, u: torch.Tensor) -> torch.Tensor:
        """Dimensionless residual of ``Ss dh/dt = div(K grad h) + W``."""
        head = self.head_scaled(u)
        gradient = grad(head, u)                      # (N, 4): d_X, d_Y, d_Z, d_T
        k = self.conductivity(u[:, :3])               # m/d

        # Interior flux components K * d_i H, then their divergence.
        divergence = None
        for axis in range(3):
            flux = k * gradient[:, axis : axis + 1]
            term = (self.mu[axis] ** 2) * grad(flux, u)[:, axis : axis + 1]
            divergence = term if divergence is None else divergence + term

        source = self.source(u, head)
        residual = gradient[:, 3:4] - self.alpha * divergence - source
        return residual / self.residual_scale(source)

    def pde_losses(self, batch) -> dict[str, torch.Tensor]:
        u = self.unit_input(batch.x, batch.y, batch.z, batch.t)
        return {"pde": torch.mean(self.pde_residual(u) ** 2)}

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
                gradient = grad(self.head_scaled(u), u)
                # Weight by mu so the penalty is on the *flux*, putting the thin
                # vertical direction on the same footing as the horizontal ones.
                flux = self.mu[face.axis] * gradient[:, face.axis : face.axis + 1]
                term = torch.mean(flux**2)
                total = term if total is None else total + term
                count += 1
            if count:
                losses["neumann"] = total / count

        return losses
