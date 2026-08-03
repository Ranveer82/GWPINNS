"""Architecture 3 -- conservative domain-decomposition PINN (cPINN).

The fault plane splits the aquifer into a western and an eastern block, each
carrying its **own** head network and its **own** conductivity network::

    block w:  H_w = N_w(X,Y,Z,T),  K_w = N_kw(X,Y,Z)     for d(x,y) < -w/2
    block e:  H_e = N_e(X,Y,Z,T),  K_e = N_ke(X,Y,Z)     for d(x,y) > +w/2

Neither network ever has to represent the discontinuity: it lives entirely in
the coupling between them, and the finite-width fault zone is *homogenised
away* into a plane condition.

Interface conditions
--------------------
Flux continuity is unconditional -- mass cannot accumulate in a zero-thickness
plane::

    Q_n^w = Q_n^e                                                        (1)

For the head, note that **strict continuity cannot represent a barrier**: a
low-permeability fault exists precisely to sustain a head jump, and imposing
``H_w = H_e`` would force the model to explain that jump with a wildly wrong
conductivity field on either side.  The physically correct thin-feature
condition is a *leaky wall*::

    Q_n = Gamma * (H_w - H_e),   Gamma = C * L0 / K0,   C = K_fault / width  (2)

``Gamma -> 0`` recovers a perfect no-flow barrier and ``Gamma -> inf`` recovers
head continuity, so the single fitted field ``Gamma`` **is** the answer to
"barrier or conduit?".  It is inferred, not prescribed.

Condition (2) is imposed in the normalised form

    [ Gamma (H_w - H_e) - Q_n ] / (1 + Gamma)

which stays well conditioned at both limits: for small ``Gamma`` it reduces to
``Q_n = 0`` (barrier) and for large ``Gamma`` to ``H_w = H_e`` (conduit).

Setting ``interface_mode="continuity"`` restores the textbook ``H_w = H_e``
condition, which is retained deliberately so the study can demonstrate that it
fails on the barrier scenario.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import BenchmarkConfig
from .base import BasePINN
from .derivatives import grad
from .forcing import ForcingTerm
from .networks import (
    HEAD_FOURIER_SIGMA,
    K_FOURIER_SIGMA,
    MLP,
    FaultConductance,
    LogConductivityNet,
)
from .scaling import Scaling

__all__ = ["ConservativePINN"]


class ConservativePINN(BasePINN):
    """Two-block conservative PINN with an inferred fault leakance."""

    def __init__(
        self,
        cfg: BenchmarkConfig,
        scaling: Scaling,
        forcing: ForcingTerm | None = None,
        head_width: int = 80,
        head_depth: int = 5,
        head_fourier: int = 48,
        head_fourier_sigma=HEAD_FOURIER_SIGMA,
        k_width: int = 80,
        k_depth: int = 5,
        k_fourier: int = 48,
        k_fourier_sigma=K_FOURIER_SIGMA,
        log_k_min: float = -4.0,
        log_k_max: float = 3.0,
        log_k_init: float = 0.0,
        interface_mode: str = "conductance",
        gamma_mode: str = "field",
        residual_weighting: str = "source",
    ):
        super().__init__(cfg, scaling, forcing, residual_weighting, log_k_prior=log_k_init)
        if interface_mode not in {"conductance", "continuity"}:
            raise ValueError("interface_mode must be 'conductance' or 'continuity'")
        self.interface_mode = interface_mode

        def make_head() -> MLP:
            return MLP(
                4, 1, width=head_width, depth=head_depth,
                fourier_features=head_fourier, fourier_sigma=head_fourier_sigma,
            )

        def make_k() -> LogConductivityNet:
            return LogConductivityNet(
                3, width=k_width, depth=k_depth,
                fourier_features=k_fourier, fourier_sigma=k_fourier_sigma,
                log_k_min=log_k_min, log_k_max=log_k_max, log_k_init=log_k_init,
            )

        self.head_nets = nn.ModuleList([make_head(), make_head()])   # [west, east]
        self.k_nets = nn.ModuleList([make_k(), make_k()])
        self.gamma_net = FaultConductance(mode=gamma_mode)

        nx, ny = cfg.fault.normal
        dtype = torch.get_default_dtype()
        self.register_buffer("fault_normal", torch.tensor([nx, ny], dtype=dtype))
        self.fault_x0 = float(cfg.fault.x0)
        self.fault_y0 = float(cfg.fault.y0)
        self.fault_half_width = 0.5 * float(cfg.fault.width)
        self.k_ref = float(scaling.k_ref)

    # -- block routing --------------------------------------------------------- #
    def signed_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        nx, ny = self.fault_normal[0], self.fault_normal[1]
        return (x - self.fault_x0) * nx + (y - self.fault_y0) * ny

    def _west_mask(self, u: torch.Tensor) -> torch.Tensor:
        x, y, _, _ = self.physical(u)
        return (self.signed_distance(x, y) < 0.0)

    def head_scaled(self, u: torch.Tensor) -> torch.Tensor:
        """Routed head: evaluates whichever block's network owns the point."""
        west = self._west_mask(u)
        return torch.where(west, self.head_nets[0](u), self.head_nets[1](u))

    def conductivity(self, u3: torch.Tensor) -> torch.Tensor:
        """Routed conductivity. Fault-zone cells are handled by :meth:`fault_k_grid`."""
        s = self.scaling
        x = u3[:, 0:1] * s.a_x + s.x_c
        y = u3[:, 1:2] * s.a_y + s.y_c
        west = self.signed_distance(x, y) < 0.0
        return torch.where(west, self.k_nets[0](u3), self.k_nets[1](u3))

    def head_scaled_block(self, u: torch.Tensor, block: int) -> torch.Tensor:
        return self.head_nets[block](u)

    def conductivity_block(self, u3: torch.Tensor, block: int) -> torch.Tensor:
        return self.k_nets[block](u3)

    # -- physics within each block --------------------------------------------- #
    def _block_residual(self, u: torch.Tensor, block: int) -> torch.Tensor:
        head = self.head_scaled_block(u, block)
        gradient = grad(head, u)
        k = self.conductivity_block(u[:, :3], block)

        divergence = None
        for axis in range(3):
            flux = k * gradient[:, axis : axis + 1]
            term = (self.mu[axis] ** 2) * grad(flux, u)[:, axis : axis + 1]
            divergence = term if divergence is None else divergence + term

        source = self.source(u, head)
        residual = gradient[:, 3:4] - self.alpha * divergence - source
        return residual / self.residual_scale(source)

    def pde_losses(self, batch) -> dict[str, torch.Tensor]:
        """``batch`` is a mapping with ``"west"`` and ``"east"`` collocation sets."""
        if not isinstance(batch, dict):
            raise TypeError(
                "ConservativePINN.pde_losses expects {'west': batch, 'east': batch}"
            )
        total = None
        for block, key in enumerate(("west", "east")):
            part = batch[key]
            u = self.unit_input(part.x, part.y, part.z, part.t)
            term = torch.mean(self._block_residual(u, block) ** 2)
            total = term if total is None else total + term
        return {"pde": total / 2.0}

    # -- interface -------------------------------------------------------------- #
    def _normal_flux(self, u: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled head and fault-normal Darcy flux ``Q_n`` for one block."""
        head = self.head_scaled_block(u, block)
        gradient = grad(head, u)
        k = self.conductivity_block(u[:, :3], block)
        nx, ny = self.fault_normal[0], self.fault_normal[1]
        q_n = -(k / self.k_ref) * (
            nx * self.mu[0] * gradient[:, 0:1] + ny * self.mu[1] * gradient[:, 1:2]
        )
        return head, q_n

    def gamma(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Dimensionless fault leakance at in-plane coordinates ``(y, z)``."""
        s = self.scaling
        yz = torch.cat([(y - s.y_c) / s.a_y, (z - s.z_c) / s.a_z], dim=1)
        return self.gamma_net(yz)

    def interface_losses(self, iface) -> dict[str, torch.Tensor]:
        u_west = self.unit_input(iface.x_west, iface.y_west, iface.z, iface.t)
        u_east = self.unit_input(iface.x_east, iface.y_east, iface.z, iface.t)

        head_w, flux_w = self._normal_flux(u_west, 0)
        head_e, flux_e = self._normal_flux(u_east, 1)

        # (1) Mass conservation across the plane -- always enforced.
        losses = {"interface_flux": torch.mean((flux_w - flux_e) ** 2)}

        # (2) Head coupling.
        if self.interface_mode == "continuity":
            losses["interface_head"] = torch.mean((head_w - head_e) ** 2)
        else:
            gamma = self.gamma(iface.y, iface.z)
            flux_mean = 0.5 * (flux_w + flux_e)
            residual = (gamma * (head_w - head_e) - flux_mean) / (1.0 + gamma)
            losses["interface_head"] = torch.mean(residual**2)

        return losses

    # -- boundaries -------------------------------------------------------------- #
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
                flux = self.mu[face.axis] * gradient[:, face.axis : face.axis + 1]
                term = torch.mean(flux**2)
                total = term if total is None else total + term
                count += 1
            if count:
                losses["neumann"] = total / count

        return losses

    # -- inferred fault behaviour ------------------------------------------------ #
    @torch.no_grad()
    def fault_conductance(self, n_y: int = 60, n_z: int = 12) -> dict[str, np.ndarray]:
        """Sample the inferred fault leakance over the fault plane.

        Returns ``C`` in 1/day and the equivalent fault conductivity
        ``K_f = C * width`` in m/d -- directly comparable with the value the
        benchmark used.
        """
        grid = self.cfg.grid
        device = self.mu.device
        dtype = self.mu.dtype

        y = np.linspace(0.0, grid.Ly, n_y)
        z = np.linspace(grid.zbot, grid.top, n_z)
        yy, zz = np.meshgrid(y, z, indexing="ij")

        y_col = torch.as_tensor(yy.ravel(), dtype=dtype, device=device).reshape(-1, 1)
        z_col = torch.as_tensor(zz.ravel(), dtype=dtype, device=device).reshape(-1, 1)
        gamma = self.gamma(y_col, z_col).cpu().numpy().reshape(yy.shape)

        conductance = self.scaling.conductance_from_gamma(gamma)
        return {
            "y": yy,
            "z": zz,
            "gamma": gamma,
            "conductance_per_day": conductance,
            "equivalent_k_m_per_day": conductance * self.cfg.fault.width,
        }

    @torch.no_grad()
    def predict_conductivity_grid(self, batch_size: int = 20000) -> np.ndarray:
        """Inferred K grid, with fault-zone cells filled from the leakance."""
        k = super().predict_conductivity_grid(batch_size=batch_size)
        if self.interface_mode != "conductance":
            return k

        grid = self.cfg.grid
        x, y, z = grid.cell_center_arrays()
        mask = self.cfg.fault.in_fault_zone(x, y)
        if not mask.any():
            return k

        device = self.mu.device
        dtype = self.mu.dtype
        y_col = torch.as_tensor(y[mask], dtype=dtype, device=device).reshape(-1, 1)
        z_col = torch.as_tensor(z[mask], dtype=dtype, device=device).reshape(-1, 1)
        gamma = self.gamma(y_col, z_col).cpu().numpy().reshape(-1)
        k[mask] = self.scaling.conductance_from_gamma(gamma) * self.cfg.fault.width
        return k
