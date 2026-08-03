"""Shared machinery for the three inverse-PINN architectures.

Every architecture works on the same non-dimensional residual derived in
:mod:`gwpinns.pinn.scaling`.  What differs is *how* the divergence term is
formed:

============================  ==================================================
Baseline (:mod:`.baseline`)   second order, ``d_i(K d_i H)`` via nested autograd
Mixed (:mod:`.mixed`)         first order, Darcy + continuity on ``(H, U)``
cPINN (:mod:`.cpinn`)         one model per fault block + interface conditions
============================  ==================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from ..config import BenchmarkConfig
from .derivatives import grad
from .forcing import ForcingTerm
from .scaling import Scaling

__all__ = ["LossWeights", "BasePINN"]


@dataclass
class LossWeights:
    """Relative weights of the loss terms.

    Defaults put the data term an order of magnitude above the physics: the
    heads are the only real information, and an over-weighted PDE term drives
    the conductivity field towards whatever is smoothest rather than whatever
    fits the observations.
    """

    data: float = 10.0
    pde: float = 1.0
    darcy: float = 1.0
    dirichlet: float = 1.0
    neumann: float = 1.0
    interface_flux: float = 1.0
    interface_head: float = 1.0
    k_smoothness: float = 0.0
    k_prior: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class BasePINN(nn.Module):
    """Common coordinate handling, forcing evaluation and loss assembly."""

    def __init__(
        self,
        cfg: BenchmarkConfig,
        scaling: Scaling,
        forcing: ForcingTerm | None = None,
        residual_weighting: str = "source",
        log_k_prior: float = 0.0,
    ):
        super().__init__()
        if residual_weighting not in {"source", "none"}:
            raise ValueError("residual_weighting must be 'source' or 'none'")
        self.cfg = cfg
        self.scaling = scaling
        self.residual_weighting = residual_weighting
        self.log_k_prior = float(log_k_prior)
        self.forcing = forcing if forcing is not None else ForcingTerm(cfg)

        mu = scaling.mu
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.get_default_dtype()))
        self.alpha = float(scaling.alpha)
        self.beta = float(scaling.beta)

    # -- coordinate plumbing -------------------------------------------------- #
    def unit_input(self, x, y, z, t) -> torch.Tensor:
        """Physical columns -> a differentiable leaf tensor in unit coordinates.

        All autograd in the residuals is taken with respect to *this* tensor, so
        the derivatives are exactly the ``d_X, d_Y, d_Z, d_T`` of the scaled
        equation and no chain-rule factors are applied twice.
        """
        s = self.scaling
        u = torch.cat(
            [
                (x - s.x_c) / s.a_x,
                (y - s.y_c) / s.a_y,
                (z - s.z_c) / s.a_z,
                (t - s.t_c) / s.a_t,
            ],
            dim=1,
        )
        return u.detach().requires_grad_(True)

    def unit_input_xyz(self, x, y, z) -> torch.Tensor:
        s = self.scaling
        u = torch.cat(
            [(x - s.x_c) / s.a_x, (y - s.y_c) / s.a_y, (z - s.z_c) / s.a_z], dim=1
        )
        return u.detach().requires_grad_(True)

    def physical(self, u: torch.Tensor):
        """Unit coordinates -> physical ``(x, y, z, t)`` columns."""
        s = self.scaling
        return (
            u[:, 0:1] * s.a_x + s.x_c,
            u[:, 1:2] * s.a_y + s.y_c,
            u[:, 2:3] * s.a_z + s.z_c,
            u[:, 3:4] * s.a_t + s.t_c,
        )

    # -- forcing --------------------------------------------------------------- #
    def source(self, u: torch.Tensor, head_scaled: torch.Tensor) -> torch.Tensor:
        """Evaluate ``beta * W`` at the collocation points (dimensionless)."""
        x, y, z, t = self.physical(u)
        h = self.scaling.decode_head(head_scaled)
        return self.beta * self.forcing(x, y, z, t, h)

    def residual_scale(self, source: torch.Tensor) -> torch.Tensor:
        """Local normaliser for the PDE residual.

        Inside a pumping cell ``beta*W`` is ~1000x its bulk value, so an
        unweighted mean-square residual is effectively a well-cell-only loss.
        Dividing by ``1 + |beta W|`` restores a comparable *relative* accuracy
        target everywhere without changing where the residual is zero.
        """
        if self.residual_weighting == "none":
            return torch.ones_like(source)
        return 1.0 + source.abs()

    # -- interface every architecture implements ------------------------------ #
    def head_scaled(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def conductivity(self, u3: torch.Tensor) -> torch.Tensor:
        """Hydraulic conductivity in m/d at unit spatial coordinates."""
        raise NotImplementedError

    def pde_losses(self, batch) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def boundary_losses(self, dirichlet, neumann) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def interface_losses(self, interface) -> dict[str, torch.Tensor]:
        """Only the domain-decomposition architecture couples across the fault."""
        return {}

    # -- losses shared by all architectures ----------------------------------- #
    def data_loss(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean-square misfit on the scaled head at the monitoring points."""
        u = self.unit_input(obs["x"], obs["y"], obs["z"], obs["t"])
        pred = self.head_scaled(u)
        return torch.mean((pred - obs["head"]) ** 2)

    def k_prior_loss(self, u3: torch.Tensor) -> torch.Tensor:
        """Weak Tikhonov pull of ``log10 K`` towards a prior bulk value.

        Without it the inverse problem is degenerate: away from wells and
        observation points, shrinking ``K`` towards zero satisfies the PDE for
        *any* smooth head field, and the optimiser reliably takes that route.
        The prior is a round-number bulk estimate of the kind any site
        investigation supplies -- deliberately not the benchmark's true
        geometric mean -- and its weight is small enough that data and physics
        override it wherever they carry information.  Its influence is
        quantified by the prior-sensitivity ablation in the study.
        """
        log_k = torch.log10(self.conductivity(u3))
        return torch.mean((log_k - self.log_k_prior) ** 2)

    def k_smoothness_loss(self, u: torch.Tensor) -> torch.Tensor:
        """Optional Tikhonov penalty on ``grad log10 K`` (off by default).

        Useful as an ablation: turning it up demonstrably smears the fault,
        which is why the default weight is zero.
        """
        u3 = u[:, :3].detach().requires_grad_(True)
        log_k = torch.log10(self.conductivity(u3))
        gradient = grad(log_k, u3)
        return torch.mean(gradient**2)

    def loss_terms(
        self,
        obs: dict[str, torch.Tensor],
        collocation,
        dirichlet=None,
        neumann=None,
        interface=None,
        weights: LossWeights | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate every (unweighted) loss component as a live tensor."""
        weights = weights or LossWeights()
        terms: dict[str, torch.Tensor] = {"data": self.data_loss(obs)}
        terms.update(self.pde_losses(collocation))
        if dirichlet is not None or neumann is not None:
            terms.update(self.boundary_losses(dirichlet, neumann))
        if interface is not None:
            terms.update(self.interface_losses(interface))

        # The cPINN passes a per-block mapping rather than a single batch.
        parts = (
            list(collocation.values()) if isinstance(collocation, dict)
            else [collocation]
        )

        if weights.k_prior > 0:
            penalty = None
            for part in parts:
                u3 = self.unit_input_xyz(part.x, part.y, part.z)
                term = self.k_prior_loss(u3)
                penalty = term if penalty is None else penalty + term
            terms["k_prior"] = penalty / len(parts)

        if weights.k_smoothness > 0:
            penalty = None
            for part in parts:
                u = self.unit_input(part.x, part.y, part.z, part.t)
                term = self.k_smoothness_loss(u)
                penalty = term if penalty is None else penalty + term
            terms["k_smoothness"] = penalty / len(parts)

        return terms

    @staticmethod
    def combine(
        terms: dict[str, torch.Tensor], weights: LossWeights
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Weighted sum of loss components, plus a float report of each."""
        weight_map = weights.as_dict()
        total = None
        for name, value in terms.items():
            w = weight_map.get(name, 1.0)
            if w == 0.0:
                continue
            contribution = w * value
            total = contribution if total is None else total + contribution

        report = {name: float(value.detach()) for name, value in terms.items()}
        report["total"] = float(total.detach())
        return total, report

    def total_loss(
        self,
        obs: dict[str, torch.Tensor],
        collocation,
        dirichlet=None,
        neumann=None,
        interface=None,
        weights: LossWeights | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Assemble the weighted objective and a report of its components."""
        weights = weights or LossWeights()
        terms = self.loss_terms(
            obs, collocation, dirichlet, neumann, interface, weights
        )
        return self.combine(terms, weights)

    # -- prediction ------------------------------------------------------------ #
    @torch.no_grad()
    def predict_conductivity_grid(self, batch_size: int = 20000) -> np.ndarray:
        """Inferred K on the benchmark grid, shape ``(nlay, nrow, ncol)``."""
        grid = self.cfg.grid
        x, y, z = grid.cell_center_arrays()
        shape = x.shape
        device = self.mu.device
        dtype = self.mu.dtype

        cols = [
            torch.as_tensor(a.ravel(), dtype=dtype, device=device).reshape(-1, 1)
            for a in (x, y, z)
        ]
        out = []
        total = cols[0].shape[0]
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            u3 = torch.cat([c[start:stop] for c in cols], dim=1)
            u3 = (u3 - torch.tensor(
                [self.scaling.x_c, self.scaling.y_c, self.scaling.z_c],
                dtype=dtype, device=device,
            )) / torch.tensor(
                [self.scaling.a_x, self.scaling.a_y, self.scaling.a_z],
                dtype=dtype, device=device,
            )
            out.append(self.conductivity(u3).reshape(-1))
        return torch.cat(out).cpu().numpy().reshape(shape)

    @torch.no_grad()
    def predict_head_grid(self, times: np.ndarray, batch_size: int = 20000) -> np.ndarray:
        """Predicted heads on the grid, shape ``(len(times), nlay, nrow, ncol)``."""
        grid = self.cfg.grid
        x, y, z = grid.cell_center_arrays()
        shape = x.shape
        device = self.mu.device
        dtype = self.mu.dtype

        cols = [
            torch.as_tensor(a.ravel(), dtype=dtype, device=device).reshape(-1, 1)
            for a in (x, y, z)
        ]
        frames = []
        for time in np.atleast_1d(times):
            t_col = torch.full_like(cols[0], float(time))
            out = []
            total = cols[0].shape[0]
            for start in range(0, total, batch_size):
                stop = min(start + batch_size, total)
                u = self.unit_input(
                    cols[0][start:stop], cols[1][start:stop],
                    cols[2][start:stop], t_col[start:stop],
                )
                with torch.enable_grad():
                    head = self.head_scaled(u)
                out.append(self.scaling.decode_head(head).detach().reshape(-1))
            frames.append(torch.cat(out).cpu().numpy().reshape(shape))
        return np.stack(frames, axis=0)
