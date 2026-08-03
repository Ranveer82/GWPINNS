"""Non-dimensionalisation of the groundwater flow problem.

Getting this right is most of the battle in a 3-D transient inverse PINN.  The
domain is 5000 x 3000 x 60 m over 360 days with heads of order 50 m and
conductivities spanning seven orders of magnitude; feeding those numbers to a
network raw does not train.

Convention
----------
Network inputs are mapped to ``[-1, 1]`` per axis::

    X = (x - x_c)/a_x,   Y = (y - y_c)/a_y,   Z = (z - z_c)/a_z,   T = (t - t_c)/a_t

and the head is centred and scaled, ``H = (h - h_c)/dh``.  Derivatives written
``d_X`` below are with respect to these unit variables.

Starting from ``Ss dh/dt = div(K grad h) + W`` and dividing through by
``Ss*dh/a_t`` gives the **residual used by every architecture**::

    d_T H  -  alpha * sum_i mu_i^2 d_i( K d_i H )  -  beta * W  =  0

with

    mu_i  = L0 / a_i          (L0 = a_x, so mu_x = 1)
    alpha = a_t / (Ss * L0^2)
    beta  = a_t / (Ss * dh)

Note that ``K`` enters in **physical units (m/d)** and the reference
conductivity ``K0`` has cancelled out entirely -- the residual does not depend
on any prior guess of conductivity.  ``K0`` survives only as the scale of the
Darcy flux in the mixed-variable formulation, where it is a pure unit choice.

The first-order (mixed) form of the same equation is::

    Darcy_i     :  U_i + (K/K0) * mu_i * d_i H            = 0
    continuity  :  d_T H + alpha*K0 * sum_i mu_i d_i U_i - beta*W = 0

where ``U_i = q_i / q0`` and ``q0 = K0*dh/L0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import BenchmarkConfig

__all__ = ["Scaling"]


@dataclass(frozen=True)
class Scaling:
    """Affine scaling of coordinates, head and flux, plus the PDE coefficients."""

    x_c: float
    y_c: float
    z_c: float
    t_c: float
    a_x: float
    a_y: float
    a_z: float
    a_t: float
    h_c: float
    dh: float
    specific_storage: float
    k_ref: float = 1.0

    # -- derived coefficients ------------------------------------------------ #
    @property
    def L0(self) -> float:
        """Reference length: the horizontal half-length, so ``mu_x == 1``."""
        return self.a_x

    @property
    def mu(self) -> tuple[float, float, float]:
        """Anisotropy factors ``L0/a_i`` for the x, y and z axes."""
        return (self.L0 / self.a_x, self.L0 / self.a_y, self.L0 / self.a_z)

    @property
    def alpha(self) -> float:
        """Diffusion coefficient ``a_t / (Ss * L0^2)`` -- multiplies K in m/d."""
        return self.a_t / (self.specific_storage * self.L0**2)

    @property
    def beta(self) -> float:
        """Source coefficient ``a_t / (Ss * dh)`` -- multiplies W in 1/d."""
        return self.a_t / (self.specific_storage * self.dh)

    @property
    def q_ref(self) -> float:
        """Darcy flux scale ``K0 * dh / L0`` in m/d."""
        return self.k_ref * self.dh / self.L0

    def gamma_from_conductance(self, conductance: float) -> float:
        """Dimensionless fault leakance ``Gamma = C * L0 / K0`` from ``C = K_f/w``."""
        return conductance * self.L0 / self.k_ref

    def conductance_from_gamma(self, gamma):
        """Inverse of :meth:`gamma_from_conductance`, returning ``C`` in 1/day."""
        return gamma * self.k_ref / self.L0

    # -- construction -------------------------------------------------------- #
    @classmethod
    def from_config(
        cls,
        cfg: BenchmarkConfig,
        head_center: float | None = None,
        head_scale: float | None = None,
        k_ref: float = 1.0,
    ) -> "Scaling":
        """Build a scaling from the benchmark geometry.

        ``head_center``/``head_scale`` default to the boundary heads.  Prefer
        :meth:`from_observations`, which calibrates them on the observed data
        (and therefore uses no information the inverse problem would not have).
        """
        grid, tim, bc = cfg.grid, cfg.time, cfg.boundary
        h_c = head_center if head_center is not None else 0.5 * (bc.head_west + bc.head_east)
        dh = head_scale if head_scale is not None else 0.5 * abs(bc.head_west - bc.head_east)
        return cls(
            x_c=0.5 * grid.Lx,
            y_c=0.5 * grid.Ly,
            z_c=0.5 * (grid.top + grid.zbot),
            t_c=0.5 * tim.total_time,
            a_x=0.5 * grid.Lx,
            a_y=0.5 * grid.Ly,
            a_z=0.5 * grid.Lz,
            a_t=0.5 * tim.total_time,
            h_c=h_c,
            dh=max(dh, 1e-6),
            specific_storage=cfg.aquifer.specific_storage,
            k_ref=k_ref,
        )

    @classmethod
    def from_observations(
        cls, cfg: BenchmarkConfig, heads: np.ndarray, k_ref: float = 1.0
    ) -> "Scaling":
        """Calibrate the head scaling on the observed head values only."""
        h_c = float(np.mean(heads))
        dh = float(np.max(np.abs(heads - h_c)))
        return cls.from_config(cfg, head_center=h_c, head_scale=dh, k_ref=k_ref)

    # -- transforms ---------------------------------------------------------- #
    def encode_xyzt(self, x, y, z, t):
        """Map physical ``(x, y, z, t)`` columns to a unit-cube input tensor."""
        stack = torch.stack if torch.is_tensor(x) else np.stack
        return stack(
            [
                (x - self.x_c) / self.a_x,
                (y - self.y_c) / self.a_y,
                (z - self.z_c) / self.a_z,
                (t - self.t_c) / self.a_t,
            ],
            axis=-1 if not torch.is_tensor(x) else -1,
        )

    def encode_xyz(self, x, y, z):
        """Map physical ``(x, y, z)`` columns to a unit-cube input tensor."""
        stack = torch.stack if torch.is_tensor(x) else np.stack
        return stack(
            [
                (x - self.x_c) / self.a_x,
                (y - self.y_c) / self.a_y,
                (z - self.z_c) / self.a_z,
            ],
            axis=-1,
        )

    def decode_xyzt(self, u):
        """Inverse of :meth:`encode_xyzt`; returns physical ``(x, y, z, t)``."""
        x = u[..., 0] * self.a_x + self.x_c
        y = u[..., 1] * self.a_y + self.y_c
        z = u[..., 2] * self.a_z + self.z_c
        t = u[..., 3] * self.a_t + self.t_c
        return x, y, z, t

    def encode_head(self, h):
        return (h - self.h_c) / self.dh

    def decode_head(self, H):
        return H * self.dh + self.h_c

    def summary(self) -> dict[str, float]:
        mu = self.mu
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "mu_x": mu[0],
            "mu_y": mu[1],
            "mu_z": mu[2],
            "head_center_m": self.h_c,
            "head_scale_m": self.dh,
            "q_ref_m_per_d": self.q_ref,
        }
