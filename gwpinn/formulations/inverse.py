"""Ways of parameterising the unknown conductivity field.

The inverse problem is where the choice of *parameterisation* matters more than
the choice of network.  Only the divergence of ``T grad h`` is observable, so
the data leave a large null space; what fills it is the parameterisation's
implicit prior.

``net``
    A coordinate network with its own high-frequency Fourier band.  Flexible,
    but its prior is "whatever a Fourier-featured MLP finds easy", which is not
    a geostatistical statement about the aquifer.

``grid``
    One unknown per cell (10 000 of them) plus an explicit edge-preserving
    (pseudo-Huber / total-variation) penalty.  Maximum flexibility, honest about
    the dimensionality of the problem, and the penalty is a statement one can
    argue about.  Needs the penalty: without it this is hopelessly ill-posed.

``kl``
    Truncated Karhunen-Loeve expansion.  For a *stationary* covariance the KL
    eigenfunctions on a periodic domain are exactly the Fourier modes and the
    eigenvalues are the power spectral density, so the expansion can be built in
    closed form from the variogram - no 10 000 x 10 000 eigendecomposition.
    Keeping the ``M`` most energetic modes reduces the unknowns from 10 000 to a
    few hundred **and** guarantees that every field the optimiser can reach has
    the right spatial statistics.  This is the classical geostatistical
    inversion prior (Kitanidis' principal-component parameterisation) written as
    a differentiable layer.

The benchmark's truth is an exponential-covariance field with a known range and
variance, so ``kl`` is being given the correct prior family.  That is deliberate:
it measures the value of *having* a correct structural prior, which is the
question a practitioner actually faces.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from gwpinn.formulations.arch import CoordinateNet
from gwpinn.formulations.problem import Problem


class KField(nn.Module):
    """Base: maps ``(x, y)`` to hydraulic conductivity in m/d."""

    def __init__(self, prob: Problem, k_min: float = 1e-3, k_max: float = 2e3) -> None:
        super().__init__()
        self.prob = prob
        self.log_min = math.log10(k_min)
        self.log_max = math.log10(k_max)

    def squash(self, raw: torch.Tensor) -> torch.Tensor:
        """Map an unbounded field onto ``[k_min, k_max]`` in log space."""
        frac = torch.sigmoid(raw)
        return torch.pow(10.0, self.log_min + (self.log_max - self.log_min) * frac)

    def regulariser(self) -> torch.Tensor:
        return torch.zeros((), device=self.prob.device, dtype=self.prob.dtype)

    def as_grid(self) -> torch.Tensor:
        """Evaluate on the cell centres, ``(nrow, ncol)``."""
        x, y = self.prob.grid_points()
        return self(x, y).view(self.prob.nrow, self.prob.ncol)


class NetKField(KField):
    """Coordinate network with a high-frequency Fourier band."""

    def __init__(self, prob: Problem, width: int = 64, depth: int = 3,
                 fourier: int = 64, fourier_sigma: float = 6.0,
                 arch: str = "mlp", k0: float = 5.0, seed: int = 0, **kw) -> None:
        super().__init__(prob, **kw)
        self.net = CoordinateNet(2, 1, arch=arch, width=width, depth=depth,
                                 fourier=fourier, fourier_sigma=fourier_sigma,
                                 seed=seed + 77)
        # Start at the prior geometric mean rather than at K = midpoint of the
        # bounds, which would be ~1.4 m/d here and put the whole field in the
        # wrong regime for the first thousand iterations.
        frac = (math.log10(k0) - self.log_min) / (self.log_max - self.log_min)
        self.bias = nn.Parameter(torch.tensor(math.log(frac / (1 - frac))))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xn = self.prob.scales.norm_xy(x, y)
        return self.squash(self.net(xn).squeeze(-1) + self.bias)


class GridKField(KField):
    """One unknown per cell, bilinearly interpolated, with a TV-like penalty."""

    def __init__(self, prob: Problem, k0: float = 5.0, tv_weight: float = 1e-3,
                 tv_delta: float = 0.15, **kw) -> None:
        super().__init__(prob, **kw)
        frac = (math.log10(k0) - self.log_min) / (self.log_max - self.log_min)
        init = math.log(frac / (1 - frac))
        self.raw = nn.Parameter(torch.full((prob.nrow, prob.ncol), init,
                                           dtype=prob.dtype))
        self.tv_weight = float(tv_weight)
        self.tv_delta = float(tv_delta)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = self.prob
        # Normalised grid coordinates in [-1, 1] for grid_sample (align_corners
        # so cell centres map exactly onto sample points).
        gx = (x - p.xmin) / (p.xmax - p.xmin) * 2.0 - 1.0
        gy = (p.ymax - y) / (p.ymax - p.ymin) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
        raw = torch.nn.functional.grid_sample(
            self.raw[None, None], grid, mode="bilinear",
            padding_mode="border", align_corners=False,
        ).view(-1)
        return self.squash(raw)

    def regulariser(self) -> torch.Tensor:
        """Pseudo-Huber total variation: suppresses speckle, keeps facies edges.

        A plain squared-gradient penalty can only push variance down, so it
        smears the very contacts the inversion is supposed to find.  The
        pseudo-Huber form is quadratic for small jumps and linear for large
        ones, so a genuine order-of-magnitude contact is cheap to keep.
        """
        d = self.tv_delta
        gx = self.raw[:, 1:] - self.raw[:, :-1]
        gy = self.raw[1:, :] - self.raw[:-1, :]
        ph = lambda g: (torch.sqrt(g * g + d * d) - d).mean()  # noqa: E731
        return self.tv_weight * (ph(gx) + ph(gy))


class KLKField(KField):
    """Truncated Karhunen-Loeve expansion with the benchmark's covariance.

    For a stationary random field on a periodic box the KL modes are the Fourier
    modes and the eigenvalues are the spectral density ``S(k)``.  For an
    exponential covariance in 2-D,

    ``S(k) ∝ var * l^2 / (1 + (2 pi l |k|)^2)^{3/2}``.

    Keeping the ``n_modes`` largest-``S`` wavenumbers gives an expansion whose
    every realisation has (approximately) the right variogram by construction.
    The unknowns are the mode coefficients, initialised at zero so the field
    starts at its prior mean.
    """

    def __init__(self, prob: Problem, n_modes: int = 384, var: float = 2.0,
                 len_scale: float = 200.0, k0: float = 5.0, seed: int = 0, **kw) -> None:
        super().__init__(prob, **kw)
        p = prob
        Lx = p.xmax - p.xmin
        Ly = p.ymax - p.ymin

        # Candidate wavenumbers on the box's reciprocal lattice.
        nmax = 24
        kk = np.arange(-nmax, nmax + 1)
        KX, KY = np.meshgrid(kk / Lx, kk / Ly, indexing="ij")
        kmag = np.sqrt(KX**2 + KY**2)
        psd = var * len_scale**2 / (1.0 + (2 * np.pi * len_scale * kmag) ** 2) ** 1.5
        psd[kmag == 0] = 0.0

        order = np.argsort(psd.ravel())[::-1][:n_modes]
        kx = KX.ravel()[order]
        ky = KY.ravel()[order]
        amp = np.sqrt(psd.ravel()[order])
        amp = amp / np.sqrt((amp**2).sum()) * math.sqrt(var)  # unit total variance

        self.register_buffer("kx", torch.as_tensor(kx, dtype=p.dtype))
        self.register_buffer("ky", torch.as_tensor(ky, dtype=p.dtype))
        self.register_buffer("amp", torch.as_tensor(amp, dtype=p.dtype))
        g = torch.Generator().manual_seed(seed)
        self.coef_cos = nn.Parameter(torch.zeros(n_modes, dtype=p.dtype))
        self.coef_sin = nn.Parameter(torch.zeros(n_modes, dtype=p.dtype))
        # log10 K is modelled as prior mean + KL perturbation, and mapped to K
        # directly (no sigmoid squash: the KL expansion is already bounded in
        # practice and the squash would distort the imposed variogram).
        self.log_k0 = nn.Parameter(torch.tensor(math.log10(k0), dtype=p.dtype))
        self.log_sigma = nn.Parameter(torch.tensor(0.0, dtype=p.dtype))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ph = 2.0 * math.pi * (x[:, None] * self.kx[None] + y[:, None] * self.ky[None])
        field = ((self.amp * self.coef_cos)[None] * torch.cos(ph)
                 + (self.amp * self.coef_sin)[None] * torch.sin(ph)).sum(-1)
        log_k = self.log_k0 + torch.exp(self.log_sigma) * field
        return torch.pow(10.0, log_k.clamp(self.log_min, self.log_max))

    def regulariser(self) -> torch.Tensor:
        """Gaussian prior on the KL coefficients - the -log density of N(0, I)."""
        return 1e-3 * (self.coef_cos.pow(2).mean() + self.coef_sin.pow(2).mean())


def build_k_field(kind: str, prob: Problem, seed: int = 0, **kw) -> KField:
    if kind == "net":
        return NetKField(prob, seed=seed, **kw)
    if kind == "grid":
        return GridKField(prob, **kw)
    if kind == "kl":
        return KLKField(prob, seed=seed, **kw)
    if kind == "true":
        return TrueKField(prob)
    raise ValueError(f"unknown K parameterisation {kind!r}")


class TrueKField(KField):
    """The reference field, frozen.  Used by the forward-only task."""

    def __init__(self, prob: Problem) -> None:
        super().__init__(prob)
        self.register_buffer("k", prob.kh_true)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Interpolation-aware: the strong form needs grad K to be defined.
        return self.prob.coeff(self.k, x, y)


__all__ = ["KField", "NetKField", "GridKField", "KLKField", "TrueKField", "build_k_field"]
