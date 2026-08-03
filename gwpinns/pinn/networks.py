"""Network building blocks shared by the three architectures."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "FourierFeatures",
    "MLP",
    "LogConductivityNet",
    "FaultConductance",
    "HEAD_FOURIER_SIGMA",
    "K_FOURIER_SIGMA",
]

# Per-axis Fourier bandwidths.  Horizontal directions carry the structure
# (fault, drawdown cones, heterogeneity); the vertical is nearly hydrostatic
# and time is smooth apart from the stress-period steps.
HEAD_FOURIER_SIGMA = (2.0, 2.0, 0.25, 1.0)      # (X, Y, Z, T)
K_FOURIER_SIGMA = (3.0, 3.0, 0.4)               # (X, Y, Z)


class FourierFeatures(nn.Module):
    """Fixed random Fourier encoding ``x -> [x, sin(pi B x), cos(pi B x)]``.

    Plain tanh MLPs are strongly biased towards low frequencies, which is fatal
    for a problem whose defining feature is a sharp fault.  A random Fourier
    encoding lifts that bias.  ``B`` is a fixed (non-trained) buffer so the
    encoding stays a deterministic property of the model.

    ``sigma`` may be a **per-axis** sequence, and for this problem it must be.
    An isotropic encoding puts the same high-frequency content on ``z`` as on
    ``x``, and the vertical diffusion term carries a factor ``mu_z^2 ~ 7000``;
    the untrained residual is then ~10^13 and the only cheap way for the
    optimiser to reduce it is to drive ``K`` to its lower bound, from which the
    bounded parameterisation cannot recover.  Since a 60 m-thick aquifer is
    nearly hydrostatic in the vertical, a low vertical bandwidth costs nothing
    physically and fixes the conditioning.
    """

    def __init__(
        self,
        in_dim: int,
        n_features: int = 64,
        sigma: float | Sequence[float] = 2.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_features = n_features

        scales = torch.as_tensor(
            [float(sigma)] * in_dim if np.isscalar(sigma) else list(sigma),
            dtype=torch.get_default_dtype(),
        )
        if scales.numel() != in_dim:
            raise ValueError(
                f"sigma must be scalar or length {in_dim}, got {scales.numel()}"
            )
        self.register_buffer("sigma", scales)

        if n_features > 0:
            b = torch.randn(in_dim, n_features) * scales[:, None]
            self.register_buffer("B", b)
        else:
            self.register_buffer("B", torch.zeros(in_dim, 0))

    @property
    def out_dim(self) -> int:
        return self.in_dim + 2 * self.n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_features == 0:
            return x
        proj = math.pi * (x @ self.B)
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    """Fully connected tanh network with an optional Fourier input encoding."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 96,
        depth: int = 5,
        fourier_features: int = 64,
        fourier_sigma: float | Sequence[float] = 2.0,
    ):
        super().__init__()
        self.encoding = FourierFeatures(in_dim, fourier_features, fourier_sigma)

        dims = [self.encoding.out_dim] + [width] * depth + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.activation = torch.tanh

        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoding(x)
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
        return self.layers[-1](h)


class LogConductivityNet(nn.Module):
    """Predicts ``log10 K`` bounded to a physically sensible interval.

    The bounded parameterisation ``log10 K = mid + half * tanh(raw)`` is what
    keeps the inverse problem stable: an unbounded K head diverges as soon as
    the data are locally uninformative, whereas this simply saturates towards
    the prior bounds.  The default window spans 1e-4 to 1e3 m/d -- wide enough
    to contain both a clay-gouge barrier and an open conduit.
    """

    def __init__(
        self,
        in_dim: int = 3,
        width: int = 96,
        depth: int = 5,
        fourier_features: int = 64,
        fourier_sigma: float | Sequence[float] = K_FOURIER_SIGMA,
        log_k_min: float = -4.0,
        log_k_max: float = 3.0,
        log_k_init: float = 0.0,
    ):
        super().__init__()
        if log_k_max <= log_k_min:
            raise ValueError("log_k_max must exceed log_k_min")
        if not log_k_min < log_k_init < log_k_max:
            raise ValueError("log_k_init must lie strictly inside the bounds")
        self.net = MLP(
            in_dim, 1, width=width, depth=depth,
            fourier_features=fourier_features, fourier_sigma=fourier_sigma,
        )
        self.register_buffer("log_mid", torch.tensor(0.5 * (log_k_max + log_k_min)))
        self.register_buffer("log_half", torch.tensor(0.5 * (log_k_max - log_k_min)))

        # Start the field at a plausible bulk conductivity rather than at the
        # centre of the bounds.  The tanh squash has vanishing gradient near its
        # saturation points, so a field that collapses to the lower bound early
        # in training can never climb back out; starting mid-range avoids that
        # trap without prescribing the answer.
        #
        # The output layer is also damped so the initial field is nearly
        # *uniform* at that value.  A randomly structured starting K interacts
        # with an equally random starting head field to produce a very large
        # initial residual, which is what pushes K towards the bound.
        target = (log_k_init - float(self.log_mid)) / float(self.log_half)
        with torch.no_grad():
            self.net.layers[-1].weight.mul_(0.05)
            self.net.layers[-1].bias.fill_(math.atanh(max(min(target, 0.99), -0.99)))

    def log10_k(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_mid + self.log_half * torch.tanh(self.net(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hydraulic conductivity in m/d."""
        return torch.pow(10.0, self.log10_k(x))


class FaultConductance(nn.Module):
    """Inferred fault-plane leakance, the cPINN's headline unknown.

    Parameterised as ``log10 Gamma`` where ``Gamma = C * L0 / K0`` is the
    dimensionless leakance and ``C = K_fault / width`` in 1/day.  Two modes:

    ``"scalar"``
        a single learnable number -- the fault behaves uniformly;
    ``"field"``
        a small MLP over the in-plane coordinates ``(Y, Z)``, letting the fault
        be a barrier in one place and a conduit in another.

    ``Gamma -> 0`` is a perfect barrier and ``Gamma -> inf`` recovers head
    continuity, so a single fitted number classifies the fault's behaviour.
    """

    def __init__(
        self,
        mode: str = "field",
        width: int = 32,
        depth: int = 3,
        fourier_features: int = 16,
        fourier_sigma: float | Sequence[float] = (1.5, 0.4),
        log_gamma_min: float = -5.0,
        log_gamma_max: float = 5.0,
        log_gamma_init: float = 0.0,
    ):
        super().__init__()
        if mode not in {"scalar", "field"}:
            raise ValueError("mode must be 'scalar' or 'field'")
        self.mode = mode
        self.register_buffer("log_mid", torch.tensor(0.5 * (log_gamma_max + log_gamma_min)))
        self.register_buffer("log_half", torch.tensor(0.5 * (log_gamma_max - log_gamma_min)))

        # Invert the tanh squash so training starts at ``log_gamma_init``.
        target = (log_gamma_init - float(self.log_mid)) / float(self.log_half)
        raw_init = math.atanh(max(min(target, 0.99), -0.99))

        if mode == "scalar":
            self.raw = nn.Parameter(torch.tensor([[raw_init]]))
        else:
            self.net = MLP(
                2, 1, width=width, depth=depth,
                fourier_features=fourier_features, fourier_sigma=fourier_sigma,
            )
            nn.init.constant_(self.net.layers[-1].bias, raw_init)

    def log10_gamma(self, yz: torch.Tensor) -> torch.Tensor:
        if self.mode == "scalar":
            raw = self.raw.expand(yz.shape[0], 1)
        else:
            raw = self.net(yz)
        return self.log_mid + self.log_half * torch.tanh(raw)

    def forward(self, yz: torch.Tensor) -> torch.Tensor:
        """Dimensionless leakance ``Gamma``."""
        return torch.pow(10.0, self.log10_gamma(yz))
