"""Backbone architectures.

Four field representations, all exposing the same interface

    ``forward(feats, coords) -> (N, out_dim)``

and all twice-differentiable with respect to ``coords`` by autograd, so the same
PDE-residual code drives every one of them and the architecture comparison is
like-for-like:

``mlp``
    Fourier-feature MLP - the standard PINN backbone.
``resnet``
    Same, with residual blocks. Deeper networks stay trainable because the skip
    connections keep the gradient of the PDE residual from vanishing.
``modified_mlp``
    The gated architecture of Wang, Teng & Perdikaris (2021). Two "encoder"
    projections U, V of the input modulate every hidden layer, which damps the
    stiff gradient interactions between the data and residual losses.
``cnn``
    A convolutional decoder that emits the field on a regular grid, which is
    then read at arbitrary points through a **cubic B-spline** reconstruction.
    Bilinear sampling would have zero second derivative and could not feed a
    second-order PDE at all; the B-spline is C^2, so the grid representation
    stays usable inside the same residual.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Activations and init
# --------------------------------------------------------------------------- #


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "sin":
        return Sine()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "softplus":
        return nn.Softplus()
    raise ValueError(f"unknown activation {name!r}")


def xavier_init(module: nn.Module) -> None:
    """Glorot-normal init, the usual choice for tanh PINNs."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# --------------------------------------------------------------------------- #
# MLP family
# --------------------------------------------------------------------------- #


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 128,
        depth: int = 5,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        act = activation
        layers: list[nn.Module] = [nn.Linear(in_dim, width), get_activation(act)]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(width, width), get_activation(act)]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(xavier_init)

    def forward(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None):
        return self.net(feats)


class _ResBlock(nn.Module):
    def __init__(self, width: int, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(x + h)


class ResNetMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 128,
        depth: int = 5,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.act = get_activation(activation)
        n_blocks = max(1, depth // 2)
        self.blocks = nn.ModuleList([_ResBlock(width, activation) for _ in range(n_blocks)])
        self.out = nn.Linear(width, out_dim)
        self.apply(xavier_init)

    def forward(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None):
        h = self.act(self.inp(feats))
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


class ModifiedMLP(nn.Module):
    """Wang, Teng & Perdikaris (2021), eq. 3.4.

    ``U`` and ``V`` are computed once from the input and then mix into every
    hidden layer::

        H^{k+1} = (1 - Z^k) * U + Z^k * V,    Z^k = act(W^k H^k + b^k)

    The multiplicative paths give the residual gradients a direct route back to
    the input, which is what keeps the data loss and the PDE loss from fighting.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 128,
        depth: int = 5,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.act = get_activation(activation)
        self.enc_u = nn.Linear(in_dim, width)
        self.enc_v = nn.Linear(in_dim, width)
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim if i == 0 else width, width) for i in range(max(depth, 1))]
        )
        self.out = nn.Linear(width, out_dim)
        self.apply(xavier_init)

    def forward(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None):
        u = self.act(self.enc_u(feats))
        v = self.act(self.enc_v(feats))
        h = feats
        for layer in self.layers:
            z = self.act(layer(h))
            h = (1.0 - z) * u + z * v
        return self.out(h)


# --------------------------------------------------------------------------- #
# Grid CNN with C^2 spline reconstruction
# --------------------------------------------------------------------------- #


def cubic_bspline_weights(t: torch.Tensor) -> torch.Tensor:
    """Uniform cubic B-spline weights for the four knots around ``t`` in [0, 1).

    Returns a ``(..., 4)`` tensor. The basis is C^2, so a field built from it has
    continuous second derivatives and can be substituted straight into a
    second-order PDE residual.
    """
    t2 = t * t
    t3 = t2 * t
    w0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0
    w1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0
    w2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0
    w3 = t3 / 6.0
    return torch.stack([w0, w1, w2, w3], dim=-1)


def bspline_sample(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Evaluate a ``(C, H, W)`` coefficient grid at ``coords`` in ``[-1, 1]^2``.

    ``coords[:, 0]`` is x (columns) and ``coords[:, 1]`` is y (rows).
    """
    C, H, W = grid.shape
    x = (coords[:, 0] * 0.5 + 0.5) * (W - 1)
    y = (coords[:, 1] * 0.5 + 0.5) * (H - 1)

    ix = torch.floor(x)
    iy = torch.floor(y)
    tx = x - ix
    ty = y - iy

    wx = cubic_bspline_weights(tx)          # (N, 4)
    wy = cubic_bspline_weights(ty)          # (N, 4)

    ix = ix.long()
    iy = iy.long()
    offs = torch.arange(-1, 3, device=grid.device)
    cols = torch.clamp(ix[:, None] + offs[None, :], 0, W - 1)   # (N, 4)
    rows = torch.clamp(iy[:, None] + offs[None, :], 0, H - 1)   # (N, 4)

    # (C, N, 4, 4) patch of coefficients, then contract with the separable weights.
    patch = grid[:, rows[:, :, None].expand(-1, 4, 4), cols[:, None, :].expand(-1, 4, 4)]
    out = torch.einsum("cnij,ni,nj->nc", patch, wy, wx)
    return out


class GridCNN(nn.Module):
    """Convolutional decoder from a learned latent grid to a continuous field.

    This is the grid-based counterpart of the coordinate networks: the field is
    stored as pixels and read back with a smooth interpolant. It gets the CNN's
    spatial inductive bias, at the cost of tying resolution to the grid and
    losing the mesh-free treatment of the domain outline.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        latent: int = 16,
        channels: int = 64,
        n_up: int = 3,
        activation: str = "tanh",
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.latent_size = latent
        self.act = get_activation(activation)
        self.z = nn.Parameter(torch.randn(1, channels, latent, latent) * 0.05)

        # Optional FiLM conditioning (used for time in transient runs).
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.film = nn.Sequential(
                nn.Linear(cond_dim, channels), get_activation(activation),
                nn.Linear(channels, 2 * channels),
            )

        blocks = []
        for _ in range(n_up):
            blocks += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                get_activation(activation),
                nn.Conv2d(channels, channels, 3, padding=1),
                get_activation(activation),
            ]
        self.decoder = nn.Sequential(*blocks)
        self.head = nn.Conv2d(channels, out_dim, 3, padding=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def field(self, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.z
        if self.cond_dim > 0 and cond is not None:
            gb = self.film(cond.reshape(1, -1))
            gamma, beta = gb.chunk(2, dim=-1)
            z = z * (1.0 + gamma[..., None, None]) + beta[..., None, None]
        return self.head(self.decoder(z))[0]         # (out_dim, H, W)

    def forward(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None):
        if coords is None:
            raise ValueError("GridCNN needs explicit normalised coordinates")
        cond = feats[:1, -self.cond_dim:] if self.cond_dim > 0 else None
        return bspline_sample(self.field(cond), coords[:, :2])


# --------------------------------------------------------------------------- #


def build_backbone(
    arch: str,
    in_dim: int,
    out_dim: int,
    width: int = 128,
    depth: int = 5,
    activation: str = "tanh",
    cnn_latent: int = 16,
    cnn_channels: int = 64,
    cond_dim: int = 0,
) -> nn.Module:
    arch = arch.lower()
    if arch == "mlp":
        return MLP(in_dim, out_dim, width, depth, activation)
    if arch == "resnet":
        return ResNetMLP(in_dim, out_dim, width, depth, activation)
    if arch in ("modified_mlp", "modified", "mmlp"):
        return ModifiedMLP(in_dim, out_dim, width, depth, activation)
    if arch == "cnn":
        n_up = max(1, int(round(math.log2(max(width, 32) / max(cnn_latent, 1)))))
        return GridCNN(
            in_dim, out_dim, latent=cnn_latent, channels=cnn_channels,
            n_up=n_up, activation=activation, cond_dim=cond_dim,
        )
    raise ValueError(f"unknown architecture {arch!r}")


__all__ = [
    "MLP",
    "ResNetMLP",
    "ModifiedMLP",
    "GridCNN",
    "build_backbone",
    "bspline_sample",
    "cubic_bspline_weights",
    "get_activation",
]
