"""Coordinate-network backbones for physics-informed surrogates.

The architectures here are the ones that have actually moved the needle on PINN
accuracy in the literature, implemented so they can be swapped on one axis of
the study while everything else is held fixed.

``mlp``
    Plain tanh MLP.  The control.

``modified``
    Wang, Teng & Perdikaris (2021), "Understanding and mitigating gradient
    pathologies in PINNs".  Two global encoders ``U``, ``V`` gate every hidden
    layer multiplicatively.  Costs one extra matmul and consistently beats the
    plain MLP on stiff problems, because the gating gives the network a cheap
    way to represent multiplicative interactions between inputs.

``pirate``
    Wang et al. (2024), "PirateNets: physics-informed deep learning with
    residual adaptive networks".  Adds (a) an adaptive
    residual connection per block with a *trainable* skip weight initialised at
    zero, so the network starts as a shallow linear map and deepens itself only
    as the data demands, and (b) random weight factorisation.  This is the
    current state of the art for deep PINNs and is the only one here that does
    not degrade as depth grows.

``spinn``
    Cho et al. (2023), "Separable PINN".  Instead of one network on
    ``(x, y, t)``, it uses one small network *per axis* and combines them with a
    tensor product.  For a rank-``r`` factorisation, evaluating an
    ``nx x ny x nt`` lattice costs ``nx + ny + nt`` network calls instead of
    ``nx*ny*nt``.  On a transient problem with a 2-hourly tide - where the time
    axis needs hundreds of samples - that is the difference between a tractable
    and an intractable collocation budget.

Random weight factorisation (``rwf``) parameterises every weight as
``W = diag(exp(s)) V`` with ``s`` trainable.  It rescales each neuron's learning
rate individually and measurably accelerates convergence (Wang, Sankaran, Wang &
Perdikaris, 2023); it is available to every backbone here.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class RWFLinear(nn.Module):
    """Linear layer with optional random weight factorisation.

    ``W = diag(exp(s)) V``.  The scale vector ``s`` is initialised from
    ``N(mu, sigma)`` and trained, which gives every output neuron its own
    effective learning rate.  With ``rwf=False`` this is a plain ``nn.Linear``
    with the same initialisation, so the two are directly comparable.
    """

    def __init__(self, n_in: int, n_out: int, rwf: bool = False,
                 mu: float = 1.0, sigma: float = 0.1) -> None:
        super().__init__()
        w = torch.empty(n_out, n_in)
        nn.init.xavier_normal_(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.rwf = bool(rwf)
        if self.rwf:
            s = torch.randn(n_out) * sigma + mu
            self.s = nn.Parameter(s)
            self.v = nn.Parameter(w / torch.exp(s)[:, None])
        else:
            self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.exp(self.s)[:, None] * self.v if self.rwf else self.weight
        return torch.nn.functional.linear(x, w, self.bias)


class FourierFeatures(nn.Module):
    """Random Fourier embedding ``x -> [sin(Bx), cos(Bx)]``.

    Tancik et al. (2020).  Without it a tanh network is spectrally biased toward
    smooth functions and cannot represent a head field shaped by a K field with
    a two-cell correlation length.  ``B`` is fixed (not trained): training it
    tends to collapse the bandwidth early and lose the high frequencies for good.
    """

    def __init__(self, n_in: int, n_features: int, sigma: float = 3.0,
                 seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(n_in, n_features, generator=g) * sigma)

    @property
    def n_out(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return {
        "tanh": torch.tanh,
        "gelu": torch.nn.functional.gelu,
        "silu": torch.nn.functional.silu,
        "sin": torch.sin,
    }[name]


# --------------------------------------------------------------------------- #
# Backbones
# --------------------------------------------------------------------------- #


class MLP(nn.Module):
    """Plain fully-connected network."""

    def __init__(self, n_in: int, n_out: int, width: int = 96, depth: int = 4,
                 activation: str = "tanh", rwf: bool = False) -> None:
        super().__init__()
        self.act = _activation(activation)
        dims = [n_in] + [width] * depth
        self.layers = nn.ModuleList(
            [RWFLinear(a, b, rwf) for a, b in zip(dims[:-1], dims[1:])]
        )
        self.out = RWFLinear(width, n_out, rwf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.layers:
            x = self.act(lin(x))
        return self.out(x)


class ModifiedMLP(nn.Module):
    """Wang et al. (2021) gated MLP."""

    def __init__(self, n_in: int, n_out: int, width: int = 96, depth: int = 4,
                 activation: str = "tanh", rwf: bool = False) -> None:
        super().__init__()
        self.act = _activation(activation)
        self.enc_u = RWFLinear(n_in, width, rwf)
        self.enc_v = RWFLinear(n_in, width, rwf)
        dims = [n_in] + [width] * depth
        self.layers = nn.ModuleList(
            [RWFLinear(a, b, rwf) for a, b in zip(dims[:-1], dims[1:])]
        )
        self.out = RWFLinear(width, n_out, rwf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act(self.enc_u(x))
        v = self.act(self.enc_v(x))
        z = x
        for lin in self.layers:
            z = self.act(lin(z))
            # The gate: a convex-ish blend of the two global encodings, which
            # lets the network modulate features multiplicatively.
            z = z * u + (1.0 - z) * v
        return self.out(z)


class PirateBlock(nn.Module):
    """One adaptive residual block with a trainable, zero-initialised skip."""

    def __init__(self, width: int, activation: str, rwf: bool) -> None:
        super().__init__()
        self.act = _activation(activation)
        self.f1 = RWFLinear(width, width, rwf)
        self.f2 = RWFLinear(width, width, rwf)
        self.f3 = RWFLinear(width, width, rwf)
        # alpha = 0 => the block is the identity at initialisation, so a deep
        # PirateNet starts out behaving like its (well-conditioned) shallow
        # counterpart and only recruits depth as training needs it.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        f = self.act(self.f1(x))
        f = f * u + (1.0 - f) * v
        g = self.act(self.f2(f))
        g = g * u + (1.0 - g) * v
        h = self.act(self.f3(g))
        return self.alpha * h + (1.0 - self.alpha) * x


class PirateNet(nn.Module):
    """Wang et al. (2024) PirateNet: gated encoders + adaptive residual blocks."""

    def __init__(self, n_in: int, n_out: int, width: int = 96, depth: int = 3,
                 activation: str = "tanh", rwf: bool = True) -> None:
        super().__init__()
        self.act = _activation(activation)
        self.enc_u = RWFLinear(n_in, width, rwf)
        self.enc_v = RWFLinear(n_in, width, rwf)
        self.proj = RWFLinear(n_in, width, rwf)
        self.blocks = nn.ModuleList(
            [PirateBlock(width, activation, rwf) for _ in range(depth)]
        )
        self.out = RWFLinear(width, n_out, rwf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act(self.enc_u(x))
        v = self.act(self.enc_v(x))
        z = self.act(self.proj(x))
        for blk in self.blocks:
            z = blk(z, u, v)
        return self.out(z)


class SeparableNet(nn.Module):
    """Separable PINN (Cho et al., 2023): one sub-network per coordinate axis.

    Each axis ``i`` has its own MLP ``f_i: R -> R^{r*n_out}``.  The output is the
    rank-``r`` tensor contraction

    ``h(x, y, t) = sum_j f_x[j] * f_y[j] * f_t[j]``.

    Two consequences matter here.  First, on a lattice the cost is additive in
    the axis sizes rather than multiplicative, so a dense time axis is nearly
    free - exactly what a 2-hourly tide needs.  Second, the factorisation is a
    low-rank prior: it represents smooth, separable structure very efficiently
    and genuinely non-separable structure (a meandering river, an oblique fault)
    only as ``r`` grows.  That tension is the interesting thing to measure.
    """

    def __init__(self, n_in: int, n_out: int, width: int = 64, depth: int = 3,
                 rank: int = 32, activation: str = "tanh", rwf: bool = False) -> None:
        super().__init__()
        self.n_in, self.n_out, self.rank = n_in, n_out, rank
        self.nets = nn.ModuleList([
            MLP(1, rank * n_out, width, depth, activation, rwf) for _ in range(n_in)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pointwise evaluation: ``x`` is ``(N, n_in)`` and the output ``(N, n_out)``.

        Lattice evaluation would be the cheap path, but every formulation in this
        study samples scattered collocation points (they have to - the river and
        the faults are not lattice-aligned), so the pointwise contraction is what
        is actually used.  The saving that remains is in parameter count and in
        the conditioning of the per-axis problems.
        """
        feats = [net(x[:, i : i + 1]).view(-1, self.rank, self.n_out)
                 for i, net in enumerate(self.nets)]
        prod = feats[0]
        for f in feats[1:]:
            prod = prod * f
        return prod.sum(dim=1)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


class CoordinateNet(nn.Module):
    """A backbone plus its input feature map.

    Input assembly is ``[x_hat, gamma(x_hat), extra]`` where ``x_hat`` is the
    normalised coordinate, ``gamma`` the optional Fourier embedding and ``extra``
    any problem-specific features (the signed fault-side indicators).  Keeping
    this in one place means every architecture sees exactly the same inputs, so
    the architecture axis of the study is not silently confounded with a feature
    axis.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        arch: str = "mlp",
        width: int = 96,
        depth: int = 4,
        activation: str = "tanh",
        fourier: int = 0,
        fourier_sigma: float = 3.0,
        n_extra: int = 0,
        extra_in_fourier: bool = False,
        rank: int = 32,
        rwf: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_in, self.n_extra = n_in, n_extra
        self.extra_in_fourier = bool(extra_in_fourier)

        embed_in = n_in + (n_extra if extra_in_fourier else 0)
        self.embed = (
            FourierFeatures(embed_in, fourier, fourier_sigma, seed) if fourier > 0 else None
        )
        feat = embed_in + (self.embed.n_out if self.embed is not None else 0)
        if not extra_in_fourier:
            feat += n_extra

        kw = dict(width=width, depth=depth, activation=activation, rwf=rwf)
        if arch == "mlp":
            self.body: nn.Module = MLP(feat, n_out, **kw)
        elif arch == "modified":
            self.body = ModifiedMLP(feat, n_out, **kw)
        elif arch == "pirate":
            self.body = PirateNet(feat, n_out, **kw)
        elif arch == "spinn":
            # Separable networks cannot consume a Fourier embedding of the joint
            # coordinate (the embedding is not separable), so they take the raw
            # normalised coordinates and any extra features as additional axes.
            self.body = SeparableNet(n_in + n_extra, n_out, width=width, depth=depth,
                                     rank=rank, activation=activation, rwf=rwf)
            self.embed = None
        else:
            raise ValueError(f"unknown arch {arch!r}")
        self.arch = arch

    def forward(self, coords: torch.Tensor,
                extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.arch == "spinn":
            z = coords if extra is None else torch.cat([coords, extra], dim=-1)
            return self.body(z)
        if self.embed is None:
            parts = [coords]
        elif self.extra_in_fourier and extra is not None:
            xi = torch.cat([coords, extra], dim=-1)
            parts = [xi, self.embed(xi)]
        else:
            parts = [coords, self.embed(coords)]
        if extra is not None and not self.extra_in_fourier:
            parts.append(extra)
        return self.body(torch.cat(parts, dim=-1))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


__all__ = [
    "CoordinateNet", "FourierFeatures", "MLP", "ModifiedMLP", "PirateNet",
    "SeparableNet", "RWFLinear", "count_parameters",
]
