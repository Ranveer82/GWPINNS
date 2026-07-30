"""Field wrappers: normalisation, feature assembly, and physical output ranges.

Two networks are trained jointly:

``HeadField``
    ``(x, y[, t]) -> h_l``, the piezometric head in every layer.
``PropertyField``
    ``(x, y) -> K_l, S_l``, hydraulic conductivity and storage coefficient.

Keeping them separate matters. The head field is smoothed by the flow equation
and is comparatively low-frequency; the property field is rough and is what the
inversion is really after. Giving the property network its own, higher-frequency
Fourier band lets each be represented at its natural scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from gwpinn.models.backbones import build_backbone
from gwpinn.models.fourier import RandomFourierFeatures


@dataclass
class Normalizer:
    """Maps physical coordinates/heads to the O(1) ranges networks train well on.

    Non-dimensionalisation is not cosmetic here: with raw metre-scale coordinates
    and metre-scale heads the PDE residual spans many orders of magnitude and the
    optimiser stalls almost immediately.
    """

    x0: float = 0.0
    y0: float = 0.0
    scale: float = 1.0            # half-diagonal, so x_n stays within ~[-1, 1]
    h_mean: float = 0.0
    h_std: float = 1.0
    t_max: float = 1.0

    @classmethod
    def from_bounds(
        cls,
        bounds: Tuple[float, float, float, float],
        heads: Optional[np.ndarray] = None,
        t_max: float = 1.0,
    ) -> "Normalizer":
        xmin, ymin, xmax, ymax = bounds
        scale = 0.5 * max(xmax - xmin, ymax - ymin)
        h_mean, h_std = 0.0, 1.0
        if heads is not None:
            h = np.asarray(heads, dtype=float)
            h = h[np.isfinite(h)]
            if h.size:
                h_mean = float(h.mean())
                h_std = float(max(h.std(), 1e-3))
        return cls(
            x0=0.5 * (xmin + xmax),
            y0=0.5 * (ymin + ymax),
            scale=float(max(scale, 1e-9)),
            h_mean=h_mean,
            h_std=h_std,
            t_max=float(max(t_max, 1e-9)),
        )

    # ------------------------------------------------------------------ #

    def norm_xy(self, xy: torch.Tensor) -> torch.Tensor:
        origin = torch.tensor([self.x0, self.y0], dtype=xy.dtype, device=xy.device)
        return (xy - origin) / self.scale

    def denorm_xy(self, xyn: torch.Tensor) -> torch.Tensor:
        origin = torch.tensor([self.x0, self.y0], dtype=xyn.dtype, device=xyn.device)
        return xyn * self.scale + origin

    def norm_t(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * t / self.t_max - 1.0

    def norm_h(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.h_mean) / self.h_std

    def denorm_h(self, hn: torch.Tensor) -> torch.Tensor:
        return hn * self.h_std + self.h_mean


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


# --------------------------------------------------------------------------- #


class _FeatureMixin(nn.Module):
    """Shared input assembly: normalised coords + Fourier bands + fault sides."""

    def _build_features(
        self,
        in_dim: int,
        n_fourier: int,
        sigmas: Sequence[float],
        n_fault_feats: int,
        dtype: torch.dtype,
        generator: Optional[torch.Generator],
    ) -> int:
        self.embed = RandomFourierFeatures(
            in_dim, n_fourier, sigmas, dtype=dtype, generator=generator
        )
        self.n_fault_feats = int(n_fault_feats)
        return in_dim + self.embed.out_dim + self.n_fault_feats

    def _assemble(
        self, coords: torch.Tensor, fault_feats: Optional[torch.Tensor]
    ) -> torch.Tensor:
        parts = [coords, self.embed(coords)]
        if self.n_fault_feats > 0:
            if fault_feats is None:
                parts.append(coords.new_zeros(coords.shape[0], self.n_fault_feats))
            else:
                parts.append(fault_feats)
        return torch.cat(parts, dim=-1)


class HeadField(_FeatureMixin):
    """Piezometric head in every aquifer layer."""

    def __init__(
        self,
        n_layers: int,
        normalizer: Normalizer,
        arch: str = "modified_mlp",
        width: int = 128,
        depth: int = 5,
        activation: str = "tanh",
        n_fourier: int = 64,
        fourier_sigma: float = 3.0,
        n_fault_feats: int = 0,
        transient: bool = False,
        dtype: torch.dtype = torch.float32,
        cnn_latent: int = 16,
        cnn_channels: int = 64,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.norm = normalizer
        self.transient = transient

        coord_dim = 3 if transient else 2
        sigmas = (0.5 * fourier_sigma, fourier_sigma, 2.0 * fourier_sigma)
        in_dim = self._build_features(
            coord_dim, n_fourier, sigmas, n_fault_feats, dtype, generator
        )

        self.backbone = build_backbone(
            arch, in_dim, n_layers, width, depth, activation,
            cnn_latent=cnn_latent, cnn_channels=cnn_channels,
            cond_dim=1 if (transient and arch == "cnn") else 0,
        )
        self.to(dtype)

    def forward(
        self,
        xy: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        fault_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        coords = self.norm.norm_xy(xy)
        if self.transient:
            if t is None:
                t = xy.new_zeros(xy.shape[0], 1)
            coords = torch.cat([coords, self.norm.norm_t(t.reshape(-1, 1))], dim=-1)
        feats = self._assemble(coords, fault_feats)
        return self.norm.denorm_h(self.backbone(feats, coords))


class PropertyField(_FeatureMixin):
    """Hydraulic conductivity and storage coefficient of every layer.

    Both are predicted in log space and squashed into physical bounds with a
    sigmoid, which keeps them strictly positive without a penalty term and stops
    the optimiser walking off into unphysical values early in training.
    """

    def __init__(
        self,
        n_layers: int,
        normalizer: Normalizer,
        arch: str = "modified_mlp",
        width: int = 96,
        depth: int = 4,
        activation: str = "tanh",
        n_fourier: int = 64,
        fourier_sigma: float = 6.0,
        n_fault_feats: int = 0,
        k_bounds: Tuple[float, float] = (1e-3, 5e2),
        s_bounds: Tuple[float, float] = (1e-5, 3e-1),
        dtype: torch.dtype = torch.float32,
        cnn_latent: int = 16,
        cnn_channels: int = 64,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.norm = normalizer

        sigmas = (0.5 * fourier_sigma, fourier_sigma, 2.0 * fourier_sigma)
        in_dim = self._build_features(2, n_fourier, sigmas, n_fault_feats, dtype, generator)

        self.backbone = build_backbone(
            arch, in_dim, 2 * n_layers, width, depth, activation,
            cnn_latent=cnn_latent, cnn_channels=cnn_channels,
        )

        self.logk_min = float(np.log10(k_bounds[0]))
        self.logk_max = float(np.log10(k_bounds[1]))
        self.logs_min = float(np.log10(s_bounds[0]))
        self.logs_max = float(np.log10(s_bounds[1]))

        # Per-layer offsets, set from the point measurements so the field starts
        # at the observed geometric mean instead of the middle of the bounds.
        self.offset = nn.Parameter(torch.zeros(2 * n_layers, dtype=dtype))
        self.to(dtype)

    # ------------------------------------------------------------------ #

    def set_prior(
        self,
        log10k: Optional[Sequence[float]] = None,
        log10s: Optional[Sequence[float]] = None,
    ) -> None:
        """Centre the field on given per-layer log10 means."""
        with torch.no_grad():
            if log10k is not None:
                for l, v in enumerate(log10k[: self.n_layers]):
                    if np.isfinite(v):
                        frac = (float(v) - self.logk_min) / (self.logk_max - self.logk_min)
                        self.offset[l] = _logit(frac)
            if log10s is not None:
                for l, v in enumerate(log10s[: self.n_layers]):
                    if np.isfinite(v):
                        frac = (float(v) - self.logs_min) / (self.logs_max - self.logs_min)
                        self.offset[self.n_layers + l] = _logit(frac)

    def forward(
        self, xy: torch.Tensor, fault_feats: Optional[torch.Tensor] = None
    ) -> dict:
        coords = self.norm.norm_xy(xy)
        feats = self._assemble(coords, fault_feats)
        raw = self.backbone(feats, coords) + self.offset[None, :]

        gk = torch.sigmoid(raw[:, : self.n_layers])
        gs = torch.sigmoid(raw[:, self.n_layers :])

        log10k = self.logk_min + (self.logk_max - self.logk_min) * gk
        log10s = self.logs_min + (self.logs_max - self.logs_min) * gs

        return {
            "log10K": log10k,
            "K": torch.pow(10.0, log10k),
            "log10S": log10s,
            "S": torch.pow(10.0, log10s),
        }


__all__ = ["Normalizer", "HeadField", "PropertyField"]
