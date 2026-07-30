"""Loss terms beyond the plain data and PDE misfits.

The inverse problem is badly under-determined: many transmissivity fields
reproduce the same sparse heads, because only the divergence of ``T grad(h)`` is
observed. The terms here supply the missing information from geostatistics -
the spatial *structure* implied by the pumping-test points - rather than from an
arbitrary smoothness assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from gwpinn.geo.grid import ModelDomain
from gwpinn.stats.variogram import VariogramModel, ordinary_kriging


# --------------------------------------------------------------------------- #
# Variogram structure
# --------------------------------------------------------------------------- #


def sample_lag_pairs(
    domain: ModelDomain,
    model: VariogramModel,
    n_pairs: int,
    rng: np.random.Generator,
    n_bins: int = 8,
    max_lag: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Point pairs at controlled separations.

    Random pairs of uniform points would concentrate at large lags and barely
    sample the short ones that carry the correlation-range information, so lags
    are stratified: pick an anchor, a target lag and a random direction.

    Returns ``(p1, p2, lag, bin_index)``.
    """
    if max_lag is None:
        max_lag = min(1.5 * model.range_, 0.5 * domain.diagonal)
    max_lag = float(max(max_lag, 1e-6))

    centers = np.linspace(max_lag / n_bins, max_lag, n_bins)
    per_bin = max(8, n_pairs // n_bins)

    p1 = domain.sample_interior(per_bin * n_bins, rng)
    lag = np.repeat(centers, per_bin)[: len(p1)]
    bin_idx = np.repeat(np.arange(n_bins), per_bin)[: len(p1)]

    theta = rng.uniform(0.0, 2.0 * np.pi, len(p1))
    p2 = p1 + lag[:, None] * np.column_stack([np.cos(theta), np.sin(theta)])

    keep = domain.contains(p2[:, 0], p2[:, 1])
    return p1[keep], p2[keep], lag[keep], bin_idx[keep]


def variogram_loss(
    values1: torch.Tensor,
    values2: torch.Tensor,
    lag: torch.Tensor,
    bin_idx: torch.Tensor,
    model: VariogramModel,
    n_bins: int = 8,
) -> torch.Tensor:
    """Mismatch between the field's own semivariogram and the fitted model.

    For each lag bin the empirical semivariance of the *predicted* field,
    :math:`\\hat\\gamma_k = \\langle \\tfrac12 (Y_1-Y_2)^2 \\rangle`, is compared
    with the theoretical :math:`\\gamma(d_k)`. Matching both the sill and the
    range forces the learned field to have the right variance *and* the right
    correlation length - a constraint a smoothness penalty cannot express, since
    smoothness only ever pushes the variance down.

    Normalised by the sill, so the term is dimensionless and comparable across
    properties.
    """
    if values1.numel() == 0:
        return values1.new_zeros(())

    semi = 0.5 * (values1 - values2) ** 2
    target = model.gamma(lag)

    # index_add rather than bincount: bincount has no derivative, and the whole
    # point of this term is to backpropagate through the predicted semivariance.
    zeros = semi.new_zeros(n_bins)
    counts = zeros.index_add(0, bin_idx, torch.ones_like(semi))
    pred_g = zeros.index_add(0, bin_idx, semi) / counts.clamp_min(1.0)
    targ_g = zeros.index_add(0, bin_idx, target) / counts.clamp_min(1.0)

    present = counts > 0
    if not bool(present.any()):
        return values1.new_zeros(())

    sill = max(model.sill, 1e-12)
    return (((pred_g[present] - targ_g[present]) / sill) ** 2).mean()


# --------------------------------------------------------------------------- #
# Kriging prior
# --------------------------------------------------------------------------- #


@dataclass
class KrigingPrior:
    """Ordinary-kriging estimate of a property field, with its own confidence.

    Kriging alone would give a field that honours the measurements but ignores
    the flow equation; the PINN alone would give one that fits the heads but
    drifts wherever the heads are insensitive. Weighting the pull toward the
    kriged surface by ``1/(1 + var/sill)`` hands control to the measurements
    near a pumping test and to the physics far from one.
    """

    xy: np.ndarray
    estimate: np.ndarray
    weight: np.ndarray
    name: str = "T"

    @classmethod
    def build(
        cls,
        domain: ModelDomain,
        obs_xy: np.ndarray,
        obs_values: np.ndarray,
        model: VariogramModel,
        n_points: int = 4096,
        rng: Optional[np.random.Generator] = None,
        name: str = "T",
    ) -> Optional["KrigingPrior"]:
        rng = rng or np.random.default_rng(0)
        obs_xy = np.asarray(obs_xy, dtype=float).reshape(-1, 2)
        obs_values = np.asarray(obs_values, dtype=float).ravel()
        ok = np.isfinite(obs_values)
        if ok.sum() < 3:
            return None

        pts = domain.sample_interior(n_points, rng)
        est, var = ordinary_kriging(obs_xy[ok], obs_values[ok], pts, model)
        good = np.isfinite(est)
        if not good.any():
            return None

        w = 1.0 / (1.0 + np.maximum(var[good], 0.0) / max(model.sill, 1e-12))
        return cls(xy=pts[good], estimate=est[good], weight=w, name=name)

    def batch(
        self, n: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(self.xy), min(n, len(self.xy)))
        return self.xy[idx], self.estimate[idx], self.weight[idx]


def kriging_loss(
    predicted: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    if predicted.numel() == 0:
        return predicted.new_zeros(())
    return (weight * (predicted - target) ** 2).sum() / weight.sum().clamp_min(1e-12)


# --------------------------------------------------------------------------- #
# Regularisation
# --------------------------------------------------------------------------- #


def smoothness_loss(
    field: torch.Tensor, xy: torch.Tensor, delta: float = 0.1, scale: float = 1.0
) -> torch.Tensor:
    """Pseudo-Huber penalty on the property gradient - a smooth total variation.

    A squared-gradient (Tikhonov) penalty smears out genuine facies contacts,
    because its cost grows without bound as a contrast sharpens. The pseudo-Huber
    is quadratic for small gradients and linear for large ones, so it suppresses
    speckle while leaving sharp boundaries essentially free - the standard
    edge-preserving choice in geophysical inversion.
    """
    from gwpinn.physics.operators import grad

    total = field.new_zeros(())
    n_cols = field.shape[1] if field.dim() > 1 else 1
    for c in range(n_cols):
        col = field[:, c] if field.dim() > 1 else field
        g = grad(col, xy) * scale
        mag2 = (g**2).sum(dim=1)
        total = total + (delta**2 * (torch.sqrt(1.0 + mag2 / delta**2) - 1.0)).mean()
    return total / max(n_cols, 1)


def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    noise: float = 0.0,
) -> torch.Tensor:
    """Mean squared error, optionally insensitive to residuals within ``noise``.

    Driving the misfit below the measurement accuracy is not a better fit, it is
    fitting the noise: the network spends capacity contorting the field through
    each well and the error between wells grows. With ``noise`` set, a residual
    smaller than the stated accuracy costs nothing, so once the data are honoured
    to within their uncertainty the physics and the geostatistics decide the
    rest. This is the same criterion as a target chi-square of one in classical
    groundwater calibration.
    """
    if pred.numel() == 0:
        return pred.new_zeros(())
    err = pred - target
    if noise > 0:
        err = torch.clamp(err.abs() - noise, min=0.0)
    err2 = err**2
    if weight is None:
        return err2.mean()
    return (weight * err2).sum() / weight.sum().clamp_min(1e-12)


__all__ = [
    "sample_lag_pairs",
    "variogram_loss",
    "KrigingPrior",
    "kriging_loss",
    "smoothness_loss",
    "weighted_mse",
]
