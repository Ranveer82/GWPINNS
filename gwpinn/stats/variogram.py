"""Variograms, variogram fitting, and ordinary kriging.

The point aquifer-property measurements are too sparse to pin down a
heterogeneous field on their own, but they do carry information about its
*spatial structure*: how quickly transmissivity decorrelates with distance, and
how much total variance there is. That structure is summarised by a variogram
and then imposed on the network's property field as a soft constraint (see
:func:`gwpinn.train.losses.variogram_loss`), which is what keeps the inversion
from producing a field that fits the heads but is geologically implausible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist

MODELS = ("exponential", "spherical", "gaussian")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass
class VariogramModel:
    """A fitted theoretical variogram.

    ``sill`` is the *total* sill (nugget included); ``range_`` is the practical
    range, i.e. the lag at which 95% of the sill is reached.
    """

    model: str = "exponential"
    nugget: float = 0.0
    sill: float = 1.0
    range_: float = 1.0

    # ------------------------------------------------------------------ #

    @property
    def partial_sill(self) -> float:
        return max(self.sill - self.nugget, 0.0)

    def _shape(self, h):
        """Normalised structure function, 0 at h=0 and ->1 at large h."""
        a = max(self.range_, 1e-9)
        if self.model == "exponential":
            return 1.0 - _exp(-3.0 * h / a)
        if self.model == "gaussian":
            return 1.0 - _exp(-3.0 * (h / a) ** 2)
        if self.model == "spherical":
            r = _clip(h / a, 0.0, 1.0)
            return 1.5 * r - 0.5 * r**3
        raise ValueError(f"unknown variogram model {self.model!r}")

    def gamma(self, h):
        """Semivariance at lag ``h`` (numpy array or torch tensor)."""
        zero = _is_zero(h)
        g = self.nugget + self.partial_sill * self._shape(h)
        return _where(zero, _zeros_like(g), g)

    def covariance(self, h):
        return self.sill - self.gamma(h)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "nugget": float(self.nugget),
            "sill": float(self.sill),
            "range": float(self.range_),
            "partial_sill": float(self.partial_sill),
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"VariogramModel({self.model}, nugget={self.nugget:.4g}, "
            f"sill={self.sill:.4g}, range={self.range_:.4g})"
        )


# Small helpers so ``gamma`` works for both numpy arrays and torch tensors.
def _exp(x):
    return torch.exp(x) if torch.is_tensor(x) else np.exp(x)


def _clip(x, lo, hi):
    return torch.clamp(x, lo, hi) if torch.is_tensor(x) else np.clip(x, lo, hi)


def _is_zero(x):
    return (x == 0) if torch.is_tensor(x) else (np.asarray(x) == 0)


def _where(cond, a, b):
    return torch.where(cond, a, b) if torch.is_tensor(b) else np.where(cond, a, b)


def _zeros_like(x):
    return torch.zeros_like(x) if torch.is_tensor(x) else np.zeros_like(np.asarray(x, dtype=float))


# --------------------------------------------------------------------------- #
# Experimental variogram
# --------------------------------------------------------------------------- #


def experimental_variogram(
    xy: np.ndarray,
    values: np.ndarray,
    n_lags: int = 12,
    max_lag: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matheron estimator of the semivariogram.

    Returns ``(lag_centres, gamma, counts)`` for bins that contain at least one
    pair; empty bins are dropped.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    v = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(v)
    xy, v = xy[ok], v[ok]
    if len(v) < 3:
        raise ValueError("need at least 3 finite values for a variogram")

    d = pdist(xy)
    dv = 0.5 * pdist(v[:, None]) ** 2  # 0.5 * (v_i - v_j)^2

    if max_lag is None:
        max_lag = float(np.percentile(d, 60))
    max_lag = max(max_lag, np.min(d[d > 0]) * 2 if np.any(d > 0) else 1.0)

    edges = np.linspace(0.0, max_lag, n_lags + 1)
    idx = np.digitize(d, edges) - 1
    keep = (idx >= 0) & (idx < n_lags)
    idx, d, dv = idx[keep], d[keep], dv[keep]

    counts = np.bincount(idx, minlength=n_lags)
    lags = np.bincount(idx, weights=d, minlength=n_lags)
    gam = np.bincount(idx, weights=dv, minlength=n_lags)

    good = counts > 0
    return lags[good] / counts[good], gam[good] / counts[good], counts[good]


def fit_variogram(
    xy: np.ndarray,
    values: np.ndarray,
    model: str = "exponential",
    n_lags: int = 12,
    max_lag: Optional[float] = None,
    nugget: Optional[float] = None,
    sill: Optional[float] = None,
    range_: Optional[float] = None,
) -> Tuple[VariogramModel, dict]:
    """Fit a theoretical variogram by count-weighted least squares.

    Any of ``nugget`` / ``sill`` / ``range_`` that is supplied is held fixed.
    Returns the model and a diagnostics dict holding the experimental points.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")

    lags, gam, counts = experimental_variogram(xy, values, n_lags, max_lag)
    var = float(np.nanvar(np.asarray(values, dtype=float)))
    span = float(lags.max()) if lags.size else 1.0

    # Starting guesses: sill ~ sample variance, range ~ half the sampled span,
    # nugget ~ the smallest-lag semivariance.
    p0 = np.array(
        [
            nugget if nugget is not None else min(gam[0] * 0.5, 0.5 * max(var, 1e-12)),
            sill if sill is not None else max(var, float(gam.max()), 1e-12),
            range_ if range_ is not None else max(0.5 * span, 1e-6),
        ]
    )
    lo = np.array([0.0, 1e-12, 1e-6])
    # The range is capped just above the sampled extent. A correlation length
    # longer than the data cannot be resolved, and with few points the fit will
    # happily run it to infinity - which then reads as "no spatial structure"
    # and silently disables the variogram constraint.
    hi = np.array(
        [max(var, gam.max()) * 1.5 + 1e-9, max(var, gam.max()) * 5 + 1e-9, span * 1.5]
    )
    free = np.array([nugget is None, sill is None, range_ is None])

    w = np.sqrt(counts / counts.sum())

    def residual(theta_free):
        theta = p0.copy()
        theta[free] = theta_free
        m = VariogramModel(model, float(theta[0]), float(max(theta[1], theta[0])), float(theta[2]))
        return w * (m.gamma(lags) - gam)

    if free.any():
        sol = least_squares(
            residual,
            np.clip(p0[free], lo[free], hi[free]),
            bounds=(lo[free], hi[free]),
            max_nfev=2000,
        )
        theta = p0.copy()
        theta[free] = sol.x
    else:
        theta = p0

    fitted = VariogramModel(
        model=model,
        nugget=float(theta[0]),
        sill=float(max(theta[1], theta[0])),
        range_=float(theta[2]),
    )

    pred = fitted.gamma(lags)
    ss_res = float(np.sum(counts * (pred - gam) ** 2))
    ss_tot = float(np.sum(counts * (gam - np.average(gam, weights=counts)) ** 2))

    diag = {
        "lags": lags,
        "gamma": gam,
        "counts": counts,
        "sample_variance": var,
        "fit_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_points": int(np.isfinite(np.asarray(values, dtype=float)).sum()),
    }
    return fitted, diag


def fit_variogram_by_layer(
    xy: np.ndarray,
    values: np.ndarray,
    layer: np.ndarray,
    n_layers: int,
    model: str = "exponential",
    n_lags: int = 12,
    max_lag: Optional[float] = None,
    min_points: int = 10,
    **fixed,
) -> Tuple[Dict[int, VariogramModel], Dict[int, dict]]:
    """Fit one variogram per aquifer layer.

    Fitting a single variogram across all layers is wrong and quietly ruinous:
    different layers have different means, so pooling them raw reports the
    *between-layer* contrast as spatial variance. A confined layer with
    S ~ 2e-4 pooled with an unconfined one at S ~ 0.1 yields a sill of order 1
    in log10 units and no usable range at all.

    Layers with at least ``min_points`` measurements get their own fit. Sparser
    layers borrow the correlation structure (range, relative nugget) from a
    pooled fit of the layer-mean-removed residuals, but keep their own sample
    variance as the sill - the standard way to share structure across
    populations that differ in level but not in texture.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    v = np.asarray(values, dtype=float).ravel()
    layer = np.asarray(layer, dtype=int).ravel()
    finite = np.isfinite(v)

    # ---- pooled structure from mean-removed residuals --------------------
    pooled: Optional[VariogramModel] = None
    resid = v.copy()
    for l in np.unique(layer[finite]):
        m = finite & (layer == l)
        if m.sum() >= 2:
            resid[m] = v[m] - v[m].mean()
    m = finite
    if m.sum() >= 5:
        try:
            pooled, _ = fit_variogram(
                xy[m], resid[m], model=model, n_lags=n_lags, max_lag=max_lag, **fixed
            )
        except (ValueError, np.linalg.LinAlgError):
            pooled = None

    models: Dict[int, VariogramModel] = {}
    diags: Dict[int, dict] = {}

    for l in range(n_layers):
        m = finite & (layer == l)
        n = int(m.sum())

        if n >= min_points:
            try:
                models[l], diags[l] = fit_variogram(
                    xy[m], v[m], model=model, n_lags=n_lags, max_lag=max_lag, **fixed
                )
                diags[l]["source"] = "per-layer fit"
                continue
            except (ValueError, np.linalg.LinAlgError):
                pass

        if pooled is None:
            continue

        sill = float(np.var(v[m])) if n >= 4 else pooled.sill
        sill = max(sill, 1e-9)
        frac_nugget = pooled.nugget / max(pooled.sill, 1e-12)
        models[l] = VariogramModel(
            model=pooled.model,
            nugget=frac_nugget * sill,
            sill=sill,
            range_=pooled.range_,
        )
        diags[l] = {
            "source": f"pooled structure (n={n} < {min_points})",
            "n_points": n,
            "lags": np.zeros(0),
            "gamma": np.zeros(0),
            "counts": np.zeros(0),
            "fit_r2": float("nan"),
            "sample_variance": sill,
        }

    return models, diags


# --------------------------------------------------------------------------- #
# Kriging
# --------------------------------------------------------------------------- #


def ordinary_kriging(
    xy: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    model: VariogramModel,
    max_points: int = 400,
    ridge: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging with the fitted variogram.

    Returns ``(estimate, variance)``. The kriging variance is used to weight the
    geostatistical prior: near a measurement the prior is trusted, far from one
    it is all but switched off and the physics decides.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    v = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(v)
    xy, v = xy[ok], v[ok]
    query = np.asarray(query, dtype=float).reshape(-1, 2)

    n = len(v)
    if n == 0:
        return np.full(len(query), np.nan), np.full(len(query), np.nan)
    if n == 1:
        return np.full(len(query), v[0]), np.full(len(query), model.sill)
    if n > max_points:  # keep the linear solve small
        sel = np.random.default_rng(0).choice(n, max_points, replace=False)
        xy, v, n = xy[sel], v[sel], max_points

    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    G = model.gamma(d)
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = G + ridge * np.eye(n)
    A[n, n] = 0.0

    dq = np.linalg.norm(query[:, None, :] - xy[None, :, :], axis=2)
    b = np.ones((n + 1, len(query)))
    b[:n, :] = model.gamma(dq).T

    try:
        lam = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(A, b, rcond=None)[0]

    est = lam[:n, :].T @ v
    var = np.einsum("ij,ij->j", lam, b)
    return est, np.maximum(var, 0.0)


__all__ = [
    "VariogramModel",
    "experimental_variogram",
    "fit_variogram",
    "ordinary_kriging",
    "MODELS",
]
