"""Accuracy metrics - aggregate and spatial."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from gwpinn.stats.variogram import experimental_variogram


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #


def regression_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Standard goodness-of-fit measures for a head or property comparison.

    Includes the hydrology-specific ones (NSE, KGE, PBIAS) alongside RMSE/MAE so
    the result is directly comparable with the calibration statistics reported
    for conventional groundwater models.
    """
    obs = np.asarray(obs, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    n = obs.size
    if n == 0:
        return {k: float("nan") for k in _METRIC_KEYS} | {"n": 0}

    err = pred - obs
    obs_mean = obs.mean()
    ss_tot = float(((obs - obs_mean) ** 2).sum())

    rmse = float(np.sqrt((err**2).mean()))
    obs_range = float(obs.max() - obs.min())
    obs_std = float(obs.std(ddof=1)) if n > 1 else float("nan")

    # Kling-Gupta efficiency (Gupta et al., 2009).
    if n > 1 and obs_std > 0 and pred.std(ddof=1) > 0:
        r = float(np.corrcoef(obs, pred)[0, 1])
        beta = float(pred.mean() / obs_mean) if obs_mean != 0 else float("nan")
        gamma = float((pred.std(ddof=1) / pred.mean()) / (obs_std / obs_mean)) if (
            obs_mean != 0 and pred.mean() != 0
        ) else float("nan")
        kge = (
            1.0 - float(np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))
            if np.isfinite(beta) and np.isfinite(gamma)
            else float("nan")
        )
    else:
        r, kge = float("nan"), float("nan")

    return {
        "n": int(n),
        "rmse": rmse,
        "mae": float(np.abs(err).mean()),
        "me": float(err.mean()),                       # bias
        "mae_std": float(np.abs(err).std()),
        "max_abs_err": float(np.abs(err).max()),
        "r2": float(1.0 - (err**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson_r": r,
        "nse": float(1.0 - (err**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "kge": kge,
        "pbias_pct": float(100.0 * err.sum() / obs.sum()) if obs.sum() != 0 else float("nan"),
        "nrmse_range_pct": float(100.0 * rmse / obs_range) if obs_range > 0 else float("nan"),
        "rsr": float(rmse / obs_std) if np.isfinite(obs_std) and obs_std > 0 else float("nan"),
        "willmott_d": _willmott(obs, pred),
    }


_METRIC_KEYS = (
    "rmse", "mae", "me", "mae_std", "max_abs_err", "r2", "pearson_r", "nse",
    "kge", "pbias_pct", "nrmse_range_pct", "rsr", "willmott_d",
)


def _willmott(obs: np.ndarray, pred: np.ndarray) -> float:
    """Willmott's index of agreement."""
    om = obs.mean()
    denom = float(((np.abs(pred - om) + np.abs(obs - om)) ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float(1.0 - ((pred - obs) ** 2).sum() / denom)


# --------------------------------------------------------------------------- #
# Spatial diagnostics
# --------------------------------------------------------------------------- #


def morans_i(
    xy: np.ndarray,
    values: np.ndarray,
    max_dist: Optional[float] = None,
    n_permutations: int = 199,
    seed: int = 0,
) -> Dict[str, float]:
    """Moran's I of the residuals with inverse-distance weights.

    Residuals from a well-specified model should be spatially unstructured. A
    significantly positive I means the fit is leaving coherent patches of
    over- or under-prediction behind - the signature of a property field that is
    still too smooth, or of a missing boundary condition.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    z = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(z)
    xy, z = xy[ok], z[ok]
    n = len(z)
    if n < 8:
        return {"morans_i": float("nan"), "p_value": float("nan"), "n": int(n)}

    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    if max_dist is None:
        max_dist = float(np.percentile(d[d > 0], 25))
    with np.errstate(divide="ignore"):
        W = np.where((d > 0) & (d <= max_dist), 1.0 / np.maximum(d, 1e-12), 0.0)
    np.fill_diagonal(W, 0.0)
    if W.sum() <= 0:
        return {"morans_i": float("nan"), "p_value": float("nan"), "n": int(n)}

    def compute(zv: np.ndarray) -> float:
        zc = zv - zv.mean()
        denom = float((zc**2).sum())
        if denom <= 0:
            return float("nan")
        return float(n / W.sum() * (zc @ W @ zc) / denom)

    obs_i = compute(z)

    rng = np.random.default_rng(seed)
    null = np.array([compute(rng.permutation(z)) for _ in range(n_permutations)])
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(obs_i):
        p = float("nan")
    else:
        # Two-sided pseudo p-value.
        more = float((np.abs(null) >= abs(obs_i)).sum())
        p = (more + 1.0) / (null.size + 1.0)

    return {
        "morans_i": obs_i,
        "expected_i": float(-1.0 / (n - 1)),
        "p_value": p,
        "max_dist": float(max_dist),
        "n": int(n),
    }


def spatial_error_summary(
    xy: np.ndarray,
    residuals: np.ndarray,
    n_lags: int = 10,
    seed: int = 0,
) -> Dict[str, object]:
    """Spatial structure of the residuals: Moran's I plus a residual variogram.

    A flat residual variogram (pure nugget) is the target: it means the error is
    white noise in space and nothing systematic is left to extract.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    r = np.asarray(residuals, dtype=float).ravel()
    out: Dict[str, object] = {"morans": morans_i(xy, r, seed=seed)}

    try:
        lags, gam, counts = experimental_variogram(xy, r, n_lags=n_lags)
        out["residual_variogram"] = {
            "lags": lags.tolist(),
            "gamma": gam.tolist(),
            "counts": counts.tolist(),
        }
        # Ratio of the semivariance at short lags to the overall variance: ~1
        # means uncorrelated, <<1 means spatially structured residuals.
        var = float(np.var(r))
        out["nugget_ratio"] = float(gam[0] / var) if var > 0 and gam.size else float("nan")
    except (ValueError, IndexError):
        out["residual_variogram"] = None
        out["nugget_ratio"] = float("nan")

    return out


def per_bin_metrics(
    values: np.ndarray,
    obs: np.ndarray,
    pred: np.ndarray,
    n_bins: int = 8,
    label: str = "bin",
) -> Dict[str, list]:
    """Break the error down by a covariate (e.g. distance to the nearest well)."""
    values = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(values) & np.isfinite(obs) & np.isfinite(pred)
    values, obs, pred = values[ok], obs[ok], pred[ok]
    if values.size == 0:
        return {label: [], "rmse": [], "me": [], "n": []}

    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)

    centers, rmses, mes, ns = [], [], [], []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() == 0:
            continue
        e = pred[m] - obs[m]
        centers.append(float(0.5 * (edges[k] + edges[k + 1])))
        rmses.append(float(np.sqrt((e**2).mean())))
        mes.append(float(e.mean()))
        ns.append(int(m.sum()))

    return {label: centers, "rmse": rmses, "me": mes, "n": ns}


__all__ = [
    "regression_metrics",
    "morans_i",
    "spatial_error_summary",
    "per_bin_metrics",
]
