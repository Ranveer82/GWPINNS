"""Geostatistics and accuracy assessment."""

from gwpinn.stats.variogram import (  # noqa: F401
    VariogramModel,
    experimental_variogram,
    fit_variogram,
    ordinary_kriging,
)
from gwpinn.stats.metrics import (  # noqa: F401
    morans_i,
    regression_metrics,
    spatial_error_summary,
)

__all__ = [
    "VariogramModel",
    "experimental_variogram",
    "fit_variogram",
    "ordinary_kriging",
    "regression_metrics",
    "morans_i",
    "spatial_error_summary",
]
