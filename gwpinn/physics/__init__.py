"""Groundwater flow physics expressed through automatic differentiation."""

from gwpinn.physics.operators import (  # noqa: F401
    divergence,
    grad,
    smooth_clamp,
)
from gwpinn.physics.gwflow import (  # noqa: F401
    GroundwaterFlow,
    LayerElevations,
    Sources,
)
from gwpinn.physics.bcs import BoundaryConditions, boundary_residual  # noqa: F401

__all__ = [
    "grad",
    "divergence",
    "smooth_clamp",
    "GroundwaterFlow",
    "LayerElevations",
    "Sources",
    "BoundaryConditions",
    "boundary_residual",
]
