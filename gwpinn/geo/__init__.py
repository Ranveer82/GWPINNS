"""Geometric preprocessing: domain, river stage, fault fields."""

from gwpinn.geo.grid import ModelDomain, build_domain  # noqa: F401
from gwpinn.geo.river import RiverStage, polygon_centerline, project_to_polyline  # noqa: F401
from gwpinn.geo.faults import FaultField  # noqa: F401

__all__ = [
    "ModelDomain",
    "build_domain",
    "RiverStage",
    "polygon_centerline",
    "project_to_polyline",
    "FaultField",
]
