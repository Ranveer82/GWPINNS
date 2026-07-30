"""Network architectures for the head and aquifer-property fields."""

from gwpinn.models.fourier import RandomFourierFeatures  # noqa: F401
from gwpinn.models.backbones import (  # noqa: F401
    MLP,
    GridCNN,
    ModifiedMLP,
    ResNetMLP,
    build_backbone,
)
from gwpinn.models.fields import HeadField, PropertyField  # noqa: F401

__all__ = [
    "RandomFourierFeatures",
    "MLP",
    "ResNetMLP",
    "ModifiedMLP",
    "GridCNN",
    "build_backbone",
    "HeadField",
    "PropertyField",
]
