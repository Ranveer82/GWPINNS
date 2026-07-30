"""gwpinn - physics-informed neural networks for groundwater flow inversion.

Estimates a multi-layer groundwater head field together with heterogeneous
aquifer property fields (transmissivity / hydraulic conductivity, storage
coefficient) from sparse field observations, subject to the quasi-3D
groundwater flow equation.
"""

__version__ = "0.1.0"

from gwpinn.config import Config, load_config  # noqa: F401

__all__ = ["Config", "load_config", "__version__"]
