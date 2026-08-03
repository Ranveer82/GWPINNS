"""GWPINNS -- inverse PINNs for 3D transient groundwater flow across a fault.

Sub-packages
------------
``gwpinns.benchmark``
    Phase 1: FloPy/MODFLOW 6 synthetic benchmark plus a reference
    finite-difference solver and the sparse monitoring-well sampler.
``gwpinns.pinn``
    Phase 2: the three PyTorch architectures (baseline, mixed-variable, cPINN),
    the hand-written autograd physics residuals and the trainer.
``gwpinns.evaluation``
    Metrics and figures for comparing recovered K fields and fault behaviour.
"""

from .config import BenchmarkConfig, SCENARIOS

__version__ = "0.1.0"
__all__ = ["BenchmarkConfig", "SCENARIOS", "__version__"]
