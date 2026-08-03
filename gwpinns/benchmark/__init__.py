"""Phase 1 -- synthetic benchmark generation."""

from .fields import (
    build_conductivity,
    build_et_max_rate,
    build_recharge,
    fault_mask,
    well_cells,
)
from .generate import (
    BenchmarkData,
    fault_signature,
    generate_benchmark,
    head_jump,
    load_benchmark,
)
from .modflow6 import ForwardSolution, build_simulation, find_mf6_executable, run_modflow
from .observations import ObservationSet, sample_observations

__all__ = [
    "BenchmarkData",
    "ForwardSolution",
    "ObservationSet",
    "build_conductivity",
    "build_et_max_rate",
    "build_recharge",
    "build_simulation",
    "fault_mask",
    "fault_signature",
    "find_mf6_executable",
    "generate_benchmark",
    "head_jump",
    "load_benchmark",
    "run_modflow",
    "sample_observations",
    "well_cells",
]
