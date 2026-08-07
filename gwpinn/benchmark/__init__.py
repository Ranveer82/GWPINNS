"""Benchmark cases derived from the MODFLOW 6 heterogeneous test model.

``scripts/build_mf6_hetero_benchmark.py`` produces a five-layer transient
MODFLOW 6 model together with a complete set of georeferenced exports.  This
subpackage turns those exports into concrete, self-consistent learning tasks
for physics-informed surrogates, and supplies the reference solution every
formulation is scored against.
"""

from gwpinn.benchmark.mf6case import (
    ReducedCase,
    build_reduced_case,
    load_reduced_case,
)

__all__ = ["ReducedCase", "build_reduced_case", "load_reduced_case"]
