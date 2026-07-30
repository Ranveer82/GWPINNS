"""Synthetic test case: ground-truth solver and sample-data generation."""

from gwpinn.data.grf import gaussian_random_field  # noqa: F401
from gwpinn.data.fdsolver import fault_face_multipliers, solve_steady, solve_transient  # noqa: F401

__all__ = [
    "gaussian_random_field",
    "solve_steady",
    "solve_transient",
    "fault_face_multipliers",
]
