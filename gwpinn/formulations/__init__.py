"""Composable physics-informed formulations for the groundwater benchmark."""

from gwpinn.formulations.problem import Problem
from gwpinn.formulations.runner import Runner, Spec, Surrogate

__all__ = ["Problem", "Runner", "Spec", "Surrogate"]
