"""Scoring a surrogate against the MODFLOW reference on the five criteria."""

from gwpinn.eval.criteria import (
    CRITERIA, evaluate, null_scores, reference_scores, skill,
)

__all__ = ["evaluate", "CRITERIA", "null_scores", "reference_scores", "skill"]
