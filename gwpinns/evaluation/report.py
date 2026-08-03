"""Assembling a trained model's evaluation into a serialisable record."""

from __future__ import annotations

import numpy as np

from ..benchmark.generate import BenchmarkData
from ..pinn.base import BasePINN
from ..pinn.cpinn import ConservativePINN
from .metrics import evaluate

__all__ = ["inferred_fault_conductivity", "evaluate_model"]


def inferred_fault_conductivity(model: BasePINN) -> float | None:
    """The cPINN's own estimate of ``K_fault``; ``None`` for the other models.

    The decomposed model never places collocation points inside the fault zone,
    so reading K from those grid cells would score a network on a region it was
    never asked to represent.  Its leakance parameter is the honest estimate:
    ``K_f = C * width``.  The geometric mean is used because the quantity is
    log-distributed over the fault plane.
    """
    if not isinstance(model, ConservativePINN):
        return None
    if model.interface_mode != "conductance":
        return None
    sample = model.fault_conductance()
    return float(np.exp(np.mean(np.log(np.clip(sample["equivalent_k_m_per_day"], 1e-12, None)))))


def evaluate_model(model: BasePINN, data: BenchmarkData) -> dict:
    """Predict on the benchmark grid and score against the truth."""
    k_pred = model.predict_conductivity_grid()
    h_pred = model.predict_head_grid(data.times)
    report = evaluate(
        data.cfg,
        h_pred,
        data.heads,
        k_pred,
        data.k_true,
        obs=data.obs,
        inferred_fault_k=inferred_fault_conductivity(model),
    )
    return {"metrics": report, "k_pred": k_pred, "h_pred": h_pred}
