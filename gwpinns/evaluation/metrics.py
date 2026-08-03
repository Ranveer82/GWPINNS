"""Scoring of a trained inverse model against the benchmark truth.

Three families of metric, in increasing order of what we actually care about:

1. **Head reproduction** -- can the model reproduce the state it was fitted to,
   and does it generalise to cells with no observation well?
2. **Conductivity recovery** -- error in ``log10 K``, reported separately for
   the background aquifer and the fault zone, because a model can score well on
   the bulk while getting the fault completely wrong.
3. **Fault characterisation** -- the question the study exists to answer: does
   the model identify the fault as a barrier or a conduit, and how close does
   it get to the true fault conductivity?
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..config import BenchmarkConfig
from ..benchmark.fields import fault_mask

__all__ = [
    "FaultVerdict",
    "head_metrics",
    "conductivity_metrics",
    "fault_metrics",
    "evaluate",
    "classify_fault",
]


@dataclass
class FaultVerdict:
    """Classification of the inferred fault behaviour."""

    label: str                 # "barrier" | "conduit" | "neutral"
    contrast_log10: float      # log10(K_fault / K_background)
    k_fault: float
    k_background: float

    def as_dict(self) -> dict:
        return asdict(self)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    true = np.asarray(true)
    residual = np.sum((np.asarray(pred) - true) ** 2)
    total = np.sum((true - true.mean()) ** 2)
    return float(1.0 - residual / total) if total > 0 else float("nan")


def head_metrics(
    predicted: np.ndarray, true: np.ndarray, obs_cells: np.ndarray | None = None
) -> dict[str, float]:
    """Head errors over the whole grid, in metres.

    ``obs_cells`` is an optional boolean mask of cells containing a monitoring
    well; when given, errors are also reported for the *unobserved* cells,
    which is the honest measure of generalisation.
    """
    metrics = {
        "head_rmse_m": _rmse(predicted, true),
        "head_mae_m": float(np.mean(np.abs(predicted - true))),
        "head_max_abs_error_m": float(np.max(np.abs(predicted - true))),
        "head_r2": _r2(predicted.ravel(), true.ravel()),
    }
    if obs_cells is not None:
        unobserved = ~np.broadcast_to(obs_cells, true.shape[-3:])
        mask = np.broadcast_to(unobserved, true.shape)
        metrics["head_rmse_unobserved_m"] = _rmse(predicted[mask], true[mask])
    return metrics


def conductivity_metrics(
    cfg: BenchmarkConfig, predicted: np.ndarray, true: np.ndarray
) -> dict[str, float]:
    """Errors in ``log10 K``, split into background and fault zone."""
    mask = fault_mask(cfg)
    log_pred = np.log10(np.clip(predicted, 1e-12, None))
    log_true = np.log10(np.clip(true, 1e-12, None))

    return {
        "logk_rmse": _rmse(log_pred, log_true),
        "logk_rmse_background": _rmse(log_pred[~mask], log_true[~mask]),
        "logk_rmse_fault": _rmse(log_pred[mask], log_true[mask]),
        "logk_bias_background": float(np.mean(log_pred[~mask] - log_true[~mask])),
        "logk_r2_background": _r2(log_pred[~mask], log_true[~mask]),
        "k_background_gmean_pred": float(10.0 ** np.mean(log_pred[~mask])),
        "k_background_gmean_true": float(10.0 ** np.mean(log_true[~mask])),
        "k_fault_gmean_pred": float(10.0 ** np.mean(log_pred[mask])),
        "k_fault_gmean_true": float(10.0 ** np.mean(log_true[mask])),
    }


def classify_fault(
    k_fault: float, k_background: float, threshold: float = 0.5
) -> FaultVerdict:
    """Label a fault from its conductivity contrast with the host rock.

    ``threshold`` is in log10 units: half an order of magnitude either way is
    treated as hydraulically neutral.
    """
    contrast = float(np.log10(max(k_fault, 1e-12) / max(k_background, 1e-12)))
    if contrast <= -threshold:
        label = "barrier"
    elif contrast >= threshold:
        label = "conduit"
    else:
        label = "neutral"
    return FaultVerdict(
        label=label,
        contrast_log10=contrast,
        k_fault=float(k_fault),
        k_background=float(k_background),
    )


def fault_metrics(
    cfg: BenchmarkConfig,
    predicted_k: np.ndarray,
    true_k: np.ndarray,
    inferred_fault_k: float | None = None,
    predicted_heads: np.ndarray | None = None,
    true_heads: np.ndarray | None = None,
) -> dict[str, object]:
    """Did the model get the fault right?

    ``inferred_fault_k`` lets an architecture supply its own estimate -- the
    cPINN reports ``K_f = C * width`` from its leakance parameter rather than
    from grid cells it never models.
    """
    from ..benchmark.generate import head_jump

    mask = fault_mask(cfg)
    log_pred = np.log10(np.clip(predicted_k, 1e-12, None))
    log_true = np.log10(np.clip(true_k, 1e-12, None))

    k_background_pred = float(10.0 ** np.mean(log_pred[~mask]))
    k_background_true = float(10.0 ** np.mean(log_true[~mask]))
    k_fault_pred = (
        float(inferred_fault_k)
        if inferred_fault_k is not None
        else float(10.0 ** np.mean(log_pred[mask]))
    )
    k_fault_true = float(cfg.fault_k)

    predicted_verdict = classify_fault(k_fault_pred, k_background_pred)
    true_verdict = classify_fault(k_fault_true, k_background_true)

    report: dict[str, object] = {
        "true_scenario": cfg.scenario,
        "predicted_label": predicted_verdict.label,
        "correct_classification": predicted_verdict.label == cfg.scenario,
        "contrast_log10_pred": predicted_verdict.contrast_log10,
        "contrast_log10_true": true_verdict.contrast_log10,
        "contrast_error_log10": abs(
            predicted_verdict.contrast_log10 - true_verdict.contrast_log10
        ),
        "k_fault_pred_m_per_d": k_fault_pred,
        "k_fault_true_m_per_d": k_fault_true,
        "k_fault_error_log10": abs(
            np.log10(max(k_fault_pred, 1e-12)) - np.log10(max(k_fault_true, 1e-12))
        ),
        "k_background_pred_m_per_d": k_background_pred,
        "k_background_true_m_per_d": k_background_true,
    }

    # Does the *head field* carry the fault's hydraulic signature?  This is
    # independent of the recovered K, and it is the quantity a hydrogeologist
    # would actually check: a barrier sustains a head jump across the zone, a
    # conduit erases it.
    if predicted_heads is not None and true_heads is not None:
        report["head_jump_pred_m"] = head_jump(cfg, predicted_heads[-1])
        report["head_jump_true_m"] = head_jump(cfg, true_heads[-1])

    return report


def observation_cell_mask(cfg: BenchmarkConfig, obs) -> np.ndarray:
    """Boolean ``(nlay, nrow, ncol)`` mask of cells holding a monitoring well."""
    grid = cfg.grid
    mask = np.zeros((grid.nlay, grid.nrow, grid.ncol), dtype=bool)
    for x, y, layer in zip(obs.x, obs.y, obs.layer):
        row, col = grid.locate(float(x), float(y))
        mask[int(layer), row, col] = True
    return mask


def evaluate(
    cfg: BenchmarkConfig,
    predicted_heads: np.ndarray,
    true_heads: np.ndarray,
    predicted_k: np.ndarray,
    true_k: np.ndarray,
    obs=None,
    inferred_fault_k: float | None = None,
) -> dict[str, object]:
    """Full metric report for one trained model."""
    obs_cells = observation_cell_mask(cfg, obs) if obs is not None else None
    report: dict[str, object] = {}
    report.update(head_metrics(predicted_heads, true_heads, obs_cells))
    report.update(conductivity_metrics(cfg, predicted_k, true_k))
    report.update(
        fault_metrics(
            cfg, predicted_k, true_k, inferred_fault_k,
            predicted_heads=predicted_heads, true_heads=true_heads,
        )
    )
    return report
