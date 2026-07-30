"""Differential operators and smooth constraint helpers."""

from __future__ import annotations

from typing import Optional

import torch


def grad(
    y: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = True,
    retain_graph: Optional[bool] = None,
) -> torch.Tensor:
    """Gradient of a scalar-per-sample output ``y`` w.r.t. ``x``.

    ``y`` must be shape ``(N,)`` or ``(N, 1)``; the result has the shape of ``x``.
    """
    if y.dim() > 1:
        y = y.reshape(y.shape[0], -1)
        if y.shape[1] != 1:
            raise ValueError("grad expects one output column; index it first")
        y = y[:, 0]

    # A field that is exactly constant or linear in x has an empty derivative
    # graph, so autograd has nothing to differentiate. That is a valid answer
    # (the derivative is zero), not an error - it shows up in the second
    # derivative of a linear head field and in degenerate test cases.
    if not y.requires_grad:
        return torch.zeros_like(x)

    (g,) = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=create_graph if retain_graph is None else retain_graph,
        allow_unused=True,
    )
    return torch.zeros_like(x) if g is None else g


def divergence(flux: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """``div(flux)`` for a vector field sampled at ``x``.

    ``flux`` is ``(N, d)`` and ``x`` is ``(N, d)``; each component is
    differentiated with respect to its own coordinate.
    """
    out = None
    for i in range(flux.shape[1]):
        gi = grad(flux[:, i], x, create_graph=create_graph)[:, i]
        out = gi if out is None else out + gi
    return out


def laplacian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return divergence(grad(y, x), x)


# --------------------------------------------------------------------------- #
# Smooth constraints
# --------------------------------------------------------------------------- #


def soft_lower(x: torch.Tensor, lo: torch.Tensor | float, beta: float = 1.0) -> torch.Tensor:
    """Smooth ``max(x, lo)``."""
    return lo + torch.nn.functional.softplus(x - lo, beta=beta)


def soft_upper(x: torch.Tensor, hi: torch.Tensor | float, beta: float = 1.0) -> torch.Tensor:
    """Smooth ``min(x, hi)``."""
    return hi - torch.nn.functional.softplus(hi - x, beta=beta)


def smooth_clamp(
    x: torch.Tensor,
    lo: torch.Tensor | float,
    hi: torch.Tensor | float,
    beta: float = 1.0,
) -> torch.Tensor:
    """Differentiable clamp that is guaranteed to respect its bounds.

    .. math::
        g(x) = lo + \\frac{1}{\\beta}\\Big(
            \\mathrm{softplus}(\\beta (x - lo)) -
            \\mathrm{softplus}(\\beta (x - hi))\\Big)

    ``g`` is smooth and strictly increasing, equals ``x`` well inside the
    interval, and tends to ``lo`` / ``hi`` outside it - so the result never
    leaves ``[lo, hi]``. Composing two one-sided softplus clamps instead
    (``min`` after ``max``) undershoots by up to ``ln 2 / beta`` whenever the
    bounds are close together, which for a thin aquifer layer means a negative
    saturated thickness and a negative transmissivity.

    A hard ``clamp`` is not an option: it zeroes the gradient wherever the bound
    is active, so the optimiser gets no signal from dry or over-full cells.
    ``beta`` sets the blend width in the units of ``x`` (metres here).
    """
    sp = torch.nn.functional.softplus
    return lo + (sp(beta * (x - lo)) - sp(beta * (x - hi))) / beta


__all__ = ["grad", "divergence", "laplacian", "smooth_clamp", "soft_lower", "soft_upper"]
