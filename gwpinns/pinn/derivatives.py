"""Autograd helpers for the physics residuals.

Deliberately thin: the PDE terms are assembled by hand in the model classes so
that every spatial gradient and every interface condition stays explicit and
inspectable, rather than being hidden inside a framework.
"""

from __future__ import annotations

import torch

__all__ = ["grad", "partial", "divergence"]


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Gradient of a scalar-valued column ``outputs`` w.r.t. ``inputs``.

    ``inputs`` must be a leaf-like tensor with ``requires_grad=True`` of shape
    ``(N, d)``; the result has the same shape.  ``create_graph=True`` keeps the
    result differentiable, which is what allows the second-order term
    ``d_i(K d_i H)`` to be built from two nested calls.
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def partial(outputs: torch.Tensor, inputs: torch.Tensor, index: int) -> torch.Tensor:
    """Single partial derivative ``d(outputs)/d(inputs[:, index])``, shape ``(N, 1)``."""
    return grad(outputs, inputs)[:, index : index + 1]


def divergence(
    components: tuple[torch.Tensor, ...],
    inputs: torch.Tensor,
    weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Weighted divergence ``sum_i w_i * d(components[i])/d(inputs[:, i])``.

    The weights carry the per-axis scaling factors ``mu_i`` from
    :class:`~gwpinns.pinn.scaling.Scaling`.
    """
    if weights is None:
        weights = (1.0,) * len(components)
    total = None
    for axis, (component, weight) in enumerate(zip(components, weights)):
        term = weight * partial(component, inputs, axis)
        total = term if total is None else total + term
    return total
