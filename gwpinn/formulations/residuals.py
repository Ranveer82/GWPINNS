"""Three ways of writing the same groundwater equation, and why they differ.

All three discretise nothing in time - the head is a continuous function of
``t`` and ``dh/dt`` comes from automatic differentiation.  They differ in how
the *spatial* operator ``div(T A grad h)`` is expressed, and that choice is what
decides the mass-balance and fault behaviour.

``strong``
    The textbook PINN.  Write the residual directly, take two derivatives of the
    head network.  Simple and mesh-free, but it enforces the PDE only at the
    points sampled, so nothing prevents the network from leaking mass between
    them; and the second derivative of a smooth network is a poor way to
    represent a coefficient that jumps by three orders of magnitude across a
    fault.

``mixed``
    Introduce the Darcy flux as an independent output and split the equation
    into a constitutive law ``q = -T A grad h`` and a continuity equation
    ``S dh/dt + div q = f``.  Only first derivatives are needed, so the network
    never has to be twice differentiable across a barrier.  More importantly the
    *flux* is now the represented quantity, and flux - unlike head - really is
    continuous across a fault, so the representation matches the physics.  This
    is the PINN analogue of a mixed finite-element method.

``fv``
    Keep the network continuous but evaluate the balance over the benchmark's own
    control volumes: face conductances by harmonic mean, barriers in series on the
    faces they block, and a cell balance that telescopes.  Because every face flux
    enters exactly two cells with opposite sign, the residual being zero implies
    *local* conservation - the property strong-form collocation cannot give at
    any finite number of points.  It also lets the fault be applied exactly where
    MODFLOW applies it, on the face, instead of as a smeared anisotropy.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from gwpinn.formulations.problem import Problem


def _grad(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(u, v, torch.ones_like(u), create_graph=True)[0]


# --------------------------------------------------------------------------- #
# Strong form
# --------------------------------------------------------------------------- #


def strong_residual(
    prob: Problem, head_fn, kfun, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
) -> torch.Tensor:
    """``S dh/dt - div(T A grad h) - f`` in m/d, at scattered points."""
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    h = head_fn(x, y, t)
    K = kfun(x, y)
    b = prob.thickness(h, x, y)
    T = K * b
    S = prob.storage(h, x, y, b)

    hx, hy = _grad(h, x), _grad(h, y)
    A = prob.anisotropy(x, y, K)
    # q = -T A grad h  (sign folded in below)
    gx = A[:, 0, 0] * hx + A[:, 0, 1] * hy
    gy = A[:, 1, 0] * hx + A[:, 1, 1] * hy
    fx, fy = T * gx, T * gy
    div = _grad(fx, x) + _grad(fy, y)

    src = prob.sources(x, y, t, h)
    f = src["recharge"] + src["wel"] + src["riv"]
    return S * _grad(h, t) - div - f


# --------------------------------------------------------------------------- #
# Mixed (dual) form
# --------------------------------------------------------------------------- #


def mixed_residual(
    prob: Problem, head_fn, flux_fn, kfun,
    x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(constitutive, continuity)`` residuals, both in consistent units.

    The constitutive residual is divided by a characteristic transmissivity so
    that both terms are O(1) relative to each other; otherwise the flux equation,
    whose natural magnitude is ``T * grad h`` ~ 1e3 m2/d, drowns the continuity
    equation, whose magnitude is ~1e-1 m/d.
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    h = head_fn(x, y, t)
    q = flux_fn(x, y, t)                      # (N, 2), m2/d
    qx, qy = q[:, 0], q[:, 1]

    K = kfun(x, y)
    b = prob.thickness(h, x, y)
    T = K * b
    S = prob.storage(h, x, y, b)

    hx, hy = _grad(h, x), _grad(h, y)
    A = prob.anisotropy(x, y, K)
    gx = A[:, 0, 0] * hx + A[:, 0, 1] * hy
    gy = A[:, 1, 0] * hx + A[:, 1, 1] * hy

    t_ref = T.detach().mean().clamp_min(1e-6)
    r_const = torch.stack([(qx + T * gx) / t_ref, (qy + T * gy) / t_ref], dim=-1)

    src = prob.sources(x, y, t, h)
    f = src["recharge"] + src["wel"] + src["riv"]
    r_cont = S * _grad(h, t) + _grad(qx, x) + _grad(qy, y) - f
    return r_const, r_cont


# --------------------------------------------------------------------------- #
# Control-volume (finite-volume informed) form
# --------------------------------------------------------------------------- #


class ControlVolume:
    """Precomputed cell-centre geometry and face bookkeeping for the FV residual."""

    def __init__(self, prob: Problem) -> None:
        self.prob = prob
        self.nrow, self.ncol = prob.nrow, prob.ncol
        x, y = prob.grid_points()
        self.x = x
        self.y = y
        self.n_cells = x.numel()

    def _harmonic_face(self, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Harmonic-mean face transmissivities, matching MODFLOW's CV scheme."""
        Tg = T.view(self.nrow, self.ncol)
        tx = 2.0 * Tg[:, :-1] * Tg[:, 1:] / (Tg[:, :-1] + Tg[:, 1:]).clamp_min(1e-12)
        ty = 2.0 * Tg[:-1, :] * Tg[1:, :] / (Tg[:-1, :] + Tg[1:, :]).clamp_min(1e-12)
        return tx, ty

    def face_conductances(self, T: torch.Tensor, b: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Face conductance in m2/d, with any barrier added in series.

        MODFLOW's HFB puts a resistance ``1 / (hydchr * b * w)`` in series with
        the two half-cell resistances.  Reproducing that exactly - rather than
        approximating it with a smeared anisotropy - is the whole point of the
        FV formulation.
        """
        p = self.prob
        tx, ty = self._harmonic_face(T)
        cx = tx * p.dy / p.dx
        cy = ty * p.dx / p.dy

        bg = b.view(self.nrow, self.ncol)
        bx = 0.5 * (bg[:, :-1] + bg[:, 1:])
        by = 0.5 * (bg[:-1, :] + bg[1:, :])

        hcx, hcy = p.t_face_hydchr_x, p.t_face_hydchr_y
        finite_x, finite_y = torch.isfinite(hcx), torch.isfinite(hcy)
        if finite_x.any():
            cbar = torch.where(finite_x, hcx, torch.ones_like(hcx)) * bx * p.dy
            cx = torch.where(finite_x,
                             1.0 / (1.0 / cx.clamp_min(1e-30) + 1.0 / cbar.clamp_min(1e-30)),
                             cx)
        if finite_y.any():
            cbar = torch.where(finite_y, hcy, torch.ones_like(hcy)) * by * p.dx
            cy = torch.where(finite_y,
                             1.0 / (1.0 / cy.clamp_min(1e-30) + 1.0 / cbar.clamp_min(1e-30)),
                             cy)
        return cx, cy

    def cell_balance(self, h: torch.Tensor, T: torch.Tensor, b: torch.Tensor,
                     ) -> torch.Tensor:
        """Net lateral inflow to every cell, m3/d, ``(nrow, ncol)``.

        No-flow on the outer edges is implicit: there is simply no face there.
        """
        cx, cy = self.face_conductances(T, b)
        hg = h.view(self.nrow, self.ncol)
        fx = cx * (hg[:, 1:] - hg[:, :-1])      # flow from left cell to right, m3/d
        fy = cy * (hg[1:, :] - hg[:-1, :])

        net = torch.zeros_like(hg)
        net[:, :-1] = net[:, :-1] + fx
        net[:, 1:] = net[:, 1:] - fx
        net[:-1, :] = net[:-1, :] + fy
        net[1:, :] = net[1:, :] - fy
        return net


def fv_residual(
    prob: Problem, cv: ControlVolume, head_fn, kfun, t_slices: torch.Tensor,
) -> torch.Tensor:
    """Cell-balance residual, m/d, shape ``(n_times, nrow, ncol)``.

    Continuous in time, conservative in space.  ``dh/dt`` still comes from
    autograd, so the scheme has no time-stepping error to speak of, while the
    spatial operator telescopes exactly.
    """
    out = []
    for tv in t_slices:
        x = cv.x.clone().requires_grad_(True)
        y = cv.y.clone().requires_grad_(True)
        t = torch.full_like(cv.x, float(tv)).requires_grad_(True)

        h = head_fn(x, y, t)
        K = kfun(x, y)
        b = prob.thickness(h, x, y)
        T = K * b
        S = prob.storage(h, x, y, b)
        dhdt = _grad(h, t)

        net = cv.cell_balance(h, T, b)                     # m3/d
        src = prob.sources(x, y, t, h)
        f = (src["recharge"] + src["wel"] + src["riv"]).view(cv.nrow, cv.ncol)

        r = (S * dhdt).view(cv.nrow, cv.ncol) - net / prob.area - f
        out.append(r)
    return torch.stack(out)


__all__ = ["strong_residual", "mixed_residual", "fv_residual", "ControlVolume"]
