"""The flow residual must vanish on an exact solution of the flow equation.

This is the strongest single check in the suite: it exercises the sign
conventions, the divergence operator, the transmissivity assembly and the source
terms together, against a closed-form answer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gwpinn.physics.gwflow import GroundwaterFlow, LayerElevations, Sources
from gwpinn.physics.operators import divergence, grad, smooth_clamp


def _confined_layer(n: int, thickness: float = 1.0, dtype=torch.float64):
    top = torch.full((n,), 10.0, dtype=dtype)
    bottoms = torch.full((n, 1), 10.0 - thickness, dtype=dtype)
    return LayerElevations(top=top, bottoms=bottoms)


def test_residual_zero_for_analytic_1d_solution():
    """T h'' + R = 0  =>  h(x) = -R x^2 / (2T) + a x + b."""
    torch.manual_seed(0)
    K, thickness, R = 12.0, 1.0, 3.0e-4
    T = K * thickness
    a, b = 0.02, 50.0

    xy = torch.rand(256, 2, dtype=torch.float64) * 1000.0
    xy.requires_grad_(True)
    x = xy[:, 0]
    h = (-R * x**2 / (2 * T) + a * x + b).reshape(-1, 1)

    flow = GroundwaterFlow(
        n_layers=1, unconfined_top=False, residual_scale=1.0, dtype=torch.float64
    )
    res = flow.residual(
        xy,
        h,
        torch.full((256, 1), K, dtype=torch.float64),
        _confined_layer(256, thickness),
        sources=Sources(recharge=torch.full((256,), R, dtype=torch.float64)),
    )
    assert torch.isfinite(res).all()
    assert res.abs().max().item() < 1e-9


def test_residual_zero_for_unconfined_boussinesq_solution():
    """Unconfined, no recharge: h^2 is linear in x (the Dupuit solution)."""
    K = 8.0
    h0, h1, L = 30.0, 20.0, 500.0

    x = torch.linspace(20.0, L - 20.0, 200, dtype=torch.float64)
    xy = torch.stack([x, torch.zeros_like(x)], dim=1).requires_grad_(True)
    # h(x)^2 = h0^2 + (h1^2 - h0^2) x / L, with the aquifer base at z = 0.
    h = torch.sqrt(h0**2 + (h1**2 - h0**2) * xy[:, 0] / L).reshape(-1, 1)

    elev = LayerElevations(
        top=torch.full((200,), 60.0, dtype=torch.float64),
        bottoms=torch.zeros((200, 1), dtype=torch.float64),
    )
    flow = GroundwaterFlow(
        n_layers=1, unconfined_top=True, min_thickness=0.1,
        residual_scale=1.0, dtype=torch.float64,
    )
    res = flow.residual(
        xy, h, torch.full((200, 1), K, dtype=torch.float64), elev, sources=None
    )
    # Softplus smoothing of the saturated thickness leaves a small residue.
    assert res.abs().max().item() < 1e-6


def test_leakage_between_layers_is_antisymmetric():
    """What leaks out of one layer must arrive in the other."""
    n = 64
    xy = (torch.rand(n, 2, dtype=torch.float64) * 100).requires_grad_(True)
    # Constant in space, but routed through xy so the graph exists.
    zero = 0.0 * xy[:, 0]
    h = torch.stack([30.0 + zero, 25.0 + zero], dim=1)
    elev = LayerElevations(
        top=torch.full((n,), 60.0, dtype=torch.float64),
        bottoms=torch.stack(
            [torch.full((n,), 20.0, dtype=torch.float64),
             torch.full((n,), 0.0, dtype=torch.float64)], dim=1
        ),
    )
    flow = GroundwaterFlow(
        n_layers=2, unconfined_top=False, leakance=[2.5e-3],
        residual_scale=1.0, dtype=torch.float64,
    )
    res = flow.residual(
        xy, h, torch.full((n, 2), 5.0, dtype=torch.float64), elev, sources=None
    )
    # Uniform heads => no lateral flux, so only the leakage term survives.
    assert torch.allclose(res[:, 0], -res[:, 1], atol=1e-12)
    assert res[:, 0].mean().item() == pytest.approx(2.5e-3 * (25.0 - 30.0), rel=1e-9)


def test_river_term_drives_head_towards_stage():
    n = 32
    xy = (torch.rand(n, 2, dtype=torch.float64) * 100).requires_grad_(True)
    elev = LayerElevations(
        top=torch.full((n,), 60.0, dtype=torch.float64),
        bottoms=torch.zeros((n, 1), dtype=torch.float64),
    )
    flow = GroundwaterFlow(
        n_layers=1, unconfined_top=False, river_conductance=0.05,
        residual_scale=1.0, dtype=torch.float64,
    )
    src = Sources(
        river_stage=torch.full((n,), 40.0, dtype=torch.float64),
        river_mask=torch.ones(n, dtype=torch.float64),
    )
    for head, sign in ((30.0, +1.0), (50.0, -1.0)):
        h = (head + 0.0 * xy[:, 0]).reshape(-1, 1)
        res = flow.residual(
            xy, h, torch.full((n, 1), 5.0, dtype=torch.float64), elev, sources=src
        )
        # Head below stage => the river is a source (positive residual term).
        assert torch.sign(res).unique().tolist() == [sign]


def test_smooth_clamp_keeps_gradient_at_the_bound():
    x = torch.linspace(-5, 5, 41, dtype=torch.float64, requires_grad=True)
    y = smooth_clamp(x, 0.0, 2.0, beta=1.0)
    assert (y >= -1e-9).all() and (y <= 2.0 + 1e-9).all()
    g = torch.autograd.grad(y.sum(), x)[0]
    # A hard clamp would give exactly zero gradient outside the range.
    assert (g > 0).all()


def test_divergence_matches_analytic():
    xy = (torch.rand(128, 2, dtype=torch.float64) * 4 - 2).requires_grad_(True)
    x, y = xy[:, 0], xy[:, 1]
    flux = torch.stack([x**2 * y, x * y**3], dim=1)   # div = 2xy + 3xy^2
    got = divergence(flux, xy)
    want = 2 * x * y + 3 * x * y**2
    assert torch.allclose(got, want, atol=1e-10)


def test_grad_matches_analytic():
    xy = (torch.rand(64, 2, dtype=torch.float64)).requires_grad_(True)
    f = (xy[:, 0] ** 3 + torch.sin(xy[:, 1]))
    g = grad(f, xy)
    assert torch.allclose(g[:, 0], 3 * xy[:, 0] ** 2, atol=1e-10)
    assert torch.allclose(g[:, 1], torch.cos(xy[:, 1]), atol=1e-10)
