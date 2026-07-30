"""The reference finite-difference solver, checked against closed-form answers.

The synthetic experiment scores the PINN against this solver, so the solver
itself has to be right. Both are compared with the same analytic solutions used
in ``test_physics.py``, which is what makes the PINN-vs-reference comparison
meaningful: the two implementations agree with theory independently.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwpinn.data.fdsolver import fault_face_multipliers, solve_steady, solve_transient
from gwpinn.train.losses import weighted_mse


def _grid(nx=60, ny=5, cs=20.0):
    xs = (np.arange(nx) + 0.5) * cs
    ys = (np.arange(ny)[::-1] + 0.5) * cs
    return xs, ys, cs


def test_confined_flow_between_fixed_heads_is_linear():
    """Constant T, no sources: the head profile must be a straight line."""
    nx, ny, cs = 60, 5, 20.0
    active = np.ones((ny, nx), dtype=bool)
    K = [np.full((ny, nx), 10.0)]
    top = np.full((ny, nx), 100.0)
    bottoms = [np.full((ny, nx), 90.0)]

    fixed = np.zeros((ny, nx), dtype=bool)
    fixed[:, 0] = True
    fixed[:, -1] = True
    head = np.zeros((ny, nx))
    head[:, 0] = 50.0
    head[:, -1] = 30.0

    h = solve_steady(
        K, top, bottoms, cs, active, unconfined_top=False,
        fixed_mask=fixed, fixed_head=head, n_picard=2, min_thickness=1.0,
    )
    mid = h[0, ny // 2, :]
    expect = np.linspace(50.0, 30.0, nx)
    # Cell centres are half a cell inside the fixed columns; compare the interior.
    assert np.allclose(mid[1:-1], expect[1:-1], atol=0.35)


def test_recharge_gives_the_parabolic_profile():
    """T h'' + R = 0 between two fixed heads -> a parabola."""
    nx, ny, cs = 80, 5, 25.0
    active = np.ones((ny, nx), dtype=bool)
    K, b = 20.0, 10.0
    T = K * b
    R = 2.0e-4

    fixed = np.zeros((ny, nx), dtype=bool)
    fixed[:, 0] = fixed[:, -1] = True
    head = np.zeros((ny, nx))
    head[:, 0] = head[:, -1] = 40.0

    h = solve_steady(
        [np.full((ny, nx), K)], np.full((ny, nx), 100.0),
        [np.full((ny, nx), 90.0)], cs, active,
        recharge=np.full((ny, nx), R), unconfined_top=False,
        fixed_mask=fixed, fixed_head=head, n_picard=2, min_thickness=1.0,
    )
    x = (np.arange(nx) + 0.5) * cs
    x0, x1 = x[0], x[-1]
    L = x1 - x0
    analytic = 40.0 + R / (2 * T) * (x - x0) * (x1 - x)
    got = h[0, ny // 2, :]
    assert np.allclose(got[2:-2], analytic[2:-2], rtol=0.02, atol=0.05)


def test_impermeable_fault_creates_a_head_jump():
    nx, ny, cs = 60, 9, 20.0
    active = np.ones((ny, nx), dtype=bool)
    xs, ys, _ = _grid(nx, ny, cs)

    fixed = np.zeros((ny, nx), dtype=bool)
    fixed[:, 0] = fixed[:, -1] = True
    head = np.zeros((ny, nx))
    head[:, 0], head[:, -1] = 50.0, 30.0

    line = np.column_stack([np.full(5, xs[nx // 2] + 0.5 * cs),
                            np.linspace(-100, ys[0] + 100, 5)])
    mx, my = fault_face_multipliers([line], [1e-4], xs, ys)
    assert (mx < 1).any()

    args = dict(
        top=np.full((ny, nx), 100.0), bottoms=[np.full((ny, nx), 90.0)],
        cellsize=cs, active=active, unconfined_top=False, fixed_mask=fixed,
        fixed_head=head, n_picard=2, min_thickness=1.0,
    )
    K = [np.full((ny, nx), 10.0)]
    h_open = solve_steady(K, **args)
    h_fault = solve_steady(K, fault_mx=mx, fault_my=my, **args)

    row = ny // 2
    jump_open = abs(h_open[0, row, nx // 2] - h_open[0, row, nx // 2 + 1])
    jump_fault = abs(h_fault[0, row, nx // 2] - h_fault[0, row, nx // 2 + 1])
    assert jump_fault > 10 * jump_open
    # A barrier steepens the gradient locally but cannot change the end points.
    assert h_fault[0, row, 1] == pytest.approx(h_open[0, row, 1], abs=2.0)


def test_leakage_pulls_layers_together():
    nx, ny, cs = 30, 6, 50.0
    active = np.ones((ny, nx), dtype=bool)
    top = np.full((ny, nx), 100.0)
    bottoms = [np.full((ny, nx), 80.0), np.full((ny, nx), 50.0)]
    K = [np.full((ny, nx), 10.0), np.full((ny, nx), 10.0)]

    river_cond = np.zeros((ny, nx))
    river_cond[:, 0] = 1.0
    stage = np.zeros((ny, nx))
    stage[:, 0] = 60.0

    h = solve_steady(
        K, top, bottoms, cs, active, leakance=[1e-2],
        recharge=np.full((ny, nx), 1e-4),
        river_cond=river_cond, river_stage=stage,
        unconfined_top=False, n_picard=3, min_thickness=1.0,
    )
    # Strong leakance => the two layers should track each other closely.
    assert np.nanmax(np.abs(h[0] - h[1])) < 1.0
    assert np.isfinite(h[:, :, 1:]).all()


def test_transient_relaxes_towards_the_steady_state():
    nx, ny, cs = 24, 6, 50.0
    active = np.ones((ny, nx), dtype=bool)
    top = np.full((ny, nx), 100.0)
    bottoms = [np.full((ny, nx), 70.0)]
    K = [np.full((ny, nx), 15.0)]
    S = [np.full((ny, nx), 0.1)]

    river_cond = np.zeros((ny, nx))
    river_cond[:, 0] = 0.5
    stage = np.zeros((ny, nx))
    stage[:, 0] = 80.0

    steady = solve_steady(
        K, top, bottoms, cs, active, river_cond=river_cond, river_stage=stage,
        unconfined_top=False, n_picard=3, min_thickness=1.0,
    )
    times = list(np.linspace(0.0, 4000.0, 25))
    h0 = np.full((1, ny, nx), 90.0)
    series = solve_transient(
        K, S, top, bottoms, cs, active, times, h0,
        river_cond=river_cond,
        river_stage_series=[stage] * len(times),
        unconfined_top=False, min_thickness=1.0,
    )
    err = [np.nanmax(np.abs(series[i] - steady)) for i in (1, 12, 24)]
    assert err[0] > err[1] > err[2]
    assert err[2] < 0.5 * err[0]


def test_noise_floored_loss_ignores_residuals_within_tolerance():
    import torch

    pred = torch.tensor([1.0, 1.02, 0.98], dtype=torch.float64)
    target = torch.ones(3, dtype=torch.float64)
    assert float(weighted_mse(pred, target, noise=0.05)) == pytest.approx(0.0)
    assert float(weighted_mse(pred, target, noise=0.0)) > 0
    # Beyond the tolerance only the excess is charged.
    pred2 = torch.tensor([1.15], dtype=torch.float64)
    got = float(weighted_mse(pred2, torch.ones(1, dtype=torch.float64), noise=0.05))
    assert got == pytest.approx(0.10**2)
