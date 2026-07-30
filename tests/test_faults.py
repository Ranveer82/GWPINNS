"""Fault anisotropy field."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gwpinn.geo.faults import FaultField


def _vertical_fault(perm=1.0, width=50.0, perm_min=1e-3):
    line = np.column_stack([np.full(9, 500.0), np.linspace(0, 1000, 9)])
    return FaultField([line], np.array([perm]), width=width, perm_min=perm_min,
                      dtype=torch.float64)


def test_identity_far_from_fault():
    ff = _vertical_fault()
    xy = torch.tensor([[0.0, 500.0], [1000.0, 500.0]], dtype=torch.float64)
    A = ff.anisotropy(xy)
    assert torch.allclose(A, torch.eye(2, dtype=torch.float64), atol=1e-6)


def test_blocks_normal_flow_but_not_along_strike():
    """On a north-south fault, x-conductivity drops and y-conductivity does not."""
    ff = _vertical_fault(perm=0.0)          # flagged impermeable -> alpha = perm_min
    xy = torch.tensor([[500.0, 500.0]], dtype=torch.float64)
    A = ff.anisotropy(xy)[0]
    assert A[0, 0].item() == pytest.approx(1e-3, abs=5e-3)   # across the fault
    assert A[1, 1].item() == pytest.approx(1.0, abs=1e-6)    # along the fault
    assert abs(A[0, 1].item()) < 1e-9


def test_eigenvalues_stay_in_bounds_where_faults_cross():
    """Two crossing faults must not drive the tensor singular or negative."""
    a = np.column_stack([np.full(5, 500.0), np.linspace(0, 1000, 5)])
    b = np.column_stack([np.linspace(0, 1000, 5), np.full(5, 500.0)])
    ff = FaultField([a, b], np.array([0.0, 0.0]), width=60.0, perm_min=1e-3,
                    dtype=torch.float64)
    xy = torch.tensor([[500.0, 500.0], [505.0, 495.0]], dtype=torch.float64)
    A = ff.anisotropy(xy)
    eig = torch.linalg.eigvalsh(0.5 * (A + A.transpose(1, 2)))
    assert (eig > 0).all()
    assert eig.min().item() >= 1e-3 - 1e-6
    assert eig.max().item() <= 1.0 + 1e-6


def test_anisotropy_is_differentiable_in_position():
    ff = _vertical_fault(perm=0.0)
    xy = torch.tensor([[480.0, 500.0]], dtype=torch.float64, requires_grad=True)
    A = ff.anisotropy(xy)
    g = torch.autograd.grad(A.sum(), xy)[0]
    assert torch.isfinite(g).all()
    assert g.abs().sum().item() > 0     # the barrier really does vary in space


def test_side_features_change_sign_across_the_trace():
    ff = _vertical_fault(perm=1.0, width=40.0)
    xy = torch.tensor([[300.0, 500.0], [700.0, 500.0]], dtype=torch.float64)
    s = ff.side_features(xy)
    assert s.shape == (2, 1)
    assert s[0, 0].item() * s[1, 0].item() < 0
    assert abs(s[0, 0].item()) == pytest.approx(1.0, abs=1e-3)


def test_permeability_parameterisation():
    # perm == 1 -> trainable; perm == 0 -> pinned at perm_min; else fixed.
    ff = FaultField(
        [np.array([[0.0, 0.0], [1.0, 0.0]])] * 3,
        np.array([1.0, 0.0, 0.25]),
        width=10.0, perm_min=1e-3, dtype=torch.float64,
    )
    alpha = ff.alpha()
    assert ff.raw.requires_grad
    assert ff.trainable.tolist() == [True, False, False]
    assert alpha[1].item() == pytest.approx(1e-3)
    assert alpha[2].item() == pytest.approx(0.25)
    assert 1e-3 <= alpha[0].item() <= 1.0


def test_evaluate_matches_separate_calls():
    ff = _vertical_fault(perm=0.0)
    xy = torch.tensor([[470.0, 400.0], [530.0, 600.0]], dtype=torch.float64)
    A1, s1 = ff.evaluate(xy)
    assert torch.allclose(A1, ff.anisotropy(xy))
    assert torch.allclose(s1, ff.side_features(xy))


def test_empty_fault_set_is_a_no_op():
    ff = FaultField([], None, width=10.0, dtype=torch.float64)
    xy = torch.rand(5, 2, dtype=torch.float64)
    assert not ff.has_faults
    assert torch.allclose(ff.anisotropy(xy), torch.eye(2, dtype=torch.float64))
    assert ff.side_features(xy).shape == (5, 0)
