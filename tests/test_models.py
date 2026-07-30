"""Every backbone must support the second derivatives the PDE residual needs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gwpinn.models.backbones import (
    bspline_sample,
    build_backbone,
    cubic_bspline_weights,
)
from gwpinn.models.fields import HeadField, Normalizer, PropertyField
from gwpinn.models.fourier import RandomFourierFeatures

ARCHS = ["mlp", "resnet", "modified_mlp", "cnn"]


@pytest.mark.parametrize("arch", ARCHS)
def test_backbone_has_nonzero_second_derivative(arch):
    """A field with a vanishing Laplacian everywhere could not solve the PDE."""
    torch.manual_seed(0)
    net = build_backbone(arch, in_dim=2, out_dim=1, width=32, depth=3,
                         cnn_latent=8, cnn_channels=8).double()
    coords = (torch.rand(64, 2, dtype=torch.float64) * 1.6 - 0.8).requires_grad_(True)
    y = net(coords, coords)

    g = torch.autograd.grad(y.sum(), coords, create_graph=True)[0]
    lap = torch.autograd.grad(g[:, 0].sum(), coords, create_graph=True)[0][:, 0]
    lap = lap + torch.autograd.grad(g[:, 1].sum(), coords, create_graph=True)[0][:, 1]

    assert torch.isfinite(g).all() and torch.isfinite(lap).all()
    assert g.abs().mean().item() > 0
    assert lap.abs().mean().item() > 0


def test_cubic_bspline_weights_partition_unity():
    t = torch.linspace(0, 1, 17, dtype=torch.float64)
    w = cubic_bspline_weights(t)
    assert torch.allclose(w.sum(dim=-1), torch.ones_like(t))
    assert (w >= -1e-12).all()


def test_bspline_reproduces_a_constant_grid():
    grid = torch.full((1, 12, 12), 3.5, dtype=torch.float64)
    coords = (torch.rand(40, 2, dtype=torch.float64) * 1.2 - 0.6)
    out = bspline_sample(grid, coords)
    assert torch.allclose(out, torch.full_like(out, 3.5), atol=1e-9)


def test_bspline_is_twice_differentiable():
    """Bilinear sampling would give an identically zero second derivative."""
    torch.manual_seed(1)
    grid = torch.randn(1, 16, 16, dtype=torch.float64)
    coords = (torch.rand(32, 2, dtype=torch.float64) * 1.2 - 0.6).requires_grad_(True)
    y = bspline_sample(grid, coords)
    g = torch.autograd.grad(y.sum(), coords, create_graph=True)[0]
    h = torch.autograd.grad(g[:, 0].sum(), coords, create_graph=True)[0]
    assert torch.isfinite(h).all()
    assert h.abs().mean().item() > 0


def test_fourier_features_shape_and_range():
    emb = RandomFourierFeatures(2, n_features=30, sigmas=(1.0, 3.0, 6.0),
                                dtype=torch.float64)
    x = torch.rand(20, 2, dtype=torch.float64)
    out = emb(x)
    assert out.shape == (20, emb.out_dim)
    assert out.abs().max() <= 1.0 + 1e-12


def test_property_field_respects_bounds():
    norm = Normalizer.from_bounds((0.0, 0.0, 1000.0, 1000.0))
    pf = PropertyField(2, norm, arch="mlp", width=24, depth=2, n_fourier=12,
                       k_bounds=(1e-2, 50.0), s_bounds=(1e-5, 0.3),
                       dtype=torch.float64)
    xy = torch.rand(200, 2, dtype=torch.float64) * 1000.0
    out = pf(xy)
    assert (out["K"] >= 1e-2 - 1e-9).all() and (out["K"] <= 50.0 + 1e-6).all()
    assert (out["S"] >= 1e-5 - 1e-12).all() and (out["S"] <= 0.3 + 1e-9).all()
    assert out["K"].shape == (200, 2)


def test_property_prior_centres_the_field():
    norm = Normalizer.from_bounds((0.0, 0.0, 1000.0, 1000.0))
    pf = PropertyField(1, norm, arch="mlp", width=16, depth=2, n_fourier=8,
                       k_bounds=(1e-3, 1e3), dtype=torch.float64)
    pf.set_prior(log10k=[1.0])
    # Zero out the backbone so only the offset remains.
    with torch.no_grad():
        for p in pf.backbone.parameters():
            p.zero_()
    out = pf(torch.rand(5, 2, dtype=torch.float64) * 1000.0)
    assert out["log10K"].mean().item() == pytest.approx(1.0, abs=1e-6)


def test_head_field_normalisation_roundtrip():
    heads = np.array([40.0, 45.0, 50.0, 55.0])
    norm = Normalizer.from_bounds((0.0, 0.0, 2000.0, 1000.0), heads)
    assert norm.h_mean == pytest.approx(47.5)
    xy = torch.tensor([[0.0, 0.0], [2000.0, 1000.0]], dtype=torch.float64)
    n = norm.norm_xy(xy)
    assert n.abs().max().item() <= 1.0 + 1e-12
    assert torch.allclose(norm.denorm_xy(n), xy, atol=1e-9)


def test_head_field_output_shape_and_transient():
    norm = Normalizer.from_bounds((0.0, 0.0, 1000.0, 1000.0), np.array([10.0, 20.0]),
                                  t_max=100.0)
    hf = HeadField(3, norm, arch="mlp", width=16, depth=2, n_fourier=8,
                   transient=True, dtype=torch.float64)
    xy = torch.rand(7, 2, dtype=torch.float64) * 1000.0
    t = torch.rand(7, 1, dtype=torch.float64) * 100.0
    assert hf(xy, t).shape == (7, 3)


def test_fault_features_are_wired_into_the_networks():
    norm = Normalizer.from_bounds((0.0, 0.0, 1000.0, 1000.0))
    hf = HeadField(1, norm, arch="mlp", width=16, depth=2, n_fourier=8,
                   n_fault_feats=2, dtype=torch.float64)
    xy = torch.rand(5, 2, dtype=torch.float64) * 1000.0
    a = hf(xy, None, torch.zeros(5, 2, dtype=torch.float64))
    b = hf(xy, None, torch.ones(5, 2, dtype=torch.float64))
    assert not torch.allclose(a, b)     # the side indicator must actually matter
