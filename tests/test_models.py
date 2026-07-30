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


# --------------------------------------------------------------------------- #
# Explicit fault representation by coordinate mapping
# --------------------------------------------------------------------------- #


def _fault_setup(fault_coords: bool, width: float = 50.0,
                 fault_sigma_scale: float = 0.15):
    """A single north-south fault at x = 500 inside a 1 km square."""
    from gwpinn.geo.faults import FaultField

    line = np.column_stack([np.full(9, 500.0), np.linspace(0, 1000, 9)])
    ff = FaultField([line], np.array([0.0]), width=width, perm_min=1e-3,
                    dtype=torch.float64)
    norm = Normalizer.from_bounds((0.0, 0.0, 1000.0, 1000.0),
                                  np.array([10.0, 20.0]))
    net = HeadField(1, norm, arch="mlp", width=32, depth=2, n_fourier=24,
                    fourier_sigma=3.0, n_fault_feats=1,
                    fault_coords=fault_coords,
                    fault_sigma_scale=fault_sigma_scale, dtype=torch.float64)
    return ff, net


def test_fault_coords_keeps_the_input_width_unchanged():
    """Both modes must present the backbone the same number of inputs, or a
    comparison between them would be confounded by capacity."""
    _, off = _fault_setup(False)
    _, on = _fault_setup(True)
    assert off.backbone.net[0].in_features == on.backbone.net[0].in_features
    assert off.fault_coords is False and on.fault_coords is True
    # Off: coords + embedding + appended indicator. On: the indicator is a
    # coordinate, so it is inside the embedding's input instead.
    assert on.embed.in_dim == off.embed.in_dim + 1


def _across_fault_gradient_ratio(fault_coords: bool, scale: float,
                                 seeds: int = 6) -> float:
    """Mean |dh/dn| just inside the barrier, over the same far from it."""
    rng = np.random.default_rng(0)
    near = np.column_stack([rng.uniform(480, 520, 300), rng.uniform(0, 1000, 300)])
    far = np.column_stack([rng.uniform(150, 190, 300), rng.uniform(0, 1000, 300)])

    def grad_n(net, ff, pts):
        xy = torch.as_tensor(pts, dtype=torch.float64).requires_grad_(True)
        h = net(xy, None, ff.side_features(xy))[:, 0]
        return float(torch.autograd.grad(h.sum(), xy)[0][:, 0].abs().mean())

    out = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        ff, net = _fault_setup(fault_coords, fault_sigma_scale=scale)
        out.append(grad_n(net, ff, near) / grad_n(net, ff, far))
    return float(np.mean(out))


def test_coordinate_mapping_concentrates_gradient_across_the_fault():
    """Appending the indicator leaves the field no steeper at the trace than
    anywhere else; mapping the coordinate concentrates gradient there, by an
    amount the bandwidth setting controls."""
    appended = _across_fault_gradient_ratio(False, 0.15)
    assert appended == pytest.approx(1.0, abs=0.25)

    default = _across_fault_gradient_ratio(True, 0.15)
    assert default > 1.4 * appended


def test_fault_bandwidth_controls_the_steepness_monotonically():
    """The knob has to do what it says, since it trades representational
    sharpness against the stiffness of the second-order residual."""
    ratios = [_across_fault_gradient_ratio(True, s, seeds=4)
              for s in (0.05, 0.15, 0.5, 1.0)]
    assert all(a < b for a, b in zip(ratios, ratios[1:]))
    # Full bandwidth is where training was observed to diverge - the basis then
    # oscillates inside the barrier rather than stepping across it.
    assert ratios[-1] > 5.0 * ratios[0]


def test_coordinate_mapping_is_twice_differentiable():
    """The warped coordinates feed the PDE residual, so second derivatives of
    the mapped field must exist and be finite."""
    ff, net = _fault_setup(True)
    xy = torch.tensor([[480.0, 500.0], [520.0, 400.0], [500.0, 600.0]],
                      dtype=torch.float64, requires_grad=True)
    h = net(xy, None, ff.side_features(xy))[:, 0]
    g = torch.autograd.grad(h.sum(), xy, create_graph=True)[0]
    hess = torch.autograd.grad(g[:, 0].sum(), xy, create_graph=True)[0]
    assert torch.isfinite(g).all() and torch.isfinite(hess).all()
    assert g.abs().sum() > 0 and hess.abs().sum() > 0


def test_coordinate_mapping_can_represent_a_sharper_jump():
    """Fit both modes to a step across the trace; the mapped one should get
    closer, because a smooth function of the warped coordinate *is* a step."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    xy = torch.as_tensor(rng.uniform(0, 1000, (600, 2)), dtype=torch.float64)
    target = torch.where(xy[:, 0] > 500.0, 20.0, 10.0)     # exact step at x=500

    errors = {}
    for mode in (False, True):
        ff, net = _fault_setup(mode)
        feats = ff.side_features(xy)
        opt = torch.optim.Adam(net.parameters(), lr=5e-3)
        for _ in range(400):
            opt.zero_grad()
            loss = ((net(xy, None, feats)[:, 0] - target) ** 2).mean()
            loss.backward()
            opt.step()
        errors[mode] = float(loss.detach())

    assert errors[True] < errors[False]
