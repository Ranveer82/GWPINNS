"""Variogram fitting, kriging, and the variogram loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gwpinn.data.grf import gaussian_random_field
from gwpinn.stats.variogram import (
    VariogramModel,
    experimental_variogram,
    fit_variogram,
    fit_variogram_by_layer,
    ordinary_kriging,
)
from gwpinn.train.losses import variogram_loss


def test_model_shapes():
    for name in ("exponential", "spherical", "gaussian"):
        m = VariogramModel(name, nugget=0.1, sill=1.0, range_=100.0)
        assert m.gamma(np.array([0.0]))[0] == pytest.approx(0.0)
        # 95% of the sill is reached at the practical range, by definition.
        assert m.gamma(np.array([100.0]))[0] == pytest.approx(1.0, rel=0.06)
        assert m.gamma(np.array([1e6]))[0] == pytest.approx(1.0, rel=1e-3)
        assert np.all(np.diff(m.gamma(np.linspace(0, 300, 50))) >= -1e-12)


def test_recovers_the_generating_variogram():
    """Sample a field with a known variogram; the fit must find it back."""
    truth = VariogramModel("exponential", nugget=0.0, sill=0.5, range_=1500.0)
    field = gaussian_random_field((160, 160), 50.0, truth,
                                  rng=np.random.default_rng(3), mean=1.0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 160, (400, 2))
    xy = idx[:, ::-1] * 50.0
    v = field[idx[:, 0], idx[:, 1]]

    fitted, diag = fit_variogram(xy, v, model="exponential", n_lags=14, max_lag=4000.0)
    assert fitted.sill == pytest.approx(truth.sill, rel=0.45)
    assert fitted.range_ == pytest.approx(truth.range_, rel=0.6)
    assert diag["fit_r2"] > 0.8


def test_range_cannot_exceed_the_sampled_extent():
    """A pure-nugget field must not report an unbounded correlation range."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 1000, (120, 2))
    v = rng.standard_normal(120)          # white noise: no spatial structure
    fitted, _ = fit_variogram(xy, v, n_lags=10, max_lag=500.0)
    assert fitted.range_ <= 500.0 * 1.5 + 1e-6


def test_per_layer_fitting_separates_populations():
    """Layers with very different means must not pool into one huge sill."""
    rng = np.random.default_rng(5)
    n = 60
    xy = np.vstack([rng.uniform(0, 5000, (n, 2)), rng.uniform(0, 5000, (n, 2))])
    layer = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    # Same texture, wildly different level - like S in an unconfined vs a
    # confined layer (0.1 against 2e-4).
    v = np.concatenate([
        -1.0 + 0.2 * rng.standard_normal(n),
        -3.7 + 0.2 * rng.standard_normal(n),
    ])

    models, diags = fit_variogram_by_layer(xy, v, layer, 2, n_lags=8, max_lag=2500.0)
    assert set(models) == {0, 1}
    for l in (0, 1):
        # Pooling would give a sill near (1.35)^2 ~ 1.8 instead of ~0.04.
        assert models[l].sill < 0.3

    pooled, _ = fit_variogram(xy, v, n_lags=8, max_lag=2500.0)
    assert pooled.sill > 1.0


def test_kriging_honours_the_measurements():
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 1000, (30, 2))
    v = np.sin(xy[:, 0] / 300.0) + 0.5 * np.cos(xy[:, 1] / 250.0)
    model = VariogramModel("exponential", nugget=0.0, sill=1.0, range_=400.0)

    est, var = ordinary_kriging(xy, v, xy, model)
    assert np.allclose(est, v, atol=1e-3)
    # Variance vanishes at a datum and grows away from the data.
    assert np.max(var) < 1e-3
    _, var_far = ordinary_kriging(xy, v, np.array([[5000.0, 5000.0]]), model)
    assert var_far[0] > 0.3


def test_variogram_loss_is_zero_for_a_matching_field():
    model = VariogramModel("exponential", nugget=0.0, sill=1.0, range_=500.0)
    lag = torch.tensor([100.0, 300.0, 600.0, 900.0], dtype=torch.float64).repeat(50)
    bins = torch.arange(4).repeat(50)
    gam = model.gamma(lag)
    # Construct a pair difference with exactly the target semivariance.
    d = torch.sqrt(2.0 * gam)
    v1 = torch.zeros_like(d)
    loss = variogram_loss(v1, d, lag, bins, model, n_bins=4)
    assert float(loss) < 1e-12


def test_variogram_loss_backpropagates():
    model = VariogramModel("exponential", nugget=0.0, sill=1.0, range_=500.0)
    lag = torch.tensor([100.0, 400.0], dtype=torch.float64).repeat(20)
    bins = torch.arange(2).repeat(20)
    v1 = torch.zeros(40, dtype=torch.float64, requires_grad=True)
    v2 = torch.full((40,), 0.05, dtype=torch.float64, requires_grad=True)
    loss = variogram_loss(v1, v2, lag, bins, model, n_bins=2)
    loss.backward()
    assert v2.grad is not None and torch.isfinite(v2.grad).all()
    assert v2.grad.abs().sum() > 0


def test_experimental_variogram_bins():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (60, 2))
    v = rng.standard_normal(60)
    lags, gam, counts = experimental_variogram(xy, v, n_lags=6, max_lag=80.0)
    assert len(lags) == len(gam) == len(counts)
    assert np.all(counts > 0)
    assert np.all(np.diff(lags) > 0)
