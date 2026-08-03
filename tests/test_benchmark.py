"""Tests for Phase 1: geometry, fields and the forward solver.

The important one here is :func:`test_fd_solver_matches_theis` -- it validates
the reference finite-difference solver against a closed-form analytical
solution, which is what lets the benchmark be trusted in environments where the
compiled MODFLOW 6 binary is unavailable.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import scipy.special

from gwpinns.benchmark.fdsolver import build_conductance_matrix, solve_forward
from gwpinns.benchmark.fields import (
    build_conductivity,
    build_recharge,
    fault_mask,
    gaussian_random_field_2d,
)
from gwpinns.benchmark.generate import head_jump
from gwpinns.benchmark.observations import sample_observations
from gwpinns.config import (
    AquiferConfig,
    BenchmarkConfig,
    BoundaryConfig,
    EvapotranspirationConfig,
    GridConfig,
    RechargeConfig,
    TimeConfig,
    WellConfig,
)


def test_grid_geometry_is_consistent():
    grid = GridConfig()
    assert grid.nlay == 3
    assert grid.dx == pytest.approx(100.0)
    assert grid.dy == pytest.approx(100.0)
    assert grid.dz.sum() == pytest.approx(grid.Lz)
    assert grid.z_centers()[0] > grid.z_centers()[-1]      # z increases upwards

    # Row 0 is the northern edge.
    assert grid.y_centers()[0] > grid.y_centers()[-1]
    row, col = grid.locate(250.0, grid.Ly - 150.0)
    assert (row, col) == (1, 2)

    for layer, z in enumerate(grid.z_centers()):
        assert grid.layer_of(float(z)) == layer


def test_signed_distance_and_fault_zone():
    cfg = BenchmarkConfig()
    fault = cfg.fault
    assert fault.signed_distance(fault.x0, fault.y0) == pytest.approx(0.0)

    # The normal has unit length, so signed distance is a true distance.
    nx, ny = fault.normal
    assert nx**2 + ny**2 == pytest.approx(1.0)

    step = 200.0
    assert fault.signed_distance(fault.x0 + step * nx, fault.y0 + step * ny) == pytest.approx(step)
    assert fault.in_fault_zone(fault.x0, fault.y0)
    assert not fault.in_fault_zone(fault.x0 + 500.0, fault.y0)


def test_random_field_is_standardised():
    rng = np.random.default_rng(0)
    field = gaussian_random_field_2d(30, 50, 100.0, 100.0, 700.0, 900.0, rng)
    assert field.shape == (30, 50)
    assert field.mean() == pytest.approx(0.0, abs=1e-10)
    assert field.std() == pytest.approx(1.0, rel=1e-10)


def test_conductivity_has_a_sharp_fault():
    for scenario, expected in (("barrier", 1.0e-3), ("conduit", 200.0)):
        cfg = BenchmarkConfig(scenario=scenario)
        k = build_conductivity(cfg)
        mask = fault_mask(cfg)
        assert mask.any()
        assert np.allclose(k[mask], expected)
        # The background is untouched and log-normally spread.
        assert k[~mask].min() > 0.0
        assert not np.allclose(k[~mask], expected)


def test_recharge_is_non_negative_and_varies():
    cfg = BenchmarkConfig()
    recharge = build_recharge(cfg)
    assert recharge.shape == (cfg.grid.nrow, cfg.grid.ncol)
    assert (recharge >= 0).all()
    assert recharge.std() > 0


def test_config_json_round_trip(tmp_path):
    cfg = BenchmarkConfig(scenario="conduit")
    path = tmp_path / "config.json"
    cfg.to_json(path)
    restored = BenchmarkConfig.from_json(path)
    assert restored == cfg


def test_conductance_matrix_is_symmetric_with_zero_row_sums():
    cfg = BenchmarkConfig()
    k = build_conductivity(cfg)
    mat = build_conductance_matrix(cfg, k)
    assert (abs(mat - mat.T) > 1e-9).nnz == 0
    # A pure conductance operator annihilates a uniform head field.
    assert np.abs(mat @ np.ones(mat.shape[0])).max() < 1e-6


def test_harmonic_mean_conductance():
    """Two cells of different K must combine harmonically, as MODFLOW does."""
    grid = GridConfig(Lx=200.0, Ly=100.0, ncol=2, nrow=1, top=10.0, botm=(0.0,))
    cfg = BenchmarkConfig(grid=grid, wells=())
    k = np.array([[[1.0, 9.0]]])
    mat = build_conductance_matrix(cfg, k)
    expected = (2.0 * grid.dy * 10.0) / (grid.dx / 1.0 + grid.dx / 9.0)
    assert mat[0, 1] == pytest.approx(-expected)
    assert mat[0, 0] == pytest.approx(expected)


def _uniform_config(k: float = 10.0, **overrides) -> BenchmarkConfig:
    """A homogeneous, unstressed configuration for analytical comparisons.

    The fault is given the host conductivity so it is hydraulically invisible --
    the fault zone is always burned into the K field, so it has to be
    neutralised rather than simply left unconfigured.
    """
    base = dict(
        grid=GridConfig(Lx=5000.0, Ly=5000.0, ncol=50, nrow=50, top=50.0, botm=(0.0,)),
        aquifer=AquiferConfig(k_layer_gmean=(k,), sigma_lnk=0.0, specific_storage=2.0e-4),
        recharge=RechargeConfig(base_rate=0.0, amplitude=0.0),
        et=EvapotranspirationConfig(max_rate=0.0, amplitude=0.0),
        boundary=BoundaryConfig(head_west=50.0, head_east=50.0),
        time=TimeConfig(n_periods=1, period_length=10.0, steps_per_period=40),
        fault=replace(BenchmarkConfig().fault, k_barrier=k, k_conduit=k),
        wells=(),
    )
    base.update(overrides)
    return BenchmarkConfig(**base)


def test_fd_solver_matches_theis():
    """Drawdown around a pumped well must follow the Theis solution.

    A single fully penetrating well in a homogeneous confined aquifer with no
    other stresses.  Comparison is restricted to radii between 4 and 12 cells:
    closer in, the finite-difference cell average legitimately differs from the
    point solution; further out, the constant-head boundaries intrude.
    """
    rate = -2000.0
    cfg = _uniform_config(
        wells=(WellConfig(name="T1", x=2525.0, y=2475.0, layer=0, rates=(rate,)),)
    )
    grid = cfg.grid
    thickness = grid.Lz
    transmissivity = 10.0 * thickness
    storativity = cfg.aquifer.specific_storage * thickness

    solution = solve_forward(cfg)
    elapsed = solution.times[-1]
    drawdown = solution.heads[0, 0] - solution.heads[-1, 0]

    well_row, well_col = grid.locate(2525.0, 2475.0)
    x = grid.x_centers()
    y = grid.y_centers()
    xx, yy = np.meshgrid(x, y, indexing="xy")
    radius = np.hypot(xx - x[well_col], yy - y[well_row])

    band = (radius > 4 * grid.dx) & (radius < 12 * grid.dx)
    u = radius[band] ** 2 * storativity / (4.0 * transmissivity * elapsed)
    theis = -rate / (4.0 * np.pi * transmissivity) * scipy.special.exp1(u)

    numerical = drawdown[band]
    assert np.allclose(numerical, theis, rtol=0.05), (
        f"max relative error {np.max(np.abs(numerical - theis) / theis):.3f}"
    )


def test_fd_solver_matches_one_dimensional_barrier():
    """Steady 1-D flow through a low-K slab must match the series-resistance law."""
    grid = GridConfig(Lx=1000.0, Ly=100.0, ncol=100, nrow=1, top=10.0, botm=(0.0,))
    k_host, k_slab, width = 10.0, 0.01, 50.0
    cfg = BenchmarkConfig(
        grid=grid,
        aquifer=AquiferConfig(k_layer_gmean=(k_host,), sigma_lnk=0.0),
        recharge=RechargeConfig(base_rate=0.0, amplitude=0.0),
        et=EvapotranspirationConfig(max_rate=0.0, amplitude=0.0),
        boundary=BoundaryConfig(head_west=20.0, head_east=10.0),
        time=TimeConfig(n_periods=1, period_length=1.0, steps_per_period=1),
        wells=(),
        fault=replace(
            BenchmarkConfig().fault, x0=500.0, y0=50.0, strike_deg=0.0,
            width=width, k_barrier=k_slab,
        ),
    )
    solution = solve_forward(cfg)
    head = solution.heads[0, 0, 0]

    # Series resistance: total head drop splits in proportion to L/K.
    length_host = grid.Lx - width - grid.dx      # minus the constant-head cell widths
    resistance_host = length_host / k_host
    resistance_slab = width / k_slab
    total_drop = 20.0 - 10.0
    expected_slab_drop = total_drop * resistance_slab / (resistance_host + resistance_slab)

    observed = head_jump(cfg, solution.heads[0], band=300.0)
    assert observed == pytest.approx(expected_slab_drop, rel=0.05)


def test_barrier_and_conduit_produce_opposite_signatures():
    barrier = solve_forward(BenchmarkConfig(scenario="barrier"))
    conduit = solve_forward(BenchmarkConfig(scenario="conduit"))
    barrier_jump = head_jump(BenchmarkConfig(scenario="barrier"), barrier.heads[0])
    conduit_jump = head_jump(BenchmarkConfig(scenario="conduit"), conduit.heads[0])

    assert barrier_jump > 3.0, "a barrier must sustain a large head jump"
    assert abs(conduit_jump) < 0.2, "a conduit must nearly erase the head jump"


def test_observations_avoid_the_fault_zone_and_carry_noise():
    cfg = BenchmarkConfig(scenario="barrier")
    solution = solve_forward(cfg)
    obs = sample_observations(cfg, solution)

    assert len(obs) > 0
    assert obs.n_points == np.unique(obs.name).size
    min_offset = 0.5 * cfg.fault.width + cfg.observations.min_fault_offset
    assert np.all(np.abs(obs.signed_distance) >= min_offset)

    residual = obs.head_obs - obs.head_true
    assert residual.std() == pytest.approx(cfg.observations.noise_std, rel=0.25)
    assert obs.head_obs is not obs.head_true
