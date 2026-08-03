"""Tests for Phase 2: scaling, forcing, the physics residuals and the models.

The centrepiece is :func:`test_baseline_residual_matches_manufactured_solution`
-- an analytic head and conductivity field are pushed through the autograd
residual and compared against derivatives worked out by hand.  That is what
proves the PDE loss implements the intended equation rather than merely
converging to something plausible.  A companion test checks that the
second-order and first-order (mixed) formulations agree on the same fields.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gwpinns.benchmark.fields import build_et_max_rate, build_recharge, well_cells
from gwpinns.config import BenchmarkConfig
from gwpinns.pinn.baseline import BaselinePINN
from gwpinns.pinn.cpinn import ConservativePINN
from gwpinns.pinn.forcing import ForcingTerm
from gwpinns.pinn.mixed import MixedPINN
from gwpinns.pinn.networks import FaultConductance, LogConductivityNet
from gwpinns.pinn.sampling import DomainSampler
from gwpinns.pinn.scaling import Scaling

torch.set_default_dtype(torch.float64)

# Analytic manufactured fields, expressed directly in unit coordinates.
A, B, C, D = 1.3, 0.9, 0.4, 0.7
E, F, K0 = 1.1, 0.8, 2.5


def _analytic_head(u: torch.Tensor) -> torch.Tensor:
    X, Y, Z, T = (u[:, i : i + 1] for i in range(4))
    return torch.sin(A * X) * torch.cos(B * Y) * (1.0 + C * Z) * torch.exp(-D * T)


def _analytic_k(u3: torch.Tensor) -> torch.Tensor:
    X, Y = u3[:, 0:1], u3[:, 1:2]
    return K0 * (1.0 + 0.3 * torch.sin(E * X) * torch.cos(F * Y))


def _analytic_divergence(u: torch.Tensor, mu) -> torch.Tensor:
    """``sum_i mu_i^2 d_i(K d_i H)`` worked out by hand."""
    X, Y, Z, T = (u[:, i : i + 1] for i in range(4))
    decay = torch.exp(-D * T)
    sinX, cosX = torch.sin(A * X), torch.cos(A * X)
    cosY, sinY = torch.cos(B * Y), torch.sin(B * Y)
    depth = 1.0 + C * Z

    head = sinX * cosY * depth * decay
    k = _analytic_k(u[:, :3])
    dk_dX = K0 * 0.3 * E * torch.cos(E * X) * torch.cos(F * Y)
    dk_dY = -K0 * 0.3 * F * torch.sin(E * X) * torch.sin(F * Y)

    dH_dX = A * cosX * cosY * depth * decay
    dH_dY = -B * sinX * sinY * depth * decay
    dH_dZ = C * sinX * cosY * decay

    d2H_dX2 = -(A**2) * head
    d2H_dY2 = -(B**2) * head
    d2H_dZ2 = torch.zeros_like(head)

    term_x = mu[0] ** 2 * (dk_dX * dH_dX + k * d2H_dX2)
    term_y = mu[1] ** 2 * (dk_dY * dH_dY + k * d2H_dY2)
    term_z = mu[2] ** 2 * (k * d2H_dZ2)          # K has no z-dependence here
    return term_x + term_y + term_z


class _AnalyticBaseline(BaselinePINN):
    def head_scaled(self, u):
        return _analytic_head(u)

    def conductivity(self, u3):
        return _analytic_k(u3)


class _AnalyticMixed(MixedPINN):
    def fields(self, u):
        head = _analytic_head(u)
        gradient = torch.autograd.grad(
            head, u, grad_outputs=torch.ones_like(head), create_graph=True
        )[0]
        k = _analytic_k(u[:, :3])
        flux = torch.cat(
            [-(k / self.k_ref) * self.mu[i] * gradient[:, i : i + 1] for i in range(3)],
            dim=1,
        )
        return head, flux

    def head_scaled(self, u):
        return self.fields(u)[0]

    def flux_scaled(self, u):
        return self.fields(u)[1]

    def conductivity(self, u3):
        return _analytic_k(u3)


def _unforced_config() -> BenchmarkConfig:
    """Same geometry, but with every source and sink switched off."""
    from dataclasses import replace

    cfg = BenchmarkConfig()
    return replace(
        cfg,
        recharge=replace(cfg.recharge, base_rate=0.0, amplitude=0.0),
        et=replace(cfg.et, max_rate=0.0, amplitude=0.0),
        wells=(),
    )


@pytest.fixture
def unforced():
    cfg = _unforced_config()
    scaling = Scaling.from_config(cfg)
    return cfg, scaling


def _sample_points(cfg, scaling, n=256, seed=3):
    rng = np.random.default_rng(seed)
    grid = cfg.grid
    x = rng.uniform(0, grid.Lx, n)
    y = rng.uniform(0, grid.Ly, n)
    z = rng.uniform(grid.zbot, grid.top, n)
    t = rng.uniform(0, cfg.time.total_time, n)
    cols = [torch.tensor(a).reshape(-1, 1) for a in (x, y, z, t)]
    return cols


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
def test_scaling_maps_domain_to_unit_cube():
    cfg = BenchmarkConfig()
    scaling = Scaling.from_config(cfg)
    corners = torch.tensor(
        [
            [0.0, 0.0, cfg.grid.zbot, 0.0],
            [cfg.grid.Lx, cfg.grid.Ly, cfg.grid.top, cfg.time.total_time],
        ]
    )
    encoded = scaling.encode_xyzt(*(corners[:, i] for i in range(4)))
    assert torch.allclose(encoded[0], torch.tensor([-1.0, -1.0, -1.0, -1.0]))
    assert torch.allclose(encoded[1], torch.tensor([1.0, 1.0, 1.0, 1.0]))


def test_scaling_head_round_trip():
    cfg = BenchmarkConfig()
    scaling = Scaling.from_config(cfg)
    heads = np.array([40.0, 50.0, 63.0])
    assert np.allclose(scaling.decode_head(scaling.encode_head(heads)), heads)


def test_gamma_conductance_round_trip():
    cfg = BenchmarkConfig()
    scaling = Scaling.from_config(cfg)
    conductance = cfg.fault.conductance("barrier")
    gamma = scaling.gamma_from_conductance(conductance)
    assert scaling.conductance_from_gamma(gamma) == pytest.approx(conductance)


def test_pde_coefficients_match_their_definitions():
    cfg = BenchmarkConfig()
    scaling = Scaling.from_config(cfg)
    ss = cfg.aquifer.specific_storage
    assert scaling.alpha == pytest.approx(scaling.a_t / (ss * scaling.L0**2))
    assert scaling.beta == pytest.approx(scaling.a_t / (ss * scaling.dh))
    assert scaling.mu[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Forcing
# --------------------------------------------------------------------------- #
def test_forcing_matches_the_discrete_source_term():
    """``W`` from the PINN must equal what the forward model applied per cell."""
    cfg = BenchmarkConfig()
    forcing = ForcingTerm(cfg).double()
    grid = cfg.grid

    heads = np.full((grid.nlay, grid.nrow, grid.ncol), 52.0)
    time = 3.5 * cfg.time.period_length          # inside stress period 4 (0-based 3)
    w = forcing.on_grid(cfg, heads, time)

    recharge = build_recharge(cfg)
    et_max = build_et_max_rate(cfg)
    expected = np.zeros_like(w)
    expected[0] += recharge / grid.dz[0]
    expected[0] -= cfg.et.rate(heads[0], et_max) / grid.dz[0]
    period = cfg.time.period_of(time)
    for well, (k, i, j) in zip(cfg.wells, well_cells(cfg)):
        expected[k, i, j] += well.rate_at(period) / grid.cell_volume[k]

    assert np.allclose(w, expected, atol=1e-12)


def test_forcing_wells_are_off_during_spin_up():
    cfg = BenchmarkConfig()
    forcing = ForcingTerm(cfg).double()
    grid = cfg.grid
    heads = np.full((grid.nlay, grid.nrow, grid.ncol), 40.0)  # below ET extinction
    w = forcing.on_grid(cfg, heads, 0.0)
    assert np.allclose(w[1:], 0.0)                            # only layer 0 has recharge


def test_evapotranspiration_ramp_is_clamped():
    cfg = BenchmarkConfig()
    forcing = ForcingTerm(cfg).double()
    et_max = torch.tensor([[1.0], [1.0], [1.0]])
    below = cfg.et.surface - cfg.et.extinction_depth - 5.0
    heads = torch.tensor([[below], [cfg.et.surface - 0.5 * cfg.et.extinction_depth],
                          [cfg.et.surface + 5.0]])
    rate = forcing.evapotranspiration_rate(heads, et_max)
    assert rate[0].item() == pytest.approx(0.0)
    assert rate[1].item() == pytest.approx(0.5)
    assert rate[2].item() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Physics residuals -- the manufactured-solution checks
# --------------------------------------------------------------------------- #
def test_baseline_residual_matches_manufactured_solution(unforced):
    cfg, scaling = unforced
    model = _AnalyticBaseline(cfg, scaling, residual_weighting="none").double()
    cols = _sample_points(cfg, scaling)
    u = model.unit_input(*cols)

    computed = model.pde_residual(u)

    head = _analytic_head(u)
    head_t = -D * head
    divergence = _analytic_divergence(u, scaling.mu)
    expected = head_t - scaling.alpha * divergence

    assert torch.allclose(computed, expected, atol=1e-8, rtol=1e-8)


def test_mixed_and_baseline_residuals_agree(unforced):
    """The first-order and second-order formulations are the same equation."""
    cfg, scaling = unforced
    baseline = _AnalyticBaseline(cfg, scaling, residual_weighting="none").double()
    mixed = _AnalyticMixed(cfg, scaling, residual_weighting="none").double()
    cols = _sample_points(cfg, scaling, n=128, seed=11)

    r_baseline = baseline.pde_residual(baseline.unit_input(*cols))
    r_mixed = mixed.continuity_residual(mixed.unit_input(*cols))

    assert torch.allclose(r_baseline, r_mixed, atol=1e-8, rtol=1e-8)


def test_exact_darcy_flux_gives_zero_darcy_residual(unforced):
    cfg, scaling = unforced
    mixed = _AnalyticMixed(cfg, scaling, residual_weighting="none").double()
    cols = _sample_points(cfg, scaling, n=128, seed=5)
    residual = mixed.darcy_residuals(mixed.unit_input(*cols))
    assert residual.abs().max().item() < 1e-10


def test_residual_source_weighting_is_bounded():
    cfg = BenchmarkConfig()
    scaling = Scaling.from_config(cfg)
    model = BaselinePINN(cfg, scaling, residual_weighting="source").double()
    source = torch.tensor([[0.0], [10.0], [-1000.0]])
    scale = model.residual_scale(source)
    assert torch.allclose(scale, torch.tensor([[1.0], [11.0], [1001.0]]))

    plain = BaselinePINN(cfg, scaling, residual_weighting="none").double()
    assert torch.allclose(plain.residual_scale(source), torch.ones_like(source))


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
def test_log_conductivity_is_bounded_and_initialised():
    net = LogConductivityNet(3, log_k_min=-4.0, log_k_max=3.0, log_k_init=0.5).double()
    x = torch.randn(500, 3, dtype=torch.float64) * 3.0
    log_k = net.log10_k(x)
    assert log_k.min().item() >= -4.0
    assert log_k.max().item() <= 3.0
    # Near the origin the field starts close to the requested initial value.
    assert net.log10_k(torch.zeros(1, 3, dtype=torch.float64)).item() == pytest.approx(0.5, abs=0.6)


def test_fault_conductance_modes_and_bounds():
    for mode in ("scalar", "field"):
        net = FaultConductance(mode=mode, log_gamma_init=-1.0).double()
        yz = torch.randn(64, 2, dtype=torch.float64)
        gamma = net(yz)
        assert gamma.shape == (64, 1)
        assert (gamma > 0).all()
        assert torch.isfinite(gamma).all()
    scalar = FaultConductance(mode="scalar", log_gamma_init=-1.0).double()
    value = scalar(torch.zeros(4, 2, dtype=torch.float64))
    assert torch.allclose(value, value[0].expand_as(value))
    assert value[0].item() == pytest.approx(0.1, rel=1e-6)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_collocation_points_lie_in_the_domain():
    cfg = BenchmarkConfig()
    sampler = DomainSampler(cfg, seed=0)
    batch = sampler.collocation(2000)
    assert len(batch) == 2000
    assert (batch.x >= 0).all() and (batch.x <= cfg.grid.Lx).all()
    assert (batch.y >= 0).all() and (batch.y <= cfg.grid.Ly).all()
    assert (batch.z >= cfg.grid.zbot).all() and (batch.z <= cfg.grid.top).all()
    assert (batch.t > 0).all() and (batch.t <= cfg.time.total_time).all()


def test_side_restricted_sampling_stays_in_its_block():
    cfg = BenchmarkConfig()
    sampler = DomainSampler(cfg, seed=1)
    half = 0.5 * cfg.fault.width
    for side in (-1, 1):
        batch = sampler.collocation(1500, side=side)
        dist = cfg.fault.signed_distance(
            batch.x.detach().numpy(), batch.y.detach().numpy()
        )
        assert np.all(side * dist > half - 1e-9)


def test_interface_points_sit_on_the_fault_walls():
    cfg = BenchmarkConfig()
    sampler = DomainSampler(cfg, seed=2)
    iface = sampler.interface(600)
    half = 0.5 * cfg.fault.width

    west = cfg.fault.signed_distance(
        iface.x_west.detach().numpy(), iface.y_west.detach().numpy()
    )
    east = cfg.fault.signed_distance(
        iface.x_east.detach().numpy(), iface.y_east.detach().numpy()
    )
    assert np.allclose(west, -half)
    assert np.allclose(east, half)

    # The two walls are exactly one fault width apart, measured normal to the plane.
    separation = np.hypot(
        (iface.x_east - iface.x_west).detach().numpy(),
        (iface.y_east - iface.y_west).detach().numpy(),
    )
    assert np.allclose(separation, cfg.fault.width)


def test_dirichlet_samples_carry_the_right_boundary_head():
    cfg = BenchmarkConfig()
    sampler = DomainSampler(cfg, seed=4)
    batch = sampler.dirichlet(400)
    x = batch.x.detach().numpy().ravel()
    value = batch.value.detach().numpy().ravel()
    assert np.allclose(value[x == 0.0], cfg.boundary.head_west)
    assert np.allclose(value[x == cfg.grid.Lx], cfg.boundary.head_east)


# --------------------------------------------------------------------------- #
# Model plumbing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("architecture", ["baseline", "mixed", "cpinn"])
def test_models_produce_finite_losses_and_gradients(architecture):
    from gwpinns.pinn.trainer import TrainConfig, _draw_batches, build_model

    cfg = BenchmarkConfig()
    scaling = Scaling.from_config(cfg)
    train_cfg = TrainConfig(architecture=architecture, collocation_points=300,
                            boundary_points=120, interface_points=120)
    model = build_model(cfg, scaling, train_cfg).double()
    sampler = DomainSampler(cfg, seed=7)
    batches = _draw_batches(model, sampler, train_cfg)

    n = 64
    obs = {
        "x": torch.rand(n, 1, dtype=torch.float64) * cfg.grid.Lx,
        "y": torch.rand(n, 1, dtype=torch.float64) * cfg.grid.Ly,
        "z": torch.rand(n, 1, dtype=torch.float64) * cfg.grid.Lz + cfg.grid.zbot,
        "t": torch.rand(n, 1, dtype=torch.float64) * cfg.time.total_time,
        "head": torch.randn(n, 1, dtype=torch.float64) * 0.1,
    }
    loss, report = model.total_loss(
        obs, batches["collocation"], batches["dirichlet"],
        batches["neumann"], batches["interface"],
    )
    assert torch.isfinite(loss)
    assert all(np.isfinite(v) for v in report.values())

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_cpinn_reports_interface_terms_and_baseline_does_not():
    from gwpinns.pinn.trainer import TrainConfig, _draw_batches, build_model

    cfg = BenchmarkConfig()
    scaling = Scaling.from_config(cfg)
    sampler = DomainSampler(cfg, seed=8)

    cpinn_cfg = TrainConfig(architecture="cpinn", collocation_points=200,
                            boundary_points=80, interface_points=100)
    cpinn = build_model(cfg, scaling, cpinn_cfg).double()
    iface = sampler.interface(100)
    terms = cpinn.interface_losses(iface)
    assert set(terms) == {"interface_flux", "interface_head"}

    base_cfg = TrainConfig(architecture="baseline", collocation_points=200)
    baseline = build_model(cfg, scaling, base_cfg).double()
    assert baseline.interface_losses(iface) == {}


def test_cpinn_fault_conductance_reports_physical_units():
    cfg = BenchmarkConfig()
    scaling = Scaling.from_config(cfg)
    model = ConservativePINN(cfg, scaling, gamma_mode="scalar").double()
    sample = model.fault_conductance(n_y=8, n_z=3)

    assert sample["gamma"].shape == (8, 3)
    assert np.all(sample["conductance_per_day"] > 0)
    # K_f = C * width, consistently with how the benchmark defines the zone.
    assert np.allclose(
        sample["equivalent_k_m_per_day"],
        sample["conductance_per_day"] * cfg.fault.width,
    )


def test_cpinn_routes_points_to_the_owning_block():
    cfg = BenchmarkConfig()
    scaling = Scaling.from_config(cfg)
    model = ConservativePINN(cfg, scaling).double()

    west = torch.tensor([[500.0, 1500.0, 30.0, 100.0]], dtype=torch.float64)
    east = torch.tensor([[4500.0, 1500.0, 30.0, 100.0]], dtype=torch.float64)
    cols_w = [west[:, i : i + 1] for i in range(4)]
    cols_e = [east[:, i : i + 1] for i in range(4)]

    u_w = model.unit_input(*cols_w)
    u_e = model.unit_input(*cols_e)
    assert torch.allclose(model.head_scaled(u_w), model.head_nets[0](u_w))
    assert torch.allclose(model.head_scaled(u_e), model.head_nets[1](u_e))
