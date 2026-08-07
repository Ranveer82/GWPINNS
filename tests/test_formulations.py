"""Tests for the benchmark case, the formulation components and the criteria.

These are the properties the formulation study *relies* on being true.  If the
control volume does not telescope, the mass-balance criterion means nothing; if
the anisotropy tensor does not actually block flow, the fault criterion means
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gwpinn.benchmark.mf6case import ReducedCase
from gwpinn.formulations.arch import (
    CoordinateNet, MLP, ModifiedMLP, PirateNet, SeparableNet, count_parameters,
)
from gwpinn.formulations.inverse import GridKField, KLKField, NetKField
from gwpinn.formulations.problem import Problem
from gwpinn.formulations.residuals import ControlVolume


# --------------------------------------------------------------------------- #
# A tiny synthetic case, so the tests need no MODFLOW run
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def case() -> ReducedCase:
    rng = np.random.default_rng(0)
    nr, nc, nper = 12, 14, 5
    kh = 10.0 ** rng.normal(0.7, 0.4, (nr, nc))
    botm = np.full((nr, nc), -50.0)
    top = np.full((nr, nc), 20.0)
    faces = np.array([
        [3, 4, 3, 5, 1e-8],
        [4, 4, 4, 5, 1e-8],
        [5, 4, 6, 4, 1e-3],
    ], dtype=float)
    riv = np.array([[nr - 1, c] for c in range(nc)])
    wel = np.array([[4, 6], [8, 3]])
    head = np.stack([np.full((nr, nc), 10.0) + 0.1 * i + 0.05 * rng.standard_normal((nr, nc))
                     for i in range(nper)])
    return ReducedCase(
        nrow=nr, ncol=nc, delr=100.0, delc=100.0, x_origin=0.0, y_origin=0.0,
        crs_epsg=32644, kh=kh, sy=np.full((nr, nc), 0.15),
        ss=np.full((nr, nc), 1e-4), top=top, botm=botm,
        fault_faces=faces, fault_names=["F1", "F2"],
        riv_cells=riv, riv_cond=5000.0, riv_bottom=5.0,
        wel_cells=wel, wel_names=["W1", "W2"],
        times=np.arange(1, nper + 1) * 0.0833, dt=0.0833,
        riv_stage=np.full(nper, 9.0), recharge=np.full(nper, 5e-4),
        wel_q=np.zeros((2, nper)),
        head=head, head_init=head[0],
        budget={}, riv_leakage=np.zeros((nper, len(riv))),
    )


@pytest.fixture(scope="module")
def prob(case) -> Problem:
    return Problem(case)


# --------------------------------------------------------------------------- #
# Control volume: the property the mass criterion depends on
# --------------------------------------------------------------------------- #


def test_cell_balance_telescopes_to_zero(prob):
    """Every face flux enters two cells with opposite sign, so the domain sums to 0.

    This is what makes the FV residual a genuine *local* conservation statement
    rather than a pointwise one, and it must hold for an arbitrary head field -
    not just a converged one.
    """
    cv = ControlVolume(prob)
    g = torch.Generator().manual_seed(0)
    h = 10.0 + torch.randn(cv.n_cells, generator=g)
    b = prob.thickness(h, cv.x, cv.y)
    T = prob.coeff(prob.kh_true, cv.x, cv.y) * b
    net = cv.cell_balance(h, T, b)
    scale = net.abs().sum().item()
    assert abs(float(net.sum())) < 1e-6 * max(scale, 1.0)


def test_barrier_reduces_face_conductance(prob):
    """A blocked face must carry far less flow than an equivalent open one."""
    cv = ControlVolume(prob)
    h = torch.full((cv.n_cells,), 10.0)
    b = prob.thickness(h, cv.x, cv.y)
    T = prob.coeff(prob.kh_true, cv.x, cv.y) * b
    cx, cy = cv.face_conductances(T, b)

    blocked = torch.isfinite(prob.t_face_hydchr_x)
    assert blocked.any()
    assert float(cx[blocked].max()) < 1e-3 * float(cx[~blocked].median())


# --------------------------------------------------------------------------- #
# Faults
# --------------------------------------------------------------------------- #


def test_fault_traces_recovered_from_staircase(prob):
    """Two distinct faults, each with its own hydraulic characteristic."""
    assert prob.n_faults == 2
    assert sorted(prob.fault_hydchr) == [1e-8, 1e-3]


def test_anisotropy_blocks_normal_flow_only(prob):
    """On the trace, the tensor must kill the normal component and keep the tangential."""
    a, b = prob.fault_lines[0]
    mid = torch.tensor([[0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]], dtype=prob.dtype)
    x, y = mid[:, 0], mid[:, 1]
    K = torch.tensor([5.0], dtype=prob.dtype)
    A = prob.anisotropy(x, y, K)[0]

    d = torch.tensor(b - a, dtype=prob.dtype)
    tang = d / d.norm()
    norm = torch.tensor([-tang[1], tang[0]])

    assert float(A @ norm @ norm) < 1e-3          # normal flow suppressed
    assert float(A @ tang @ tang) == pytest.approx(1.0, abs=1e-3)  # along-strike free


def test_fault_features_change_sign_across_trace(prob):
    a, b = prob.fault_lines[0]
    mid = np.array([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])])
    d = b - a
    n = np.array([-d[1], d[0]])
    n = n / np.linalg.norm(n)
    p = mid + 400.0 * n
    m = mid - 400.0 * n
    xs = torch.tensor([p[0], m[0]], dtype=prob.dtype)
    ys = torch.tensor([p[1], m[1]], dtype=prob.dtype)
    f = prob.fault_features(xs, ys)[:, 0]
    assert float(f[0]) * float(f[1]) < 0


# --------------------------------------------------------------------------- #
# Coefficient sampling
# --------------------------------------------------------------------------- #


def test_bilinear_coefficients_have_nonzero_gradient(prob):
    """The strong form needs grad K; nearest-neighbour lookup would zero it."""
    prob.coeff_mode = "bilinear"
    x = torch.tensor([550.0], dtype=prob.dtype, requires_grad=True)
    y = torch.tensor([550.0], dtype=prob.dtype, requires_grad=True)
    k = prob.coeff(prob.kh_true, x, y)
    (gx,) = torch.autograd.grad(k.sum(), x)
    assert abs(float(gx)) > 0.0

    # Nearest lookup is an integer index, so the result is detached from the
    # coordinates entirely - which is exactly why it must not be used where a
    # coefficient gradient is needed.
    prob.coeff_mode = "nearest"
    x2 = torch.tensor([550.0], dtype=prob.dtype, requires_grad=True)
    y2 = torch.tensor([550.0], dtype=prob.dtype, requires_grad=True)
    k2 = prob.coeff(prob.kh_true, x2, y2)
    assert not k2.requires_grad
    prob.coeff_mode = "bilinear"


def test_sources_only_inside_river_cells(prob, case):
    """The river term must be exactly zero outside the RIV cell set."""
    x = torch.tensor([650.0, 650.0], dtype=prob.dtype)
    y = torch.tensor([prob.ymax - 50.0, prob.ymin + 50.0], dtype=prob.dtype)
    t = torch.full((2,), float(case.times[0]), dtype=prob.dtype)
    h = torch.tensor([10.0, 10.0], dtype=prob.dtype)
    src = prob.sources(x, y, t, h)
    assert float(src["riv_mask"][0]) == 0.0    # northern edge: no river
    assert float(src["riv_mask"][1]) == 1.0    # southern edge: river row
    assert float(src["riv"][0]) == 0.0


def test_river_leakage_saturates_below_river_bottom(prob, case):
    """MODFLOW's conductance-limited switch: below rbot the flux stops depending on h."""
    x = torch.tensor([650.0, 650.0], dtype=prob.dtype)
    y = torch.full((2,), prob.ymin + 50.0, dtype=prob.dtype)
    t = torch.full((2,), float(case.times[0]), dtype=prob.dtype)
    deep = torch.tensor([-10.0, -100.0], dtype=prob.dtype)   # both below rbot = 5
    q = prob.sources(x, y, t, deep)["riv"]
    assert float(q[0]) == pytest.approx(float(q[1]), rel=1e-6)


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arch", ["mlp", "modified", "pirate", "spinn"])
def test_backbones_are_twice_differentiable(arch):
    """The strong form takes two derivatives; every backbone must survive it."""
    net = CoordinateNet(3, 1, arch=arch, width=16, depth=2, fourier=8, seed=0)
    x = torch.randn(7, 3, requires_grad=True)
    y = net(x).sum()
    g = torch.autograd.grad(y, x, create_graph=True)[0]
    h = torch.autograd.grad(g[:, 0].sum(), x)[0]
    assert torch.isfinite(h).all()
    assert count_parameters(net) > 0


def test_pirate_block_starts_as_identity():
    """PirateNet's zero-initialised skip means depth costs nothing at init."""
    net = PirateNet(4, 1, width=16, depth=3)
    for blk in net.blocks:
        assert float(blk.alpha) == 0.0


def test_fault_features_reach_the_network():
    """``sidefeat`` appends features; ``faultcoord`` puts them inside the embedding."""
    a = CoordinateNet(3, 1, arch="mlp", width=8, depth=2, fourier=4, n_extra=2,
                      extra_in_fourier=False)
    b = CoordinateNet(3, 1, arch="mlp", width=8, depth=2, fourier=4, n_extra=2,
                      extra_in_fourier=True)
    coords = torch.randn(5, 3)
    extra = torch.randn(5, 2)
    assert a(coords, extra).shape == (5, 1)
    assert b(coords, extra).shape == (5, 1)
    # The embedding sees 5 inputs in the faultcoord case and 3 otherwise.
    assert a.embed.B.shape[0] == 3
    assert b.embed.B.shape[0] == 5


# --------------------------------------------------------------------------- #
# Inverse parameterisations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", [NetKField, GridKField, KLKField])
def test_k_fields_are_positive_and_shaped(prob, cls):
    kf = cls(prob)
    grid = kf.as_grid()
    assert grid.shape == (prob.nrow, prob.ncol)
    assert float(grid.min()) > 0.0
    assert torch.isfinite(grid).all()


def test_kl_field_starts_at_prior_mean_and_responds_to_coefficients(prob):
    """Zero coefficients => the prior mean; perturbing them must vary the field."""
    kf = KLKField(prob, n_modes=64, k0=5.0)
    flat = kf.as_grid()
    assert float(flat.std()) < 1e-6
    assert float(flat.mean()) == pytest.approx(5.0, rel=1e-3)

    with torch.no_grad():
        kf.coef_cos.normal_(0.0, 1.0)
    varied = kf.as_grid()
    assert float(varied.std()) > 0.0


def test_grid_tv_penalty_prefers_piecewise_constant(prob):
    """The pseudo-Huber penalty must charge less for one big step than for noise."""
    smooth = GridKField(prob)
    noisy = GridKField(prob)
    with torch.no_grad():
        smooth.raw[:, prob.ncol // 2:] += 3.0            # a single facies contact
        noisy.raw.normal_(0.0, 1.0)                       # speckle of similar range
    assert float(smooth.regulariser()) < float(noisy.regulariser())
