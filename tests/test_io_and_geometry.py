"""Layer geometry, config round-trip, river stage, and the model domain."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from gwpinn.config import Config, load_config
from gwpinn.geo.grid import build_domain
from gwpinn.geo.river import (
    build_river_stage,
    polygon_centerline,
    project_to_polyline,
)
from gwpinn.io.layers import correct_layer_overlaps
from gwpinn.io.raster import Raster, grid_from_bounds, make_transform
from gwpinn.io.vector import resolve_field


# --------------------------------------------------------------------------- #
# layer geometry
# --------------------------------------------------------------------------- #


def test_overlap_correction_enforces_minimum_thickness():
    top = np.full((10, 10), 100.0)
    b0 = np.full((10, 10), 80.0)
    b1 = np.full((10, 10), 60.0)
    # A ridge that pushes the deeper surface up through the one above it.
    b1[3:6, 3:6] = 95.0

    _, fixed, report = correct_layer_overlaps(top, [b0, b1], min_thickness=2.0)

    assert np.all(top - fixed[0] >= 2.0 - 1e-9)
    assert np.all(fixed[0] - fixed[1] >= 2.0 - 1e-9)
    assert report["layers"][1]["n_cells_corrected"] == 9
    assert report["layers"][1]["max_shift_m"] == pytest.approx(95.0 - 78.0)
    # Untouched cells must keep their original elevation.
    assert fixed[1][0, 0] == pytest.approx(60.0)


def test_overlap_correction_cascades_downward():
    top = np.full((4, 4), 50.0)
    b0 = np.full((4, 4), 60.0)     # above the ground surface
    b1 = np.full((4, 4), 55.0)
    _, fixed, _ = correct_layer_overlaps(top, [b0, b1], min_thickness=1.0)
    assert np.allclose(fixed[0], 49.0)
    assert np.allclose(fixed[1], 48.0)


def test_overlap_correction_fills_missing_data():
    top = np.full((3, 3), 20.0)
    b0 = np.full((3, 3), np.nan)
    _, fixed, report = correct_layer_overlaps(top, [b0], min_thickness=1.0)
    assert np.all(np.isfinite(fixed[0]))
    assert report["layers"][0]["n_cells_filled"] == 9


# --------------------------------------------------------------------------- #
# raster
# --------------------------------------------------------------------------- #


def test_raster_sampling_is_exact_on_a_linear_field():
    tf = make_transform(0.0, 100.0, 10.0)
    xs = np.arange(10) * 10.0 + 5.0
    ys = 100.0 - (np.arange(10) * 10.0 + 5.0)
    xx, yy = np.meshgrid(xs, ys)
    r = Raster(2.0 * xx + 3.0 * yy, tf, None)
    qx = np.array([12.0, 47.0, 88.0])
    qy = np.array([23.0, 51.0, 77.0])
    assert np.allclose(r.sample(qx, qy), 2.0 * qx + 3.0 * qy, atol=1e-9)


def test_raster_sampling_ignores_nodata_neighbours():
    tf = make_transform(0.0, 20.0, 10.0)
    vals = np.array([[1.0, np.nan], [1.0, 1.0]])
    r = Raster(vals, tf, None)
    got = r.sample(np.array([12.0]), np.array([12.0]))
    assert np.isfinite(got[0])
    assert got[0] == pytest.approx(1.0)


def test_raster_bounds_and_centers():
    r = grid_from_bounds((0.0, 0.0, 100.0, 50.0), 10.0)
    assert r.shape == (5, 10)
    assert r.bounds == (0.0, 0.0, 100.0, 50.0)
    xs, ys = r.cell_centers()
    assert xs[0] == pytest.approx(5.0)
    assert ys[0] == pytest.approx(45.0)     # north-up: first row is the top


# --------------------------------------------------------------------------- #
# domain
# --------------------------------------------------------------------------- #


def test_domain_boundary_normals_point_outward():
    poly = Polygon([(0, 0), (1000, 0), (1000, 800), (0, 800)])
    dom = build_domain(domain_polygon=poly, cellsize=50.0)
    rng = np.random.default_rng(0)
    xy, nrm = dom.sample_boundary(200, rng)
    assert len(xy) == 200
    assert np.allclose(np.hypot(nrm[:, 0], nrm[:, 1]), 1.0)
    step = xy + 5.0 * nrm
    assert not dom.contains(step[:, 0], step[:, 1]).any()
    back = xy - 5.0 * nrm
    assert dom.contains(back[:, 0], back[:, 1]).all()


def test_interior_sampling_stays_inside():
    poly = Polygon([(0, 0), (1000, 200), (900, 900), (100, 700)])
    dom = build_domain(domain_polygon=poly, cellsize=25.0)
    pts = dom.sample_interior(500, np.random.default_rng(1))
    assert len(pts) == 500
    assert dom.contains(pts[:, 0], pts[:, 1]).all()


# --------------------------------------------------------------------------- #
# river
# --------------------------------------------------------------------------- #


def test_centerline_follows_a_curved_channel():
    t = np.linspace(0, 4000, 120)
    line = np.column_stack([1000 + 400 * np.sin(t / 900.0), t])
    poly = LineString(line).buffer(120.0)
    cl = polygon_centerline(poly, n_nodes=40, n_samples=4000,
                            rng=np.random.default_rng(0))
    # Every centerline node should sit close to the true axis.
    _, d, _ = project_to_polyline(line, cl)
    assert np.median(d) < 45.0
    assert d.max() < 160.0


def test_stage_interpolation_is_monotone_downstream():
    line = np.column_stack([np.zeros(60), np.linspace(3000, 0, 60)])
    poly = LineString(line).buffer(80.0)
    # Gauges given upstream-first, with one noisy reading that breaks monotonicity.
    g_xy = np.column_stack([np.zeros(5), np.array([3000, 2200, 1500, 800, 0.0])])
    g_stage = np.array([60.0, 57.0, 57.6, 53.0, 50.0])

    rs = build_river_stage(poly, g_xy, g_stage, monotonic=True,
                           rng=np.random.default_rng(0))
    s = np.linspace(rs.s_gauge.min(), rs.s_gauge.max(), 200)
    z = np.asarray(rs.interpolator(s))
    assert np.all(np.diff(z) <= 1e-6), "stage must not rise downstream"
    assert z[0] == pytest.approx(60.0, abs=1.0)
    assert z[-1] == pytest.approx(50.0, abs=1.0)


def test_stage_query_inside_polygon():
    line = np.column_stack([np.zeros(40), np.linspace(2000, 0, 40)])
    poly = LineString(line).buffer(60.0)
    g_xy = np.column_stack([np.zeros(3), np.array([2000, 1000, 0.0])])
    rs = build_river_stage(poly, g_xy, np.array([40.0, 35.0, 30.0]),
                           rng=np.random.default_rng(0))
    z = rs.stage_at(np.array([20.0, -20.0]), np.array([1000.0, 1000.0]))
    assert np.allclose(z, 35.0, atol=1.5)
    assert rs.inside(np.array([0.0]), np.array([1000.0]))[0]
    assert not rs.inside(np.array([500.0]), np.array([1000.0]))[0]


# --------------------------------------------------------------------------- #
# config / attribute resolution
# --------------------------------------------------------------------------- #


def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.model.arch = "resnet"
    cfg.physics.n_layers = 3
    cfg.train.weights.variogram = 0.25
    cfg.paths.dtm = "dtm.tif"
    cfg.save(tmp_path / "c.yaml")

    back = load_config(tmp_path / "c.yaml")
    assert back.model.arch == "resnet"
    assert back.physics.n_layers == 3
    assert back.train.weights.variogram == 0.25
    assert back.paths.dtm == str(tmp_path / "dtm.tif")   # resolved relative to file


def test_config_rejects_unknown_keys(tmp_path):
    (tmp_path / "bad.yaml").write_text("model:\n  arhc: mlp\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(tmp_path / "bad.yaml")


def test_attribute_alias_resolution():
    assert resolve_field(["GW_HEAD", "geometry"], "head") == "GW_HEAD"
    assert resolve_field(["water_leve", "id"], "head") == "water_leve"
    assert resolve_field(["T", "S", "layer"], "T") == "T"
    assert resolve_field(["Transmissi"], "T") == "Transmissi"
    assert resolve_field(["perm_flag"], "perm") == "perm_flag"
    assert resolve_field(["foo", "bar"], "head") is None
