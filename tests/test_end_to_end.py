"""A tiny but complete run: generate data, fit, predict, export, report.

Deliberately fast (a coarse grid and a handful of iterations), so it checks that
the pieces fit together rather than that the answer is accurate.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from gwpinn.config import load_config
from gwpinn.dataset import build_dataset
from gwpinn.data.synthetic import make_synthetic_case
from gwpinn.io.raster import read_raster
from gwpinn.postproc.predict import export_rasters, predict_grid
from gwpinn.postproc.report import build_report, format_report, load_truth
from gwpinn.train.trainer import train_ensemble


@pytest.fixture(scope="module")
def tiny_case(tmp_path_factory):
    out = tmp_path_factory.mktemp("case")
    make_synthetic_case(out, nx=48, ny=40, cellsize=200.0, n_wells=30,
                        n_gauges=4, n_pumping_tests=14, seed=7, verbose=False)
    return out


def _tiny_config(case_dir: pathlib.Path):
    cfg = load_config(case_dir / "config.yaml")
    cfg.model.width, cfg.model.depth = 24, 2
    cfg.model.prop_width, cfg.model.prop_depth = 20, 2
    cfg.model.fourier_features = 12
    cfg.train.adam_iters = 12
    cfg.train.lbfgs_iters = 0
    cfg.train.n_collocation = 128
    cfg.train.n_river = 32
    cfg.train.n_boundary = 32
    cfg.train.n_variogram_pairs = 96
    cfg.train.log_every = 10**6
    return cfg


def test_full_pipeline(tiny_case, tmp_path):
    torch.set_num_threads(1)
    cfg = _tiny_config(tiny_case)
    ds = build_dataset(cfg, verbose=False)

    assert ds.n_layers == 2
    assert len(ds.head_obs) > 10
    assert ds.prop_obs is not None and len(ds.prop_obs) > 5
    assert ds.river is not None
    assert ds.faults is not None and ds.faults.has_faults
    assert len(ds.train_idx) + len(ds.val_idx) == len(ds.head_obs)
    # Layer bottoms are written before correction, so the pipeline must fix them.
    assert ds.layers.report["layers"][1]["n_cells_corrected"] > 0
    for l in range(ds.n_layers):
        th = ds.layers.thickness(l)
        assert np.nanmin(th) >= cfg.domain.min_thickness - 1e-6

    models, histories = train_ensemble(ds, cfg, verbose=False)
    assert len(models) == 1
    assert len(histories[0]) >= 1
    assert np.isfinite(histories[0][-1]["total"])

    pred = predict_grid(models, ds, verbose=False)
    for key in ("head", "K", "T", "S"):
        arr = pred.mean[key]
        assert arr.shape[0] == ds.n_layers
        inside = arr[:, ds.domain.active]
        assert np.isfinite(inside).all(), f"{key} has non-finite values inside domain"
    assert (pred.mean["K"][:, ds.domain.active] > 0).all()
    assert (pred.mean["T"][:, ds.domain.active] > 0).all()
    # Outside the active area everything must stay masked.
    assert np.isnan(pred.mean["head"][0][~ds.domain.active]).all()

    out = tmp_path / "rasters"
    written = export_rasters(pred, ds, out, verbose=False)
    assert len(written) >= 2 * ds.n_layers
    r = read_raster(out / "head_L0.tif")
    assert r.shape == ds.domain.template.shape
    assert np.isfinite(r.values[ds.domain.active]).all()

    truth = load_truth(tiny_case / "truth", ds)
    assert "head" in truth and "T" in truth

    report = build_report(models, ds, pred, histories, truth)
    for tag in ("train", "validation", "all"):
        assert np.isfinite(report["head"][tag]["rmse"])
    assert "log10T" in report["properties"]
    assert report["grid"], "grid comparison against truth should be populated"
    assert "fault_permeability" in report["parameters"]
    assert report["variograms"]

    text = format_report(report)
    assert "gwpinn accuracy report" in text
    assert "Moran's I" in text
    # The report must survive a JSON round-trip (numpy types included).
    json.loads(json.dumps(report, default=str))


def test_plots_are_written(tiny_case, tmp_path):
    torch.set_num_threads(1)
    cfg = _tiny_config(tiny_case)
    ds = build_dataset(cfg, verbose=False)
    models, histories = train_ensemble(ds, cfg, verbose=False)
    pred = predict_grid(models, ds, verbose=False)
    truth = load_truth(tiny_case / "truth", ds)
    report = build_report(models, ds, pred, histories, truth)

    from gwpinn.postproc.plots import make_all_plots

    paths = make_all_plots(models, ds, pred, histories, report, truth, tmp_path / "p")
    assert len(paths) >= 8
    for p in paths:
        assert pathlib.Path(p).stat().st_size > 5000


def test_lbfgs_stage_runs_and_is_logged(tiny_case):
    """The second-order stage only starts after the whole Adam budget, so
    without this it would first be exercised an hour into a production run."""
    torch.set_num_threads(1)
    cfg = _tiny_config(tiny_case)
    cfg.train.adam_iters = 6
    cfg.train.lbfgs_iters = 8
    cfg.train.log_every = 4

    from gwpinn.train.trainer import Trainer

    ds = build_dataset(cfg, verbose=False)
    tr = Trainer(ds, cfg, verbose=False)
    hist = tr.fit()

    stages = {r["stage"] for r in hist}
    assert stages == {"adam", "lbfgs"}
    assert all(np.isfinite(r["total"]) for r in hist)
    assert all(torch.isfinite(p.detach()).all() for p in tr.model.parameters())


def test_ensemble_produces_uncertainty(tiny_case):
    torch.set_num_threads(1)
    cfg = _tiny_config(tiny_case)
    cfg.train.n_ensemble = 2
    ds = build_dataset(cfg, verbose=False)
    models, _ = train_ensemble(ds, cfg, verbose=False)
    assert len(models) == 2
    # Ensemble members must not share fault parameters.
    a = models[0].faults.raw
    b = models[1].faults.raw
    assert a is not b

    pred = predict_grid(models, ds, verbose=False)
    assert pred.n_members == 2
    assert "head" in pred.std
    assert np.isfinite(pred.std["head"][0][ds.domain.active]).all()
    assert (pred.std["head"][0][ds.domain.active] >= 0).all()
