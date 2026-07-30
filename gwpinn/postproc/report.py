"""Accuracy reporting: aggregate metrics, spatial diagnostics, parameter recovery."""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from gwpinn.dataset import GWDataset
from gwpinn.io.raster import Raster, read_raster
from gwpinn.postproc.predict import Prediction, predict_points
from gwpinn.stats.metrics import (
    per_bin_metrics,
    regression_metrics,
    spatial_error_summary,
)
from gwpinn.stats.variogram import experimental_variogram


# --------------------------------------------------------------------------- #
# Reference ("truth") rasters, when a synthetic case is being scored
# --------------------------------------------------------------------------- #


def load_truth(
    truth_dir: str | pathlib.Path, ds: GWDataset, n_layers: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Load reference rasters and resample them onto the model grid."""
    truth_dir = pathlib.Path(truth_dir)
    if not truth_dir.is_dir():
        return {}
    n_layers = n_layers or ds.n_layers
    template = ds.domain.template
    out: Dict[str, np.ndarray] = {}
    for name in ("head", "K", "T", "S"):
        layers = []
        for l in range(n_layers):
            path = truth_dir / f"{name}_L{l}.tif"
            if not path.exists():
                layers = []
                break
            layers.append(read_raster(path).resample_to(template).values)
        if layers:
            out[name] = np.stack(layers)
    return out


# --------------------------------------------------------------------------- #


def build_report(
    models: Sequence,
    ds: GWDataset,
    pred: Prediction,
    histories: Optional[List[List[dict]]] = None,
    truth: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """Assemble every accuracy figure the run can produce."""
    report: Dict = {"config": {}, "head": {}, "properties": {}, "spatial": {},
                    "parameters": {}, "variograms": {}, "grid": {}}

    report["config"] = {
        "architecture": ds.cfg.model.arch,
        "n_layers": ds.n_layers,
        "regime": ds.cfg.physics.regime,
        "n_ensemble": len(models),
        "n_head_obs_train": int(len(ds.train_idx)),
        "n_head_obs_val": int(len(ds.val_idx)),
        "n_property_obs": 0 if ds.prop_obs is None else int(len(ds.prop_obs)),
        "residual_scale": ds.residual_scale,
        "parameters_trained": int(
            sum(p.numel() for p in models[0].parameters() if p.requires_grad)
        ),
    }

    # ---- heads at observation wells --------------------------------------
    obs = ds.head_obs
    obs_layer = np.asarray(obs.get("layer", 0.0), dtype=int)
    p = predict_points(models, ds, obs.xy(), time=None)
    pred_head = p["head"][np.arange(len(obs)), np.clip(obs_layer, 0, ds.n_layers - 1)]
    truth_head = np.asarray(obs.attrs["head"], dtype=float)

    is_val = np.isin(np.arange(len(obs)), ds.val_idx)
    report["head"]["train"] = regression_metrics(truth_head[~is_val], pred_head[~is_val])
    report["head"]["validation"] = regression_metrics(truth_head[is_val], pred_head[is_val])
    report["head"]["all"] = regression_metrics(truth_head, pred_head)
    for l in range(ds.n_layers):
        m = obs_layer == l
        if m.sum() >= 3:
            report["head"][f"layer_{l}"] = regression_metrics(truth_head[m], pred_head[m])

    report["head"]["_obs"] = {
        "x": obs.x.tolist(), "y": obs.y.tolist(),
        "observed": truth_head.tolist(), "predicted": pred_head.tolist(),
        "layer": obs_layer.tolist(), "is_validation": is_val.tolist(),
    }

    # ---- aquifer properties at pumping tests -----------------------------
    if ds.prop_obs is not None and len(ds.prop_obs):
        po = ds.prop_obs
        pl = np.clip(np.asarray(po.get("layer", 0.0), dtype=int), 0, ds.n_layers - 1)
        pp = predict_points(models, ds, po.xy(), time=None)
        rows = np.arange(len(po))
        for name in ("T", "S"):
            if name not in po.attrs:
                continue
            v = np.asarray(po.attrs[name], dtype=float)
            ok = np.isfinite(v) & (v > 0)
            if ok.sum() < 3:
                continue
            pv = pp[f"log10{name}"][rows, pl]
            report["properties"][f"log10{name}"] = regression_metrics(
                np.log10(v[ok]), pv[ok]
            )
            report["properties"][f"_{name}_obs"] = {
                "x": po.x[ok].tolist(), "y": po.y[ok].tolist(),
                "layer": pl[ok].tolist(),
                "observed_log10": np.log10(v[ok]).tolist(),
                "predicted_log10": pv[ok].tolist(),
            }

    # ---- spatial structure of the head residuals -------------------------
    resid = pred_head - truth_head
    report["spatial"]["residuals"] = spatial_error_summary(obs.xy(), resid)

    # How the error grows away from the nearest calibration well is the single
    # most informative spatial diagnostic for an interpolation-type model.
    if len(ds.train_idx) >= 3:
        tree = cKDTree(obs.xy()[~is_val])
        dist, _ = tree.query(obs.xy(), k=1)
        dist_val = dist[is_val]
        if is_val.sum() >= 4:
            report["spatial"]["error_vs_distance"] = per_bin_metrics(
                dist_val, truth_head[is_val], pred_head[is_val],
                n_bins=4, label="distance_m",
            )
        report["spatial"]["_distance_to_train_well"] = dist.tolist()

    # ---- fitted physical parameters --------------------------------------
    m0 = models[0]
    params: Dict = {}
    if m0.faults is not None and m0.faults.has_faults:
        alphas = np.stack([m.faults.alpha().detach().cpu().numpy() for m in models])
        params["fault_permeability"] = {
            "mean": alphas.mean(axis=0).tolist(),
            "std": alphas.std(axis=0).tolist(),
            "trainable": m0.faults.trainable.cpu().numpy().tolist(),
        }
    params["river_conductance"] = float(
        np.mean([float(m.flow.river_conductance.detach()) for m in models])
    )
    if m0.flow.leakance.numel():
        lk = np.stack([m.flow.leakance.detach().cpu().numpy() for m in models])
        params["leakance"] = {"mean": lk.mean(axis=0).tolist(), "std": lk.std(axis=0).tolist()}
    report["parameters"] = params

    # ---- variograms: fitted from points vs realised by the model ---------
    for name, per_layer in ds.variograms.items():
        for layer, model in per_layer.items():
            key = f"{name}_L{layer}"
            entry = {"fitted": model.as_dict()}
            d = ds.variogram_diag.get(name, {}).get(layer, {})
            if d.get("lags") is not None and len(np.atleast_1d(d.get("lags", []))):
                entry["experimental"] = {
                    "lags": np.asarray(d["lags"]).tolist(),
                    "gamma": np.asarray(d["gamma"]).tolist(),
                }
            entry["source"] = d.get("source", "")
            field = pred.mean["log10T" if name == "T" else "log10S"][layer]
            entry["realised"] = _field_variogram(field, ds, model.range_)
            report["variograms"][key] = entry

    # ---- grid-wide comparison against the reference solution --------------
    if truth:
        report["grid"] = _truth_metrics(pred, truth, ds)

    return report


def _field_variogram(field: np.ndarray, ds: GWDataset, range_hint: float) -> Dict:
    """Experimental variogram of a predicted raster, on a random subsample."""
    xx, yy = ds.domain.template.meshgrid()
    ok = np.isfinite(field) & ds.domain.active
    if ok.sum() < 50:
        return {}
    rng = np.random.default_rng(0)
    idx = np.nonzero(ok.ravel())[0]
    take = rng.choice(idx, min(1200, len(idx)), replace=False)
    xy = np.column_stack([xx.ravel()[take], yy.ravel()[take]])
    try:
        lags, gam, _ = experimental_variogram(
            xy, field.ravel()[take], n_lags=12, max_lag=1.5 * range_hint
        )
    except (ValueError, IndexError):
        return {}
    return {"lags": lags.tolist(), "gamma": gam.tolist()}


def _truth_metrics(pred: Prediction, truth: Dict[str, np.ndarray], ds: GWDataset) -> Dict:
    """Cell-by-cell accuracy against the reference rasters."""
    out: Dict = {}
    active = ds.domain.active
    for name, key in (("head", "head"), ("T", "log10T"), ("K", "log10K"), ("S", "log10S")):
        if name not in truth:
            continue
        for l in range(min(ds.n_layers, truth[name].shape[0])):
            t = truth[name][l]
            p = pred.mean[key][l]
            if key.startswith("log10"):
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.log10(np.where(t > 0, t, np.nan))
            m = active & np.isfinite(t) & np.isfinite(p)
            if m.sum() < 10:
                continue
            label = f"{key}_L{l}" if key.startswith("log10") else f"{name}_L{l}"
            out[label] = regression_metrics(t[m], p[m])
    return out


# --------------------------------------------------------------------------- #


def save_report(report: Dict, path: str | pathlib.Path) -> None:
    """Write the report as JSON plus a human-readable summary alongside it."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    with open(path.with_suffix(".txt"), "w") as fh:
        fh.write(format_report(report))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return str(o)


def _fmt(d: Dict, keys=("n", "rmse", "mae", "me", "r2", "nse", "kge", "pbias_pct")) -> str:
    parts = []
    for k in keys:
        if k not in d:
            continue
        v = d[k]
        parts.append(f"{k}={v:d}" if k == "n" else f"{k}={v:.4g}")
    return "  ".join(parts)


def format_report(report: Dict) -> str:
    """Readable summary of the accuracy assessment."""
    L: List[str] = []
    add = L.append

    c = report.get("config", {})
    add("=" * 78)
    add("gwpinn accuracy report")
    add("=" * 78)
    add(
        f"architecture={c.get('architecture')}  layers={c.get('n_layers')}  "
        f"regime={c.get('regime')}  ensemble={c.get('n_ensemble')}  "
        f"trainable params={c.get('parameters_trained')}"
    )
    add(
        f"head observations: {c.get('n_head_obs_train')} train / "
        f"{c.get('n_head_obs_val')} validation   "
        f"property points: {c.get('n_property_obs')}"
    )

    add("")
    add("-- groundwater head at observation wells (m) " + "-" * 33)
    for tag in ("train", "validation", "all"):
        if tag in report.get("head", {}):
            add(f"  {tag:<12s} {_fmt(report['head'][tag])}")
    for k in sorted(report.get("head", {})):
        if k.startswith("layer_"):
            add(f"  {k:<12s} {_fmt(report['head'][k])}")

    props = report.get("properties", {})
    if any(not k.startswith("_") for k in props):
        add("")
        add("-- aquifer properties at pumping tests (log10 units) " + "-" * 25)
        for k in sorted(props):
            if not k.startswith("_"):
                add(f"  {k:<12s} {_fmt(props[k])}")

    grid = report.get("grid", {})
    if grid:
        add("")
        add("-- cell-by-cell vs reference solution " + "-" * 40)
        for k in sorted(grid):
            add(f"  {k:<14s} {_fmt(grid[k])}")

    sp = report.get("spatial", {})
    if sp.get("residuals"):
        mo = sp["residuals"].get("morans", {})
        add("")
        add("-- spatial structure of head residuals " + "-" * 39)
        add(
            f"  Moran's I = {mo.get('morans_i', float('nan')):.4f} "
            f"(expected {mo.get('expected_i', float('nan')):.4f}, "
            f"p = {mo.get('p_value', float('nan')):.3f})"
        )
        nr = sp["residuals"].get("nugget_ratio", float("nan"))
        add(f"  residual variogram nugget / variance = {nr:.3f}  (1.0 = no structure left)")
    if sp.get("error_vs_distance"):
        e = sp["error_vs_distance"]
        add("  validation error vs distance to nearest calibration well:")
        for d, r, n in zip(e["distance_m"], e["rmse"], e["n"]):
            add(f"     {d:8.0f} m   RMSE = {r:6.3f} m   (n={n})")

    par = report.get("parameters", {})
    if par:
        add("")
        add("-- fitted physical parameters " + "-" * 48)
        fp = par.get("fault_permeability")
        if fp:
            for i, (v, tr) in enumerate(zip(fp["mean"], fp["trainable"])):
                kind = "trainable" if tr else "fixed"
                add(f"  fault {i} permeability multiplier = {v:.5f}  ({kind})")
        add(f"  riverbed conductance = {par.get('river_conductance', float('nan')):.5f} 1/d")
        if par.get("leakance"):
            vals = ", ".join(f"{v:.3e}" for v in par["leakance"]["mean"])
            add(f"  vertical leakance    = [{vals}] 1/d")

    vg = report.get("variograms", {})
    if vg:
        add("")
        add("-- geostatistics " + "-" * 61)
        for k, v in sorted(vg.items()):
            f = v["fitted"]
            add(
                f"  {k}: model={f['model']} sill={f['sill']:.4g} "
                f"range={f['range']:.0f} m nugget={f['nugget']:.3g}  [{v.get('source','')}]"
            )
    add("=" * 78)
    return "\n".join(L)


__all__ = ["build_report", "save_report", "format_report", "load_truth"]
