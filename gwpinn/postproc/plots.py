"""Validation figures.

Colour conventions, applied consistently so the figures read as one set:

* **magnitude** (head, transmissivity) - a perceptually uniform sequential map
  (``viridis`` / ``magma``), monotone in lightness and colourblind-safe.
* **signed error** - a diverging map with a neutral midpoint (``RdBu_r``) and
  limits forced symmetric about zero, so the sign of a residual is never an
  artefact of the colour scaling.
* **series** - at most a few, always with a legend and a distinct marker, so
  identity never rests on colour alone.

Quantities of different scale go in separate panels rather than on a second
y-axis.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from gwpinn.dataset import GWDataset
from gwpinn.postproc.predict import Prediction
from gwpinn.stats.variogram import VariogramModel

# --------------------------------------------------------------------------- #

SEQ_HEAD = "viridis"
SEQ_PROP = "magma"
DIVERGING = "RdBu_r"
C_TRAIN = "#4269D0"
C_VAL = "#EFB118"
C_MODEL = "#3CA951"
C_REF = "#333333"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "legend.frameon": False,
    }
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _extent(ds: GWDataset):
    xmin, ymin, xmax, ymax = ds.domain.template.bounds
    return (xmin, xmax, ymin, ymax)


def _imshow(ax, arr, ds, cmap=SEQ_HEAD, symmetric=False, vmin=None, vmax=None, **kw):
    a = np.asarray(arr, dtype=float)
    if symmetric:
        lim = np.nanpercentile(np.abs(a), 99) if np.isfinite(a).any() else 1.0
        lim = max(float(lim), 1e-9)
        vmin, vmax = -lim, lim
    im = ax.imshow(
        a, extent=_extent(ds), origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", **kw,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return im


def _overlay(ax, ds, faults=True, river=True, wells=False, lw=1.1):
    if faults and ds.fault_lines is not None:
        perm = ds.fault_lines.attrs.get("perm")
        for i, ln in enumerate(ds.fault_lines.lines):
            style = "-" if (perm is None or not np.isclose(perm[i], 1.0)) else "--"
            ax.plot(ln[:, 0], ln[:, 1], color="#D62728", lw=lw, ls=style, zorder=5)
    if river and ds.river is not None and ds.river.polygon is not None:
        polys = getattr(ds.river.polygon, "geoms", [ds.river.polygon])
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color="#7EC8E3", alpha=0.55, zorder=4, lw=0)
    if wells:
        ax.scatter(
            ds.head_obs.x, ds.head_obs.y, s=9, c="white", edgecolors="black",
            linewidths=0.4, zorder=6,
        )


def _cb(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


def _save(fig, path: pathlib.Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# 1. training history
# --------------------------------------------------------------------------- #


def plot_training(histories: List[List[dict]], path: pathlib.Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    h = histories[0]
    it = [r["iter"] + (0 if r["stage"] == "adam" else max(x["iter"] for x in h
          if x["stage"] == "adam")) for r in h]

    keys = [k for k in h[0] if k.startswith("loss_")]
    ax = axes[0]
    for k in keys:
        v = np.array([r[k] for r in h], dtype=float)
        if np.nanmax(v) <= 0:
            continue
        ax.plot(it, np.maximum(v, 1e-16), lw=1.2, label=k[5:])
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss term")
    ax.set_title("Loss components")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for k, c, m in (("train_rmse", C_TRAIN, "o"), ("val_rmse", C_VAL, "s")):
        if k in h[0]:
            ax.plot(it, [r[k] for r in h], color=c, lw=1.6, marker=m,
                    markevery=max(len(it) // 12, 1), ms=4, label=k.replace("_", " "))
    ax.set_xlabel("iteration")
    ax.set_ylabel("head RMSE (m)")
    ax.set_title("Fit to observation wells")
    ax.legend(fontsize=8)

    ax = axes[2]
    wkeys = [k for k in h[0] if k.startswith("w_")]
    for k in wkeys:
        ax.plot(it, [r[k] for r in h], lw=1.2, label=k[2:])
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("adaptive weight")
    ax.set_title("Gradient-norm loss balancing")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Training history", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 2. observed vs predicted
# --------------------------------------------------------------------------- #


def plot_head_scatter(report: Dict, path: pathlib.Path) -> str:
    o = report["head"]["_obs"]
    obs = np.array(o["observed"])
    pred = np.array(o["predicted"])
    isval = np.array(o["is_validation"], dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    lo, hi = np.nanmin([obs, pred]), np.nanmax([obs, pred])
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=C_REF, lw=1, ls="--",
            label="1:1", zorder=1)
    ax.scatter(obs[~isval], pred[~isval], s=26, c=C_TRAIN, marker="o",
               edgecolors="white", linewidths=0.5, label="calibration", zorder=3)
    ax.scatter(obs[isval], pred[isval], s=42, c=C_VAL, marker="s",
               edgecolors="black", linewidths=0.5, label="validation (held out)",
               zorder=4)
    ax.set_xlabel("observed head (m a.s.l.)")
    ax.set_ylabel("predicted head (m a.s.l.)")
    m = report["head"]["validation"]
    t = report["head"]["train"]
    ax.set_title(
        f"RMSE cal {t['rmse']:.2f} m / val {m['rmse']:.2f} m\n"
        f"R² val {m['r2']:.3f}   NSE val {m['nse']:.3f}"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    res = pred - obs
    bins = np.linspace(np.nanmin(res), np.nanmax(res), 22)
    ax.hist(res[~isval], bins=bins, color=C_TRAIN, alpha=0.75, label="calibration")
    ax.hist(res[isval], bins=bins, color=C_VAL, alpha=0.85, label="validation")
    ax.axvline(0, color=C_REF, lw=1, ls="--")
    ax.set_xlabel("residual, predicted − observed (m)")
    ax.set_ylabel("count")
    ax.set_title(f"Bias {report['head']['all']['me']:+.3f} m")
    ax.legend(fontsize=8)

    ax = axes[2]
    props = {k: v for k, v in report.get("properties", {}).items()
             if not k.startswith("_")}
    if props:
        markers = {"log10T": ("o", C_TRAIN), "log10S": ("^", C_MODEL)}
        allv = []
        for name in props:
            d = report["properties"].get(f"_{name.replace('log10', '')}_obs")
            if not d:
                continue
            mk, c = markers.get(name, ("o", C_TRAIN))
            ax.scatter(d["observed_log10"], d["predicted_log10"], s=34, c=c,
                       marker=mk, edgecolors="white", linewidths=0.5,
                       label=f"{name} (RMSE {props[name]['rmse']:.2f})")
            allv += list(d["observed_log10"]) + list(d["predicted_log10"])
        if allv:
            lo, hi = min(allv), max(allv)
            ax.plot([lo, hi], [lo, hi], color=C_REF, lw=1, ls="--", label="1:1")
        ax.set_xlabel("observed (log10)")
        ax.set_ylabel("predicted (log10)")
        ax.set_title("Aquifer properties at pumping tests")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    fig.suptitle("Goodness of fit", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 3./4. maps
# --------------------------------------------------------------------------- #


def plot_field_maps(
    pred: Prediction,
    ds: GWDataset,
    truth: Optional[Dict[str, np.ndarray]],
    key: str,
    truth_key: str,
    label: str,
    path: pathlib.Path,
    cmap: str = SEQ_HEAD,
    log_truth: bool = False,
) -> str:
    n_layers = ds.n_layers
    has_truth = truth is not None and truth_key in truth
    ncols = 3 if has_truth else 1
    fig, axes = plt.subplots(
        n_layers, ncols, figsize=(4.6 * ncols, 3.7 * n_layers), squeeze=False
    )

    for l in range(n_layers):
        p = pred.mean[key][l]
        t = None
        if has_truth and l < truth[truth_key].shape[0]:
            t = truth[truth_key][l].astype(float)
            if log_truth:
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.log10(np.where(t > 0, t, np.nan))

        vmin = np.nanpercentile(np.concatenate([
            p[np.isfinite(p)].ravel(),
            t[np.isfinite(t)].ravel() if t is not None else np.zeros(0)]), 1)
        vmax = np.nanpercentile(np.concatenate([
            p[np.isfinite(p)].ravel(),
            t[np.isfinite(t)].ravel() if t is not None else np.zeros(0)]), 99)

        ax = axes[l][0]
        im = _imshow(ax, p, ds, cmap=cmap, vmin=vmin, vmax=vmax)
        _overlay(ax, ds)
        ax.set_title(f"PINN estimate — layer {l}")
        _cb(fig, im, ax, label)

        if t is not None:
            ax = axes[l][1]
            im = _imshow(ax, t, ds, cmap=cmap, vmin=vmin, vmax=vmax)
            _overlay(ax, ds)
            ax.set_title(f"reference solution — layer {l}")
            _cb(fig, im, ax, label)

            ax = axes[l][2]
            err = p - t
            im = _imshow(ax, err, ds, cmap=DIVERGING, symmetric=True)
            _overlay(ax, ds, wells=True)
            good = np.isfinite(err) & ds.domain.active
            rmse = float(np.sqrt(np.nanmean(err[good] ** 2))) if good.any() else np.nan
            ax.set_title(f"error (PINN − reference)\nRMSE {rmse:.3g}")
            _cb(fig, im, ax, f"Δ {label}")

    fig.suptitle(label, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 5. variograms
# --------------------------------------------------------------------------- #


def plot_variograms(report: Dict, ds: GWDataset, path: pathlib.Path) -> Optional[str]:
    vg = report.get("variograms", {})
    if not vg:
        return None
    keys = sorted(vg)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.3 * len(keys), 3.7), squeeze=False)

    for ax, k in zip(axes[0], keys):
        entry = vg[k]
        f = entry["fitted"]
        model = VariogramModel(f["model"], f["nugget"], f["sill"], f["range"])

        hmax = 1.5 * model.range_
        hh = np.linspace(0, hmax, 200)
        ax.plot(hh, model.gamma(hh), color=C_REF, lw=1.8,
                label=f"fitted model\n(range {model.range_:.0f} m, sill {model.sill:.3f})")

        exp = entry.get("experimental")
        if exp and len(exp.get("lags", [])):
            ax.scatter(exp["lags"], exp["gamma"], s=34, c=C_TRAIN, marker="o",
                       edgecolors="white", linewidths=0.5,
                       label="experimental (point data)", zorder=4)

        real = entry.get("realised")
        if real and len(real.get("lags", [])):
            ax.plot(real["lags"], real["gamma"], color=C_MODEL, lw=1.8, ls="--",
                    marker="^", ms=4, label="realised by PINN field")

        ax.axvline(model.range_, color=C_REF, lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel("lag distance (m)")
        ax.set_ylabel("semivariance")
        ax.set_title(f"log10 {k}")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Spatial structure: does the fitted field reproduce the measured variogram?",
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 6. spatial residual diagnostics
# --------------------------------------------------------------------------- #


def plot_residual_space(report: Dict, ds: GWDataset, path: pathlib.Path) -> str:
    o = report["head"]["_obs"]
    x, y = np.array(o["x"]), np.array(o["y"])
    res = np.array(o["predicted"]) - np.array(o["observed"])
    isval = np.array(o["is_validation"], dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    lim = float(np.nanpercentile(np.abs(res), 98)) or 1.0
    sc = ax.scatter(x, y, c=res, s=40 + 130 * np.abs(res) / max(lim, 1e-9),
                    cmap=DIVERGING, vmin=-lim, vmax=lim, edgecolors="black",
                    linewidths=0.4, zorder=5)
    ax.scatter(x[isval], y[isval], s=170, facecolors="none", edgecolors=C_VAL,
               linewidths=1.6, zorder=6, label="validation wells")
    _overlay(ax, ds)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    mo = report["spatial"]["residuals"]["morans"]
    ax.set_title(
        f"Head residuals in space\nMoran's I = {mo.get('morans_i', float('nan')):.3f} "
        f"(p = {mo.get('p_value', float('nan')):.3f})"
    )
    ax.legend(fontsize=8, loc="lower left")
    _cb(fig, sc, ax, "residual (m)")

    ax = axes[1]
    rv = report["spatial"]["residuals"].get("residual_variogram")
    if rv and len(rv.get("lags", [])):
        ax.plot(rv["lags"], rv["gamma"], color=C_TRAIN, lw=1.8, marker="o", ms=4,
                label="residual semivariogram")
        ax.axhline(float(np.var(res)), color=C_REF, ls="--", lw=1,
                   label="residual variance")
        ax.set_xlabel("lag distance (m)")
        ax.set_ylabel("semivariance (m²)")
        ax.legend(fontsize=8)
    ax.set_title("Flat ⇒ no spatial structure left in the error")

    ax = axes[2]
    e = report["spatial"].get("error_vs_distance")
    if e and e.get("distance_m"):
        ax.bar(range(len(e["distance_m"])), e["rmse"], color=C_VAL,
               edgecolor="white", width=0.7)
        ax.set_xticks(range(len(e["distance_m"])))
        ax.set_xticklabels([f"{d:.0f}" for d in e["distance_m"]])
        for i, (r, n) in enumerate(zip(e["rmse"], e["n"])):
            ax.text(i, r, f"n={n}", ha="center", va="bottom", fontsize=7)
        ax.set_xlabel("distance to nearest calibration well (m)")
        ax.set_ylabel("validation RMSE (m)")
    ax.set_title("Error growth away from data")

    fig.suptitle("Spatial accuracy", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 7. fault behaviour
# --------------------------------------------------------------------------- #


def plot_fault_section(
    models: Sequence,
    ds: GWDataset,
    pred: Prediction,
    truth: Optional[Dict[str, np.ndarray]],
    report: Dict,
    path: pathlib.Path,
) -> Optional[str]:
    if ds.fault_lines is None or len(ds.fault_lines) == 0:
        return None

    from gwpinn.postproc.predict import predict_points

    n_f = len(ds.fault_lines)
    fig, axes = plt.subplots(1, n_f + 1, figsize=(4.6 * (n_f + 1), 3.9), squeeze=False)
    axes = axes[0]

    for i, line in enumerate(ds.fault_lines.lines):
        ax = axes[i]
        mid = len(line) // 2
        p0 = line[max(mid - 1, 0)]
        p1 = line[min(mid + 1, len(line) - 1)]
        tang = p1 - p0
        tang = tang / max(np.hypot(*tang), 1e-9)
        nrm = np.array([-tang[1], tang[0]])
        centre = line[mid]

        span = 6.0 * ds.cfg.physics.fault_width
        s = np.linspace(-span, span, 160)
        xy = centre[None, :] + s[:, None] * nrm[None, :]
        inside = ds.domain.contains(xy[:, 0], xy[:, 1])
        xy, s = xy[inside], s[inside]
        if len(xy) < 10:
            ax.axis("off")
            continue

        ph = predict_points(models, ds, xy)["head"][:, 0]
        ax.plot(s, ph, color=C_MODEL, lw=2.0, label="PINN")

        if truth and "head" in truth:
            from gwpinn.io.raster import Raster

            r = Raster(truth["head"][0], ds.domain.template.transform,
                       ds.domain.template.crs)
            th = r.sample(xy[:, 0], xy[:, 1])
            ax.plot(s, th, color=C_REF, lw=1.6, ls="--", label="reference")

        ax.axvline(0, color="#D62728", lw=1.4, alpha=0.8, label="fault trace")
        ax.axvspan(-ds.cfg.physics.fault_width, ds.cfg.physics.fault_width,
                   color="#D62728", alpha=0.08)
        fp = report.get("parameters", {}).get("fault_permeability", {})
        alpha = fp.get("mean", [np.nan] * n_f)[i] if fp else float("nan")
        kind = "trainable" if (fp and fp["trainable"][i]) else "fixed"
        ax.set_xlabel("distance across fault (m)")
        ax.set_ylabel("head, layer 0 (m a.s.l.)")
        ax.set_title(f"Fault {i} — fitted permeability {alpha:.4f} ({kind})")
        ax.legend(fontsize=8)

    ax = axes[-1]
    fp = report.get("parameters", {}).get("fault_permeability")
    if fp:
        idx = np.arange(len(fp["mean"]))
        cols = [C_MODEL if t else "#999999" for t in fp["trainable"]]
        ax.bar(idx, fp["mean"], yerr=fp.get("std"), color=cols, edgecolor="white",
               width=0.6, capsize=4)
        ax.set_yscale("log")
        ax.set_xticks(idx)
        ax.set_xticklabels([f"fault {i}" for i in idx])
        ax.set_ylabel("permeability multiplier α (−)")
        ax.set_title("Fitted fault permeability")
        ax.legend(handles=[
            Line2D([], [], color=C_MODEL, lw=6, label="trainable"),
            Line2D([], [], color="#999999", lw=6, label="fixed (perm = 0 flag)"),
        ], fontsize=8)
    else:
        ax.axis("off")

    fig.suptitle("Flow barriers", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 8./9. water table and uncertainty
# --------------------------------------------------------------------------- #


def plot_water_table(pred: Prediction, ds: GWDataset, path: pathlib.Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    im = _imshow(ax, pred.extra["top"], ds, cmap="terrain")
    _overlay(ax, ds)
    ax.set_title("Ground surface (DTM)")
    _cb(fig, im, ax, "m a.s.l.")

    ax = axes[1]
    h = pred.mean["head"][0]
    im = _imshow(ax, h, ds, cmap=SEQ_HEAD)
    xx, yy = ds.domain.template.meshgrid()
    with np.errstate(invalid="ignore"):
        cs = ax.contour(xx, yy, h, levels=14, colors="black", linewidths=0.5,
                        alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    _overlay(ax, ds, wells=True)
    ax.set_title("Fitted water table, layer 0")
    _cb(fig, im, ax, "head (m a.s.l.)")

    ax = axes[2]
    d = pred.extra["depth_to_water"]
    im = _imshow(ax, d, ds, cmap="YlGnBu")
    _overlay(ax, ds)
    frac = float(np.nanmean(d > 0))
    ax.set_title(f"Depth to water table\n({100 * frac:.0f}% of cells below ground)")
    _cb(fig, im, ax, "m below ground")

    fig.suptitle("Groundwater table", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_uncertainty(pred: Prediction, ds: GWDataset, path: pathlib.Path) -> Optional[str]:
    if not pred.std:
        return None
    n = ds.n_layers
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 7.4), squeeze=False)
    for l in range(n):
        ax = axes[0][l]
        im = _imshow(ax, pred.std["head"][l], ds, cmap="YlOrRd")
        _overlay(ax, ds, wells=True)
        ax.set_title(f"Head uncertainty — layer {l}")
        _cb(fig, im, ax, "ensemble s.d. (m)")

        ax = axes[1][l]
        im = _imshow(ax, pred.std["log10T"][l], ds, cmap="YlOrRd")
        _overlay(ax, ds)
        ax.set_title(f"log10 T uncertainty — layer {l}")
        _cb(fig, im, ax, "ensemble s.d. (log10)")
    fig.suptitle(
        f"Ensemble spread over {pred.n_members} members "
        "(high where neither data nor physics constrains the field)",
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 10. inputs overview
# --------------------------------------------------------------------------- #


def plot_inputs(ds: GWDataset, path: pathlib.Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    ax = axes[0]
    im = _imshow(ax, ds.layers.top.values, ds, cmap="terrain")
    _overlay(ax, ds, wells=True)
    if ds.prop_obs is not None:
        ax.scatter(ds.prop_obs.x, ds.prop_obs.y, s=32, marker="^", c="#EFB118",
                   edgecolors="black", linewidths=0.4, zorder=7)
    handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="head wells"),
        Line2D([], [], marker="^", ls="", mfc="#EFB118", mec="black", label="pumping tests"),
        Line2D([], [], color="#D62728", lw=1.5, label="fault (impermeable)"),
        Line2D([], [], color="#D62728", lw=1.5, ls="--", label="fault (trainable)"),
        Line2D([], [], color="#7EC8E3", lw=5, label="river"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower left")
    ax.set_title("Inputs on the DTM")
    _cb(fig, im, ax, "elevation (m a.s.l.)")

    ax = axes[1]
    th = ds.layers.thickness(0)
    im = _imshow(ax, th, ds, cmap="BuPu")
    _overlay(ax, ds)
    ax.set_title("Layer 0 geometric thickness\n(after overlap correction)")
    _cb(fig, im, ax, "m")

    ax = axes[2]
    if ds.river is not None:
        ax.plot(ds.river.s_gauge, ds.river.stage_gauge, "o", color=C_VAL, ms=7,
                mec="black", mew=0.5, label="gauge observations", zorder=5)
        ss = np.linspace(ds.river.s_gauge.min(), ds.river.s_gauge.max(), 300)
        ax.plot(ss, ds.river.interpolator(ss), color=C_MODEL, lw=2,
                label="along-stream interpolation")
        ax.set_xlabel("distance downstream along centerline (m)")
        ax.set_ylabel("river stage (m a.s.l.)")
        ax.legend(fontsize=8)
        ax.set_title("River stage profile")
    else:
        ax.axis("off")

    fig.suptitle("Model inputs", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #


def plot_fault_steps(report: Dict, path: pathlib.Path) -> Optional[str]:
    """Head held up across each fault, sampled along the whole trace.

    The single transect in ``plot_fault_section`` can land on an unrepresentative
    stretch. This walks the trace and plots the paired head difference at every
    station, which is the quantity a barrier exists to produce.
    """
    faults = report.get("faults", {}).get("faults")
    if not faults:
        return None

    panels = [
        (f, rec)
        for f in faults
        for rec in f["by_offset"]
        if rec.get("_profile")
    ]
    if not panels:
        return None

    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.6 * len(panels), 3.8), squeeze=False
    )
    for ax, (f, rec) in zip(axes[0], panels):
        prof = rec["_profile"]
        s = np.asarray(prof["s"])
        ax.plot(s, prof["reference"], color=C_REF, lw=1.8, ls="--",
                label="reference")
        ax.plot(s, prof["predicted"], color=C_MODEL, lw=2.0, label="PINN")
        ax.axhline(0.0, color="#999999", lw=0.8)
        flag = f.get("perm_flag")
        kind = ("impermeable" if flag == 0 else
                "trainable" if flag == 1 else f"perm={flag}")
        frac = rec.get("recovered_fraction", float("nan"))
        ax.set_xlabel("position along fault trace (normalised)")
        ax.set_ylabel("head difference across trace (m)")
        ax.set_title(
            f"Fault {f['fault']} ({kind}), ±{rec['offset_m']:.0f} m\n"
            f"{100 * frac:.0f}% of the reference step recovered"
        )
        ax.legend(fontsize=8)

    fig.suptitle("Head held up by each barrier", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def make_all_plots(
    models: Sequence,
    ds: GWDataset,
    pred: Prediction,
    histories: List[List[dict]],
    report: Dict,
    truth: Optional[Dict[str, np.ndarray]],
    outdir: str | pathlib.Path,
) -> List[str]:
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: List[Optional[str]] = []

    paths.append(plot_inputs(ds, outdir / "01_inputs.png"))
    paths.append(plot_training(histories, outdir / "02_training.png"))
    paths.append(plot_head_scatter(report, outdir / "03_goodness_of_fit.png"))
    paths.append(plot_water_table(pred, ds, outdir / "04_water_table.png"))
    paths.append(
        plot_field_maps(pred, ds, truth, "head", "head", "Groundwater head (m a.s.l.)",
                        outdir / "05_head_maps.png", cmap=SEQ_HEAD)
    )
    paths.append(
        plot_field_maps(pred, ds, truth, "log10T", "T",
                        "log10 transmissivity (m²/d)",
                        outdir / "06_transmissivity_maps.png", cmap=SEQ_PROP,
                        log_truth=True)
    )
    paths.append(
        plot_field_maps(pred, ds, truth, "log10S", "S",
                        "log10 storage coefficient (−)",
                        outdir / "07_storage_maps.png", cmap=SEQ_PROP, log_truth=True)
    )
    paths.append(plot_variograms(report, ds, outdir / "08_variograms.png"))
    paths.append(plot_residual_space(report, ds, outdir / "09_spatial_accuracy.png"))
    paths.append(
        plot_fault_section(models, ds, pred, truth, report, outdir / "10_faults.png")
    )
    paths.append(plot_fault_steps(report, outdir / "10b_fault_steps.png"))
    paths.append(plot_uncertainty(pred, ds, outdir / "11_uncertainty.png"))

    return [p for p in paths if p]


__all__ = ["make_all_plots"]
