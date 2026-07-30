"""Evaluate a trained model on the output grid and write rasters."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from gwpinn.dataset import GWDataset
from gwpinn.io.raster import write_raster

_FIELDS = ("head", "K", "T", "S", "log10K", "log10T", "log10S", "b")


@dataclass
class Prediction:
    """Gridded model output.

    Every entry is ``(n_layers, ny, nx)`` with NaN outside the active domain.
    ``std`` holds the between-member standard deviation when an ensemble was
    trained, which is the model's own statement of where it is unconstrained.
    """

    mean: Dict[str, np.ndarray] = field(default_factory=dict)
    std: Dict[str, np.ndarray] = field(default_factory=dict)
    extra: Dict[str, np.ndarray] = field(default_factory=dict)
    n_members: int = 1

    def layer(self, name: str, l: int) -> np.ndarray:
        return self.mean[name][l]


# --------------------------------------------------------------------------- #


@torch.no_grad()
def _evaluate(
    model, ds: GWDataset, xy: np.ndarray, time: Optional[float], chunk: int
) -> Dict[str, np.ndarray]:
    out: Dict[str, List[np.ndarray]] = {k: [] for k in _FIELDS}
    for i in range(0, len(xy), chunk):
        sub = xy[i : i + chunk]
        times = None if time is None else np.full(len(sub), float(time))
        batch = ds.make_batch(sub, times, requires_grad=False)
        f = model.fields(batch)
        out["head"].append(f["h"].cpu().numpy())
        for k in ("K", "T", "S", "log10K", "log10T", "log10S", "b"):
            out[k].append(f[k].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items() if v}


def predict_points(
    models: Sequence,
    ds: GWDataset,
    xy: np.ndarray,
    time: Optional[float] = None,
    chunk: int = 8192,
) -> Dict[str, np.ndarray]:
    """Ensemble-mean prediction at arbitrary points; shapes ``(n_points, n_layers)``."""
    stacks = [_evaluate(m, ds, np.asarray(xy, dtype=float), time, chunk) for m in models]
    return {k: np.mean([s[k] for s in stacks], axis=0) for k in stacks[0]}


def predict_grid(
    models: Sequence,
    ds: GWDataset,
    time: Optional[float] = None,
    downsample: int = 1,
    chunk: int = 8192,
    verbose: bool = True,
) -> Prediction:
    """Evaluate the model(s) on the output raster grid."""
    template = ds.domain.template
    active = ds.domain.active
    ny, nx = template.shape

    if downsample > 1:
        mask = np.zeros_like(active)
        mask[::downsample, ::downsample] = True
        active = active & mask

    xx, yy = template.meshgrid()
    idx = np.nonzero(active)
    xy = np.column_stack([xx[idx], yy[idx]])
    if verbose:
        print(f"  predicting on {len(xy)} active cells x {len(models)} member(s)")

    stacks = [_evaluate(m, ds, xy, time, chunk) for m in models]

    pred = Prediction(n_members=len(models))
    for key in stacks[0]:
        arr = np.stack([s[key] for s in stacks])          # (M, N, L)
        mean = arr.mean(axis=0)
        pred.mean[key] = _scatter(mean, idx, (ny, nx))
        if len(models) > 1:
            pred.std[key] = _scatter(arr.std(axis=0), idx, (ny, nx))

    # Depth to the water table: the form most field reports actually use.
    top = template.copy_like(ds.layers.top.values).values
    pred.extra["depth_to_water"] = top - pred.mean["head"][0]
    pred.extra["top"] = top
    return pred


def _scatter(values: np.ndarray, idx, shape) -> np.ndarray:
    """Place ``(N, L)`` point values back onto an ``(L, ny, nx)`` grid."""
    n_layers = values.shape[1]
    out = np.full((n_layers, *shape), np.nan)
    for l in range(n_layers):
        out[l][idx] = values[:, l]
    return out


# --------------------------------------------------------------------------- #


def export_rasters(
    pred: Prediction,
    ds: GWDataset,
    outdir: str | pathlib.Path,
    nodata: float = -9999.0,
    dtype: str = "float32",
    verbose: bool = True,
) -> List[str]:
    """Write the fitted head and aquifer-property rasters, layer by layer."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tf, crs = ds.domain.template.transform, ds.domain.template.crs
    written: List[str] = []

    def w(name: str, arr: np.ndarray) -> None:
        path = outdir / f"{name}.tif"
        write_raster(path, arr, tf, crs, nodata=nodata, dtype=dtype)
        written.append(str(path))

    n_layers = ds.n_layers
    for l in range(n_layers):
        w(f"head_L{l}", pred.mean["head"][l])
        w(f"transmissivity_L{l}", pred.mean["T"][l])
        w(f"conductivity_L{l}", pred.mean["K"][l])
        w(f"storage_L{l}", pred.mean["S"][l])
        w(f"saturated_thickness_L{l}", pred.mean["b"][l])
        if pred.std:
            w(f"head_std_L{l}", pred.std["head"][l])
            w(f"log10T_std_L{l}", pred.std["log10T"][l])

    w("depth_to_water", pred.extra["depth_to_water"])

    if verbose:
        print(f"  wrote {len(written)} rasters to {outdir}")
    return written


__all__ = ["Prediction", "predict_grid", "predict_points", "export_rasters"]
