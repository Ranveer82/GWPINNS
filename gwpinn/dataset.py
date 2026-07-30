"""Assemble every input into the tensors the trainer consumes.

This module owns all the messy real-world handling - missing attributes, mixed
resolutions, layer surfaces that cross - so that the trainer only ever sees
clean, consistent batches.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from gwpinn.config import Config
from gwpinn.geo.faults import FaultField
from gwpinn.geo.grid import ModelDomain, build_domain
from gwpinn.geo.river import RiverStage, build_river_stage
from gwpinn.io.layers import LayerGeometry, build_layer_geometry
from gwpinn.io.raster import Raster, read_raster
from gwpinn.io.vector import (
    PointSet,
    PolylineSet,
    read_points,
    read_polygon,
    read_polylines,
    simplify_polylines,
)
from gwpinn.models.fields import Normalizer
from gwpinn.physics.bcs import BoundaryConditions, assign_boundary_conditions
from gwpinn.physics.gwflow import LayerElevations, Sources, estimate_residual_scale
from gwpinn.stats.variogram import VariogramModel, fit_variogram_by_layer


@dataclass
class Batch:
    """One batch of collocation points with everything the residual needs."""

    xy: torch.Tensor
    elev: LayerElevations
    sources: Sources
    t: Optional[torch.Tensor] = None


@dataclass
class GWDataset:
    """Everything the model needs, in one place."""

    cfg: Config
    domain: ModelDomain
    layers: LayerGeometry
    normalizer: Normalizer

    head_obs: PointSet
    prop_obs: Optional[PointSet] = None
    river: Optional[RiverStage] = None
    faults: Optional[FaultField] = None
    fault_lines: Optional[PolylineSet] = None
    boundary_lines: Optional[PolylineSet] = None
    recharge_raster: Optional[Raster] = None

    #: ``variograms[property][layer]`` - fitted separately per aquifer layer.
    variograms: Dict[str, Dict[int, VariogramModel]] = field(default_factory=dict)
    variogram_diag: Dict[str, Dict[int, dict]] = field(default_factory=dict)

    train_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    val_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    residual_scale: float = 1.0
    flux_scale: float = 1.0
    t_ref: float = 1.0

    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device("cpu")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @property
    def n_layers(self) -> int:
        return self.layers.n_layers

    @property
    def transient(self) -> bool:
        return self.cfg.physics.regime == "transient"

    def _tt(self, arr) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arr), dtype=self.dtype, device=self.device)

    def fault_features(self, xy: torch.Tensor) -> Optional[torch.Tensor]:
        if self.faults is None or not self.faults.has_faults:
            return None
        return self.faults.side_features(xy)

    @property
    def n_fault_feats(self) -> int:
        if self.faults is None or not self.faults.has_faults:
            return 0
        return self.faults.n_faults

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def _elevations(self, x: np.ndarray, y: np.ndarray) -> LayerElevations:
        top = self.layers.sample_top(x, y)
        bots = np.column_stack(
            [self.layers.sample_bottom(l, x, y) for l in range(self.n_layers)]
        )
        # Points can land just outside the DTM footprint; fall back to a nominal
        # stack rather than propagating NaN into the residual.
        bad = ~np.isfinite(top)
        if bad.any():
            top = np.where(bad, np.nanmedian(self.layers.top.values), top)
        for l in range(self.n_layers):
            col = bots[:, l]
            m = ~np.isfinite(col)
            if m.any():
                ref = top - (l + 1) * max(self.cfg.domain.min_thickness, 1.0)
                bots[m, l] = ref[m]
        # Re-assert monotonicity after the fallback fill.
        upper = top
        for l in range(self.n_layers):
            bots[:, l] = np.minimum(bots[:, l], upper - self.cfg.domain.min_thickness)
            upper = bots[:, l]

        return LayerElevations(top=self._tt(top), bottoms=self._tt(bots))

    def _sources(self, x: np.ndarray, y: np.ndarray) -> Sources:
        n = len(x)
        if self.recharge_raster is not None:
            rch = self.recharge_raster.sample(x, y)
            rch = np.where(np.isfinite(rch), rch, self.cfg.physics.recharge)
        else:
            rch = np.full(n, self.cfg.physics.recharge)

        river_mask = np.zeros(n)
        river_stage = np.zeros(n)
        if self.river is not None:
            inside = self.river.inside(x, y)
            if inside.any():
                river_mask[inside] = 1.0
                river_stage[inside] = self.river.stage_at(x[inside], y[inside])

        return Sources(
            recharge=self._tt(rch),
            river_stage=self._tt(river_stage),
            river_mask=self._tt(river_mask),
        )

    def make_batch(
        self,
        xy: np.ndarray,
        times: Optional[np.ndarray] = None,
        requires_grad: bool = True,
        with_sources: bool = True,
    ) -> Batch:
        """Build a batch. ``with_sources=False`` skips the recharge and river
        lookups, which only the PDE residual needs - worth it for the geostatistical
        batches, which are rebuilt every iteration."""
        x, y = xy[:, 0], xy[:, 1]
        xyt = self._tt(xy).requires_grad_(requires_grad)
        t = None
        if self.transient:
            tv = np.zeros(len(xy)) if times is None else np.asarray(times, dtype=float)
            t = self._tt(tv.reshape(-1, 1)).requires_grad_(requires_grad)
        sources = self._sources(x, y) if with_sources else Sources()
        return Batch(xy=xyt, elev=self._elevations(x, y), sources=sources, t=t)

    def sample_collocation(
        self, n: int, rng: np.random.Generator, n_fault: int = 0
    ) -> Batch:
        xy = self.domain.sample_interior(n, rng)
        near_fault = self.sample_fault_zone(n_fault, rng)
        if near_fault is not None and len(near_fault):
            xy = np.vstack([xy, near_fault])
        times = None
        if self.transient:
            tt = np.asarray(self.cfg.physics.times or [0.0], dtype=float)
            times = rng.choice(tt, size=len(xy))
        return self.make_batch(xy, times)

    def sample_river(self, n: int, rng: np.random.Generator):
        if self.river is None or n <= 0:
            return None, None
        pts, stage = self.river.sample_river(n, rng)
        if len(pts) == 0:
            return None, None
        # These are collocation points, not data points: the flow equation is
        # enforced on them so the river exchange term is actually sampled.
        # Uniform interior sampling almost never lands in a narrow channel.
        return self.make_batch(pts, requires_grad=True), self._tt(stage)

    def sample_fault_zone(self, n: int, rng: np.random.Generator) -> Optional[np.ndarray]:
        """Points concentrated in the narrow zone where a fault bends the flow.

        A barrier is a few tens of metres wide across a trace kilometres long, so
        uniform collocation lands almost nothing inside it - for the demo case,
        about 0.8% of points - and the residual is effectively never evaluated
        where the barrier acts. The head drop across the fault is then set by the
        data alone and comes out far too small. These extra points put the
        equation back where the interesting physics is.
        """
        if self.fault_lines is None or len(self.fault_lines) == 0 or n <= 0:
            return None
        segs = self.fault_lines.segments()
        if len(segs) == 0:
            return None

        a = segs[:, :2]
        b = segs[:, 2:]
        d = b - a
        length = np.hypot(d[:, 0], d[:, 1])
        if length.sum() <= 0:
            return None

        width = max(self.cfg.physics.fault_width, 1e-6)
        out = np.zeros((0, 2))
        for _ in range(8):
            k = rng.choice(len(segs), size=3 * n, p=length / length.sum())
            t = rng.uniform(0.0, 1.0, len(k))[:, None]
            on = a[k] + d[k] * t
            nrm = np.column_stack([-d[k, 1], d[k, 0]]) / np.maximum(length[k], 1e-12)[:, None]
            # Spread over a couple of barrier widths so the residual sees the
            # whole transition, not just its centre.
            off = rng.normal(0.0, 1.5 * width, len(k))[:, None]
            cand = on + nrm * off
            out = np.vstack([out, cand[self.domain.contains(cand[:, 0], cand[:, 1])]])
            if len(out) >= n:
                break
        return out[:n] if len(out) else None

    def sample_boundary(self, n: int, rng: np.random.Generator):
        xy, nrm = self.domain.sample_boundary(n, rng)
        if len(xy) == 0:
            return None, None, None
        bc = assign_boundary_conditions(
            xy,
            None if self.boundary_lines is None else self.boundary_lines.lines,
            None if self.boundary_lines is None else self.boundary_lines.attrs.get("bctype"),
            None if self.boundary_lines is None else self.boundary_lines.attrs.get("value"),
            None if self.boundary_lines is None else self.boundary_lines.attrs.get("cond"),
            snap_distance=3.0 * self.domain.template.cellsize,
        )
        batch = self.make_batch(xy, requires_grad=True)
        return batch, self._tt(nrm), bc

    # ------------------------------------------------------------------ #

    def head_observation_tensors(self, idx: np.ndarray):
        """Batch + targets for a subset of the head observations."""
        pts = self.head_obs.subset(np.isin(np.arange(len(self.head_obs)), idx))
        times = pts.get("time", 0.0) if self.transient else None
        batch = self.make_batch(pts.xy(), times, requires_grad=False)
        return (
            batch,
            self._tt(pts.attrs["head"]),
            torch.as_tensor(
                np.asarray(pts.get("layer", 0.0), dtype=int), device=self.device
            ),
            self._tt(pts.get("weight", 1.0)),
        )

    def property_observation_tensors(self):
        if self.prop_obs is None or len(self.prop_obs) == 0:
            return None
        pts = self.prop_obs
        batch = self.make_batch(pts.xy(), requires_grad=False)
        out = {
            "batch": batch,
            "layer": torch.as_tensor(
                np.asarray(pts.get("layer", 0.0), dtype=int), device=self.device
            ),
        }
        for name in ("T", "S"):
            if name in pts.attrs:
                v = np.asarray(pts.attrs[name], dtype=float)
                out[f"log10{name}"] = self._tt(np.log10(np.clip(v, 1e-12, None)))
                out[f"{name}_valid"] = torch.as_tensor(
                    np.isfinite(v) & (v > 0), device=self.device
                )
        return out


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def build_dataset(
    cfg: Config,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> GWDataset:
    """Read every configured input and assemble a :class:`GWDataset`."""
    p = cfg.paths
    log = print if verbose else (lambda *a, **k: None)

    # ---- rasters ---------------------------------------------------------
    if not p.dtm:
        raise ValueError("paths.dtm is required (top elevation of layer 0)")
    dtm = read_raster(p.dtm)
    bottoms = [read_raster(b) for b in p.layer_bottoms]
    if not bottoms:
        raise ValueError("paths.layer_bottoms must list at least one raster")

    domain_poly = None
    if p.domain:
        domain_poly, _ = read_polygon(p.domain)

    domain = build_domain(
        dtm=dtm,
        domain_polygon=domain_poly,
        cellsize=cfg.domain.cellsize,
        use_dtm_footprint=cfg.domain.use_dtm_footprint,
    )
    layers = build_layer_geometry(
        dtm, bottoms, grid=domain.template, min_thickness=cfg.domain.min_thickness
    )
    log(layers.summary())

    recharge_raster = read_raster(p.recharge) if p.recharge else None

    # ---- observations ----------------------------------------------------
    if not p.head_obs:
        raise ValueError("paths.head_obs is required")
    head_obs = read_points(p.head_obs, ("head", "layer", "time", "weight"))
    if "head" not in head_obs.attrs:
        raise ValueError(f"{p.head_obs} has no recognisable head attribute")

    keep = np.isfinite(head_obs.attrs["head"]) & domain.contains(head_obs.x, head_obs.y)
    dropped = int((~keep).sum())
    head_obs = head_obs.subset(keep)
    if dropped:
        log(f"  dropped {dropped} head observations (outside domain or no value)")
    if len(head_obs) == 0:
        raise ValueError("no usable head observations inside the domain")

    prop_obs = None
    if p.prop_obs:
        prop_obs = read_points(p.prop_obs, ("T", "S", "layer"))
        keep = domain.contains(prop_obs.x, prop_obs.y)
        prop_obs = prop_obs.subset(keep)
        log(f"  {len(prop_obs)} aquifer-property points inside the domain")

    # ---- river -----------------------------------------------------------
    river = None
    if p.river_polygon and p.gauge_obs:
        poly, _ = read_polygon(p.river_polygon)
        gauges = read_points(p.gauge_obs, ("stage", "time"))
        if "stage" not in gauges.attrs:
            raise ValueError(f"{p.gauge_obs} has no recognisable stage attribute")
        centerline = None
        if p.river_centerline:
            cl = read_polylines(p.river_centerline)
            if len(cl):
                centerline = max(cl.lines, key=len)
        river = build_river_stage(
            poly,
            gauges.xy(),
            gauges.attrs["stage"],
            centerline=centerline,
            rng=np.random.default_rng(cfg.train.seed),
        )
        log(
            f"  river: {len(gauges)} gauges, stage "
            f"{river.stage_gauge.min():.2f}-{river.stage_gauge.max():.2f} m over "
            f"{river.s_gauge.max() - river.s_gauge.min():.0f} m of channel"
        )

    # ---- faults ----------------------------------------------------------
    fault_lines, faults = None, None
    if p.faults:
        fault_lines = read_polylines(p.faults, ("perm",))
        if len(fault_lines):
            # Light simplification only: the segment count drives the cost of
            # every fault-distance evaluation, but over-simplifying a curved
            # trace moves the barrier away from where it actually is.
            fault_lines = simplify_polylines(fault_lines, 0.1 * cfg.physics.fault_width)
            perm = fault_lines.attrs.get("perm")
            faults = FaultField(
                fault_lines.lines,
                perm_flags=perm,
                width=cfg.physics.fault_width,
                perm_min=cfg.physics.fault_perm_min,
                dtype=dtype,
            ).to(device)
            n_train = int(np.sum(np.isclose(perm, 1.0))) if perm is not None else len(fault_lines)
            log(
                f"  faults: {len(fault_lines)} traces "
                f"({faults.a.shape[0]} segments), {n_train} with trainable permeability"
            )

    boundary_lines = None
    if p.boundary:
        boundary_lines = read_polylines(p.boundary, ("bctype", "value", "cond"))
        log(f"  boundary: {len(boundary_lines)} segments with prescribed conditions")

    # ---- normalisation and scales ---------------------------------------
    t_max = 1.0
    if cfg.physics.regime == "transient":
        tt = list(cfg.physics.times) or list(np.unique(head_obs.get("time", 0.0)))
        t_max = float(max(tt) if tt else 1.0)
        cfg.physics.times = [float(v) for v in (cfg.physics.times or tt)]

    normalizer = Normalizer.from_bounds(
        domain.bounds, head_obs.attrs["head"], t_max=t_max
    )

    t_ref = _reference_transmissivity(prop_obs, layers, cfg)
    residual_scale = estimate_residual_scale(
        t_ref, normalizer.h_std, 0.5 * domain.diagonal, cfg.physics.recharge
    )
    flux_scale = float(t_ref * normalizer.h_std / max(0.5 * domain.diagonal, 1e-9))

    # ---- variograms (one per property per layer) -------------------------
    variograms: Dict[str, Dict[int, VariogramModel]] = {}
    diag: Dict[str, Dict[int, dict]] = {}
    if prop_obs is not None and len(prop_obs) >= 5:
        n_layers = layers.n_layers
        max_lag = cfg.variogram.max_lag_fraction * domain.diagonal
        obs_layer = np.asarray(prop_obs.get("layer", 0.0), dtype=int)
        for name in ("T", "S"):
            if name not in prop_obs.attrs:
                continue
            v = np.asarray(prop_obs.attrs[name], dtype=float)
            ok = np.isfinite(v) & (v > 0)
            if ok.sum() < 5:
                continue
            y = np.full(len(v), np.nan)
            y[ok] = np.log10(v[ok]) if cfg.variogram.log_transform else v[ok]
            models, diags = fit_variogram_by_layer(
                prop_obs.xy(),
                y,
                obs_layer,
                n_layers,
                model=cfg.variogram.model,
                n_lags=cfg.variogram.n_lags,
                max_lag=max_lag,
                nugget=cfg.variogram.nugget,
                sill=cfg.variogram.sill,
                range_=cfg.variogram.range_,
            )
            if models:
                variograms[name] = models
                diag[name] = diags
            for l, mdl in models.items():
                log(
                    f"  variogram log10({name}) layer {l}: {mdl} "
                    f"[{diags[l].get('source', '')}]"
                )

    # ---- train / validation split ---------------------------------------
    rng = np.random.default_rng(cfg.train.seed)
    n = len(head_obs)
    perm_idx = rng.permutation(n)
    n_val = int(round(cfg.train.val_fraction * n))
    val_idx = np.sort(perm_idx[:n_val])
    train_idx = np.sort(perm_idx[n_val:])
    log(f"  head observations: {len(train_idx)} train / {len(val_idx)} validation")

    return GWDataset(
        cfg=cfg,
        domain=domain,
        layers=layers,
        normalizer=normalizer,
        head_obs=head_obs,
        prop_obs=prop_obs,
        river=river,
        faults=faults,
        fault_lines=fault_lines,
        boundary_lines=boundary_lines,
        recharge_raster=recharge_raster,
        variograms=variograms,
        variogram_diag=diag,
        train_idx=train_idx,
        val_idx=val_idx,
        residual_scale=residual_scale,
        flux_scale=flux_scale,
        t_ref=t_ref,
        dtype=dtype,
        device=torch.device(device),
    )


def _reference_transmissivity(
    prop_obs: Optional[PointSet], layers: LayerGeometry, cfg: Config
) -> float:
    """Characteristic transmissivity, used only to non-dimensionalise."""
    if prop_obs is not None and "T" in prop_obs.attrs:
        v = np.asarray(prop_obs.attrs["T"], dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            return float(np.exp(np.log(v).mean()))  # geometric mean
    th = layers.thickness(0)
    th = th[np.isfinite(th)]
    mean_b = float(th.mean()) if th.size else 10.0
    k_geom = float(np.sqrt(cfg.physics.k_min * cfg.physics.k_max))
    return float(max(k_geom * mean_b, 1e-6))


__all__ = ["GWDataset", "Batch", "build_dataset"]
