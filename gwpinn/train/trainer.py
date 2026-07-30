"""The PINN model and its training loop."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from gwpinn.config import Config
from gwpinn.dataset import Batch, GWDataset
from gwpinn.models.fields import HeadField, PropertyField
from gwpinn.physics.bcs import boundary_residual
from gwpinn.physics.gwflow import GroundwaterFlow
from gwpinn.train.losses import (
    KrigingPrior,
    kriging_loss,
    sample_lag_pairs,
    smoothness_loss,
    variogram_loss,
    weighted_mse,
)

_VARIOGRAM_BINS = 8


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class PINN(nn.Module):
    """Head network + property network + the flow operator that couples them."""

    def __init__(self, ds: GWDataset, cfg: Config, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(int(seed))
        torch.manual_seed(int(seed))

        dtype = ds.dtype
        m, ph = cfg.model, cfg.physics
        n_ff = ds.n_fault_feats if m.fault_features else 0

        self.head = HeadField(
            n_layers=ds.n_layers,
            normalizer=ds.normalizer,
            arch=m.arch,
            width=m.width,
            depth=m.depth,
            activation=m.activation,
            n_fourier=m.fourier_features,
            fourier_sigma=m.fourier_sigma,
            n_fault_feats=n_ff,
            fault_coords=m.fault_coords,
            fault_sigma_scale=m.fault_sigma_scale,
            transient=ds.transient,
            dtype=dtype,
            cnn_latent=m.cnn_latent,
            cnn_channels=m.cnn_channels,
            generator=gen,
        )
        self.prop = PropertyField(
            n_layers=ds.n_layers,
            normalizer=ds.normalizer,
            arch=m.arch,
            width=m.prop_width,
            depth=m.prop_depth,
            activation=m.activation,
            n_fourier=m.fourier_features,
            fourier_sigma=m.prop_fourier_sigma,
            n_fault_feats=n_ff,
            fault_coords=m.fault_coords,
            fault_sigma_scale=m.fault_sigma_scale,
            k_bounds=(ph.k_min, ph.k_max),
            s_bounds=(ph.s_min, ph.s_max),
            dtype=dtype,
            cnn_latent=m.cnn_latent,
            cnn_channels=m.cnn_channels,
            generator=gen,
        )

        # Each ensemble member needs its own fault permeabilities.
        self.faults = copy.deepcopy(ds.faults) if ds.faults is not None else None
        self.use_fault_feats = bool(m.fault_features) and n_ff > 0

        self.flow = GroundwaterFlow(
            n_layers=ds.n_layers,
            faults=self.faults,
            leakance=ph.leakance,
            leakance_default=ph.leakance_default,
            train_leakance=ph.train_leakance,
            river_conductance=ph.river_conductance,
            train_river_conductance=ph.train_river_conductance,
            unconfined_top=ph.unconfined_top,
            min_thickness=cfg.domain.min_thickness,
            residual_scale=ds.residual_scale,
            dtype=dtype,
        )

        self._set_property_prior(ds)
        self.to(ds.device)

    # ------------------------------------------------------------------ #

    def _set_property_prior(self, ds: GWDataset) -> None:
        """Start the property field at the observed geometric mean per layer."""
        if ds.prop_obs is None:
            return
        layer = np.asarray(ds.prop_obs.get("layer", 0.0), dtype=int)
        k_prior: List[float] = []
        s_prior: List[float] = []
        for l in range(ds.n_layers):
            m = layer == l
            kv, sv = np.nan, np.nan
            if "T" in ds.prop_obs.attrs and m.any():
                tv = np.asarray(ds.prop_obs.attrs["T"], dtype=float)[m]
                tv = tv[np.isfinite(tv) & (tv > 0)]
                if tv.size:
                    th = ds.layers.sample_thickness(l, ds.prop_obs.x[m], ds.prop_obs.y[m])
                    th = np.nanmedian(th[np.isfinite(th)]) if np.isfinite(th).any() else 10.0
                    kv = float(np.log10(np.exp(np.log(tv).mean()) / max(th, 1e-6)))
            if "S" in ds.prop_obs.attrs and m.any():
                svv = np.asarray(ds.prop_obs.attrs["S"], dtype=float)[m]
                svv = svv[np.isfinite(svv) & (svv > 0)]
                if svv.size:
                    sv = float(np.log10(np.exp(np.log(svv).mean())))
            k_prior.append(kv)
            s_prior.append(sv)
        self.prop.set_prior(k_prior, s_prior)

    # ------------------------------------------------------------------ #

    def fault_feats(self, xy: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_fault_feats or self.faults is None:
            return None
        return self.faults.side_features(xy)

    def fields(self, batch: Batch, with_anis: bool = False) -> Dict[str, torch.Tensor]:
        """Head and property predictions at a batch's points."""
        anis = None
        if self.faults is not None and self.faults.has_faults:
            if with_anis:
                anis, side = self.faults.evaluate(batch.xy)
                ff = side if self.use_fault_feats else None
            else:
                ff = self.fault_feats(batch.xy)
        else:
            ff = None

        h = self.head(batch.xy, batch.t, ff)
        props = self.prop(batch.xy, ff)
        T = self.flow.transmissivity(props["K"], h, batch.elev)
        return {
            "h": h,
            "K": props["K"],
            "log10K": props["log10K"],
            "S": props["S"],
            "log10S": props["log10S"],
            "T": T,
            "log10T": torch.log10(T.clamp_min(1e-12)),
            "b": self.flow.saturated_thickness(h, batch.elev),
            "anis": anis,
        }

    def residual(
        self, batch: Batch, fields: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        f = self.fields(batch, with_anis=True) if fields is None else fields
        return self.flow.residual(
            batch.xy,
            f["h"],
            f["K"],
            batch.elev,
            sources=batch.sources,
            S=f["S"] if batch.t is not None else None,
            t=batch.t,
            anis=f.get("anis"),
        )


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


@dataclass
class _Cache:
    """Collocation batches, refreshed periodically.

    The variogram and kriging samples for every (property, layer) combination
    are concatenated into one batch each. Evaluating them separately meant six
    extra forward passes through both networks per iteration, which dominated
    the step time.
    """

    colloc: Optional[Batch] = None
    river: Optional[Batch] = None
    boundary: Optional[tuple] = None
    vario_batch: Optional[Batch] = None
    vario_specs: List[tuple] = field(default_factory=list)
    krig_batch: Optional[Batch] = None
    krig_specs: List[tuple] = field(default_factory=list)


class Trainer:
    """Adam warm-up followed by L-BFGS refinement, with adaptive loss balancing."""

    def __init__(
        self,
        ds: GWDataset,
        cfg: Optional[Config] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.ds = ds
        self.cfg = cfg or ds.cfg
        self.seed = self.cfg.train.seed if seed is None else int(seed)
        self.verbose = verbose

        self.model = PINN(ds, self.cfg, self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.cache = _Cache()
        self.history: List[dict] = []

        # Fixed observation tensors.
        self.obs_train = ds.head_observation_tensors(ds.train_idx)
        self.obs_val = (
            ds.head_observation_tensors(ds.val_idx) if len(ds.val_idx) else None
        )
        self.prop_obs = ds.property_observation_tensors()

        self.weights: Dict[str, float] = {
            k: 1.0
            for k in ("head", "pde", "river", "prop", "variogram", "boundary",
                      "smooth", "kriging")
        }
        self.kriging_priors = self._build_kriging_priors()

    # ------------------------------------------------------------------ #

    def _build_kriging_priors(self) -> Dict[tuple, KrigingPrior]:
        """One kriged prior per (property, layer), keyed ``(name, layer)``."""
        out: Dict[tuple, KrigingPrior] = {}
        if not self.cfg.variogram.use_kriging_prior or self.ds.prop_obs is None:
            return out
        obs_layer = np.asarray(self.ds.prop_obs.get("layer", 0.0), dtype=int)
        for name in ("T", "S"):
            if name not in self.ds.variograms or name not in self.ds.prop_obs.attrs:
                continue
            v = np.asarray(self.ds.prop_obs.attrs[name], dtype=float)
            for layer, model in self.ds.variograms[name].items():
                ok = np.isfinite(v) & (v > 0) & (obs_layer == layer)
                if ok.sum() < 3:
                    continue
                prior = KrigingPrior.build(
                    self.ds.domain,
                    self.ds.prop_obs.xy()[ok],
                    np.log10(v[ok]),
                    model,
                    n_points=2048,
                    rng=np.random.default_rng(self.seed + 17 + layer),
                    name=name,
                )
                if prior is not None:
                    out[(name, layer)] = prior
        return out

    # ------------------------------------------------------------------ #

    def resample(self) -> None:
        """Draw a fresh set of collocation points."""
        tr = self.cfg.train
        self.cache.colloc = self.ds.sample_collocation(
            tr.n_collocation, self.rng, n_fault=tr.n_fault_colloc
        )

        river_batch, _ = self.ds.sample_river(tr.n_river, self.rng)
        self.cache.river = river_batch

        if tr.n_boundary > 0:
            b, n, bc = self.ds.sample_boundary(tr.n_boundary, self.rng)
            self.cache.boundary = (b, n, bc) if b is not None else None
        else:
            self.cache.boundary = None

        self.resample_geostat()

    def resample_geostat(self) -> None:
        """Redraw the variogram pairs and kriging anchors.

        Done **every** iteration, unlike the collocation points. A fixed sample
        of point pairs can be satisfied by nudging the field at those particular
        points instead of by getting the spatial structure right: with the pairs
        held for 100 iterations the loss fell to ~1e-4 on the current sample
        while a fresh sample still scored ~0.2. These batches need no source
        terms and no autograd graph, so redrawing them is cheap.
        """
        tr = self.cfg.train

        # ---- variogram pairs, all combinations in one batch ---------------
        n_combos = max(sum(len(v) for v in self.ds.variograms.values()), 1)
        n_per = max(tr.n_variogram_pairs // n_combos, 96)
        pts: List[np.ndarray] = []
        specs: List[tuple] = []
        off = 0
        for name, per_layer in self.ds.variograms.items():
            for layer, model in per_layer.items():
                p1, p2, lag, bins = sample_lag_pairs(
                    self.ds.domain, model, n_per, self.rng, _VARIOGRAM_BINS
                )
                if len(p1) == 0:
                    continue
                pts += [p1, p2]
                specs.append(
                    (
                        name,
                        layer,
                        off,
                        len(p1),
                        self.ds._tt(lag),
                        torch.as_tensor(bins, dtype=torch.long, device=self.ds.device),
                    )
                )
                off += 2 * len(p1)
        self.cache.vario_specs = specs
        self.cache.vario_batch = (
            self.ds.make_batch(np.vstack(pts), requires_grad=False, with_sources=False)
            if pts else None
        )

        # ---- kriging anchors, likewise batched ---------------------------
        pts, specs, off = [], [], 0
        n_krig = max(1536 // max(len(self.kriging_priors), 1), 128)
        for (name, layer), prior in self.kriging_priors.items():
            xy, est, w = prior.batch(n_krig, self.rng)
            pts.append(xy)
            specs.append((name, layer, off, len(xy), self.ds._tt(est), self.ds._tt(w)))
            off += len(xy)
        self.cache.krig_specs = specs
        self.cache.krig_batch = (
            self.ds.make_batch(np.vstack(pts), requires_grad=False, with_sources=False)
            if pts else None
        )

    # ------------------------------------------------------------------ #

    def compute_losses(self) -> Dict[str, torch.Tensor]:
        ds, model = self.ds, self.model
        h_std = ds.normalizer.h_std
        zero = torch.zeros((), dtype=ds.dtype, device=ds.device)
        terms: Dict[str, torch.Tensor] = {}

        # -- observed heads ------------------------------------------------
        batch, target, layer, w = self.obs_train
        ff = model.fault_feats(batch.xy)
        h_obs = model.head(batch.xy, batch.t, ff)
        pred = h_obs.gather(1, layer.clamp(0, ds.n_layers - 1)[:, None])[:, 0]
        terms["head"] = weighted_mse(
            pred / h_std, target / h_std, w,
            noise=self.cfg.train.head_noise / h_std,
        )

        # -- flow equation -------------------------------------------------
        cb = self.cache.colloc
        fc = model.fields(cb, with_anis=True)
        terms["pde"] = (model.residual(cb, fc) ** 2).mean()

        terms["river"] = (
            (model.residual(self.cache.river) ** 2).mean()
            if self.cache.river is not None
            else zero
        )

        # -- outer boundary ------------------------------------------------
        if self.cache.boundary is not None:
            bb, nrm, bc = self.cache.boundary
            fb = model.fields(bb, with_anis=True)
            parts = boundary_residual(
                bb.xy, nrm, fb["h"], fb["T"], bc, anis=fb.get("anis"),
                flux_scale=ds.flux_scale, head_scale=h_std,
            )
            terms["boundary"] = (
                sum((v**2).mean() for v in parts.values()) if parts else zero
            )
        else:
            terms["boundary"] = zero

        # -- point aquifer properties --------------------------------------
        terms["prop"] = self._property_loss()

        # -- geostatistical structure --------------------------------------
        terms["variogram"] = self._variogram_loss()
        terms["kriging"] = self._kriging_loss()

        # -- edge-preserving regularisation --------------------------------
        # Reuses the collocation property field, so no extra forward pass. The
        # gradient is taken in normalised coordinates (hence the length scale)
        # to keep the penalty independent of the map units.
        terms["smooth"] = smoothness_loss(
            fc["log10K"], cb.xy, delta=0.1, scale=ds.normalizer.scale
        )

        return terms

    def _property_loss(self) -> torch.Tensor:
        zero = torch.zeros((), dtype=self.ds.dtype, device=self.ds.device)
        po = self.prop_obs
        if po is None:
            return zero
        f = self.model.fields(po["batch"])
        idx = po["layer"].clamp(0, self.ds.n_layers - 1)[:, None]
        total, n = zero, 0
        for name, key in (("T", "log10T"), ("S", "log10S")):
            if f"log10{name}" not in po:
                continue
            valid = po[f"{name}_valid"]
            if not bool(valid.any()):
                continue
            pred = f[key].gather(1, idx)[:, 0][valid]
            total = total + weighted_mse(
                pred, po[f"log10{name}"][valid], noise=self.cfg.train.prop_noise
            )
            n += 1
        return total / max(n, 1)

    def _variogram_loss(self) -> torch.Tensor:
        zero = torch.zeros((), dtype=self.ds.dtype, device=self.ds.device)
        if self.cache.vario_batch is None or not self.cache.vario_specs:
            return zero
        f = self.model.fields(self.cache.vario_batch)
        total, n = zero, 0
        for name, layer, off, n1, lag, bins in self.cache.vario_specs:
            v = f["log10T" if name == "T" else "log10S"][:, layer]
            total = total + variogram_loss(
                v[off : off + n1], v[off + n1 : off + 2 * n1], lag, bins,
                self.ds.variograms[name][layer], _VARIOGRAM_BINS,
            )
            n += 1
        return total / max(n, 1)

    def _kriging_loss(self) -> torch.Tensor:
        zero = torch.zeros((), dtype=self.ds.dtype, device=self.ds.device)
        if self.cache.krig_batch is None or not self.cache.krig_specs:
            return zero
        f = self.model.fields(self.cache.krig_batch)
        total, n = zero, 0
        for name, layer, off, cnt, est, w in self.cache.krig_specs:
            v = f["log10T" if name == "T" else "log10S"][off : off + cnt, layer]
            total = total + kriging_loss(v, est, w)
            n += 1
        return total / max(n, 1)

    # ------------------------------------------------------------------ #

    def _static(self, name: str) -> float:
        w = self.cfg.train.weights
        return float(getattr(w, {"prop": "prop_obs", "head": "head_obs"}.get(name, name)))

    def total_loss(self, terms: Dict[str, torch.Tensor], ramp: float = 1.0) -> torch.Tensor:
        out = None
        for name, value in terms.items():
            scale = ramp if name in ("pde", "river", "boundary") else 1.0
            contrib = scale * self.weights[name] * self._static(name) * value
            out = contrib if out is None else out + contrib
        return out

    def _update_adaptive_weights(self, terms: Dict[str, torch.Tensor]) -> None:
        """Gradient-norm balancing (Wang, Teng & Perdikaris, 2021).

        Each loss term is rescaled so its gradient on the shared head network has
        the same magnitude as the PDE residual's. Without it the stiff residual
        gradients swamp the data terms and the model converges to a smooth field
        that ignores the observations.

        Both sides use the *mean* absolute gradient. The original rule takes the
        max for the residual, but max/mean over ~10^5 parameters is itself a
        factor of hundreds, so with a stiff residual every other term pins to the
        ceiling and the balancing stops discriminating between them.
        """
        params = [p for p in self.model.head.parameters() if p.requires_grad]
        if not params or "pde" not in terms:
            return

        def gnorm(t: torch.Tensor, reduce: str):
            # A term can be an exact constant (no boundary segments of a given
            # kind, no variogram fitted), in which case it has no gradient on
            # the head network and must simply be skipped.
            if not t.requires_grad or float(t.detach()) == 0.0:
                return None
            gs = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
            parts = [g.abs().reshape(-1) for g in gs if g is not None]
            if not parts:
                return None
            vals = torch.cat(parts)
            if vals.numel() == 0:
                return None
            return float(vals.max() if reduce == "max" else vals.mean())

        ref = gnorm(terms["pde"], "mean")
        if ref is None or ref <= 0:
            return

        alpha = self.cfg.train.adaptive_alpha
        for name, value in terms.items():
            if name == "pde":
                continue
            g = gnorm(value, "mean")
            if g is None or g <= 1e-30:
                continue
            hat = float(np.clip(ref / g, 1e-4, self.cfg.train.max_adaptive_weight))
            self.weights[name] = min(
                (1 - alpha) * self.weights[name] + alpha * hat,
                self.cfg.train.max_adaptive_weight,
            )

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for tag, obs in (("train", self.obs_train), ("val", self.obs_val)):
            if obs is None:
                continue
            batch, target, layer, _ = obs
            h = self.model.head(batch.xy, batch.t, self.model.fault_feats(batch.xy))
            pred = h.gather(1, layer.clamp(0, self.ds.n_layers - 1)[:, None])[:, 0]
            err = (pred - target).cpu().numpy()
            out[f"{tag}_rmse"] = float(np.sqrt((err**2).mean()))
            out[f"{tag}_mae"] = float(np.abs(err).mean())
        return out

    # ------------------------------------------------------------------ #

    def fit(self) -> List[dict]:
        tr = self.cfg.train
        t0 = time.time()
        self.resample()

        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=tr.lr)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(tr.lr_decay_every, 1), gamma=tr.lr_decay
        )

        for it in range(tr.adam_iters):
            if it > 0 and it % max(tr.resample_every, 1) == 0:
                self.resample()
            elif it > 0:
                self.resample_geostat()

            ramp = min(1.0, (it + 1) / max(tr.pde_warmup, 1))
            terms = self.compute_losses()

            if tr.adaptive_weights and it % max(tr.adaptive_every, 1) == 0 and it > 0:
                self._update_adaptive_weights(terms)

            loss = self.total_loss(terms, ramp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()

            if it % max(tr.log_every, 1) == 0 or it == tr.adam_iters - 1:
                self._log(it, "adam", loss, terms, t0)

        if tr.lbfgs_iters > 0:
            self._run_lbfgs(t0)

        return self.history

    def _run_lbfgs(self, t0: float) -> None:
        """Second-order polish on a frozen set of collocation points.

        L-BFGS assumes a deterministic objective - resampling underneath it makes
        the line search meaningless - so the points are held fixed here.
        """
        tr = self.cfg.train
        self.resample()
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=tr.lbfgs_iters,
            history_size=50,
            tolerance_grad=1e-11,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        state = {"n": 0, "last": None}

        def closure():
            opt.zero_grad(set_to_none=True)
            terms = self.compute_losses()
            loss = self.total_loss(terms, ramp=1.0)
            loss.backward()
            state["n"] += 1
            state["last"] = (loss, terms)
            if state["n"] % max(tr.log_every, 1) == 0:
                self._log(state["n"], "lbfgs", loss, terms, t0)
            return loss

        try:
            opt.step(closure)
        except RuntimeError as exc:  # pragma: no cover - numerical breakdown
            if self.verbose:
                print(f"  L-BFGS stopped early: {exc}")

        if state["last"] is not None:
            self._log(state["n"], "lbfgs", *state["last"], t0)

    def _log(self, it: int, stage: str, loss, terms, t0: float) -> None:
        rec = {
            "iter": int(it),
            "stage": stage,
            "total": float(loss.detach()),
            "elapsed": time.time() - t0,
            **{f"loss_{k}": float(v.detach()) for k, v in terms.items()},
            **{f"w_{k}": float(v) for k, v in self.weights.items()},
            **self.validate(),
        }
        self.history.append(rec)
        if self.verbose:
            print(
                f"  [{stage:5s} {it:6d}] total={rec['total']:.4e} "
                f"head={rec['loss_head']:.3e} pde={rec['loss_pde']:.3e} "
                f"prop={rec['loss_prop']:.3e} vario={rec['loss_variogram']:.3e} "
                f"| RMSE train={rec.get('train_rmse', float('nan')):.3f} "
                f"val={rec.get('val_rmse', float('nan')):.3f} m "
                f"| {rec['elapsed']:.0f}s",
                flush=True,
            )


# --------------------------------------------------------------------------- #


def train_ensemble(
    ds: GWDataset, cfg: Optional[Config] = None, verbose: bool = True
) -> tuple[List[PINN], List[List[dict]]]:
    """Train ``cfg.train.n_ensemble`` independently seeded models.

    The spread between members is the model's own estimate of where it is
    unconstrained - which for a sparse inverse problem is at least as useful as
    the point estimate, and is what the uncertainty maps are built from.
    """
    cfg = cfg or ds.cfg
    models, histories = [], []
    for k in range(max(cfg.train.n_ensemble, 1)):
        if verbose and cfg.train.n_ensemble > 1:
            print(f"\n--- ensemble member {k + 1}/{cfg.train.n_ensemble} ---")
        tr = Trainer(ds, cfg, seed=cfg.train.seed + 1000 * k, verbose=verbose)
        histories.append(tr.fit())
        models.append(tr.model)
    return models, histories


__all__ = ["PINN", "Trainer", "train_ensemble"]
