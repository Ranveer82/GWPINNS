"""Composing a formulation from its axes, and training it.

A *specification* is a point in the design space:

    Spec(arch=..., form=..., fault=..., temporal=..., balance=..., kfield=...)

Every axis is independent, so the study can vary one at a time from a common
baseline and attribute a change in a metric to the axis that moved.  That is the
whole reason for the indirection: a comparison of six named methods from six
papers confounds architecture with formulation with training schedule, and tells
you almost nothing about which ingredient mattered.

Shared across every variant (so they never become confounders):

* the same hard initial condition ``h = h_init + s(t) * hs * NN``, which removes
  the IC loss term entirely and guarantees ``h(t0) = h_init`` exactly;
* the same normalisation and residual scaling;
* the same collocation budget and resampling period;
* the same optimiser and learning-rate schedule.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from gwpinn.formulations.arch import CoordinateNet, count_parameters
from gwpinn.formulations.inverse import KField, build_k_field
from gwpinn.formulations.problem import Problem
from gwpinn.formulations.residuals import (
    ControlVolume, fv_residual, mixed_residual, strong_residual,
)
from gwpinn.formulations.strategy import (
    CausalWeighting, LossBalancer, ResidualAttention, TimeMarching,
)


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #


@dataclass
class Spec:
    """One point in the formulation design space."""

    name: str

    # --- axes -------------------------------------------------------------
    arch: str = "modified"       # mlp | modified | pirate | spinn
    form: str = "strong"         # strong | mixed | fv
    fault: str = "smeared"       # none | smeared | sidefeat | faultcoord
    temporal: str = "plain"      # plain | causal | march
    balance: str = "gradnorm"    # fixed | gradnorm | ntk | rba
    kfield: str = "true"         # true | net | grid | kl

    # --- capacity ---------------------------------------------------------
    width: int = 96
    depth: int = 4
    fourier: int = 64
    fourier_sigma: float = 2.5
    rank: int = 48
    rwf: bool = False

    # --- optimisation -----------------------------------------------------
    iters: int = 4000
    lr: float = 2.0e-3
    lr_decay: float = 0.7
    lr_decay_every: int = 1200
    n_colloc: int = 3072
    n_boundary: int = 384
    fv_time_slices: int = 3
    resample_every: int = 50
    seed: int = 0
    #: Wall-clock budget in seconds.  The study compares formulations at *equal
    #: compute*, not equal iterations: per-iteration cost varies four-fold across
    #: these variants, and "what should I run for the next ten minutes" is the
    #: question a practitioner actually has.  ``iters`` is then an upper bound.
    max_seconds: float = 0.0

    # --- loss weights (before adaptive balancing) -------------------------
    w_pde: float = 1.0
    w_bc: float = 1.0
    w_obs: float = 10.0
    w_prop: float = 1.0
    w_reg: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class Surrogate(nn.Module):
    """Head (and optionally flux) networks plus the conductivity parameterisation."""

    def __init__(self, prob: Problem, spec: Spec) -> None:
        super().__init__()
        self.prob = prob
        self.spec = spec
        torch.manual_seed(spec.seed)

        n_extra = prob.n_faults if spec.fault in ("sidefeat", "faultcoord") else 0
        self.use_fault_features = n_extra > 0
        self.extra_in_fourier = spec.fault == "faultcoord"

        self.head_net = CoordinateNet(
            3, 1, arch=spec.arch, width=spec.width, depth=spec.depth,
            fourier=spec.fourier, fourier_sigma=spec.fourier_sigma,
            n_extra=n_extra, extra_in_fourier=self.extra_in_fourier,
            rank=spec.rank, rwf=spec.rwf, seed=spec.seed,
        )
        self.flux_net = (
            CoordinateNet(
                3, 2, arch=spec.arch, width=spec.width, depth=spec.depth,
                fourier=spec.fourier, fourier_sigma=spec.fourier_sigma,
                n_extra=n_extra, extra_in_fourier=self.extra_in_fourier,
                rank=spec.rank, rwf=spec.rwf, seed=spec.seed + 1,
            ) if spec.form == "mixed" else None
        )
        self.k_field: KField = build_k_field(spec.kfield, prob, seed=spec.seed)

    # ------------------------------------------------------------------ #

    def _inputs(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        coords = self.prob.scales.norm(x, y, t)
        extra = self.prob.fault_features(x, y) if self.use_fault_features else None
        return coords, extra

    def head(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Head in metres, with the initial condition satisfied exactly.

        ``h = h_init(x, y) + s(t) * hs * NN`` with ``s(t0) = 0``.  Making the IC
        a hard constraint rather than a penalty removes a loss term, removes the
        weight that would have to be chosen for it, and guarantees that every
        variant starts from the same state - so a difference between variants is
        a difference in how they propagate it, not in how well they fitted it.
        """
        p = self.prob
        coords, extra = self._inputs(x, y, t)
        raw = self.head_net(coords, extra).squeeze(-1)
        h0 = p.interp(p.head_init, x, y)
        s = (t - p.t0) / (p.t1 - p.t0)
        return h0 + s * p.scales.hs * raw

    def flux(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Depth-integrated Darcy flux, m2/d."""
        coords, extra = self._inputs(x, y, t)
        # Scaled by the characteristic flux of the reference solution so the
        # network output stays O(1); see Problem.flux_scale.
        return self.flux_net(coords, extra) * self.prob.flux_scale

    def k(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.k_field(x, y)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_grid(self, times: torch.Tensor, chunk: int = 20000) -> torch.Tensor:
        """Head on the cell-centre grid at each requested time, ``(nt, nrow, ncol)``."""
        p = self.prob
        gx, gy = p.grid_points()
        out = []
        for tv in times:
            t = torch.full_like(gx, float(tv))
            vals = [self.head(gx[i:i + chunk], gy[i:i + chunk], t[i:i + chunk])
                    for i in range(0, gx.numel(), chunk)]
            out.append(torch.cat(vals).view(p.nrow, p.ncol))
        return torch.stack(out)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


class Runner:
    """Trains one :class:`Surrogate` and reports its history."""

    def __init__(self, prob: Problem, spec: Spec, inverse: bool = False,
                 verbose: bool = True) -> None:
        self.prob = prob
        self.spec = spec
        self.inverse = bool(inverse)
        self.verbose = verbose
        prob.coeff_mode = "nearest" if spec.form == "fv" else "bilinear"

        self.model = Surrogate(prob, spec)
        self.gen = torch.Generator(device=prob.device).manual_seed(spec.seed + 1234)
        self.cv = ControlVolume(prob) if spec.form == "fv" else None

        base = {"pde": spec.w_pde, "bc": spec.w_bc, "obs": spec.w_obs,
                "prop": spec.w_prop, "reg": spec.w_reg}
        self.balancer = LossBalancer(
            kind="gradnorm" if spec.balance == "rba" else spec.balance,
            base=base,
        )
        self.rba = (ResidualAttention(spec.n_colloc, device=str(prob.device),
                                      dtype=prob.dtype)
                    if spec.balance == "rba" else None)
        self.causal = (CausalWeighting(n_bins=16, eps=1.0, t0=prob.t0, t1=prob.t1)
                       if spec.temporal == "causal" else None)
        self.march = (TimeMarching(prob.t0, prob.t1, n_stages=4)
                      if spec.temporal == "march" else None)

        self.batch: Optional[Tuple[torch.Tensor, ...]] = None
        self.history: List[dict] = []

    # ------------------------------------------------------------------ #

    def _resample(self, progress: float) -> None:
        p, s = self.prob, self.spec
        x, y, t = p.sample_interior(s.n_colloc, self.gen)
        if self.march is not None:
            hi = self.march.horizon(progress)
            t = p.t0 + (t - p.t0) * (hi - p.t0) / (p.t1 - p.t0)
        bx, by, bt, bnx, bny = p.sample_boundary(s.n_boundary, self.gen)
        self.batch = (x, y, t, bx, by, bt, bnx, bny)
        if self.rba is not None:
            self.rba.reset(x.numel())

    def _pde_loss(self, progress: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss, raw residual) with the residual non-dimensionalised."""
        p, s, m = self.prob, self.spec, self.model
        scale = p.scales.residual
        x, y, t = self.batch[0], self.batch[1], self.batch[2]

        if s.form == "strong":
            r = strong_residual(p, m.head, m.k, x, y, t) / scale
            sq = r.pow(2)
        elif s.form == "mixed":
            r_const, r_cont = mixed_residual(p, m.head, m.flux, m.k, x, y, t)
            sq = r_const.pow(2).sum(-1) + (r_cont / scale).pow(2)
            r = r_cont / scale
        elif s.form == "fv":
            lo = p.t0 if self.march is None else p.t0
            hi = p.t1 if self.march is None else self.march.horizon(progress)
            tt = (torch.rand(s.fv_time_slices, generator=self.gen, device=p.device,
                             dtype=p.dtype) * (hi - lo) + lo)
            r = fv_residual(p, self.cv, m.head, m.k, tt) / scale
            sq = r.pow(2)
            if self.causal is not None:
                tvec = tt[:, None, None].expand_as(r).reshape(-1)
                return self.causal(sq.reshape(-1), tvec), r
            return sq.mean(), r
        else:
            raise ValueError(f"unknown form {s.form!r}")

        if self.causal is not None:
            return self.causal(sq, t), r
        if self.rba is not None:
            self.rba.update(r)
            return self.rba.apply(sq), r
        return sq.mean(), r

    def _bc_loss(self) -> torch.Tensor:
        """No-flow on the four outer edges (the FV form has it structurally)."""
        p, m = self.prob, self.model
        if self.spec.form == "fv":
            return torch.zeros((), device=p.device, dtype=p.dtype)
        _, _, _, bx, by, bt, nx, ny = self.batch
        bx = bx.clone().requires_grad_(True)
        by = by.clone().requires_grad_(True)
        h = m.head(bx, by, bt)
        K = m.k(bx, by)
        b = p.thickness(h, bx, by)
        T = K * b
        hx = torch.autograd.grad(h, bx, torch.ones_like(h), create_graph=True)[0]
        hy = torch.autograd.grad(h, by, torch.ones_like(h), create_graph=True)[0]
        A = p.anisotropy(bx, by, K)
        qx = -T * (A[:, 0, 0] * hx + A[:, 0, 1] * hy)
        qy = -T * (A[:, 1, 0] * hx + A[:, 1, 1] * hy)
        t_ref = T.detach().mean().clamp_min(1e-6)
        return ((qx * nx + qy * ny) / t_ref).pow(2).mean()

    def _data_losses(self) -> Dict[str, torch.Tensor]:
        p, m = self.prob, self.model
        out: Dict[str, torch.Tensor] = {}
        if not self.inverse or not p.obs:
            return out
        o = p.obs
        hp = m.head(o["x"], o["y"], o["t"])
        out["obs"] = ((hp - o["h"]) / p.scales.hs).pow(2).mean()
        kp = m.k(o["prop_x"], o["prop_y"])
        # Pumping tests pin log10 K to about +/-0.18; fitting them exactly makes
        # the field spike at the points and collapse between them.
        resid = (torch.log10(kp.clamp_min(1e-8)) - o["prop_logk"]).abs()
        out["prop"] = torch.nn.functional.relu(resid - 0.18).pow(2).mean()
        return out

    # ------------------------------------------------------------------ #

    def fit(self) -> "Runner":
        s, p, m = self.spec, self.prob, self.model
        opt = torch.optim.Adam(m.parameters(), lr=s.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, s.lr_decay_every, s.lr_decay)
        params = [q for q in m.parameters() if q.requires_grad]
        t_start = time.time()

        for it in range(s.iters):
            progress = it / max(s.iters - 1, 1)
            if self.batch is None or it % s.resample_every == 0:
                self._resample(progress)

            opt.zero_grad(set_to_none=True)
            pde, _ = self._pde_loss(progress)
            terms: Dict[str, torch.Tensor] = {"pde": pde, "bc": self._bc_loss()}
            terms.update(self._data_losses())
            reg = m.k_field.regulariser()
            if reg.requires_grad:
                terms["reg"] = reg

            if it % self.balancer.every == 0 and it > 0:
                self.balancer.update({k: v for k, v in terms.items() if k != "reg"},
                                     params, it)
            loss = self.balancer.total(terms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()

            if self.verbose and (it % max(s.iters // 8, 1) == 0 or it == s.iters - 1):
                msg = " ".join(f"{k}={float(v):.3e}" for k, v in terms.items())
                print(f"   [{s.name}] it {it:5d}  loss={float(loss):.4e}  {msg}")
            if s.max_seconds and (time.time() - t_start) > s.max_seconds:
                if self.verbose:
                    print(f"   [{s.name}] budget reached at iteration {it}")
                break
            if it % 100 == 0:
                self.history.append({"iter": it, "loss": float(loss.detach()),
                                     **{k: float(v.detach()) for k, v in terms.items()}})

        self.train_seconds = time.time() - t_start
        self.iters_done = it + 1
        self.n_params = count_parameters(m)
        if self.verbose:
            print(f"   [{s.name}] {self.train_seconds:.1f} s, {self.n_params:,} params")
        return self


__all__ = ["Spec", "Surrogate", "Runner"]
