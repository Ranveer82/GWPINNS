"""Training strategies: how the loss terms are balanced and how time is handled.

Loss balancing
--------------
A PINN loss is a sum of terms with incommensurate units - metres of head misfit
against m/d of residual - and the answer depends strongly on their ratio.  Three
published schemes are implemented, plus a fixed-weight control.

``fixed``    hand-set multipliers; the control.
``gradnorm`` Wang, Teng & Perdikaris (2021).  Scale each term so its gradient
             norm matches the mean.  Cheap and robust; the current gwpinn default.
``ntk``      Wang, Yu & Perdikaris (2022).  Balance the *neural tangent kernel*
             traces instead of the gradient norms, which is the quantity that
             actually governs each term's convergence rate.
``rba``      Anagnostopoulos et al. (2024), residual-based attention.  A
             per-point multiplier updated by an exponential moving average of the
             normalised residual.  Unlike the other three this reweights *within*
             the PDE term, concentrating effort where the residual is stubborn -
             which on this problem means the fault zones and the river.

Time
----
``plain``    sample the whole space-time domain at once.  The default, and the
             reason many transient PINNs quietly fail: nothing stops the network
             from fitting late times before early ones, and a solution that is
             wrong at t=0 can still have a small residual everywhere.
``causal``   Wang, Sankaran & Perdikaris (2022).  Split the window into bins and
             weight bin ``i`` by ``exp(-eps * sum_{j<i} L_j)``, so a bin is only
             "unlocked" once its predecessors are converged.  Restores the
             temporal causality an all-at-once loss destroys.
``march``    train on an expanding time window.  Cruder than causal weighting but
             it also bounds the space-time volume the network must fit at once,
             which matters when the tide needs six samples per 12.4 h cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


# --------------------------------------------------------------------------- #
# Loss balancing
# --------------------------------------------------------------------------- #


class LossBalancer:
    """Adaptive multipliers over the named loss terms."""

    def __init__(self, kind: str = "gradnorm", alpha: float = 0.9,
                 every: int = 100, max_weight: float = 1e4,
                 base: Optional[Dict[str, float]] = None) -> None:
        self.kind = kind
        self.alpha = float(alpha)
        self.every = int(every)
        self.max_weight = float(max_weight)
        self.base = dict(base or {})
        self.w: Dict[str, float] = {}

    def weight(self, name: str) -> float:
        return self.base.get(name, 1.0) * self.w.get(name, 1.0)

    def total(self, terms: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = None
        for k, v in terms.items():
            term = self.weight(k) * v
            out = term if out is None else out + term
        return out

    @torch.no_grad()
    def _set(self, name: str, value: float) -> None:
        prev = self.w.get(name, 1.0)
        new = self.alpha * prev + (1.0 - self.alpha) * value
        self.w[name] = float(min(max(new, 1e-6), self.max_weight))

    def update(self, terms: Dict[str, torch.Tensor], params: List[torch.Tensor],
               step: int) -> None:
        if self.kind == "fixed" or step % self.every != 0:
            return
        if self.kind not in ("gradnorm", "ntk"):
            return

        stats: Dict[str, float] = {}
        for name, value in terms.items():
            if not value.requires_grad:
                continue
            grads = torch.autograd.grad(value, params, retain_graph=True,
                                        allow_unused=True)
            flat = torch.cat([g.reshape(-1) for g in grads if g is not None]) \
                if any(g is not None for g in grads) else None
            if flat is None or flat.numel() == 0:
                continue
            # gradnorm balances |grad L_i|; the NTK rule balances |grad L_i|^2,
            # which is the diagonal of the tangent kernel for a squared loss.
            stats[name] = float(flat.norm()) if self.kind == "gradnorm" \
                else float(flat.pow(2).sum())
        if len(stats) < 2:
            return
        ref = sum(stats.values()) / len(stats)
        for name, s in stats.items():
            self._set(name, ref / max(s, 1e-12))


class ResidualAttention:
    """Per-point multipliers for the PDE residual (RBA).

    ``lambda <- gamma * lambda + eta * |r| / max|r|``, clipped.  Points whose
    residual refuses to fall accumulate weight, so the optimiser is pushed onto
    them instead of averaging them away.  The buffer is keyed by position in the
    collocation batch, so it only works when the batch is *not* resampled every
    step - hence the resample period is a real hyper-parameter here.
    """

    def __init__(self, n: int, gamma: float = 0.999, eta: float = 0.01,
                 device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.lam = torch.ones(n, device=device, dtype=dtype)
        self.gamma = float(gamma)
        self.eta = float(eta)

    def reset(self, n: int) -> None:
        if n != self.lam.numel():
            self.lam = torch.ones(n, device=self.lam.device, dtype=self.lam.dtype)
        else:
            self.lam.fill_(1.0)

    @torch.no_grad()
    def update(self, residual: torch.Tensor) -> None:
        r = residual.detach().abs().reshape(-1)
        if r.numel() != self.lam.numel():
            self.lam = torch.ones_like(r)
        self.lam.mul_(self.gamma).add_(self.eta * r / r.max().clamp_min(1e-12))
        self.lam.clamp_(0.1, 10.0)

    def apply(self, sq_residual: torch.Tensor) -> torch.Tensor:
        lam = self.lam
        if lam.numel() != sq_residual.reshape(-1).shape[0]:
            return sq_residual.mean()
        return (lam * sq_residual.reshape(-1)).mean()


# --------------------------------------------------------------------------- #
# Temporal strategies
# --------------------------------------------------------------------------- #


@dataclass
class CausalWeighting:
    """Wang et al. (2022) causal weights over ``n_bins`` time bins."""

    n_bins: int = 16
    eps: float = 1.0
    t0: float = 0.0
    t1: float = 1.0

    def bin_of(self, t: torch.Tensor) -> torch.Tensor:
        u = (t - self.t0) / max(self.t1 - self.t0, 1e-12)
        return (u * self.n_bins).floor().long().clamp(0, self.n_bins - 1)

    def __call__(self, sq_residual: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        idx = self.bin_of(t)
        per_bin = torch.zeros(self.n_bins, device=sq_residual.device,
                              dtype=sq_residual.dtype)
        counts = torch.zeros_like(per_bin)
        per_bin = per_bin.index_add(0, idx, sq_residual.reshape(-1))
        counts = counts.index_add(0, idx, torch.ones_like(sq_residual.reshape(-1)))
        per_bin = per_bin / counts.clamp_min(1.0)

        # w_i = exp(-eps * sum_{j<i} L_j); detached so the weights steer the
        # optimiser without contributing gradients of their own.
        cum = torch.cumsum(per_bin, 0) - per_bin
        w = torch.exp(-self.eps * cum).detach()
        return (w * per_bin).sum() / w.sum().clamp_min(1e-12)


@dataclass
class TimeMarching:
    """Expanding-window curriculum over the simulation period."""

    t0: float
    t1: float
    n_stages: int = 4
    warm_fraction: float = 0.25

    def horizon(self, progress: float) -> float:
        """Upper limit of the sampled time window at training ``progress`` in [0,1]."""
        if self.n_stages <= 1:
            return self.t1
        p = min(max(progress / max(1.0 - self.warm_fraction, 1e-6), 0.0), 1.0)
        stage = min(int(p * self.n_stages) + 1, self.n_stages)
        return self.t0 + (self.t1 - self.t0) * stage / self.n_stages


__all__ = ["LossBalancer", "ResidualAttention", "CausalWeighting", "TimeMarching"]
