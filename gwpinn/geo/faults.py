"""Faults as anisotropic flow barriers.

A fault is a surface across which the head drops sharply. Two ways of putting
that into a PINN are common:

* **Domain decomposition** (XPINN / cPINN): cut the domain along the faults,
  fit one network per sub-domain, and couple them with interface conditions.
  Exact, but it needs the fault network to partition the domain cleanly and it
  multiplies the number of networks.
* **Smeared anisotropic barrier**, used here. The fault is represented as a
  narrow zone in which conductivity *normal to the fault* is multiplied by
  ``alpha`` while the along-fault direction is untouched:

  .. math::  q = -T \\,\\big(I - (1-\\alpha)\\,\\psi(x)\\, n n^{T}\\big)\\, \\nabla h

  with :math:`\\psi` a Gaussian bump of half-width ``w`` centred on the fault.
  This is the continuous analogue of MODFLOW's Horizontal Flow Barrier package,
  it keeps a single global network, and ``alpha`` is a plain trainable scalar.

The head drop across a smeared barrier is roughly :math:`q w / (T \\alpha)`, so
a perfectly impermeable fault would need an infinite gradient. ``alpha`` is
therefore floored at ``perm_min``: an "impermeable" fault is modelled as one
that passes a thousandth of the surrounding conductivity, which for the head
field is indistinguishable from a true barrier.

Every quantity here is computed analytically from the line segments, so it is
exactly differentiable with respect to the collocation coordinates - which is
what the PDE residual needs.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class FaultField(nn.Module):
    """Differentiable anisotropy and side-indicator fields induced by faults."""

    def __init__(
        self,
        lines: Sequence[np.ndarray],
        perm_flags: Optional[np.ndarray] = None,
        width: float = 60.0,
        perm_min: float = 1e-3,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.width = float(width)
        self.perm_min = float(perm_min)
        self.n_faults = len(lines)

        if self.n_faults == 0:
            self.register_buffer("a", torch.zeros(0, 2, dtype=dtype))
            self.register_buffer("ab", torch.zeros(0, 2, dtype=dtype))
            self.register_buffer("inv_l2", torch.zeros(0, dtype=dtype))
            self.register_buffer("normal", torch.zeros(0, 2, dtype=dtype))
            self.seg_index: List[torch.Tensor] = []
            self.register_buffer("fixed_alpha", torch.zeros(0, dtype=dtype))
            self.register_buffer("trainable", torch.zeros(0, dtype=torch.bool))
            self.raw = nn.Parameter(torch.zeros(0, dtype=dtype))
            return

        a_list, ab_list, n_list, owner = [], [], [], []
        for f, ln in enumerate(lines):
            ln = np.asarray(ln, dtype=float)[:, :2]
            if len(ln) < 2:
                continue
            a = ln[:-1]
            b = ln[1:]
            ab = b - a
            seglen = np.hypot(ab[:, 0], ab[:, 1])
            keep = seglen > 1e-9
            a, ab, seglen = a[keep], ab[keep], seglen[keep]
            if len(a) == 0:
                continue
            tang = ab / seglen[:, None]
            nrm = np.column_stack([-tang[:, 1], tang[:, 0]])

            # Give every segment of one fault a consistent normal orientation so
            # the signed side-indicator does not flip sign along the trace.
            ref = nrm.mean(axis=0)
            if np.hypot(*ref) < 1e-6:
                ref = nrm[0]
            nrm[nrm @ ref < 0] *= -1.0

            a_list.append(a)
            ab_list.append(ab)
            n_list.append(nrm)
            owner.append(np.full(len(a), f, dtype=int))

        a_all = np.vstack(a_list)
        ab_all = np.vstack(ab_list)
        n_all = np.vstack(n_list)
        owner = np.concatenate(owner)

        self.register_buffer("a", torch.as_tensor(a_all, dtype=dtype))
        self.register_buffer("ab", torch.as_tensor(ab_all, dtype=dtype))
        self.register_buffer(
            "inv_l2",
            torch.as_tensor(1.0 / np.maximum((ab_all**2).sum(1), 1e-12), dtype=dtype),
        )
        self.register_buffer("normal", torch.as_tensor(n_all, dtype=dtype))

        # Segment indices per fault, kept as a plain list of index tensors.
        self.seg_index = []
        for f in range(self.n_faults):
            idx = np.nonzero(owner == f)[0]
            self.register_buffer(f"_idx_{f}", torch.as_tensor(idx, dtype=torch.long))
            self.seg_index.append(getattr(self, f"_idx_{f}"))

        # ---- permeability parameterisation -------------------------------
        flags = (
            np.ones(self.n_faults)
            if perm_flags is None
            else np.asarray(perm_flags, dtype=float).ravel()
        )
        if flags.size < self.n_faults:
            flags = np.pad(flags, (0, self.n_faults - flags.size), constant_values=1.0)

        trainable = np.isclose(flags, 1.0)
        fixed = np.where(np.isclose(flags, 0.0), self.perm_min, flags)
        fixed = np.clip(fixed, self.perm_min, 1.0)

        self.register_buffer("fixed_alpha", torch.as_tensor(fixed, dtype=dtype))
        self.register_buffer("trainable", torch.as_tensor(trainable, dtype=torch.bool))
        # alpha = perm_min ** sigmoid(raw): log-uniform in [perm_min, 1], which
        # matches how permeability actually varies. raw = 0 starts mid-decade.
        self.raw = nn.Parameter(torch.zeros(self.n_faults, dtype=dtype))

    # ------------------------------------------------------------------ #

    @property
    def has_faults(self) -> bool:
        return self.n_faults > 0 and self.a.numel() > 0

    def alpha(self) -> torch.Tensor:
        """Permeability multiplier of each fault, in ``[perm_min, 1]``."""
        if not self.has_faults:
            return torch.zeros(0, device=self.a.device, dtype=self.a.dtype)
        trained = torch.exp(np.log(self.perm_min) * torch.sigmoid(self.raw))
        return torch.where(self.trainable, trained, self.fixed_alpha)

    # ------------------------------------------------------------------ #

    def _segment_terms(self, xy: torch.Tensor):
        """Squared distance and offset vector from each point to each segment."""
        ap = xy[:, None, :] - self.a[None, :, :]                    # (N, S, 2)
        t = (ap * self.ab[None, :, :]).sum(-1) * self.inv_l2[None, :]
        t = t.clamp(0.0, 1.0)
        delta = ap - t[..., None] * self.ab[None, :, :]             # (N, S, 2)
        d2 = (delta**2).sum(-1)                                     # (N, S)
        return d2, delta

    def _per_fault(self, xy: torch.Tensor):
        """Blend the segments of each fault into one bump, normal and side sign.

        Returns ``(psi, nnT, side)`` with shapes ``(N, F)``, ``(N, F, 2, 2)``
        and ``(N, F)``. Segments are combined with softmax weights of
        ``-d^2/w^2``, i.e. a smooth nearest-segment selection, so the result
        stays differentiable where the closest segment changes.
        """
        d2, delta = self._segment_terms(xy)
        w2 = self.width**2

        psi_list, nnt_list, side_list = [], [], []
        for f in range(self.n_faults):
            idx = self.seg_index[f]
            if idx.numel() == 0:
                n = xy.shape[0]
                psi_list.append(xy.new_zeros(n))
                nnt_list.append(xy.new_zeros(n, 2, 2))
                side_list.append(xy.new_zeros(n))
                continue

            d2f = d2[:, idx]                                        # (N, Sf)
            logits = -d2f / w2
            w = torch.softmax(logits, dim=1)                        # (N, Sf)

            psi_list.append((w * torch.exp(logits)).sum(1))         # smooth max

            nf = self.normal[idx]                                   # (Sf, 2)
            nnt = nf[:, :, None] * nf[:, None, :]                   # (Sf, 2, 2)
            nnt_list.append(torch.einsum("ns,sij->nij", w, nnt))

            signed = (delta[:, idx, :] * nf[None, :, :]).sum(-1)    # (N, Sf)
            side_list.append((w * signed).sum(1))

        psi = torch.stack(psi_list, dim=1)
        nnT = torch.stack(nnt_list, dim=1)
        side = torch.stack(side_list, dim=1)
        return psi, nnT, side

    # ------------------------------------------------------------------ #

    def evaluate(self, xy: torch.Tensor):
        """Anisotropy tensor and side indicators in one pass.

        The per-segment distance computation is the expensive part and both
        outputs need it, so callers that want both should use this rather than
        calling :meth:`anisotropy` and :meth:`side_features` separately.
        """
        if not self.has_faults:
            n = xy.shape[0]
            eye = torch.eye(2, dtype=xy.dtype, device=xy.device).expand(n, 2, 2)
            return eye.clone(), xy.new_zeros(n, 0)
        psi, nnT, side = self._per_fault(xy)
        return self._anisotropy_from(xy, psi, nnT), torch.tanh(side / self.width)

    def anisotropy(self, xy: torch.Tensor) -> torch.Tensor:
        """Conductivity-multiplier tensor ``A(x)``, shape ``(N, 2, 2)``.

        The effective flux is ``q = -T A grad(h)``. ``A`` is the identity away
        from every fault, and is rescaled where faults overlap so its smallest
        eigenvalue never drops below ``perm_min`` (which keeps the operator
        elliptic and the residual finite).
        """
        n = xy.shape[0]
        if not self.has_faults:
            return torch.eye(2, dtype=xy.dtype, device=xy.device).expand(n, 2, 2).clone()
        psi, nnT, _ = self._per_fault(xy)
        return self._anisotropy_from(xy, psi, nnT)

    def _anisotropy_from(
        self, xy: torch.Tensor, psi: torch.Tensor, nnT: torch.Tensor
    ) -> torch.Tensor:
        n = xy.shape[0]
        eye = torch.eye(2, dtype=xy.dtype, device=xy.device).expand(n, 2, 2)
        alpha = self.alpha()                                        # (F,)
        coef = psi * (1.0 - alpha)[None, :]                         # (N, F)
        M = torch.einsum("nf,nfij->nij", coef, nnT)                 # (N, 2, 2)

        # Largest eigenvalue of a symmetric 2x2, in closed form.
        tr = M[:, 0, 0] + M[:, 1, 1]
        det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        disc = torch.clamp(0.25 * tr**2 - det, min=0.0)
        lam_max = 0.5 * tr + torch.sqrt(disc + 1e-30)

        cap = 1.0 - self.perm_min
        scale = torch.clamp(cap / torch.clamp(lam_max, min=1e-12), max=1.0)
        return eye - scale[:, None, None] * M

    def side_features(self, xy: torch.Tensor) -> torch.Tensor:
        """Signed side indicators, shape ``(N, F)``, saturating to +/-1.

        Fed to the network as extra inputs these act as a learned basis for a
        jump across the fault, which is what lets a smooth coordinate network
        produce a sharp head offset without needing extreme Fourier frequencies.
        """
        if not self.has_faults:
            return xy.new_zeros(xy.shape[0], 0)
        _, _, side = self._per_fault(xy)
        return torch.tanh(side / self.width)

    def barrier_strength(self, xy: torch.Tensor) -> torch.Tensor:
        """``psi * (1 - alpha)`` summed over faults - a 0..1 "how blocked" map."""
        if not self.has_faults:
            return xy.new_zeros(xy.shape[0])
        psi, _, _ = self._per_fault(xy)
        return (psi * (1.0 - self.alpha())[None, :]).sum(1)

    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        return (
            f"n_faults={self.n_faults}, n_segments={self.a.shape[0]}, "
            f"width={self.width}, perm_min={self.perm_min}"
        )


__all__ = ["FaultField"]
