"""The reduced benchmark expressed for torch: geometry, forcing, faults, sampling.

Everything that is common to all formulations lives here, so that a formulation
module only has to say *how* it writes the residual, never *what* the physics is.
That separation is what makes the study a controlled comparison rather than a
collection of loosely related models.

Design decisions worth stating
------------------------------
**Sources use the exact cell indicator, not a smeared bump.**  The river is a
meandering set of 139 cells and the wells are 8 cells; MODFLOW applies their
conductance to those cells and nothing else.  A PINN *could* smear them into
Gaussian ridges, but then every variant would carry the same smearing error and
the river-exchange metric would be measuring my smoothing kernel rather than the
surrogate.  Instead each collocation point is looked up in the grid and gets the
source of the cell it falls in.  The source field is then piecewise constant -
which is exactly what the reference solves.

**Faults get a physically calibrated barrier width.**  MODFLOW's HFB gives a face
a conductance ``hydchr * b * w_face`` in series with the cells.  The continuous
analogue is a zone of width ``w`` whose normal conductivity is ``K_n = hydchr * w``.
So the anisotropy multiplier is ``alpha = hydchr * w / K``, not a tuned constant.
For the impermeable fault that is ~2e-7 and for the leaky one ~2e-2 - three
orders of magnitude apart, as they should be.

**Time is physical.**  Derivatives are taken with respect to the physical
coordinates and the network normalises internally, so a residual is in m/d and
can be compared with the reference budget without bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gwpinn.benchmark.mf6case import ReducedCase


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


@dataclass
class Scales:
    """Physical <-> network units."""

    x0: float
    y0: float
    L: float
    t0: float
    t1: float
    h0: float
    hs: float
    residual: float

    def norm(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            (x - self.x0) / self.L,
            (y - self.y0) / self.L,
            2.0 * (t - self.t0) / (self.t1 - self.t0) - 1.0,
        ], dim=-1)

    def norm_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.stack([(x - self.x0) / self.L, (y - self.y0) / self.L], dim=-1)

    def denorm_h(self, hn: torch.Tensor) -> torch.Tensor:
        return hn * self.hs + self.h0


# --------------------------------------------------------------------------- #
# Problem
# --------------------------------------------------------------------------- #


class Problem:
    """Torch-side view of a :class:`ReducedCase`."""

    def __init__(self, case: ReducedCase, device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 fault_width: float = 150.0) -> None:
        self.case = case
        self.device = torch.device(device)
        self.dtype = dtype
        self.fault_width = float(fault_width)
        #: "bilinear" for pointwise formulations, "nearest" for finite volume.
        self.coeff_mode = "bilinear"

        nr, nc = case.nrow, case.ncol
        self.nrow, self.ncol = nr, nc
        self.dx, self.dy = case.delr, case.delc
        self.area = self.dx * self.dy
        self.xmin, self.xmax = case.x_origin, case.x_origin + nc * case.delr
        self.ymin, self.ymax = case.y_origin, case.y_origin + nr * case.delc
        self.t0 = float(case.times[0] - case.dt)
        self.t1 = float(case.times[-1])

        def T(a, d=None):
            return torch.as_tensor(np.asarray(a), dtype=d or dtype, device=self.device)

        # ---- static fields -------------------------------------------------
        self.kh_true = T(case.kh)
        self.sy = T(case.sy)
        self.ss = T(case.ss)
        self.top = T(case.top)
        self.botm = T(case.botm)
        self.head_ref = T(case.head)
        self.head_init = T(case.head_init)

        h = case.head
        self.scales = Scales(
            x0=0.5 * (self.xmin + self.xmax), y0=0.5 * (self.ymin + self.ymax),
            L=0.5 * max(self.xmax - self.xmin, self.ymax - self.ymin),
            t0=self.t0, t1=self.t1,
            h0=float(h.mean()), hs=float(max(h.std(), 1.0)),
            residual=1.0,
        )
        # Characteristic residual magnitude: the storage term of the reference
        # solution.  Without this the PDE loss sits orders of magnitude away from
        # the data loss and no fixed weighting works.
        dhdt = np.gradient(h, case.dt, axis=0)
        self.scales.residual = float(max(np.abs(case.sy * dhdt).mean(), 1e-6))

        # Characteristic depth-integrated Darcy flux, used to scale the mixed
        # formulation's flux output so the network emits O(1) numbers.  Deriving
        # it from the reference rather than hard-coding a round number matters:
        # a flux head asked to produce 0.02 trains far more slowly than one
        # asked to produce 1, and that would penalise the mixed formulation for
        # a scaling mistake rather than for anything about the formulation.
        gy_ref, gx_ref = np.gradient(h.mean(0), case.delc, case.delr)
        grad_ref = float(np.abs(np.hypot(gx_ref, gy_ref)).mean())
        b_ref = float(np.mean(np.clip(h.mean(0) - case.botm, 1.0, None)))
        self.flux_scale = float(max(np.median(case.kh) * b_ref * grad_ref, 1e-3))

        # ---- forcing -------------------------------------------------------
        self.times = T(case.times)
        self.dt = float(case.dt)
        self.riv_stage = T(case.riv_stage)
        self.recharge = T(case.recharge)
        self.riv_cond_area = float(case.riv_cond) / self.area   # 1/d
        self.riv_bottom = float(case.riv_bottom)

        # Per-cell source maps, rebuilt per stress period only where needed.
        riv_mask = np.zeros((nr, nc), dtype=bool)
        riv_mask[case.riv_cells[:, 0], case.riv_cells[:, 1]] = True
        self.riv_mask = T(riv_mask.astype(np.float32))
        self.riv_cells = torch.as_tensor(case.riv_cells, dtype=torch.long,
                                         device=self.device)

        wel_map = np.zeros((case.wel_q.shape[1], nr, nc), dtype=np.float32)
        for i, (r, c) in enumerate(case.wel_cells):
            wel_map[:, r, c] += case.wel_q[i] / self.area   # m/d
        self.wel_map = T(wel_map)

        # ---- faults --------------------------------------------------------
        self._build_faults(case)

        # ---- observation design (filled by :meth:`make_observations`) -------
        self.obs: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Faults
    # ------------------------------------------------------------------ #

    def _build_faults(self, case: ReducedCase) -> None:
        """Fit a straight trace to each fault's staircase and precompute geometry.

        MODFLOW represents each oblique fault as a staircase of blocked faces.
        The underlying geological object is a straight line, and that is what the
        continuous formulations should see: least-squares fitting the trace back
        out of the face midpoints recovers it without needing the generator's
        parameters.
        """
        faces = case.fault_faces
        segs: List[Tuple[np.ndarray, np.ndarray, float]] = []
        groups: Dict[float, List[np.ndarray]] = {}
        for r1, c1, r2, c2, hc in faces:
            mx = case.x_origin + 0.5 * (c1 + c2 + 1.0) * case.delr
            my = case.y_origin + (case.nrow - 0.5 * (r1 + r2 + 1.0)) * case.delc
            groups.setdefault(float(hc), []).append(np.array([mx, my]))

        self.fault_lines: List[Tuple[np.ndarray, np.ndarray]] = []
        self.fault_hydchr: List[float] = []
        for hc, pts in sorted(groups.items()):
            P = np.stack(pts)
            centre = P.mean(0)
            # Principal direction of the face midpoints = the fault strike.
            _, _, vt = np.linalg.svd(P - centre, full_matrices=False)
            direction = vt[0]
            s = (P - centre) @ direction
            self.fault_lines.append((centre + s.min() * direction,
                                     centre + s.max() * direction))
            self.fault_hydchr.append(hc)

        self.n_faults = len(self.fault_lines)
        if self.n_faults:
            a = np.stack([l[0] for l in self.fault_lines])
            b = np.stack([l[1] for l in self.fault_lines])
            self._fa = torch.as_tensor(a, dtype=self.dtype, device=self.device)
            self._fb = torch.as_tensor(b, dtype=self.dtype, device=self.device)
            self._fhc = torch.as_tensor(self.fault_hydchr, dtype=self.dtype,
                                        device=self.device)

        # The blocked faces themselves, for the finite-volume formulation, as
        # dense multiplier arrays on the x- and y-faces of the grid.
        mx = np.ones((case.nrow, case.ncol - 1), dtype=np.float32)
        my = np.ones((case.nrow - 1, case.ncol), dtype=np.float32)
        self.face_hydchr_x = np.full((case.nrow, case.ncol - 1), np.inf, dtype=np.float32)
        self.face_hydchr_y = np.full((case.nrow - 1, case.ncol), np.inf, dtype=np.float32)
        for r1, c1, r2, c2, hc in faces:
            r1, c1, r2, c2 = int(r1), int(c1), int(r2), int(c2)
            if r1 == r2:
                self.face_hydchr_x[r1, min(c1, c2)] = min(
                    self.face_hydchr_x[r1, min(c1, c2)], hc)
            else:
                self.face_hydchr_y[min(r1, r2), c1] = min(
                    self.face_hydchr_y[min(r1, r2), c1], hc)
        self.t_face_hydchr_x = torch.as_tensor(self.face_hydchr_x, dtype=self.dtype,
                                               device=self.device)
        self.t_face_hydchr_y = torch.as_tensor(self.face_hydchr_y, dtype=self.dtype,
                                               device=self.device)

    def signed_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Signed perpendicular distance to each fault trace, ``(N, n_faults)``."""
        if not self.n_faults:
            return torch.zeros(x.shape[0], 0, dtype=self.dtype, device=self.device)
        p = torch.stack([x, y], dim=-1)[:, None, :]        # (N, 1, 2)
        a, b = self._fa[None], self._fb[None]              # (1, F, 2)
        d = b - a
        n = torch.stack([-d[..., 1], d[..., 0]], dim=-1)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return ((p - a) * n).sum(-1)                        # (N, F)

    def fault_features(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``tanh(s / w)`` per fault - a ready-made basis for a head jump."""
        return torch.tanh(self.signed_distance(x, y) / self.fault_width)

    def anisotropy(self, x: torch.Tensor, y: torch.Tensor,
                   K: torch.Tensor) -> torch.Tensor:
        """Fault anisotropy tensor ``A``, ``(N, 2, 2)``.

        Inside a barrier of half-width ``w`` the conductivity normal to the trace
        is scaled by ``alpha = hydchr * w / K`` - the value that reproduces the
        discrete HFB conductance - while flow along the trace is untouched.
        """
        N = x.shape[0]
        A = torch.eye(2, dtype=self.dtype, device=self.device).expand(N, 2, 2).clone()
        if not self.n_faults:
            return A
        s = self.signed_distance(x, y)                                  # (N, F)
        psi = torch.exp(-((s / self.fault_width) ** 2))                 # bump
        d = self._fb - self._fa
        nvec = torch.stack([-d[:, 1], d[:, 0]], dim=-1)
        nvec = nvec / nvec.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (F, 2)
        for f in range(self.n_faults):
            alpha = (self._fhc[f] * self.fault_width / K.clamp_min(1e-6)).clamp(1e-8, 1.0)
            nn_outer = torch.outer(nvec[f], nvec[f])[None]              # (1, 2, 2)
            scale = (1.0 - alpha) * psi[:, f]                           # (N,)
            A = A - scale[:, None, None] * nn_outer
        return A

    # ------------------------------------------------------------------ #
    # Grid lookup and sources
    # ------------------------------------------------------------------ #

    def cell_index(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        col = ((x - self.xmin) / self.dx).floor().long().clamp(0, self.ncol - 1)
        row = ((self.ymax - y) / self.dy).floor().long().clamp(0, self.nrow - 1)
        return row, col

    def period_index(self, t: torch.Tensor) -> torch.Tensor:
        """Stress period containing ``t`` (forcing is piecewise constant)."""
        k = ((t - self.t0) / self.dt).floor().long()
        return k.clamp(0, len(self.times) - 1)

    def gather(self, field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Nearest-cell lookup - piecewise constant, zero gradient."""
        r, c = self.cell_index(x, y)
        return field[r, c]

    def interp(self, field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of a cell-centred field."""
        gx = (x - self.xmin) / (self.xmax - self.xmin) * 2.0 - 1.0
        gy = (self.ymax - y) / (self.ymax - self.ymin) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
        return torch.nn.functional.grid_sample(
            field[None, None], grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        ).view(-1)

    def coeff(self, field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sample a PDE *coefficient*.

        This distinction matters and is easy to get wrong.  The strong and mixed
        forms differentiate ``T = K b`` with respect to position, so
        ``div(T grad h)`` contains ``grad T . grad h``.  Sampling ``K``, the layer
        top and the layer bottom by nearest cell would make those gradients
        identically zero and silently delete a term from the PDE - the surrogate
        would then be solving a different equation from the reference and would
        be blamed for the discrepancy.  Coefficients are therefore interpolated
        bilinearly for the pointwise formulations.

        The finite-volume form is the opposite case: it wants genuine
        cell-averaged coefficients, because that is what its face conductances
        are built from and what MODFLOW itself uses.  It sets
        ``coeff_mode = "nearest"``.

        Source terms (river mask, wells, recharge) are always nearest: they are
        genuinely piecewise constant per cell in the reference and never appear
        under a spatial derivative.
        """
        if self.coeff_mode == "nearest":
            return self.gather(field, x, y)
        return self.interp(field, x, y)

    def sources(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
                h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Areal source terms in m/d at the given points.

        The river term uses MODFLOW's conductance-limited form: once the head
        falls below the river bottom the leakage saturates at
        ``C (stage - rbot)`` and stops depending on the aquifer head at all.
        That kink is a genuine feature of the RIV package and a surrogate that
        smooths it over will get the exchange wrong at low stage.
        """
        r, c = self.cell_index(x, y)
        k = self.period_index(t)
        rch = self.recharge[k]
        wel = self.wel_map[k, r, c]
        stage = self.riv_stage[k]
        in_riv = self.riv_mask[r, c]
        h_eff = torch.maximum(h, torch.full_like(h, self.riv_bottom))
        riv = self.riv_cond_area * in_riv * (stage - h_eff)
        return {"recharge": rch, "wel": wel, "riv": riv, "riv_mask": in_riv}

    # ------------------------------------------------------------------ #
    # Aquifer coefficients
    # ------------------------------------------------------------------ #

    def thickness(self, h: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                  beta: float = 2.0) -> torch.Tensor:
        """Saturated thickness, smoothly clamped between 1 m and the full section."""
        zb = self.coeff(self.botm, x, y)
        zt = self.coeff(self.top, x, y)
        full = (zt - zb).clamp_min(1.0)
        raw = h - zb
        soft = torch.nn.functional.softplus(raw - 1.0, beta=beta) + 1.0
        return full - torch.nn.functional.softplus(full - soft, beta=beta)

    def storage(self, h: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                b: torch.Tensor, eps: float = 0.5) -> torch.Tensor:
        """Effective storage coefficient.

        MODFLOW switches from specific yield to confined storage once the head
        rises above the cell top.  A hard switch would put a discontinuity in the
        residual, so the indicator is replaced by a sigmoid of width ``eps``;
        with ``eps = 0.5 m`` the smoothing is well below the metre-scale head
        changes the study cares about.
        """
        zt = self.coeff(self.top, x, y)
        sy = self.coeff(self.sy, x, y)
        ss = self.coeff(self.ss, x, y)
        unconfined = torch.sigmoid((zt - h) / eps)
        return ss * b + sy * unconfined

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def sample_interior(self, n: int, gen: torch.Generator,
                        fault_fraction: float = 0.25,
                        river_fraction: float = 0.15) -> Tuple[torch.Tensor, ...]:
        """Collocation points, over-sampled where the solution is hardest.

        A uniform sample puts almost nothing inside a 150 m barrier or on a
        one-cell-wide river, and those are precisely the places the residual has
        to be enforced.  Roughly 40 % of the budget is therefore steered onto
        them - stratified sampling, not importance sampling, so the loss stays an
        unbiased estimate of a *weighted* residual norm with known weights.
        """
        n_f = int(n * fault_fraction) if self.n_faults else 0
        n_r = int(n * river_fraction)
        n_u = n - n_f - n_r

        xs = [torch.rand(n_u, generator=gen, device=self.device, dtype=self.dtype)
              * (self.xmax - self.xmin) + self.xmin]
        ys = [torch.rand(n_u, generator=gen, device=self.device, dtype=self.dtype)
              * (self.ymax - self.ymin) + self.ymin]

        if n_f:
            f = torch.randint(0, self.n_faults, (n_f,), generator=gen, device=self.device)
            u = torch.rand(n_f, generator=gen, device=self.device, dtype=self.dtype)
            a, b = self._fa[f], self._fb[f]
            base = a + (b - a) * u[:, None]
            d = b - a
            nvec = torch.stack([-d[:, 1], d[:, 0]], dim=-1)
            nvec = nvec / nvec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            off = (torch.rand(n_f, generator=gen, device=self.device, dtype=self.dtype)
                   * 6.0 - 3.0) * self.fault_width
            p = base + nvec * off[:, None]
            xs.append(p[:, 0].clamp(self.xmin, self.xmax))
            ys.append(p[:, 1].clamp(self.ymin, self.ymax))

        if n_r:
            idx = torch.randint(0, self.riv_cells.shape[0], (n_r,), generator=gen,
                                device=self.device)
            rr, cc = self.riv_cells[idx, 0], self.riv_cells[idx, 1]
            jx = torch.rand(n_r, generator=gen, device=self.device, dtype=self.dtype)
            jy = torch.rand(n_r, generator=gen, device=self.device, dtype=self.dtype)
            xs.append(self.xmin + (cc.to(self.dtype) + jx) * self.dx)
            ys.append(self.ymax - (rr.to(self.dtype) + jy) * self.dy)

        x = torch.cat(xs)
        y = torch.cat(ys)
        t = (torch.rand(x.shape[0], generator=gen, device=self.device, dtype=self.dtype)
             * (self.t1 - self.t0) + self.t0)
        return x, y, t

    def sample_boundary(self, n: int, gen: torch.Generator) -> Tuple[torch.Tensor, ...]:
        """Points on the four no-flow edges, with their outward normals."""
        side = torch.randint(0, 4, (n,), generator=gen, device=self.device)
        u = torch.rand(n, generator=gen, device=self.device, dtype=self.dtype)
        x = torch.where(side == 0, self.xmin, torch.where(side == 1, self.xmax,
                        self.xmin + u * (self.xmax - self.xmin)))
        y = torch.where(side <= 1, self.ymin + u * (self.ymax - self.ymin),
                        torch.where(side == 2, self.ymin, self.ymax))
        nx = torch.where(side == 0, -1.0, torch.where(side == 1, 1.0, 0.0)).to(self.dtype)
        ny = torch.where(side == 2, -1.0, torch.where(side == 3, 1.0, 0.0)).to(self.dtype)
        t = (torch.rand(n, generator=gen, device=self.device, dtype=self.dtype)
             * (self.t1 - self.t0) + self.t0)
        return x, y, t, nx, ny

    def grid_points(self, stride: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cell-centre coordinates as flat tensors (for evaluation and FV)."""
        cols = torch.arange(0, self.ncol, stride, device=self.device)
        rows = torch.arange(0, self.nrow, stride, device=self.device)
        X = self.xmin + (cols.to(self.dtype) + 0.5) * self.dx
        Y = self.ymax - (rows.to(self.dtype) + 0.5) * self.dy
        yy, xx = torch.meshgrid(Y, X, indexing="ij")
        return xx.reshape(-1), yy.reshape(-1)

    # ------------------------------------------------------------------ #
    # Observation design for the inverse task
    # ------------------------------------------------------------------ #

    def make_observations(self, n_wells: int = 60, noise_m: float = 0.02,
                          n_prop: int = 12, seed: int = 0) -> Dict[str, torch.Tensor]:
        """Sparse-in-space, dense-in-time head loggers plus a few pumping tests.

        This is what a real monitoring network looks like: a few dozen boreholes
        logging every couple of hours, and a handful of pumping tests giving a
        point value of K to about a factor of 1.5.  Both are needed - heads alone
        fix only the divergence of ``T grad h`` and leave the overall level of the
        K field unidentified.
        """
        rng = np.random.default_rng(seed)
        nr, nc = self.nrow, self.ncol
        rows = rng.integers(2, nr - 2, n_wells)
        cols = rng.integers(2, nc - 2, n_wells)
        x = self.xmin + (cols + 0.5) * self.dx
        y = self.ymax - (rows + 0.5) * self.dy

        h = self.case.head[:, rows, cols]                       # (nper, n_wells)
        h = h + rng.normal(0.0, noise_m, h.shape)
        tt = np.repeat(self.case.times[:, None], n_wells, axis=1)
        xx = np.repeat(x[None, :], len(self.case.times), axis=0)
        yy = np.repeat(y[None, :], len(self.case.times), axis=0)

        pr = rng.integers(2, nr - 2, n_prop)
        pc = rng.integers(2, nc - 2, n_prop)
        k_true = self.case.kh[pr, pc]
        log_k = np.log10(k_true) + rng.normal(0.0, 0.18, n_prop)  # factor ~1.5

        def T(a):
            return torch.as_tensor(np.asarray(a).ravel(), dtype=self.dtype,
                                   device=self.device)

        self.obs = {
            "x": T(xx), "y": T(yy), "t": T(tt), "h": T(h),
            "well_x": T(x), "well_y": T(y),
            "well_row": torch.as_tensor(rows, device=self.device),
            "well_col": torch.as_tensor(cols, device=self.device),
            "prop_x": T(self.xmin + (pc + 0.5) * self.dx),
            "prop_y": T(self.ymax - (pr + 0.5) * self.dy),
            "prop_logk": T(log_k),
        }
        return self.obs


__all__ = ["Problem", "Scales"]
