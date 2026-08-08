"""The five criteria the study scores every formulation on.

Each is measured against the MODFLOW 6 reference, which solves the same equation
on the same grid and closes its budget to ~1e-7 relative - so any error reported
here belongs to the surrogate.

1. ``head``    - can it reproduce the transient head field at all?
2. ``fault``   - does it get the head *jump* across the barriers right?  A model
                 can have an excellent domain-wide RMSE and still smear both
                 faults into nothing, which is the failure mode that matters for
                 compartmentalised aquifers.
3. ``mass``    - does it conserve water, globally and locally?  Global closure is
                 easy to fake by symmetric errors; the local statistic is the
                 honest one.
4. ``river``   - does it get the aquifer-river exchange right, in total and cell
                 by cell, including its sign?  This is the quantity a surface
                 water licence is written against.
5. ``inverse`` - if K was unknown, how well was it recovered - in value, in
                 spatial pattern, and in *texture* (the variogram)?

Every metric is reported so that **lower is better**, and each criterion also
gets a single headline number so the summary table is readable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gwpinn.formulations.problem import Problem
from gwpinn.formulations.residuals import ControlVolume

CRITERIA = ("head", "fault", "mass", "river", "inverse")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _nse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency; 1 is perfect, 0 means "no better than the mean"."""
    denom = float(((ref - ref.mean()) ** 2).sum())
    return float(1.0 - ((pred - ref) ** 2).sum() / max(denom, 1e-12))


def _variogram(field: np.ndarray, dx: float, n_lags: int = 10,
               max_lag_cells: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Omnidirectional experimental variogram of a gridded field, by shifts."""
    lags, gam = [], []
    for lag in np.unique(np.linspace(1, max_lag_cells, n_lags).astype(int)):
        d = []
        if lag < field.shape[1]:
            d.append((field[:, lag:] - field[:, :-lag]).ravel())
        if lag < field.shape[0]:
            d.append((field[lag:, :] - field[:-lag, :]).ravel())
        if not d:
            continue
        diff = np.concatenate(d)
        lags.append(lag * dx)
        gam.append(0.5 * float(np.mean(diff ** 2)))
    return np.array(lags), np.array(gam)


# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #


def head_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """Criterion 1: transient head accuracy, ``(nt, nrow, ncol)`` arrays."""
    err = pred - ref
    # The transient signal is only ~1 m on top of a ~10 m spatial range, so a
    # model can score a fine absolute RMSE while getting none of the dynamics.
    # The amplitude error is what separates the two.
    amp_p = pred.max(0) - pred.min(0)
    amp_r = ref.max(0) - ref.min(0)
    return {
        "head_rmse_m": float(np.sqrt((err ** 2).mean())),
        "head_mae_m": float(np.abs(err).mean()),
        "head_max_err_m": float(np.abs(err).max()),
        "head_nse": _nse(pred.ravel(), ref.ravel()),
        "head_amp_rmse_m": float(np.sqrt(((amp_p - amp_r) ** 2).mean())),
        "head_final_rmse_m": float(np.sqrt(((pred[-1] - ref[-1]) ** 2).mean())),
    }


def fault_metrics(pred: np.ndarray, ref: np.ndarray,
                  fault_faces: np.ndarray) -> Dict[str, float]:
    """Criterion 2: the head discontinuity across each barrier."""
    out: Dict[str, float] = {}
    hyd = fault_faces[:, 4]
    for tag, mask in (("all", np.ones(len(hyd), bool)),
                      ("tight", hyd <= 1e-6),
                      ("leaky", hyd > 1e-6)):
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        r1 = fault_faces[idx, 0].astype(int); c1 = fault_faces[idx, 1].astype(int)
        r2 = fault_faces[idx, 2].astype(int); c2 = fault_faces[idx, 3].astype(int)
        jp = pred[:, r1, c1] - pred[:, r2, c2]
        jr = ref[:, r1, c1] - ref[:, r2, c2]
        out[f"fault_{tag}_jump_rmse_m"] = float(np.sqrt(((jp - jr) ** 2).mean()))
        # Recovered fraction of the true jump: 1.0 is perfect, 0 means the
        # barrier was smeared away completely.
        out[f"fault_{tag}_jump_recovery"] = float(
            np.abs(jp).mean() / max(np.abs(jr).mean(), 1e-9))
        out[f"fault_{tag}_ref_jump_m"] = float(np.abs(jr).mean())
        # The *evolution* of the jump, with the initial value removed.
        #
        # This exists because the hard initial condition hands every model the
        # jump that exists at t0 - here 2.25 m of the eventual 2.46 m - so the
        # raw numbers above mostly measure an inherited quantity, not a learned
        # one.  Subtracting each model's own t0 jump leaves only what it did
        # during the simulation, which on this case is a signal worth 0.29 m.
        out[f"fault_{tag}_evolution_rmse_m"] = float(
            np.sqrt((((jp - jp[0]) - (jr - jr[0])) ** 2).mean()))
        out[f"fault_{tag}_ref_evolution_m"] = float(np.abs(jr - jr[0]).mean())
    return out


def mass_metrics(model, prob: Problem, times: torch.Tensor,
                 n_times: int = 6) -> Dict[str, float]:
    """Criterion 3: water balance of the surrogate's own solution.

    The surrogate is asked the same question MODFLOW answers exactly: for each
    control volume, does storage change equal net lateral inflow plus sources?
    The residual is evaluated with the conservative finite-volume operator
    regardless of which formulation was trained, so the comparison is fair - a
    strong-form PINN is not allowed to grade its own homework with its own
    collocation points.
    """
    cv = ControlVolume(prob)
    prev_mode = prob.coeff_mode
    prob.coeff_mode = "nearest"
    idx = np.linspace(0, len(times) - 1, n_times).astype(int)

    local, glob, scale = [], [], []
    for i in idx:
        tv = float(times[i])
        x = cv.x.clone().requires_grad_(True)
        y = cv.y.clone().requires_grad_(True)
        t = torch.full_like(cv.x, tv).requires_grad_(True)
        h = model.head(x, y, t)
        K = model.k(x, y)
        b = prob.thickness(h, x, y)
        T = K * b
        S = prob.storage(h, x, y, b)
        dhdt = torch.autograd.grad(h, t, torch.ones_like(h), create_graph=False)[0]
        net = cv.cell_balance(h.detach(), T.detach(), b.detach())     # m3/d
        src = prob.sources(x, y, t, h)
        f = (src["recharge"] + src["wel"] + src["riv"]).view(cv.nrow, cv.ncol)

        sto = (S * dhdt).view(cv.nrow, cv.ncol).detach() * prob.area  # m3/d
        imbalance = sto - net - f.detach() * prob.area                 # m3/d per cell
        gross = (sto.abs() + net.abs() + (f.detach() * prob.area).abs())
        local.append(float(imbalance.abs().sum() / gross.abs().sum().clamp_min(1e-9)))
        glob.append(float(imbalance.sum()))
        scale.append(float(gross.sum()))

    prob.coeff_mode = prev_mode
    return {
        "mass_local_err_frac": float(np.mean(local)),
        "mass_global_err_frac": float(np.mean(np.abs(glob)) / max(np.mean(scale), 1e-9)),
        "mass_global_err_m3d": float(np.mean(np.abs(glob))),
    }


def river_metrics(model, prob: Problem, case, n_times: int = 12) -> Dict[str, float]:
    """Criterion 4: aquifer-river exchange against the reference RIV budget."""
    idx = np.linspace(0, case.nper - 1, n_times).astype(int)
    rows = torch.as_tensor(case.riv_cells[:, 0], device=prob.device)
    cols = torch.as_tensor(case.riv_cells[:, 1], device=prob.device)
    x = prob.xmin + (cols.to(prob.dtype) + 0.5) * prob.dx
    y = prob.ymax - (rows.to(prob.dtype) + 0.5) * prob.dy

    pred_tot, ref_tot, cellwise = [], [], []
    with torch.no_grad():
        for i in idx:
            tv = float(case.times[i])
            t = torch.full_like(x, tv)
            h = model.head(x, y, t)
            stage = float(case.riv_stage[i])
            h_eff = torch.maximum(h, torch.full_like(h, prob.riv_bottom))
            q = (case.riv_cond * (stage - h_eff)).cpu().numpy()   # m3/d per cell
            qr = case.riv_leakage[i]
            pred_tot.append(float(q.sum()))
            ref_tot.append(float(qr.sum()))
            cellwise.append(np.sqrt(np.mean((q - qr) ** 2)))

    pred_tot = np.array(pred_tot); ref_tot = np.array(ref_tot)
    denom = max(np.abs(ref_tot).mean(), 1e-9)
    sign_ok = float(np.mean(np.sign(pred_tot) == np.sign(ref_tot)))
    return {
        "river_total_rel_err": float(np.abs(pred_tot - ref_tot).mean() / denom),
        "river_cell_rmse_m3d": float(np.mean(cellwise)),
        "river_nse": _nse(pred_tot, ref_tot),
        "river_sign_agreement": sign_ok,
    }


def inverse_metrics(k_pred: np.ndarray, k_true: np.ndarray,
                    dx: float) -> Dict[str, float]:
    """Criterion 5: conductivity recovery, in value, pattern and texture."""
    lp = np.log10(np.clip(k_pred, 1e-8, None))
    lt = np.log10(np.clip(k_true, 1e-8, None))
    err = lp - lt

    # Texture: does the recovered field have the right variogram?  A smooth
    # field can have a decent RMSE and completely the wrong spatial statistics,
    # which is what makes it useless for transport or for uncertainty.
    lag_p, gam_p = _variogram(lp, dx)
    lag_t, gam_t = _variogram(lt, dx)
    sill_t = max(gam_t[-1], 1e-9)
    var_err = float(np.mean(np.abs(gam_p - gam_t)) / sill_t)

    cor = float(np.corrcoef(lp.ravel(), lt.ravel())[0, 1]) if lp.std() > 1e-9 else 0.0
    return {
        "k_log_rmse": float(np.sqrt((err ** 2).mean())),
        "k_log_bias": float(err.mean()),
        "k_pattern_corr": cor,
        "k_variogram_err": var_err,
        "k_sill_ratio": float(gam_p[-1] / sill_t),
    }


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #


def evaluate(model, prob: Problem, case, inverse: bool = False,
             n_times: Optional[int] = None) -> Dict[str, float]:
    """Score a trained surrogate on every criterion."""
    times = torch.as_tensor(case.times, dtype=prob.dtype, device=prob.device)
    pred = model.predict_grid(times).cpu().numpy()
    ref = case.head

    out: Dict[str, float] = {}
    out.update(head_metrics(pred, ref))
    out.update(fault_metrics(pred, ref, case.fault_faces))
    out.update(mass_metrics(model, prob, times))
    out.update(river_metrics(model, prob, case))
    if inverse:
        with torch.no_grad():
            k_pred = model.k_field.as_grid().cpu().numpy()
        out.update(inverse_metrics(k_pred, case.kh, case.delr))

    # Headline number per criterion, all "lower is better".
    out["score_head"] = out["head_rmse_m"]
    out["score_fault"] = out["fault_all_jump_rmse_m"]
    out["score_mass"] = out["mass_local_err_frac"]
    out["score_river"] = out["river_total_rel_err"]
    if inverse:
        out["score_inverse"] = out["k_log_rmse"]
    return out


# --------------------------------------------------------------------------- #
# The metric floor
# --------------------------------------------------------------------------- #


class ReferenceModel:
    """The MODFLOW solution itself, wrapped in the surrogate interface.

    Scoring the reference with the same code that scores the surrogates is the
    only way to know what a metric's *achievable* value is.  It is not zero: the
    reference is stored at stress-period ends, so ``dh/dt`` here is a
    piecewise-linear reconstruction rather than the backward-Euler slope MODFLOW
    actually used, and the storage switch is smoothed.  Those two approximations
    put a floor under the mass-balance criterion, and a surrogate that reaches it
    is perfect as far as this study can tell.
    """

    def __init__(self, prob: Problem, case) -> None:
        self.p = prob
        self.times = torch.as_tensor(case.times, dtype=prob.dtype, device=prob.device)
        self.h = prob.head_ref

    def head(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p = self.p
        tt = t.clamp(float(self.times[0]), float(self.times[-1]))
        idx = torch.searchsorted(self.times, tt.detach().contiguous())
        idx = idx.clamp(1, len(self.times) - 1)
        t0, t1 = self.times[idx - 1], self.times[idx]
        w = (tt - t0) / (t1 - t0)
        stack = torch.stack([p.interp(self.h[i], x, y) for i in range(len(self.times))])
        lo = stack.gather(0, (idx - 1).unsqueeze(0)).squeeze(0)
        hi = stack.gather(0, idx.unsqueeze(0)).squeeze(0)
        return lo + w * (hi - lo)

    def k(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.p.coeff(self.p.kh_true, x, y)

    @torch.no_grad()
    def predict_grid(self, times: torch.Tensor) -> torch.Tensor:
        return self.h


def null_scores(case) -> Dict[str, float]:
    """Score the *persistence* model: hold the head at its initial condition.

    This is the baseline every result must be read against, and on this case it
    is a strong one.  Because the initial condition is a hard constraint, every
    surrogate starts from the true head field - complete with the 2.25 m head
    jump already present across the barriers.  A model that learns nothing at
    all therefore scores a head RMSE of 0.91 m and a fault-jump RMSE of 0.56 m.

    Without this number the study would credit formulations for reproducing
    something they were handed.  With it, the question becomes the right one:
    does the surrogate extract any transient signal *beyond* the initial state?
    """
    ref = case.head
    null = np.repeat(case.head_init[None], len(ref), axis=0)
    ff = case.fault_faces.astype(int)
    jn = null[:, ff[:, 0], ff[:, 1]] - null[:, ff[:, 2], ff[:, 3]]
    jr = ref[:, ff[:, 0], ff[:, 1]] - ref[:, ff[:, 2], ff[:, 3]]
    return {
        "null_head_rmse_m": float(np.sqrt(((null - ref) ** 2).mean())),
        "null_head_nse": _nse(null.ravel(), ref.ravel()),
        "null_fault_jump_rmse_m": float(np.sqrt(((jn - jr) ** 2).mean())),
        "null_fault_jump_recovery": float(np.abs(jn).mean() / max(np.abs(jr).mean(), 1e-9)),
        "ref_transient_amplitude_m": float((ref.max(0) - ref.min(0)).mean()),
    }


def skill(score: float, null: float) -> float:
    """Skill score against the persistence baseline: 1 perfect, 0 no better, <0 worse."""
    return float(1.0 - score / max(null, 1e-12))


def reference_scores(prob: Problem, case, n_times: int = 4) -> Dict[str, float]:
    """Score the reference solution - the floor of every criterion."""
    model = ReferenceModel(prob, case)
    times = torch.as_tensor(case.times, dtype=prob.dtype, device=prob.device)
    out = {}
    out.update(mass_metrics(model, prob, times, n_times=n_times))
    out.update(river_metrics(model, prob, case, n_times=6))
    return out


__all__ = ["evaluate", "CRITERIA", "head_metrics", "fault_metrics",
           "mass_metrics", "river_metrics", "inverse_metrics",
           "ReferenceModel", "reference_scores", "null_scores", "skill"]
