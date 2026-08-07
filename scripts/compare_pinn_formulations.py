#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 WHICH PHYSICS-INFORMED FORMULATION FOR A HETEROGENEOUS, FAULTED, TIDAL AQUIFER?
================================================================================

A controlled screening study over the design space of physics-informed
surrogates, scored on the five things this problem actually needs:

    1. head simulation      2. head across faults      3. mass balance
    4. river-aquifer exchange                          5. inverse recovery of K

The reference is the MODFLOW 6 reduced case (``gwpinn.benchmark``): the same
equation, the same grid, the same heterogeneous K, the same meandering river and
the same two barriers, solved by a finite-volume code whose budget closes to
~1e-7 relative.  Every error reported therefore belongs to the surrogate.

Experimental design
-------------------
One factor at a time from a common baseline.  Comparing six named methods from
six papers confounds architecture with formulation with training schedule; this
design attributes each change to the axis that moved.  Axes:

    form      strong | mixed | fv          how the PDE is written
    arch      mlp | modified | pirate | spinn
    fault     none | smeared | sidefeat | faultcoord
    temporal  plain | causal | march
    balance   fixed | gradnorm | ntk | rba
    kfield    true | net | grid | kl       (inverse task only)

**Equal compute, not equal iterations.**  Per-iteration cost varies four-fold
across these variants, so every run gets the same wall-clock budget and the
number of iterations completed is reported as a result.  "What should I run for
the next N minutes" is the question a practitioner has.

Honest limitations, stated up front
-----------------------------------
* The budget (default 240 s on 4 CPU threads) is far short of PINN convergence.
  This measures **which formulation gets furthest per unit compute**, which is a
  well-defined and practically useful question, but it is *not* a measurement of
  asymptotic accuracy.  A method that starts slowly and finishes strongly would
  be ranked unfairly here; ``--budget`` exists to test that.
* Screening runs use one seed.  The top variants are then re-run with
  ``--seeds`` to check that the ranking survives initialisation noise.
* The case is a single layer.  Conclusions about vertical discretisation, and
  about the quasi-3D leakage terms in the full benchmark, are not tested.

Usage
-----
    python scripts/compare_pinn_formulations.py                    # full study
    python scripts/compare_pinn_formulations.py --budget 60        # quick pass
    python scripts/compare_pinn_formulations.py --only form,arch   # some axes
    python scripts/compare_pinn_formulations.py --seeds 3 --only best
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gwpinn.benchmark import build_reduced_case, load_reduced_case
from gwpinn.eval import evaluate
from gwpinn.eval.criteria import reference_scores
from gwpinn.formulations import Problem, Runner, Spec


# --------------------------------------------------------------------------- #
# Experimental design
# --------------------------------------------------------------------------- #

BASELINE = dict(
    form="strong", arch="modified", fault="smeared", temporal="plain",
    balance="gradnorm", width=64, depth=4, fourier=48, n_colloc=2048,
    iters=1_000_000,
)


def design(budget: float, seed: int = 0) -> List[Spec]:
    """The screening design: baseline plus one-factor-at-a-time variations."""
    def S(name: str, group: str, **kw) -> Spec:
        cfg = dict(BASELINE)
        cfg.update(kw)
        spec = Spec(name=name, seed=seed, max_seconds=budget, **cfg)
        spec.group = group          # type: ignore[attr-defined]
        spec.inverse = "kfield" in kw and kw["kfield"] != "true"  # type: ignore
        return spec

    runs: List[Spec] = [
        S("baseline", "baseline"),

        # --- how the PDE is written ----------------------------------------
        S("form:mixed", "form", form="mixed"),
        S("form:fv", "form", form="fv"),

        # --- network architecture ------------------------------------------
        S("arch:mlp", "arch", arch="mlp"),
        S("arch:pirate", "arch", arch="pirate", depth=3),
        S("arch:spinn", "arch", arch="spinn"),
        S("arch:mlp+rwf", "arch", arch="mlp", rwf=True),

        # --- fault representation ------------------------------------------
        S("fault:none", "fault", fault="none"),
        S("fault:sidefeat", "fault", fault="sidefeat"),
        S("fault:faultcoord", "fault", fault="faultcoord"),

        # --- temporal strategy ---------------------------------------------
        S("time:causal", "temporal", temporal="causal"),
        S("time:march", "temporal", temporal="march"),

        # --- loss balancing -------------------------------------------------
        S("bal:fixed", "balance", balance="fixed"),
        S("bal:ntk", "balance", balance="ntk"),
        S("bal:rba", "balance", balance="rba"),

        # --- promising combinations ----------------------------------------
        S("combo:fv+causal", "combo", form="fv", temporal="causal"),
        S("combo:fv+sidefeat", "combo", form="fv", fault="sidefeat"),
        S("combo:mixed+sidefeat", "combo", form="mixed", fault="sidefeat"),
        S("combo:fv+pirate", "combo", form="fv", arch="pirate", depth=3),

        # --- inverse problem: the K parameterisation is the axis ------------
        S("inv:net", "inverse", kfield="net"),
        S("inv:grid", "inverse", kfield="grid"),
        S("inv:kl", "inverse", kfield="kl"),
        S("inv:fv+kl", "inverse", form="fv", kfield="kl"),
        S("inv:fv+grid", "inverse", form="fv", kfield="grid"),
    ]
    return runs


def followup(budget: float, seed: int = 0) -> List[Spec]:
    """Follow-ups that check whether a screening conclusion is real.

    Two screening results deserve challenging before they are written down:

    * the mixed formulation was run with a mis-scaled flux head (it was asked to
      emit ~0.02 instead of ~1), so its numbers may say more about that than
      about the formulation.  Re-run with the flux scale derived from the
      reference;
    * causal weighting scored badly at ``eps = 1``.  The scheme is known to be
      sensitive to ``eps`` - too large and every bin after the first is frozen
      out - and reporting "causal weighting hurts" from a single untuned value
      would be a wrong negative.  Sweep it.
    """
    def S(name: str, **kw) -> Spec:
        cfg = dict(BASELINE)
        cfg.update(kw)
        spec = Spec(name=name, seed=seed, max_seconds=budget, **cfg)
        spec.group = "followup"     # type: ignore[attr-defined]
        spec.inverse = False        # type: ignore[attr-defined]
        return spec

    return [
        S("mixed:rescaled", form="mixed"),
        S("mixed+sidefeat:rescaled", form="mixed", fault="sidefeat"),
        S("causal:eps0.1", temporal="causal", causal_eps=0.1),
        S("causal:eps0.01", temporal="causal", causal_eps=0.01),
    ]


def budget_check(budget: float, seed: int = 0) -> List[Spec]:
    """The same two leaders at a much larger budget.

    The headline caveat of this study is that every run is far short of
    convergence, so the ranking could be an artefact of who starts fastest.
    Re-running the baseline and the control-volume form at 4x the budget is the
    direct test of that, and it also shows whether the mass-balance criterion -
    stuck near 0.8 against a floor of 0.079 - is budget-limited or structural.
    """
    def S(name: str, **kw) -> Spec:
        cfg = dict(BASELINE)
        cfg.update(kw)
        spec = Spec(name=name, seed=seed, max_seconds=budget, **cfg)
        spec.group = "budget"       # type: ignore[attr-defined]
        spec.inverse = False        # type: ignore[attr-defined]
        return spec

    return [S("budget:strong"), S("budget:fv", form="fv")]


DESIGNS = {"screen": design, "followup": followup, "budget": budget_check}


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def run_one(prob: Problem, case, spec: Spec, verbose: bool = False) -> Dict[str, object]:
    inverse = bool(getattr(spec, "inverse", False))
    if inverse and not prob.obs:
        prob.make_observations(n_wells=60, noise_m=0.02, n_prop=12, seed=7)
    runner = Runner(prob, spec, inverse=inverse, verbose=verbose).fit()
    metrics = evaluate(runner.model, prob, case, inverse=inverse)

    row: Dict[str, object] = {
        "name": spec.name,
        "group": getattr(spec, "group", ""),
        "form": spec.form, "arch": spec.arch, "fault": spec.fault,
        "temporal": spec.temporal, "balance": spec.balance, "kfield": spec.kfield,
        "seed": spec.seed,
        "iters": runner.iters_done,
        "seconds": round(runner.train_seconds, 1),
        "params": runner.n_params,
        "inverse": inverse,
    }
    row.update({k: float(v) for k, v in metrics.items()})
    return row, runner


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo = pathlib.Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(repo / "runs" / "mf6_hetero_benchmark"))
    ap.add_argument("--out", default=str(repo / "runs" / "pinn_formulation_study"))
    ap.add_argument("--budget", type=float, default=240.0,
                    help="Wall-clock seconds per run.")
    ap.add_argument("--only", default="",
                    help="Comma-separated group names to run (default: all).")
    ap.add_argument("--seeds", type=int, default=1,
                    help="Seeds per run; >1 repeats the design.")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--design", default="screen", choices=sorted(DESIGNS),
                    help="Which set of runs to execute.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = pathlib.Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    results_csv = out / "results.csv"

    # ---- case ---------------------------------------------------------------
    case_path = pathlib.Path(args.bench) / "reduced_case" / "case.npz"
    if case_path.exists():
        case = load_reduced_case(case_path)
        print(f"[case] loaded {case_path}")
    else:
        print("[case] building the reduced case (this runs MODFLOW twice)")
        case = build_reduced_case(args.bench)
        case.save(case_path)
    prob = Problem(case)
    print(f"[case] {case.nrow}x{case.ncol} cells, {case.nper} periods of "
          f"{case.dt * 24:g} h, K {case.kh.min():.3g}-{case.kh.max():.3g} m/d")
    print(f"[case] reference fault jump "
          f"{np.abs(case.head[-1][case.fault_faces[:, 0].astype(int), case.fault_faces[:, 1].astype(int)] - case.head[-1][case.fault_faces[:, 2].astype(int), case.fault_faces[:, 3].astype(int)]).mean():.2f} m")

    floor = reference_scores(prob, case)
    print("[floor] scoring the MODFLOW reference with the same metrics: "
          + ", ".join(f"{k}={v:.4g}" for k, v in floor.items()
                      if k in ("mass_local_err_frac", "river_total_rel_err")))
    (out / "metric_floor.json").write_text(json.dumps(floor, indent=2))

    # ---- resume -------------------------------------------------------------
    done: set = set()
    rows: List[Dict[str, object]] = []
    if results_csv.exists():
        prev = pd.read_csv(results_csv)
        rows = prev.to_dict("records")
        done = {(r["name"], r["seed"]) for r in rows}
        print(f"[resume] {len(done)} runs already recorded")

    groups = {g.strip() for g in args.only.split(",") if g.strip()}
    keep_runner: Dict[str, Runner] = {}
    t_all = time.time()

    for seed in range(args.seeds):
        for spec in DESIGNS[args.design](args.budget, seed=seed):
            if groups and getattr(spec, "group", "") not in groups:
                continue
            if (spec.name, seed) in done:
                continue
            print(f"\n=== {spec.name} (seed {seed}) ===")
            try:
                row, runner = run_one(prob, case, spec, verbose=args.verbose)
            except Exception as exc:  # noqa: BLE001 - one bad variant must not
                print(f"    FAILED: {exc.__class__.__name__}: {exc}")  # kill the study
                row = {"name": spec.name, "group": getattr(spec, "group", ""),
                       "seed": seed, "error": f"{exc.__class__.__name__}: {exc}"}
                runner = None
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_csv, index=False)
            if runner is not None:
                if "score_head" in row:
                    print(f"    head {row['score_head']:.3f} m | fault "
                          f"{row['score_fault']:.3f} m | mass {row['score_mass']:.3f} | "
                          f"river {row['score_river']:.3f} | {row['iters']} iters")
                if spec.name in ("baseline", "form:fv", "combo:fv+causal",
                                 "inv:kl", "inv:grid", "inv:net", "form:mixed"):
                    keep_runner[spec.name] = runner

    df = pd.DataFrame(rows)
    df.to_csv(results_csv, index=False)
    print(f"\n[done] {len(df)} runs in {(time.time() - t_all) / 60:.1f} min "
          f"-> {results_csv}")

    if not args.no_plots and "score_head" in df.columns:
        make_plots(df, case, prob, keep_runner, out / "figures")
    summarise(df, out)
    return 0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def summarise(df: pd.DataFrame, out: pathlib.Path) -> None:
    """Per-criterion winner tables, written to CSV and printed."""
    if "score_head" not in df.columns:
        return
    ok = df[df.get("error").isna()] if "error" in df.columns else df
    fwd = ok[~ok["inverse"].astype(bool)] if "inverse" in ok.columns else ok

    crit = {
        "head simulation": ("score_head", "m RMSE", fwd),
        "head across faults": ("score_fault", "m RMSE of jump", fwd),
        "mass balance": ("score_mass", "local imbalance frac", fwd),
        "river exchange": ("score_river", "relative error", fwd),
    }
    inv = ok[ok["inverse"].astype(bool)] if "inverse" in ok.columns else ok.iloc[0:0]
    if len(inv) and "score_inverse" in inv.columns:
        crit["inverse (K recovery)"] = ("score_inverse", "log10 RMSE", inv)

    lines = []
    for label, (col, unit, sub) in crit.items():
        if col not in sub.columns or sub[col].isna().all():
            continue
        s = sub.dropna(subset=[col]).sort_values(col)
        lines.append(f"\n### {label}  ({unit}, lower is better)")
        lines.append(s[["name", col, "iters"]].head(6).to_string(index=False))
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text)


def make_plots(df: pd.DataFrame, case, prob, runners: Dict[str, Runner],
               figdir: pathlib.Path) -> None:
    """Criterion bar charts plus qualitative head / fault / K comparisons."""
    ok = df[df["score_head"].notna()].copy()
    fwd = ok[~ok["inverse"].astype(bool)]

    # --- 1. one bar chart per criterion -------------------------------------
    crits = [("score_head", "Head RMSE (m)"),
             ("score_fault", "Fault jump RMSE (m)"),
             ("score_mass", "Local mass imbalance (fraction)"),
             ("score_river", "River exchange relative error")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, (col, label) in zip(axes.flat, crits):
        s = fwd.dropna(subset=[col]).sort_values(col)
        colors = ["#2ca02c" if n == s["name"].iloc[0] else
                  ("#7f7f7f" if n == "baseline" else "#1f77b4") for n in s["name"]]
        ax.barh(s["name"], s[col], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(label, fontsize=11, loc="left")
        if s[col].max() / max(s[col].min(), 1e-9) > 30:
            ax.set_xscale("log")
    fig.suptitle("Formulation screening at equal wall-clock budget "
                 "(green = best, grey = baseline)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figdir / "fig_criteria.png", dpi=140)
    plt.close(fig)

    # --- 2. head field comparison -------------------------------------------
    if runners:
        names = [n for n in ("baseline", "form:mixed", "form:fv", "combo:fv+causal")
                 if n in runners]
        if names:
            k = case.nper - 1
            ref = case.head[k]
            fig, axes = plt.subplots(2, len(names) + 1, figsize=(4.2 * (len(names) + 1), 8))
            axes = np.atleast_2d(axes)
            im = axes[0, 0].imshow(ref, extent=case.extent, origin="upper", cmap="viridis")
            axes[0, 0].set_title("MODFLOW reference", fontsize=10)
            fig.colorbar(im, ax=axes[0, 0], shrink=0.8)
            axes[1, 0].axis("off")
            times = torch.as_tensor(case.times, dtype=prob.dtype, device=prob.device)
            for j, n in enumerate(names, start=1):
                pred = runners[n].model.predict_grid(times).cpu().numpy()[k]
                a = axes[0, j].imshow(pred, extent=case.extent, origin="upper",
                                      cmap="viridis", vmin=ref.min(), vmax=ref.max())
                axes[0, j].set_title(n, fontsize=10)
                fig.colorbar(a, ax=axes[0, j], shrink=0.8)
                e = axes[1, j].imshow(pred - ref, extent=case.extent, origin="upper",
                                      cmap="RdBu_r", vmin=-2, vmax=2)
                axes[1, j].set_title(f"error (RMSE {np.sqrt(((pred-ref)**2).mean()):.2f} m)",
                                     fontsize=9)
                fig.colorbar(e, ax=axes[1, j], shrink=0.8)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"Head at the end of the window (t = {case.times[-1]:.2f} d)",
                         fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(figdir / "fig_head_fields.png", dpi=140)
            plt.close(fig)

    # --- 3. recovered K fields ----------------------------------------------
    inv_names = [n for n in ("inv:net", "inv:grid", "inv:kl") if n in runners]
    if inv_names:
        fig, axes = plt.subplots(1, len(inv_names) + 1, figsize=(4.5 * (len(inv_names) + 1), 4.4))
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(case.kh.min(), 1e-2), vmax=case.kh.max())
        im = axes[0].imshow(case.kh, extent=case.extent, origin="upper", cmap="turbo", norm=norm)
        axes[0].set_title("truth", fontsize=10)
        fig.colorbar(im, ax=axes[0], shrink=0.85)
        for ax, n in zip(axes[1:], inv_names):
            with torch.no_grad():
                kk = runners[n].model.k_field.as_grid().cpu().numpy()
            a = ax.imshow(kk, extent=case.extent, origin="upper", cmap="turbo", norm=norm)
            row = df[df["name"] == n].iloc[0]
            ax.set_title(f"{n}  (log RMSE {row.get('k_log_rmse', float('nan')):.2f})",
                         fontsize=10)
            fig.colorbar(a, ax=ax, shrink=0.85)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Inverse problem: recovered hydraulic conductivity", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(figdir / "fig_inverse_k.png", dpi=140)
        plt.close(fig)

    print(f"[plots] -> {figdir}")


if __name__ == "__main__":
    raise SystemExit(main())
