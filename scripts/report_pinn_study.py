#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the formulation study into a PDF report and a results markdown page.

Reads ``runs/pinn_formulation_study/results.csv`` (written by
``compare_pinn_formulations.py``) plus the metric floor and the figures, and
produces:

* ``docs/pinn_formulation_study.pdf`` - the report, text pages plus every figure
* ``docs/pinn_formulation_results.md`` - the results tables as markdown

Only reporting lives here; nothing is retrained, so this is cheap to re-run
after adding seeds or extending the design.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PAGE = (11.69, 8.27)
TEXT_WIDTH = 108
LINES_PER_PAGE = 50

CRITERIA: Tuple[Tuple[str, str, str, bool], ...] = (
    ("head simulation", "score_head", "head RMSE (m)", False),
    ("head across faults", "score_fault", "RMSE of the head jump (m)", False),
    ("mass balance", "score_mass", "local imbalance fraction", False),
    ("river exchange", "score_river", "relative error in total exchange", False),
    ("inverse (K recovery)", "score_inverse", "log10 RMSE of K", True),
)


# --------------------------------------------------------------------------- #
# Text rendering (shared style with the benchmark report)
# --------------------------------------------------------------------------- #


def _layout(blocks: Sequence[Tuple[str, str]]) -> List[str]:
    lines: List[str] = []
    for kind, text in blocks:
        if kind == "h":
            if lines:
                lines.append("")
            lines.append(text.upper())
            lines.append("-" * min(len(text), TEXT_WIDTH))
        elif kind == "p":
            for para in textwrap.dedent(text).strip("\n").split("\n\n"):
                collapsed = " ".join(para.split())
                if collapsed:
                    lines.extend("  " + ln for ln in
                                 textwrap.wrap(collapsed, width=TEXT_WIDTH - 2))
                    lines.append("")
        elif kind == "pre":
            lines.extend("  " + ln for ln in
                         textwrap.dedent(text).strip("\n").split("\n"))
            lines.append("")
    return lines


def _pages(title: str, blocks: Sequence[Tuple[str, str]], pdf: PdfPages) -> None:
    lines = _layout(blocks)
    n_pages = max(1, -(-len(lines) // LINES_PER_PAGE))
    per = max(1, -(-len(lines) // n_pages))
    for i in range(0, max(len(lines), 1), per):
        fig = plt.figure(figsize=PAGE)
        head = title if i == 0 else f"{title}  (continued)"
        fig.text(0.055, 0.95, head, fontsize=15, weight="bold", va="top")
        fig.text(0.055, 0.885, "\n".join(lines[i:i + per]), fontsize=7.4,
                 family="monospace", va="top", linespacing=1.4)
        pdf.savefig(fig)
        plt.close(fig)


def _image_page(path: pathlib.Path, pdf: PdfPages, caption: str = "") -> None:
    if not path.exists():
        return
    img = mpimg.imread(str(path))
    h, w = img.shape[:2]
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
    ax.imshow(img)
    ax.axis("off")
    if caption:
        fig.text(0.5, 0.025, caption, ha="center", fontsize=9, color="0.3")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def ranking(df: pd.DataFrame, col: str, inverse: bool, n: int = 8) -> pd.DataFrame:
    sub = df[df["inverse"].astype(bool)] if inverse else df[~df["inverse"].astype(bool)]
    if col not in sub.columns:
        return pd.DataFrame()
    sub = sub.dropna(subset=[col]).sort_values(col)
    keep = ["name", col, "iters"]
    for extra in ("head_nse", "fault_all_jump_recovery", "k_pattern_corr"):
        if extra in sub.columns and extra not in keep:
            keep.append(extra)
    return sub[keep].head(n)


def axis_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Change in each criterion relative to the baseline, per screening run."""
    fwd = df[~df["inverse"].astype(bool)]
    if "baseline" not in set(fwd["name"]):
        return pd.DataFrame()
    base = fwd[fwd["name"] == "baseline"].iloc[0]
    rows = []
    for _, r in fwd.iterrows():
        if r["name"] == "baseline":
            continue
        row = {"name": r["name"], "group": r.get("group", "")}
        for _, col, _, is_inv in CRITERIA:
            if is_inv or col not in fwd.columns or not np.isfinite(r.get(col, np.nan)):
                continue
            b = float(base[col])
            row[col.replace("score_", "")] = (float(r[col]) - b) / max(abs(b), 1e-12) * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(df: pd.DataFrame, floats: str = "%.4g") -> str:
    if df.empty:
        return "(no data)"
    return df.to_string(index=False, float_format=lambda v: floats % v)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo = pathlib.Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default=str(repo / "runs" / "pinn_formulation_study"))
    ap.add_argument("--pdf", default=str(repo / "docs" / "pinn_formulation_study.pdf"))
    ap.add_argument("--md", default=str(repo / "docs" / "pinn_formulation_results.md"))
    args = ap.parse_args(argv)

    study = pathlib.Path(args.study)
    df = pd.read_csv(study / "results.csv")
    if "error" in df.columns:
        failed = df[df["error"].notna()]
        df = df[df["error"].isna()]
    else:
        failed = pd.DataFrame()
    df["inverse"] = df["inverse"].astype(bool)

    floor = {}
    fp = study / "metric_floor.json"
    if fp.exists():
        floor = json.loads(fp.read_text())

    budget = float(df["seconds"].median()) if "seconds" in df else float("nan")
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------------- markdown ------------------------------------------------
    md = [
        "# Which physics-informed formulation wins on this benchmark?",
        "",
        "Results of the controlled screening study defined in "
        "[`pinn_formulation_design.md`](pinn_formulation_design.md) and run by "
        "`scripts/compare_pinn_formulations.py`.",
        "",
        f"- Reference: MODFLOW 6 reduced case, 100x100 cells, 48 stress periods of 2 h",
        f"- Budget: **{budget:.0f} s of wall clock per run** on 4 CPU threads "
        f"(equal compute, not equal iterations)",
        f"- Runs: {len(df)} completed" + (f", {len(failed)} failed" if len(failed) else ""),
        f"- Generated: {now}",
        "",
        "## Metric floors",
        "",
        "Every metric applied to the MODFLOW reference itself, so the numbers below "
        "are interpretable:",
        "",
    ]
    for k, v in floor.items():
        md.append(f"- `{k}` = **{v:.4g}**")
    md.append("")

    for label, col, unit, is_inv in CRITERIA:
        tab = ranking(df, col, is_inv)
        if tab.empty:
            continue
        md += [f"## {label}", "", f"_{unit}, lower is better_", "",
               tab.to_markdown(index=False, floatfmt=".4g"), ""]

    eff = axis_effects(df)
    if not eff.empty:
        md += ["## Effect of each axis relative to the baseline", "",
               "_percent change; negative is an improvement_", "",
               eff.to_markdown(index=False, floatfmt=".1f"), ""]

    pathlib.Path(args.md).write_text("\n".join(md))
    print(f"[md] {args.md}")

    # ---------------- pdf -----------------------------------------------------
    out_pdf = pathlib.Path(args.pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out_pdf)) as pdf:
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.70, "Which Physics-Informed Formulation?", ha="center",
                 fontsize=23, weight="bold")
        fig.text(0.5, 0.635, "A controlled screening study on the heterogeneous "
                 "faulted tidal aquifer benchmark", ha="center", fontsize=12.5,
                 color="0.3")
        summary = (
            f"Generated        {now}\n"
            f"Reference        MODFLOW 6, 100 x 100 cells, 48 stress periods of 2 h\n"
            f"Runs             {len(df)} completed"
            + (f", {len(failed)} failed\n" if len(failed) else "\n") +
            f"Budget           {budget:.0f} s wall clock per run, 4 CPU threads\n"
            f"Design           one factor at a time from a common baseline\n"
            f"Axes             form, arch, fault, temporal, balance, kfield\n"
        )
        for k, v in floor.items():
            summary += f"floor {k:<24s} {v:.4g}\n"
        fig.text(0.11, 0.50, summary, fontsize=9, family="monospace", va="top",
                 linespacing=1.75)
        pdf.savefig(fig)
        plt.close(fig)

        blocks: List[Tuple[str, str]] = [
            ("h", "How to read these tables"),
            ("p", """
                Every metric is "lower is better".  Each run got the same wall-clock
                budget, so the iteration count is a result, not a control: a formulation
                that is cheap per step gets more steps, and that is part of its merit.
                The floors listed on the title page are what the MODFLOW reference itself
                scores under the same metrics - a surrogate reaching the floor is perfect
                as far as this study can measure.
            """),
        ]
        for label, col, unit, is_inv in CRITERIA:
            tab = ranking(df, col, is_inv)
            if tab.empty:
                continue
            blocks += [("h", f"{label}  ({unit})"), ("pre", fmt(tab))]
        if not eff.empty:
            blocks += [("h", "Effect of each axis vs the baseline (% change, negative is better)"),
                       ("pre", fmt(eff, "%.1f"))]
        if len(failed):
            blocks += [("h", "Failed runs"),
                       ("pre", fmt(failed[["name", "error"]]))]
        _pages("Results", blocks, pdf)

        figs = study / "figures"
        _image_page(figs / "fig_criteria.png", pdf,
                    "Screening result per criterion; green = best, grey = baseline.")
        _image_page(figs / "fig_head_fields.png", pdf,
                    "Head field at the end of the window, and the error against MODFLOW.")
        _image_page(figs / "fig_inverse_k.png", pdf,
                    "Recovered hydraulic conductivity for each parameterisation.")

        meta = pdf.infodict()
        meta["Title"] = "Physics-informed formulation study - groundwater benchmark"
        meta["Creator"] = "report_pinn_study.py"

    print(f"[pdf] {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
