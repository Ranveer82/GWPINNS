"""Figures for the benchmark and the architecture comparison.

Colour policy
-------------
* The three architectures are a **categorical** encoding, so they get a fixed
  hue order that never cycles and never depends on rank.  The palette is
  colourblind-safe (validated: worst adjacent deuteranope separation dE 11.0,
  normal-vision 25.8, contrast >3:1 on both light and dark surfaces).
* Ground truth is not a peer series -- it is the reference -- so it is drawn as
  neutral ink, dashed, and is always direct-labelled.
* ``log10 K`` and head are **magnitude**, so they use a single-hue sequential
  ramp, never a rainbow.
* Prediction-minus-truth is **polarity**, so it uses a two-hue diverging ramp
  with a neutral grey midpoint, symmetric about zero.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from ..config import BenchmarkConfig

__all__ = [
    "ARCH_COLORS",
    "ARCH_LABELS",
    "SEQUENTIAL",
    "DIVERGING",
    "plot_benchmark_overview",
    "plot_conductivity_comparison",
    "plot_fault_transect",
    "plot_fault_summary",
    "plot_training_history",
    "plot_head_timeseries",
]

# Fixed categorical order -- assigned by identity, never by rank.
ARCH_COLORS = {
    "baseline": "#0072B2",
    "mixed": "#D55E00",
    "cpinn": "#009E73",
}
ARCH_LABELS = {
    "baseline": "Baseline PINN",
    "mixed": "Mixed-variable PINN",
    "cpinn": "cPINN (decomposed)",
}
TRUTH_INK = "#3A3A3A"
MUTED_INK = "#6B6B6B"

# Single-hue sequential ramp (light -> dark), monotone in lightness.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "gw_sequential",
    ["#F2F7FB", "#CDE1F0", "#9CC4E0", "#6BA3CC", "#3F7FB3", "#1F5B8F", "#0D3A62"],
)
# Two-hue diverging with a neutral grey midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "gw_diverging",
    ["#7B3200", "#C06A20", "#E0A970", "#E8E8E6", "#7FB6AE", "#2E8577", "#0B4F45"],
)


def _style_axes(ax) -> None:
    """Recessive grid and axes; the data should be the loudest thing present."""
    ax.set_facecolor("#FCFCFB")
    ax.grid(True, color="#E3E3E1", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#C8C8C6")
    ax.tick_params(colors=MUTED_INK, labelsize=8)


def _fault_trace(cfg: BenchmarkConfig, ax, color: str = "#B00020") -> None:
    """Overlay the fault-zone walls on a map view."""
    grid, fault = cfg.grid, cfg.fault
    y = np.linspace(0.0, grid.Ly, 50)
    nx, ny = fault.normal
    for offset, style in ((-0.5 * fault.width, "--"), (0.5 * fault.width, "--")):
        x = fault.x0 + (offset - (y - fault.y0) * ny) / nx
        ax.plot(x, y, style, color=color, linewidth=1.2, zorder=6)


def plot_benchmark_overview(
    cfg: BenchmarkConfig,
    k_true: np.ndarray,
    heads: np.ndarray,
    times: np.ndarray,
    obs,
    path: str | Path,
) -> Path:
    """True conductivity, the fault, the stress network and the head response."""
    grid = cfg.grid
    extent = [0, grid.Lx, 0, grid.Ly]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    log_k = np.log10(k_true)
    vmin, vmax = log_k.min(), log_k.max()

    for col, layer in enumerate(range(grid.nlay)):
        ax = axes[0, col]
        im = ax.imshow(
            log_k[layer], origin="upper", extent=extent, cmap=SEQUENTIAL,
            vmin=vmin, vmax=vmax, aspect="auto",
        )
        _fault_trace(cfg, ax)
        ax.set_title(f"log₁₀ K — layer {layer}", fontsize=10, color=TRUTH_INK)
        ax.set_xlabel("x (m)", fontsize=8, color=MUTED_INK)
        if col == 0:
            ax.set_ylabel("y (m)", fontsize=8, color=MUTED_INK)
        ax.tick_params(colors=MUTED_INK, labelsize=8)
    cbar = fig.colorbar(im, ax=axes[0, :], shrink=0.85, pad=0.01)
    cbar.set_label("log₁₀ K (m/d)", fontsize=8, color=MUTED_INK)
    cbar.ax.tick_params(colors=MUTED_INK, labelsize=8)

    # Head at the final time, layer 0.
    ax = axes[1, 0]
    im = ax.imshow(
        heads[-1, 0], origin="upper", extent=extent, cmap=SEQUENTIAL, aspect="auto"
    )
    _fault_trace(cfg, ax)
    ax.set_title(f"head at t = {times[-1]:.0f} d — layer 0", fontsize=10, color=TRUTH_INK)
    ax.set_xlabel("x (m)", fontsize=8, color=MUTED_INK)
    ax.set_ylabel("y (m)", fontsize=8, color=MUTED_INK)
    ax.tick_params(colors=MUTED_INK, labelsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("head (m)", fontsize=8, color=MUTED_INK)
    cb.ax.tick_params(colors=MUTED_INK, labelsize=8)

    # Drawdown relative to the steady-state spin-up.
    ax = axes[1, 1]
    drawdown = heads[-1, 0] - heads[0, 0]
    limit = float(np.max(np.abs(drawdown))) or 1.0
    im = ax.imshow(
        drawdown, origin="upper", extent=extent, cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), aspect="auto",
    )
    _fault_trace(cfg, ax)
    ax.set_title("head change from steady state — layer 0", fontsize=10, color=TRUTH_INK)
    ax.set_xlabel("x (m)", fontsize=8, color=MUTED_INK)
    ax.tick_params(colors=MUTED_INK, labelsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("Δ head (m)", fontsize=8, color=MUTED_INK)
    cb.ax.tick_params(colors=MUTED_INK, labelsize=8)

    # Stress and monitoring network.
    ax = axes[1, 2]
    _style_axes(ax)
    _fault_trace(cfg, ax)
    unique = {}
    for name, x, y in zip(obs.name, obs.x, obs.y):
        unique[str(name)] = (x, y)
    xs = [v[0] for v in unique.values()]
    ys = [v[1] for v in unique.values()]
    ax.scatter(xs, ys, s=26, facecolor="#FCFCFB", edgecolor="#0072B2",
               linewidth=1.4, zorder=5, label="monitoring wells")
    for well in cfg.wells:
        ax.scatter([well.x], [well.y], s=90, marker="v", color="#D55E00",
                   edgecolor="#FCFCFB", linewidth=1.5, zorder=6)
        ax.annotate(well.name, (well.x, well.y), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color=TRUTH_INK)
    ax.scatter([], [], s=90, marker="v", color="#D55E00", label="pumping wells")
    ax.plot([], [], "--", color="#B00020", label="fault zone")
    ax.set_xlim(0, grid.Lx)
    ax.set_ylim(0, grid.Ly)
    ax.set_title("sources, sinks and monitoring network", fontsize=10, color=TRUTH_INK)
    ax.set_xlabel("x (m)", fontsize=8, color=MUTED_INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED_INK, loc="upper left")

    fig.suptitle(
        f"Benchmark — fault as {cfg.scenario} (K_fault = {cfg.fault_k:g} m/d)",
        fontsize=12, color=TRUTH_INK,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def plot_conductivity_comparison(
    cfg: BenchmarkConfig,
    k_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: str | Path,
    layer: int = 0,
) -> Path:
    """True vs recovered ``log10 K`` for every architecture, on one layer."""
    grid = cfg.grid
    extent = [0, grid.Lx, 0, grid.Ly]
    n = len(predictions) + 1
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), constrained_layout=True)
    fig.patch.set_facecolor("white")

    log_true = np.log10(k_true[layer])
    vmin, vmax = log_true.min(), log_true.max()

    ax = axes[0]
    im = ax.imshow(log_true, origin="upper", extent=extent, cmap=SEQUENTIAL,
                   vmin=vmin, vmax=vmax, aspect="auto")
    _fault_trace(cfg, ax)
    ax.set_title("truth", fontsize=10, color=TRUTH_INK)
    ax.set_ylabel("y (m)", fontsize=8, color=MUTED_INK)
    ax.tick_params(colors=MUTED_INK, labelsize=8)

    for ax, (name, k_pred) in zip(axes[1:], predictions.items()):
        im = ax.imshow(np.log10(k_pred[layer]), origin="upper", extent=extent,
                       cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, aspect="auto")
        _fault_trace(cfg, ax)
        ax.set_title(ARCH_LABELS.get(name, name), fontsize=10,
                     color=ARCH_COLORS.get(name, TRUTH_INK))
        ax.set_xlabel("x (m)", fontsize=8, color=MUTED_INK)
        ax.tick_params(colors=MUTED_INK, labelsize=8)

    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("log₁₀ K (m/d)", fontsize=8, color=MUTED_INK)
    cb.ax.tick_params(colors=MUTED_INK, labelsize=8)
    fig.suptitle(
        f"Recovered conductivity — layer {layer}, {cfg.scenario} scenario",
        fontsize=12, color=TRUTH_INK,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def plot_fault_transect(
    cfg: BenchmarkConfig,
    k_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: str | Path,
    layer: int = 0,
    band: float = 1200.0,
) -> Path:
    """``log10 K`` against signed fault distance -- the sharpness test.

    Everything is binned on distance to the fault plane, so the oblique strike
    does not smear the profile.  This is the figure that shows whether an
    architecture reproduces a step or a slope.
    """
    x, y, _ = cfg.grid.cell_center_arrays()
    dist = cfg.fault.signed_distance(x, y)[layer]
    keep = np.abs(dist) <= band
    edges = np.linspace(-band, band, 41)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def profile(field: np.ndarray) -> np.ndarray:
        values = np.log10(field[layer])[keep]
        d = dist[keep]
        idx = np.digitize(d, edges) - 1
        out = np.full(centers.size, np.nan)
        for b in range(centers.size):
            sel = idx == b
            if sel.any():
                out[b] = values[sel].mean()
        return out

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    _style_axes(ax)

    half = 0.5 * cfg.fault.width
    ax.axvspan(-half, half, color="#F0DADA", zorder=1, label="fault zone")

    ax.plot(centers, profile(k_true), "--", color=TRUTH_INK, linewidth=2.0,
            zorder=5, label="truth")
    for name, k_pred in predictions.items():
        ax.plot(centers, profile(k_pred), "-", color=ARCH_COLORS.get(name, MUTED_INK),
                linewidth=2.0, zorder=4, label=ARCH_LABELS.get(name, name))

    ax.set_xlabel("signed distance from fault plane (m)", fontsize=9, color=MUTED_INK)
    ax.set_ylabel("log₁₀ K (m/d)", fontsize=9, color=MUTED_INK)
    ax.set_title(
        f"Fault-normal conductivity profile — layer {layer}, {cfg.scenario} scenario",
        fontsize=11, color=TRUTH_INK,
    )
    ax.legend(fontsize=9, frameon=False, labelcolor=MUTED_INK, loc="best")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def plot_fault_summary(results: dict, path: str | Path) -> Path:
    """Inferred fault conductivity per architecture, against the truth.

    Log scale, because the quantity spans five orders of magnitude between the
    two scenarios; every bar is direct-labelled so the value never has to be
    read off the axis.
    """
    scenarios = list(results.keys())
    fig, axes = plt.subplots(
        1, len(scenarios), figsize=(6.0 * len(scenarios), 4.4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for ax, scenario in zip(axes, scenarios):
        _style_axes(ax)
        entries = results[scenario]
        names = list(entries.keys())
        values = [max(entries[n]["k_fault_pred_m_per_d"], 1e-6) for n in names]
        truth = entries[names[0]]["k_fault_true_m_per_d"]
        background = entries[names[0]]["k_background_true_m_per_d"]

        positions = np.arange(len(names))
        for pos, name, value in zip(positions, names, values):
            ax.bar(pos, value, width=0.62, color=ARCH_COLORS.get(name, MUTED_INK),
                   zorder=4, edgecolor="#FCFCFB", linewidth=2.0)
            ax.annotate(f"{value:.3g}", (pos, value), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=9, color=TRUTH_INK)

        # Reference lines are annotated in axes fractions so the labels cannot
        # be clipped by the data limits.
        ax.axhline(truth, linestyle="--", color=TRUTH_INK, linewidth=1.8, zorder=5)
        ax.annotate(
            f"true K_fault = {truth:g}", xy=(0.99, truth),
            xycoords=ax.get_yaxis_transform(), textcoords="offset points",
            xytext=(0, 4), ha="right", fontsize=9, color=TRUTH_INK,
            bbox=dict(facecolor="#FCFCFB", edgecolor="none", pad=1.0),
        )
        ax.axhline(background, linestyle=":", color=MUTED_INK, linewidth=1.5, zorder=5)
        ax.annotate(
            f"background K = {background:.2g}", xy=(0.01, background),
            xycoords=ax.get_yaxis_transform(), textcoords="offset points",
            xytext=(0, 4), ha="left", fontsize=9, color=MUTED_INK,
            bbox=dict(facecolor="#FCFCFB", edgecolor="none", pad=1.0),
        )

        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels([ARCH_LABELS.get(n, n) for n in names], fontsize=9,
                           color=MUTED_INK, rotation=12, ha="right")
        ax.set_ylabel("inferred K_fault (m/d)", fontsize=9, color=MUTED_INK)
        ax.set_title(f"{scenario} scenario", fontsize=11, color=TRUTH_INK)

    fig.suptitle("Fault characterisation: inferred vs true fault conductivity",
                 fontsize=12, color=TRUTH_INK)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def plot_training_history(histories: dict, path: str | Path) -> Path:
    """Total objective and data misfit against iteration, per architecture."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, key, title in (
        (axes[0], "total", "total objective"),
        (axes[1], "data", "head data misfit"),
    ):
        _style_axes(ax)
        for name, history in histories.items():
            iterations = history["iterations"]
            values = [r.get(key, np.nan) for r in history["records"]]
            ax.plot(iterations, values, color=ARCH_COLORS.get(name, MUTED_INK),
                    linewidth=1.8, zorder=4, label=ARCH_LABELS.get(name, name))
        ax.set_yscale("log")
        ax.set_xlabel("Adam iteration", fontsize=9, color=MUTED_INK)
        ax.set_ylabel(title, fontsize=9, color=MUTED_INK)
        ax.set_title(title, fontsize=11, color=TRUTH_INK)
    axes[0].legend(fontsize=9, frameon=False, labelcolor=MUTED_INK, loc="best")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def plot_head_timeseries(
    cfg: BenchmarkConfig,
    obs,
    times: np.ndarray,
    true_heads: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: str | Path,
    n_wells: int = 4,
) -> Path:
    """Head histories at a few monitoring wells: truth, data and each model."""
    grid = cfg.grid
    names = list(dict.fromkeys(str(n) for n in obs.name))[:n_wells]
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.0 * len(names), 3.8), constrained_layout=True, sharex=True
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for ax, name in zip(axes, names):
        _style_axes(ax)
        sel = obs.name == name
        layer = int(obs.layer[sel][0])
        row, col = grid.locate(float(obs.x[sel][0]), float(obs.y[sel][0]))

        ax.plot(times, true_heads[:, layer, row, col], "--", color=TRUTH_INK,
                linewidth=2.0, zorder=5, label="truth")
        ax.scatter(obs.t[sel], obs.head_obs[sel], s=12, facecolor="#FCFCFB",
                   edgecolor=MUTED_INK, linewidth=0.8, zorder=6, label="observations")
        for arch, heads in predictions.items():
            ax.plot(times, heads[:, layer, row, col], "-",
                    color=ARCH_COLORS.get(arch, MUTED_INK), linewidth=1.8,
                    zorder=4, label=ARCH_LABELS.get(arch, arch))

        ax.set_title(name, fontsize=10, color=TRUTH_INK)
        ax.set_xlabel("time (d)", fontsize=8, color=MUTED_INK)
    axes[0].set_ylabel("head (m)", fontsize=9, color=MUTED_INK)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=MUTED_INK, loc="best")

    fig.suptitle(f"Head reproduction at monitoring wells — {cfg.scenario} scenario",
                 fontsize=12, color=TRUTH_INK)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path
