#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 HETEROGENEOUS TRANSIENT MODFLOW 6 BENCHMARK FOR ML / PINN SURROGATE MODELS
================================================================================

This script builds, runs, post-processes and *documents* a deliberately hostile
conceptual groundwater model.  It is not a calibrated representation of a real
basin - it is a **strict benchmark** for machine-learning surrogates (PINNs,
FNOs, CNN-LSTM emulators, DeepONets, ...) covering both the **forward
(simulation)** problem and the **inverse (parameter estimation)** problem.

Why every design choice below makes life hard for a surrogate
-------------------------------------------------------------
1.  SHORT-RANGE HETEROGENEITY.  The geostatistical correlation length is
    200 m = 2 cells.  A surrogate cannot get away with learning a smooth,
    low-rank K-field; it has to resolve near-cell-scale structure.
2.  SHARP INTERNAL DISCONTINUITIES.  Two Horizontal-Flow-Barrier faults cut all
    five layers.  One is effectively impermeable (1e-8 d^-1) and produces a head
    *jump* across a single cell face - a genuine discontinuity that smooth
    neural representations (and PINN residual losses) struggle with.
3.  CURVILINEAR, NON-GRID-ALIGNED FEATURES.  Both watercourses meander.  The
    incised valley is a sinuous alluvial corridor that follows the stream, not a
    rectangle, so Layer 1's active domain is an irregular, non-convex mask.  A
    surrogate cannot exploit axis-aligned structure.
4.  REALISTIC 3-D GEOMETRY.  Land surface, the base of the alluvium and all four
    deeper stratigraphic contacts are undulating surfaces, so layer thickness -
    and therefore transmissivity and storativity - varies smoothly on top of the
    short-range K noise.  Cell geometry is an input the surrogate must consume.
5.  FAST, MULTI-SCALE FORCING.  360 stress periods of exactly 2 hours each.  The
    tidal river carries a 12.4 h semi-diurnal signal (resolved by only ~6 stress
    periods per cycle), pumping follows a diurnal cycle with abrupt on/off
    events, recharge has two 12-h convective storms and the stream carries
    flash-flood hydrographs.  Temporal aliasing is a real risk.
6.  MIXED BOUNDARY PHYSICS.  RIV (head-dependent, time-varying stage), SFR
    (fully routed surface water with its own stage-discharge non-linearity),
    GHB (regional inflow), WEL (point sinks), RCH (areal source).
7.  UNCONFINED / CONFINED SWITCHING.  Layers 1-2 are convertible, so storage
    coefficients change in time and the PDE is non-linear.

Known behaviour of the prescribed conceptual model
--------------------------------------------------
The prescribed base recharge (5e-4 m/d over 100 km^2 = 50 000 m^3/d) is large
relative to what the prescribed transmissivities can route to the only two
outlets - the tidal river and the central stream.  In the most remote,
fault-compartmentalised parts of the plateau the water table therefore reaches
land surface, and because MODFLOW switches those cells to *confined* storage
above the layer top, the two 12-hour storms produce head spikes of several
metres there.  This is a property of the prescribed parameter set, not a
numerical failure: the run converges with a 0.00 % volumetric budget
discrepancy, and ``report_benchmark_diagnostics`` prints the exceedance every
time.  If a strictly sub-surface water table is wanted, either lower
``Config.rch_base`` or add a DRN "spring/seepage" package at land surface.

Outputs (all written under the run workspace)
---------------------------------------------
  * A complete MODFLOW 6 simulation (written, run and mass-balance checked)
  * ``gis/*.tif``   - GeoTIFF rasters of every static input field (K, Ss, Sy,
                      topography, layer tops/bottoms/thicknesses, idomain,
                      distance-to-feature covariates) **and** every simulated
                      state (per-layer multi-band head time stacks, water table,
                      drawdown, specific discharge, temporal statistics).
  * ``gis/*.shp``   - ESRI Shapefiles of the meandering river and stream
                      centrelines, the alluvial corridor, faults, all boundary
                      condition cells, wells and observation points.
  * ``tables/*.csv``- Every boundary-condition time series, stress period by
                      stress period, plus the stream long-profile and the
                      observation hydrographs.
  * ``arrays/*.npz``- The full head tensor (nper, nlay, nrow, ncol) as float32,
                      i.e. the surrogate's training target.
  * ``figures/*.png`` - Diagnostic plots.
  * ``report/*.pdf``  - A full model report with every plot and table
                      (also copied to ``docs/`` in the repository).

Usage
-----
    python scripts/build_mf6_hetero_benchmark.py                 # everything
    python scripts/build_mf6_hetero_benchmark.py --nper 24       # fast smoke test
    python scripts/build_mf6_hetero_benchmark.py --no-run        # build + export only
    python scripts/build_mf6_hetero_benchmark.py --seed 12345    # new realisation

Dependencies: flopy, numpy, pandas, geopandas, rasterio, shapely, gstools,
scipy, matplotlib.  The MODFLOW 6 executable is resolved automatically.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import textwrap
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import flopy
import geopandas as gpd
import gstools as gs
import rasterio
from flopy.utils.postprocessing import get_specific_discharge
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from rasterio import features as rio_features
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, Point, Polygon, box, shape as shapely_shape
from shapely.ops import unary_union

# =============================================================================
# SECTION 0 - GLOBAL CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Every knob of the benchmark in one immutable place."""

    # ---------------- simulation identity ----------------
    sim_name: str = "hetbench"
    model_name: str = "hetbench"

    # ---------------- spatial discretisation ----------------
    nlay: int = 5
    nrow: int = 100
    ncol: int = 100
    delr: float = 100.0  # column width  (x, m)
    delc: float = 100.0  # row height    (y, m)

    # Synthetic georeference.  A real projected CRS is used so the exported
    # GeoTIFFs/Shapefiles line up in any GIS and so distance-based feature
    # engineering (metres) is meaningful.  UTM 44N, arbitrary origin.
    crs_epsg: int = 32644
    x_origin: float = 500_000.0  # model (0,0) lower-left easting
    y_origin: float = 3_000_000.0  # model (0,0) lower-left northing

    # ---------------- meandering central stream ----------------
    # The stream is a two-harmonic meander train about column 50.  Two
    # incommensurate wavelengths give an irregular, non-repeating planform.
    stream_col_mean: float = 50.0
    stream_amp1: float = 6.5
    stream_wave1: float = 42.0  # rows per meander
    stream_phase1: float = 0.4
    stream_amp2: float = 2.6
    stream_wave2: float = 17.0
    stream_phase2: float = 1.9

    # ---------------- meandering tidal river ----------------
    river_row_mean: float = 92.0
    river_amp1: float = 4.2
    river_wave1: float = 61.0  # columns per meander
    river_phase1: float = 1.1
    river_amp2: float = 1.8
    river_wave2: float = 23.0
    river_phase2: float = 0.3
    river_row_min: int = 84
    river_row_max: int = 98

    # ---------------- topography ----------------
    # The plateau sits well above the drainage lines and the watercourses are
    # deeply incised.  Those two numbers are coupled: raising the interfluves
    # while deepening the incision by the same amount leaves every *drainage*
    # elevation (streambed, tidal stage) untouched, so the head field is
    # unchanged, but it gives the water table the headroom it needs to stay
    # below ground on the interfluves.  Shallower relief here makes the
    # prescribed recharge emerge at surface over much of the plateau.
    plateau_north: float = 58.0  # mean land surface at row 0
    plateau_south: float = 50.0  # mean land surface at row nrow-1
    topo_undulation_var: float = 3.2  # m^2 -> ~1.8 m std, long wavelength
    topo_undulation_len: float = 2000.0
    topo_roughness_var: float = 0.25  # m^2 -> 0.5 m std, short wavelength
    topo_roughness_len: float = 400.0
    valley_incision: float = 9.0  # m below the plateau at the valley axis
    valley_incision_width: float = 900.0  # e-folding half width (m)
    river_incision: float = 6.0  # m below the plateau at the river axis
    river_incision_width: float = 600.0

    # ---------------- alluvial fill (Layer 1) ----------------
    alluv_thick_north: float = 10.0  # axis thickness at row 0
    alluv_thick_south: float = 18.0  # axis thickness at the confluence
    alluv_width: float = 800.0  # e-folding half width about the stream
    floodplain_thick: float = 12.0  # axis thickness of the tidal floodplain
    floodplain_width: float = 500.0
    alluv_min_thick: float = 2.0  # below this Layer 1 is switched off

    # ---------------- deeper stratigraphic contacts ----------------
    # Mean elevation of each layer base plus an undulating (paleo-)surface.
    botm_l2_north: float = 20.0
    botm_l2_south: float = 16.0
    botm_l2_var: float = 9.0  # m^2 -> 3 m std
    botm_l2_len: float = 2500.0
    botm_l3_mean: float = 0.0
    botm_l3_var: float = 16.0  # 4 m std
    botm_l3_len: float = 3000.0
    botm_l4_mean: float = -50.0
    botm_l4_var: float = 36.0  # 6 m std
    botm_l4_len: float = 3500.0
    botm_l5_mean: float = -150.0
    botm_l5_var: float = 64.0  # 8 m std
    botm_l5_len: float = 4000.0
    min_thickness: Tuple[float, ...] = (0.0, 3.0, 5.0, 5.0, 5.0)

    # ---------------- temporal discretisation ----------------
    nper: int = 360  # 360 x 2 h = 30 days
    dt_days: float = 2.0 / 24.0  # exactly 2 hours

    # ---------------- geostatistics ----------------
    # Very short range + high variance => extreme, near-cell-scale heterogeneity.
    len_scale: float = 200.0  # m  (= 2 cells!)
    var_lnk: float = 2.0  # variance of ln(K)
    var_lns: float = 2.0  # variance of ln(Ss)
    seed: int = 20260805

    # Geometric-mean horizontal K per layer (m/d)
    kh_mean: Tuple[float, ...] = (15.0, 5.0, 25.0, 1.0, 0.05)
    kz_ratio: float = 10.0  # Kz = Kh / 10
    ss_mean: Tuple[float, ...] = (2.0e-4, 1.0e-4, 5.0e-5, 3.0e-5, 1.0e-5)
    sy_mean: Tuple[float, ...] = (0.22, 0.15, 0.08, 0.05, 0.02)
    sy_bounds: Tuple[float, float] = (0.01, 0.38)  # physical clamp for Sy
    channel_facies_gain: float = 1.5  # Layer 1 K uplift along the palaeo-channel
    channel_facies_width: float = 500.0

    # ---------------- tidal river (RIV) ----------------
    riv_mean_stage: float = 40.0
    riv_amplitude: float = 2.5
    riv_period_h: float = 12.4  # semi-diurnal M2
    riv_spring_neap_d: float = 14.77  # amplitude modulation period
    riv_spring_neap_frac: float = 0.18
    riv_bottom: float = 36.0
    riv_bed_k: float = 1.0  # m/d
    riv_bed_thick: float = 1.0  # m
    riv_width: float = 80.0  # m

    # ---------------- central stream (SFR) ----------------
    sfr_incision: float = 1.0  # streambed below the valley floor (m)
    sfr_min_grad: float = 1.0e-4
    sfr_width: float = 15.0
    sfr_bed_thick: float = 1.0
    sfr_bed_k: float = 0.5
    sfr_manning: float = 0.030
    sfr_base_inflow: float = 4_000.0  # m3/d

    # ---------------- eastern regional inflow (GHB) ----------------
    ghb_col: int = 99
    ghb_layers: Tuple[int, ...] = (1, 2, 3)  # 0-based -> model layers 2,3,4
    ghb_head: float = 48.0

    # ---------------- faults (HFB) ----------------
    fault1: Tuple[int, int, int, int] = (10, 10, 40, 35)  # r0,c0,r1,c1
    fault1_hydchr: float = 1.0e-8  # effectively impermeable
    fault2: Tuple[int, int, int, int] = (85, 99, 65, 65)
    fault2_hydchr: float = 1.0e-3  # semi-permeable / leaky

    # ---------------- recharge ----------------
    rch_base: float = 5.0e-4  # m/d
    storm_rate: float = 6.0e-2  # m/d  (=30 mm over a 12 h event)
    storm_windows_d: Tuple[Tuple[float, float], ...] = ((6.0, 6.5), (20.5, 21.0))

    # ---------------- initial condition ----------------
    strt_fallback: float = 45.0

    # ---------------- output control ----------------
    budget_every_nper: int = 12  # save full budget once per day (12 x 2 h)


CFG = Config()

# 8 production wells, all screened in Layer 2 (0-based layer index 1) inside the
# alluvial corridor.  ``base`` is the mean daily abstraction in m3/d.
# ``phase_h`` shifts each well's diurnal peak so the composite stress is not a
# single clean harmonic.  ``start_d`` / ``stop_d`` create abrupt step changes
# mid-simulation (a well switched on at day 12, another shut down at day 20)
# which are the hardest events for a time-series surrogate to anticipate.
WELLS: Tuple[dict, ...] = (
    dict(name="W1", row=12, col=45, base=520.0, phase_h=0.0, start_d=0.0, stop_d=30.0),
    dict(name="W2", row=22, col=55, base=760.0, phase_h=1.5, start_d=0.0, stop_d=30.0),
    dict(name="W3", row=35, col=44, base=430.0, phase_h=-2.0, start_d=0.0, stop_d=30.0),
    dict(name="W4", row=46, col=57, base=880.0, phase_h=0.5, start_d=12.0, stop_d=30.0),
    dict(name="W5", row=58, col=46, base=610.0, phase_h=2.5, start_d=0.0, stop_d=30.0),
    dict(name="W6", row=70, col=54, base=500.0, phase_h=-1.0, start_d=0.0, stop_d=20.0),
    dict(name="W7", row=80, col=43, base=690.0, phase_h=3.0, start_d=0.0, stop_d=30.0),
    dict(name="W8", row=90, col=58, base=350.0, phase_h=-3.0, start_d=0.0, stop_d=30.0),
)

# Flash floods injected at the SFR head reach: (peak day, peak m3/d, rise h,
# recession time-constant in days).  Spike 1 follows storm 1 and spike 3 follows
# storm 2, but spike 2 is *uncorrelated* with recharge so a surrogate cannot
# simply infer stream inflow from the rainfall signal.
FLOOD_EVENTS: Tuple[Tuple[float, float, float, float], ...] = (
    (6.6, 120_000.0, 4.0, 1.2),
    (13.2, 45_000.0, 3.0, 0.6),
    (21.1, 200_000.0, 5.0, 1.8),
)

LAYER_NAMES: Tuple[str, ...] = (
    "Recent Alluvium",
    "Older Alluvium",
    "Massive Carbonate 1",
    "Massive Carbonate 2",
    "Fractured Granite",
)


# =============================================================================
# SECTION 1 - UTILITIES: WORKSPACE + MODFLOW 6 EXECUTABLE RESOLUTION
# =============================================================================


def make_workspace(root: pathlib.Path) -> Dict[str, pathlib.Path]:
    """Create (and return) the full output directory tree."""
    paths = {
        "root": root,
        "sim": root / "mf6",
        "warmup": root / "mf6_warmup",
        "gis": root / "gis",
        "tables": root / "tables",
        "arrays": root / "arrays",
        "figures": root / "figures",
        "report": root / "report",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _is_working_mf6(path: pathlib.Path) -> bool:
    """Return True if *path* is an executable file that looks like mf6."""
    return path.is_file() and os.access(str(path), os.X_OK)


def resolve_mf6_executable(bindir: Optional[pathlib.Path] = None) -> str:
    """Locate a MODFLOW 6 executable, downloading one if necessary.

    Resolution order (first hit wins):

    1. ``MF6_EXE`` environment variable.
    2. ``mf6`` already on ``PATH``.
    3. Well-known FloPy / conda install locations.
    4. ``flopy.utils.get_modflow`` (the officially supported installer).
    5. Direct download of the ``MODFLOW-ORG/executables`` release asset.  This
       fallback exists because ``get_modflow`` queries the GitHub *API*, which
       is blocked on many corporate/CI networks while the release *asset* URL
       still resolves.

    Raising is preferred over silently continuing: a benchmark that cannot run
    is worthless.
    """
    env_exe = os.environ.get("MF6_EXE")
    if env_exe and _is_working_mf6(pathlib.Path(env_exe)):
        print(f"[mf6] using MF6_EXE -> {env_exe}")
        return env_exe

    which = shutil.which("mf6")
    if which:
        print(f"[mf6] found on PATH -> {which}")
        return which

    candidates = [
        pathlib.Path.home() / ".local" / "bin" / "mf6",
        pathlib.Path.home() / "bin" / "mf6",
        pathlib.Path(sys.prefix) / "bin" / "mf6",
        pathlib.Path("/usr/local/bin/mf6"),
    ]
    if bindir is not None:
        candidates.insert(0, bindir / "mf6")
    for cand in candidates:
        if _is_working_mf6(cand):
            print(f"[mf6] found locally -> {cand}")
            return str(cand)

    target = bindir if bindir is not None else pathlib.Path.home() / ".local" / "bin"
    target.mkdir(parents=True, exist_ok=True)

    try:
        print("[mf6] not found; trying flopy.utils.get_modflow() ...")
        from flopy.utils import get_modflow

        get_modflow(str(target), subset="mf6", quiet=False)
        if _is_working_mf6(target / "mf6"):
            print(f"[mf6] installed via get_modflow -> {target / 'mf6'}")
            return str(target / "mf6")
    except Exception as exc:  # noqa: BLE001 - any failure falls through to (5)
        print(f"[mf6] get_modflow() failed ({exc.__class__.__name__}: {exc})")

    asset = "linux.zip"
    if sys.platform == "darwin":
        asset = "mac.zip"
    elif sys.platform.startswith("win"):
        asset = "win64.zip"
    for release in ("16.0", "15.0"):
        url = (
            "https://github.com/MODFLOW-ORG/executables/releases/download/"
            f"{release}/{asset}"
        )
        try:
            print(f"[mf6] downloading {url}")
            with tempfile.TemporaryDirectory() as tmp:
                zpath = pathlib.Path(tmp) / asset
                with urllib.request.urlopen(url, timeout=180) as resp, open(
                    zpath, "wb"
                ) as fh:
                    shutil.copyfileobj(resp, fh)
                with zipfile.ZipFile(zpath) as zf:
                    members = [m for m in zf.namelist() if pathlib.Path(m).stem == "mf6"]
                    zf.extractall(str(target), members=members or None)
            exe = target / ("mf6.exe" if sys.platform.startswith("win") else "mf6")
            if exe.is_file():
                exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                print(f"[mf6] installed via direct download -> {exe}")
                return str(exe)
        except Exception as exc:  # noqa: BLE001
            print(f"[mf6] download of release {release} failed: {exc}")

    raise RuntimeError(
        "Could not resolve a MODFLOW 6 executable.  Install it manually "
        "(https://github.com/MODFLOW-ORG/modflow6/releases) and either put it "
        "on PATH or set the MF6_EXE environment variable."
    )


# =============================================================================
# SECTION 2 - MEANDERING WATERCOURSES
# =============================================================================


@dataclass
class Channels:
    """Planform geometry of the two meandering watercourses.

    Both are stored as **4-connected** cell chains: consecutive cells always
    share a face.  For the stream that is a hard requirement (SFR reaches must
    form a physically contiguous network); for the river it simply prevents
    diagonal gaps that would look wrong on a map and would leak groundwater
    past the boundary.
    """

    stream_path: List[Tuple[int, int]]  # headwater -> confluence
    river_path: List[Tuple[int, int]]  # west -> east
    confluence: Tuple[int, int]
    stream_rtp: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stream_grad: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_reaches(self) -> int:
        return len(self.stream_path)

    def sinuosity(self, cfg: Config) -> float:
        """Channel length divided by straight-line valley length."""
        (r0, c0), (r1, c1) = self.stream_path[0], self.stream_path[-1]
        straight = np.hypot((r1 - r0) * cfg.delc, (c1 - c0) * cfg.delr)
        return float(len(self.stream_path) - 1) * cfg.delr / max(straight, 1.0)


def _meander(
    t: np.ndarray, mean: float, a1: float, w1: float, p1: float,
    a2: float, w2: float, p2: float,
) -> np.ndarray:
    """Two-harmonic meander train.

    Superposing two incommensurate wavelengths produces a planform that never
    exactly repeats, which stops a surrogate from learning a single periodic
    template for the channel position.
    """
    return (
        mean
        + a1 * np.sin(2.0 * np.pi * t / w1 + p1)
        + a2 * np.sin(2.0 * np.pi * t / w2 + p2)
    )


def _walk_4connected(anchor: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Turn a sequence of (row, col) anchors into a 4-connected cell chain.

    Between consecutive anchors the walk moves one index at a time - never
    diagonally - so every consecutive pair of cells in the result shares a face.
    """
    path: List[Tuple[int, int]] = [tuple(anchor[0])]  # type: ignore[list-item]
    for r_next, c_next in anchor[1:]:
        r, c = path[-1]
        while c != c_next:
            c += 1 if c_next > c else -1
            path.append((r, c))
        while r != r_next:
            r += 1 if r_next > r else -1
            path.append((r, c))
    return path


def build_channels(cfg: Config) -> Channels:
    """Generate the meandering stream and tidal river planforms.

    The stream runs north -> south down the valley and stops at the first cell
    that touches the tidal river: that cell is the confluence.  The river runs
    west -> east across the southern part of the domain.
    """
    # --- tidal river: one anchor per column, meandering about river_row_mean --
    cols = np.arange(cfg.ncol)
    river_rows = _meander(
        cols.astype(float), cfg.river_row_mean,
        cfg.river_amp1, cfg.river_wave1, cfg.river_phase1,
        cfg.river_amp2, cfg.river_wave2, cfg.river_phase2,
    )
    river_rows = np.clip(np.round(river_rows), cfg.river_row_min, cfg.river_row_max)
    river_anchor = [(int(r), int(c)) for c, r in zip(cols, river_rows)]
    river_path = _walk_4connected(river_anchor)
    river_cells = set(river_path)

    # --- central stream: one anchor per row, meandering about stream_col_mean -
    rows = np.arange(cfg.nrow)
    stream_cols = _meander(
        rows.astype(float), cfg.stream_col_mean,
        cfg.stream_amp1, cfg.stream_wave1, cfg.stream_phase1,
        cfg.stream_amp2, cfg.stream_wave2, cfg.stream_phase2,
    )
    stream_cols = np.clip(np.round(stream_cols), 1, cfg.ncol - 2)
    stream_anchor = [(int(r), int(c)) for r, c in zip(rows, stream_cols)]
    full_path = _walk_4connected(stream_anchor)

    # Truncate at the confluence - the first cell that is part of the river.
    confluence_idx = next(
        (i for i, cell in enumerate(full_path) if cell in river_cells),
        len(full_path) - 1,
    )
    stream_path = full_path[: confluence_idx + 1]

    ch = Channels(
        stream_path=stream_path,
        river_path=river_path,
        confluence=stream_path[-1],
    )
    print(
        f"[channels] stream: {ch.n_reaches} reaches, sinuosity "
        f"{ch.sinuosity(cfg):.3f}, confluence at r{ch.confluence[0]}/c{ch.confluence[1]}"
    )
    print(f"[channels] tidal river: {len(river_path)} cells, rows "
          f"{min(r for r, _ in river_path)}-{max(r for r, _ in river_path)}")
    return ch


# =============================================================================
# SECTION 3 - TOPOGRAPHY, STRATIGRAPHY AND GEOREFERENCING
# =============================================================================


@dataclass
class Grid:
    """Structured grid geometry + stratigraphy + active-cell bookkeeping."""

    cfg: Config
    top: np.ndarray  # (nrow, ncol) land surface
    botm: np.ndarray  # (nlay, nrow, ncol)
    idomain: np.ndarray  # (nlay, nrow, ncol) int
    valley_mask: np.ndarray  # (nrow, ncol) bool - Layer 1 present
    alluv_thick: np.ndarray  # (nrow, ncol) alluvial fill thickness
    dist_stream: np.ndarray  # (nrow, ncol) metres to the stream centreline
    dist_river: np.ndarray  # (nrow, ncol) metres to the river centreline
    xc: np.ndarray  # (ncol,) cell-centre easting
    yc: np.ndarray  # (nrow,) cell-centre northing (row 0 = north)
    transform: rasterio.Affine  # north-up raster transform

    # ------------------------------------------------------------------ #
    @property
    def shape2d(self) -> Tuple[int, int]:
        return self.cfg.nrow, self.cfg.ncol

    @property
    def thickness(self) -> np.ndarray:
        """(nlay, nrow, ncol) geometric thickness of each layer."""
        tops = np.empty_like(self.botm)
        tops[0] = self.top
        tops[1:] = self.botm[:-1]
        return tops - self.botm

    def top_of_layer(self, k: int) -> np.ndarray:
        return self.top if k == 0 else self.botm[k - 1]

    def uppermost_active(self) -> np.ndarray:
        """(nrow, ncol) 0-based index of the highest active layer per column."""
        upper = np.full(self.shape2d, -1, dtype=int)
        for k in range(self.cfg.nlay - 1, -1, -1):
            upper = np.where(self.idomain[k] > 0, k, upper)
        return upper

    # --- coordinate helpers (model row/col -> real-world metres) --------- #
    def cell_center(self, row: int, col: int) -> Tuple[float, float]:
        return float(self.xc[col]), float(self.yc[row])

    def x_edge(self, col: int) -> float:
        return self.cfg.x_origin + col * self.cfg.delr

    def y_edge(self, row: int) -> float:
        """Northing of the *northern* edge of ``row``."""
        return self.cfg.y_origin + (self.cfg.nrow - row) * self.cfg.delc


def _gaussian_field(
    cfg: Config, seed: int, var: float, len_scale: float, model: str = "Exponential"
) -> np.ndarray:
    """One zero-mean Gaussian random field on the model grid, shaped (nrow, ncol).

    ``Exponential`` covariance is the default because it is far rougher than a
    Gaussian covariance at short lag - combined with ``len_scale = 200 m`` (two
    cells) it produces the near-white, high-contrast K fields that make this a
    hard benchmark.  The *geological surfaces* instead use a smooth ``Gaussian``
    covariance at kilometre scale, because real stratigraphic contacts undulate
    smoothly rather than rattling from cell to cell.
    """
    cov = getattr(gs, model)(dim=2, var=var, len_scale=len_scale)
    srf = gs.SRF(cov, mean=0.0, seed=seed)

    # gstools works in ascending-coordinate order; ``structured`` returns
    # (nx, ny).  Transpose to (ny, nx) then flip so that row 0 = north, which
    # matches both MODFLOW's row convention and the GeoTIFF transform.
    x = (np.arange(cfg.ncol) + 0.5) * cfg.delr
    y = (np.arange(cfg.nrow) + 0.5) * cfg.delc
    fld = srf.structured((x, y))
    return np.flipud(np.asarray(fld).T)


def _distance_to_cells(
    cfg: Config, cells: Sequence[Tuple[int, int]]
) -> np.ndarray:
    """Euclidean distance (m) from every cell centre to the nearest listed cell."""
    mask = np.ones((cfg.nrow, cfg.ncol), dtype=bool)
    for r, c in cells:
        mask[r, c] = False
    return distance_transform_edt(mask, sampling=(cfg.delc, cfg.delr))


def build_grid(cfg: Config, channels: Channels) -> Grid:
    """Assemble topography and the five-layer stratigraphy.

    Geometry is built in the order a geomorphologist would reason about it:

    1. A regional land surface sloping from the northern plateau to the
       southern coastal plain, plus a smooth kilometre-scale undulation and a
       little hectometre-scale roughness.
    2. The two watercourses **incise** that surface: a broad valley following
       the meandering stream and a shallower floodplain following the tidal
       river.  The deeper of the two incisions wins where they overlap, which
       is what produces a proper confluence rather than a double-cut trench.
    3. Recent Alluvium (Layer 1) fills the incised corridor, thickening
       downstream (10 m at the headwaters to 18 m at the confluence) and
       tapering to nothing at the corridor margins.  Where the fill is thinner
       than ``alluv_min_thick`` the layer is *pinched out*: zero thickness and
       ``idomain = 0``.  MODFLOW 6 only enforces positive thickness on active
       cells, so this is the canonical structured-grid pinch-out and keeps
       ``botm[0]`` a valid top for Layer 2.
    4. The four deeper contacts are undulating (paleo-)surfaces about their
       prescribed mean elevations, so layer thickness - and hence
       transmissivity and storativity - varies smoothly across the domain.

    A final pass enforces a minimum thickness on every layer so no active cell
    can ever be inverted or degenerate.
    """
    nrow, ncol, nlay = cfg.nrow, cfg.ncol, cfg.nlay
    rr = np.arange(nrow)[:, None].astype(float)  # row index, broadcastable

    # --- distances to the two channel centrelines ---------------------------
    dist_stream = _distance_to_cells(cfg, channels.stream_path)
    dist_river = _distance_to_cells(cfg, channels.river_path)

    # --- 1. regional land surface -------------------------------------------
    regional = cfg.plateau_north + (
        cfg.plateau_south - cfg.plateau_north
    ) * (rr / (nrow - 1.0))
    undulation = _gaussian_field(
        cfg, cfg.seed + 9001, cfg.topo_undulation_var, cfg.topo_undulation_len,
        model="Gaussian",
    )
    roughness = _gaussian_field(
        cfg, cfg.seed + 9002, cfg.topo_roughness_var, cfg.topo_roughness_len,
        model="Gaussian",
    )

    # --- 2. fluvial incision -------------------------------------------------
    cut_valley = cfg.valley_incision * np.exp(
        -((dist_stream / cfg.valley_incision_width) ** 2)
    )
    cut_river = cfg.river_incision * np.exp(
        -((dist_river / cfg.river_incision_width) ** 2)
    )
    top = np.broadcast_to(regional, (nrow, ncol)) + undulation + roughness
    top = top - np.maximum(cut_valley, cut_river)

    # --- 3. alluvial fill ----------------------------------------------------
    axis_thick = cfg.alluv_thick_north + (
        cfg.alluv_thick_south - cfg.alluv_thick_north
    ) * (rr / (nrow - 1.0))
    fill_valley = np.broadcast_to(axis_thick, (nrow, ncol)) * np.exp(
        -((dist_stream / cfg.alluv_width) ** 2)
    )
    fill_river = cfg.floodplain_thick * np.exp(
        -((dist_river / cfg.floodplain_width) ** 2)
    )
    alluv_thick = np.maximum(fill_valley, fill_river)
    valley = alluv_thick >= cfg.alluv_min_thick
    alluv_thick = np.where(valley, alluv_thick, 0.0)

    # --- 4. stratigraphic contacts ------------------------------------------
    botm = np.empty((nlay, nrow, ncol), dtype=float)
    botm[0] = top - alluv_thick  # equals ``top`` where pinched out

    l2_mean = cfg.botm_l2_north + (cfg.botm_l2_south - cfg.botm_l2_north) * (
        rr / (nrow - 1.0)
    )
    botm[1] = np.broadcast_to(l2_mean, (nrow, ncol)) + _gaussian_field(
        cfg, cfg.seed + 9003, cfg.botm_l2_var, cfg.botm_l2_len, model="Gaussian"
    )
    botm[2] = cfg.botm_l3_mean + _gaussian_field(
        cfg, cfg.seed + 9004, cfg.botm_l3_var, cfg.botm_l3_len, model="Gaussian"
    )
    botm[3] = cfg.botm_l4_mean + _gaussian_field(
        cfg, cfg.seed + 9005, cfg.botm_l4_var, cfg.botm_l4_len, model="Gaussian"
    )
    botm[4] = cfg.botm_l5_mean + _gaussian_field(
        cfg, cfg.seed + 9006, cfg.botm_l5_var, cfg.botm_l5_len, model="Gaussian"
    )

    # --- 5. guarantee a well-posed geometry ---------------------------------
    for k in range(1, nlay):
        botm[k] = np.minimum(botm[k], botm[k - 1] - cfg.min_thickness[k])

    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    idomain[0] = valley.astype(int)

    # Cell centres.  Row 0 is the northern-most row (raster convention).
    xc = cfg.x_origin + (np.arange(ncol) + 0.5) * cfg.delr
    yc = cfg.y_origin + (nrow - np.arange(nrow) - 0.5) * cfg.delc
    transform = from_origin(
        cfg.x_origin, cfg.y_origin + nrow * cfg.delc, cfg.delr, cfg.delc
    )

    grid = Grid(
        cfg=cfg, top=top, botm=botm, idomain=idomain, valley_mask=valley,
        alluv_thick=alluv_thick, dist_stream=dist_stream, dist_river=dist_river,
        xc=xc, yc=yc, transform=transform,
    )

    thick = grid.thickness
    print(f"[grid] land surface {top.min():.2f} - {top.max():.2f} m "
          f"(mean {top.mean():.2f})")
    print(f"[grid] alluvial corridor: {int(valley.sum()):,} cells "
          f"({100.0 * valley.mean():.1f} % of the domain), fill up to "
          f"{alluv_thick.max():.1f} m")
    for k in range(nlay):
        act = idomain[k] > 0
        if act.any():
            print(f"[grid] L{k + 1} {LAYER_NAMES[k]:20s} thickness "
                  f"{thick[k][act].min():6.2f} - {thick[k][act].max():6.2f} m "
                  f"(mean {thick[k][act].mean():6.2f})")

    # --- 6. streambed long profile ------------------------------------------
    # The bed sits a fixed incision below the valley floor, then a running
    # minimum enforces a monotonically falling profile: a stream cannot flow
    # uphill, and MODFLOW's SFR needs a strictly positive gradient.
    bed = np.array([top[r, c] for r, c in channels.stream_path]) - cfg.sfr_incision
    bed = np.minimum.accumulate(bed)
    seg = cfg.delr
    grad = np.gradient(-bed, seg)
    channels.stream_rtp = bed
    channels.stream_grad = np.maximum(grad, cfg.sfr_min_grad)
    print(f"[grid] streambed {bed[0]:.2f} m (headwater) -> {bed[-1]:.2f} m "
          f"(confluence), mean gradient {np.mean(channels.stream_grad):.2e}")

    return grid


def water_table(grid: Grid, head: np.ndarray) -> np.ndarray:
    """(nrow, ncol) water-table / potentiometric elevation.

    FloPy's ``get_water_table`` keys off MODFLOW's ``hdry``/``hnoflo`` sentinels,
    but this script replaces those with NaN so that inactive cells can never
    contaminate a training tensor.  The consequence is that ``get_water_table``
    would return NaN for every column where Layer 1 is pinched out - which here
    is ~80 % of the domain.  This helper does the search explicitly instead:
    descend from the top and take the head of the first **active and saturated**
    cell (one whose head is above its own bottom).  Where the uppermost active
    cell is also confined the value is the potentiometric surface, which is the
    physically correct thing to contour.
    """
    nrow, ncol = grid.shape2d
    wt = np.full((nrow, ncol), np.nan)
    filled = np.zeros((nrow, ncol), dtype=bool)
    for k in range(grid.cfg.nlay):
        active = (grid.idomain[k] > 0) & np.isfinite(head[k])
        saturated = active & (head[k] > grid.botm[k])
        take = saturated & ~filled
        wt = np.where(take, head[k], wt)
        filled |= take
    # Fallback for fully dry columns: use the deepest active head available.
    if not filled.all():
        for k in range(grid.cfg.nlay - 1, -1, -1):
            active = (grid.idomain[k] > 0) & np.isfinite(head[k])
            take = active & ~filled
            wt = np.where(take, head[k], wt)
            filled |= take
    return wt


def build_icelltype(cfg: Config) -> np.ndarray:
    """Convertible (water-table) flag per layer.

    Layers 1-2 are convertible; the deeper carbonates and the granite stay
    confined.  This removes a large source of solver pain without sacrificing
    any intended non-linearity, because the water table lives in Layers 1-2.
    """
    icelltype = np.zeros((cfg.nlay, cfg.nrow, cfg.ncol), dtype=int)
    icelltype[0] = 1
    icelltype[1] = 1
    return icelltype


# =============================================================================
# SECTION 4 - GEOSTATISTICS (gstools)
# =============================================================================


@dataclass
class PropertyFields:
    """Per-layer hydraulic property realisations."""

    kh: np.ndarray  # (nlay, nrow, ncol) m/d
    k33: np.ndarray  # (nlay, nrow, ncol) m/d
    ss: np.ndarray  # (nlay, nrow, ncol) 1/m
    sy: np.ndarray  # (nlay, nrow, ncol) -

    def summary(self, grid: Grid) -> pd.DataFrame:
        """Per-layer descriptive statistics over the active cells."""
        rows = []
        for k in range(grid.cfg.nlay):
            act = grid.idomain[k] > 0
            if not act.any():
                continue
            kh = self.kh[k][act]
            rows.append(
                dict(
                    layer=k + 1,
                    unit=LAYER_NAMES[k],
                    n_active=int(act.sum()),
                    kh_geomean=float(np.exp(np.log(kh).mean())),
                    kh_min=float(kh.min()),
                    kh_p05=float(np.percentile(kh, 5)),
                    kh_p95=float(np.percentile(kh, 95)),
                    kh_max=float(kh.max()),
                    kh_log10_range=float(np.log10(kh.max() / kh.min())),
                    ss_mean=float(self.ss[k][act].mean()),
                    sy_mean=float(self.sy[k][act].mean()),
                )
            )
        return pd.DataFrame(rows)


def build_property_fields(cfg: Config, grid: Grid) -> PropertyFields:
    """Generate correlated log-normal Kh / Ss and bounded Sy fields per layer.

    * ``Kh = kh_mean * exp(Z)`` with ``Z ~ N(0, var_lnk)`` -> ``kh_mean`` is the
      *geometric* mean; with var = 2.0 the field spans two to three orders of
      magnitude within a single layer.
    * ``Ss`` uses the same recipe (log-normal is physically appropriate).
    * ``Sy`` cannot be log-normal - it is a bounded volume fraction.  The same
      correlated field is therefore squashed through a logistic map onto
      ``sy_bounds`` while preserving the spatial structure (and hence the
      correlation between high-K and high-Sy zones, which is what a surrogate
      would exploit).
    * Each layer gets an independent seed, so vertical correlation is zero.
    * Layer 1 additionally carries a **channel-facies trend**: hydraulic
      conductivity is boosted near the palaeo-channel axis, because coarse
      channel-lag and point-bar deposits sit there while overbank silts sit at
      the corridor margins.  This gives the *inverse* problem a genuine,
      physically meaningful structure to recover rather than pure noise.
    """
    nlay, nrow, ncol = cfg.nlay, cfg.nrow, cfg.ncol
    kh = np.empty((nlay, nrow, ncol))
    ss = np.empty((nlay, nrow, ncol))
    sy = np.empty((nlay, nrow, ncol))

    for k in range(nlay):
        z_k = _gaussian_field(cfg, cfg.seed + 1000 * k + 1, cfg.var_lnk, cfg.len_scale)
        z_s = _gaussian_field(cfg, cfg.seed + 1000 * k + 2, cfg.var_lnk, cfg.len_scale)
        z_y = _gaussian_field(cfg, cfg.seed + 1000 * k + 3, cfg.var_lnk, cfg.len_scale)

        kh[k] = cfg.kh_mean[k] * np.exp(z_k)

        # Ss: same log-normal treatment, scaled to keep the variance sane.
        ss[k] = cfg.ss_mean[k] * np.exp(np.sqrt(cfg.var_lns / cfg.var_lnk) * z_s * 0.5)

        # Sy: logistic squash of the correlated field onto physical bounds.
        lo, hi = cfg.sy_bounds
        centre = np.clip(cfg.sy_mean[k], lo + 1e-3, hi - 1e-3)
        p0 = (centre - lo) / (hi - lo)
        logit0 = np.log(p0 / (1.0 - p0))
        p = 1.0 / (1.0 + np.exp(-(logit0 + 0.9 * z_y)))
        sy[k] = lo + (hi - lo) * p

    # Channel-facies trend in the Recent Alluvium.
    facies = 1.0 + cfg.channel_facies_gain * np.exp(
        -((grid.dist_stream / cfg.channel_facies_width) ** 2)
    )
    kh[0] = kh[0] * facies

    # Guard rails: MODFLOW is unforgiving of pathological property values.
    kh = np.clip(kh, 1.0e-6, 5.0e3)
    ss = np.clip(ss, 1.0e-7, 1.0e-2)
    sy = np.clip(sy, cfg.sy_bounds[0], cfg.sy_bounds[1])
    k33 = kh / cfg.kz_ratio

    fields = PropertyFields(kh=kh, k33=k33, ss=ss, sy=sy)
    for _, row in fields.summary(grid).iterrows():
        print(
            f"[props] L{int(row.layer)}: Kh geo-mean={row.kh_geomean:8.4g} "
            f"min={row.kh_min:9.4g} max={row.kh_max:9.4g} "
            f"({row.kh_log10_range:.1f} orders) | Ss={row.ss_mean:.3g} "
            f"| Sy={row.sy_mean:.3f}"
        )
    return fields


# =============================================================================
# SECTION 5 - FAULTS: CELL-FACE BARRIERS FOR THE HFB PACKAGE
# =============================================================================


def four_connected_line(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    """Rasterise a straight line into a *4-connected* chain of cells.

    A classic (8-connected) Bresenham line takes diagonal steps, and diagonally
    adjacent cells do **not** share a face - so they cannot host an HFB barrier.
    This variant advances exactly one index per step, producing the "staircase"
    chain in which every consecutive pair is face-adjacent and therefore a valid
    HFB connection.  That staircase is the standard way to represent an oblique
    fault on a structured MODFLOW grid.
    """
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    r, c = r0, c0
    cells = [(r, c)]
    ir = ic = 0
    while ir < dr or ic < dc:
        if ic < dc and ((1 + 2 * ic) * dr <= (1 + 2 * ir) * dc or ir >= dr):
            c += sc
            ic += 1
        else:
            r += sr
            ir += 1
        cells.append((r, c))
    return cells


@dataclass
class Fault:
    """A fault discretised as a set of blocked cell faces."""

    name: str
    hydchr: float
    endpoints: Tuple[int, int, int, int]
    chain: List[Tuple[int, int]]  # 4-connected cell chain
    pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]]  # face-adjacent pairs


def build_faults(cfg: Config) -> List[Fault]:
    """Discretise both faults into face-adjacent cell pairs."""
    specs = [
        ("FAULT1_IMPERMEABLE", cfg.fault1, cfg.fault1_hydchr),
        ("FAULT2_SEMIPERMEABLE", cfg.fault2, cfg.fault2_hydchr),
    ]
    faults: List[Fault] = []
    for name, (r0, c0, r1, c1), hydchr in specs:
        chain = four_connected_line(r0, c0, r1, c1)
        pairs = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
        faults.append(
            Fault(name=name, hydchr=hydchr, endpoints=(r0, c0, r1, c1),
                  chain=chain, pairs=pairs)
        )
        print(f"[hfb] {name}: {len(chain)} cells -> {len(pairs)} blocked faces/layer")
    return faults


def hfb_stress_period_data(cfg: Config, grid: Grid, faults: Sequence[Fault]) -> List[list]:
    """Expand the fault face-pairs over **all five layers**.

    Only pairs where *both* cells are active in that layer are emitted, because
    Layer 1 exists only inside the sinuous alluvial corridor.
    """
    data: List[list] = []
    per_layer = {k: 0 for k in range(cfg.nlay)}
    for flt in faults:
        for (r1, c1), (r2, c2) in flt.pairs:
            for k in range(cfg.nlay):
                if grid.idomain[k, r1, c1] > 0 and grid.idomain[k, r2, c2] > 0:
                    data.append([(k, r1, c1), (k, r2, c2), flt.hydchr])
                    per_layer[k] += 1
    print("[hfb] barrier faces per layer: "
          + ", ".join(f"L{k + 1}={n}" for k, n in per_layer.items()))
    return data


def fault_face_geometries(grid: Grid, flt: Fault) -> List[LineString]:
    """Real-world geometry of every blocked cell face (for the shapefile)."""
    segs: List[LineString] = []
    for (r1, c1), (r2, c2) in flt.pairs:
        if r1 == r2:  # horizontal neighbours -> shared face is a vertical line
            x = grid.x_edge(max(c1, c2))
            segs.append(LineString([(x, grid.y_edge(r1 + 1)), (x, grid.y_edge(r1))]))
        else:  # vertical neighbours -> shared face is a horizontal line
            y = grid.y_edge(max(r1, r2))
            segs.append(LineString([(grid.x_edge(c1), y), (grid.x_edge(c1 + 1), y)]))
    return segs


def fault_observation_cells(
    grid: Grid, faults: Sequence[Fault], obs_layer: int = 1
) -> List[Tuple[str, Tuple[int, int, int]]]:
    """Pick the 4 head-observation cells straddling the two faults.

    For each fault we take the *middle* blocked face and observe the cell on
    either side of it.  These two cells are 100 m apart yet separated by the
    barrier, so their head difference is a direct, high-signal measurement of
    the discontinuity the surrogate has to reproduce.
    """
    obs: List[Tuple[str, Tuple[int, int, int]]] = []
    for i, flt in enumerate(faults, start=1):
        candidates = [
            p for p in flt.pairs
            if grid.idomain[obs_layer, p[0][0], p[0][1]] > 0
            and grid.idomain[obs_layer, p[1][0], p[1][1]] > 0
        ]
        if not candidates:
            raise RuntimeError(f"No active HFB pair for {flt.name} in layer {obs_layer}")
        (ra, ca), (rb, cb) = candidates[len(candidates) // 2]
        obs.append((f"F{i}_SIDE_A", (obs_layer, ra, ca)))
        obs.append((f"F{i}_SIDE_B", (obs_layer, rb, cb)))
        print(f"[obs] {flt.name}: F{i}_SIDE_A=(L{obs_layer + 1},r{ra},c{ca})  "
              f"F{i}_SIDE_B=(L{obs_layer + 1},r{rb},c{cb})")
    return obs


# =============================================================================
# SECTION 6 - TRANSIENT FORCING TIME SERIES
# =============================================================================


def stress_period_times(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (start, mid, end) time in days for every stress period."""
    t0 = np.arange(cfg.nper) * cfg.dt_days
    t1 = t0 + cfg.dt_days
    return t0, 0.5 * (t0 + t1), t1


def tidal_stage(cfg: Config, t_days: np.ndarray) -> np.ndarray:
    """Semi-diurnal tidal stage with a spring-neap envelope.

    The dominant M2 constituent has a 12.4 h period.  With 2-hour stress periods
    that is only ~6.2 samples per cycle - deliberately close to the Nyquist
    limit, so any surrogate that coarsens time will alias the tide.  The
    spring-neap envelope adds a slow amplitude modulation that cannot be
    captured by a single-frequency fit.
    """
    envelope = 1.0 + cfg.riv_spring_neap_frac * np.sin(
        2.0 * np.pi * t_days / cfg.riv_spring_neap_d
    )
    tide = np.sin(2.0 * np.pi * t_days / (cfg.riv_period_h / 24.0))
    return cfg.riv_mean_stage + cfg.riv_amplitude * envelope * tide


def stream_inflow(cfg: Config, t_days: np.ndarray) -> np.ndarray:
    """Baseflow plus asymmetric flash-flood hydrographs.

    Each event is a fast linear rise followed by an exponential recession - the
    classic asymmetric flood wave.  The sharp rising limbs stress a surrogate's
    temporal responsiveness for river-aquifer exchange: the induced bank storage
    reverses the flux direction within a couple of stress periods.
    """
    q = np.full_like(t_days, cfg.sfr_base_inflow)
    q *= 1.0 + 0.25 * np.sin(2.0 * np.pi * (t_days - 3.0) / 17.0)

    for t_peak, q_peak, rise_h, rec_d in FLOOD_EVENTS:
        rise_d = rise_h / 24.0
        rising = (t_days >= t_peak - rise_d) & (t_days < t_peak)
        falling = t_days >= t_peak
        add = np.zeros_like(t_days)
        add[rising] = q_peak * (t_days[rising] - (t_peak - rise_d)) / rise_d
        add[falling] = q_peak * np.exp(-(t_days[falling] - t_peak) / rec_d)
        q = q + add
    return q


def recharge_series(cfg: Config, t_days: np.ndarray) -> np.ndarray:
    """Base recharge with two intense 12-hour convective rainfall events."""
    rch = np.full_like(t_days, cfg.rch_base)
    for t_start, t_end in cfg.storm_windows_d:
        rch[(t_days >= t_start) & (t_days < t_end)] = cfg.storm_rate
    return rch


def well_rates(cfg: Config, t_days: np.ndarray) -> np.ndarray:
    """(nwell, nper) abstraction rates, negative = extraction.

    Each well follows a diurnal irrigation/supply schedule: a low night-time
    baseline, a smooth daytime peak between 06:00 and 18:00, a well-specific
    phase offset, a weekend reduction, and hard on/off step changes for two of
    the wells.  The composite pumping signal therefore contains a diurnal
    harmonic, a weekly harmonic and two discontinuities.
    """
    rates = np.zeros((len(WELLS), len(t_days)))
    hour = (t_days * 24.0) % 24.0
    weekend = (np.floor(t_days).astype(int) % 7) >= 5

    for i, w in enumerate(WELLS):
        h = (hour - w["phase_h"]) % 24.0
        day_shape = np.where(
            (h >= 6.0) & (h <= 18.0), np.sin(np.pi * (h - 6.0) / 12.0) ** 0.7, 0.0
        )
        factor = 0.25 + 1.35 * day_shape
        factor = np.where(weekend, factor * 0.45, factor)
        active = (t_days >= w["start_d"]) & (t_days < w["stop_d"])
        rates[i] = -w["base"] * factor * active
    return rates


@dataclass
class Forcing:
    """All transient boundary forcing, tabulated per stress period."""

    t_start: np.ndarray
    t_mid: np.ndarray
    t_end: np.ndarray
    stage: np.ndarray  # (nper,) river stage, m
    inflow: np.ndarray  # (nper,) SFR head-reach inflow, m3/d
    rch: np.ndarray  # (nper,) recharge, m/d
    wel: np.ndarray  # (nwell, nper) m3/d (negative)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "kper": np.arange(len(self.t_mid)) + 1,
                "t_start_d": self.t_start,
                "t_mid_d": self.t_mid,
                "t_end_d": self.t_end,
                "riv_stage_m": self.stage,
                "sfr_inflow_m3d": self.inflow,
                "rch_m_per_d": self.rch,
            }
        )
        for i, w in enumerate(WELLS):
            df[f"wel_{w['name']}_m3d"] = self.wel[i]
        df["wel_total_m3d"] = self.wel.sum(axis=0)
        return df


def build_forcing(cfg: Config) -> Forcing:
    """Evaluate every forcing series at the *midpoint* of each stress period.

    Midpoint sampling (rather than start-of-period) keeps a piecewise-constant
    representation second-order accurate for the smooth harmonics and avoids
    systematically lagging the tide by one stress period.
    """
    t0, tm, t1 = stress_period_times(cfg)
    forcing = Forcing(
        t_start=t0, t_mid=tm, t_end=t1,
        stage=tidal_stage(cfg, tm),
        inflow=stream_inflow(cfg, tm),
        rch=recharge_series(cfg, tm),
        wel=well_rates(cfg, tm),
    )
    print(
        f"[forcing] stage {forcing.stage.min():.2f}-{forcing.stage.max():.2f} m | "
        f"SFR inflow {forcing.inflow.min():,.0f}-{forcing.inflow.max():,.0f} m3/d | "
        f"recharge {forcing.rch.min():.2e}-{forcing.rch.max():.2e} m/d | "
        f"pumping {forcing.wel.sum(axis=0).min():,.0f}-"
        f"{forcing.wel.sum(axis=0).max():,.0f} m3/d"
    )
    return forcing


# =============================================================================
# SECTION 7 - BOUNDARY-CONDITION PACKAGE DATA
# =============================================================================


def riv_cells(cfg: Config, grid: Grid, channels: Channels) -> List[dict]:
    """One RIV cell per cell of the meandering tidal-river planform.

    The river is placed in the *uppermost active* of Layers 1-2: Layer 1 where
    its own floodplain alluvium exists (almost everywhere along the channel) and
    Layer 2 where the channel clips bedrock.  Putting it in both layers would
    double-count the conductance.
    """
    upper = grid.uppermost_active()
    cells = []
    cond = cfg.riv_bed_k * cfg.riv_width * cfg.delr / cfg.riv_bed_thick
    for r, c in channels.river_path:
        k = int(upper[r, c])
        if k < 0 or k > 1:
            continue
        cells.append(dict(layer=k, row=r, col=c, cond=cond))
    n_l1 = sum(1 for c in cells if c["layer"] == 0)
    print(f"[riv] {len(cells)} tidal-river cells ({n_l1} in L1, "
          f"{len(cells) - n_l1} in L2)")
    return cells


def riv_period_data(cells: Sequence[dict], forcing: Forcing) -> Dict[int, List[tuple]]:
    """Full RIV list re-specified every stress period so the stage follows the
    tide at 2-hour resolution."""
    return {
        kper: [
            ((c["layer"], c["row"], c["col"]), float(stage), c["cond"], CFG.riv_bottom)
            for c in cells
        ]
        for kper, stage in enumerate(forcing.stage)
    }


def ghb_cells(cfg: Config, grid: Grid, fields: PropertyFields,
              riv_cellids: Sequence[Tuple[int, int, int]]) -> List[tuple]:
    """Regional inflow along the eastern edge, Layers 2-4.

    Conductance is derived cell-by-cell from the local heterogeneous Kh and the
    local layer thickness, so the *boundary itself* inherits both the
    geostatistical roughness and the undulating stratigraphy - a surrogate
    cannot assume a smooth edge flux.  Cells shared with the tidal river are
    skipped so the two head-dependent boundaries do not fight over one cell.
    """
    thick = grid.thickness
    taken = set(riv_cellids)
    rows = []
    for k in cfg.ghb_layers:
        for r in range(cfg.nrow):
            if grid.idomain[k, r, cfg.ghb_col] <= 0:
                continue
            if (k, r, cfg.ghb_col) in taken:
                continue
            b = max(thick[k, r, cfg.ghb_col], 1.0e-3)
            cond = fields.kh[k, r, cfg.ghb_col] * b * cfg.delc / (0.5 * cfg.delr)
            rows.append(((k, r, cfg.ghb_col), cfg.ghb_head, float(cond)))
    print(f"[ghb] {len(rows)} regional-inflow cells on column {cfg.ghb_col}")
    return rows


def sfr_package_data(
    cfg: Config, grid: Grid, channels: Channels
) -> Tuple[List[list], List[list], List[Tuple[int, int, int]]]:
    """Build SFR ``packagedata`` and ``connectiondata`` for the meandering stream.

    Reaches follow the 4-connected meander chain from the headwater to the
    confluence in a single linearly connected network.  The most-downstream
    reach has no downstream connection, so its outflow leaves the model domain -
    conceptually it discharges into the tidal river it has just met.  (MODFLOW 6
    cannot route a mover from SFR into RIV, because RIV is not an advanced
    package, so an outflow boundary at the confluence is the correct construct.)
    """
    upper = grid.uppermost_active()
    nreach = len(channels.stream_path)

    pkgdata: List[list] = []
    conndata: List[list] = []
    cellids: List[Tuple[int, int, int]] = []
    for i, (r, c) in enumerate(channels.stream_path):
        k = int(upper[r, c])
        cellid = (k, r, c)
        cellids.append(cellid)
        conn = [i]
        if i > 0:
            conn.append(i - 1)
        if i < nreach - 1:
            conn.append(-(i + 1))
        conndata.append(conn)
        pkgdata.append(
            [
                i,                                  # ifno (0-based reach number)
                cellid,                             # connected GWF cell
                cfg.delr,                           # rlen
                cfg.sfr_width,                      # rwid
                float(channels.stream_grad[i]),     # rgrd
                float(channels.stream_rtp[i]),      # rtp (streambed top)
                cfg.sfr_bed_thick,
                cfg.sfr_bed_k,
                cfg.sfr_manning,
                len(conn) - 1,                      # ncon
                1.0,                                # ustrf - single downstream reach
                0,                                  # ndv
            ]
        )
    n_l1 = sum(1 for cid in cellids if cid[0] == 0)
    print(f"[sfr] {nreach} reaches ({n_l1} in L1), bed "
          f"{channels.stream_rtp[0]:.2f} -> {channels.stream_rtp[-1]:.2f} m")
    return pkgdata, conndata, cellids


def wel_period_data(forcing: Forcing) -> Dict[int, List[tuple]]:
    """Per-stress-period WEL list for the 8 valley production wells."""
    return {
        kper: [
            ((1, w["row"], w["col"]), float(forcing.wel[i, kper]), w["name"])
            for i, w in enumerate(WELLS)
        ]
        for kper in range(forcing.wel.shape[1])
    }


def recharge_layer_map(grid: Grid) -> np.ndarray:
    """0-based layer index that receives areal recharge in each column.

    MODFLOW 6's array-based RCH applies recharge to the *specified* layer, and
    when IRCH is omitted it uses the top grid layer - which is inactive outside
    the alluvial corridor.  IRCH is therefore set explicitly to the uppermost
    active layer.  FloPy expects a 0-based array here and writes the 1-based
    values MODFLOW requires.
    """
    upper = grid.uppermost_active()
    if (upper < 0).any():
        raise RuntimeError("Some columns have no active cell at all.")
    return upper


# =============================================================================
# SECTION 8 - MODEL CONSTRUCTION
# =============================================================================


def build_simulation(
    cfg: Config,
    grid: Grid,
    fields: PropertyFields,
    faults: Sequence[Fault],
    channels: Channels,
    forcing: Forcing,
    obs_cells: Sequence[Tuple[str, Tuple[int, int, int]]],
    ws: pathlib.Path,
    exe: str,
    strt: np.ndarray,
    transient: bool,
) -> flopy.mf6.MFSimulation:
    """Assemble the complete GWF simulation.

    ``transient=False`` produces the single-stress-period steady-state spin-up
    used to generate a physically consistent initial head field; ``True``
    produces the 360-stress-period benchmark itself.  Sharing one builder
    guarantees the two models are identical in every respect except time.
    """
    sim = flopy.mf6.MFSimulation(
        sim_name=cfg.sim_name, version="mf6", exe_name=exe, sim_ws=str(ws),
        memory_print_option="summary",
    )

    # ---------------- TDIS ----------------
    if transient:
        # 360 stress periods x 2 h, one time step each.  One time step per
        # stress period means the boundary conditions are refreshed on exactly
        # the same clock as the solution is reported - which is what makes this
        # dataset directly usable as (forcing_t -> head_t) training pairs.
        perioddata = [(cfg.dt_days, 1, 1.0)] * cfg.nper
    else:
        perioddata = [(1.0, 1, 1.0)]
    flopy.mf6.ModflowTdis(
        sim, time_units="days", nper=len(perioddata), perioddata=perioddata
    )

    # ---------------- IMS ----------------
    # COMPLEX complexity + BICGSTAB + Delta-Bar-Delta under-relaxation +
    # backtracking.  This combination is what keeps a Newton solution alive when
    # (a) K varies by orders of magnitude between neighbouring cells, (b) HFB
    # barriers create near-zero inter-cell conductances, and (c) the water table
    # crosses layer tops during the tidal cycle.
    flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="COMPLEX",
        outer_dvclose=1.0e-3,
        outer_maximum=250,
        under_relaxation="DBD",
        under_relaxation_theta=0.85,
        under_relaxation_kappa=0.05,
        under_relaxation_gamma=0.05,
        under_relaxation_momentum=0.05,
        backtracking_number=15,
        backtracking_tolerance=1.05,
        backtracking_reduction_factor=0.2,
        backtracking_residual_limit=0.002,
        inner_maximum=300,
        inner_dvclose=1.0e-4,
        rcloserecord=[1.0e-2, "STRICT"],
        linear_acceleration="BICGSTAB",
        scaling_method="DIAGONAL",
        reordering_method="NONE",
        relaxation_factor=0.0,
        preconditioner_levels=5,
        preconditioner_drop_tolerance=1.0e-4,
        number_orthogonalizations=2,
    )

    # ---------------- GWF model ----------------
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=cfg.model_name,
        model_nam_file=f"{cfg.model_name}.nam",
        save_flows=True,
        # Newton with under-relaxation: mandatory here.  Cells in the thin
        # valley alluvium go partially dry under pumping, and the standard
        # wetting/drying formulation would simply deactivate them.
        newtonoptions="NEWTON UNDER_RELAXATION",
    )

    # ---------------- DIS ----------------
    flopy.mf6.ModflowGwfdis(
        gwf, length_units="meters", nlay=cfg.nlay, nrow=cfg.nrow, ncol=cfg.ncol,
        delr=cfg.delr, delc=cfg.delc, top=grid.top, botm=grid.botm,
        idomain=grid.idomain, xorigin=cfg.x_origin, yorigin=cfg.y_origin, angrot=0.0,
    )
    gwf.modelgrid.set_coord_info(
        xoff=cfg.x_origin, yoff=cfg.y_origin, angrot=0.0, crs=f"EPSG:{cfg.crs_epsg}"
    )

    # ---------------- IC ----------------
    flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # ---------------- NPF ----------------
    icelltype = build_icelltype(cfg)
    flopy.mf6.ModflowGwfnpf(
        gwf, save_flows=True, save_specific_discharge=True, icelltype=icelltype,
        k=fields.kh, k33=fields.k33, k33overk=False,
    )

    # ---------------- STO ----------------
    if transient:
        flopy.mf6.ModflowGwfsto(
            gwf, save_flows=True, iconvert=icelltype, ss=fields.ss, sy=fields.sy,
            steady_state={0: False}, transient={0: True},
        )

    # ---------------- HFB (faults) ----------------
    hfb_data = hfb_stress_period_data(cfg, grid, faults)
    flopy.mf6.ModflowGwfhfb(
        gwf, print_input=True, maxhfb=len(hfb_data), stress_period_data={0: hfb_data}
    )

    # ---------------- RIV (meandering tidal boundary) ----------------
    rcells = riv_cells(cfg, grid, channels)
    if transient:
        riv_spd = riv_period_data(rcells, forcing)
    else:
        riv_spd = {
            0: [
                ((c["layer"], c["row"], c["col"]), cfg.riv_mean_stage, c["cond"],
                 cfg.riv_bottom)
                for c in rcells
            ]
        }
    flopy.mf6.ModflowGwfriv(
        gwf, save_flows=True, maxbound=len(rcells), stress_period_data=riv_spd,
        pname="riv_tidal",
    )

    # ---------------- GHB (eastern regional inflow) ----------------
    riv_ids = [(c["layer"], c["row"], c["col"]) for c in rcells]
    gdata = ghb_cells(cfg, grid, fields, riv_ids)
    flopy.mf6.ModflowGwfghb(
        gwf, save_flows=True, maxbound=len(gdata), stress_period_data={0: gdata},
        pname="ghb_east",
    )

    # ---------------- WEL (8 valley production wells) ----------------
    if transient:
        wel_spd = wel_period_data(forcing)
    else:
        wel_spd = {
            0: [
                ((1, w["row"], w["col"]), float(forcing.wel[i].mean()), w["name"])
                for i, w in enumerate(WELLS)
            ]
        }
    flopy.mf6.ModflowGwfwel(
        gwf, save_flows=True,
        # Boundnames (not aux) carry the well IDs through to the budget file so
        # each well's abstraction can be recovered by name during training-data
        # assembly.
        boundnames=True, maxbound=len(WELLS), stress_period_data=wel_spd,
        pname="wel_valley",
    )

    # ---------------- RCHA (areal recharge) ----------------
    irch = recharge_layer_map(grid)
    if transient:
        rch_spd = {kper: float(v) for kper, v in enumerate(forcing.rch)}
    else:
        # Antecedent conditions = the *base* rate.  Using the mean of the series
        # would fold the two 12-hour storms into the steady state and start the
        # benchmark from an artificially wet aquifer.
        rch_spd = {0: cfg.rch_base}
    flopy.mf6.ModflowGwfrcha(
        gwf, save_flows=True, readasarrays=True,
        irch={0: irch},  # specified once; MODFLOW 6 carries it forward
        recharge=rch_spd, pname="rcha",
    )

    # ---------------- SFR (meandering central stream) ----------------
    pkgdata, conndata, _ = sfr_package_data(cfg, grid, channels)
    if transient:
        sfr_spd = {kper: [(0, "inflow", float(q))] for kper, q in enumerate(forcing.inflow)}
    else:
        # Median, not mean: the flash-flood peaks are two orders of magnitude
        # above baseflow and would otherwise dominate the steady-state inflow.
        sfr_spd = {0: [(0, "inflow", float(np.median(forcing.inflow)))]}
    flopy.mf6.ModflowGwfsfr(
        gwf, save_flows=True, print_stage=False, print_flows=False,
        budget_filerecord=f"{cfg.model_name}.sfr.cbc",
        stage_filerecord=f"{cfg.model_name}.sfr.stage",
        length_conversion=1.0,     # metres
        time_conversion=86400.0,   # Manning's equation works in seconds
        nreaches=len(pkgdata), packagedata=pkgdata, connectiondata=conndata,
        perioddata=sfr_spd, pname="sfr_central",
    )

    # ---------------- OBS (fault-straddling head observations) ----------------
    flopy.mf6.ModflowUtlobs(
        gwf, digits=10, print_input=False,
        continuous={
            f"{cfg.model_name}.head.obs.csv": [
                (name, "HEAD", cellid) for name, cellid in obs_cells
            ]
        },
    )

    # ---------------- OC ----------------
    # Heads every single 2-hour step (that is the surrogate's target tensor);
    # the full cell-by-cell budget only once a day, because FLOW-JA-FACE for
    # 50 000 cells x 360 steps would be ~1 GB of disk for little extra value.
    if transient:
        saverecord: Dict[int, list] = {}
        for kper in range(cfg.nper):
            recs = [("HEAD", "ALL")]
            if (kper + 1) % cfg.budget_every_nper == 0:
                recs.append(("BUDGET", "ALL"))
            saverecord[kper] = recs
    else:
        saverecord = {0: [("HEAD", "ALL"), ("BUDGET", "ALL")]}

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{cfg.model_name}.hds",
        budget_filerecord=f"{cfg.model_name}.cbc",
        saverecord=saverecord,
        printrecord={0: [("BUDGET", "LAST")]},
    )

    return sim


def steady_state_initial_heads(
    cfg: Config, grid: Grid, fields: PropertyFields, faults: Sequence[Fault],
    channels: Channels, forcing: Forcing,
    obs_cells: Sequence[Tuple[str, Tuple[int, int, int]]],
    ws: pathlib.Path, exe: str,
) -> np.ndarray:
    """Run a steady-state spin-up and return its head field.

    Starting the transient benchmark from an arbitrary flat head would inject a
    huge, purely numerical relaxation transient that swamps the physical signals
    we actually want the surrogate to learn.  Spinning up under the *base*
    forcing gives a head field already in equilibrium with the heterogeneity,
    the geometry and the faults, so day 1 of the benchmark is dominated by the
    tide, the storms and the pumping - not by initialisation shock.
    """
    print("\n=== STEADY-STATE SPIN-UP (initial condition) ===")
    strt0 = np.maximum(
        np.broadcast_to(grid.top, (cfg.nlay, cfg.nrow, cfg.ncol)) - 2.0,
        cfg.riv_mean_stage,
    ).copy()
    sim = build_simulation(
        cfg, grid, fields, faults, channels, forcing, obs_cells, ws, exe, strt0,
        transient=False,
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    if not success:
        print("[warn] steady-state spin-up did not converge; falling back to a")
        print("       linear head ramp between the river stage and the GHB head.")
        for line in buff[-25:]:
            print("       " + str(line).rstrip())
        ramp = np.linspace(cfg.riv_mean_stage, cfg.ghb_head, cfg.ncol)
        return np.broadcast_to(ramp[None, None, :], (cfg.nlay, cfg.nrow, cfg.ncol)).copy()

    hds = flopy.utils.HeadFile(str(ws / f"{cfg.model_name}.hds"))
    head = hds.get_data(kstpkper=(0, 0)).astype(float)
    hds.close()

    # Sanitise: inactive / dry cells carry sentinel values that must never be
    # fed back in as an initial condition.
    bad = ~np.isfinite(head) | (np.abs(head) > 1.0e6)
    bad |= grid.idomain <= 0
    if bad.any():
        good_mean = float(np.mean(head[~bad])) if (~bad).any() else cfg.strt_fallback
        head[bad] = good_mean
    act = grid.idomain > 0
    print(f"[spinup] converged. head range {head[act].min():.2f} - "
          f"{head[act].max():.2f} m")
    return head


# =============================================================================
# SECTION 9 - GIS EXPORT (GeoTIFF + ESRI Shapefile)
# =============================================================================


def write_geotiff(
    path: pathlib.Path, array: np.ndarray, grid: Grid,
    nodata: float = -9999.0, dtype: str = "float32",
    band_descriptions: Optional[Sequence[str]] = None,
) -> None:
    """Write a north-up, compressed GeoTIFF (single- or multi-band).

    A 3-D ``array`` is written as a multi-band stack, which is how the head time
    series is stored: band *n* is stress period *n*.  That keeps 360 time slices
    in one georeferenced file instead of 360 separate ones.
    """
    arr = np.asarray(array, dtype=dtype)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    arr = arr.copy()
    arr[~np.isfinite(arr)] = nodata
    with rasterio.open(
        str(path), "w", driver="GTiff",
        height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
        dtype=dtype, crs=f"EPSG:{grid.cfg.crs_epsg}", transform=grid.transform,
        nodata=nodata, compress="deflate", predictor=2, tiled=True,
    ) as dst:
        dst.write(arr)
        if band_descriptions is not None:
            for i, desc in enumerate(band_descriptions, start=1):
                dst.set_band_description(i, desc)


def export_static_rasters(
    cfg: Config, grid: Grid, fields: PropertyFields, faults: Sequence[Fault],
    channels: Channels, strt: Optional[np.ndarray], gis_dir: pathlib.Path,
) -> int:
    """Export every static input field as a GeoTIFF feature layer.

    These are exactly the channels a convolutional surrogate stacks into its
    input tensor, and exactly the fields an inverse problem has to recover:

    * per-layer ``Kh``/``log10 Kh``/``Kz``/``Ss``/``Sy``  - the inversion targets;
    * per-layer top, bottom, thickness, ``idomain``, ``icelltype`` - the geometry
      the forward operator needs;
    * land surface, alluvial thickness, corridor mask - the geological context;
    * **distance-to-feature covariates** (stream, river, each fault, nearest
      well) - usually the single most informative engineered inputs for a
      groundwater surrogate, because head gradients organise themselves around
      exactly these features;
    * ``x``/``y`` coordinate rasters - drop-in collocation inputs for a PINN;
    * the steady-state initial head, without which the transient run cannot be
      reproduced.

    Inactive cells are written as NoData so masking is unambiguous downstream.
    """
    icelltype = build_icelltype(cfg)
    n = 0

    def _w(name: str, arr: np.ndarray, **kw) -> None:
        nonlocal n
        write_geotiff(gis_dir / name, arr, grid, **kw)
        n += 1

    for k in range(cfg.nlay):
        mask = grid.idomain[k] > 0
        lay = k + 1

        def masked(a: np.ndarray) -> np.ndarray:
            return np.where(mask, a, np.nan)

        _w(f"kh_L{lay}.tif", masked(fields.kh[k]))
        _w(f"log10_kh_L{lay}.tif", masked(np.log10(fields.kh[k])))
        _w(f"kz_L{lay}.tif", masked(fields.k33[k]))
        _w(f"ss_L{lay}.tif", masked(fields.ss[k]))
        _w(f"sy_L{lay}.tif", masked(fields.sy[k]))
        _w(f"transmissivity_L{lay}.tif", masked(fields.kh[k] * grid.thickness[k]))
        _w(f"top_L{lay}.tif", grid.top_of_layer(k))
        _w(f"botm_L{lay}.tif", grid.botm[k])
        _w(f"thickness_L{lay}.tif", masked(grid.thickness[k]))
        _w(f"idomain_L{lay}.tif", grid.idomain[k].astype(float))
        _w(f"icelltype_L{lay}.tif", icelltype[k].astype(float))
        if strt is not None:
            _w(f"strt_L{lay}.tif", masked(strt[k]))

    # --- 2-D geological context ---------------------------------------------
    _w("land_surface.tif", grid.top)
    _w("alluvium_thickness.tif", np.where(grid.valley_mask, grid.alluv_thick, np.nan))
    _w("valley_corridor_mask.tif", grid.valley_mask.astype(float))
    _w("recharge_layer_irch.tif", recharge_layer_map(grid).astype(float) + 1.0)
    _w("uppermost_active_layer.tif", grid.uppermost_active().astype(float) + 1.0)

    # --- distance-to-feature covariates -------------------------------------
    _w("dist_to_stream.tif", grid.dist_stream)
    _w("dist_to_river.tif", grid.dist_river)
    for i, flt in enumerate(faults, start=1):
        _w(f"dist_to_fault{i}.tif", _distance_to_cells(cfg, flt.chain))
    all_fault_cells = [c for f in faults for c in f.chain]
    _w("dist_to_any_fault.tif", _distance_to_cells(cfg, all_fault_cells))
    _w("dist_to_nearest_well.tif",
       _distance_to_cells(cfg, [(w["row"], w["col"]) for w in WELLS]))

    # --- coordinate rasters (PINN collocation inputs) -----------------------
    xx, yy = np.meshgrid(grid.xc, grid.yc)
    _w("x_coordinate.tif", xx)
    _w("y_coordinate.tif", yy)

    print(f"[gis] wrote {n} static GeoTIFF feature rasters -> {gis_dir}")
    return n


def export_head_rasters(
    cfg: Config, grid: Grid, head_stack: np.ndarray, times: np.ndarray,
    strt: np.ndarray, gwf: flopy.mf6.ModflowGwf, sim_ws: pathlib.Path,
    map_kper: int, gis_dir: pathlib.Path,
) -> int:
    """Export every simulated state as georeferenced rasters.

    Written for both benchmark modes:

    * **forward / simulation** - the full ``(nper, nlay, nrow, ncol)`` head
      tensor as one multi-band GeoTIFF per layer (band *n* = stress period *n*),
      so a surrogate's prediction can be compared cell-by-cell and step-by-step
      in a GIS as well as in NumPy;
    * **inverse** - the water table, the drawdown relative to the spin-up state,
      the temporal statistics (min/max/mean/range) that an inversion typically
      fits, and the specific-discharge vector components that a
      Darcy-constrained PINN residual needs.
    """
    n = 0

    def _w(name: str, arr: np.ndarray, **kw) -> None:
        nonlocal n
        write_geotiff(gis_dir / name, arr, grid, **kw)
        n += 1

    band_desc = [f"kper={i + 1} t={t:.4f}d" for i, t in enumerate(times)]
    idx = int(np.clip(map_kper - 1, 0, len(times) - 1))
    snap = head_stack[idx]

    for k in range(cfg.nlay):
        lay = k + 1
        _w(f"head_timeseries_L{lay}.tif", head_stack[:, k], band_descriptions=band_desc)
        _w(f"head_L{lay}_sp{map_kper}.tif", snap[k])
        _w(f"drawdown_L{lay}_sp{map_kper}.tif",
           np.where(grid.idomain[k] > 0, strt[k] - snap[k], np.nan))
        _w(f"head_min_L{lay}.tif", np.nanmin(head_stack[:, k], axis=0))
        _w(f"head_max_L{lay}.tif", np.nanmax(head_stack[:, k], axis=0))
        _w(f"head_mean_L{lay}.tif", np.nanmean(head_stack[:, k], axis=0))
        _w(f"head_range_L{lay}.tif",
           np.nanmax(head_stack[:, k], axis=0) - np.nanmin(head_stack[:, k], axis=0))

    # --- water table and unsaturated thickness ------------------------------
    wt = water_table(grid, snap)
    _w(f"watertable_sp{map_kper}.tif", wt)
    _w(f"watertable_depth_sp{map_kper}.tif", grid.top - wt)
    wt_stack = np.stack([water_table(grid, head_stack[i]) for i in range(len(times))])
    _w("watertable_timeseries.tif", wt_stack, band_descriptions=band_desc)
    _w("watertable_range.tif", np.nanmax(wt_stack, axis=0) - np.nanmin(wt_stack, axis=0))

    # --- specific discharge (Darcy flux) ------------------------------------
    try:
        cbc = flopy.utils.CellBudgetFile(str(sim_ws / f"{cfg.model_name}.cbc"))
        available = cbc.get_kstpkper()
        target = (0, map_kper - 1)
        if target not in available:
            target = min(available, key=lambda kk: abs(kk[1] - (map_kper - 1)))
        spdis = cbc.get_data(text="DATA-SPDIS", kstpkper=target)[0]
        qx, qy, qz = get_specific_discharge(spdis, gwf)
        cbc.close()
        for k in range(cfg.nlay):
            mask = grid.idomain[k] > 0
            _w(f"qx_L{k + 1}_sp{target[1] + 1}.tif", np.where(mask, qx[k], np.nan))
            _w(f"qy_L{k + 1}_sp{target[1] + 1}.tif", np.where(mask, qy[k], np.nan))
            _w(f"qz_L{k + 1}_sp{target[1] + 1}.tif", np.where(mask, qz[k], np.nan))
        _w("specific_discharge_magnitude_L2.tif",
           np.where(grid.idomain[1] > 0, np.hypot(qx[1], qy[1]), np.nan))
    except Exception as exc:  # noqa: BLE001
        print(f"[gis] specific discharge unavailable ({exc})")

    print(f"[gis] wrote {n} simulated-state GeoTIFF rasters -> {gis_dir}")
    return n


def _cell_polygon(grid: Grid, row: int, col: int) -> Polygon:
    return box(grid.x_edge(col), grid.y_edge(row + 1),
               grid.x_edge(col + 1), grid.y_edge(row))


def _path_line(grid: Grid, path: Sequence[Tuple[int, int]]) -> LineString:
    return LineString([grid.cell_center(r, c) for r, c in path])


def export_shapefiles(
    cfg: Config, grid: Grid, fields: PropertyFields, faults: Sequence[Fault],
    channels: Channels, forcing: Forcing,
    obs_cells: Sequence[Tuple[str, Tuple[int, int, int]]],
    gis_dir: pathlib.Path,
) -> int:
    """Export every boundary condition and structural feature as a Shapefile.

    Vector exports serve a different purpose to the rasters: they are what you
    compute distance-to-feature covariates from, what you clip and mask with,
    and what you draw on a map.  DBF field names are kept to <= 10 characters so
    no silent truncation occurs.
    """
    crs = f"EPSG:{cfg.crs_epsg}"
    n = 0

    def _write(gdf: gpd.GeoDataFrame, name: str) -> None:
        nonlocal n
        gdf.to_file(gis_dir / name, driver="ESRI Shapefile")
        n += 1

    # --- 1. domain outline and the sinuous alluvial corridor -----------------
    domain = box(cfg.x_origin, cfg.y_origin,
                 cfg.x_origin + cfg.ncol * cfg.delr,
                 cfg.y_origin + cfg.nrow * cfg.delc)
    _write(gpd.GeoDataFrame({"name": ["model_domain"], "kind": ["domain"]},
                            geometry=[domain], crs=crs), "domain_outline.shp")

    # Polygonise the Layer 1 active mask - this is the alluvial corridor, whose
    # shape is now a meander belt rather than a rectangle.
    mask = grid.valley_mask.astype(np.uint8)
    polys = [
        shapely_shape(geom)
        for geom, val in rio_features.shapes(mask, mask=mask.astype(bool),
                                             transform=grid.transform)
        if val == 1
    ]
    corridor = unary_union(polys)
    parts = list(corridor.geoms) if corridor.geom_type == "MultiPolygon" else [corridor]
    _write(
        gpd.GeoDataFrame(
            [dict(name="alluvial_corridor", part=i + 1,
                  area_km2=p.area / 1.0e6, unit=LAYER_NAMES[0])
             for i, p in enumerate(parts)],
            geometry=parts, crs=crs,
        ),
        "valley_alluvial_corridor.shp",
    )

    # --- 2. faults: idealised trace + every blocked cell face ----------------
    trace_geoms, trace_rows, face_geoms, face_rows = [], [], [], []
    for flt in faults:
        r0, c0, r1, c1 = flt.endpoints
        trace_geoms.append(LineString([grid.cell_center(r0, c0), grid.cell_center(r1, c1)]))
        trace_rows.append(dict(name=flt.name[:10], hydchr=flt.hydchr, r0=r0, c0=c0,
                               r1=r1, c1=c1, nfaces=len(flt.pairs)))
        for seg, ((ra, ca), (rb, cb)) in zip(fault_face_geometries(grid, flt), flt.pairs):
            face_geoms.append(seg)
            face_rows.append(dict(name=flt.name[:10], hydchr=flt.hydchr,
                                  r1=ra, c1=ca, r2=rb, c2=cb))
    _write(gpd.GeoDataFrame(trace_rows, geometry=trace_geoms, crs=crs), "faults_trace.shp")
    _write(gpd.GeoDataFrame(face_rows, geometry=face_geoms, crs=crs),
           "faults_hfb_faces.shp")

    # --- 3. meandering tidal river ------------------------------------------
    rcells = riv_cells(cfg, grid, channels)
    _write(
        gpd.GeoDataFrame(
            [dict(layer=c["layer"] + 1, row=c["row"], col=c["col"], cond=c["cond"],
                  rbot=cfg.riv_bottom, stage_mean=cfg.riv_mean_stage,
                  stage_amp=cfg.riv_amplitude, land_elev=float(grid.top[c["row"], c["col"]]))
             for c in rcells],
            geometry=[_cell_polygon(grid, c["row"], c["col"]) for c in rcells], crs=crs,
        ),
        "bc_riv_cells.shp",
    )
    _write(
        gpd.GeoDataFrame(
            [dict(name="tidal_river", ncells=len(channels.river_path),
                  length_m=(len(channels.river_path) - 1) * cfg.delr)],
            geometry=[_path_line(grid, channels.river_path)], crs=crs,
        ),
        "bc_riv_centerline.shp",
    )

    # --- 4. meandering stream (SFR) -----------------------------------------
    pkgdata, _, cellids = sfr_package_data(cfg, grid, channels)
    _write(
        gpd.GeoDataFrame(
            [dict(ifno=int(rec[0]), layer=cellids[i][0] + 1, row=cellids[i][1],
                  col=cellids[i][2], rlen=rec[2], rwid=rec[3], rgrd=rec[4],
                  rtp=rec[5], rhk=rec[7], man=rec[8],
                  dist_m=i * cfg.delr,
                  land_elev=float(grid.top[cellids[i][1], cellids[i][2]]))
             for i, rec in enumerate(pkgdata)],
            geometry=[Point(grid.cell_center(cid[1], cid[2])) for cid in cellids], crs=crs,
        ),
        "bc_sfr_reaches.shp",
    )
    _write(
        gpd.GeoDataFrame(
            [dict(name="central_stream", nreach=len(cellids),
                  length_m=(len(cellids) - 1) * cfg.delr,
                  sinuosity=channels.sinuosity(cfg))],
            geometry=[_path_line(grid, channels.stream_path)], crs=crs,
        ),
        "bc_sfr_centerline.shp",
    )

    # --- 5. GHB cells --------------------------------------------------------
    riv_ids = [(c["layer"], c["row"], c["col"]) for c in rcells]
    gdata = ghb_cells(cfg, grid, fields, riv_ids)
    _write(
        gpd.GeoDataFrame(
            [dict(layer=cid[0] + 1, row=cid[1], col=cid[2], bhead=bh, cond=cd)
             for cid, bh, cd in gdata],
            geometry=[_cell_polygon(grid, cid[1], cid[2]) for cid, _, _ in gdata], crs=crs,
        ),
        "bc_ghb_cells.shp",
    )

    # --- 6. wells ------------------------------------------------------------
    _write(
        gpd.GeoDataFrame(
            [dict(name=w["name"], layer=2, row=w["row"], col=w["col"],
                  q_base=w["base"], q_mean=float(forcing.wel[i].mean()),
                  q_min=float(forcing.wel[i].min()), phase_h=w["phase_h"],
                  start_d=w["start_d"], stop_d=w["stop_d"],
                  land_elev=float(grid.top[w["row"], w["col"]]))
             for i, w in enumerate(WELLS)],
            geometry=[Point(grid.cell_center(w["row"], w["col"])) for w in WELLS], crs=crs,
        ),
        "bc_wells.shp",
    )

    # --- 7. observation points ----------------------------------------------
    _write(
        gpd.GeoDataFrame(
            [dict(name=name, layer=cid[0] + 1, row=cid[1], col=cid[2],
                  land_elev=float(grid.top[cid[1], cid[2]]))
             for name, cid in obs_cells],
            geometry=[Point(grid.cell_center(cid[1], cid[2])) for _, cid in obs_cells],
            crs=crs,
        ),
        "obs_points.shp",
    )

    print(f"[gis] wrote {n} ESRI Shapefiles -> {gis_dir}")
    return n


def write_manifest(paths: Dict[str, pathlib.Path]) -> pd.DataFrame:
    """Catalogue every exported file so the dataset is self-describing."""
    rules = [
        ("head_timeseries_", "Multi-band head time stack; band n = stress period n (m)"),
        ("watertable_timeseries", "Multi-band water-table time stack (m)"),
        ("head_min_", "Minimum simulated head over the 30 days (m)"),
        ("head_max_", "Maximum simulated head over the 30 days (m)"),
        ("head_mean_", "Time-mean simulated head (m)"),
        ("head_range_", "Max-minus-min simulated head, i.e. transient amplitude (m)"),
        ("head_L", "Simulated head snapshot (m)"),
        ("drawdown_", "Spin-up head minus snapshot head, i.e. drawdown (m)"),
        ("watertable_depth", "Land surface minus water table (m below ground)"),
        ("watertable_range", "Water-table fluctuation amplitude (m)"),
        ("watertable_", "Water-table elevation (m)"),
        ("qx_", "Specific discharge, x component (m/d)"),
        ("qy_", "Specific discharge, y component (m/d)"),
        ("qz_", "Specific discharge, z component (m/d)"),
        ("specific_discharge_magnitude", "Horizontal Darcy flux magnitude (m/d)"),
        ("log10_kh_", "log10 horizontal hydraulic conductivity (log10 m/d)"),
        ("kh_", "Horizontal hydraulic conductivity - INVERSION TARGET (m/d)"),
        ("kz_", "Vertical hydraulic conductivity (m/d)"),
        ("ss_", "Specific storage - INVERSION TARGET (1/m)"),
        ("sy_", "Specific yield - INVERSION TARGET (-)"),
        ("transmissivity_", "Kh x layer thickness (m2/d)"),
        ("strt_", "Steady-state initial head used to start the transient run (m)"),
        ("top_", "Layer top elevation (m)"),
        ("botm_", "Layer bottom elevation (m)"),
        ("thickness_", "Layer geometric thickness (m)"),
        ("idomain_", "Active-cell mask (1 active, 0 pinched out)"),
        ("icelltype_", "Convertible flag (1 water table, 0 confined)"),
        ("land_surface", "Land surface elevation (m)"),
        ("alluvium_thickness", "Recent Alluvium fill thickness (m)"),
        ("valley_corridor_mask", "Layer 1 presence mask (meander belt)"),
        ("recharge_layer_irch", "1-based layer receiving areal recharge"),
        ("uppermost_active_layer", "1-based index of the highest active layer"),
        ("dist_to_", "Euclidean distance covariate (m)"),
        ("x_coordinate", "Cell-centre easting (m) - PINN collocation input"),
        ("y_coordinate", "Cell-centre northing (m) - PINN collocation input"),
        ("domain_outline", "Model domain rectangle"),
        ("valley_alluvial_corridor", "Polygonised meander-belt alluvium"),
        ("faults_trace", "Idealised straight fault traces with hydchr"),
        ("faults_hfb_faces", "Every blocked cell face used by the HFB package"),
        ("bc_riv_cells", "Tidal-river RIV cells"),
        ("bc_riv_centerline", "Meandering tidal-river centreline"),
        ("bc_sfr_reaches", "SFR reaches with streambed geometry"),
        ("bc_sfr_centerline", "Meandering stream centreline"),
        ("bc_ghb_cells", "Eastern regional-inflow GHB cells"),
        ("bc_wells", "Production wells with scheduling metadata"),
        ("obs_points", "Fault-straddling head observation points"),
        ("forcing_by_stress_period", "Every boundary forcing, per stress period"),
        ("fault_observations", "Observed head hydrographs at the 4 obs points"),
        ("fault_contrast", "Head jump across faults vs background gradient"),
        ("stream_long_profile", "SFR streambed long profile"),
        ("property_statistics", "Per-layer hydraulic property statistics"),
        ("wells", "Well locations and pumping metadata"),
        ("observation_points", "Observation point coordinates"),
        ("heads_transient", "Full head tensor (nper, nlay, nrow, ncol) float32"),
    ]

    def describe(name: str) -> str:
        for prefix, desc in rules:
            if name.startswith(prefix):
                return desc
        return ""

    rows = []
    for key in ("gis", "tables", "arrays", "figures", "report"):
        d = paths[key]
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".shx", ".dbf", ".prj", ".cpg"):
                continue  # shapefile sidecars - listed under the .shp
            rows.append(
                dict(folder=key, file=f.name, size_kb=round(f.stat().st_size / 1024, 1),
                     description=describe(f.stem))
            )
    df = pd.DataFrame(rows)
    df.to_csv(paths["tables"] / "MANIFEST.csv", index=False)
    print(f"[gis] catalogued {len(df)} output files -> tables/MANIFEST.csv")
    return df


# =============================================================================
# SECTION 10 - PLOTS
# =============================================================================


def _save(fig: plt.Figure, path: pathlib.Path) -> plt.Figure:
    fig.savefig(path, dpi=150)
    print(f"[plot] {path}")
    return fig


def plot_topography_geology(
    cfg: Config, grid: Grid, channels: Channels, faults: Sequence[Fault],
    out_png: pathlib.Path,
) -> plt.Figure:
    """Land surface, the meander belt and the undulating stratigraphy in plan."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    extent = (cfg.x_origin, cfg.x_origin + cfg.ncol * cfg.delr,
              cfg.y_origin, cfg.y_origin + cfg.nrow * cfg.delc)

    def overlay(ax):
        for path, col, lw in ((channels.stream_path, "cyan", 1.6),
                              (channels.river_path, "navy", 2.2)):
            xy = np.array([grid.cell_center(r, c) for r, c in path])
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=lw)
        for flt, col in zip(faults, ("red", "darkorange")):
            for seg in fault_face_geometries(grid, flt):
                xs, ys = seg.xy
                ax.plot(xs, ys, color=col, lw=1.4, solid_capstyle="butt")
        ax.set_xticks([])
        ax.set_yticks([])

    panels = [
        ("Land surface (m a.s.l.)", grid.top, "terrain", None),
        ("Recent Alluvium thickness (m)", np.where(grid.valley_mask, grid.alluv_thick, np.nan), "YlGnBu", None),
        ("Layer 2 thickness (m)", grid.thickness[1], "viridis", None),
        ("Base of Older Alluvium (m)", grid.botm[1], "cividis", None),
        ("Base of Carbonate 2 (m)", grid.botm[3], "cividis", None),
        ("Basement top / L5 base (m)", grid.botm[4], "cividis", None),
    ]
    for ax, (title, arr, cmap, norm) in zip(axes.flat, panels):
        im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm)
        overlay(ax)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.82)

    fig.suptitle(
        "Topography and stratigraphic geometry - meandering stream (cyan), tidal "
        "river (blue), faults (red/orange)", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_png)


def plot_k_field(
    cfg: Config, grid: Grid, fields: PropertyFields, faults: Sequence[Fault],
    channels: Channels, out_png: pathlib.Path,
) -> plt.Figure:
    """Log-K panels for all five layers with the faults and channels overlaid."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    extent = (cfg.x_origin, cfg.x_origin + cfg.ncol * cfg.delr,
              cfg.y_origin, cfg.y_origin + cfg.nrow * cfg.delc)
    for k in range(cfg.nlay):
        ax = axes.flat[k]
        arr = np.where(grid.idomain[k] > 0, fields.kh[k], np.nan)
        im = ax.imshow(arr, extent=extent, origin="upper", cmap="turbo",
                       norm=LogNorm(vmin=np.nanpercentile(arr, 1),
                                    vmax=np.nanpercentile(arr, 99)))
        for flt, col in zip(faults, ("k", "w")):
            for seg in fault_face_geometries(grid, flt):
                xs, ys = seg.xy
                ax.plot(xs, ys, color=col, lw=1.5, solid_capstyle="butt")
        ax.set_title(f"Layer {k + 1} - {LAYER_NAMES[k]}\n$K_h$ (m/d), geo-mean "
                     f"{cfg.kh_mean[k]:g}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes.flat[5]
    arr = np.where(grid.idomain[1] > 0, fields.sy[1], np.nan)
    im = ax.imshow(arr, extent=extent, origin="upper", cmap="magma")
    ax.set_title("Layer 2 - specific yield $S_y$ (-)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"Geostatistical realisation: exponential covariance, len_scale = "
        f"{cfg.len_scale:g} m ({cfg.len_scale / cfg.delr:.0f} cells), "
        f"var(lnK) = {cfg.var_lnk:g}  |  seed = {cfg.seed}", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_png)


# Conventional geological shades: alluvium in ochres, carbonates in blue-greys,
# basement in pink - so a hydrogeologist reads the section without a legend.
UNIT_COLORS: Tuple[str, ...] = ("#f2d492", "#d9a441", "#9fc4d6", "#6b9dc2", "#c98fb0")


def _section_specs(cfg: Config, grid: Grid, channels: Channels):
    """The three section lines used by both section figures."""
    return [
        (
            "W-E transverse, row 25 (crosses Fault 1 and the meander belt)",
            {"row": 25}, 25, None,
        ),
        (
            "W-E transverse, row 75 (crosses Fault 2 and the meander belt)",
            {"row": 75}, 75, None,
        ),
        (
            "Longitudinal down the meandering stream (headwater -> confluence)",
            {"line": [grid.cell_center(r, c) for r, c in channels.stream_path]},
            None, channels,
        ),
    ]


def _decorate_section(
    ax, cfg: Config, grid: Grid, faults: Sequence[Fault], row: Optional[int],
    chan: Optional[Channels], wt: Optional[np.ndarray],
) -> None:
    """Overlay land surface, water table, streambed and fault crossings."""
    if row is not None:
        dist = (np.arange(cfg.ncol) + 0.5) * cfg.delr
        ax.plot(dist, grid.top[row], color="saddlebrown", lw=1.8, label="Land surface")
        if wt is not None:
            ax.plot(dist, wt[row], color="blue", lw=1.6, ls="--", label="Water table")
        for flt, col in zip(faults, ("red", "darkorange")):
            for c in sorted({c for r, c in flt.chain if r == row}):
                ax.axvline((c + 1) * cfg.delr, color=col, lw=2.0, alpha=0.85)
    else:
        dist = np.arange(len(chan.stream_path)) * cfg.delr
        land = np.array([grid.top[r, c] for r, c in chan.stream_path])
        ax.plot(dist, land, color="saddlebrown", lw=1.8, label="Land surface")
        ax.plot(dist, chan.stream_rtp, color="#00bcd4", lw=1.8, label="Streambed")
        if wt is not None:
            ax.plot(dist, [wt[r, c] for r, c in chan.stream_path], color="blue",
                    lw=1.6, ls="--", label="Water table")
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_xlabel("Distance along section (m)")


def plot_geological_sections(
    cfg: Config, grid: Grid, faults: Sequence[Fault], channels: Channels,
    gwf: flopy.mf6.ModflowGwf, head: Optional[np.ndarray], out_png: pathlib.Path,
) -> plt.Figure:
    """Classic geological cross-sections: fill is the stratigraphic unit.

    The first three panels show the full 200 m section so the undulating
    carbonate and basement contacts are visible; the fourth repeats the
    transverse section at a shallow zoom, because the Recent Alluvium is only
    10-18 m thick and would otherwise be a hairline at full scale.  That shallow
    panel is where the pinch-out, the incised valley and the water table are
    actually legible.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    unit = np.empty((cfg.nlay, cfg.nrow, cfg.ncol), dtype=float)
    for k in range(cfg.nlay):
        unit[k] = np.where(grid.idomain[k] > 0, k, np.nan)
    cmap = ListedColormap(UNIT_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, cfg.nlay), cmap.N)
    wt = water_table(grid, head) if head is not None else None

    specs = _section_specs(cfg, grid, channels)
    layout = [
        (specs[0], None), (specs[1], None), (specs[2], None),
        (specs[0], (5.0, 60.0)),  # shallow zoom on the alluvial valley
    ]

    fig, axes = plt.subplots(4, 1, figsize=(15, 16.5))
    for ax, ((title, line, row, chan), ylim) in zip(axes, layout):
        xs = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line=line,
                                         geographic_coords=False)
        xs.plot_array(unit, cmap=cmap, norm=norm, alpha=1.0)
        xs.plot_grid(lw=0.12, color="0.35", alpha=0.4)
        xs.plot_inactive(color_noflow="white")
        _decorate_section(ax, cfg, grid, faults, row, chan, wt)
        if ylim is not None:
            ax.set_ylim(*ylim)
            title = title + "  -  SHALLOW ZOOM on the alluvial valley"
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    handles = [Patch(facecolor=UNIT_COLORS[k], edgecolor="0.3",
                     label=f"L{k + 1}  {LAYER_NAMES[k]}") for k in range(cfg.nlay)]
    handles += [
        plt.Line2D([], [], color="red", lw=2, label="Fault 1 (impermeable)"),
        plt.Line2D([], [], color="darkorange", lw=2, label="Fault 2 (semi-permeable)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Geological cross-sections - stratigraphic units, undulating contacts "
        "and the alluvial pinch-out", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    return _save(fig, out_png)


def plot_k_sections(
    cfg: Config, grid: Grid, fields: PropertyFields, faults: Sequence[Fault],
    channels: Channels, gwf: flopy.mf6.ModflowGwf, head: Optional[np.ndarray],
    out_png: pathlib.Path,
) -> plt.Figure:
    """The same section lines, filled with log10 Kh instead of the unit.

    Side by side with the geological sections these show the point of the
    benchmark: the stratigraphy is smooth and layer-cake, but the conductivity
    inside each unit varies by two to three orders of magnitude over two cells.
    """
    log_kh = np.log10(np.where(grid.idomain > 0, fields.kh, np.nan))
    vmin, vmax = np.nanpercentile(log_kh, [2, 98])
    wt = water_table(grid, head) if head is not None else None

    specs = _section_specs(cfg, grid, channels)
    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    for ax, (title, line, row, chan) in zip(axes, specs):
        xs = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line=line,
                                         geographic_coords=False)
        pc = xs.plot_array(log_kh, cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.95)
        xs.plot_grid(lw=0.12, color="0.3", alpha=0.35)
        xs.plot_inactive(color_noflow="0.85")
        _decorate_section(ax, cfg, grid, faults, row, chan, wt)
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        cb = fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.01)
        cb.set_label("$\\log_{10} K_h$ (m/d)", fontsize=8)

    fig.suptitle(
        "Hydraulic conductivity in cross-section - two to three orders of "
        "magnitude of variation within every unit", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_png)


def plot_forcing(cfg: Config, forcing: Forcing, out_png: pathlib.Path) -> plt.Figure:
    """Four-panel summary of everything that drives the model."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(forcing.t_mid, forcing.stage, lw=0.8, color="#1f77b4")
    axes[0].axhline(cfg.riv_bottom, color="k", ls=":", lw=0.8, label="River bottom")
    axes[0].set_ylabel("Stage (m)")
    axes[0].set_title("Tidal river stage - M2 semi-diurnal (12.4 h) with spring-neap "
                      "envelope", fontsize=10, loc="left")
    axes[0].legend(fontsize=8)

    axes[1].plot(forcing.t_mid, forcing.inflow, lw=0.9, color="#17becf")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Inflow (m$^3$/d)")
    axes[1].set_title("SFR head-reach inflow - baseflow plus three flash-flood "
                      "hydrographs", fontsize=10, loc="left")

    axes[2].plot(forcing.t_mid, forcing.rch, lw=0.9, color="#2ca02c")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Recharge (m/d)")
    axes[2].set_title("Areal recharge - base rate plus two intense 12-hour storms",
                      fontsize=10, loc="left")

    for i, w in enumerate(WELLS):
        axes[3].plot(forcing.t_mid, -forcing.wel[i], lw=0.6, label=w["name"])
    axes[3].plot(forcing.t_mid, -forcing.wel.sum(axis=0), lw=1.4, color="k", label="Total")
    axes[3].set_ylabel("Abstraction (m$^3$/d)")
    axes[3].set_xlabel("Time (days)")
    axes[3].set_title("Well abstraction - diurnal + weekly cycles, W4 starts at day 12, "
                      "W6 stops at day 20", fontsize=10, loc="left")
    axes[3].legend(fontsize=7, ncol=5, loc="upper left")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlim(0, cfg.nper * cfg.dt_days)

    fig.suptitle("Transient forcing at 2-hour stress-period resolution", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_png)


def plot_head_map(
    cfg: Config, grid: Grid, gwf: flopy.mf6.ModflowGwf, head: np.ndarray,
    faults: Sequence[Fault], channels: Channels,
    obs_cells: Sequence[Tuple[str, Tuple[int, int, int]]],
    kper: int, time_d: float, out_png: pathlib.Path,
) -> plt.Figure:
    """Filled + line contour map of the Layer 2 head field with all features."""
    fig, ax = plt.subplots(figsize=(11, 9.5))
    pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=1)

    h2 = np.where(grid.idomain[1] > 0, head[1], np.nan)
    finite = h2[np.isfinite(h2)]
    levels = np.linspace(np.nanpercentile(finite, 0.5), np.nanpercentile(finite, 99.5), 30)

    cf = pmv.plot_array(h2, cmap="viridis", alpha=0.95)
    cs = pmv.contour_array(h2, levels=levels, colors="k", linewidths=0.45, alpha=0.75)
    ax.clabel(cs, cs.levels[::4], inline=True, fontsize=6, fmt="%.1f")
    pmv.plot_inactive(color_noflow="0.85")

    pmv.plot_bc("GHB", color="#9467bd", alpha=0.55)
    pmv.plot_bc("RIV", color="#1f77b4", alpha=0.7)
    pmv.plot_bc("SFR", color="#17becf", alpha=0.85)

    for path, col, lw, lab in (
        (channels.stream_path, "#17becf", 1.6, "Central stream (SFR)"),
        (channels.river_path, "#1f77b4", 2.4, "Tidal river (RIV)"),
    ):
        xy = np.array([grid.cell_center(r, c) for r, c in path])
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=lw, label=lab)

    for flt, col in zip(faults, ("red", "darkorange")):
        for seg in fault_face_geometries(grid, flt):
            xs, ys = seg.xy
            ax.plot(xs, ys, color=col, lw=2.2, solid_capstyle="butt")
        ax.plot([], [], color=col, lw=2.2, label=f"{flt.name} (K'={flt.hydchr:g}/d)")

    wx = [grid.cell_center(w["row"], w["col"])[0] for w in WELLS]
    wy = [grid.cell_center(w["row"], w["col"])[1] for w in WELLS]
    ax.scatter(wx, wy, marker="o", s=48, facecolor="white", edgecolor="k", zorder=6,
               label="Pumping wells (L2)")
    for w, x, y in zip(WELLS, wx, wy):
        ax.annotate(w["name"], (x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=7, zorder=7)

    ox = [grid.cell_center(cid[1], cid[2])[0] for _, cid in obs_cells]
    oy = [grid.cell_center(cid[1], cid[2])[1] for _, cid in obs_cells]
    ax.scatter(ox, oy, marker="^", s=70, facecolor="yellow", edgecolor="k", zorder=8,
               label="Fault observation points")

    cbar = fig.colorbar(cf, ax=ax, shrink=0.82)
    cbar.set_label("Head in Layer 2 - Older Alluvium (m a.s.l.)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Layer 2 simulated head - stress period {kper} (t = {time_d:.3f} d)\n"
                 "Sharp offsets across the fault traces are the HFB barriers", fontsize=12)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, out_png)


def plot_water_table(
    cfg: Config, grid: Grid, head: np.ndarray, channels: Channels,
    faults: Sequence[Fault], kper: int, out_png: pathlib.Path,
) -> plt.Figure:
    """Water table, depth to water and transient amplitude."""
    wt = water_table(grid, head)
    extent = (cfg.x_origin, cfg.x_origin + cfg.ncol * cfg.delr,
              cfg.y_origin, cfg.y_origin + cfg.nrow * cfg.delc)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0))
    panels = [
        ("Water-table elevation (m a.s.l.)", wt, "viridis", None),
        ("Land surface (m a.s.l.)", grid.top, "terrain", None),
        ("Depth to water (m below ground)", grid.top - wt, "RdYlBu",
         Normalize(vmin=-3, vmax=12)),
    ]
    for ax, (title, arr, cmap, norm) in zip(axes, panels):
        im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm)
        for path, col in ((channels.stream_path, "cyan"), (channels.river_path, "blue")):
            xy = np.array([grid.cell_center(r, c) for r, c in path])
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.4)
        for flt, col in zip(faults, ("red", "darkorange")):
            for seg in fault_face_geometries(grid, flt):
                xsg, ysg = seg.xy
                ax.plot(xsg, ysg, color=col, lw=1.4, solid_capstyle="butt")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"Water table at stress period {kper}.  Negative depth = water table "
                 "above land surface (see the module docstring)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_png)


def plot_obs_hydrographs(
    cfg: Config, obs: pd.DataFrame, forcing: Forcing, out_png: pathlib.Path
) -> plt.Figure:
    """Hydrographs of the four fault-straddling observation points.

    Each panel shows the pair on either side of one fault plus, on a secondary
    axis, the head *difference* across the barrier.  For the impermeable fault
    that difference is a persistent step of several metres; for the
    semi-permeable fault it is small and strongly modulated by the transient
    stresses.  Reproducing both behaviours is the acid test for a surrogate.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    pairs = [
        ("F1_SIDE_A", "F1_SIDE_B", "Fault 1 - impermeable (hydchr = 1e-8 d$^{-1}$)"),
        ("F2_SIDE_A", "F2_SIDE_B", "Fault 2 - semi-permeable (hydchr = 1e-3 d$^{-1}$)"),
    ]
    colors = [("#1f77b4", "#d62728"), ("#2ca02c", "#ff7f0e")]

    for ax, (a, b, title), (ca, cb) in zip(axes[:2], pairs, colors):
        ax.plot(obs["time_d"], obs[a], color=ca, lw=1.1, label=f"{a} (west side)")
        ax.plot(obs["time_d"], obs[b], color=cb, lw=1.1, label=f"{b} (east side)")
        ax.set_ylabel("Head (m a.s.l.)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax2 = ax.twinx()
        ax2.plot(obs["time_d"], obs[a] - obs[b], color="0.35", lw=0.8, ls="--")
        ax2.set_ylabel("$\\Delta h$ across fault (m)", color="0.35", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="0.35", labelsize=8)

    ax = axes[2]
    ax.plot(forcing.t_mid, forcing.stage, color="#17becf", lw=0.9)
    ax.set_ylabel("Tidal stage (m)", color="#17becf")
    ax.tick_params(axis="y", labelcolor="#17becf")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Time (days)")
    ax3 = ax.twinx()
    ax3.plot(forcing.t_mid, -forcing.wel.sum(axis=0), color="#8c564b", lw=0.8)
    ax3.set_ylabel("Total abstraction (m$^3$/d)", color="#8c564b")
    ax3.tick_params(axis="y", labelcolor="#8c564b")
    ax.set_title("Driving stresses (context)", fontsize=11, loc="left")

    for a in axes:
        a.set_xlim(0, cfg.nper * cfg.dt_days)

    fig.suptitle("Fault-straddling head hydrographs - 2-hourly resolution over 30 days",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_png)


def plot_stream_profile(
    cfg: Config, grid: Grid, channels: Channels, out_png: pathlib.Path
) -> plt.Figure:
    """Planform and long profile of the meandering stream."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})

    extent = (cfg.x_origin, cfg.x_origin + cfg.ncol * cfg.delr,
              cfg.y_origin, cfg.y_origin + cfg.nrow * cfg.delc)
    ax = axes[0]
    im = ax.imshow(grid.top, extent=extent, origin="upper", cmap="terrain")
    xy = np.array([grid.cell_center(r, c) for r, c in channels.stream_path])
    ax.plot(xy[:, 0], xy[:, 1], color="cyan", lw=2.0, label="Central stream (SFR)")
    xy2 = np.array([grid.cell_center(r, c) for r, c in channels.river_path])
    ax.plot(xy2[:, 0], xy2[:, 1], color="blue", lw=2.6, label="Tidal river (RIV)")
    cx, cy = grid.cell_center(*channels.confluence)
    ax.scatter([cx], [cy], s=90, marker="*", color="yellow", edgecolor="k", zorder=6,
               label="Confluence")
    ax.set_title(f"Planform on the land surface - sinuosity "
                 f"{channels.sinuosity(cfg):.3f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.85, label="Land surface (m)")

    ax = axes[1]
    dist = np.arange(len(channels.stream_path)) * cfg.delr / 1000.0
    land = np.array([grid.top[r, c] for r, c in channels.stream_path])
    l1_base = np.array([grid.botm[0, r, c] for r, c in channels.stream_path])
    l2_base = np.array([grid.botm[1, r, c] for r, c in channels.stream_path])
    ax.fill_between(dist, l1_base, land, color="#f4e2b8", label="Recent Alluvium (L1)")
    ax.fill_between(dist, l2_base, l1_base, color="#d9c39a", label="Older Alluvium (L2)")
    ax.plot(dist, land, color="saddlebrown", lw=1.6, label="Land surface")
    ax.plot(dist, channels.stream_rtp, color="#1f77b4", lw=1.8, label="Streambed top")
    ax.axhline(cfg.riv_mean_stage, color="teal", ls=":", lw=1.2,
               label="Mean tidal stage")
    ax.set_xlabel("Distance along the stream (km)")
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_title("Long profile - bed forced monotonically downhill", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Meandering central stream", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_png)


# =============================================================================
# SECTION 11 - DIAGNOSTICS AND REPORT
# =============================================================================


def report_mass_balance(cfg: Config, ws: pathlib.Path) -> Dict[str, float]:
    """Return and print the volumetric budget discrepancy."""
    out = {"max_incremental_pct": float("nan"), "final_cumulative_pct": float("nan")}
    try:
        mf_list = flopy.utils.Mf6ListBudget(str(ws / f"{cfg.model_name}.lst"))
        inc, cum = mf_list.get_budget()
        out["max_incremental_pct"] = float(np.max(np.abs(inc["PERCENT_DISCREPANCY"])))
        out["final_cumulative_pct"] = float(np.abs(cum["PERCENT_DISCREPANCY"][-1]))
        print(f"[budget] max incremental discrepancy = "
              f"{out['max_incremental_pct']:.4f} % | final cumulative discrepancy = "
              f"{out['final_cumulative_pct']:.4f} %")
        if out["max_incremental_pct"] > 1.0:
            print("[budget] WARNING: discrepancy > 1 %, tighten the IMS settings.")
    except Exception as exc:  # noqa: BLE001
        print(f"[budget] could not parse listing file ({exc})")
    return out


def report_benchmark_diagnostics(
    cfg: Config, grid: Grid, faults: Sequence[Fault], head: np.ndarray, time_d: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Quantify *how hard* the generated benchmark actually is.

    * **Fault contrast** - the mean head jump across the blocked faces divided
      by the mean head difference between ordinary neighbouring cells.  A ratio
      near 1 would mean the HFB barriers are invisible and the benchmark is
      pointless; values of several times the background confirm a genuine
      internal discontinuity that a smooth surrogate cannot represent.
    * **Water table above land surface** - reported honestly rather than hidden;
      see the module docstring for why the prescribed parameter set produces it.
    """
    rows = []
    print(f"\n[diagnostics] evaluated on the head field at t = {time_d:.3f} d")
    for k in range(cfg.nlay):
        active = grid.idomain[k] > 0
        if active.sum() < 10:
            continue
        hk = np.where(active, head[k], np.nan)
        dx = np.abs(np.diff(hk, axis=1))
        dy = np.abs(np.diff(hk, axis=0))
        background = np.concatenate([dx.ravel(), dy.ravel()])
        background = background[np.isfinite(background)]
        bg_mean = float(background.mean()) if background.size else np.nan

        for flt in faults:
            jumps = [abs(hk[a] - hk[b]) for a, b in flt.pairs
                     if np.isfinite(hk[a]) and np.isfinite(hk[b])]
            if not jumps:
                continue
            ratio = float(np.mean(jumps) / bg_mean) if bg_mean else np.nan
            rows.append(dict(layer=k + 1, fault=flt.name, hydchr=flt.hydchr,
                             n_faces=len(jumps), dh_mean_m=float(np.mean(jumps)),
                             dh_max_m=float(np.max(jumps)), background_dh_m=bg_mean,
                             contrast_ratio=ratio))
            print(f"  L{k + 1} {flt.name:22s} mean|dh|={np.mean(jumps):6.3f} m  "
                  f"max={np.max(jumps):6.3f} m  background={bg_mean:6.3f} m  "
                  f"contrast={ratio:5.1f}x")

    wt = water_table(grid, head)
    above = np.isfinite(wt) & (wt > grid.top)
    stats = {
        "flooded_fraction_pct": float(100.0 * above.mean()),
        "max_exceedance_m": float(np.nanmax(np.where(above, wt - grid.top, np.nan)))
        if above.any() else 0.0,
        "mean_contrast_ratio": float(np.mean([r["contrast_ratio"] for r in rows]))
        if rows else float("nan"),
    }
    print(f"  water table above land surface in {int(above.sum()):,} of "
          f"{above.size:,} columns ({stats['flooded_fraction_pct']:.1f} %), "
          f"max exceedance {stats['max_exceedance_m']:.2f} m")
    return pd.DataFrame(rows), stats


def load_observations(cfg: Config, ws: pathlib.Path) -> pd.DataFrame:
    """Read the MODFLOW 6 continuous-observation CSV into a DataFrame."""
    df = pd.read_csv(ws / f"{cfg.model_name}.head.obs.csv")
    df.columns = [c.strip().upper() for c in df.columns]
    return df.rename(columns={"TIME": "time_d"})


# ---- report text rendering ------------------------------------------------
# A report page is built from a list of typed blocks rather than one long
# string, because the two kinds of content need opposite treatment: prose must
# be re-flowed to the page width (so it never depends on how the source happens
# to be indented) while tables and code must be reproduced verbatim.  The
# renderer then paginates automatically, which is what stops long sections from
# running off the bottom of a page.

_PAGE_SIZE = (11.69, 8.27)  # A4 landscape, inches
_TEXT_WIDTH = 108  # characters per line
_LINES_PER_PAGE = 50


def _layout_blocks(blocks: Sequence[Tuple[str, str]]) -> List[str]:
    """Flatten typed blocks into a single list of fixed-width text lines.

    Block kinds: ``h`` section heading, ``p`` prose paragraph(s) (re-flowed),
    ``pre`` pre-formatted text reproduced as-is (tables, listings).
    """
    lines: List[str] = []
    for kind, text in blocks:
        if kind == "h":
            if lines:
                lines.append("")
            lines.append(text.upper())
            lines.append("-" * min(len(text), _TEXT_WIDTH))
        elif kind == "p":
            for para in textwrap.dedent(text).strip("\n").split("\n\n"):
                collapsed = " ".join(para.split())
                if not collapsed:
                    continue
                lines.extend("  " + ln for ln in
                             textwrap.wrap(collapsed, width=_TEXT_WIDTH - 2))
                lines.append("")
        elif kind == "pre":
            body = textwrap.dedent(text).strip("\n")
            lines.extend("  " + ln for ln in body.split("\n"))
            lines.append("")
        else:  # pragma: no cover - programming error
            raise ValueError(f"unknown block kind {kind!r}")
    return lines


def _render_text_pages(title: str, blocks: Sequence[Tuple[str, str]],
                       pdf: PdfPages) -> None:
    """Render blocks across as many pages as they need.

    Once the page count is known the lines are spread *evenly* over that many
    pages rather than packed greedily, so a section that overruns by a handful
    of lines produces two balanced pages instead of a full one followed by a
    nearly empty orphan.
    """
    lines = _layout_blocks(blocks)
    n_pages = max(1, -(-len(lines) // _LINES_PER_PAGE))  # ceiling division
    per_page = max(1, -(-len(lines) // n_pages))
    pages = [lines[i:i + per_page] for i in range(0, max(len(lines), 1), per_page)]
    for i, page_lines in enumerate(pages):
        fig = plt.figure(figsize=_PAGE_SIZE)
        heading = title if i == 0 else f"{title}  (continued)"
        fig.text(0.055, 0.95, heading, fontsize=15, weight="bold", va="top")
        fig.text(0.055, 0.885, "\n".join(page_lines), fontsize=7.4,
                 family="monospace", va="top", linespacing=1.4)
        pdf.savefig(fig)
        plt.close(fig)


def build_pdf_report(
    cfg: Config, grid: Grid, fields: PropertyFields, faults: Sequence[Fault],
    channels: Channels, forcing: Forcing,
    obs_cells: Sequence[Tuple[str, Tuple[int, int, int]]],
    figures: Sequence[plt.Figure], diagnostics: pd.DataFrame,
    diag_stats: Dict[str, float], budget: Dict[str, float],
    manifest: pd.DataFrame, run_seconds: Optional[float], out_pdf: pathlib.Path,
) -> None:
    """Assemble the full model report as a multi-page PDF.

    Text pages are rendered with matplotlib so the report needs no LaTeX and no
    extra dependency; the figures are the very same ``Figure`` objects already
    written as PNGs, so the PDF and the PNGs can never drift apart.
    """
    nan = float("nan")
    thick = grid.thickness
    strat = pd.DataFrame([
        dict(
            Layer=k + 1, Unit=LAYER_NAMES[k],
            Top_m=f"{grid.top_of_layer(k)[grid.idomain[k] > 0].min():.1f}..{grid.top_of_layer(k)[grid.idomain[k] > 0].max():.1f}",
            Base_m=f"{grid.botm[k][grid.idomain[k] > 0].min():.1f}..{grid.botm[k][grid.idomain[k] > 0].max():.1f}",
            Thick_m=f"{thick[k][grid.idomain[k] > 0].min():.1f}..{thick[k][grid.idomain[k] > 0].max():.1f}",
            Active=int((grid.idomain[k] > 0).sum()),
            Kh_geomean=f"{cfg.kh_mean[k]:g}",
            Convertible="yes" if k < 2 else "no",
        )
        for k in range(cfg.nlay)
    ])
    fmt = lambda v: f"{v:.4g}"  # noqa: E731
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    with PdfPages(str(out_pdf)) as pdf:
        # ---------------- page 1: title ----------------
        fig = plt.figure(figsize=_PAGE_SIZE)
        fig.text(0.5, 0.74, "Heterogeneous Transient MODFLOW 6 Benchmark",
                 ha="center", fontsize=23, weight="bold")
        fig.text(0.5, 0.675, "A strict test case for PINN and ML groundwater surrogates",
                 ha="center", fontsize=13, color="0.3")
        fig.text(0.5, 0.625, "Forward (simulation) and inverse (parameter estimation) "
                 "problems", ha="center", fontsize=10.5, color="0.45")
        summary = (
            f"Generated            {now}\n"
            f"Realisation seed     {cfg.seed}\n"
            f"Grid                 {cfg.nlay} layers x {cfg.nrow} rows x {cfg.ncol} cols "
            f"@ {cfg.delr:g} m  ({cfg.ncol * cfg.delr / 1000:g} x "
            f"{cfg.nrow * cfg.delc / 1000:g} km)\n"
            f"Active cells         {int((grid.idomain > 0).sum()):,} of {grid.idomain.size:,}\n"
            f"Time                 {cfg.nper} stress periods x {cfg.dt_days * 24:g} h "
            f"= {cfg.nper * cfg.dt_days:g} days, 1 time step each\n"
            f"Packages             DIS NPF STO IC HFB RIV GHB WEL RCHA SFR OBS OC\n"
            f"Solver               IMS COMPLEX / BICGSTAB / DBD under-relaxation, "
            f"Newton formulation\n"
            f"Run time             {f'{run_seconds:.1f} s' if run_seconds else 'not run'}\n"
            f"Budget discrepancy   max incremental {budget.get('max_incremental_pct', nan):.4f} %"
            f", cumulative {budget.get('final_cumulative_pct', nan):.4f} %\n"
            f"Fault head contrast  {diag_stats.get('mean_contrast_ratio', nan):.1f}x the "
            f"background neighbour-cell head gradient\n"
            f"Stream sinuosity     {channels.sinuosity(cfg):.3f}  "
            f"({channels.n_reaches} SFR reaches, {len(channels.river_path)} RIV cells)\n"
        )
        fig.text(0.10, 0.51, summary, fontsize=9, family="monospace", va="top",
                 linespacing=1.75)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- section 1: conceptual model ----------------
        _render_text_pages("1.  Conceptual model", [
            ("h", "Purpose"),
            ("p", """
                A deliberately hostile synthetic aquifer for benchmarking machine-learning
                surrogates.  Every feature below exists to defeat a specific shortcut a
                surrogate might otherwise take: short-range heterogeneity defeats low-rank
                approximation, the fault barriers defeat smooth function representations,
                the meandering channels and the pinched-out layer defeat grid-aligned
                assumptions, and the 2-hour forcing defeats temporal subsampling.
            """),
            ("h", "Domain and geometry"),
            ("p", f"""
                10 x 10 km, {cfg.delr:g} m cells, five layers.  The land surface falls from a
                northern plateau (mean {cfg.plateau_north:g} m) to a southern coastal plain
                (mean {cfg.plateau_south:g} m), carrying a kilometre-scale undulation
                (std {np.sqrt(cfg.topo_undulation_var):.1f} m) and hectometre-scale roughness
                (std {np.sqrt(cfg.topo_roughness_var):.1f} m).  Two meandering watercourses
                incise it: the central stream cuts a valley {cfg.valley_incision:g} m deep and
                the tidal river a floodplain {cfg.river_incision:g} m deep.  Where the two
                overlap the deeper incision wins, which produces a proper confluence rather
                than a double-cut trench.

                Recent Alluvium (Layer 1) fills that incised corridor, thickening downstream
                from {cfg.alluv_thick_north:g} m to {cfg.alluv_thick_south:g} m on the axis and
                tapering to nothing at the margins.  Where the fill is thinner than
                {cfg.alluv_min_thick:g} m the layer is pinched out (idomain = 0), so Layer 1's
                active domain is an irregular meander belt covering
                {100.0 * grid.valley_mask.mean():.1f} % of the plan area - not a rectangle.

                The four deeper contacts are undulating palaeo-surfaces about their mean
                elevations, so layer thickness - and therefore transmissivity and storativity -
                varies smoothly beneath the short-range K noise.  A final pass enforces a
                minimum thickness on every layer so no active cell can be degenerate.
            """),
            ("h", "Stratigraphy"),
            ("pre", strat.to_string(index=False)),
            ("h", "Heterogeneity"),
            ("p", f"""
                gstools exponential-covariance Gaussian random fields, correlation length
                {cfg.len_scale:g} m = {cfg.len_scale / cfg.delr:.0f} cells, var(lnK) = {cfg.var_lnk:g}.
                Kh is log-normal about the per-layer geometric means; Kz = Kh / {cfg.kz_ratio:g};
                Ss is log-normal; Sy is the same correlated field squashed logistically onto
                {cfg.sy_bounds} because a volume fraction cannot be log-normal.  Each layer uses
                an independent seed, so there is zero vertical correlation - deep K cannot be
                inferred from shallow K.

                Layer 1 additionally carries a channel-facies trend: K is boosted by up to
                {1 + cfg.channel_facies_gain:g}x within {cfg.channel_facies_width:g} m of the
                palaeo-channel axis, because coarse channel-lag and point-bar deposits sit
                there while overbank silts sit at the corridor margins.  This gives the inverse
                problem a physically meaningful structure to recover rather than pure noise.
            """),
            ("h", "Faults (HFB, all five layers)"),
            ("pre", f"""
                Fault 1  impermeable     hydchr = {cfg.fault1_hydchr:g} 1/d   "
                         (r{cfg.fault1[0]},c{cfg.fault1[1]}) -> (r{cfg.fault1[2]},c{cfg.fault1[3]})
                Fault 2  semi-permeable  hydchr = {cfg.fault2_hydchr:g} 1/d   "
                         (r{cfg.fault2[0]},c{cfg.fault2[1]}) -> (r{cfg.fault2[2]},c{cfg.fault2[3]})
            """.replace('"', "")),
            ("p", """
                Each oblique trace is discretised as a 4-connected staircase of shared cell
                faces, which is the only construct MODFLOW's HFB package accepts: diagonally
                adjacent cells do not share a face and cannot host a barrier.
            """),
        ], pdf)

        # ---------------- section 2: boundaries, stresses, numerics ------------
        riv_cond = cfg.riv_bed_k * cfg.riv_width * cfg.delr / cfg.riv_bed_thick
        _render_text_pages("2.  Boundary conditions, stresses and numerics", [
            ("h", "RIV - meandering tidal river"),
            ("p", f"""
                {len(channels.river_path)} cells tracing a two-harmonic meander across the
                southern domain, placed in the uppermost active of Layers 1-2.  Stage follows an
                M2 semi-diurnal tide: mean {cfg.riv_mean_stage:g} m, amplitude
                {cfg.riv_amplitude:g} m, period {cfg.riv_period_h:g} h, modulated by a
                {cfg.riv_spring_neap_d:g}-day spring-neap envelope.  With 2-hour stress periods
                the tide is sampled only ~6.2 times per cycle - close to Nyquist, so a surrogate
                that coarsens time will alias it.  Bed at {cfg.riv_bottom:g} m, conductance
                {riv_cond:,.0f} m2/d per cell.
            """),
            ("h", "SFR - meandering central stream"),
            ("p", f"""
                {channels.n_reaches} reaches from the northern headwater to the confluence,
                sinuosity {channels.sinuosity(cfg):.3f}.  The bed falls monotonically from
                {channels.stream_rtp[0]:.2f} m to {channels.stream_rtp[-1]:.2f} m (a running
                minimum enforces this - a stream cannot flow uphill and SFR needs a strictly
                positive gradient).  Width {cfg.sfr_width:g} m, Manning {cfg.sfr_manning:g},
                bed K {cfg.sfr_bed_k:g} m/d.

                Head-reach inflow is baseflow ({cfg.sfr_base_inflow:,.0f} m3/d) plus three
                asymmetric flash floods peaking at {max(f[1] for f in FLOOD_EVENTS):,.0f} m3/d.
                Two of the floods follow storms; one does not, so stream inflow cannot be
                inferred from the rainfall signal.  The downstream reach discharges out of the
                model at the confluence - MODFLOW cannot route a mover from SFR into RIV,
                because RIV is not an advanced package.
            """),
            ("h", "GHB - eastern regional inflow"),
            ("p", f"""
                Column {cfg.ghb_col}, Layers 2-4, head {cfg.ghb_head:g} m.  Conductance is
                computed cell-by-cell from the local heterogeneous Kh and the local layer
                thickness, so even the boundary flux inherits the geostatistical roughness and
                the undulating stratigraphy.  Cells shared with the tidal river are skipped so
                the two head-dependent boundaries never fight over one cell.
            """),
            ("h", "WEL - production wells"),
            ("p", f"""
                {len(WELLS)} wells in Layer 2 within the alluvial corridor.  Diurnal schedule
                (night baseline 0.25, smooth daytime peak 06:00-18:00), a weekend reduction to
                45 %, per-well phase offsets so the composite is not a single clean harmonic,
                and two hard step changes: W4 starts at day 12, W6 stops at day 20.  Total
                abstraction swings between {-forcing.wel.sum(axis=0).max():,.0f} and
                {-forcing.wel.sum(axis=0).min():,.0f} m3/d.
            """),
            ("h", "RCH - areal recharge"),
            ("p", f"""
                Base {cfg.rch_base:g} m/d with two intense 12-hour storms at
                {cfg.storm_windows_d[0][0]:g}-{cfg.storm_windows_d[0][1]:g} d and
                {cfg.storm_windows_d[1][0]:g}-{cfg.storm_windows_d[1][1]:g} d at
                {cfg.storm_rate:g} m/d (30 mm each).  IRCH is set explicitly to the uppermost
                active layer: MODFLOW's array recharge does NOT fall through a pinched-out top
                layer, and omitting IRCH silently loses all recharge outside the meander belt.
            """),
            ("h", "Numerics"),
            ("p", """
                Newton formulation with under-relaxation; IMS COMPLEX complexity, BICGSTAB
                acceleration, Delta-Bar-Delta under-relaxation and backtracking.  Layers 1-2 are
                convertible, 3-5 confined.  Initial heads come from a steady-state spin-up under
                base forcing - base recharge and median baseflow, because using the series means
                would fold the storms and the floods into the initial condition and start the
                benchmark from an artificially wet aquifer.
            """),
            ("h", "Result quality"),
            ("pre", f"""
                Maximum incremental budget discrepancy   {budget.get('max_incremental_pct', nan):.4f} %
                Final cumulative budget discrepancy      {budget.get('final_cumulative_pct', nan):.4f} %
                Mean fault head-jump contrast            {diag_stats.get('mean_contrast_ratio', nan):.1f}x background
                Water table above land surface           {diag_stats.get('flooded_fraction_pct', nan):.1f} % of columns
                                                         (max {diag_stats.get('max_exceedance_m', nan):.2f} m)
            """),
            ("h", "Caveat on the prescribed parameter set"),
            ("p", """
                The prescribed base recharge over 100 km2 is large relative to what the
                prescribed transmissivities can route to the two outlets.  In the small
                fault-dammed compartment behind the impermeable fault the water table therefore
                still reaches land surface, and because MODFLOW switches those cells to confined
                storage above the layer top, the storms produce head spikes of several metres
                there.  That is a property of the parameter set, not a numerical failure - the
                budget closes to 0.00 %.  Physically it is where springs and seepage faces would
                form.  Lower Config.rch_base or add a DRN seepage package at land surface if a
                strictly sub-surface water table is required everywhere.
            """),
        ], pdf)

        # ---------------- section 3: properties and diagnostics ----------------
        obs_table = "\n".join(
            f"{name:12s}  layer {cid[0] + 1}   row {cid[1]:3d}   col {cid[2]:3d}"
            for name, cid in obs_cells
        )
        _render_text_pages("3.  Properties and diagnostics", [
            ("h", "Hydraulic property statistics (active cells only)"),
            ("pre", fields.summary(grid).to_string(index=False, float_format=fmt)),
            ("h", "Fault head-jump diagnostics"),
            ("p", """
                Mean absolute head difference across the blocked cell faces, compared with the
                mean head difference between ordinary neighbouring cells in the same layer.  A
                contrast ratio near 1 would mean the barriers are invisible and the benchmark
                pointless; these values confirm a genuine internal discontinuity that a smooth
                neural representation cannot reproduce.
            """),
            ("pre", diagnostics.to_string(index=False, float_format=fmt)
             if len(diagnostics) else "(not available - model not run)"),
            ("h", "Observation points"),
            ("p", """
                Four head observations, two on each side of the middle blocked face of each
                fault.  The paired cells are 100 m apart but separated by the barrier, so their
                difference measures the discontinuity directly, at 2-hourly resolution.
            """),
            ("pre", obs_table),
        ], pdf)

        # ---------------- figure pages ----------------
        for fg in figures:
            pdf.savefig(fg)

        # ---------------- section 4: data inventory ----------------
        by_folder = manifest.groupby("folder").agg(
            files=("file", "count"),
            size_mb=("size_kb", lambda s: round(s.sum() / 1024, 1)),
        )
        gis_files = manifest[manifest.folder == "gis"]
        rasters = gis_files[gis_files.file.str.endswith(".tif")]
        shapes = gis_files[gis_files.file.str.endswith(".shp")]
        tabs = manifest[manifest.folder.isin(["tables", "arrays"])]

        _render_text_pages("4.  Data inventory for surrogate training", [
            ("p", f"""
                Every field the benchmark uses or produces is written to georeferenced GIS files
                in EPSG:{cfg.crs_epsg}, so a surrogate can be trained, validated and inspected
                without ever reading a MODFLOW binary.
            """),
            ("pre", by_folder.to_string()),
            ("h", "For the forward (simulation) problem"),
            ("pre", f"""
                inputs   kh_L*, kz_L*, ss_L*, sy_L*, top_L*, botm_L*, thickness_L*, idomain_L*,
                         icelltype_L*, land_surface, alluvium_thickness, recharge_layer_irch,
                         strt_L*  (the initial condition)
                forcing  tables/forcing_by_stress_period.csv - stage, stream inflow, recharge
                         and all 8 well rates for each of the {cfg.nper} stress periods
                targets  head_timeseries_L*.tif  (multi-band, band n = stress period n)
                         arrays/heads_transient.npz  (nper, nlay, nrow, ncol) float32
            """),
            ("h", "For the inverse (parameter estimation) problem"),
            ("pre", """
                unknowns kh_L*, ss_L*, sy_L*  (and the fault hydchr values in faults_trace.shp)
                data     head_timeseries_L*.tif, watertable_timeseries.tif,
                         tables/fault_observations.csv (dense 2-hourly hydrographs),
                         qx_L*/qy_L*/qz_L* specific discharge for Darcy-residual terms
                priors   dist_to_stream, dist_to_river, dist_to_fault1, dist_to_fault2,
                         dist_to_any_fault, dist_to_nearest_well, valley_corridor_mask -
                         the covariates that carry the structural information
            """),
            ("p", """
                The PINN collocation inputs x_coordinate.tif and y_coordinate.tif give
                cell-centre coordinates in metres on the same grid, so (x, y, t) samples can be
                drawn directly without re-deriving the affine transform.
            """),
            ("h", f"Rasters ({len(rasters)} GeoTIFFs)"),
            ("pre", rasters[["file", "description"]].to_string(index=False, max_colwidth=58)),
        ], pdf)

        _render_text_pages("5.  Data inventory (continued)", [
            ("h", f"Shapefiles ({len(shapes)})"),
            ("pre", shapes[["file", "description"]].to_string(index=False, max_colwidth=62)),
            ("h", "Tables and arrays"),
            ("pre", tabs[["file", "size_kb", "description"]].to_string(index=False,
                                                                      max_colwidth=52)),
            ("h", "Reproducing this dataset"),
            ("pre", f"""
                python scripts/build_mf6_hetero_benchmark.py --seed {cfg.seed}
            """),
            ("p", """
                Change --seed for an independent realisation of the same conceptual model: the
                geometry, the channel planforms and the faults are deterministic, while the K,
                Ss, Sy and geological-surface fields are not.  --nper shortens the run for smoke
                tests and --no-run stops after writing the model and the static exports.
            """),
        ], pdf)

        meta = pdf.infodict()
        meta["Title"] = "Heterogeneous Transient MODFLOW 6 Benchmark for ML Surrogates"
        meta["Subject"] = f"Model report, realisation seed {cfg.seed}"
        meta["Creator"] = "build_mf6_hetero_benchmark.py"

    print(f"[report] {out_pdf}")

def export_head_tensor(
    cfg: Config, grid: Grid, head_stack: np.ndarray, times: np.ndarray,
    arrays_dir: pathlib.Path,
) -> None:
    """Dump the full head tensor for surrogate training.

    Stored as float32 in a compressed ``.npz`` together with the time vector,
    the active-cell mask and the layer geometry, so a downstream training script
    needs nothing but NumPy to consume the benchmark.
    """
    out = arrays_dir / "heads_transient.npz"
    np.savez_compressed(
        out, head=head_stack, time_days=times.astype(np.float32),
        idomain=grid.idomain.astype(np.int8), top=grid.top.astype(np.float32),
        botm=grid.botm.astype(np.float32),
    )
    print(f"[arrays] head tensor {head_stack.shape} -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB compressed)")


def load_head_stack(
    cfg: Config, grid: Grid, ws: pathlib.Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Read every saved head record into one (nper, nlay, nrow, ncol) float32 array."""
    hds = flopy.utils.HeadFile(str(ws / f"{cfg.model_name}.hds"))
    times = np.array(hds.get_times())
    stack = np.empty((len(times), cfg.nlay, cfg.nrow, cfg.ncol), dtype=np.float32)
    for i, t in enumerate(times):
        stack[i] = hds.get_data(totim=t).astype(np.float32)
    hds.close()

    # Sentinel values (inactive / dry) -> NaN so they cannot pollute training.
    stack[np.broadcast_to(grid.idomain <= 0, stack.shape)] = np.nan
    stack[np.abs(stack) > 1.0e20] = np.nan
    return stack, times


# =============================================================================
# SECTION 12 - MAIN
# =============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Build, run, export and document a heterogeneous transient "
                    "MODFLOW 6 benchmark for ML / PINN surrogate models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-w", "--workspace",
                   default=str(repo_root / "runs" / "mf6_hetero_benchmark"),
                   help="Output root directory.")
    p.add_argument("--seed", type=int, default=CFG.seed,
                   help="Master seed for the geostatistical realisation.")
    p.add_argument("--nper", type=int, default=CFG.nper,
                   help="Number of 2-hour stress periods (360 = the full 30-day "
                        "benchmark; lower values are useful for smoke tests).")
    p.add_argument("--no-run", action="store_true",
                   help="Write the model and the static exports but do not execute "
                        "MODFLOW.")
    p.add_argument("--no-spinup", action="store_true",
                   help="Skip the steady-state spin-up and start from a flat head.")
    p.add_argument("--no-npz", action="store_true",
                   help="Skip writing the full head tensor npz.")
    p.add_argument("--map-kper", type=int, default=180,
                   help="1-based stress period used for the head map and snapshots.")
    p.add_argument("--report-copy",
                   default=str(repo_root / "docs" / "mf6_hetero_benchmark_report.pdf"),
                   help="Where to copy the PDF report inside the repository "
                        "(empty string disables the copy).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = Config(seed=args.seed, nper=max(1, args.nper))
    root = pathlib.Path(args.workspace).resolve()
    paths = make_workspace(root)

    print("=" * 78)
    print(" HETEROGENEOUS TRANSIENT MODFLOW 6 BENCHMARK")
    print("=" * 78)
    print(f" workspace : {root}")
    print(f" grid      : {cfg.nlay} x {cfg.nrow} x {cfg.ncol} @ {cfg.delr:g} m "
          f"({cfg.ncol * cfg.delr / 1000:g} x {cfg.nrow * cfg.delc / 1000:g} km)")
    print(f" time      : {cfg.nper} stress periods x {cfg.dt_days * 24:g} h = "
          f"{cfg.nper * cfg.dt_days:g} d")
    print(f" seed      : {cfg.seed}")
    print("=" * 78)

    exe = resolve_mf6_executable(bindir=root / "bin")

    # ---- 1. static structure -------------------------------------------------
    channels = build_channels(cfg)
    grid = build_grid(cfg, channels)
    fields = build_property_fields(cfg, grid)
    faults = build_faults(cfg)
    obs_cells = fault_observation_cells(grid, faults, obs_layer=1)
    forcing = build_forcing(cfg)

    # ---- 2. tables (surrogate feature matrices) ------------------------------
    forcing.to_dataframe().to_csv(
        paths["tables"] / "forcing_by_stress_period.csv", index=False)
    fields.summary(grid).to_csv(
        paths["tables"] / "property_statistics.csv", index=False)
    pd.DataFrame(
        [dict(name=w["name"], layer=2, row=w["row"], col=w["col"],
              x=grid.cell_center(w["row"], w["col"])[0],
              y=grid.cell_center(w["row"], w["col"])[1],
              land_elev=float(grid.top[w["row"], w["col"]]),
              q_base_m3d=w["base"], phase_h=w["phase_h"],
              start_d=w["start_d"], stop_d=w["stop_d"])
         for w in WELLS]
    ).to_csv(paths["tables"] / "wells.csv", index=False)
    pd.DataFrame(
        [dict(name=name, layer=cid[0] + 1, row=cid[1], col=cid[2],
              x=grid.cell_center(cid[1], cid[2])[0],
              y=grid.cell_center(cid[1], cid[2])[1])
         for name, cid in obs_cells]
    ).to_csv(paths["tables"] / "observation_points.csv", index=False)
    pd.DataFrame(
        dict(
            ifno=np.arange(channels.n_reaches),
            row=[r for r, _ in channels.stream_path],
            col=[c for _, c in channels.stream_path],
            dist_m=np.arange(channels.n_reaches) * cfg.delr,
            land_elev_m=[grid.top[r, c] for r, c in channels.stream_path],
            streambed_m=channels.stream_rtp,
            gradient=channels.stream_grad,
        )
    ).to_csv(paths["tables"] / "stream_long_profile.csv", index=False)
    print(f"[tables] wrote 5 CSV tables -> {paths['tables']}")

    # ---- 3. static figures ---------------------------------------------------
    figures: List[plt.Figure] = []
    figures.append(plot_topography_geology(
        cfg, grid, channels, faults, paths["figures"] / "fig_topography_geology.png"))
    figures.append(plot_k_field(
        cfg, grid, fields, faults, channels, paths["figures"] / "fig_k_fields.png"))
    figures.append(plot_stream_profile(
        cfg, grid, channels, paths["figures"] / "fig_stream_profile.png"))
    figures.append(plot_forcing(
        cfg, forcing, paths["figures"] / "fig_forcing_timeseries.png"))

    # ---- 4. initial condition ------------------------------------------------
    if args.no_run or args.no_spinup:
        strt = np.broadcast_to(
            np.maximum(grid.top - 2.0, cfg.riv_mean_stage),
            (cfg.nlay, cfg.nrow, cfg.ncol)).copy()
        print("[spinup] skipped - starting from a land-surface-following head field.")
    else:
        strt = steady_state_initial_heads(
            cfg, grid, fields, faults, channels, forcing, obs_cells,
            paths["warmup"], exe)

    # ---- 5. transient benchmark ---------------------------------------------
    print("\n=== TRANSIENT BENCHMARK SIMULATION ===")
    sim = build_simulation(cfg, grid, fields, faults, channels, forcing, obs_cells,
                           paths["sim"], exe, strt, transient=True)
    sim.write_simulation(silent=True)
    print(f"[write] simulation written to {paths['sim']}")
    gwf = sim.get_model(cfg.model_name)

    # ---- 6. static GIS exports ----------------------------------------------
    export_static_rasters(cfg, grid, fields, faults, channels, strt, paths["gis"])
    export_shapefiles(cfg, grid, fields, faults, channels, forcing, obs_cells,
                      paths["gis"])

    if args.no_run:
        figures.append(plot_geological_sections(
            cfg, grid, faults, channels, gwf, None,
            paths["figures"] / "fig_geological_sections.png"))
        figures.append(plot_k_sections(
            cfg, grid, fields, faults, channels, gwf, None,
            paths["figures"] / "fig_k_sections.png"))
        manifest = write_manifest(paths)
        build_pdf_report(cfg, grid, fields, faults, channels, forcing, obs_cells,
                         figures, pd.DataFrame(), {}, {}, manifest, None,
                         paths["report"] / "model_report.pdf")
        print("[run] --no-run requested; stopping before execution.")
        return 0

    t_start = _dt.datetime.now()
    success, buff = sim.run_simulation(silent=False, report=True)
    run_seconds = (_dt.datetime.now() - t_start).total_seconds()
    if not success:
        print("\n[run] MODFLOW 6 FAILED. Tail of the simulation output:")
        for line in buff[-40:]:
            print("   " + str(line).rstrip())
        return 1
    print(f"[run] MODFLOW 6 completed normally in {run_seconds:.1f} s.")

    budget = report_mass_balance(cfg, paths["sim"])

    # ---- 7. post-processing --------------------------------------------------
    head_stack, times = load_head_stack(cfg, grid, paths["sim"])
    map_kper = int(np.clip(args.map_kper, 1, len(times)))
    head_map = head_stack[map_kper - 1].astype(float)
    time_d = float(times[map_kper - 1])

    obs = load_observations(cfg, paths["sim"])
    obs.to_csv(paths["tables"] / "fault_observations.csv", index=False)

    figures.append(plot_geological_sections(
        cfg, grid, faults, channels, gwf, head_map,
        paths["figures"] / "fig_geological_sections.png"))
    figures.append(plot_k_sections(
        cfg, grid, fields, faults, channels, gwf, head_map,
        paths["figures"] / "fig_k_sections.png"))
    figures.append(plot_head_map(
        cfg, grid, gwf, head_map, faults, channels, obs_cells, map_kper, time_d,
        paths["figures"] / f"fig_head_layer2_sp{map_kper}.png"))
    figures.append(plot_water_table(
        cfg, grid, head_map, channels, faults, map_kper,
        paths["figures"] / f"fig_water_table_sp{map_kper}.png"))
    figures.append(plot_obs_hydrographs(
        cfg, obs, forcing, paths["figures"] / "fig_obs_hydrographs.png"))

    diagnostics, diag_stats = report_benchmark_diagnostics(
        cfg, grid, faults, head_map, time_d)
    diagnostics.to_csv(paths["tables"] / "fault_contrast_diagnostics.csv", index=False)

    # ---- 8. simulated-state GIS exports -------------------------------------
    export_head_rasters(cfg, grid, head_stack, times, strt, gwf, paths["sim"],
                        map_kper, paths["gis"])
    if not args.no_npz:
        export_head_tensor(cfg, grid, head_stack, times, paths["arrays"])

    # ---- 9. report -----------------------------------------------------------
    manifest = write_manifest(paths)
    report_pdf = paths["report"] / "model_report.pdf"
    build_pdf_report(cfg, grid, fields, faults, channels, forcing, obs_cells,
                     figures, diagnostics, diag_stats, budget, manifest,
                     run_seconds, report_pdf)
    if args.report_copy:
        dest = pathlib.Path(args.report_copy)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report_pdf, dest)
        print(f"[report] copied to {dest}")

    for fg in figures:
        plt.close(fg)

    print("\n" + "=" * 78)
    print(" BENCHMARK COMPLETE")
    print("=" * 78)
    for key in ("sim", "gis", "tables", "arrays", "figures", "report"):
        print(f"  {key:8s} -> {paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
