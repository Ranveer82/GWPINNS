#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 HETEROGENEOUS TRANSIENT MODFLOW 6 BENCHMARK FOR ML SURROGATE MODELS
================================================================================

This script builds, runs and post-processes a *deliberately hostile* conceptual
groundwater model.  It is not meant to be a calibrated representation of a real
basin - it is meant to be a **strict benchmark** for machine-learning surrogates
(PINNs, FNOs, CNN-LSTM emulators, DeepONets, ...).

Why every design choice below makes life hard for a surrogate
-------------------------------------------------------------
1.  SHORT-RANGE HETEROGENEITY.  The geostatistical correlation length is
    200 m = 2 cells.  A surrogate cannot get away with learning a smooth,
    low-rank K-field; it has to resolve near-cell-scale structure.
2.  SHARP INTERNAL DISCONTINUITIES.  Two Horizontal-Flow-Barrier faults cut all
    five layers.  One is effectively impermeable (1e-8 d^-1) and produces a head
    *jump* across a single cell face - a genuine discontinuity that smooth
    neural representations (and PINN residual losses) struggle with.
3.  PINCHED-OUT STRATIGRAPHY.  Layer 1 exists only inside the incised valley
    (idomain = 0 elsewhere), so the surrogate must handle a non-rectangular,
    layer-dependent active domain.
4.  FAST, MULTI-SCALE FORCING.  360 stress periods of exactly 2 hours each.  The
    tidal river carries a 12.4-h semi-diurnal signal (so the response is
    resolved by only ~6 stress periods per cycle), pumping follows a diurnal
    cycle with abrupt on/off events, recharge has two 12-h convective storms and
    the stream carries flash-flood hydrographs.  Temporal aliasing is a real
    risk for any surrogate that subsamples time.
5.  MIXED BOUNDARY PHYSICS.  RIV (head-dependent, time-varying stage), SFR
    (fully routed surface water with its own stage-discharge non-linearity),
    GHB (regional inflow), WEL (point sinks), RCH (areal source).  The surrogate
    must learn several different flux laws at once.
6.  UNCONFINED / CONFINED SWITCHING.  Layers 1-2 are convertible, so storage
    coefficients change in time and the PDE is non-linear.

Known behaviour of the prescribed conceptual model
--------------------------------------------------
The specified base recharge (5e-4 m/d over 100 km^2 = 50 000 m^3/d) is larger
than what the specified transmissivities (bulk sum T ~ 6e2 m^2/d) can route to
the only two outlets - the southern tidal river and the central stream - from
the most remote, fault-compartmentalised parts of the plateau.  The steady
water table therefore sits at or slightly above the 50 m ground surface over
roughly 10 % of the domain, and because MODFLOW switches those cells to
*confined* storage above the layer top, the two 12-hour storms produce head
spikes of several metres there rather than the ~0.2 m a specific-yield response
would give.  This is a property of the prescribed parameter set, not a
numerical failure: the run converges with a 0.00 % volumetric budget
discrepancy, and ``report_benchmark_diagnostics`` prints the exceedance every
time.  If a strictly sub-surface water table is wanted, either lower
``Config.rch_base`` (~1e-4 m/d balances) or add a DRN "spring/seepage" package
at land surface to reject the excess.

Outputs (all written under the run workspace)
---------------------------------------------
  * A complete MODFLOW 6 simulation (written, run and mass-balance checked)
  * ``gis/*.tif``   - GeoTIFF rasters of every static input field (K, Ss, Sy,
                      tops, bottoms, idomain, recharge-layer map) plus the
                      simulated head snapshot - ready to be stacked into a
                      CNN input tensor.
  * ``gis/*.shp``   - ESRI Shapefiles of faults, river cells, stream reaches,
                      GHB cells, wells, observation points and the domain /
                      valley outlines - ready for distance-to-feature and
                      rasterised-mask feature engineering.
  * ``tables/*.csv``- Every boundary-condition time series, stress period by
                      stress period (the surrogate's forcing matrix).
  * ``arrays/*.npz``- The full head tensor (nper, nlay, nrow, ncol) as float32,
                      i.e. the surrogate's training target.
  * ``figures/*.png`` - Diagnostic plots (fault hydrographs, head contour map,
                      forcing time series, log-K field).

Usage
-----
    python scripts/build_mf6_hetero_benchmark.py                 # build + run + export
    python scripts/build_mf6_hetero_benchmark.py --no-run        # build + export only
    python scripts/build_mf6_hetero_benchmark.py --seed 12345    # different realisation
    python scripts/build_mf6_hetero_benchmark.py -w /tmp/bench   # custom workspace

Dependencies: flopy, numpy, pandas, geopandas, rasterio, shapely, gstools,
matplotlib.  The MODFLOW 6 executable is resolved automatically.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import stat
import sys
import tempfile
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
from matplotlib.colors import LogNorm
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

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

    # Synthetic georeference.  A real projected CRS is used so that the exported
    # GeoTIFFs/Shapefiles line up in any GIS and so that distance-based feature
    # engineering (metres) is meaningful.  UTM 44N, arbitrary origin.
    crs_epsg: int = 32644
    x_origin: float = 500_000.0  # model (0,0) lower-left easting
    y_origin: float = 3_000_000.0  # model (0,0) lower-left northing

    # Incised valley: columns [valley_c0, valley_c1] inclusive
    valley_c0: int = 40
    valley_c1: int = 60

    # Stratigraphic surfaces (m a.s.l.)
    top_plateau: float = 50.0  # ground surface outside the valley
    valley_floor: float = 35.0  # base of Layer 1 = top of Layer 2 in valley
    botm_l2: float = 20.0
    botm_l3: float = 0.0
    botm_l4: float = -50.0
    botm_l5: float = -150.0

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

    # ---------------- tidal river (RIV) ----------------
    riv_row: int = 99
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
    sfr_col: int = 50
    sfr_row0: int = 0
    sfr_row1: int = 98  # inclusive -> 99 reaches
    sfr_top_up: float = 46.0  # streambed top at row 0
    sfr_top_dn: float = 40.5  # streambed top at row 98
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
# incised valley.  ``base`` is the mean daily abstraction in m3/d.  ``phase_h``
# shifts each well's diurnal peak so the composite stress is not a single clean
# harmonic - a surrogate cannot fit one sinusoid and be done.  ``start_d`` /
# ``stop_d`` create abrupt step changes mid-simulation (a well switched on at
# day 12, another shut down at day 20) which are the hardest events for a
# time-series surrogate to anticipate.
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
# recession time-constant in days).  Spike 1 follows storm 1, spike 3 follows
# storm 2, and spike 2 is *uncorrelated* with recharge so the surrogate cannot
# simply infer stream inflow from the rainfall signal.
FLOOD_EVENTS: Tuple[Tuple[float, float, float, float], ...] = (
    (6.6, 120_000.0, 4.0, 1.2),
    (13.2, 45_000.0, 3.0, 0.6),
    (21.1, 200_000.0, 5.0, 1.8),
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
    # --- 1. explicit override -------------------------------------------------
    env_exe = os.environ.get("MF6_EXE")
    if env_exe and _is_working_mf6(pathlib.Path(env_exe)):
        print(f"[mf6] using MF6_EXE -> {env_exe}")
        return env_exe

    # --- 2. PATH --------------------------------------------------------------
    which = shutil.which("mf6")
    if which:
        print(f"[mf6] found on PATH -> {which}")
        return which

    # --- 3. common install locations -----------------------------------------
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

    # --- 4. official FloPy installer -----------------------------------------
    try:
        print("[mf6] not found; trying flopy.utils.get_modflow() ...")
        from flopy.utils import get_modflow

        get_modflow(str(target), subset="mf6", quiet=False)
        if _is_working_mf6(target / "mf6"):
            print(f"[mf6] installed via get_modflow -> {target / 'mf6'}")
            return str(target / "mf6")
    except Exception as exc:  # noqa: BLE001 - any failure falls through to (5)
        print(f"[mf6] get_modflow() failed ({exc.__class__.__name__}: {exc})")

    # --- 5. direct release-asset download ------------------------------------
    asset = {"linux": "linux.zip", "darwin": "mac.zip", "win32": "win64.zip"}.get(
        "linux"
        if sys.platform.startswith("linux")
        else ("darwin" if sys.platform == "darwin" else "win32")
    )
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
            exe = target / ("mf6.exe" if sys.platform == "win32" else "mf6")
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
# SECTION 2 - GRID, STRATIGRAPHY AND GEOREFERENCING
# =============================================================================


@dataclass
class Grid:
    """Structured grid geometry + stratigraphy + active-cell bookkeeping."""

    cfg: Config
    top: np.ndarray  # (nrow, ncol)
    botm: np.ndarray  # (nlay, nrow, ncol)
    idomain: np.ndarray  # (nlay, nrow, ncol) int
    valley_mask: np.ndarray  # (nrow, ncol) bool
    xc: np.ndarray  # (ncol,) cell-centre easting
    yc: np.ndarray  # (nrow,) cell-centre northing (row 0 = north)
    transform: rasterio.Affine  # north-up raster transform

    # ------------------------------------------------------------------ #
    @property
    def shape2d(self) -> Tuple[int, int]:
        return self.cfg.nrow, self.cfg.ncol

    @property
    def thickness(self) -> np.ndarray:
        """(nlay, nrow, ncol) saturated-geometry thickness of each layer."""
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


def build_grid(cfg: Config) -> Grid:
    """Assemble the five-layer incised-valley stratigraphy.

    The valley is the only place Layer 1 (Recent Alluvium) exists.  Outside the
    valley Layer 1 is *pinched out*: it is given zero thickness and switched off
    with ``idomain = 0``.  MODFLOW 6 only enforces positive thickness on active
    cells, so a zero-thickness inactive layer is the canonical way to represent
    a pinch-out on a structured grid while keeping ``botm[0]`` a valid top for
    Layer 2 (= 50 m outside, 35 m inside).
    """
    nrow, ncol, nlay = cfg.nrow, cfg.ncol, cfg.nlay

    valley = np.zeros((nrow, ncol), dtype=bool)
    valley[:, cfg.valley_c0 : cfg.valley_c1 + 1] = True

    # Model top is the plateau surface everywhere; inside the valley the true
    # ground surface is also 50 m (the valley is *incised* into the section
    # below, i.e. the alluvium fill reaches the plateau level).
    top = np.full((nrow, ncol), cfg.top_plateau, dtype=float)

    botm = np.empty((nlay, nrow, ncol), dtype=float)
    # Layer 1: 50 -> 35 in the valley, zero thickness (50 -> 50) outside.
    botm[0] = np.where(valley, cfg.valley_floor, cfg.top_plateau)
    botm[1] = cfg.botm_l2  # Older Alluvium base, global
    botm[2] = cfg.botm_l3  # Massive Carbonate 1 base
    botm[3] = cfg.botm_l4  # Massive Carbonate 2 base
    botm[4] = cfg.botm_l5  # Fractured Granite base

    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    idomain[0] = valley.astype(int)  # Layer 1 exists only in the valley

    # Cell centres.  Row 0 is the northern-most row (raster convention).
    xc = cfg.x_origin + (np.arange(ncol) + 0.5) * cfg.delr
    yc = cfg.y_origin + (nrow - np.arange(nrow) - 0.5) * cfg.delc

    transform = from_origin(
        cfg.x_origin, cfg.y_origin + nrow * cfg.delc, cfg.delr, cfg.delc
    )

    return Grid(
        cfg=cfg,
        top=top,
        botm=botm,
        idomain=idomain,
        valley_mask=valley,
        xc=xc,
        yc=yc,
        transform=transform,
    )


# =============================================================================
# SECTION 3 - GEOSTATISTICS (gstools)
# =============================================================================


@dataclass
class PropertyFields:
    """Per-layer hydraulic property realisations."""

    kh: np.ndarray  # (nlay, nrow, ncol) m/d
    k33: np.ndarray  # (nlay, nrow, ncol) m/d
    ss: np.ndarray  # (nlay, nrow, ncol) 1/m
    sy: np.ndarray  # (nlay, nrow, ncol) -


def _gaussian_field(cfg: Config, seed: int) -> np.ndarray:
    """One zero-mean, unit-structure Gaussian random field on the model grid.

    An **exponential** covariance is used on purpose: it is far rougher than a
    Gaussian covariance at short lag, so combined with ``len_scale = 200 m``
    (two cells) it produces the near-white, high-contrast fields that make this
    a hard benchmark.  A smooth (Gaussian-covariance) field would be trivially
    compressible by a low-rank surrogate.
    """
    model = gs.Exponential(dim=2, var=cfg.var_lnk, len_scale=cfg.len_scale)
    srf = gs.SRF(model, mean=0.0, seed=seed)

    # gstools works in ascending-coordinate order; ``structured`` returns
    # (nx, ny).  Transpose to (ny, nx) then flip so that row 0 = north, which
    # matches both MODFLOW's row convention and the GeoTIFF transform.
    x = (np.arange(cfg.ncol) + 0.5) * cfg.delr
    y = (np.arange(cfg.nrow) + 0.5) * cfg.delc
    fld = srf.structured((x, y))
    return np.flipud(np.asarray(fld).T)


def build_property_fields(cfg: Config, grid: Grid) -> PropertyFields:
    """Generate correlated log-normal Kh / Ss and bounded Sy fields per layer.

    * ``Kh = kh_mean * exp(Z)`` with ``Z ~ N(0, var_lnk)``  -> ``kh_mean`` is the
      *geometric* mean; with var = 2.0 the 5th-95th percentile spans roughly two
      and a half orders of magnitude within a single layer.
    * ``Ss`` uses the same recipe (log-normal is physically appropriate).
    * ``Sy`` cannot be log-normal - it is a bounded volume fraction.  The same
      correlated field is therefore squashed through a logistic map onto
      ``sy_bounds`` while preserving the spatial structure (and hence the
      correlation between high-K and high-Sy zones, which is what a surrogate
      would exploit).
    * Each layer gets an independent seed, so vertical correlation is zero.  A
      surrogate cannot infer deep K from shallow K.
    """
    nlay, nrow, ncol = cfg.nlay, cfg.nrow, cfg.ncol
    kh = np.empty((nlay, nrow, ncol))
    ss = np.empty((nlay, nrow, ncol))
    sy = np.empty((nlay, nrow, ncol))

    for k in range(nlay):
        z_k = _gaussian_field(cfg, seed=cfg.seed + 1000 * k + 1)
        z_s = _gaussian_field(cfg, seed=cfg.seed + 1000 * k + 2)
        z_y = _gaussian_field(cfg, seed=cfg.seed + 1000 * k + 3)

        kh[k] = cfg.kh_mean[k] * np.exp(z_k)

        # Ss: same log-normal treatment, scaled to keep the variance sane.
        ss[k] = cfg.ss_mean[k] * np.exp(np.sqrt(cfg.var_lns / cfg.var_lnk) * z_s * 0.5)

        # Sy: logistic squash of the correlated field onto physical bounds.
        lo, hi = cfg.sy_bounds
        centre = np.clip(cfg.sy_mean[k], lo + 1e-3, hi - 1e-3)
        # logit of the mean, then add the (scaled) correlated perturbation.
        p0 = (centre - lo) / (hi - lo)
        logit0 = np.log(p0 / (1.0 - p0))
        p = 1.0 / (1.0 + np.exp(-(logit0 + 0.9 * z_y)))
        sy[k] = lo + (hi - lo) * p

    # Guard rails: MODFLOW is unforgiving of pathological property values.
    kh = np.clip(kh, 1.0e-6, 5.0e3)
    ss = np.clip(ss, 1.0e-7, 1.0e-2)
    sy = np.clip(sy, cfg.sy_bounds[0], cfg.sy_bounds[1])
    k33 = kh / cfg.kz_ratio

    for k in range(nlay):
        act = grid.idomain[k] > 0
        if act.any():
            print(
                f"[props] L{k + 1}: Kh geo-mean={np.exp(np.log(kh[k][act]).mean()):9.4g} "
                f"min={kh[k][act].min():9.4g} max={kh[k][act].max():9.4g} "
                f"| Ss mean={ss[k][act].mean():.3g} | Sy mean={sy[k][act].mean():.3f}"
            )

    return PropertyFields(kh=kh, k33=k33, ss=ss, sy=sy)


# =============================================================================
# SECTION 4 - FAULTS: CELL-FACE BARRIERS FOR THE HFB PACKAGE
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
        # Compare the "cost" of the next horizontal vs. vertical step.
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
            Fault(
                name=name,
                hydchr=hydchr,
                endpoints=(r0, c0, r1, c1),
                chain=chain,
                pairs=pairs,
            )
        )
        print(f"[hfb] {name}: {len(chain)} cells -> {len(pairs)} blocked faces/layer")
    return faults


def hfb_stress_period_data(
    cfg: Config, grid: Grid, faults: Sequence[Fault]
) -> List[list]:
    """Expand the fault face-pairs over **all five layers**.

    Only pairs where *both* cells are active in that layer are emitted - Layer 1
    exists only in the valley, and both faults lie outside it, so in practice the
    barriers populate Layers 2-5 while the code stays general.
    """
    data: List[list] = []
    per_layer = {k: 0 for k in range(cfg.nlay)}
    for flt in faults:
        for (r1, c1), (r2, c2) in flt.pairs:
            for k in range(cfg.nlay):
                if grid.idomain[k, r1, c1] > 0 and grid.idomain[k, r2, c2] > 0:
                    data.append([(k, r1, c1), (k, r2, c2), flt.hydchr])
                    per_layer[k] += 1
    print(
        "[hfb] barrier faces per layer: "
        + ", ".join(f"L{k + 1}={n}" for k, n in per_layer.items())
    )
    return data


def fault_face_geometries(grid: Grid, flt: Fault) -> List[LineString]:
    """Real-world geometry of every blocked cell face (for the shapefile)."""
    segs: List[LineString] = []
    for (r1, c1), (r2, c2) in flt.pairs:
        if r1 == r2:  # horizontal neighbours -> shared face is a vertical line
            cmax = max(c1, c2)
            x = grid.x_edge(cmax)
            segs.append(
                LineString([(x, grid.y_edge(r1 + 1)), (x, grid.y_edge(r1))])
            )
        else:  # vertical neighbours -> shared face is a horizontal line
            rmax = max(r1, r2)
            y = grid.y_edge(rmax)
            segs.append(
                LineString([(grid.x_edge(c1), y), (grid.x_edge(c1 + 1), y)])
            )
    return segs


def fault_observation_cells(
    grid: Grid, faults: Sequence[Fault], obs_layer: int = 1
) -> List[Tuple[str, Tuple[int, int, int]]]:
    """Pick the 4 head-observation cells straddling the two faults.

    For each fault we take the *middle* blocked face and observe the cell on
    either side of it.  These two cells are 100 m apart yet separated by the
    barrier, so their head difference is a direct, high-signal measurement of the
    discontinuity the surrogate has to reproduce.
    """
    obs: List[Tuple[str, Tuple[int, int, int]]] = []
    for i, flt in enumerate(faults, start=1):
        # Choose a mid-fault pair that is active in the observation layer.
        candidates = [
            p
            for p in flt.pairs
            if grid.idomain[obs_layer, p[0][0], p[0][1]] > 0
            and grid.idomain[obs_layer, p[1][0], p[1][1]] > 0
        ]
        if not candidates:
            raise RuntimeError(f"No active HFB pair for {flt.name} in layer {obs_layer}")
        (ra, ca), (rb, cb) = candidates[len(candidates) // 2]
        obs.append((f"F{i}_SIDE_A", (obs_layer, ra, ca)))
        obs.append((f"F{i}_SIDE_B", (obs_layer, rb, cb)))
        print(
            f"[obs] {flt.name}: F{i}_SIDE_A=(L{obs_layer + 1},r{ra},c{ca})  "
            f"F{i}_SIDE_B=(L{obs_layer + 1},r{rb},c{cb})"
        )
    return obs


# =============================================================================
# SECTION 5 - TRANSIENT FORCING TIME SERIES
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
    classic asymmetric flood wave.  The sharp rising limbs are what stress a
    surrogate's temporal responsiveness for river-aquifer exchange: the induced
    bank storage reverses the flux direction within a couple of stress periods.
    """
    q = np.full_like(t_days, cfg.sfr_base_inflow)
    # A gentle multi-day baseflow drift so the "background" is not constant.
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
        inside = (t_days >= t_start) & (t_days < t_end)
        rch[inside] = cfg.storm_rate
    return rch


def well_rates(cfg: Config, t_days: np.ndarray) -> np.ndarray:
    """(nwell, nper) abstraction rates, negative = extraction.

    Each well follows a diurnal irrigation/supply schedule: a low night-time
    baseline, a smooth daytime peak between 06:00 and 18:00, a well-specific
    phase offset, a weekend reduction, and hard on/off step changes for two of
    the wells.  The composite pumping signal therefore contains a diurnal
    harmonic, a weekly harmonic and two discontinuities.
    """
    nwell = len(WELLS)
    rates = np.zeros((nwell, len(t_days)))
    hour = (t_days * 24.0) % 24.0
    day_index = np.floor(t_days).astype(int)
    weekend = (day_index % 7) >= 5  # 2 low-demand days out of every 7

    for i, w in enumerate(WELLS):
        h = (hour - w["phase_h"]) % 24.0
        # Smooth daytime bell between 06:00 and 18:00, 0.25 baseline at night.
        day_shape = np.where(
            (h >= 6.0) & (h <= 18.0),
            np.sin(np.pi * (h - 6.0) / 12.0) ** 0.7,
            0.0,
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
        t_start=t0,
        t_mid=tm,
        t_end=t1,
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
# SECTION 6 - BOUNDARY-CONDITION PACKAGE DATA
# =============================================================================


def riv_cells(cfg: Config, grid: Grid, fields: PropertyFields) -> List[dict]:
    """One RIV cell per column along the southern boundary row.

    The river is placed in the *uppermost active* of Layers 1-2: Layer 1 inside
    the valley (where the alluvium exists) and Layer 2 on the plateau flanks.
    Putting it in both layers would double-count the conductance.
    """
    upper = grid.uppermost_active()
    cells = []
    for c in range(cfg.ncol):
        k = int(upper[cfg.riv_row, c])
        if k < 0 or k > 1:
            continue
        cond = cfg.riv_bed_k * cfg.riv_width * cfg.delr / cfg.riv_bed_thick
        cells.append(dict(layer=k, row=cfg.riv_row, col=c, cond=cond))
    print(f"[riv] {len(cells)} tidal-river cells on row {cfg.riv_row}")
    return cells


def riv_period_data(
    cells: Sequence[dict], forcing: Forcing
) -> Dict[int, List[tuple]]:
    """Full RIV list re-specified every stress period so the stage follows the
    tide at 2-hour resolution.  ``aux`` carries the stage as a convenience field
    for post-processing."""
    spd: Dict[int, List[tuple]] = {}
    for kper, stage in enumerate(forcing.stage):
        spd[kper] = [
            ((c["layer"], c["row"], c["col"]), float(stage), c["cond"], CFG.riv_bottom)
            for c in cells
        ]
    return spd


def ghb_cells(cfg: Config, grid: Grid, fields: PropertyFields) -> List[tuple]:
    """Regional inflow along the eastern edge, Layers 2-4.

    Conductance is derived cell-by-cell from the local heterogeneous Kh and the
    layer thickness, so the *boundary itself* inherits the geostatistical
    roughness - the surrogate cannot assume a smooth edge flux.  The one cell
    shared with the tidal river (row 99, Layer 2) is skipped so the two
    head-dependent boundaries do not fight over the same cell.
    """
    thick = grid.thickness
    rows = []
    for k in cfg.ghb_layers:
        for r in range(cfg.nrow):
            if grid.idomain[k, r, cfg.ghb_col] <= 0:
                continue
            if k == 1 and r == cfg.riv_row:  # avoid overlap with the RIV cell
                continue
            b = max(thick[k, r, cfg.ghb_col], 1.0e-3)
            cond = fields.kh[k, r, cfg.ghb_col] * b * cfg.delc / (0.5 * cfg.delr)
            rows.append(((k, r, cfg.ghb_col), cfg.ghb_head, float(cond)))
    print(f"[ghb] {len(rows)} regional-inflow cells on column {cfg.ghb_col}")
    return rows


def sfr_package_data(
    cfg: Config, grid: Grid
) -> Tuple[List[list], List[list], List[Tuple[int, int, int]]]:
    """Build SFR ``packagedata`` and ``connectiondata`` for the central stream.

    Reaches run down column 50 from row 0 to row 98 in a single, linearly
    connected chain.  The most-downstream reach has no downstream connection, so
    its outflow leaves the model domain - conceptually it discharges into the
    tidal river occupying row 99.
    """
    nreach = cfg.sfr_row1 - cfg.sfr_row0 + 1
    upper = grid.uppermost_active()

    # Linear streambed profile from the headwater to the tidal confluence.
    rtp = np.linspace(cfg.sfr_top_up, cfg.sfr_top_dn, nreach)
    rgrd = (cfg.sfr_top_up - cfg.sfr_top_dn) / ((nreach - 1) * cfg.delc)

    pkgdata: List[list] = []
    conndata: List[list] = []
    cellids: List[Tuple[int, int, int]] = []
    for i in range(nreach):
        r = cfg.sfr_row0 + i
        k = int(upper[r, cfg.sfr_col])
        cellid = (k, r, cfg.sfr_col)
        cellids.append(cellid)
        pkgdata.append(
            [
                i,               # ifno (0-based reach number)
                cellid,          # connected GWF cell
                cfg.delc,        # rlen
                cfg.sfr_width,   # rwid
                rgrd,            # rgrd
                float(rtp[i]),   # rtp  (streambed top)
                cfg.sfr_bed_thick,
                cfg.sfr_bed_k,
                cfg.sfr_manning,
                0 if i == 0 else 1,  # ncon set below (placeholder, fixed next)
                1.0,             # ustrf - all flow to the single downstream reach
                0,               # ndv
            ]
        )
        # Connections: negative ids are downstream, positive are upstream.
        conn = [i]
        if i > 0:
            conn.append(i - 1)
        if i < nreach - 1:
            conn.append(-(i + 1))
        conndata.append(conn)
        pkgdata[-1][9] = len(conn) - 1  # ncon = number of connected reaches

    print(f"[sfr] {nreach} reaches down column {cfg.sfr_col} (rows "
          f"{cfg.sfr_row0}-{cfg.sfr_row1}), slope={rgrd:.2e}")
    return pkgdata, conndata, cellids


def wel_period_data(forcing: Forcing) -> Dict[int, List[tuple]]:
    """Per-stress-period WEL list for the 8 valley production wells."""
    spd: Dict[int, List[tuple]] = {}
    for kper in range(forcing.wel.shape[1]):
        spd[kper] = [
            ((1, w["row"], w["col"]), float(forcing.wel[i, kper]), w["name"])
            for i, w in enumerate(WELLS)
        ]
    return spd


def recharge_layer_map(grid: Grid) -> np.ndarray:
    """0-based layer index that receives areal recharge in each column.

    MODFLOW 6's array-based RCH applies recharge to the *specified* layer, and
    when IRCH is omitted it uses the top grid layer - which is inactive outside
    the valley.  IRCH is therefore set explicitly to the uppermost active layer
    (Layer 1 in the valley, Layer 2 on the plateau).  FloPy expects a 0-based
    array here and writes the 1-based values MODFLOW requires.
    """
    upper = grid.uppermost_active()
    if (upper < 0).any():
        raise RuntimeError("Some columns have no active cell at all.")
    return upper


# =============================================================================
# SECTION 7 - MODEL CONSTRUCTION
# =============================================================================


def build_simulation(
    cfg: Config,
    grid: Grid,
    fields: PropertyFields,
    faults: Sequence[Fault],
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
        sim_name=cfg.sim_name,
        version="mf6",
        exe_name=exe,
        sim_ws=str(ws),
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
        sim,
        time_units="days",
        nper=len(perioddata),
        perioddata=perioddata,
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
        gwf,
        length_units="meters",
        nlay=cfg.nlay,
        nrow=cfg.nrow,
        ncol=cfg.ncol,
        delr=cfg.delr,
        delc=cfg.delc,
        top=grid.top,
        botm=grid.botm,
        idomain=grid.idomain,
        xorigin=cfg.x_origin,
        yorigin=cfg.y_origin,
        angrot=0.0,
    )
    gwf.modelgrid.set_coord_info(
        xoff=cfg.x_origin, yoff=cfg.y_origin, angrot=0.0, crs=f"EPSG:{cfg.crs_epsg}"
    )

    # ---------------- IC ----------------
    flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # ---------------- NPF ----------------
    # Layers 1-2 are convertible (water-table); the deeper carbonates and
    # granite stay confined, which removes a large source of solver pain without
    # sacrificing any of the intended non-linearity (the water table lives in
    # Layers 1-2).
    icelltype = np.zeros((cfg.nlay, cfg.nrow, cfg.ncol), dtype=int)
    icelltype[0] = 1
    icelltype[1] = 1
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_specific_discharge=True,
        icelltype=icelltype,
        k=fields.kh,
        k33=fields.k33,
        k33overk=False,
    )

    # ---------------- STO ----------------
    if transient:
        flopy.mf6.ModflowGwfsto(
            gwf,
            save_flows=True,
            iconvert=icelltype,
            ss=fields.ss,
            sy=fields.sy,
            steady_state={0: False},
            transient={0: True},
        )

    # ---------------- HFB (faults) ----------------
    hfb_data = hfb_stress_period_data(cfg, grid, faults)
    flopy.mf6.ModflowGwfhfb(
        gwf,
        print_input=True,
        maxhfb=len(hfb_data),
        stress_period_data={0: hfb_data},
    )

    # ---------------- RIV (tidal southern boundary) ----------------
    rcells = riv_cells(cfg, grid, fields)
    if transient:
        riv_spd = riv_period_data(rcells, forcing)
    else:
        riv_spd = {
            0: [
                (
                    (c["layer"], c["row"], c["col"]),
                    cfg.riv_mean_stage,
                    c["cond"],
                    cfg.riv_bottom,
                )
                for c in rcells
            ]
        }
    flopy.mf6.ModflowGwfriv(
        gwf,
        save_flows=True,
        maxbound=len(rcells),
        stress_period_data=riv_spd,
        pname="riv_tidal",
    )

    # ---------------- GHB (eastern regional inflow) ----------------
    gdata = ghb_cells(cfg, grid, fields)
    flopy.mf6.ModflowGwfghb(
        gwf,
        save_flows=True,
        maxbound=len(gdata),
        stress_period_data={0: gdata},
        pname="ghb_east",
    )

    # ---------------- WEL (8 valley production wells) ----------------
    if transient:
        wel_spd = wel_period_data(forcing)
    else:
        # Steady state uses the time-averaged abstraction of each well.
        wel_spd = {
            0: [
                ((1, w["row"], w["col"]), float(forcing.wel[i].mean()), w["name"])
                for i, w in enumerate(WELLS)
            ]
        }
    flopy.mf6.ModflowGwfwel(
        gwf,
        save_flows=True,
        # Boundnames (not aux) carry the well IDs through to the budget file so
        # each well's abstraction can be recovered by name during training-data
        # assembly.
        boundnames=True,
        maxbound=len(WELLS),
        stress_period_data=wel_spd,
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
        gwf,
        save_flows=True,
        readasarrays=True,
        irch={0: irch},  # specified once; MODFLOW 6 carries it forward
        recharge=rch_spd,
        pname="rcha",
    )

    # ---------------- SFR (central stream) ----------------
    pkgdata, conndata, _ = sfr_package_data(cfg, grid)
    if transient:
        sfr_spd = {
            kper: [(0, "inflow", float(q))] for kper, q in enumerate(forcing.inflow)
        }
    else:
        # Median, not mean: the flash-flood peaks are two orders of magnitude
        # above baseflow and would otherwise dominate the steady-state inflow.
        sfr_spd = {0: [(0, "inflow", float(np.median(forcing.inflow)))]}
    flopy.mf6.ModflowGwfsfr(
        gwf,
        save_flows=True,
        print_stage=False,
        print_flows=False,
        budget_filerecord=f"{cfg.model_name}.sfr.cbc",
        stage_filerecord=f"{cfg.model_name}.sfr.stage",
        length_conversion=1.0,  # metres
        time_conversion=86400.0,  # Manning's equation works in seconds
        nreaches=len(pkgdata),
        packagedata=pkgdata,
        connectiondata=conndata,
        perioddata=sfr_spd,
        pname="sfr_central",
    )

    # ---------------- OBS (fault-straddling head observations) ----------------
    obs_recs = [(name, "HEAD", cellid) for name, cellid in obs_cells]
    flopy.mf6.ModflowUtlobs(
        gwf,
        digits=10,
        print_input=False,
        continuous={f"{cfg.model_name}.head.obs.csv": obs_recs},
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
        printrecord = {0: [("BUDGET", "LAST")]}
    else:
        saverecord = {0: [("HEAD", "ALL"), ("BUDGET", "ALL")]}
        printrecord = {0: [("BUDGET", "LAST")]}

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{cfg.model_name}.hds",
        budget_filerecord=f"{cfg.model_name}.cbc",
        saverecord=saverecord,
        printrecord=printrecord,
    )

    return sim


def steady_state_initial_heads(
    cfg: Config,
    grid: Grid,
    fields: PropertyFields,
    faults: Sequence[Fault],
    forcing: Forcing,
    obs_cells: Sequence[Tuple[str, Tuple[int, int, int]]],
    ws: pathlib.Path,
    exe: str,
) -> np.ndarray:
    """Run a steady-state spin-up and return its head field.

    Starting the transient benchmark from an arbitrary flat head would inject a
    huge, purely numerical relaxation transient that swamps the physical signals
    we actually want the surrogate to learn.  Spinning up under the *mean* of
    every forcing series gives a head field already in equilibrium with the
    heterogeneity and the faults, so day 1 of the benchmark is dominated by the
    tide, the storms and the pumping - not by initialisation shock.
    """
    print("\n=== STEADY-STATE SPIN-UP (initial condition) ===")
    strt0 = np.full((cfg.nlay, cfg.nrow, cfg.ncol), cfg.strt_fallback)
    sim = build_simulation(
        cfg, grid, fields, faults, forcing, obs_cells, ws, exe, strt0, transient=False
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    if not success:
        print("[warn] steady-state spin-up did not converge; falling back to a")
        print("       linear head ramp between the river stage and the GHB head.")
        for line in buff[-25:]:
            print("       " + str(line).rstrip())
        ramp = np.linspace(cfg.riv_mean_stage, cfg.ghb_head, cfg.ncol)
        return np.broadcast_to(
            ramp[None, None, :], (cfg.nlay, cfg.nrow, cfg.ncol)
        ).copy()

    hds = flopy.utils.HeadFile(str(ws / f"{cfg.model_name}.hds"))
    head = hds.get_data(kstpkper=(0, 0)).astype(float)
    hds.close()

    # Sanitise: inactive / dry cells carry sentinel values that must never be
    # fed back in as an initial condition.
    bad = ~np.isfinite(head) | (head < cfg.botm_l5 - 1.0) | (head > 1.0e6)
    bad |= grid.idomain <= 0
    if bad.any():
        good_mean = float(np.mean(head[~bad])) if (~bad).any() else cfg.strt_fallback
        head[bad] = good_mean
    print(
        f"[spinup] converged. head range "
        f"{head[grid.idomain > 0].min():.2f} - {head[grid.idomain > 0].max():.2f} m"
    )
    return head


# =============================================================================
# SECTION 8 - GIS EXPORT (GeoTIFF + ESRI Shapefile)
# =============================================================================


def write_geotiff(
    path: pathlib.Path,
    array: np.ndarray,
    grid: Grid,
    nodata: float = -9999.0,
    dtype: str = "float32",
) -> None:
    """Write a single-band, north-up, compressed GeoTIFF."""
    arr = np.asarray(array, dtype=dtype).copy()
    arr[~np.isfinite(arr)] = nodata
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=f"EPSG:{grid.cfg.crs_epsg}",
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
        predictor=2,
        tiled=True,
    ) as dst:
        dst.write(arr, 1)


def export_rasters(
    cfg: Config, grid: Grid, fields: PropertyFields, gis_dir: pathlib.Path
) -> None:
    """Export every static input field as a GeoTIFF feature layer.

    These are exactly the channels a convolutional surrogate would stack into
    its input tensor: per-layer Kh / Kz / Ss / Sy, the stratigraphic surfaces,
    the active-cell masks and the recharge-target-layer map.  Inactive cells are
    written as NoData so masking is unambiguous downstream.
    """
    n = 0
    for k in range(cfg.nlay):
        mask = grid.idomain[k] > 0
        lay = k + 1

        def masked(a: np.ndarray) -> np.ndarray:
            out = np.where(mask, a, np.nan)
            return out

        write_geotiff(gis_dir / f"kh_L{lay}.tif", masked(fields.kh[k]), grid)
        write_geotiff(
            gis_dir / f"log10_kh_L{lay}.tif", masked(np.log10(fields.kh[k])), grid
        )
        write_geotiff(gis_dir / f"kz_L{lay}.tif", masked(fields.k33[k]), grid)
        write_geotiff(gis_dir / f"ss_L{lay}.tif", masked(fields.ss[k]), grid)
        write_geotiff(gis_dir / f"sy_L{lay}.tif", masked(fields.sy[k]), grid)
        write_geotiff(gis_dir / f"top_L{lay}.tif", grid.top_of_layer(k), grid)
        write_geotiff(gis_dir / f"botm_L{lay}.tif", grid.botm[k], grid)
        write_geotiff(
            gis_dir / f"thickness_L{lay}.tif", masked(grid.thickness[k]), grid
        )
        write_geotiff(
            gis_dir / f"idomain_L{lay}.tif",
            grid.idomain[k].astype(float),
            grid,
            nodata=-9999.0,
        )
        n += 9

    write_geotiff(gis_dir / "model_top.tif", grid.top, grid)
    write_geotiff(gis_dir / "valley_mask.tif", grid.valley_mask.astype(float), grid)
    write_geotiff(
        gis_dir / "recharge_layer_irch.tif",
        recharge_layer_map(grid).astype(float) + 1.0,  # 1-based for GIS readability
        grid,
    )
    n += 3
    print(f"[gis] wrote {n} GeoTIFF feature rasters -> {gis_dir}")


def _cell_polygon(grid: Grid, row: int, col: int) -> Polygon:
    return box(
        grid.x_edge(col),
        grid.y_edge(row + 1),
        grid.x_edge(col + 1),
        grid.y_edge(row),
    )


def export_shapefiles(
    cfg: Config,
    grid: Grid,
    fields: PropertyFields,
    faults: Sequence[Fault],
    forcing: Forcing,
    obs_cells: Sequence[Tuple[str, Tuple[int, int, int]]],
    gis_dir: pathlib.Path,
) -> None:
    """Export every boundary condition and structural feature as a Shapefile.

    Vector exports serve a different purpose to the rasters: they are what you
    compute *distance-to-feature* covariates from (distance to the fault, to the
    stream, to the nearest well), which are usually the single most informative
    engineered features for a groundwater surrogate.  DBF field names are kept
    to <= 10 characters so no silent truncation occurs.
    """
    crs = f"EPSG:{cfg.crs_epsg}"

    # --- 1. domain outline and valley outline --------------------------------
    domain = box(
        cfg.x_origin,
        cfg.y_origin,
        cfg.x_origin + cfg.ncol * cfg.delr,
        cfg.y_origin + cfg.nrow * cfg.delc,
    )
    valley = box(
        grid.x_edge(cfg.valley_c0),
        cfg.y_origin,
        grid.x_edge(cfg.valley_c1 + 1),
        cfg.y_origin + cfg.nrow * cfg.delc,
    )
    gpd.GeoDataFrame(
        {"name": ["model_domain", "incised_valley"], "kind": ["domain", "valley"]},
        geometry=[domain, valley],
        crs=crs,
    ).to_file(gis_dir / "domain_outlines.shp", driver="ESRI Shapefile")

    # --- 2. faults: idealised trace + every blocked cell face ----------------
    trace_geoms, trace_rows = [], []
    face_geoms, face_rows = [], []
    for flt in faults:
        r0, c0, r1, c1 = flt.endpoints
        trace_geoms.append(
            LineString([grid.cell_center(r0, c0), grid.cell_center(r1, c1)])
        )
        trace_rows.append(
            dict(
                name=flt.name[:10],
                hydchr=flt.hydchr,
                r0=r0,
                c0=c0,
                r1=r1,
                c1=c1,
                nfaces=len(flt.pairs),
            )
        )
        for seg, ((ra, ca), (rb, cb)) in zip(
            fault_face_geometries(grid, flt), flt.pairs
        ):
            face_geoms.append(seg)
            face_rows.append(
                dict(name=flt.name[:10], hydchr=flt.hydchr, r1=ra, c1=ca, r2=rb, c2=cb)
            )
    gpd.GeoDataFrame(trace_rows, geometry=trace_geoms, crs=crs).to_file(
        gis_dir / "faults_trace.shp", driver="ESRI Shapefile"
    )
    gpd.GeoDataFrame(face_rows, geometry=face_geoms, crs=crs).to_file(
        gis_dir / "faults_hfb_faces.shp", driver="ESRI Shapefile"
    )

    # --- 3. tidal river cells ------------------------------------------------
    rcells = riv_cells(cfg, grid, fields)
    gpd.GeoDataFrame(
        [
            dict(
                layer=c["layer"] + 1,
                row=c["row"],
                col=c["col"],
                cond=c["cond"],
                rbot=cfg.riv_bottom,
                stage_mean=cfg.riv_mean_stage,
                stage_amp=cfg.riv_amplitude,
            )
            for c in rcells
        ],
        geometry=[_cell_polygon(grid, c["row"], c["col"]) for c in rcells],
        crs=crs,
    ).to_file(gis_dir / "bc_riv_cells.shp", driver="ESRI Shapefile")

    # --- 4. SFR reaches (centreline + per-reach points) ----------------------
    pkgdata, _, cellids = sfr_package_data(cfg, grid)
    reach_pts = [grid.cell_center(cid[1], cid[2]) for cid in cellids]
    gpd.GeoDataFrame(
        [
            dict(
                ifno=int(rec[0]),
                layer=cellids[i][0] + 1,
                row=cellids[i][1],
                col=cellids[i][2],
                rlen=rec[2],
                rwid=rec[3],
                rgrd=rec[4],
                rtp=rec[5],
                rhk=rec[7],
                man=rec[8],
            )
            for i, rec in enumerate(pkgdata)
        ],
        geometry=[Point(p) for p in reach_pts],
        crs=crs,
    ).to_file(gis_dir / "bc_sfr_reaches.shp", driver="ESRI Shapefile")
    gpd.GeoDataFrame(
        [dict(name="central_stream", nreach=len(reach_pts))],
        geometry=[LineString(reach_pts)],
        crs=crs,
    ).to_file(gis_dir / "bc_sfr_centerline.shp", driver="ESRI Shapefile")

    # --- 5. GHB cells --------------------------------------------------------
    gdata = ghb_cells(cfg, grid, fields)
    gpd.GeoDataFrame(
        [
            dict(layer=cid[0] + 1, row=cid[1], col=cid[2], bhead=bh, cond=cd)
            for cid, bh, cd in gdata
        ],
        geometry=[_cell_polygon(grid, cid[1], cid[2]) for cid, _, _ in gdata],
        crs=crs,
    ).to_file(gis_dir / "bc_ghb_cells.shp", driver="ESRI Shapefile")

    # --- 6. wells ------------------------------------------------------------
    gpd.GeoDataFrame(
        [
            dict(
                name=w["name"],
                layer=2,
                row=w["row"],
                col=w["col"],
                q_base=w["base"],
                q_mean=float(forcing.wel[i].mean()),
                q_min=float(forcing.wel[i].min()),
                phase_h=w["phase_h"],
                start_d=w["start_d"],
                stop_d=w["stop_d"],
            )
            for i, w in enumerate(WELLS)
        ],
        geometry=[Point(grid.cell_center(w["row"], w["col"])) for w in WELLS],
        crs=crs,
    ).to_file(gis_dir / "bc_wells.shp", driver="ESRI Shapefile")

    # --- 7. observation points ----------------------------------------------
    gpd.GeoDataFrame(
        [
            dict(name=name, layer=cid[0] + 1, row=cid[1], col=cid[2])
            for name, cid in obs_cells
        ],
        geometry=[Point(grid.cell_center(cid[1], cid[2])) for _, cid in obs_cells],
        crs=crs,
    ).to_file(gis_dir / "obs_points.shp", driver="ESRI Shapefile")

    print(f"[gis] wrote 8 ESRI Shapefiles -> {gis_dir}")


# =============================================================================
# SECTION 9 - POST-PROCESSING AND PLOTS
# =============================================================================


def report_mass_balance(cfg: Config, ws: pathlib.Path) -> None:
    """Print the volumetric budget discrepancy - the first sanity check."""
    lst_path = ws / f"{cfg.model_name}.lst"
    try:
        mf_list = flopy.utils.Mf6ListBudget(str(lst_path))
        inc, cum = mf_list.get_budget()
        pct_inc = float(np.max(np.abs(inc["PERCENT_DISCREPANCY"])))
        pct_cum = float(np.abs(cum["PERCENT_DISCREPANCY"][-1]))
        print(
            f"[budget] max incremental discrepancy = {pct_inc:.4f} % | "
            f"final cumulative discrepancy = {pct_cum:.4f} %"
        )
        if pct_inc > 1.0:
            print("[budget] WARNING: discrepancy > 1 %, tighten the IMS settings.")
    except Exception as exc:  # noqa: BLE001
        print(f"[budget] could not parse listing file ({exc})")


def report_benchmark_diagnostics(
    cfg: Config,
    grid: Grid,
    faults: Sequence[Fault],
    head: np.ndarray,
    time_d: float,
) -> pd.DataFrame:
    """Quantify *how hard* the generated benchmark actually is.

    Two properties decide whether this dataset is worth training on:

    * **Fault contrast** - the mean head jump across the blocked faces divided
      by the mean head difference between ordinary neighbouring cells.  A ratio
      near 1 would mean the HFB barriers are invisible and the benchmark is
      pointless; values of several times the background confirm a genuine
      internal discontinuity that a smooth surrogate cannot represent.
    * **Water table above land surface** - reported honestly rather than hidden.
      The prescribed base recharge (5e-4 m/d over 100 km^2) exceeds what the
      specified transmissivity and the river/stream outlets can drain from the
      most remote, fault-compartmentalised parts of the plateau, so the water
      table there sits slightly above the 50 m ground surface.  This is a
      property of the prescribed conceptual model, not a numerical failure
      (MODFLOW's Newton formulation handles it exactly, with a 0 % budget
      discrepancy); lower ``rch_base`` if a strictly sub-surface water table is
      required.
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
            jumps = [
                abs(hk[a] - hk[b])
                for a, b in flt.pairs
                if np.isfinite(hk[a]) and np.isfinite(hk[b])
            ]
            if not jumps:
                continue
            rows.append(
                dict(
                    layer=k + 1,
                    fault=flt.name,
                    hydchr=flt.hydchr,
                    n_faces=len(jumps),
                    dh_mean_m=float(np.mean(jumps)),
                    dh_max_m=float(np.max(jumps)),
                    background_dh_m=bg_mean,
                    contrast_ratio=float(np.mean(jumps) / bg_mean) if bg_mean else np.nan,
                )
            )
            print(
                f"  L{k + 1} {flt.name:22s} mean|dh|={np.mean(jumps):6.3f} m  "
                f"max={np.max(jumps):6.3f} m  background={bg_mean:6.3f} m  "
                f"contrast={np.mean(jumps) / bg_mean if bg_mean else float('nan'):5.1f}x"
            )

    land_surface = grid.top
    upper = grid.uppermost_active()
    wt = np.take_along_axis(head, upper[None, :, :], axis=0)[0]
    above = np.isfinite(wt) & (wt > land_surface)
    print(
        f"  water table above land surface in {above.sum():,} of "
        f"{above.size:,} columns ({100.0 * above.mean():.1f} %), "
        f"max exceedance {np.nanmax(np.where(above, wt - land_surface, np.nan)) if above.any() else 0.0:.2f} m"
    )
    return pd.DataFrame(rows)


def load_observations(cfg: Config, ws: pathlib.Path) -> pd.DataFrame:
    """Read the MODFLOW 6 continuous-observation CSV into a DataFrame."""
    obs_path = ws / f"{cfg.model_name}.head.obs.csv"
    df = pd.read_csv(obs_path)
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={"TIME": "time_d"})
    return df


def plot_obs_hydrographs(
    cfg: Config, obs: pd.DataFrame, forcing: Forcing, out_png: pathlib.Path
) -> None:
    """Hydrographs of the four fault-straddling observation points.

    Each panel shows the pair on either side of one fault plus, on a secondary
    axis, the head *difference* across the barrier.  For the impermeable fault
    that difference is a step of several metres that persists through the whole
    simulation; for the semi-permeable fault it is small and strongly modulated
    by the transient stresses.  Reproducing both behaviours is the acid test for
    a surrogate.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    pairs = [
        ("F1_SIDE_A", "F1_SIDE_B", "Fault 1 - impermeable (hydchr = 1e-8 d$^{-1}$)"),
        ("F2_SIDE_A", "F2_SIDE_B", "Fault 2 - semi-permeable (hydchr = 1e-3 d$^{-1}$)"),
    ]
    colors = [("#1f77b4", "#d62728"), ("#2ca02c", "#ff7f0e")]

    for ax, (a, b, title), (ca, cb) in zip(axes[:2], pairs, colors):
        ax.plot(obs["time_d"], obs[a], color=ca, lw=1.1, label=f"{a} (footwall side)")
        ax.plot(obs["time_d"], obs[b], color=cb, lw=1.1, label=f"{b} (hanging-wall side)")
        ax.set_ylabel("Head (m a.s.l.)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=2)

        ax2 = ax.twinx()
        dh = obs[a] - obs[b]
        ax2.plot(obs["time_d"], dh, color="0.35", lw=0.8, ls="--")
        ax2.set_ylabel("$\\Delta h$ across fault (m)", color="0.35", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="0.35", labelsize=8)

    # Context panel: what the aquifer was being asked to do.
    ax = axes[2]
    ax.plot(forcing.t_mid, forcing.stage, color="#17becf", lw=0.9, label="River stage (m)")
    ax.set_ylabel("Tidal stage (m)", color="#17becf")
    ax.tick_params(axis="y", labelcolor="#17becf")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Time (days)")
    ax3 = ax.twinx()
    ax3.plot(
        forcing.t_mid,
        -forcing.wel.sum(axis=0),
        color="#8c564b",
        lw=0.8,
        label="Total abstraction (m$^3$/d)",
    )
    ax3.set_ylabel("Total abstraction (m$^3$/d)", color="#8c564b")
    ax3.tick_params(axis="y", labelcolor="#8c564b")
    ax.set_title("Driving stresses (context)", fontsize=11, loc="left")

    for a in axes:
        a.set_xlim(0, cfg.nper * cfg.dt_days)

    fig.suptitle(
        "Fault-straddling head hydrographs - 2-hourly resolution over 30 days",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[plot] {out_png}")


def plot_head_map(
    cfg: Config,
    grid: Grid,
    gwf: flopy.mf6.ModflowGwf,
    head: np.ndarray,
    faults: Sequence[Fault],
    obs_cells: Sequence[Tuple[str, Tuple[int, int, int]]],
    kper: int,
    time_d: float,
    out_png: pathlib.Path,
) -> None:
    """Filled + line contour map of the Layer 2 head field with all features."""
    fig, ax = plt.subplots(figsize=(11, 9.5))
    pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=1)

    h2 = np.where(grid.idomain[1] > 0, head[1], np.nan)
    finite = h2[np.isfinite(h2)]
    levels = np.linspace(np.nanpercentile(finite, 0.5), np.nanpercentile(finite, 99.5), 30)

    cf = pmv.plot_array(h2, cmap="viridis", alpha=0.95)
    cs = pmv.contour_array(
        h2, levels=levels, colors="k", linewidths=0.45, alpha=0.75
    )
    ax.clabel(cs, cs.levels[::4], inline=True, fontsize=6, fmt="%.1f")
    pmv.plot_inactive(color_noflow="0.85")
    pmv.plot_grid(lw=0.05, color="0.85", alpha=0.25)

    # Boundary conditions
    pmv.plot_bc("RIV", color="#1f77b4", alpha=0.6)
    pmv.plot_bc("GHB", color="#9467bd", alpha=0.6)
    pmv.plot_bc("SFR", color="#17becf", alpha=0.8)

    # Faults, drawn as the true blocked faces (not the idealised straight line).
    for flt, col in zip(faults, ("red", "darkorange")):
        segs = fault_face_geometries(grid, flt)
        for seg in segs:
            xs, ys = seg.xy
            ax.plot(xs, ys, color=col, lw=2.2, solid_capstyle="butt")
        ax.plot([], [], color=col, lw=2.2, label=f"{flt.name} (K'={flt.hydchr:g}/d)")

    # Wells and observation points
    wx = [grid.cell_center(w["row"], w["col"])[0] for w in WELLS]
    wy = [grid.cell_center(w["row"], w["col"])[1] for w in WELLS]
    ax.scatter(wx, wy, marker="o", s=48, facecolor="white", edgecolor="k",
               zorder=6, label="Pumping wells (L2)")
    for w, x, y in zip(WELLS, wx, wy):
        ax.annotate(w["name"], (x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=7, zorder=7)

    ox = [grid.cell_center(cid[1], cid[2])[0] for _, cid in obs_cells]
    oy = [grid.cell_center(cid[1], cid[2])[1] for _, cid in obs_cells]
    ax.scatter(ox, oy, marker="^", s=70, facecolor="yellow", edgecolor="k",
               zorder=8, label="Fault observation points")

    cbar = fig.colorbar(cf, ax=ax, shrink=0.82)
    cbar.set_label("Head in Layer 2 - Older Alluvium (m a.s.l.)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(
        f"Layer 2 simulated head - stress period {kper} "
        f"(t = {time_d:.3f} d)\n"
        "Sharp offsets across the fault traces are the HFB barriers",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[plot] {out_png}")


def plot_forcing(cfg: Config, forcing: Forcing, out_png: pathlib.Path) -> None:
    """Four-panel summary of everything that drives the model."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(forcing.t_mid, forcing.stage, lw=0.8, color="#1f77b4")
    axes[0].axhline(cfg.riv_bottom, color="k", ls=":", lw=0.8, label="River bottom")
    axes[0].set_ylabel("Stage (m)")
    axes[0].set_title(
        "Tidal river stage - M2 semi-diurnal (12.4 h) with spring-neap envelope",
        fontsize=10, loc="left")
    axes[0].legend(fontsize=8)

    axes[1].plot(forcing.t_mid, forcing.inflow, lw=0.9, color="#17becf")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Inflow (m$^3$/d)")
    axes[1].set_title(
        "SFR head-reach inflow - baseflow plus three flash-flood hydrographs",
        fontsize=10, loc="left")

    axes[2].plot(forcing.t_mid, forcing.rch, lw=0.9, color="#2ca02c")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Recharge (m/d)")
    axes[2].set_title(
        "Areal recharge - base rate plus two intense 12-hour storms",
        fontsize=10, loc="left")

    for i, w in enumerate(WELLS):
        axes[3].plot(forcing.t_mid, -forcing.wel[i], lw=0.6, label=w["name"])
    axes[3].plot(forcing.t_mid, -forcing.wel.sum(axis=0), lw=1.4, color="k",
                 label="Total")
    axes[3].set_ylabel("Abstraction (m$^3$/d)")
    axes[3].set_xlabel("Time (days)")
    axes[3].set_title(
        "Well abstraction - diurnal + weekly cycles, W4 starts at day 12, "
        "W6 stops at day 20", fontsize=10, loc="left")
    axes[3].legend(fontsize=7, ncol=5, loc="upper left")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlim(0, cfg.nper * cfg.dt_days)

    fig.suptitle("Transient forcing at 2-hour stress-period resolution", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[plot] {out_png}")


def plot_k_field(
    cfg: Config,
    grid: Grid,
    fields: PropertyFields,
    faults: Sequence[Fault],
    out_png: pathlib.Path,
) -> None:
    """Log-K panels for all five layers with the fault traces overlaid."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    extent = (
        cfg.x_origin,
        cfg.x_origin + cfg.ncol * cfg.delr,
        cfg.y_origin,
        cfg.y_origin + cfg.nrow * cfg.delc,
    )
    for k in range(cfg.nlay):
        ax = axes.flat[k]
        arr = np.where(grid.idomain[k] > 0, fields.kh[k], np.nan)
        im = ax.imshow(arr, extent=extent, origin="upper", cmap="turbo",
                       norm=LogNorm(vmin=np.nanpercentile(arr, 1),
                                    vmax=np.nanpercentile(arr, 99)))
        for flt, col in zip(faults, ("k", "w")):
            for seg in fault_face_geometries(grid, flt):
                xs, ys = seg.xy
                ax.plot(xs, ys, color=col, lw=1.6, solid_capstyle="butt")
        ax.set_title(f"Layer {k + 1} - $K_h$ (m/d), geo-mean {cfg.kh_mean[k]:g}",
                     fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes.flat[5]
    arr = np.where(grid.idomain[1] > 0, fields.sy[1], np.nan)
    im = ax.imshow(arr, extent=extent, origin="upper", cmap="magma")
    ax.set_title("Layer 2 - specific yield $S_y$ (-)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"Geostatistical realisation: exponential covariance, "
        f"len_scale = {cfg.len_scale:g} m ({cfg.len_scale / cfg.delr:.0f} cells), "
        f"var(lnK) = {cfg.var_lnk:g}  |  seed = {cfg.seed}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")


def export_head_tensor(
    cfg: Config, grid: Grid, ws: pathlib.Path, arrays_dir: pathlib.Path
) -> np.ndarray:
    """Dump the full (nper, nlay, nrow, ncol) head tensor for surrogate training.

    Stored as float32 in a compressed ``.npz`` together with the time vector and
    the active-cell mask, so a downstream training script needs nothing but
    NumPy to consume the benchmark.
    """
    hds = flopy.utils.HeadFile(str(ws / f"{cfg.model_name}.hds"))
    times = np.array(hds.get_times())
    stack = np.empty((len(times), cfg.nlay, cfg.nrow, cfg.ncol), dtype=np.float32)
    for i, t in enumerate(times):
        stack[i] = hds.get_data(totim=t).astype(np.float32)
    hds.close()

    # Sentinel values (inactive / dry) -> NaN so they cannot pollute training.
    inactive = np.broadcast_to(grid.idomain <= 0, stack.shape)
    stack[inactive] = np.nan
    stack[np.abs(stack) > 1.0e20] = np.nan

    out = arrays_dir / "heads_transient.npz"
    np.savez_compressed(
        out,
        head=stack,
        time_days=times.astype(np.float32),
        idomain=grid.idomain.astype(np.int8),
        top=grid.top.astype(np.float32),
        botm=grid.botm.astype(np.float32),
    )
    print(
        f"[arrays] head tensor {stack.shape} -> {out} "
        f"({out.stat().st_size / 1e6:.1f} MB compressed)"
    )
    return stack


# =============================================================================
# SECTION 10 - MAIN
# =============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build, run and export a heterogeneous transient MODFLOW 6 "
        "benchmark for ML surrogate models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-w",
        "--workspace",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "runs" / "mf6_hetero_benchmark"),
        help="Output root directory.",
    )
    p.add_argument("--seed", type=int, default=CFG.seed,
                   help="Master seed for the geostatistical realisation.")
    p.add_argument("--nper", type=int, default=CFG.nper,
                   help="Number of 2-hour stress periods (360 = the full 30-day "
                        "benchmark; lower values are useful for smoke tests).")
    p.add_argument("--no-run", action="store_true",
                   help="Write the model and the GIS exports but do not execute MODFLOW.")
    p.add_argument("--no-spinup", action="store_true",
                   help="Skip the steady-state spin-up and start from a flat head.")
    p.add_argument("--no-npz", action="store_true",
                   help="Skip writing the full head tensor (saves ~100 MB).")
    p.add_argument("--map-kper", type=int, default=180,
                   help="1-based stress period used for the head contour map.")
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
    print(f" grid      : {cfg.nlay} x {cfg.nrow} x {cfg.ncol} @ "
          f"{cfg.delr:g} m ({cfg.ncol * cfg.delr / 1000:g} x "
          f"{cfg.nrow * cfg.delc / 1000:g} km)")
    print(f" time      : {cfg.nper} stress periods x {cfg.dt_days * 24:g} h = "
          f"{cfg.nper * cfg.dt_days:g} d")
    print(f" seed      : {cfg.seed}")
    print("=" * 78)

    exe = resolve_mf6_executable(bindir=root / "bin")

    # ---- 1. static structure -------------------------------------------------
    grid = build_grid(cfg)
    fields = build_property_fields(cfg, grid)
    faults = build_faults(cfg)
    obs_cells = fault_observation_cells(grid, faults, obs_layer=1)
    forcing = build_forcing(cfg)

    # ---- 2. forcing tables (surrogate feature matrix) ------------------------
    forcing_df = forcing.to_dataframe()
    forcing_csv = paths["tables"] / "forcing_by_stress_period.csv"
    forcing_df.to_csv(forcing_csv, index=False)
    print(f"[tables] {forcing_csv}")

    pd.DataFrame(
        [
            dict(
                name=w["name"], layer=2, row=w["row"], col=w["col"],
                x=grid.cell_center(w["row"], w["col"])[0],
                y=grid.cell_center(w["row"], w["col"])[1],
                q_base_m3d=w["base"], phase_h=w["phase_h"],
                start_d=w["start_d"], stop_d=w["stop_d"],
            )
            for w in WELLS
        ]
    ).to_csv(paths["tables"] / "wells.csv", index=False)

    pd.DataFrame(
        [
            dict(name=name, layer=cid[0] + 1, row=cid[1], col=cid[2],
                 x=grid.cell_center(cid[1], cid[2])[0],
                 y=grid.cell_center(cid[1], cid[2])[1])
            for name, cid in obs_cells
        ]
    ).to_csv(paths["tables"] / "observation_points.csv", index=False)

    # ---- 3. GIS exports ------------------------------------------------------
    export_rasters(cfg, grid, fields, paths["gis"])
    export_shapefiles(cfg, grid, fields, faults, forcing, obs_cells, paths["gis"])

    # ---- 4. static diagnostics ----------------------------------------------
    plot_k_field(cfg, grid, fields, faults, paths["figures"] / "fig_k_fields.png")
    plot_forcing(cfg, forcing, paths["figures"] / "fig_forcing_timeseries.png")

    # ---- 5. initial condition ------------------------------------------------
    if args.no_run or args.no_spinup:
        strt = np.full((cfg.nlay, cfg.nrow, cfg.ncol), cfg.strt_fallback)
        print("[spinup] skipped - starting from a flat head field.")
    else:
        strt = steady_state_initial_heads(
            cfg, grid, fields, faults, forcing, obs_cells, paths["warmup"], exe
        )

    # ---- 6. transient benchmark ---------------------------------------------
    print("\n=== TRANSIENT BENCHMARK SIMULATION ===")
    sim = build_simulation(
        cfg, grid, fields, faults, forcing, obs_cells,
        paths["sim"], exe, strt, transient=True,
    )
    sim.write_simulation(silent=False)
    print(f"[write] simulation written to {paths['sim']}")

    if args.no_run:
        print("[run] --no-run requested; stopping before execution.")
        return 0

    success, buff = sim.run_simulation(silent=False, report=True)
    if not success:
        print("\n[run] MODFLOW 6 FAILED. Tail of the simulation output:")
        for line in buff[-40:]:
            print("   " + str(line).rstrip())
        return 1
    print("[run] MODFLOW 6 completed normally.")

    report_mass_balance(cfg, paths["sim"])

    # ---- 7. post-processing --------------------------------------------------
    gwf = sim.get_model(cfg.model_name)

    obs = load_observations(cfg, paths["sim"])
    obs.to_csv(paths["tables"] / "fault_observations.csv", index=False)
    plot_obs_hydrographs(
        cfg, obs, forcing, paths["figures"] / "fig_obs_hydrographs.png"
    )

    hds = flopy.utils.HeadFile(str(paths["sim"] / f"{cfg.model_name}.hds"))
    kstpkper = hds.get_kstpkper()
    kper_1based = int(np.clip(args.map_kper, 1, cfg.nper))
    target = (0, kper_1based - 1)
    if target not in kstpkper:
        target = kstpkper[min(kper_1based - 1, len(kstpkper) - 1)]
    head_map = hds.get_data(kstpkper=target).astype(float)
    head_map[np.abs(head_map) > 1.0e20] = np.nan
    time_d = float(hds.get_times()[kstpkper.index(target)])
    hds.close()

    plot_head_map(
        cfg, grid, gwf, head_map, faults, obs_cells,
        kper=target[1] + 1, time_d=time_d,
        out_png=paths["figures"] / f"fig_head_layer2_sp{target[1] + 1}.png",
    )

    diag = report_benchmark_diagnostics(cfg, grid, faults, head_map, time_d)
    diag.to_csv(paths["tables"] / "fault_contrast_diagnostics.csv", index=False)

    # The head snapshot is also exported as a GeoTIFF so the benchmark's target
    # variable lives in the same georeferenced stack as its input features.
    for k in range(cfg.nlay):
        write_geotiff(
            paths["gis"] / f"head_L{k + 1}_sp{target[1] + 1}.tif",
            np.where(grid.idomain[k] > 0, head_map[k], np.nan),
            grid,
        )

    if not args.no_npz:
        export_head_tensor(cfg, grid, paths["sim"], paths["arrays"])

    print("\n" + "=" * 78)
    print(" BENCHMARK COMPLETE")
    print("=" * 78)
    for key in ("sim", "gis", "tables", "arrays", "figures"):
        print(f"  {key:8s} -> {paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
