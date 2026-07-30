"""Generate a complete synthetic test case with known ground truth.

Builds a two-layer aquifer crossed by a river and two faults, solves it with the
reference finite-difference model, then throws almost all of that away and keeps
only what a real project would have: scattered well readings, a handful of gauge
stages, a few dozen pumping tests, and the elevation rasters. The full truth is
written alongside so the inversion can be scored against it.

The layer-bottom rasters are written *before* the overlap correction, so the
pipeline has to detect and fix the crossing surfaces itself.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

from gwpinn.data.fdsolver import fault_face_multipliers, solve_steady
from gwpinn.data.grf import gaussian_random_field
from gwpinn.io.layers import correct_layer_overlaps
from gwpinn.io.raster import Raster, make_transform, write_raster
from gwpinn.stats.variogram import VariogramModel

CRS = "EPSG:32633"  # arbitrary projected CRS, metres


@dataclass
class SyntheticCase:
    """Everything produced by :func:`make_synthetic_case`."""

    outdir: pathlib.Path
    xs: np.ndarray
    ys: np.ndarray
    cellsize: float
    transform: object
    active: np.ndarray
    top: np.ndarray
    bottoms_raw: List[np.ndarray]
    bottoms: List[np.ndarray]
    K: List[np.ndarray]
    S: List[np.ndarray]
    T: List[np.ndarray]
    head: np.ndarray
    truth_variograms: Dict[str, VariogramModel]
    fault_alpha: List[float]
    meta: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #


def make_synthetic_case(
    outdir: str | pathlib.Path,
    nx: int = 200,
    ny: int = 160,
    cellsize: float = 50.0,
    n_wells: int = 60,
    n_gauges: int = 6,
    n_pumping_tests: int = 28,
    seed: int = 12345,
    verbose: bool = True,
) -> SyntheticCase:
    outdir = pathlib.Path(outdir)
    (outdir / "truth").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    log = print if verbose else (lambda *a, **k: None)

    xmin, ymin = 0.0, 0.0
    xmax, ymax = xmin + nx * cellsize, ymin + ny * cellsize
    xs = xmin + (np.arange(nx) + 0.5) * cellsize
    ys = ymax - (np.arange(ny) + 0.5) * cellsize          # north-up
    xx, yy = np.meshgrid(xs, ys)
    transform = make_transform(xmin, ymax, cellsize)

    # ---- domain outline (deliberately not a rectangle) -------------------
    domain_poly = Polygon(
        [
            (xmin + 300, ymin + 200), (xmax - 200, ymin + 500),
            (xmax - 150, ymax - 700), (xmax - 2200, ymax - 150),
            (xmin + 900, ymax - 300), (xmin + 150, ymax - 3000),
        ]
    )
    active = shapely.contains_xy(domain_poly, xx, yy)
    log(f"  domain: {active.sum()} active cells of {nx * ny}")

    # ---- river geometry --------------------------------------------------
    n_cl = 220
    cy = np.linspace(ymax - 250, ymin + 250, n_cl)         # north -> south
    cx = 5000.0 + 900.0 * np.sin(2 * np.pi * (ymax - cy) / 7000.0) - 0.05 * (ymax - cy)
    centerline = np.column_stack([cx, cy])
    river_line = LineString(centerline)
    river_poly = river_line.buffer(85.0).intersection(domain_poly)

    s_cl = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(centerline, axis=0).T))])
    # The river is incised 1 m into the floor of its own valley, so the aquifer
    # has somewhere to drain to and the water table stays below ground.
    stage_up, stage_dn = 62.5, 47.5
    stage_cl = stage_up - (stage_up - stage_dn) * (s_cl / s_cl[-1]) ** 0.85

    # ---- terrain ---------------------------------------------------------
    dist_river = _distance_to_polyline(centerline, xx, yy)
    relief = gaussian_random_field(
        (ny, nx), cellsize, VariogramModel("gaussian", 0.0, 9.0, 4500.0),
        rng, mean=0.0,
    )
    top = (
        60.0
        + 0.0020 * (yy - ymin)                       # regional slope to the south
        - 12.0 * np.exp(-((dist_river / 700.0) ** 2))  # valley cut by the river
        + relief
    )

    # ---- layer bottoms, with deliberate crossings ------------------------
    und1 = gaussian_random_field(
        (ny, nx), cellsize, VariogramModel("gaussian", 0.0, 9.0, 3000.0), rng
    )
    und2 = gaussian_random_field(
        (ny, nx), cellsize, VariogramModel("gaussian", 0.0, 25.0, 3800.0), rng
    )
    bot0_raw = top - 32.0 + und1
    bot1_raw = bot0_raw - 38.0 + und2
    # A ridge on the deeper surface that pushes it up through the one above,
    # exactly the kind of artefact independent surface interpolations produce.
    ridge = 46.0 * np.exp(-(((xx - 6800) / 1400.0) ** 2 + ((yy - 5200) / 1100.0) ** 2))
    bot1_raw = bot1_raw + ridge

    bottoms_raw = [bot0_raw, bot1_raw]
    _, bottoms, report = correct_layer_overlaps(top, bottoms_raw, min_thickness=2.0)
    n_fixed = sum(r["n_cells_corrected"] for r in report["layers"])
    log(f"  layer bottoms: {n_fixed} cells needed overlap correction")

    # ---- aquifer properties ---------------------------------------------
    vgm = {
        "K0": VariogramModel("exponential", 0.01, 0.18, 1800.0),
        "K1": VariogramModel("exponential", 0.01, 0.13, 2600.0),
        "S0": VariogramModel("exponential", 0.002, 0.045, 2100.0),
        "S1": VariogramModel("exponential", 0.004, 0.090, 2600.0),
    }
    logK0 = gaussian_random_field((ny, nx), cellsize, vgm["K0"], rng, mean=1.00)
    logK1 = gaussian_random_field((ny, nx), cellsize, vgm["K1"], rng, mean=0.40)
    logS0 = gaussian_random_field((ny, nx), cellsize, vgm["S0"], rng, mean=-1.00)
    logS1 = gaussian_random_field((ny, nx), cellsize, vgm["S1"], rng, mean=-3.70)

    K = [10.0**logK0, 10.0**logK1]
    S = [10.0**logS0, 10.0**logS1]

    # ---- faults ----------------------------------------------------------
    fault_a = np.column_stack(
        [np.linspace(1000, 6400, 40),
         5100 + 320 * np.sin(np.linspace(0, 2.2, 40))]
    )
    fault_b = np.column_stack(
        [np.linspace(2600, 8600, 40),
         np.linspace(1200, 6900, 40) + 200 * np.sin(np.linspace(0, 3.0, 40))]
    )
    fault_lines = [fault_a, fault_b]
    fault_perm_flag = [0.0, 1.0]        # 0 = impermeable, 1 = trainable
    fault_alpha_true = [0.002, 0.05]    # what the solver actually uses
    mx, my = fault_face_multipliers(fault_lines, fault_alpha_true, xs, ys)
    log(f"  faults: {int((mx < 1).sum() + (my < 1).sum())} blocked cell faces")

    # ---- river cells and forcing ----------------------------------------
    in_river = shapely.contains_xy(river_poly, xx, yy) & active
    river_cond = np.where(in_river, 5.0e-2, 0.0)
    river_stage = np.zeros((ny, nx))
    if in_river.any():
        s_pt = _project_arclength(centerline, s_cl, xx[in_river], yy[in_river])
        river_stage[in_river] = np.interp(s_pt, s_cl, stage_cl)

    # ~44 mm/yr. High enough to drive a clear gradient towards the river,
    # low enough that the water table stays below ground everywhere.
    recharge = np.full((ny, nx), 1.2e-4)

    # ---- ground-truth solve ---------------------------------------------
    log("  solving reference finite-difference model ...")
    head = solve_steady(
        K=K, top=top, bottoms=bottoms, cellsize=cellsize, active=active,
        leakance=[1.0e-3], recharge=recharge,
        river_cond=river_cond, river_stage=river_stage,
        fault_mx=mx, fault_my=my,
        unconfined_top=True, min_thickness=2.0,
        n_picard=60, tol=1e-6, relax=0.6, verbose=False,
    )
    log(
        f"  truth heads: L0 {np.nanmin(head[0]):.1f}-{np.nanmax(head[0]):.1f} m, "
        f"L1 {np.nanmin(head[1]):.1f}-{np.nanmax(head[1]):.1f} m"
    )

    b0 = np.clip(head[0] - bottoms[0], 2.0, top - bottoms[0])
    b1 = np.maximum(bottoms[0] - bottoms[1], 2.0)
    T = [K[0] * b0, K[1] * b1]

    # ---- write rasters ---------------------------------------------------
    def w(name: str, arr: np.ndarray, sub: str = "") -> None:
        path = outdir / sub / name if sub else outdir / name
        write_raster(path, np.where(active, arr, np.nan), transform, CRS)

    w("dtm.tif", top)
    w("bottom_layer1.tif", bottoms_raw[0])
    w("bottom_layer2.tif", bottoms_raw[1])
    for l in range(2):
        w(f"head_L{l}.tif", head[l], "truth")
        w(f"K_L{l}.tif", K[l], "truth")
        w(f"T_L{l}.tif", T[l], "truth")
        w(f"S_L{l}.tif", S[l], "truth")
    w("bottom_layer1_corrected.tif", bottoms[0], "truth")
    w("bottom_layer2_corrected.tif", bottoms[1], "truth")

    # ---- vector inputs ---------------------------------------------------
    gpd.GeoDataFrame({"id": [1]}, geometry=[domain_poly], crs=CRS).to_file(
        outdir / "domain.shp"
    )
    gpd.GeoDataFrame({"name": ["reach"]}, geometry=[river_poly], crs=CRS).to_file(
        outdir / "river_polygon.shp"
    )
    gpd.GeoDataFrame(
        {"name": ["centerline"]}, geometry=[river_line], crs=CRS
    ).to_file(outdir / "river_centerline.shp")
    gpd.GeoDataFrame(
        {"name": ["F1_barrier", "F2_leaky"], "perm": fault_perm_flag},
        geometry=[LineString(fault_a), LineString(fault_b)],
        crs=CRS,
    ).to_file(outdir / "faults.shp")

    # gauges, evenly spaced along the reach
    g_s = np.linspace(0.03, 0.97, n_gauges) * s_cl[-1]
    g_xy = np.column_stack(
        [np.interp(g_s, s_cl, centerline[:, 0]), np.interp(g_s, s_cl, centerline[:, 1])]
    )
    g_stage = np.interp(g_s, s_cl, stage_cl) + rng.normal(0, 0.03, n_gauges)
    gpd.GeoDataFrame(
        {"name": [f"GS{i + 1:02d}" for i in range(n_gauges)], "stage": g_stage},
        geometry=[Point(*p) for p in g_xy],
        crs=CRS,
    ).to_file(outdir / "gauges.shp")

    # observation wells: partly scattered, partly clustered as in real networks
    well_xy = _sample_wells(domain_poly, river_poly, n_wells, rng)
    well_layer = (rng.random(len(well_xy)) < 0.42).astype(int)
    well_head = _bilinear(head, xs, ys, well_xy, well_layer)
    keep = np.isfinite(well_head)
    well_xy, well_layer, well_head = well_xy[keep], well_layer[keep], well_head[keep]
    well_head_obs = well_head + rng.normal(0, 0.05, len(well_head))
    gpd.GeoDataFrame(
        {
            "name": [f"OW{i + 1:03d}" for i in range(len(well_xy))],
            "head": well_head_obs,
            "layer": well_layer,
        },
        geometry=[Point(*p) for p in well_xy],
        crs=CRS,
    ).to_file(outdir / "head_obs.shp")

    # pumping tests -> T and S with realistic log-scale scatter
    pt_xy = _sample_wells(domain_poly, river_poly, n_pumping_tests, rng, cluster_frac=0.2)
    pt_layer = (rng.random(len(pt_xy)) < 0.30).astype(int)
    pt_T = _bilinear(np.stack(T), xs, ys, pt_xy, pt_layer)
    pt_S = _bilinear(np.stack(S), xs, ys, pt_xy, pt_layer)
    keep = np.isfinite(pt_T) & np.isfinite(pt_S)
    pt_xy, pt_layer, pt_T, pt_S = pt_xy[keep], pt_layer[keep], pt_T[keep], pt_S[keep]
    pt_T = pt_T * 10.0 ** rng.normal(0, 0.09, len(pt_T))
    pt_S = pt_S * 10.0 ** rng.normal(0, 0.10, len(pt_S))
    gpd.GeoDataFrame(
        {
            "name": [f"PT{i + 1:03d}" for i in range(len(pt_xy))],
            "T": pt_T,
            "S": pt_S,
            "layer": pt_layer,
        },
        geometry=[Point(*p) for p in pt_xy],
        crs=CRS,
    ).to_file(outdir / "aquifer_props.shp")

    log(
        f"  observations written: {len(well_xy)} wells, {n_gauges} gauges, "
        f"{len(pt_xy)} pumping tests"
    )

    meta = {
        "crs": CRS,
        "cellsize": cellsize,
        "extent": [xmin, ymin, xmax, ymax],
        "n_layers": 2,
        "leakance": [1.0e-3],
        "recharge": 1.2e-4,
        "river_conductance": 5.0e-2,
        "fault_perm_flag": fault_perm_flag,
        "fault_alpha_true": fault_alpha_true,
        "true_variograms": {k: v.as_dict() for k, v in vgm.items()},
        "overlap_report": report,
        "noise": {"head_m": 0.05, "log10T": 0.09, "log10S": 0.10, "stage_m": 0.03},
    }
    with open(outdir / "truth" / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=float)

    _write_config(outdir)

    return SyntheticCase(
        outdir=outdir, xs=xs, ys=ys, cellsize=cellsize, transform=transform,
        active=active, top=top, bottoms_raw=bottoms_raw, bottoms=bottoms,
        K=K, S=S, T=T, head=head, truth_variograms=vgm,
        fault_alpha=fault_alpha_true, meta=meta,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _distance_to_polyline(nodes: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    a, b = nodes[:-1], nodes[1:]
    ab = b - a
    L2 = np.maximum((ab**2).sum(1), 1e-12)
    best = np.full(len(pts), np.inf)
    # Chunked to keep the (N, S) intermediate small.
    for k in range(0, len(pts), 20000):
        p = pts[k : k + 20000]
        t = np.clip(((p[:, None, :] - a) * ab).sum(-1) / L2, 0, 1)
        foot = a + t[:, :, None] * ab
        d = np.linalg.norm(p[:, None, :] - foot, axis=2).min(axis=1)
        best[k : k + 20000] = d
    return best.reshape(xx.shape)


def _project_arclength(
    nodes: np.ndarray, s_nodes: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    pts = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])
    a, b = nodes[:-1], nodes[1:]
    ab = b - a
    L2 = np.maximum((ab**2).sum(1), 1e-12)
    t = np.clip(((pts[:, None, :] - a) * ab).sum(-1) / L2, 0, 1)
    foot = a + t[:, :, None] * ab
    d = np.linalg.norm(pts[:, None, :] - foot, axis=2)
    k = np.argmin(d, axis=1)
    seglen = np.sqrt(L2)
    return s_nodes[k] + t[np.arange(len(pts)), k] * seglen[k]


def _bilinear(
    stack: np.ndarray, xs: np.ndarray, ys: np.ndarray, xy: np.ndarray, layer: np.ndarray
) -> np.ndarray:
    """Sample ``stack[layer]`` at ``xy`` with bilinear interpolation."""
    out = np.full(len(xy), np.nan)
    for l in np.unique(layer):
        m = layer == l
        grid = stack[int(l)]
        fx = np.interp(xy[m, 0], xs, np.arange(len(xs)))
        fy = np.interp(xy[m, 1], ys[::-1], np.arange(len(ys))[::-1])
        i0 = np.clip(np.floor(fy).astype(int), 0, len(ys) - 2)
        j0 = np.clip(np.floor(fx).astype(int), 0, len(xs) - 2)
        ty, tx = fy - i0, fx - j0
        v = (
            grid[i0, j0] * (1 - ty) * (1 - tx)
            + grid[i0, j0 + 1] * (1 - ty) * tx
            + grid[i0 + 1, j0] * ty * (1 - tx)
            + grid[i0 + 1, j0 + 1] * ty * tx
        )
        out[m] = v
    return out


def _sample_wells(
    domain: Polygon,
    river: Polygon,
    n: int,
    rng: np.random.Generator,
    cluster_frac: float = 0.30,
) -> np.ndarray:
    """Scatter monitoring points, partly clustered, and never inside the river."""
    xmin, ymin, xmax, ymax = domain.bounds
    keep: List[np.ndarray] = []

    n_cluster = int(round(cluster_frac * n))
    n_uniform = n - n_cluster

    def accept(p: np.ndarray) -> np.ndarray:
        ok = shapely.contains_xy(domain, p[:, 0], p[:, 1])
        ok &= ~shapely.contains_xy(river.buffer(60.0), p[:, 0], p[:, 1])
        return p[ok]

    while sum(len(k) for k in keep) < n_uniform:
        cand = np.column_stack(
            [rng.uniform(xmin, xmax, 4 * n), rng.uniform(ymin, ymax, 4 * n)]
        )
        keep.append(accept(cand))
    uniform = np.vstack(keep)[:n_uniform]

    clustered: List[np.ndarray] = []
    if n_cluster > 0:
        n_centres = max(2, n_cluster // 5)
        centres_pool = np.vstack(keep)
        centres = centres_pool[rng.integers(0, len(centres_pool), n_centres)]
        while sum(len(c) for c in clustered) < n_cluster:
            c = centres[rng.integers(0, n_centres, 4 * n_cluster)]
            cand = c + rng.normal(0, 320.0, c.shape)
            clustered.append(accept(cand))
        clustered = [np.vstack(clustered)[:n_cluster]]

    return np.vstack([uniform] + clustered) if clustered else uniform


def _write_config(outdir: pathlib.Path) -> None:
    """A ready-to-run config pointing at the generated files."""
    from gwpinn.config import Config

    cfg = Config()
    p = cfg.paths
    p.workdir = "../runs/demo"
    p.head_obs = "head_obs.shp"
    p.gauge_obs = "gauges.shp"
    p.river_polygon = "river_polygon.shp"
    p.river_centerline = "river_centerline.shp"
    p.faults = "faults.shp"
    p.dtm = "dtm.tif"
    p.layer_bottoms = ["bottom_layer1.tif", "bottom_layer2.tif"]
    p.prop_obs = "aquifer_props.shp"
    p.domain = "domain.shp"

    cfg.physics.n_layers = 2
    cfg.physics.regime = "steady"
    cfg.physics.recharge = 1.2e-4
    cfg.physics.leakance = [1.0e-3]
    cfg.physics.river_conductance = 5.0e-2
    cfg.physics.fault_width = 60.0
    cfg.domain.min_thickness = 2.0
    # Assumed accuracy of a water-level reading (dip meter + datum survey).
    # The model fits the wells to within this and no further.
    cfg.train.head_noise = 0.05
    # A pumping test constrains T to roughly a factor of 1.3.
    cfg.train.prop_noise = 0.12

    cfg.save(outdir / "config.yaml")


__all__ = ["make_synthetic_case", "SyntheticCase"]
