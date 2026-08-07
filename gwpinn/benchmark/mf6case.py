"""The reduced benchmark case: a single-layer transient aquifer with everything hard.

Why a *reduced* case
--------------------
The full benchmark is five layers x 100 x 100 cells x 360 stress periods.  Fitting
even one physics-informed surrogate to that on a CPU is a multi-day job, and
comparing a dozen formulations on it is a small research programme.  Worse, a
depth-integrated surrogate of a five-layer model carries a *structural* model
error (the vertical leakage it cannot see) which would dominate every mass-balance
and river-exchange metric and make the comparison meaningless.

So the study uses a reduction that is **exactly the equation the surrogates
solve**:

.. math::

    S_y \\frac{\\partial h}{\\partial t}
      = \\nabla\\!\\cdot\\!\\big(K\\,b(h)\\,A\\,\\nabla h\\big)
        + R(t) + \\Gamma^{riv}(h, t) + Q_{wel}(t)

on the benchmark's own grid, with the benchmark's own heterogeneous ``K``, its
meandering river, its two faults and its 2-hourly forcing.  The reference
solution is produced by MODFLOW 6 solving that same single-layer equation, so
the reference budget closes to machine precision and a surrogate's mass-balance
error is entirely its own.  Nothing about the difficulty is lost: the K field
still spans three orders of magnitude at a 200 m correlation length, the water
table is still unconfined (so the PDE is nonlinear in ``h``), the barriers still
produce genuine head discontinuities and the tide still runs at 12.4 h.

Everything is read back from the GeoTIFF / Shapefile / CSV exports rather than
re-derived, which keeps this module independent of the generator script and
doubles as a check that those exports really are sufficient to rebuild the
problem.

Time window
-----------
The default window is days 5-9 (48 stress periods of 2 h).  That is the busiest
stretch of the benchmark: it contains the first 12-hour storm, the first flash
flood, eight tidal cycles and the full diurnal pumping signal.  A surrogate that
handles this window handles the rest.
"""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Case container
# --------------------------------------------------------------------------- #


@dataclass
class ReducedCase:
    """A fully specified 2-D transient aquifer problem plus its reference solution.

    Arrays are ``(nrow, ncol)`` in raster order (row 0 = north) unless noted.
    Times are days; lengths metres; fluxes m/d per unit area unless noted.
    """

    # ---- grid -------------------------------------------------------------
    nrow: int
    ncol: int
    delr: float
    delc: float
    x_origin: float
    y_origin: float
    crs_epsg: int

    # ---- aquifer properties (the inverse target) --------------------------
    kh: np.ndarray            # (nrow, ncol) m/d - TRUTH for the inverse task
    sy: np.ndarray            # (nrow, ncol) -
    ss: np.ndarray            # (nrow, ncol) 1/m
    top: np.ndarray           # (nrow, ncol) land surface, m
    botm: np.ndarray          # (nrow, ncol) aquifer base, m

    # ---- faults -----------------------------------------------------------
    #: One row per blocked cell face: (row1, col1, row2, col2, hydchr).
    fault_faces: np.ndarray   # (nface, 5)
    fault_names: List[str]

    # ---- river ------------------------------------------------------------
    riv_cells: np.ndarray     # (nriv, 2) row, col
    riv_cond: float           # m2/d per cell
    riv_bottom: float         # m

    # ---- wells ------------------------------------------------------------
    wel_cells: np.ndarray     # (nwel, 2) row, col
    wel_names: List[str]

    # ---- time and forcing -------------------------------------------------
    times: np.ndarray         # (nper,) end-of-period time, days
    dt: float                 # stress period length, days
    riv_stage: np.ndarray     # (nper,)   m
    recharge: np.ndarray      # (nper,)   m/d
    wel_q: np.ndarray         # (nwel, nper) m3/d, negative = abstraction

    # ---- reference solution ----------------------------------------------
    head: np.ndarray          # (nper, nrow, ncol) m
    head_init: np.ndarray     # (nrow, ncol) m, start of the window
    #: Reference cell-by-cell budget terms, (nper,) totals in m3/d.
    budget: Dict[str, np.ndarray] = field(default_factory=dict)
    #: Per-cell river leakage, (nper, nriv) m3/d, positive = into the aquifer.
    riv_leakage: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> Tuple[int, int]:
        return self.nrow, self.ncol

    @property
    def nper(self) -> int:
        return len(self.times)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) for imshow."""
        return (
            self.x_origin,
            self.x_origin + self.ncol * self.delr,
            self.y_origin,
            self.y_origin + self.nrow * self.delc,
        )

    def cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """(x, y) cell-centre coordinate arrays, each ``(nrow, ncol)``."""
        xc = self.x_origin + (np.arange(self.ncol) + 0.5) * self.delr
        yc = self.y_origin + (self.nrow - np.arange(self.nrow) - 0.5) * self.delc
        return np.meshgrid(xc, yc)

    def fault_face_segments(self) -> List[Tuple[float, float, float, float]]:
        """Real-world endpoints of every blocked face, for plotting."""
        segs = []
        for r1, c1, r2, c2, _ in self.fault_faces:
            r1, c1, r2, c2 = int(r1), int(c1), int(r2), int(c2)
            if r1 == r2:  # vertical face between horizontal neighbours
                x = self.x_origin + max(c1, c2) * self.delr
                y0 = self.y_origin + (self.nrow - r1 - 1) * self.delc
                segs.append((x, y0, x, y0 + self.delc))
            else:
                y = self.y_origin + (self.nrow - max(r1, r2)) * self.delc
                x0 = self.x_origin + c1 * self.delr
                segs.append((x0, y, x0 + self.delr, y))
        return segs

    # ------------------------------------------------------------------ #

    def save(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                payload[f.name] = v
            elif f.name == "budget":
                for k, arr in v.items():
                    payload[f"budget__{k}"] = arr
            elif isinstance(v, list):
                payload[f.name] = np.array(v, dtype=object)
            elif v is not None:
                payload[f.name] = np.array(v)
        np.savez_compressed(path, **payload)


def load_reduced_case(path: pathlib.Path) -> ReducedCase:
    """Read a case written by :meth:`ReducedCase.save`."""
    z = np.load(path, allow_pickle=True)
    kwargs: Dict[str, object] = {}
    budget: Dict[str, np.ndarray] = {}
    scalars = {"nrow", "ncol", "crs_epsg"}
    floats = {"delr", "delc", "x_origin", "y_origin", "riv_cond", "riv_bottom", "dt"}
    for key in z.files:
        if key.startswith("budget__"):
            budget[key[len("budget__"):]] = z[key]
        elif key in scalars:
            kwargs[key] = int(z[key])
        elif key in floats:
            kwargs[key] = float(z[key])
        elif key in ("fault_names", "wel_names"):
            kwargs[key] = list(z[key])
        else:
            kwargs[key] = z[key]
    kwargs["budget"] = budget
    return ReducedCase(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Building the case from the benchmark exports
# --------------------------------------------------------------------------- #


def _read_tif(path: pathlib.Path) -> np.ndarray:
    import rasterio

    with rasterio.open(str(path)) as src:
        arr = src.read(1, masked=True).astype(float)
    return np.where(np.ma.getmaskarray(arr), np.nan, np.ma.getdata(arr))


def _tif_meta(path: pathlib.Path) -> Tuple[int, int, float, float, float, float, int]:
    import rasterio

    with rasterio.open(str(path)) as src:
        tr = src.transform
        nrow, ncol = src.height, src.width
        delr, delc = abs(tr.a), abs(tr.e)
        x_origin = tr.c
        y_origin = tr.f - nrow * delc
        epsg = src.crs.to_epsg() if src.crs else 0
    return nrow, ncol, delr, delc, x_origin, y_origin, int(epsg or 0)


def build_reduced_case(
    bench_dir: str | pathlib.Path,
    t_start: float = 5.0,
    n_periods: int = 48,
    layer: int = 2,
    base_layer: int = 5,
    workdir: Optional[str | pathlib.Path] = None,
    exe: Optional[str] = None,
    quiet: bool = False,
) -> ReducedCase:
    """Assemble the reduced case and solve it with MODFLOW 6.

    ``bench_dir`` is the workspace produced by
    ``scripts/build_mf6_hetero_benchmark.py`` (the one containing ``gis/`` and
    ``tables/``).  Everything is read from those exports.

    The reference run is a genuine MODFLOW 6 single-layer model: a steady-state
    spin-up under the forcing at ``t_start`` supplies the initial condition, then
    ``n_periods`` transient stress periods reproduce the requested window.  Its
    cell-by-cell budget is saved so that a surrogate's mass balance and river
    exchange can be scored against an exactly conserved reference.
    """
    import flopy
    import geopandas as gpd

    bench = pathlib.Path(bench_dir)
    gis, tables = bench / "gis", bench / "tables"
    if not gis.is_dir():
        raise FileNotFoundError(
            f"{gis} not found - run scripts/build_mf6_hetero_benchmark.py first"
        )

    def say(msg: str) -> None:
        if not quiet:
            print(msg)

    # ---- grid and properties from the rasters ------------------------------
    kh_path = gis / f"kh_L{layer}.tif"
    nrow, ncol, delr, delc, x0, y0, epsg = _tif_meta(kh_path)
    kh = _read_tif(kh_path)
    sy = _read_tif(gis / f"sy_L{layer}.tif")
    ss = _read_tif(gis / f"ss_L{layer}.tif")
    top = _read_tif(gis / "land_surface.tif")

    # The single layer spans the whole saturated section, land surface down to
    # the base of ``base_layer`` (the basement by default), while its hydraulic
    # properties are the benchmark's Layer-``layer`` realisation.  Taking the
    # thickness of Layer 2 alone would give a transmissivity ~4x smaller than the
    # five-layer system's, and the benchmark's recharge - which is prescribed and
    # unchanged - would then pile the water table tens of metres above ground.
    # Spanning the full section keeps T comparable to the parent model, so the
    # reduced case sits in the same hydraulic regime while remaining a single,
    # exactly-solvable 2-D equation.
    botm = _read_tif(gis / f"botm_L{base_layer}.tif")

    # Layer 2 is active everywhere in the benchmark, so a single-layer reduction
    # has no inactive cells - but fill any NaN defensively so MODFLOW never sees
    # one.
    for arr in (kh, sy, ss, botm, top):
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.nanmedian(arr)
    top = np.maximum(top, botm + 5.0)
    say(f"[case] grid {nrow}x{ncol} @ {delr:g} m | K {kh.min():.3g}-{kh.max():.3g} m/d")

    # ---- faults ------------------------------------------------------------
    faces = gpd.read_file(gis / "faults_hfb_faces.shp")
    fault_faces = np.column_stack([
        faces["r1"].to_numpy(float), faces["c1"].to_numpy(float),
        faces["r2"].to_numpy(float), faces["c2"].to_numpy(float),
        faces["hydchr"].to_numpy(float),
    ])
    fault_names = sorted(set(faces["name"].astype(str)))
    say(f"[case] {len(fault_faces)} blocked faces from {len(fault_names)} faults")

    # ---- river -------------------------------------------------------------
    rivs = gpd.read_file(gis / "bc_riv_cells.shp")
    riv_cells = np.column_stack([rivs["row"].to_numpy(int), rivs["col"].to_numpy(int)])
    riv_cond = float(rivs["cond"].iloc[0])
    riv_bottom = float(rivs["rbot"].iloc[0])

    # ---- wells -------------------------------------------------------------
    wells = gpd.read_file(gis / "bc_wells.shp")
    wel_cells = np.column_stack([wells["row"].to_numpy(int), wells["col"].to_numpy(int)])
    wel_names = [str(n) for n in wells["name"]]

    # ---- forcing window ----------------------------------------------------
    forcing = pd.read_csv(tables / "forcing_by_stress_period.csv")
    dt = float(forcing["t_end_d"].iloc[0] - forcing["t_start_d"].iloc[0])
    i0 = int(np.argmin(np.abs(forcing["t_start_d"].to_numpy() - t_start)))
    sel = forcing.iloc[i0 : i0 + n_periods].reset_index(drop=True)
    if len(sel) < n_periods:
        raise ValueError(f"only {len(sel)} stress periods available from t={t_start}")

    times = sel["t_end_d"].to_numpy(float)
    riv_stage = sel["riv_stage_m"].to_numpy(float)
    recharge = sel["rch_m_per_d"].to_numpy(float)
    wel_q = np.stack([sel[f"wel_{n}_m3d"].to_numpy(float) for n in wel_names])
    say(f"[case] window t={sel['t_start_d'].iloc[0]:.3f}-{times[-1]:.3f} d, "
        f"{n_periods} periods of {dt * 24:g} h")

    # ---- build and run the reference MODFLOW model -------------------------
    ws = pathlib.Path(workdir or (bench / "reduced_case" / "mf6"))
    ws.mkdir(parents=True, exist_ok=True)
    if exe is None:
        import shutil as _sh

        exe = _sh.which("mf6") or "mf6"

    head, head_init, budget, riv_leakage = _run_reference(
        ws=ws, exe=exe, nrow=nrow, ncol=ncol, delr=delr, delc=delc,
        x0=x0, y0=y0, epsg=epsg, kh=kh, sy=sy, ss=ss, top=top, botm=botm,
        fault_faces=fault_faces, riv_cells=riv_cells, riv_cond=riv_cond,
        riv_bottom=riv_bottom, wel_cells=wel_cells, wel_names=wel_names,
        dt=dt, riv_stage=riv_stage, recharge=recharge, wel_q=wel_q, say=say,
    )

    return ReducedCase(
        nrow=nrow, ncol=ncol, delr=delr, delc=delc, x_origin=x0, y_origin=y0,
        crs_epsg=epsg, kh=kh, sy=sy, ss=ss, top=top, botm=botm,
        fault_faces=fault_faces, fault_names=fault_names,
        riv_cells=riv_cells, riv_cond=riv_cond, riv_bottom=riv_bottom,
        wel_cells=wel_cells, wel_names=wel_names,
        times=times, dt=dt, riv_stage=riv_stage, recharge=recharge, wel_q=wel_q,
        head=head, head_init=head_init, budget=budget, riv_leakage=riv_leakage,
    )


def _run_reference(
    *, ws, exe, nrow, ncol, delr, delc, x0, y0, epsg, kh, sy, ss, top, botm,
    fault_faces, riv_cells, riv_cond, riv_bottom, wel_cells, wel_names,
    dt, riv_stage, recharge, wel_q, say,
):
    """Steady spin-up + transient window, both single-layer MODFLOW 6 runs."""
    import flopy

    nper = len(riv_stage)

    def _build(sim_ws, perioddata, transient, strt, stage, rch, qcols):
        sim = flopy.mf6.MFSimulation(sim_name="red", exe_name=exe, sim_ws=str(sim_ws))
        flopy.mf6.ModflowTdis(sim, time_units="days", nper=len(perioddata),
                              perioddata=perioddata)
        flopy.mf6.ModflowIms(
            sim, print_option="SUMMARY", complexity="COMPLEX",
            outer_dvclose=1e-4, outer_maximum=200, inner_dvclose=1e-5,
            inner_maximum=300, linear_acceleration="BICGSTAB",
            under_relaxation="DBD", under_relaxation_theta=0.85,
            backtracking_number=10,
        )
        gwf = flopy.mf6.ModflowGwf(sim, modelname="red", save_flows=True,
                                   newtonoptions="NEWTON UNDER_RELAXATION")
        flopy.mf6.ModflowGwfdis(
            gwf, length_units="meters", nlay=1, nrow=nrow, ncol=ncol,
            delr=delr, delc=delc, top=top, botm=botm[None, :, :],
            xorigin=x0, yorigin=y0,
        )
        flopy.mf6.ModflowGwfic(gwf, strt=strt)
        flopy.mf6.ModflowGwfnpf(gwf, save_flows=True, save_specific_discharge=True,
                                icelltype=1, k=kh, k33=kh / 10.0)
        if transient:
            flopy.mf6.ModflowGwfsto(gwf, save_flows=True, iconvert=1, ss=ss, sy=sy,
                                    steady_state={0: False}, transient={0: True})
        hfb = [[(0, int(r1), int(c1)), (0, int(r2), int(c2)), float(hc)]
               for r1, c1, r2, c2, hc in fault_faces]
        flopy.mf6.ModflowGwfhfb(gwf, maxhfb=len(hfb), stress_period_data={0: hfb})

        riv_spd = {
            k: [((0, int(r), int(c)), float(stage[k]), riv_cond, riv_bottom)
                for r, c in riv_cells]
            for k in range(len(perioddata))
        }
        flopy.mf6.ModflowGwfriv(gwf, save_flows=True, maxbound=len(riv_cells),
                                stress_period_data=riv_spd, pname="riv")
        wel_spd = {
            k: [((0, int(r), int(c)), float(qcols[i, k]))
                for i, (r, c) in enumerate(wel_cells)]
            for k in range(len(perioddata))
        }
        flopy.mf6.ModflowGwfwel(gwf, save_flows=True, maxbound=len(wel_cells),
                                stress_period_data=wel_spd, pname="wel")
        flopy.mf6.ModflowGwfrcha(gwf, save_flows=True, readasarrays=True,
                                 recharge={k: float(rch[k]) for k in range(len(perioddata))})
        flopy.mf6.ModflowGwfoc(
            gwf, head_filerecord="red.hds", budget_filerecord="red.cbc",
            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        )
        return sim

    # --- steady spin-up under the first period's forcing --------------------
    ss_ws = pathlib.Path(ws).parent / "mf6_spinup"
    ss_ws.mkdir(parents=True, exist_ok=True)
    sim = _build(ss_ws, [(1.0, 1, 1.0)], False,
                 np.maximum(top - 3.0, riv_stage[0]),
                 riv_stage[:1], recharge[:1], wel_q[:, :1])
    sim.write_simulation(silent=True)
    ok, buff = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError("reduced-case spin-up failed:\n" + "\n".join(map(str, buff[-20:])))
    h0 = flopy.utils.HeadFile(str(ss_ws / "red.hds")).get_data(kstpkper=(0, 0))[0]
    say(f"[case] spin-up head {h0.min():.2f}-{h0.max():.2f} m")

    # --- transient window ---------------------------------------------------
    sim = _build(ws, [(dt, 1, 1.0)] * nper, True, h0[None, :, :],
                 riv_stage, recharge, wel_q)
    sim.write_simulation(silent=True)
    ok, buff = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError("reduced-case run failed:\n" + "\n".join(map(str, buff[-25:])))

    hds = flopy.utils.HeadFile(str(pathlib.Path(ws) / "red.hds"))
    head = np.stack([hds.get_data(totim=t)[0] for t in hds.get_times()])
    hds.close()

    cbc = flopy.utils.CellBudgetFile(str(pathlib.Path(ws) / "red.cbc"))
    kstpkper = cbc.get_kstpkper()
    texts = [t.decode().strip() for t in cbc.get_unique_record_names()]

    budget: Dict[str, np.ndarray] = {}
    riv_leak = np.zeros((nper, len(riv_cells)))
    riv_index = {(int(r), int(c)): i for i, (r, c) in enumerate(riv_cells)}
    for name in ("RIV", "WEL", "RCHA", "STO-SS", "STO-SY"):
        if name not in texts:
            continue
        totals = np.zeros(nper)
        for k, kk in enumerate(kstpkper):
            rec = cbc.get_data(text=name, kstpkper=kk)[0]
            # List-based packages (RIV, WEL) come back as recarrays with node/q;
            # array-based ones (RCH via READASARRAYS, STO) come back as full
            # 3-D arrays.  Both have to be handled.
            if getattr(rec, "dtype", None) is not None and rec.dtype.names:
                q = np.asarray(rec["q"], dtype=float)
                nodes = np.asarray(rec["node"], dtype=int) - 1
            else:
                q = np.asarray(rec, dtype=float).ravel()
                nodes = np.arange(q.size)
            totals[k] = float(np.nansum(q))
            if name == "RIV":
                for node, qi in zip(nodes, q):
                    rc = (int(node) // ncol, int(node) % ncol)
                    if rc in riv_index:
                        riv_leak[k, riv_index[rc]] = qi
        budget[name] = totals
    cbc.close()

    inflow = sum(np.clip(v, 0, None).sum() for v in budget.values())
    say(f"[case] reference solved: head {head.min():.2f}-{head.max():.2f} m, "
        f"|budget terms| {len(budget)}, mean gross flux {inflow / nper:,.0f} m3/d")
    return head, h0, budget, riv_leak


__all__ = ["ReducedCase", "build_reduced_case", "load_reduced_case"]
