"""Configuration objects describing the synthetic fault benchmark.

Everything downstream -- the MODFLOW 6 build, the reference finite-difference
solver, the PINN forcing terms and the evaluation code -- reads its geometry,
stresses and material properties from the objects defined here.  Keeping a
single source of truth is what makes the forward model and the inverse model
provably consistent: the PINN evaluates *the same* recharge/ET/well formulas
that the groundwater simulator applied.

Units are metres and days throughout.

Sign conventions
----------------
* ``z`` increases upwards; ``top`` is the top of layer 0.
* MODFLOW row 0 sits at the *northern* (maximum ``y``) edge of the grid.
* Well rates are negative for abstraction (MODFLOW convention).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "GridConfig",
    "FaultConfig",
    "AquiferConfig",
    "BoundaryConfig",
    "RechargeConfig",
    "EvapotranspirationConfig",
    "WellConfig",
    "TimeConfig",
    "ObservationConfig",
    "BenchmarkConfig",
    "SCENARIOS",
    "default_wells",
    "default_observation_wells",
]


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridConfig:
    """Regular structured grid (MODFLOW 6 DIS)."""

    Lx: float = 5000.0
    Ly: float = 3000.0
    ncol: int = 50
    nrow: int = 30
    top: float = 60.0
    botm: tuple[float, ...] = (40.0, 20.0, 0.0)

    @property
    def nlay(self) -> int:
        return len(self.botm)

    @property
    def dx(self) -> float:
        return self.Lx / self.ncol

    @property
    def dy(self) -> float:
        return self.Ly / self.nrow

    @property
    def zbot(self) -> float:
        return float(self.botm[-1])

    @property
    def Lz(self) -> float:
        return self.top - self.zbot

    @property
    def dz(self) -> np.ndarray:
        """Thickness of each layer, shape ``(nlay,)``."""
        edges = np.concatenate([[self.top], np.asarray(self.botm, dtype=float)])
        return edges[:-1] - edges[1:]

    @property
    def cell_volume(self) -> np.ndarray:
        """Cell volume per layer, shape ``(nlay,)``."""
        return self.dx * self.dy * self.dz

    def x_centers(self) -> np.ndarray:
        return (np.arange(self.ncol) + 0.5) * self.dx

    def y_centers(self) -> np.ndarray:
        """Row 0 is the northern edge, so ``y`` decreases with row index."""
        return self.Ly - (np.arange(self.nrow) + 0.5) * self.dy

    def z_centers(self) -> np.ndarray:
        edges = np.concatenate([[self.top], np.asarray(self.botm, dtype=float)])
        return 0.5 * (edges[:-1] + edges[1:])

    def cell_center_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Broadcast cell-centre coordinates, each of shape ``(nlay, nrow, ncol)``."""
        z, y, x = np.meshgrid(
            self.z_centers(), self.y_centers(), self.x_centers(), indexing="ij"
        )
        return x, y, z

    def locate(self, x: float, y: float) -> tuple[int, int]:
        """Return the ``(row, col)`` containing the point ``(x, y)``."""
        col = int(np.clip(x // self.dx, 0, self.ncol - 1))
        row = int(np.clip((self.Ly - y) // self.dy, 0, self.nrow - 1))
        return row, col

    def layer_of(self, z: float) -> int:
        """Return the layer index containing elevation ``z``."""
        edges = np.concatenate([[self.top], np.asarray(self.botm, dtype=float)])
        for k in range(self.nlay):
            if edges[k] >= z >= edges[k + 1]:
                return k
        return self.nlay - 1 if z < edges[-1] else 0


# --------------------------------------------------------------------------- #
# Fault
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FaultConfig:
    """A vertical planar fault zone of finite width bisecting the domain.

    The fault plane passes through ``(x0, y0)`` with in-plane strike rotated by
    ``strike_deg`` -- i.e. the plane normal is ``(cos a, sin a, 0)``.  The
    *signed distance* to that plane defines both the fault zone (``|d| <= w/2``)
    and the two blocks used by the cPINN domain decomposition (``d < -w/2`` and
    ``d > +w/2``).
    """

    x0: float = 2400.0
    y0: float = 1500.0
    strike_deg: float = 12.0
    width: float = 150.0
    k_barrier: float = 1.0e-3
    k_conduit: float = 200.0

    @property
    def normal(self) -> tuple[float, float]:
        a = math.radians(self.strike_deg)
        return math.cos(a), math.sin(a)

    def signed_distance(self, x, y):
        """Signed perpendicular distance to the fault plane (negative = west block)."""
        nx, ny = self.normal
        return (x - self.x0) * nx + (y - self.y0) * ny

    def in_fault_zone(self, x, y):
        return np.abs(self.signed_distance(x, y)) <= 0.5 * self.width

    def fault_k(self, scenario: str) -> float:
        if scenario == "barrier":
            return self.k_barrier
        if scenario == "conduit":
            return self.k_conduit
        raise ValueError(f"unknown scenario {scenario!r}; expected 'barrier' or 'conduit'")

    def conductance(self, scenario: str) -> float:
        """Fault-plane leakance ``C = K_f / w`` in 1/day.

        This is the scalar the cPINN infers directly: ``C -> 0`` is a perfect
        barrier, ``C -> inf`` recovers head continuity (a transparent fault).
        """
        return self.fault_k(scenario) / self.width


# --------------------------------------------------------------------------- #
# Aquifer material properties
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AquiferConfig:
    """Heterogeneous, log-normal hydraulic conductivity plus storage."""

    k_layer_gmean: tuple[float, ...] = (3.0, 0.4, 6.0)
    sigma_lnk: float = 0.8
    corr_len: tuple[float, float, float] = (900.0, 700.0, 25.0)
    specific_storage: float = 1.0e-4
    seed: int = 20240

    def __post_init__(self) -> None:
        if self.sigma_lnk < 0:
            raise ValueError("sigma_lnk must be non-negative")


# --------------------------------------------------------------------------- #
# Boundary conditions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BoundaryConfig:
    """Constant head on the west and east faces; everything else no-flow.

    The regional gradient runs west -> east, roughly perpendicular to the fault,
    which is what makes the barrier/conduit contrast observable in the heads.
    """

    head_west: float = 55.0
    head_east: float = 45.0


# --------------------------------------------------------------------------- #
# Distributed stresses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RechargeConfig:
    """Spatially variable areal recharge applied to layer 0 (MODFLOW RCH)."""

    base_rate: float = 1.0e-4
    amplitude: float = 0.6
    x_waves: float = 1.5
    y_waves: float = 1.0

    def rate_field(self, x: np.ndarray, y: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
        """Recharge rate [m/d] as a function of position. Never negative."""
        modulation = 1.0 + self.amplitude * np.sin(
            2.0 * np.pi * self.x_waves * x / Lx
        ) * np.cos(2.0 * np.pi * self.y_waves * y / Ly)
        return np.maximum(self.base_rate * modulation, 0.0)


@dataclass(frozen=True)
class EvapotranspirationConfig:
    """Head-dependent ET with the MODFLOW 6 linear ramp (EVT package).

    ``rate(h) = max_rate``                                 for ``h >= surface``
    ``rate(h) = max_rate * (h - (surface - extinction))/extinction``  in between
    ``rate(h) = 0``                                        for ``h <= surface - extinction``

    Because this is a *known constitutive law*, the PINN can evaluate it from
    its own predicted head -- the inverse problem never needs the true ET field
    handed to it.
    """

    surface: float = 58.0
    max_rate: float = 2.0e-4
    extinction_depth: float = 8.0
    amplitude: float = 0.5
    x_waves: float = 1.0
    y_waves: float = 1.5

    def max_rate_field(self, x: np.ndarray, y: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
        modulation = 1.0 + self.amplitude * np.cos(
            2.0 * np.pi * self.x_waves * x / Lx
        ) * np.sin(2.0 * np.pi * self.y_waves * y / Ly)
        return np.maximum(self.max_rate * modulation, 0.0)

    def rate(self, head: np.ndarray, max_rate: np.ndarray) -> np.ndarray:
        """Evaluate the ramp for a head field (NumPy version)."""
        frac = (head - (self.surface - self.extinction_depth)) / self.extinction_depth
        return max_rate * np.clip(frac, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Wells
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WellConfig:
    """A transient abstraction well. ``rates`` holds one rate per stress period."""

    name: str
    x: float
    y: float
    layer: int
    rates: tuple[float, ...]

    def rate_at(self, period: int) -> float:
        """Rate during transient stress period ``period`` (0-based). 0 outside range."""
        if 0 <= period < len(self.rates):
            return float(self.rates[period])
        return 0.0


def default_wells() -> tuple[WellConfig, ...]:
    """Three abstraction wells with staggered schedules over six stress periods.

    Two sit in the western block and one in the eastern block, so that the
    drawdown cones interact with the fault from both sides at different times.
    """
    return (
        WellConfig(
            name="PW1", x=1250.0, y=1550.0, layer=1,
            rates=(-900.0, -900.0, 0.0, 0.0, -900.0, -900.0),
        ),
        WellConfig(
            name="PW2", x=3350.0, y=1150.0, layer=2,
            rates=(0.0, 0.0, -1200.0, -1200.0, -1200.0, 0.0),
        ),
        WellConfig(
            name="PW3", x=1750.0, y=850.0, layer=0,
            rates=(-300.0, -300.0, -300.0, -600.0, -600.0, -600.0),
        ),
    )


# --------------------------------------------------------------------------- #
# Time discretisation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimeConfig:
    """One steady-state spin-up period followed by transient stress periods."""

    n_periods: int = 6
    period_length: float = 60.0
    steps_per_period: int = 6
    steady_state_spinup: bool = True

    @property
    def dt(self) -> float:
        return self.period_length / self.steps_per_period

    @property
    def total_time(self) -> float:
        return self.n_periods * self.period_length

    def output_times(self) -> np.ndarray:
        """End-of-timestep times, including t=0 for the steady-state spin-up."""
        times = [0.0]
        t = 0.0
        for _ in range(self.n_periods):
            for _ in range(self.steps_per_period):
                t += self.dt
                times.append(t)
        return np.asarray(times, dtype=float)

    def period_of(self, t: float) -> int:
        """Transient stress period index containing time ``t`` (t>0)."""
        if t <= 0.0:
            return -1
        idx = int(math.ceil(t / self.period_length)) - 1
        return int(np.clip(idx, 0, self.n_periods - 1))


# --------------------------------------------------------------------------- #
# Monitoring network
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObservationConfig:
    """Sparse monitoring well network -- the *only* training signal."""

    layers: tuple[int, ...] = (0, 2)
    noise_std: float = 0.015
    seed: int = 7
    min_fault_offset: float = 120.0


def default_observation_wells() -> tuple[tuple[str, float, float], ...]:
    """18 monitoring locations, deliberately clustered around the fault trace.

    Each location is screened in the layers listed in :class:`ObservationConfig`,
    giving 36 observation points -- ~0.8% of the 4500 model cells.
    """
    return (
        ("OBS01", 400.0, 2500.0),
        ("OBS02", 900.0, 700.0),
        ("OBS03", 1500.0, 2200.0),
        ("OBS04", 1650.0, 1300.0),
        ("OBS05", 2050.0, 2600.0),
        ("OBS06", 2000.0, 1750.0),
        ("OBS07", 2150.0, 500.0),
        ("OBS08", 2350.0, 2100.0),
        ("OBS09", 2500.0, 900.0),
        ("OBS10", 2750.0, 2400.0),
        ("OBS11", 2800.0, 1450.0),
        ("OBS12", 3050.0, 350.0),
        ("OBS13", 3200.0, 2050.0),
        ("OBS14", 3600.0, 1600.0),
        ("OBS15", 3900.0, 800.0),
        ("OBS16", 4300.0, 2350.0),
        ("OBS17", 4500.0, 1200.0),
        ("OBS18", 1100.0, 2000.0),
    )


# --------------------------------------------------------------------------- #
# Top-level benchmark configuration
# --------------------------------------------------------------------------- #
SCENARIOS = ("barrier", "conduit")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Complete specification of one synthetic experiment."""

    scenario: str = "barrier"
    grid: GridConfig = field(default_factory=GridConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    aquifer: AquiferConfig = field(default_factory=AquiferConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    recharge: RechargeConfig = field(default_factory=RechargeConfig)
    et: EvapotranspirationConfig = field(default_factory=EvapotranspirationConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    observations: ObservationConfig = field(default_factory=ObservationConfig)
    wells: tuple[WellConfig, ...] = field(default_factory=default_wells)
    obs_wells: tuple[tuple[str, float, float], ...] = field(
        default_factory=default_observation_wells
    )

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got {self.scenario!r}")
        for w in self.wells:
            if len(w.rates) != self.time.n_periods:
                raise ValueError(
                    f"well {w.name} has {len(w.rates)} rates but "
                    f"{self.time.n_periods} stress periods are configured"
                )
            if not 0 <= w.layer < self.grid.nlay:
                raise ValueError(f"well {w.name} layer {w.layer} outside grid")

    def for_scenario(self, scenario: str) -> "BenchmarkConfig":
        return replace(self, scenario=scenario)

    @property
    def fault_k(self) -> float:
        return self.fault.fault_k(self.scenario)

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkConfig":
        def rebuild(kind, data):
            return kind(**data)

        return cls(
            scenario=payload["scenario"],
            grid=GridConfig(**_tuplify(payload["grid"], ("botm",))),
            fault=FaultConfig(**payload["fault"]),
            aquifer=AquiferConfig(
                **_tuplify(payload["aquifer"], ("k_layer_gmean", "corr_len"))
            ),
            boundary=BoundaryConfig(**payload["boundary"]),
            recharge=RechargeConfig(**payload["recharge"]),
            et=EvapotranspirationConfig(**payload["et"]),
            time=TimeConfig(**payload["time"]),
            observations=ObservationConfig(**_tuplify(payload["observations"], ("layers",))),
            wells=tuple(
                WellConfig(**_tuplify(w, ("rates",))) for w in payload["wells"]
            ),
            obs_wells=tuple(tuple(w) for w in payload["obs_wells"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _tuplify(data: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """JSON round-trips tuples as lists; restore them so dataclasses stay hashable."""
    out = dict(data)
    for key in keys:
        if key in out and out[key] is not None:
            out[key] = tuple(out[key])
    return out
