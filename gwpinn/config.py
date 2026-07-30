"""Configuration schema for gwpinn.

The whole pipeline is driven by a single nested dataclass tree that can be
round-tripped through YAML, so every run is reproducible from its config file.
"""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #


@dataclass
class PathsConfig:
    """Filesystem inputs. Every entry is optional so partial datasets work."""

    workdir: str = "runs/demo"

    #: Point shapefile of observed groundwater heads.
    #: Attributes: ``head`` (m a.s.l.), optional ``layer`` (0-based), optional
    #: ``time`` (days, transient runs), optional ``weight``.
    head_obs: Optional[str] = None

    #: Point shapefile of river gauge stations. Attribute: ``stage`` (m a.s.l.),
    #: optional ``time``.
    gauge_obs: Optional[str] = None

    #: Polygon shapefile of the river water surface (wetted area).
    river_polygon: Optional[str] = None

    #: Optional line shapefile giving the river centerline. If absent, a
    #: centerline is derived from ``river_polygon`` by principal-curve fitting.
    river_centerline: Optional[str] = None

    #: Line shapefile of faults / flow barriers. Attribute ``perm``:
    #: 0 -> impermeable (barrier), 1 -> permeability is a trainable parameter.
    #: Any other value in (0, 1] is used as a fixed permeability multiplier.
    faults: Optional[str] = None

    #: Digital terrain model raster - top elevation of layer 0.
    dtm: Optional[str] = None

    #: Bottom-elevation rasters, ordered top -> bottom, one per aquifer layer.
    layer_bottoms: List[str] = field(default_factory=list)

    #: Point shapefile of pumping-test derived aquifer properties.
    #: Attributes: ``T`` (m^2/d), ``S`` (-), optional ``layer``.
    prop_obs: Optional[str] = None

    #: Optional polygon shapefile restricting the active model domain.
    domain: Optional[str] = None

    #: Optional line shapefile of prescribed boundary conditions.
    #: Attributes: ``bctype`` in {"noflow", "head", "ghb"}, ``value`` (m),
    #: optional ``cond`` (conductance, m^2/d per m for "ghb").
    boundary: Optional[str] = None

    #: Optional recharge raster (m/d). Constant ``physics.recharge`` used if absent.
    recharge: Optional[str] = None


@dataclass
class DomainConfig:
    """Spatial discretisation of the model domain."""

    #: Output raster cell size in CRS units. ``None`` -> inherit from the DTM.
    cellsize: Optional[float] = None

    #: Clip the domain to the DTM's valid-data footprint.
    use_dtm_footprint: bool = True

    #: Minimum saturated / layer thickness enforced everywhere (m).
    min_thickness: float = 1.0

    #: Shrink the raster used for the report plots by this factor (speed only).
    plot_downsample: int = 1


@dataclass
class PhysicsConfig:
    """Groundwater flow physics."""

    #: Number of aquifer layers. Inferred from ``paths.layer_bottoms`` if 0.
    n_layers: int = 0

    #: "steady" or "transient". Storage coefficients are only identifiable
    #: from the flow equation in transient mode (see README).
    regime: str = "steady"

    #: Simulation times (days) used for transient collocation.
    times: List[float] = field(default_factory=list)

    #: Areal recharge to the top layer (m/d) when no recharge raster is given.
    recharge: float = 3.0e-4

    #: Treat layer 0 as unconfined (Boussinesq: T = K * saturated thickness).
    unconfined_top: bool = True

    #: Vertical leakance between adjacent layers (1/d). One entry per interface
    #: (``n_layers - 1``). Empty -> use ``leakance_default`` for all interfaces.
    leakance: List[float] = field(default_factory=list)
    leakance_default: float = 1.0e-3

    #: Train the vertical leakance values instead of holding them fixed.
    train_leakance: bool = True

    #: Riverbed conductance per unit area (1/d). Trainable when
    #: ``train_river_conductance``.
    river_conductance: float = 5.0e-2
    train_river_conductance: bool = True

    #: Gaussian smear half-width of a fault barrier, in CRS units. Faults are
    #: represented as narrow anisotropic low-permeability zones rather than as
    #: true discontinuities (see ``gwpinn.physics.faults``).
    fault_width: float = 60.0

    #: Lower bound on the fault permeability multiplier. A fault flagged
    #: ``perm == 0`` uses this value; a perfectly impermeable barrier would
    #: require an unbounded head gradient, which a continuous network cannot
    #: represent.
    fault_perm_min: float = 1.0e-3

    #: Bounds on hydraulic conductivity (m/d), used to squash the property net.
    k_min: float = 1.0e-3
    k_max: float = 5.0e2

    #: Bounds on the storage coefficient (-).
    s_min: float = 1.0e-5
    s_max: float = 3.0e-1


@dataclass
class ModelConfig:
    """Network architecture."""

    #: One of "mlp", "resnet", "modified_mlp", "cnn". The grid CNN wins on this
    #: problem because it *cannot* overfit a sparse well network - see
    #: docs/architecture_comparison.md. Use "resnet" when barriers are narrower
    #: than the CNN's pixel pitch, when the domain is fragmented, or when wall
    #: clock is the binding constraint.
    arch: str = "cnn"

    #: Hidden width / depth of the head network.
    width: int = 96
    depth: int = 4

    #: Hidden width / depth of the aquifer-property network.
    prop_width: int = 64
    prop_depth: int = 3

    activation: str = "tanh"

    #: Random Fourier feature embedding. ``mapping_size`` random frequencies are
    #: drawn from N(0, sigma^2); this is what lets the network represent the
    #: high-frequency structure of a heterogeneous property field.
    fourier_features: int = 48
    fourier_sigma: float = 3.0

    #: A second, higher-frequency Fourier bank for the property network. Property
    #: fields are rougher than head fields, which are smoothed by the PDE.
    prop_fourier_sigma: float = 6.0

    #: Feed each collocation point a signed side-indicator per fault so the
    #: network can represent the sharp head drop across a barrier.
    fault_features: bool = True

    #: Put those indicators *inside* the Fourier embedding rather than appending
    #: them after it, so the basis functions themselves are steep across a fault
    #: (see gwpinn.models.fields._FeatureMixin). Only meaningful for coordinate
    #: networks - the grid CNN reads its field by position and ignores these
    #: features entirely.
    fault_coords: bool = False

    #: Fourier bandwidth applied to the mapped fault coordinates, relative to
    #: the spatial ones. Must be small: tanh already saturates within a barrier
    #: width, so a full-bandwidth band would oscillate several times inside the
    #: barrier and make the second-order residual diverge.
    fault_sigma_scale: float = 0.15

    #: CNN-only: shape of the latent grid the decoder upsamples from.
    cnn_latent: int = 16
    cnn_channels: int = 64


@dataclass
class LossWeights:
    """Static multipliers applied on top of the adaptive weights."""

    head_obs: float = 1.0
    pde: float = 1.0
    river: float = 1.0
    prop_obs: float = 1.0
    variogram: float = 0.5
    boundary: float = 1.0
    smooth: float = 1.0e-2
    kriging: float = 1.0


@dataclass
class TrainConfig:
    """Optimisation settings."""

    seed: int = 0
    device: str = "cpu"

    #: "float32" or "float64". Second derivatives of a tanh network are noisy in
    #: single precision; float64 costs little on CPU and makes L-BFGS behave.
    dtype: str = "float32"

    #: Adam stage.
    adam_iters: int = 6000
    lr: float = 2.0e-3
    lr_decay: float = 0.9
    lr_decay_every: int = 1000

    #: L-BFGS refinement stage (0 to skip).
    lbfgs_iters: int = 300

    #: Collocation points resampled every ``resample_every`` iterations.
    n_collocation: int = 2048
    n_boundary: int = 384
    n_river: int = 384

    #: Extra collocation points drawn inside the fault barrier zones, where a
    #: uniform sample lands almost nothing.
    n_fault_colloc: int = 512
    resample_every: int = 100

    #: Assumed measurement accuracy of the head observations (m). Residuals
    #: smaller than this cost nothing, so the model fits the data to within its
    #: uncertainty and no further - the chi-square-of-one target used in
    #: conventional groundwater calibration. 0 disables it (plain least squares).
    head_noise: float = 0.0

    #: Assumed accuracy of the pumping-test properties, in log10 units. A
    #: pumping test pins transmissivity to perhaps a factor of 1.5 (0.18 in
    #: log10); fitting the points exactly makes the field spike at them and
    #: collapse in between.
    prop_noise: float = 0.0

    #: Gradient-norm adaptive loss balancing (Wang et al., 2021).
    adaptive_weights: bool = True
    adaptive_alpha: float = 0.9
    adaptive_every: int = 100

    #: Ceiling on an adaptive weight. As a loss term approaches zero its
    #: gradient does too, so the raw rule sends its weight to infinity and the
    #: optimiser chases noise in that one term.
    max_adaptive_weight: float = 1.0e3

    #: Number of point pairs per variogram evaluation.
    n_variogram_pairs: int = 2048

    #: Curriculum: ramp the PDE weight in over this many iterations so the
    #: network first fits the data, then the physics.
    pde_warmup: int = 1500

    log_every: int = 250

    #: Deep ensemble size. >1 gives a spatial uncertainty estimate.
    n_ensemble: int = 1

    #: Fraction of head observations held out for validation.
    val_fraction: float = 0.2

    weights: LossWeights = field(default_factory=LossWeights)


@dataclass
class VariogramConfig:
    """Geostatistical structure of the aquifer property fields."""

    #: "exponential", "spherical", or "gaussian".
    model: str = "exponential"

    n_lags: int = 12

    #: Maximum lag as a fraction of the domain diagonal.
    max_lag_fraction: float = 0.5

    #: Fit the variogram on log10 of the property (recommended: T and S are
    #: log-normally distributed).
    log_transform: bool = True

    #: Override the fitted parameters (nugget, sill, range). ``None`` -> fit.
    nugget: Optional[float] = None
    sill: Optional[float] = None
    range_: Optional[float] = None

    #: Also pull the field toward an ordinary-kriging interpolation of the point
    #: data, weighted by the inverse kriging variance.
    use_kriging_prior: bool = True


@dataclass
class OutputConfig:
    write_rasters: bool = True
    write_plots: bool = True
    write_metrics: bool = True
    raster_dtype: str = "float32"
    nodata: float = -9999.0


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    variogram: VariogramConfig = field(default_factory=VariogramConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #

_SECTIONS = {
    "paths": PathsConfig,
    "domain": DomainConfig,
    "physics": PhysicsConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "variogram": VariogramConfig,
    "output": OutputConfig,
}


def _build(cls, data: Optional[Dict[str, Any]]):
    """Instantiate ``cls`` from ``data``, rejecting unknown keys loudly."""
    if not data:
        return cls()
    valid = {f.name for f in dataclasses.fields(cls)}
    # ``range`` is a Python builtin so the dataclass field is ``range_``; accept
    # the natural spelling in YAML.
    data = dict(data)
    if cls is VariogramConfig and "range" in data:
        data["range_"] = data.pop("range")
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | pathlib.Path) -> Config:
    """Load a :class:`Config` from a YAML file.

    Paths inside the ``paths`` section are resolved relative to the config
    file's directory, so a config can be moved around with its data.
    """
    path = pathlib.Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")

    kwargs: Dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        if name == "train":
            sect = dict(raw.get(name) or {})
            weights = sect.pop("weights", None)
            cfg = _build(TrainConfig, sect)
            cfg.weights = _build(LossWeights, weights)
            kwargs[name] = cfg
        else:
            kwargs[name] = _build(cls, raw.get(name))

    cfg = Config(**kwargs)
    _resolve_paths(cfg, path.parent)
    return cfg


def _resolve_paths(cfg: Config, base: pathlib.Path) -> None:
    """Make relative path entries absolute w.r.t. the config directory."""

    def fix(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        p = pathlib.Path(value)
        return str(p if p.is_absolute() else (base / p).resolve())

    p = cfg.paths
    for name in (
        "workdir", "head_obs", "gauge_obs", "river_polygon", "river_centerline",
        "faults", "dtm", "prop_obs", "domain", "boundary", "recharge",
    ):
        setattr(p, name, fix(getattr(p, name)))
    p.layer_bottoms = [fix(v) for v in p.layer_bottoms]

    if cfg.physics.n_layers == 0:
        cfg.physics.n_layers = max(1, len(p.layer_bottoms))
