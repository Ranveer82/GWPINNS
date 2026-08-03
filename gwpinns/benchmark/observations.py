"""Extraction of the sparse monitoring-well dataset.

The PINNs are trained on *these arrays only*: head time series at a handful of
screened intervals, plus the known source/sink geometry.  The full head grid and
the true conductivity field are kept strictly for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BenchmarkConfig
from .modflow6 import ForwardSolution

__all__ = ["ObservationSet", "sample_observations"]


@dataclass
class ObservationSet:
    """Flattened ``(point, time)`` head observations.

    All arrays are 1-D and of equal length ``n_points * n_times``.
    """

    name: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    layer: np.ndarray
    t: np.ndarray
    head_true: np.ndarray
    head_obs: np.ndarray
    signed_distance: np.ndarray

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def n_points(self) -> int:
        return int(np.unique(self.name).size)

    def as_columns(self) -> dict[str, np.ndarray]:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "layer": self.layer,
            "time": self.t,
            "head_true": self.head_true,
            "head_obs": self.head_obs,
            "fault_distance": self.signed_distance,
        }


def sample_observations(
    cfg: BenchmarkConfig, solution: ForwardSolution
) -> ObservationSet:
    """Sample the forward solution at the monitoring network and add noise.

    Monitoring wells falling inside the fault zone (or within
    ``min_fault_offset`` of it) are dropped -- a real network would not have a
    piezometer screened in the damage zone, and it keeps the cPINN's domain
    decomposition unambiguous.
    """
    grid, obs_cfg = cfg.grid, cfg.observations
    rng = np.random.default_rng(obs_cfg.seed)
    z_centers = grid.z_centers()

    names: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    layers: list[int] = []
    ts: list[float] = []
    truth: list[float] = []
    dists: list[float] = []

    min_offset = 0.5 * cfg.fault.width + obs_cfg.min_fault_offset

    for well_name, wx, wy in cfg.obs_wells:
        dist = float(cfg.fault.signed_distance(wx, wy))
        if abs(dist) < min_offset:
            continue
        row, col = grid.locate(wx, wy)
        for layer in obs_cfg.layers:
            series = solution.heads[:, layer, row, col]
            for time, head in zip(solution.times, series):
                names.append(f"{well_name}_L{layer}")
                xs.append(wx)
                ys.append(wy)
                zs.append(float(z_centers[layer]))
                layers.append(int(layer))
                ts.append(float(time))
                truth.append(float(head))
                dists.append(dist)

    head_true = np.asarray(truth)
    noise = rng.normal(0.0, obs_cfg.noise_std, size=head_true.shape)

    return ObservationSet(
        name=np.asarray(names),
        x=np.asarray(xs),
        y=np.asarray(ys),
        z=np.asarray(zs),
        layer=np.asarray(layers, dtype=int),
        t=np.asarray(ts),
        head_true=head_true,
        head_obs=head_true + noise,
        signed_distance=np.asarray(dists),
    )
