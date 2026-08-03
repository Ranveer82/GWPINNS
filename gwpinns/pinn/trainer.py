"""Training loop shared by all three architectures.

Two stages, which is what makes an inverse problem of this stiffness converge:

1. **Adam** with cosine decay -- robust global exploration of the conductivity
   field from a random start.
2. **L-BFGS** -- a short second-order polish on a fixed set of collocation
   points, which sharpens the fault contrast that Adam leaves blurred.

An optional warm-up trains on the data term alone.  Starting with the physics
switched on from iteration zero lets the PDE residual of an untrained (nearly
constant) head field drive ``K`` straight into the bottom of its bounded range,
from which it does not recover.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..config import BenchmarkConfig
from ..benchmark.generate import BenchmarkData
from .base import BasePINN, LossWeights
from .cpinn import ConservativePINN
from .sampling import DomainSampler
from .scaling import Scaling

__all__ = ["TrainConfig", "TrainingHistory", "build_model", "train"]


@dataclass
class TrainConfig:
    """Optimisation settings."""

    architecture: str = "baseline"          # baseline | mixed | cpinn
    adam_iterations: int = 8000
    lbfgs_iterations: int = 300
    learning_rate: float = 2.0e-3
    min_learning_rate: float = 1.0e-5
    warmup_iterations: int = 400
    collocation_points: int = 3000
    boundary_points: int = 600
    interface_points: int = 800
    well_fraction: float = 0.15
    fault_fraction: float = 0.25
    resample_every: int = 20
    log_every: int = 250
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float64"
    weights: LossWeights = field(default_factory=LossWeights)

    # gradient-norm loss balancing (Wang, Teng & Perdikaris 2021)
    adaptive_weights: bool = True
    adapt_every: int = 100
    adapt_momentum: float = 0.9
    adapt_max: float = 1.0e4

    # physics curriculum: ramp the PDE terms in after the data-only warm-up
    ramp_iterations: int = 1000
    log_k_init: float = 0.0                 # prior/initial bulk log10 K, m/d
    prior_decay: bool = True                # fade the Tikhonov prior as physics ramps in
    prior_floor: float = 0.02               # residual prior strength at the end of training

    # architecture hyper-parameters
    width: int = 96
    depth: int = 5
    fourier_features: int = 64
    interface_mode: str = "conductance"     # cPINN only
    gamma_mode: str = "field"               # cPINN only
    residual_weighting: str = "source"

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["weights"] = self.weights.as_dict()
        return payload


@dataclass
class TrainingHistory:
    """Loss-component traces recorded during training."""

    iterations: list[int] = field(default_factory=list)
    records: list[dict[str, float]] = field(default_factory=list)
    wall_time_s: float = 0.0

    def append(self, iteration: int, report: dict[str, float]) -> None:
        self.iterations.append(iteration)
        self.records.append(dict(report))

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "records": self.records,
            "wall_time_s": self.wall_time_s,
        }


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


def build_model(
    cfg: BenchmarkConfig, scaling: Scaling, train_cfg: TrainConfig
) -> BasePINN:
    """Instantiate the architecture named by ``train_cfg.architecture``."""
    from .baseline import BaselinePINN
    from .mixed import MixedPINN

    common = dict(
        cfg=cfg,
        scaling=scaling,
        residual_weighting=train_cfg.residual_weighting,
        log_k_init=train_cfg.log_k_init,
    )
    name = train_cfg.architecture.lower()

    if name == "baseline":
        return BaselinePINN(
            **common,
            head_width=train_cfg.width,
            head_depth=train_cfg.depth,
            head_fourier=train_cfg.fourier_features,
            k_width=train_cfg.width,
            k_depth=train_cfg.depth,
            k_fourier=train_cfg.fourier_features,
        )
    if name == "mixed":
        return MixedPINN(
            **common,
            width=train_cfg.width,
            depth=train_cfg.depth,
            fourier=train_cfg.fourier_features,
            k_width=train_cfg.width,
            k_depth=train_cfg.depth,
            k_fourier=train_cfg.fourier_features,
        )
    if name == "cpinn":
        return ConservativePINN(
            **common,
            head_width=train_cfg.width,
            head_depth=train_cfg.depth,
            head_fourier=train_cfg.fourier_features,
            k_width=train_cfg.width,
            k_depth=train_cfg.depth,
            k_fourier=train_cfg.fourier_features,
            interface_mode=train_cfg.interface_mode,
            gamma_mode=train_cfg.gamma_mode,
        )
    raise ValueError(f"unknown architecture {train_cfg.architecture!r}")


def _observation_tensors(
    data: BenchmarkData, scaling: Scaling, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    obs = data.obs

    def col(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(values, dtype=np.float64).reshape(-1, 1), dtype=dtype, device=device
        )

    return {
        "x": col(obs.x),
        "y": col(obs.y),
        "z": col(obs.z),
        "t": col(obs.t),
        "head": col(scaling.encode_head(obs.head_obs)),
    }


def _draw_batches(
    model: BasePINN, sampler: DomainSampler, train_cfg: TrainConfig
) -> dict:
    """Draw one set of collocation, boundary and interface points."""
    is_cpinn = isinstance(model, ConservativePINN)

    if is_cpinn:
        half = max(train_cfg.collocation_points // 2, 1)
        collocation = {
            "west": sampler.collocation(
                half, train_cfg.well_fraction, train_cfg.fault_fraction, side=-1
            ),
            "east": sampler.collocation(
                half, train_cfg.well_fraction, train_cfg.fault_fraction, side=+1
            ),
        }
        neumann = sampler.neumann(train_cfg.boundary_points, side=-1)
        neumann += sampler.neumann(train_cfg.boundary_points, side=+1)
        interface = sampler.interface(train_cfg.interface_points)
    else:
        collocation = sampler.collocation(
            train_cfg.collocation_points,
            train_cfg.well_fraction,
            train_cfg.fault_fraction,
        )
        neumann = sampler.neumann(train_cfg.boundary_points)
        interface = None

    return {
        "collocation": collocation,
        "dirichlet": sampler.dirichlet(train_cfg.boundary_points),
        "neumann": neumann,
        "interface": interface,
    }


def _grad_norm(loss: torch.Tensor, params: list[torch.Tensor]) -> float:
    """L2 norm of ``d loss / d params``, used for loss balancing."""
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True, create_graph=False
    )
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().pow(2).sum())
    return math.sqrt(total)


def _rebalance(
    terms: dict[str, torch.Tensor],
    weights: LossWeights,
    params: list[torch.Tensor],
    train_cfg: TrainConfig,
) -> LossWeights:
    """Rescale loss weights so every term contributes a comparable gradient.

    The PDE residual of this problem starts ~10^5 times larger than the data
    misfit -- an artefact of the 83:1 horizontal-to-vertical aspect ratio, which
    puts a factor ``mu_z^2 ~ 7000`` on the vertical diffusion term.  Left alone,
    the optimiser spends its entire budget flattening vertical head gradients
    and drives K to the bottom of its bounded range.  Following Wang, Teng &
    Perdikaris (2021), each weight is moved towards
    ``|grad L_data| / |grad L_i|`` with momentum.
    """
    reference = _grad_norm(terms["data"], params)
    if reference <= 0.0 or not math.isfinite(reference):
        return weights

    momentum = train_cfg.adapt_momentum
    updated = LossWeights(**weights.as_dict())
    for name, value in terms.items():
        # ``data`` is the reference; the priors are deliberate regularisers whose
        # strength is a modelling choice, not something to be auto-tuned.
        if name in {"data", "k_prior", "k_smoothness"}:
            continue
        norm = _grad_norm(value, params)
        if norm <= 0.0 or not math.isfinite(norm):
            continue
        target = min(reference / norm, train_cfg.adapt_max)
        current = getattr(updated, name, 1.0)
        setattr(updated, name, momentum * current + (1.0 - momentum) * target)
    return updated


def train(
    data: BenchmarkData,
    train_cfg: TrainConfig,
    scaling: Scaling | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[BasePINN, TrainingHistory]:
    """Fit one architecture to one benchmark scenario."""
    torch.set_default_dtype(getattr(torch, train_cfg.dtype))
    _set_seed(train_cfg.seed)

    device = torch.device(train_cfg.device)
    dtype = getattr(torch, train_cfg.dtype)
    cfg = data.cfg

    if scaling is None:
        scaling = Scaling.from_observations(cfg, data.obs.head_obs)

    model = build_model(cfg, scaling, train_cfg).to(device=device, dtype=dtype)
    sampler = DomainSampler(cfg, device=device, seed=train_cfg.seed + 1, dtype=dtype)
    obs = _observation_tensors(data, scaling, device, dtype)
    history = TrainingHistory()

    weights = train_cfg.weights

    def _prior_factor(iteration: int) -> float:
        """Fade the Tikhonov prior out as the physics fades in.

        The prior exists to stop K collapsing to its lower bound before the head
        field is good enough to constrain it.  Once the physics is fully ramped
        in, leaving the prior at full strength simply pins K to the prior value
        -- the field stops being data-driven.  Decaying it to a small floor keeps
        the early stabilisation without buying it at the cost of the answer.
        """
        if not train_cfg.prior_decay:
            return 1.0
        start = train_cfg.warmup_iterations
        span = max(train_cfg.adam_iterations - start, 1)
        progress = min(max((iteration - start) / span, 0.0), 1.0)
        return float(1.0 + (train_cfg.prior_floor - 1.0) * progress)

    def _ramped(base: LossWeights, factor: float, prior: float = 1.0) -> LossWeights:
        """Scale the physics terms by ``factor`` and the prior by ``prior``."""
        scaled = LossWeights(**base.as_dict())
        for name in ("pde", "darcy", "neumann", "interface_flux", "interface_head"):
            setattr(scaled, name, getattr(base, name) * factor)
        scaled.k_prior = base.k_prior * prior
        return scaled

    warm_weights = _ramped(weights, 0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(train_cfg.adam_iterations, 1), eta_min=train_cfg.min_learning_rate
    )

    start = time.perf_counter()
    batches = _draw_batches(model, sampler, train_cfg)
    params = [p for p in model.parameters() if p.requires_grad]

    for iteration in range(train_cfg.adam_iterations):
        if train_cfg.resample_every and iteration % train_cfg.resample_every == 0:
            batches = _draw_batches(model, sampler, train_cfg)

        warming = iteration < train_cfg.warmup_iterations
        prior_factor = _prior_factor(iteration)
        if warming:
            active = warm_weights
        else:
            progress = (
                (iteration - train_cfg.warmup_iterations) / train_cfg.ramp_iterations
                if train_cfg.ramp_iterations > 0
                else 1.0
            )
            active = _ramped(weights, min(progress, 1.0), prior_factor)

        optimizer.zero_grad(set_to_none=True)
        terms = model.loss_terms(
            obs,
            batches["collocation"],
            dirichlet=batches["dirichlet"],
            neumann=batches["neumann"],
            interface=batches["interface"],
            weights=active,
        )

        if (
            train_cfg.adaptive_weights
            and not warming
            and train_cfg.adapt_every
            and iteration % train_cfg.adapt_every == 0
        ):
            weights = _rebalance(terms, weights, params, train_cfg)
            progress = (
                (iteration - train_cfg.warmup_iterations) / train_cfg.ramp_iterations
                if train_cfg.ramp_iterations > 0
                else 1.0
            )
            active = _ramped(weights, min(progress, 1.0), prior_factor)

        loss, report = model.combine(terms, active)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if not math.isfinite(report["total"]):
            raise RuntimeError(f"loss diverged at iteration {iteration}: {report}")

        if verbose and (iteration % train_cfg.log_every == 0 or iteration == train_cfg.adam_iterations - 1):
            parts = " ".join(f"{k}={v:.3e}" for k, v in report.items() if k != "total")
            print(
                f"  [{train_cfg.architecture:8s}] adam {iteration:6d} "
                f"total={report['total']:.4e} {parts} lr={scheduler.get_last_lr()[0]:.2e}"
            )
        if iteration % max(train_cfg.log_every // 5, 1) == 0:
            history.append(iteration, report)

    # --- L-BFGS polish on a fixed batch --------------------------------------- #
    if train_cfg.lbfgs_iterations > 0:
        weights = _ramped(weights, 1.0, _prior_factor(train_cfg.adam_iterations))
        batches = _draw_batches(model, sampler, train_cfg)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=train_cfg.lbfgs_iterations,
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )
        state = {"calls": 0, "report": {}}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss, report = model.total_loss(
                obs,
                batches["collocation"],
                dirichlet=batches["dirichlet"],
                neumann=batches["neumann"],
                interface=batches["interface"],
                weights=weights,
            )
            loss.backward()
            state["calls"] += 1
            state["report"] = report
            return loss

        try:
            lbfgs.step(closure)
        except RuntimeError as exc:                      # pragma: no cover
            if verbose:
                print(f"  [{train_cfg.architecture}] L-BFGS stopped early: {exc}")
        if state["report"]:
            history.append(train_cfg.adam_iterations, state["report"])
            if verbose:
                print(
                    f"  [{train_cfg.architecture:8s}] lbfgs {state['calls']:6d} "
                    f"total={state['report']['total']:.4e}"
                )

    history.wall_time_s = time.perf_counter() - start

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "model.pt")
        payload = train_cfg.as_dict()
        payload["final_weights"] = weights.as_dict()
        (output_dir / "train_config.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "history.json").write_text(json.dumps(history.to_dict(), indent=2))
        (output_dir / "scaling.json").write_text(json.dumps(scaling.summary(), indent=2))

    return model, history
