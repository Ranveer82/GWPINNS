"""Phase 2 -- the inverse PINN architectures."""

from .base import BasePINN, LossWeights
from .baseline import BaselinePINN
from .cpinn import ConservativePINN
from .forcing import ForcingTerm
from .mixed import MixedPINN
from .networks import MLP, FaultConductance, FourierFeatures, LogConductivityNet
from .sampling import DomainSampler
from .scaling import Scaling
from .trainer import TrainConfig, TrainingHistory, build_model, train

__all__ = [
    "BasePINN",
    "BaselinePINN",
    "ConservativePINN",
    "DomainSampler",
    "FaultConductance",
    "ForcingTerm",
    "FourierFeatures",
    "LogConductivityNet",
    "LossWeights",
    "MLP",
    "MixedPINN",
    "Scaling",
    "TrainConfig",
    "TrainingHistory",
    "build_model",
    "train",
]
