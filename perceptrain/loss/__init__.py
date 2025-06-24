from __future__ import annotations

from .loss import GradWeightedLoss, cross_entropy_loss, get_loss, mse_loss

# Modules to be automatically added to the perceptrain.loss namespace
__all__ = [
    "cross_entropy_loss",
    "get_loss",
    "mse_loss",
    "GradWeightedLoss",
]
