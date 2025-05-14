from __future__ import annotations

from .loss import CrossEntropyLoss, MSELoss, get_loss

# Modules to be automatically added to the perceptrain.loss namespace
__all__ = [
    "CrossEntropyLoss",
    "get_loss",
    "MSELoss",
]
