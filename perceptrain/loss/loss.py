from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from perceptrain.models import PINN

from ..types import TBatch

# TODO If the only difference between losses is `criterion`, we can refactor
# this module
# TODO Return empty metrics unless the loss has more components (none in this module)


def _compute_loss_based_on_model(
    batch: TBatch,
    model: nn.Module,
    criterion: nn.Module,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes the Mean Squared Error (MSE) loss between model.

        Basically a wrapper of perceptrain around `nn.MSELoss` of pytorch.

        The batch can be both a tuple of a single Tensor, or a tuple of two Tensors.
        In the fist case, the batch is assumed to contain only the model inputs, as it
        happens in unsupervised or semi-supervised learning.
        In the second case, the bach is assumed to contain model inputs and labels, as it
        happens in supervised learning.

    Args:
        batch:  tuple of tensors for the batch.
        model (nn.Module): The PyTorch model used for generating predictions.

    Returns:
        Tuple[torch.Tensor, dict[str, float]]:
            - loss (torch.Tensor): The computed MSE loss value.
            - metrics (dict[str, float]): A dictionary with the MSE loss value.
    """
    if isinstance(model, PINN):
        inputs = {key: value[0] for key, value in batch.items()}  # type: ignore[attr-defined]
        outputs = model(inputs)
        metrics = {
            key: criterion(outputs[key], torch.zeros_like(outputs[key])) for key in outputs.keys()
        }
        loss = sum([metrics[key] for key in outputs.keys()])
    else:
        inputs, labels = batch
        predictions = model(inputs)
        metrics: dict[str, torch.Tensor] = {}  # type: ignore[no-redef]
        loss = criterion(predictions, labels)

    return loss, metrics


def mse_loss(batch: TBatch, model: nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _compute_loss_based_on_model(batch, model, criterion=nn.MSELoss())  # type: ignore[no-any-return]


def cross_entropy_loss(
    batch: TBatch, model: nn.Module
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predictions, labels = model(batch)
    metrics: dict[str, torch.Tensor] = {}
    loss = nn.CrossEntropyLoss(predictions, labels)

    return loss, metrics


def get_loss(loss: str | Callable | None) -> Callable:
    """
    Returns the appropriate loss function based on the input argument.

    Args:
        loss_fn (str | Callable | None): The loss function to use.
            - If `loss_fn` is a callable, it will be returned directly.
            - If `loss_fn` is a string, it should be one of:
                - "mse": Returns the MSE loss function.
                - "cross_entropy": Returns the Cross Entropy function.
            - If `loss_fn` is `None`, the default MSE loss function will be returned.

    Returns:
        Callable: The corresponding loss function.

    Raises:
        ValueError: If `loss_fn` is a string but not a supported loss function name.
    """
    if callable(loss):
        return loss
    elif isinstance(loss, str):
        if loss == "mse":
            return mse_loss
        elif loss == "cross_entropy":
            return cross_entropy_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
    else:
        # default case
        return mse_loss
