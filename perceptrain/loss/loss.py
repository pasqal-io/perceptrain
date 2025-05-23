from __future__ import annotations

from functools import singledispatch
from typing import Any, Callable

import torch
import torch.nn as nn

from perceptrain.types import Loss

# TODO If the only difference between losses is `criterion`, we can refactor
# this module
# TODO Return empty metrics unless the loss has more components (none in this module)


@singledispatch
def mse_loss(batch: Any, model: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the Mean Squared Error (MSE) loss between model.

        Basically a wrapper of perceptrain around `nn.MSELoss` of pytorch.

        The batch can be both a tuple of a single Tensor, or a tuple of two Tensors.
        In the fist case, the batch is assumed to contain only the model inputs, as it
        happens in unsupervised or semi-supervised learning.
        In the second case, the bach is assumed to contain model inputs and labels, as it
        happens in supervised learning.

    Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): tuple of tensors for the batch.
        model (nn.Module): The PyTorch model used for generating predictions.

    Returns:
        Tuple[torch.Tensor, dict[str, float]]:
            - loss (torch.Tensor): The computed MSE loss value.
            - metrics (dict[str, float]): A dictionary with the MSE loss value.
    """
    raise ValueError(f"Unsupported batch type f{batch}")


@mse_loss.register(tuple[torch.Tensor,])
def _(batch: tuple[torch.Tensor,], model: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
    criterion = nn.MSELoss()
    (inputs,) = batch
    outputs = model(inputs)
    loss = criterion(outputs)

    metrics = {"mse": loss}
    return loss, metrics


@mse_loss.register(tuple[torch.Tensor, torch.Tensor])
def _(
    batch: tuple[torch.Tensor, torch.Tensor], model: nn.Module
) -> tuple[torch.Tensor, dict[str, float]]:
    criterion = nn.MSELoss()
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    metrics = {"mse": loss}
    return loss, metrics


@singledispatch
def cross_entropy_loss(batch: Any, model: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the Cross Entropy loss between model predictions and targets.

    Basically a wrapper of perceptrain around `nn.CrossEntropyLoss` of pytorch.

    The batch can be both a tuple of a single Tensor, or a tuple of two Tensors.
    In the fist case, the batch is assumed to contain only the model inputs, as it
    happens in unsupervised or semi-supervised learning.
    In the second case, the bach is assumed to contain model inputs and labels, as it
    happens in supervised learning.

    Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): tuple of tensors for the batch.
        model (nn.Module): The PyTorch model used for generating predictions.

    Returns:
        Tuple[torch.Tensor, dict[str, float]]:
            - loss (torch.Tensor): The computed cross entropy value.
            - metrics (dict[str, float]): A dictionary with the cross entropy value.
    """
    raise ValueError(f"Unsupported batch type f{batch}")


@cross_entropy_loss.register(tuple[torch.Tensor,])
def _(batch: tuple[torch.Tensor,], model: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    (inputs,) = batch
    outputs = model(inputs)
    loss = criterion(outputs)

    metrics = {"cross_entopy": loss}
    return loss, metrics


@cross_entropy_loss.register(tuple[torch.Tensor, torch.Tensor])
def _(
    batch: tuple[torch.Tensor, torch.Tensor], model: nn.Module
) -> tuple[torch.Tensor, dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    metrics = {"cross_entropy": loss}
    return loss, metrics


def get_loss(loss: str | Loss | None) -> Callable:
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
