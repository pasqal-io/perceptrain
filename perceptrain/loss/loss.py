from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from perceptrain.types import Loss


def mse_loss(
    model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the Mean Squared Error (MSE) loss between model predictions and targets.

        Basically a wrapper of perceptrain around `nn.MSELoss` of pytorch.

    Args:
        model (nn.Module): The PyTorch model used for generating predictions.
        batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing:
            - inputs (torch.Tensor): The input data.
            - targets (torch.Tensor): The ground truth labels.

    Returns:
        Tuple[torch.Tensor, dict[str, float]]:
            - loss (torch.Tensor): The computed MSE loss value.
            - metrics (dict[str, float]): A dictionary with the MSE loss value.
    """
    criterion = nn.MSELoss()
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # TODO consider returning empty metric. Metrics are for components of the loss.
    metrics = {"mse": loss}
    return loss, metrics


def cross_entropy_loss(
    model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the Cross Entropy loss between model predictions and targets.

    Basically a wrapper of perceptrain around `nn.CrossEntropyLoss` of pytorch.

    Args:
        model (nn.Module): The PyTorch model used for generating predictions.
        batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing:
            - inputs (torch.Tensor): The input data.
            - targets (torch.Tensor): The ground truth labels.

    Returns:
        Tuple[torch.Tensor, dict[str, float]]:
            - loss (torch.Tensor): The computed Cross Entropy loss value.
            - metrics (dict[str, float]): A dictionary with the Cross Entropy loss value.
    """
    criterion = nn.CrossEntropyLoss()
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # TODO consider returning empty metric. Metrics are for components of the loss.
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
