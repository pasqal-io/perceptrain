from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from multimethod import multimethod

from ..types import DictInDictOutModel, Model, TBatch, TensorInTensorOutModel

# TODO If the only difference between losses is `criterion`, we can refactor
# this module
# TODO Return empty metrics unless the loss has more components (none in this module)


@multimethod
def _compute_loss_and_metrics(
    batch: TBatch, model: Model, criterion: nn.Module
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
    raise ValueError(f"Unsupported combination of batch and model: f{type(batch)}, f{type(model)}.")


@_compute_loss_and_metrics.register
def _(
    batch: torch.Tensor, model: TensorInTensorOutModel, criterion: nn.Module
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Unsupervised loss with a tensor-based model."""
    outputs = model.forward(batch)
    loss = criterion(outputs)

    metrics: dict[str, torch.Tensor] = {}
    return loss, metrics


@_compute_loss_and_metrics.register
def _(
    batch: dict[str, torch.Tensor], model: DictInDictOutModel, criterion: nn.Module
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Unsupervised loss with a dict-based model."""
    outputs = model.forward(batch)
    metrics = {key: criterion(outputs[key]) for key in outputs.keys()}

    loss = sum([metrics[key] for key in outputs.keys()])
    return loss, metrics


@_compute_loss_and_metrics.register
def _(
    batch: tuple[torch.Tensor, torch.Tensor], model: TensorInTensorOutModel, criterion: nn.Module
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervised loss with a tensor-based model."""
    inputs, labels = batch
    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)

    metrics: dict[str, torch.Tensor] = {}
    return loss, metrics


# NOTE: the supervised, dict-based case is not implemented, simply because the only dict-based
# model supported by perceptrain is PINN, which is unsupervised by nature


def mse_loss(
    batch: TBatch, model: Model, criterion: nn.Module
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _compute_loss_and_metrics(batch, model, criterion=nn.MSELoss())  # type: ignore[no-any-return]


def cross_entropy_loss(
    batch: TBatch, model: Model, criterion: nn.Module
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _compute_loss_and_metrics(batch, model, criterion=nn.CrossEntropyLoss())  # type: ignore[no-any-return]


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
