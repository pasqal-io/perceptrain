from __future__ import annotations

from typing import Callable

import nevergrad as ng
import torch
import torch.nn as nn
from torch import Tensor

from perceptrain.models import PINN

from ..types import LossFunction, TBatch


def _compute_loss_and_metrics_standard(
    batch: tuple[Tensor, Tensor],
    model: nn.Module,
    criterion: nn.Module,
) -> tuple[Tensor, dict[str, Tensor]]:
    inputs, labels = batch
    predictions = model(inputs)
    metrics: dict[str, Tensor] = {}  # type: ignore[no-redef]
    loss = criterion(predictions, labels)
    return loss, metrics


def _compute_loss_and_metrics_pinn(
    batch: dict[str, Tensor],
    model: PINN,
    criterion: nn.Module,
) -> tuple[Tensor, dict[str, Tensor]]:
    inputs = {key: value for key, value in batch.items()}  # type: ignore[attr-defined]
    outputs = model(inputs)
    metrics = {
        key: criterion(outputs[key], torch.zeros_like(outputs[key])) for key in outputs.keys()
    }
    loss = sum([metrics[key] for key in outputs.keys()])
    return loss, metrics


def _compute_loss_and_metrics_based_on_model(
    batch: TBatch,
    model: nn.Module,
    criterion: nn.Module,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Computes loss and metrics based on the type of model.

    The model can be either:
    - PINN: loss and metrics are computed applying the criterion to the outputs (equation residuals)
    - Other models: the loss is computed in a supervised manner, considering both model outputs and
        labels. The metrics are an empty dictionary.

    Args:
        batch (dict[str, Tensor]): Data batch. The structure of the batch depends on the model
            type.
        model (nn.Module): The PyTorch model used for generating predictions.
        criterion (nn.Module): The loss criterion used for computing the loss (e.g., nn.MSELoss()).

    Returns:
        Tuple[Tensor, dict[str, float]]:
            - loss (Tensor): The computed loss value.
            - metrics (dict[str, float]): A dictionary of metrics (loss components).
    """
    if isinstance(model, PINN):
        return _compute_loss_and_metrics_pinn(batch, model, criterion)  # type: ignore[arg-type]
    else:
        return _compute_loss_and_metrics_standard(batch, model, criterion)  # type: ignore[arg-type]


def mse_loss(batch: TBatch, model: nn.Module) -> tuple[Tensor, dict[str, Tensor]]:
    """Mean Squared Error Loss.

    Args:
        batch (TBatch): The input batch.
        model (nn.Module): The model to compute the loss for.

    Returns:
        Tuple[Tensor, dict[str, float]]:
            - loss (Tensor): The computed loss value.
            - metrics (dict[str, float]): A dictionary of metrics (loss components).
    """
    return _compute_loss_and_metrics_based_on_model(batch, model, criterion=nn.MSELoss())  # type: ignore[no-any-return]


def cross_entropy_loss(batch: TBatch, model: nn.Module) -> tuple[Tensor, dict[str, Tensor]]:
    """Cross Entropy Loss.

    Args:
        batch (TBatch): The input batch.
        model (nn.Module): The model to compute the loss for.

    Returns:
        Tuple[Tensor, dict[str, float]]:
            - loss (Tensor): The computed loss value.
            - metrics (dict[str, float]): Empty dictionary. Not relevant for this loss function.
    """
    inputs, labels = batch
    predictions = model(inputs)
    metrics: dict[str, Tensor] = {}
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predictions, labels)

    return loss, metrics


class GradWeightedLoss:
    def __init__(
        self,
        batch: dict[str, Tensor],
        unweighted_loss_function: LossFunction,
        optimizer: torch.optim.Optimizer | ng.optimization.Optimizer,
        metric_weights: dict[str, float | Tensor],
        fixed_metric: str,
        alpha: float = 0.9,
    ):
        """Loss function with gradient weighting for PINN training.

        Implements the learning rate annealing algorithm in [this article](https://arxiv.org/abs/2001.04536).

        Args:
            batch (dict[str, Tensor]): Batch of data.
            unweighted_loss_function (LossFunction): Loss function applied before weighting.
            optimizer (torch.optim.Optimizer | ng.optimization.Optimizer): torch or nevergrad
                optimizer for gradient or gradient-free optimization.
            metric_weights (dict[str, float | Tensor]): Initial metric weights.
            fixed_metric (str): Metric whose weight is not updated and whose gradient determines the
                weights of the other metrics.
            alpha (float, optional): Scaling factor. Corresponds to the inertia of the weights to
                updates. Defaults to 0.9.
        """
        self.metric_names = batch.keys()
        self.metric_weights = metric_weights
        self.gradients: dict[str, dict[str, Tensor]] = {key: {} for key in self.metric_names}
        self.unweighted_loss_function = unweighted_loss_function
        self.optimizer = optimizer
        self.fixed_metric = fixed_metric
        self.alpha = alpha

    def _update_metrics_gradients(
        self,
        metrics: dict[str, Tensor],
        model_parameters: list[tuple[str, torch.nn.parameter.Parameter]],
    ) -> None:
        if isinstance(self.optimizer, torch.optim.Optimizer):
            self.optimizer.zero_grad()
        for key, metric in metrics.items():
            metric.backward(retain_graph=True)
            self.gradients[key] = {
                name: torch.clone(param.grad.flatten())
                for name, param in model_parameters
                if param.grad is not None
            }

    def _gradient_norm_weighting(
        self, metrics: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        fixed_grad = self.gradients[self.fixed_metric]
        fixed_dthetas = torch.cat(tuple(dlayer for dlayer in fixed_grad.values()))
        #
        # get max absolute gradient corresponding to residual loss term
        max_grad = fixed_dthetas.abs().max()

        # calculate weights for IC and BC terms
        for key, val in self.gradients.items():
            if key != self.fixed_metric:
                mean_grad = torch.cat(list(val.values())).abs().mean()
                self.metric_weights[key] = (1.0 - self.alpha) * self.metric_weights[
                    key
                ] + self.alpha * max_grad / mean_grad

        # calculate reweighted loss and metrics
        reweighted_metrics = {key: val * self.metric_weights[key] for key, val in metrics.items()}
        reweighted_loss = torch.sum(torch.stack([val for val in reweighted_metrics.values()]))

        return reweighted_loss, reweighted_metrics

    def __call__(
        self, batch: tuple[dict[str, Tensor],], model: nn.Module
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _, unscaled_metrics = self.unweighted_loss_function(batch, model)
        self._update_metrics_gradients(unscaled_metrics, list(model.named_parameters()))
        loss, metrics = self._gradient_norm_weighting(unscaled_metrics)

        return loss, metrics


def get_loss(loss_fn: str | Callable | None) -> Callable:
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
    if callable(loss_fn):
        return loss_fn
    elif isinstance(loss_fn, str):
        if loss_fn == "mse":
            return mse_loss
        elif loss_fn == "cross_entropy":
            return cross_entropy_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")
    else:
        # default case
        return mse_loss
