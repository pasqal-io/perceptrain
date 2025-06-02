from __future__ import annotations

from typing import Callable

import nevergrad as ng
import torch
import torch.nn as nn

from perceptrain.models import PINN

from ..types import LossFunction, TBatch

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


class GradWeightedLoss:
    def __init__(
        self,
        batch: dict[str, torch.Tensor],
        unweighted_loss_function: LossFunction,
        optimizer: torch.optim.Optimizer | ng.optimization.Optimizer,
        gradient_weights: dict[str, float | torch.Tensor],
        fixed_metric: str,
        alpha: float = 0.9,
    ):
        self.metric_names = batch.keys()
        self.gradient_weights = gradient_weights
        self.gradients: dict[str, dict[str, torch.Tensor]] = {key: {} for key in self.metric_names}
        self.unweighted_loss_function = unweighted_loss_function
        self.optimizer = optimizer
        self.fixed_metric = fixed_metric
        self.alpha = alpha

    def _update_metrics_gradients(
        self,
        metrics: dict[str, torch.Tensor],
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
        self, metrics: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fixed_grad = self.gradients[self.fixed_metric]
        fixed_dthetas = torch.cat(tuple(dlayer for dlayer in fixed_grad.values()))
        #
        # get max absolute gradient corresponding to residual loss term
        max_grad = fixed_dthetas.abs().max()

        # calculate weights for IC and BC terms
        for key, val in self.gradients.items():
            if key != self.fixed_metric:
                mean_grad = torch.cat(list(val.values())).abs().mean()
                self.gradient_weights[key] = (1.0 - self.alpha) * self.gradient_weights[
                    key
                ] + self.alpha * max_grad / mean_grad

        # calculate reweighted loss and metrics
        reweighted_metrics = {key: val * self.gradient_weights[key] for key, val in metrics.items()}
        reweighted_loss = torch.sum(torch.stack([val for val in reweighted_metrics.values()]))

        return reweighted_loss, reweighted_metrics

    def __call__(
        self, batch: tuple[dict[str, torch.Tensor],], model: nn.Module
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, unscaled_metrics = self.unweighted_loss_function(batch, model)
        self._update_metrics_gradients(unscaled_metrics, list(model.named_parameters()))
        loss, metrics = self._gradient_norm_weighting(unscaled_metrics)

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
