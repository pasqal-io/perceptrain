# Aim: train a neural network to learn the solution of the following ODE
#
# df/dx = 5(4x^3 + x^2 - 2x - 0.5)
# f(0) = 0
#

from __future__ import annotations

from typing import Callable, Iterator

import torch


class FFNN(torch.nn.Module):
    def __init__(self, layers: list[int], activation_function: Callable = torch.nn.GELU) -> None:
        if len(layers) < 2:
            raise ValueError("You must specify at least one input and one output layer.")

        self.layers = layers
        self.activation_function = activation_function

        sequence = []
        for n_i, n_o in zip(self.layers[:-2], self.layers[1:-1]):
            sequence.append(torch.nn.Linear(n_i, n_o))
            sequence.append(self.activation_function())

        sequence.append(torch.nn.Linear(self.layers[-2], self.layers[-1]))
        self.nn = torch.nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


def mse(batch_values: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(batch_values**2, dim=1), dim=0)


class GradWeightedLoss:
    def __init__(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        gradient_weights: dict[str, float | torch.Tensor],
        fixed_metric: str,
        alpha: float = 0.9,
    ):
        self.metric_names = batch.keys()
        self.gradient_weights = gradient_weights
        self.gradients = {key: {} for key in self.metric_names}
        self.optimizer = optimizer
        self.fixed_metric = fixed_metric
        self.alpha = alpha

    def _update_metrics_gradients(
        self,
        metrics: dict[str, torch.Tensor],
        model_parameters: Iterator[tuple[str, torch.nn.parameter.Parameter]],
    ) -> None:
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

    @staticmethod
    def _compute_unweighted_metrics_and_loss(features: dict[str, torch.Tensor]):
        metrics = {name: mse(feature) for name, feature in features.items()}
        loss = sum(metrics.values())
        return loss, metrics

    def __call__(
        self, model: torch.nn.Module, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = self._compute_unweighted_metrics_and_loss(model(batch))
        self._update_metrics_gradients(metrics, model.named_parameters())
        loss, metrics = self._gradient_norm_weighting(metrics)

        return loss, metrics


def main():
    def ode_rhs(x: torch.Tensor) -> torch.Tensor:
        return 5 * (4 * x**3 + x**2 - 2 * x - 0.5)

    model = FFNN(layers=[1, 10, 10, 10, 1])

    def ode_lhs(x: torch.Tensor) -> torch.Tensor:
        grad = torch.autograd.grad(
            outputs=model(x),
            inputs=x,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad


if __name__ == "__main__":
    main()
