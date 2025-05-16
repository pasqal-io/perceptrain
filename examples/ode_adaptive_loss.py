# Aim: train a neural network to learn the solution of the following ODE.
# Uses the gradient-normalized loss technique in https://arxiv.org/abs/2001.04536
#
# df/dx = 5(4x^3 + x^2 - 2x - 0.5)
# f(0) = 0
# x \in [-1, 1]
#

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from perceptrain import TrainConfig, Trainer
from perceptrain.callbacks import Callback
from perceptrain.data import DictDataLoader, to_dataloader


class FFNN(torch.nn.Module):
    def __init__(self, layers: list[int], activation_function: Callable = torch.nn.GELU) -> None:
        super().__init__()
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


def mse(residuals: torch.Tensor) -> torch.Tensor:
    return torch.mean(residuals**2)


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
        model_parameters: list[tuple[str, torch.nn.parameter.Parameter]],
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
        features = {term: model(sample[0]) for term, sample in batch.items()}
        loss, metrics = self._compute_unweighted_metrics_and_loss(features)
        self._update_metrics_gradients(metrics, list(model.named_parameters()))
        loss, metrics = self._gradient_norm_weighting(metrics)

        return loss, metrics


def evaluate_ode(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    dudx = torch.autograd.grad(
        outputs=model(x),
        inputs=x,
        grad_outputs=torch.ones_like(x),
        create_graph=True,
        retain_graph=True,
    )[0]
    return dudx - 5 * (4 * x**3 + x**2 - 2 * x - 0.5)


def evaluate_bc(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    return model(x)  # - 0.0


def make_problem_dataloaders(
    model: torch.nn.Module, batch_sizes: dict[str, int]
) -> tuple[DataLoader, DataLoader]:
    x_interior = torch.rand(size=(100, 1), requires_grad=True) * 2 - 1  # points in [-1, 1]
    x_bc = torch.tensor([0.0])

    return (
        to_dataloader(
            x_interior,
            evaluate_ode(x_interior, model),
            batch_size=batch_sizes["ode"],
            infinite=True,
        ),
        to_dataloader(
            x_bc,
            evaluate_bc(x_bc, model),
            batch_size=batch_sizes["bc"],
            infinite=True,
        ),
    )


def print_gradient_weights(trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
    print(
        f"Epoch: {trainer.current_epoch};"
        f" ODE weight: {trainer.loss_fn.gradient_weights['ode']:8.4f},"
        f" BC weight: {trainer.loss_fn.gradient_weights['bc']:8.4f}"
    )


def print_metrics_and_loss(trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
    print(
        f"Epoch: {trainer.current_epoch};"
        f" Loss: {trainer.opt_result.loss:8.4f},"
        f" ODE loss: {trainer.opt_result.metrics['train_ode']:8.4f}"
        f" BC loss: {trainer.opt_result.metrics['train_bc']:8.4f}"
    )


def main():
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = FFNN(layers=[1, 10, 10, 10, 1])

    # dataloader(s)
    dl_ode, dl_bc = make_problem_dataloaders(model, batch_sizes={"ode": 10, "bc": 1})
    ddl = DictDataLoader(dataloaders={"ode": dl_ode, "bc": dl_bc})

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = GradWeightedLoss(
        batch=next(iter(ddl)),
        optimizer=optimizer,
        gradient_weights={"ode": 1.0, "bc": 1.0},
        fixed_metric="ode",
    )

    callback_weights = Callback(on="train_epoch_end", callback=print_gradient_weights)
    callback_metrics_loss = Callback(on="train_epoch_end", callback=print_metrics_and_loss)

    # config and trainer
    train_config = TrainConfig(max_iter=100, callbacks=[callback_weights, callback_metrics_loss])
    trainer = Trainer(model, optimizer, train_config, loss_fn=loss)

    # fit
    trainer.fit(train_dataloader=ddl)


if __name__ == "__main__":
    main()
