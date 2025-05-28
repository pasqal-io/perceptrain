# Aim: train a neural network to learn the solution of the following ODE.
# Uses the gradient-normalized loss technique in https://arxiv.org/abs/2001.04536
#
# df/dx = 5(4x^3 + x^2 - 2x - 0.5)
# f(0) = 0
# x \in [-1, 1]
#

from __future__ import annotations

import argparse
import random

import nevergrad as ng
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from perceptrain import TrainConfig, Trainer
from perceptrain.callbacks import Callback
from perceptrain.data import DictDataLoader, to_dataloader
from perceptrain.loss import GradWeightedLoss, mse_loss
from perceptrain.models import FFNN, PINN
from perceptrain.parameters import num_parameters


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nograd",
        help="Run with a gradient-free optimizer.",
        action="store_true",
    )
    args = parser.parse_args()
    return args


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


def make_problem_dataloaders(batch_sizes: dict[str, int]) -> tuple[DataLoader, DataLoader]:
    x_interior = torch.rand(size=(100, 1), requires_grad=True) * 2 - 1  # points in [-1, 1]
    x_bc = torch.tensor([0.0])

    return (
        to_dataloader(
            x_interior,
            batch_size=batch_sizes["ode"],
            infinite=True,
        ),
        to_dataloader(
            x_bc,
            batch_size=batch_sizes["bc"],
            infinite=True,
        ),
    )


def print_gradient_weights(trainer: Trainer, config: TrainConfig, writer: Any) -> None:
    print(
        f"Epoch: {trainer.current_epoch};"
        f" ODE weight: {trainer.loss_fn.gradient_weights['ode']:8.4f},"
        f" BC weight: {trainer.loss_fn.gradient_weights['bc']:8.4f}"
    )


def print_metrics_and_loss(trainer: Trainer, config: TrainConfig, writer: Any) -> None:
    print(
        f"Epoch: {trainer.current_epoch};"
        f" Loss: {trainer.opt_result.loss:8.4f},"
        f" ODE loss: {trainer.opt_result.metrics['train_ode']:8.4f}"
        f" BC loss: {trainer.opt_result.metrics['train_bc']:8.4f}"
    )


def main():
    SEED = 42
    MAX_ITER = 30_000

    cli_args = parse_arguments()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    nn = FFNN(layers=[1, 10, 10, 10, 1])
    equations = {
        "ode": evaluate_ode,
        "bc": evaluate_bc,
    }
    model = PINN(nn, equations)

    # dataloader(s)
    dl_ode, dl_bc = make_problem_dataloaders(batch_sizes={"ode": 10, "bc": 1})
    ddl = DictDataLoader(dataloaders={"ode": dl_ode, "bc": dl_bc})

    # optimizer and loss
    if cli_args.nograd:
        optimizer = ng.optimizers.NGOpt(parametrization=num_parameters(model), budget=MAX_ITER)
        Trainer.set_use_grad(False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    loss = GradWeightedLoss(
        batch=next(iter(ddl)),
        unweighted_loss_function=mse_loss,
        optimizer=optimizer,
        gradient_weights={"ode": 1.0, "bc": 1.0},
        fixed_metric="ode",
    )

    callback_weights = Callback(
        on="train_epoch_end", callback=print_gradient_weights, called_every=1000
    )
    callback_metrics_loss = Callback(
        on="train_epoch_end", callback=print_metrics_and_loss, called_every=1000
    )

    # config and trainer
    train_config = TrainConfig(
        max_iter=MAX_ITER, callbacks=[callback_weights, callback_metrics_loss]
    )
    trainer = Trainer(model, optimizer, train_config, loss_fn=loss)

    # fit
    trainer.fit(train_dataloader=ddl)


if __name__ == "__main__":
    main()
