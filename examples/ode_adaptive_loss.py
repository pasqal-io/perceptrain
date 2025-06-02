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
from perceptrain.callbacks import Callback, LivePlotMetrics, PrintMetrics
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


def main():
    BATCH_SIZE_ODE = 10
    BATCH_SIZE_BC = 1
    CALLBACK_WEIGHTS_CALLED_EVERY = 1000
    CALLBACK_LOSS_CALLED_EVERY = 1000
    INITIAL_GRAD_WEIGHT_ODE = 1.0
    INITIAL_GRAD_WEIGHT_BC = 1.0
    LR = 0.01
    MAX_ITER = 30_000
    NN_LAYERS = (1, 10, 10, 10, 1)
    SEED = 42

    cli_args = parse_arguments()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    nn = FFNN(layers=NN_LAYERS)
    equations = {
        "ode": evaluate_ode,
        "bc": evaluate_bc,
    }
    model = PINN(nn, equations)

    # dataloader(s)
    dl_ode, dl_bc = make_problem_dataloaders(
        batch_sizes={"ode": BATCH_SIZE_ODE, "bc": BATCH_SIZE_BC}
    )
    ddl = DictDataLoader(dataloaders={"ode": dl_ode, "bc": dl_bc})

    # optimizer and loss
    if cli_args.nograd:
        optimizer = ng.optimizers.NGOpt(parametrization=num_parameters(model), budget=MAX_ITER)
        Trainer.set_use_grad(False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss = GradWeightedLoss(
        batch=next(iter(ddl)),
        unweighted_loss_function=mse_loss,
        optimizer=optimizer,
        gradient_weights={"ode": INITIAL_GRAD_WEIGHT_ODE, "bc": INITIAL_GRAD_WEIGHT_BC},
        fixed_metric="ode",
    )

    callback_weights = Callback(
        on="train_epoch_end",
        callback=print_gradient_weights,
        called_every=CALLBACK_WEIGHTS_CALLED_EVERY,
    )
    callback_metrics_loss = PrintMetrics(
        on="train_epoch_end",
        called_every=CALLBACK_LOSS_CALLED_EVERY,
    )
    callback_live_loss = LivePlotMetrics(
        on="train_epoch_end",
        called_every=CALLBACK_LOSS_CALLED_EVERY,
        groups={"training": ["train_loss", "train_ode", "train_bc"]},
    )
    # config and trainer
    train_config = TrainConfig(
        max_iter=MAX_ITER, callbacks=[callback_weights, callback_metrics_loss, callback_live_loss]
    )
    trainer = Trainer(model, optimizer, train_config, loss_fn=loss)

    # fit
    trainer.fit(train_dataloader=ddl)


if __name__ == "__main__":
    main()
