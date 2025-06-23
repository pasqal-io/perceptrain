# Aim: train a neural network to learn the solution of the advection equation.
# Uses the R3 sampling technique in https://arxiv.org/abs/2001.04536.
#
# du/dt + \beta(du/dx) = 0
# u(t=0, x) = h(x)
# u(t, x=0) = u(t, x=2\pi)
# \Omega = [0, 1] X [0, 2\pi]
#

from __future__ import annotations

import argparse
import random

import nevergrad as ng
import numpy as np
import torch
from torch.utils.data import DataLoader

from perceptrain import TrainConfig, Trainer
from perceptrain.callbacks import PrintMetrics, R3Sampling
from perceptrain.data import (
    DictDataLoader,
    R3Dataset,
    to_dataloader,
)
from perceptrain.loss.loss import mse_loss
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


def interior_uniform(n: int = 1) -> torch.Tensor:
    """Random uniform distribution over [0, 1] X [0, 2*pi]."""
    ts = torch.rand(size=(n,))  # \in [0, 1]
    xs = torch.rand(size=(n,)) * 2 * torch.pi  # \in [0, 2\pi]
    coords = torch.stack((ts, xs), dim=1)
    coords.requires_grad = True
    return coords


def periodic_boundary_uniform(n: int = 1) -> torch.Tensor:
    ts = torch.rand(size=(n,))  # \in [0, 1]
    xs = torch.multinomial(torch.tensor([0.5, 0.5]), n, replacement=True) * 2 * torch.pi
    return torch.stack((ts, xs), dim=1)


def initial_uniform(n: int = 1) -> torch.Tensor:
    ts = torch.zeros(size=(n,))
    xs = torch.rand(size=(n,)) * 2 * torch.pi  # \in [0, 2\pi]
    return torch.stack((ts, xs), dim=1)


def evaluate_pde(x: torch.Tensor, model: torch.nn.Module, beta: float = 1.0) -> torch.Tensor:
    u = model(x)
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    dudt = grad_u[:, 0]
    dudx = grad_u[:, 1]
    return dudt + beta * dudx


def evaluate_periodic_bc(x: torch.Tensor, model: torch.nn.Module):
    x_shifted = torch.stack((x[:, 0], (x[:, 1] + 2 * torch.pi) % (4 * torch.pi)), dim=1)
    return model(x) - model(x_shifted)


def evaluate_initial(x: torch.Tensor, model: torch.nn.Module):
    return model(x) - torch.sin(x[:, 1])


def collate_fn(batch: list[tuple[torch.Tensor]]):
    """Returns a tensor of shape (batch_size, 2)."""
    return torch.stack([x[0] for x in batch], dim=0)


def main():
    BATCH_SIZE_BC = 2
    BATCH_SIZE_IC = 2
    BATCH_SIZE_INTERIOR = 20
    BETA = 10.0
    CALLBACK_R3_CALLED_EVERY = 1000
    CALLBACK_LOSS_METRICS_CALLED_EVERY = 1000
    LR = 0.001
    MAX_ITER = 10_000
    N_SAMPLES_BC = 10
    N_SAMPLES_IC = 20
    N_SAMPLES_INTERIOR = 100
    NN_LAYERS = [2, 50, 50, 50, 50, 1]
    SEED = 42
    THRESHOLD_RESIDUAL = 1.0

    cli_args = parse_arguments()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # dataset
    ds = R3Dataset(
        proba_dist=interior_uniform,
        n_samples=N_SAMPLES_INTERIOR,
        release_threshold=THRESHOLD_RESIDUAL,
    )

    # dataloader(s)
    dl_pde = DataLoader(ds, batch_size=BATCH_SIZE_INTERIOR)
    dl_bc = to_dataloader(
        periodic_boundary_uniform(N_SAMPLES_BC),
        batch_size=BATCH_SIZE_BC,
        infinite=True,
        collate_fn=collate_fn,
    )
    dl_ic = to_dataloader(
        initial_uniform(N_SAMPLES_IC),
        batch_size=BATCH_SIZE_IC,
        infinite=True,
        collate_fn=collate_fn,
    )
    ddl = DictDataLoader(dataloaders={"pde": dl_pde, "bc": dl_bc, "ic": dl_ic})

    # model
    nn = FFNN(layers=NN_LAYERS, activation_function=torch.nn.Tanh())
    equations = {
        "pde": evaluate_pde,
        "bc": evaluate_periodic_bc,
        "ic": evaluate_initial,
    }
    model = PINN(nn, equations)

    # optimizer and loss
    if cli_args.nograd:
        optimizer = ng.optimizers.NGOpt(parametrization=num_parameters(model), budget=MAX_ITER)
        Trainer.set_use_grad(False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss = mse_loss

    # callbacks
    def fitness_function(x: torch.Tensor, model: PINN) -> torch.Tensor:
        return torch.abs(evaluate_pde(x, model.nn, beta=BETA))

    callback_r3 = R3Sampling(
        initial_dataset=ds,
        fitness_function=fitness_function,
        verbose=True,
        called_every=CALLBACK_R3_CALLED_EVERY,
    )
    callback_metrics_loss = PrintMetrics(
        on="train_epoch_end",
        called_every=CALLBACK_LOSS_METRICS_CALLED_EVERY,
    )

    # config and trainer
    train_config = TrainConfig(max_iter=MAX_ITER, callbacks=[callback_r3, callback_metrics_loss])
    trainer = Trainer(model, optimizer, train_config, loss_fn=loss)

    # fit
    trainer.fit(train_dataloader=ddl)


if __name__ == "__main__":
    main()
