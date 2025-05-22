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
from typing import Callable

import nevergrad as ng
import numpy as np
import torch
from torch.utils.data import DataLoader

from perceptrain import TrainConfig, Trainer
from perceptrain.callbacks import Callback
from perceptrain.data import DictDataLoader, GenerativeFixedDataset, to_dataloader
from perceptrain.loss.loss import MSELoss
from perceptrain.models import FFNN, PINN
from perceptrain.parameters import num_parameters


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nograd",
        help="Run with a gradient-free optimizer.",
        action="store_true",
    )
    parser.add_argument(
        "--beta",
        help="Wave propagation speed.",
        default=10.0,
    )
    args = parser.parse_args()
    return args


class R3Sampling(Callback):
    def __init__(
        self,
        initial_dataset: GenerativeFixedDataset,
        fitness_function: Callable[[torch.Tensor, torch.nn.Module], torch.Tensor],
        threshold: float = 0.1,
        dataloader_key: str | None = None,
    ):
        """Note that only the first tensor in the dataset is considered, and it is assumed to be.

        the tensor of features.

        We pass the dataset, not the single tensors, because the object is more general, because
        map/iterable-style are chosen upstream and because we can use the init constructor of
        datasets.
        Assumes supervised learning (labels).
        """
        self.dataset = initial_dataset

        self.n_samples_total = len(initial_dataset)
        self.n_features = initial_dataset.features.size(1)
        self.threshold = threshold
        self.fitness_function = fitness_function
        self.dataloader_key = dataloader_key

        self.n_retained = 0
        super().__init__(on="train_epoch_start")

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> None:
        """Eventually make a new dataloader from a dataset.__init__() call."""
        # Compute fitness function on all samples
        fitnesses = self.fitness_function(self.dataset.features, trainer.model)

        # Retain
        retained = fitnesses > self.threshold
        self.n_retained = len(retained)

        # Resample
        resampled = self.dataset.proba_dist(self.n_samples_total - self.n_retained)

        # Release
        new_features = torch.where(retained, self.dataset.features, resampled)

        # Update the dataset
        self.dataset.features = new_features

        # Update dataloader of the trainer with the re-sampled dataset.
        # NOTE a bit of un ugly hack...
        if isinstance(trainer.dataloader, DataLoader):
            trainer.dataloader.dataset = self.dataset
        elif isinstance(trainer.dataloader, DictDataLoader):
            if self.dataloader_key is not None:
                trainer.dataloader.dataloaders[self.dataloader_key].dataset = self.dataset
            else:
                raise ValueError(
                    "Updating a dictdataloader is not possible,"
                    "unless the key of the dataloader to be updated is specified."
                )


def interior_uniform(n: int = 1) -> torch.Tensor:
    """Random uniform distribution over [0, 1] X [0, 2*pi]."""
    ts = torch.rand(size=(n,))  # \in [0, 1]
    xs = torch.rand(size=(n,)) * 2 * torch.pi  # \in [0, 2\pi]
    return torch.cat((ts, xs), dim=1)


def periodic_boundary_uniform(n: int = 1) -> torch.Tensor:
    ts = torch.rand(size=(n,))  # \in [0, 1]
    xs = torch.multinomial(torch.tensor([0.5, 0.5]), n, replacement=True) * 2 * torch.pi
    return torch.cat((ts, xs), dim=1)


def initial_uniform(n: int = 1) -> torch.Tensor:
    ts = torch.zeros(size=(n,))
    xs = torch.rand(size=(n,)) * 2 * torch.pi  # \in [0, 2\pi]
    return torch.cat((ts, xs), dim=1)


def evaluate_pde(x: torch.Tensor, model: torch.nn.Module, beta: float = 1.0) -> torch.Tensor:
    dudt = torch.autograd.grad(
        outputs=model(x),
        inputs=x[:, 0],
        grad_outputs=torch.ones_like(x),
        create_graph=True,
        retain_graph=True,
    )[0]
    dudx = torch.autograd.grad(
        outputs=model(x),
        inputs=x[:, 1],
        grad_outputs=torch.ones_like(x),
        create_graph=True,
        retain_graph=True,
    )[0]
    return dudt + beta * dudx


def evaluate_periodic_bc(x: torch.Tensor, model: torch.nn.Module):
    return model(x) - model(torch.cat((x[:, 0], x[:, 1] + 2 * torch.pi), dim=1))


def evaluate_initial(x: torch.Tensor, model: torch.nn.Module):
    return model(x) - torch.sin(x[:, 1])


def main():
    BATCH_SIZE_BC = 2
    BATCH_SIZE_IC = 2
    BATCH_SIZE_INTERIOR = 20
    LR = 0.01
    MAX_ITER = 10_000
    N_SAMPLES_BC = 10
    N_SAMPLES_IC = 20
    N_SAMPLES_INTERIOR = 100
    SEED = 42

    cli_args = parse_arguments()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # dataset
    ds = GenerativeFixedDataset(
        proba_dist=interior_uniform,
        n_samples=N_SAMPLES_INTERIOR,
    )

    # dataloader(s)
    dl_pde = DataLoader(ds, batch_size=BATCH_SIZE_INTERIOR)
    dl_bc = to_dataloader(
        periodic_boundary_uniform(N_SAMPLES_BC), batch_size=BATCH_SIZE_BC, infinite=True
    )
    dl_ic = to_dataloader(initial_uniform(N_SAMPLES_IC), batch_size=BATCH_SIZE_IC, infinite=True)
    ddl = DictDataLoader(dataloaders={"pde": dl_pde, "bc": dl_bc, "ic": dl_ic})

    # model
    nn = FFNN(layers=[2, 20, 20, 20, 1])
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

    loss = MSELoss()


if __name__ == "__main__":
    main()
