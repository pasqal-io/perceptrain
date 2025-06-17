from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from pytest import fixture  # type: ignore
from torch import Tensor, tensor

from perceptrain.data import R3Dataset
from perceptrain.models import FFNN, PINN
from perceptrain.optimizers import AdamLBFGS


class BasicNetwork(nn.Module):
    def __init__(self, n_neurons: int = 5) -> None:
        super().__init__()
        network = [
            nn.Linear(1, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, 1),
        ]
        self.network = nn.Sequential(*network)
        self.n_neurons = n_neurons

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class BasicNetworkNoInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x = nn.Parameter(tensor([1.0]))
        self.scale = nn.Parameter(tensor([1.0]))

    def forward(self) -> Tensor:
        res = self.scale * (self.x - 2.0) ** 2
        return res


@fixture
def Basic() -> nn.Module:
    return BasicNetwork()


@fixture
def BasicNoInput() -> nn.Module:
    return BasicNetworkNoInput()


@fixture
def adamlbfgs_optimizer(Basic: nn.Module) -> AdamLBFGS:
    return AdamLBFGS(Basic.parameters(), switch_epoch=5)


@fixture
def make_mock_r3_dataset() -> Callable:
    """Factory function to create a mock R3Dataset."""

    def proba_dist(num_samples: int) -> Tensor:
        _dist = torch.distributions.Normal(0.0, 1.0)
        return _dist.sample((num_samples,))

    def _make_mock_r3_dataset(num_samples: int = 10, release_threshold: float = 1.0) -> R3Dataset:
        """Creates the mock R3Dataset, parametrized by the number of samples.

        and release threshold.
        """
        return R3Dataset(proba_dist, num_samples, release_threshold)

    return _make_mock_r3_dataset


@fixture
def mock_ffnn() -> FFNN:
    return FFNN(layers=[10, 2, 5, 3], activation_function=nn.Tanh())


@fixture
def mock_pde() -> Callable[[Tensor, nn.Module], Tensor]:
    alpha, beta = 0.1, 0.2

    def pde(x: Tensor, model: nn.Module) -> Tensor:
        u = model(x)
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]
        dudx = grad_u[:, 0]
        dudy = grad_u[:, 1]
        return alpha * dudx + beta * dudy - u**2

    return pde


@fixture
def mock_bc1() -> Callable[[Tensor, nn.Module], Tensor]:
    def bc(x: Tensor, model: nn.Module) -> Tensor:
        u = model(x)
        return u - 1.0

    return bc


@fixture
def mock_bc2() -> Callable[[Tensor, nn.Module], Tensor]:
    def bc(x: Tensor, model: nn.Module) -> Tensor:
        u = model(x)
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]
        dudx = grad_u[:, 0]
        return dudx - 1.0

    return bc


@fixture
def mock_pinn(
    mock_pde: Callable[[Tensor, nn.Module], Tensor],
    mock_bc1: Callable[[Tensor, nn.Module], Tensor],
    mock_bc2: Callable[[Tensor, nn.Module], Tensor],
) -> PINN:
    network = nn.Sequential(
        nn.Linear(2, 5),
        nn.Tanh(),
        nn.Linear(5, 6),
        nn.Tanh(),
        nn.Linear(6, 1),
    )
    return PINN(network, {"pde": mock_pde, "bc1": mock_bc1, "bc2": mock_bc2})
