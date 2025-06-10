from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from pytest import fixture  # type: ignore
from torch import Tensor, tensor

from perceptrain.data import R3Dataset
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
