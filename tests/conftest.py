from __future__ import annotations

import torch.nn as nn
from pytest import fixture  # type: ignore
from torch import Tensor, tensor, complex64

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
def adamlbfgs_optimizer(Basic: BasicNetwork) -> AdamLBFGS:
    return AdamLBFGS(Basic.parameters(), switch_epoch=5)
