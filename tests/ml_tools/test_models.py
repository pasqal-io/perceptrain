from __future__ import annotations

from typing import Callable, Sequence

import pytest
import torch
import torch.nn as nn
from pytest import fixture
from torch import Tensor

from perceptrain.models import FFNN, PINN


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


def count_num_ffnn_parameters(layers: Sequence[int]) -> int:
    count = 0
    for i in range(0, len(layers) - 1):
        count += layers[i + 1] * (layers[i] + 1)
    return count


def test_ffnn_init(mock_ffnn: FFNN) -> None:
    """The number of parameters should match the expected count."""
    num_params_expected = count_num_ffnn_parameters(mock_ffnn.layers)
    assert num_params_expected == sum(p.numel() for p in mock_ffnn.parameters() if p.requires_grad)


def test_ffnn_init_too_few_layers() -> None:
    """Should raise if the number of layers is less than 2."""
    with pytest.raises(ValueError):
        FFNN(layers=[10])


def test_ffnn_forward(mock_ffnn: FFNN) -> None:
    """The output shape should match the number of output neurons."""
    n_batch, n_in, n_out = 10, mock_ffnn.layers[0], mock_ffnn.layers[-1]
    x_out = mock_ffnn(torch.randn(size=(n_batch, n_in)))
    assert x_out.size() == (n_batch, n_out)


def test_ffnn_forward_wrong_input_size(mock_ffnn: FFNN) -> None:
    """Should raise if input size does not match the number of input neurons."""
    n_batch, n_in = 10, mock_ffnn.layers[0]
    with pytest.raises(ValueError):
        assert mock_ffnn(torch.randn(size=(n_batch, n_in + 1)))


def test_pinn_init(mock_pinn: PINN) -> None:
    """The PINN and the underlying network should have the same parameters."""
    assert set(mock_pinn.parameters()) == set(mock_pinn.nn.parameters())


def test_pinn_forward(mock_pinn: PINN) -> None:
    """Forward output and input should have the same keys."""
    n_batch, n_in = 10, 2
    x_in = {
        "pde": torch.randn(size=(n_batch, n_in), requires_grad=True),
        "bc1": torch.randn(size=(n_batch, n_in)),
        "bc2": torch.randn(size=(n_batch, n_in), requires_grad=True),
    }
    x_out = mock_pinn(x_in)
    assert set(x_in.keys()) == set(x_out.keys())


def test_pinn_wrong_input_keys(mock_pinn: PINN) -> None:
    """Should raise if input keys are not the same as the equation keys."""
    n_batch, n_in = 10, 2
    x_in = {
        "pde": torch.randn(size=(n_batch, n_in)),
        "boundary_cond_1": torch.randn(size=(n_batch, n_in)),
        "bc2": torch.randn(size=(n_batch, n_in)),
    }
    with pytest.raises(ValueError):
        mock_pinn(x_in)
