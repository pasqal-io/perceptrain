from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F
from pytest import fixture
from torch import Tensor, nn

from perceptrain.loss.loss import (
    _compute_loss_and_metrics_based_on_model,
    _compute_loss_and_metrics_pinn,
    _compute_loss_and_metrics_standard,
    cross_entropy_loss,
    mse_loss,
)
from perceptrain.models import FFNN, PINN


@fixture(scope="module")
def mock_standard_batch_model_setup_classification(
    mock_ffnn: FFNN,
) -> tuple[tuple[Tensor, Tensor], FFNN]:
    n_classes = 3
    x = torch.randn(size=(5, mock_ffnn.layers[0]))
    y = torch.randint(0, n_classes, size=(5,))
    return (x, y), mock_ffnn


@fixture(scope="module")
def mock_standard_batch_model_setup_regression(
    mock_ffnn: FFNN,
) -> tuple[tuple[Tensor, Tensor], FFNN]:
    x = torch.randn(size=(5, mock_ffnn.layers[0]))
    y = torch.randn(size=(5, mock_ffnn.layers[-1]))
    return (x, y), mock_ffnn


@fixture(scope="module")
def mock_pinn_batch_model_setup(mock_pinn: PINN) -> tuple[dict[str, Tensor], PINN]:
    metrics = mock_pinn.equations.keys()
    num_in_features = next(mock_pinn.nn.children()).in_features
    # batch size can be left random 4 generality
    batch = {
        m: torch.randn(size=(random.randint(2, 10), num_in_features), requires_grad=True)
        for m in metrics
    }
    return batch, mock_pinn


@fixture(scope="module")
def exact_mse_standard(
    mock_standard_batch_model_setup_regression: tuple[tuple[Tensor, Tensor], FFNN],
) -> tuple[Tensor, dict]:
    batch, model = mock_standard_batch_model_setup_regression
    x, y = batch
    return ((model(x) - y) ** 2).mean(), {}


@fixture(scope="module")
def exact_mse_pinn(
    mock_pinn_batch_model_setup: tuple[dict[str, Tensor], PINN],
) -> tuple[Tensor, dict[str, Tensor]]:
    batch, model = mock_pinn_batch_model_setup
    residuals = model(batch)
    metrics = {}
    for key, res in residuals.items():
        metrics[key] = (residuals[key] ** 2).mean()
    loss = sum(metrics.values())
    return loss, metrics


@fixture(scope="module")
def exact_cross_entropy(
    mock_standard_batch_model_setup_classification: tuple[tuple[Tensor, Tensor], FFNN],
) -> tuple[Tensor, dict[str, Tensor]]:
    batch, model = mock_standard_batch_model_setup_classification
    x, y = batch
    # 1-hot encode y
    y_one_hot = F.one_hot(y, num_classes=model.layers[-1])
    probas = F.softmax(model(x), dim=1)
    loss = -torch.log(probas[y_one_hot == 1]).mean()
    metrics: dict[str, Tensor] = {}
    return loss, metrics


def test_compute_loss_and_metrics_standard(
    mock_standard_batch_model_setup_regression: tuple[tuple[Tensor, Tensor], FFNN],
) -> None:
    """Output types should be the expected ones."""
    batch, model = mock_standard_batch_model_setup_regression
    loss, metrics = _compute_loss_and_metrics_standard(batch, model, criterion=nn.MSELoss())
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)
    assert metrics == {}


def test_compute_loss_and_metrics_pinn(
    mock_pinn_batch_model_setup: tuple[dict[str, Tensor], PINN],
) -> None:
    """Output types should be the expected ones."""
    batch, model = mock_pinn_batch_model_setup
    loss, metrics = _compute_loss_and_metrics_pinn(batch, model, criterion=nn.MSELoss())
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)


@pytest.mark.parametrize(
    "setup", ["mock_standard_batch_model_setup_regression", "mock_pinn_batch_model_setup"]
)
def test_compute_loss_and_metrics_based_on_model(
    setup: str, request: pytest.FixtureRequest
) -> None:
    """Should work as expected."""
    batch, model = request.getfixturevalue(setup)
    _compute_loss_and_metrics_based_on_model(batch, model, criterion=nn.MSELoss())


def test_mse_loss_standard(
    mock_standard_batch_model_setup_regression: tuple[tuple[Tensor, Tensor], FFNN],
    exact_mse_standard: tuple[Tensor, dict],
) -> None:
    """Test MSE loss calculation for the standard I/O model."""
    batch, model = mock_standard_batch_model_setup_regression
    loss_exact, _ = exact_mse_standard

    loss_predicted, metrics_predicted = mse_loss(batch, model)

    assert metrics_predicted == {}
    assert torch.isclose(loss_exact, loss_predicted)


def test_mse_loss_pinn(
    mock_pinn_batch_model_setup: tuple[dict[str, Tensor], PINN], exact_mse_pinn: tuple[Tensor, dict]
) -> None:
    """Test MSE loss calculation for the PINN model."""
    batch, model = mock_pinn_batch_model_setup
    loss_exact, metrics_exact = exact_mse_pinn

    loss_predicted, metrics_predicted = mse_loss(batch, model)  # type: ignore[arg-type]

    for key in metrics_exact.keys():
        assert torch.isclose(metrics_exact[key], metrics_predicted[key])
    assert torch.isclose(loss_exact, loss_predicted)


def test_cross_entropy_loss(
    mock_standard_batch_model_setup_classification: tuple[tuple[Tensor, Tensor], FFNN],
    exact_cross_entropy: tuple[Tensor, dict],
) -> None:
    """Test cross-entropy loss calculation."""
    batch, model = mock_standard_batch_model_setup_classification
    loss_exact, _ = exact_cross_entropy

    loss_predicted, metrics_predicted = cross_entropy_loss(batch, model)

    assert metrics_predicted == {}
    assert torch.isclose(loss_exact, loss_predicted)
