from __future__ import annotations

from typing import Sequence

import pytest
import torch
import torch.nn as nn
from pytest import fixture

from perceptrain.models import FFNN


@fixture
def mock_ffnn() -> FFNN:
    return FFNN(layers=[10, 2, 5, 3], activation_function=nn.Tanh())


def count_num_ffnn_parameters(layers: Sequence[int]) -> int:
    count = 0
    for i in range(0, len(layers) - 1):
        count += layers[i + 1] * (layers[i] + 1)
    return count


def test_ffnn_init(mock_ffnn: FFNN) -> None:
    num_params_expected = count_num_ffnn_parameters(mock_ffnn.layers)
    assert num_params_expected == sum(p.numel() for p in mock_ffnn.parameters() if p.requires_grad)


def test_ffnn_init_too_few_layers() -> None:
    with pytest.raises(ValueError):
        FFNN(layers=[10])


def test_ffnn_forward(mock_ffnn: FFNN) -> None:
    n_batch, n_in, n_out = 10, mock_ffnn.layers[0], mock_ffnn.layers[-1]
    x_out = mock_ffnn(torch.randn(size=(n_batch, n_in)))
    assert x_out.size() == (n_batch, n_out)


def test_ffnn_forward_wrong_input_size(mock_ffnn: FFNN) -> None:
    n_batch, n_in = 10, mock_ffnn.layers[0]
    with pytest.raises(ValueError):
        assert mock_ffnn(torch.randn(size=(n_batch, n_in + 1)))
