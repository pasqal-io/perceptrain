from __future__ import annotations

import random
from typing import Callable

import pytest
import torch
from torch import Tensor


@pytest.mark.parametrize("num_samples", [100, 0])
def test_r3_dataset_init(num_samples: int, make_mock_r3_dataset: Callable) -> None:
    dataset = make_mock_r3_dataset(num_samples)
    assert len(dataset.features) == num_samples


def test_r3_dataset_init_invalid_threshold(make_mock_r3_dataset: Callable) -> None:
    with pytest.raises(ValueError):
        dataset = make_mock_r3_dataset(release_threshold=-1.0)


@pytest.mark.parametrize("num_samples", [100, 0])
def test_r3_dataset_len(num_samples: int, make_mock_r3_dataset: Callable) -> None:
    dataset = make_mock_r3_dataset(num_samples)
    assert len(dataset) == num_samples


def test_r3_dataset_getitem(make_mock_r3_dataset: Callable) -> None:
    dataset = make_mock_r3_dataset()
    for _ in range(3):
        idx = random.randint(0, len(dataset) - 1)
        assert len(dataset[idx]) == 1
        assert isinstance(dataset[idx][0], Tensor)


def test_r3_dataset_release(make_mock_r3_dataset: Callable) -> None:
    """Case in which some samples are released."""
    num_samples = 3
    release_threshold = 1.0
    fitness_values = Tensor([0.1, 1.1, 2.1])

    dataset = make_mock_r3_dataset(num_samples, release_threshold)
    dataset._release(fitness_values)

    assert dataset.n_released == 1
    assert dataset.n_retained == 2


def test_r3_dataset_release_all(make_mock_r3_dataset: Callable) -> None:
    """Case in which all samples are released."""
    num_samples = 3
    release_threshold = 1.0
    fitness_values = Tensor([0.1, 0.3, 0.9])

    dataset = make_mock_r3_dataset(num_samples, release_threshold)
    dataset._release(fitness_values)

    assert dataset.n_released == 3
    assert dataset.n_retained == 0


def test_r3_dataset_release_none(make_mock_r3_dataset: Callable) -> None:
    """Case in which no samples are released."""
    num_samples = 3
    release_threshold = 1.0
    fitness_values = Tensor([1.1, 1.2, 2.1])

    dataset = make_mock_r3_dataset(num_samples, release_threshold)
    dataset._release(fitness_values)

    assert dataset.n_released == 0
    assert dataset.n_retained == 3


def test_r3_dataset_release_invalid(make_mock_r3_dataset: Callable) -> None:
    """Case in which the number of fitness values is not equal to the number of samples."""
    num_samples = 3
    fitness_values = Tensor([1.1, 1.2])

    dataset = make_mock_r3_dataset(num_samples)
    with pytest.raises(ValueError):
        dataset._release(fitness_values)


def test_r3_dataset_resample_before_release(make_mock_r3_dataset: Callable) -> None:
    """Case in which resampling is attempted before release."""
    dataset = make_mock_r3_dataset()
    resampled = dataset._resample()

    assert isinstance(resampled, Tensor)
    assert torch.numel(resampled) == 0


def test_r3_dataset_resample_after_release(make_mock_r3_dataset: Callable) -> None:
    """Case in which resampling is attempted after release."""
    num_samples = 3
    release_threshold = 1.0
    fitness_values = Tensor([0.1, 0.3, 1.1])

    dataset = make_mock_r3_dataset(num_samples, release_threshold)
    dataset._release(fitness_values)

    resampled = dataset._resample()

    assert isinstance(resampled, Tensor)
    assert torch.numel(resampled) == 2
