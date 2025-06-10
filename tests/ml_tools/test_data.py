from __future__ import annotations

import random
from typing import Callable

import pytest
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
