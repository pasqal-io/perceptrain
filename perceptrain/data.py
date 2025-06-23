from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Any, Callable, Iterator

import torch
from nevergrad.optimization.base import Optimizer as NGOptimizer
from torch import Tensor
from torch import device as torch_device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset


@dataclass
class OptimizeResult:
    """OptimizeResult stores many optimization intermediate values.

    We store at a current iteration,
    the model, optimizer, loss values, metrics. An extra dict
    can be used for saving other information to be used for callbacks.
    """

    iteration: int
    """Current iteration number."""
    model: Module
    """Model at iteration."""
    optimizer: Optimizer | NGOptimizer
    """Optimizer at iteration."""
    loss: Tensor | float | None = None
    """Loss value."""
    metrics: dict = field(default_factory=lambda: dict())
    """Metrics that can be saved during training."""
    extra: dict = field(default_factory=lambda: dict())
    """Extra dict for saving anything else to be used in callbacks."""
    rank: int = 0
    """Rank of the process for which this result was generated."""
    device: str | None = "cpu"
    """Device on which this result for calculated."""


@dataclass
class DictDataLoader:
    """This class only holds a dictionary of `DataLoader`s and samples from them."""

    dataloaders: dict[str, DataLoader]

    def __iter__(self) -> DictDataLoader:
        self.iters = {key: iter(dl) for key, dl in self.dataloaders.items()}
        return self

    def __next__(self) -> dict[str, Tensor]:
        return {key: next(it) for key, it in self.iters.items()}


class InfiniteTensorDataset(IterableDataset):
    def __init__(self, *tensors: Tensor):
        """Randomly sample points from the first dimension of the given tensors.

        Behaves like a normal torch `Dataset` just that we can sample from it as
        many times as we want.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        from perceptrain.data import InfiniteTensorDataset

        x_data, y_data = torch.rand(5,2), torch.ones(5,1)
        # The dataset accepts any number of tensors with the same batch dimension
        ds = InfiniteTensorDataset(x_data, y_data)

        # call `next` to get one sample from each tensor:
        xs = next(iter(ds))
        print(str(xs)) # markdown-exec: hide
        ```
        """
        if len(set([t.size(0) for t in tensors])) != 1:
            raise ValueError("Size of first dimension must be the same for all tensors.")
        self.tensors = tensors
        self.indices = list(range(tensors[0].size(0)))

    def __iter__(self) -> Iterator:
        # Shuffle the indices for every full pass
        random.shuffle(self.indices)
        while True:
            for idx in self.indices:
                yield tuple(t[idx] for t in self.tensors)


class R3Dataset(Dataset):
    def __init__(
        self, proba_dist: Callable[[int], Tensor], n_samples: int, release_threshold: float = 0.1
    ) -> None:
        """Dataset for R3 sampling (introduced in https://arxiv.org/abs/2207.02338#).

        This is an evolutionary dataset, that updates itself during training, based on the fitness values of the samples.
        It releases samples if the corresponding fitness value is below the threshold and retains them otherwise.
        The released samples are replaced by new samples generated from a probability distribution.

        While this scheme was originally proposed for training physics-informed neural networks,
        this implementation can be used for any type of data that can be sampled from a probability distribution.

        Args:
            proba_dist: Probability distribution function for generating features.
            n_samples: Number of samples to generate.
            release_threshold: Threshold for releasing samples.
        """
        if release_threshold < 0.0:
            raise ValueError("Release threshold must be non-negative.")

        self.proba_dist = proba_dist
        self.n_samples = n_samples
        self.release_threshold = release_threshold

        self.features = proba_dist(n_samples)

        self._released: Tensor | None = None
        self._released_indices: Tensor | None = None
        self._resampled: Tensor | None = None

        self.n_released: int = 0
        self.n_retained: int = 0

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tensor:
        return self.features[index]

    def _release(self, fitness_values: Tensor) -> None:
        """Release samples if the corresponding fitness value is below the threshold."""
        if len(fitness_values) != self.n_samples:
            raise ValueError("fitness_values must have the same length as the dataset")

        release_mask = fitness_values < self.release_threshold
        self._released_indices = torch.nonzero(release_mask).squeeze()  # can be empty
        self._released = self.features[self._released_indices]

        self.n_released = torch.numel(self._released_indices)
        self.n_retained = self.n_samples - self.n_released

    def _resample(self) -> Tensor:
        """Resample released samples."""
        self._resampled = self.proba_dist(self.n_released)
        return self._resampled

    def update(self, fitness_values: Tensor) -> None:
        """Update the dataset by releasing samples below fitness threshold and resampling.

        Args:
            fitness_values (Tensor): the fitness values of the samples.
        """
        self._release(fitness_values)
        if self.n_released > 0:
            new_samples = self._resample()

            with torch.no_grad():
                self.features[self._released_indices] = new_samples
        else:
            pass


class GenerativeIterableDataset(IterableDataset):
    def __init__(
        self,
        proba_dist: Callable[[], Tensor],
    ) -> None:
        """Dataset for sampling from a probability distribution.

        Samples once per iteration.

        Args:
            proba_dist: the probability distribution to be sampled.
        """
        self.proba_dist = proba_dist

    def __iter__(self) -> Iterator[Tensor]:
        while True:
            x = self.proba_dist()
            yield x


def to_dataloader(
    *tensors: Tensor,
    batch_size: int = 1,
    infinite: bool = False,
    collate_fn: Callable | None = None,
) -> DataLoader:
    """Convert torch tensors an (infinite) Dataloader.

    Arguments:
        *tensors: Torch tensors to use in the dataloader.
        batch_size: batch size of sampled tensors
        infinite: if `True`, the dataloader will keep sampling indefinitely even after the whole
            dataset was sampled once
        collate_fn: function to collate the sampled tensors. Passed to torch.utils.data.DataLoader.
            If None, defaults to torch.utils.data.default_collate.

    Examples:

    ```python exec="on" source="above" result="json"
    import torch
    from perceptrain import to_dataloader

    (x, y, z) = [torch.rand(10) for _ in range(3)]
    loader = iter(to_dataloader(x, y, z, batch_size=5, infinite=True))
    print(next(loader))
    print(next(loader))
    print(next(loader))
    ```
    """
    ds = InfiniteTensorDataset(*tensors) if infinite else TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)


@singledispatch
def data_to_device(xs: Any, *args: Any, **kwargs: Any) -> Any:
    """Utility method to move arbitrary data to 'device'."""
    raise ValueError(f"Unable to move {type(xs)} with input args: {args} and kwargs: {kwargs}.")


@data_to_device.register
def _(xs: None, *args: Any, **kwargs: Any) -> None:
    return xs


@data_to_device.register(Tensor)
def _(xs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    return xs.to(*args, **kwargs)


@data_to_device.register(list)
def _(xs: list, *args: Any, **kwargs: Any) -> list:
    return [data_to_device(x, *args, **kwargs) for x in xs]


@data_to_device.register(dict)
def _(xs: dict, *args: Any, **kwargs: Any) -> dict:
    return {key: data_to_device(val, *args, **kwargs) for key, val in xs.items()}


@data_to_device.register(DataLoader)
def _(xs: DataLoader, *args: Any, **kwargs: Any) -> DataLoader:
    return DataLoader(data_to_device(xs.dataset, *args, **kwargs))


@data_to_device.register(DictDataLoader)
def _(xs: DictDataLoader, device: torch_device) -> DictDataLoader:
    return DictDataLoader({key: data_to_device(val, device) for key, val in xs.dataloaders.items()})


@data_to_device.register(InfiniteTensorDataset)
def _(xs: InfiniteTensorDataset, device: torch_device) -> InfiniteTensorDataset:
    return InfiniteTensorDataset(*[data_to_device(val, device) for val in xs.tensors])


@data_to_device.register(TensorDataset)
def _(xs: TensorDataset, device: torch_device) -> TensorDataset:
    return TensorDataset(*[data_to_device(val, device) for val in xs.tensors])
