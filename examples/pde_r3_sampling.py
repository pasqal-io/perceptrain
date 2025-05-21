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

import torch
from torch.utils.data import DataLoader

from perceptrain.callbacks import Callback
from perceptrain.data import DictDataLoader, GenerativeLabelledFixedDataset
from perceptrain.loss.loss import MSELoss
from perceptrain.types import Loss

"""The R3 logic should be a callback.

Generalize to resampling from a continuous proba dist (or actually a function,
which does not need to be in (0, 1) and integrate to 1.
"""


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
        initial_dataset: GenerativeLabelledFixedDataset,
        fitness_function: Loss = MSELoss(),
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
        self.n_features = initial_dataset.tensors[0].size(dim=1)
        self.threshold = threshold
        self.fitness_function = fitness_function
        self.dataloader_key = dataloader_key

        self.n_retained = 0
        super().__init__(on="train_epoch_start")

    def _sample_uniform(self, n_samples: int) -> torch.Tensor:
        """Random uniform sampling in [0, 1]."""
        return torch.rand(size=(n_samples, self.n_features))

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> None:
        """Eventually make a new dataloader from a dataset.__init__() call."""
        features, labels = self.dataset.tensors[0], self.dataset.tensors[1]

        # Compute fitness function on all samples
        fitnesses, _ = self.fitness_function(trainer.model, (features, labels))

        # Retain
        retained = fitnesses > self.threshold
        self.n_retained = len(retained)

        # Resample
        new_features = self._sample_uniform(n_samples=self.n_samples_total - self.n_retained)

        # Release
        self.features = torch.where(retained, features, new_features)

        # Compute labels
        self.labels = self.dataset.labelling_function(self.features)

        # Update the dataset
        self.dataset.tensors = self.features, self.labels

        # Update dataloader of the trainer with the re-sampled dataset
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


def main():
    pass


if __name__ == "__main__":
    main()
