from __future__ import annotations

from typing import Callable, Sequence

import torch.nn as nn
from torch import Tensor

Model: nn.Module = nn.Module


class QuantumModel(nn.Module):
    """
    Base class for any quantum-based model.

    Inherits from nn.Module.
    Subclasses should implement a forward method that handles quantum logic.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Override this method in subclasses to provide.

        the forward pass for your quantum model.
        """
        return x


class QNN(QuantumModel):
    """
    A specialized quantum neural network that extends QuantumModel.

    You can define additional layers, parameters, and logic specific
    to your quantum model here.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward pass for the quantum neural network.

        Replace with your actual quantum circuit logic if you have a
        quantum simulator or hardware integration. This example just
        passes x through a classical linear layer.
        """
        return x


class FFNN(nn.Module):
    def __init__(self, layers: Sequence[int], activation_function: nn.Module = nn.GELU()) -> None:
        """
        Standard feedforward neural network.

        Args:
            layers (Sequence[int]): List of layer sizes.
            activation_function (nn.Module): Activation function to use between layers.
        """
        super().__init__()
        if len(layers) < 2:
            raise ValueError("Please specify at least one input and one output layer.")

        self.layers = layers
        self.activation_function = activation_function

        sequence = []
        for n_i, n_o in zip(self.layers[:-2], self.layers[1:-1]):
            sequence.append(nn.Linear(n_i, n_o))
            sequence.append(self.activation_function)

        sequence.append(nn.Linear(self.layers[-2], self.layers[-1]))
        self.nn = nn.Sequential(*sequence)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the neural network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, layers[0]).

        Returns:
            Tensor: Output tensor of shape (batch_size, layers[-1]).
        """
        if x.shape[1] != self.layers[0]:
            raise ValueError(f"Input tensor must have {self.layers[0]} features, got {x.shape[1]}")
        return self.nn(x)


class PINN(nn.Module):
    def __init__(
        self,
        nn: nn.Module,
        equations: dict[str, Callable[[Tensor, nn.Module], Tensor]],
    ) -> None:
        super().__init__()
        self.nn = nn
        self.equations = equations

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        return {key: self.equations[key](x_i, self.nn) for key, x_i in x.items()}
