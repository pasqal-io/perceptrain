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
        """Physics-informed neural network.

        Args:
            nn (nn.Module): Neural network module.
            equations (dict[str, Callable[[Tensor, nn.Module], Tensor]]): Dictionary of equations.
                These are assumed in the form LHS(x) = 0, so each term of `equations` should
                provide the left-hand side.

        Notes:
            Example of equations: heat equation with a Gaussian initial condition.
            ```python
            import torch

            alpha = 1.0

            def heat_eqn(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
                grad_u = torch.autograd.grad(
                    outputs=model(x),
                    inputs=x,
                    grad_outputs=torch.ones_like(u),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                dudt = grad_u[:, 0]
                dudx = grad_u[:, 1]
                grad2_u = torch.autograd.grad(
                    outputs=dudx,
                    inputs=x,
                    grad_outputs=torch.ones_like(dudx),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                d2udx2 = grad2_u[:, 1]

                return dudt - beta * d2udx2

            def initial_condition(x: torch.Tensor, model: torch.nn.Module):
                def gaussian(z):
                    return torch.exp(-z**2)

                return model(x) - gaussian(x[:, 1])

            def boundary_condition(x: torch.Tensor, model: torch.nn.Module):
                grad_u = torch.autograd.grad(
                    outputs=model(x),
                    inputs=x,
                    grad_outputs=torch.ones_like(model(x)),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                return dudx[:, 1] - torch.zeros_like(x[:, 1])

            equations = {
                "pde": heat_eqn,
                "initial_condition": initial_condition,
                "boundary_condition_left": boundary_condition,
                "boundary_condition_right": boundary_condition,
            }
            ```
        """
        super().__init__()
        self.nn = nn
        self.equations = equations

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass through the physical-informed neural network.

        Args:
            x (dict[str, Tensor]): Dictionary of input tensors. The keys of the dictionary should
                match the keys in the `equations` dictionary.

        Returns:
            dict[str, Tensor]: Dictionary of output tensors.
        """
        if not all(key in x for key in self.equations):
            raise ValueError(f"Input dictionary must contain keys {list(self.equations.keys())}")
        return {key: self.equations[key](x_i, self.nn) for key, x_i in x.items()}
