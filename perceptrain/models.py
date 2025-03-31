from __future__ import annotations

from torch.nn import Module
from torch import Tensor

Model: Module = Module


class QuantumModel(Module):
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
