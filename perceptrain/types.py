from __future__ import annotations

import importlib
from enum import Enum
from typing import Callable, Iterable, Tuple, Union

import numpy as np
import torch.nn as nn
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from torch import Tensor, pi

TNumber = Union[int, float, complex, np.int64, np.float64]
"""Union of python and numpy numeric types."""

TDrawColor = Tuple[float, float, float, float]

TParameter = Union[TNumber, Tensor, str]
"""Union of numbers, tensors, and parameter types."""

TArray = Union[Iterable, Tensor, np.ndarray]
"""Union of common array types."""

TGenerator = Union[Tensor]
"""Union of torch tensors and numpy arrays."""

TData = Union[Tensor, dict[str, Tensor]]
TBatch = tuple[TData, ...]


PI = pi

# Modules to be automatically added to the preceptrain namespace
__all__ = [
    "QuantumModel",
    "QNN",
    "ResultType",
    "SerializationFormat",
    "FigFormat",
    "ParamDictType",
    "DifferentiableExpression",
    "ExperimentTrackingTool",
    "ExecutionType",
    "LoggablePlotFunction",
]  # type: ignore

# Basic models for trainer
QuantumModel = nn.Module
QNN = nn.Module


class StrEnum(str, Enum):
    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))  # type: ignore


class ResultType(StrEnum):
    """Available data types for generating certain results."""

    STRING = "String"
    """String Type."""
    TORCH = "Torch"
    """Torch Tensor Type."""
    NUMPY = "Numpy"
    """Numpy Array Type."""


class _DiffMode(StrEnum):
    """Differentiation modes to choose from."""

    GPSR = "gpsr"
    """Basic generalized parameter shift rule."""
    AD = "ad"
    """Automatic Differentiation."""
    ADJOINT = "adjoint"
    """Adjoint Differentiation."""


class _BackendName(StrEnum):
    """The available backends for running circuits."""

    PYQTORCH = "pyqtorch"
    """The Pyqtorch backend."""
    PULSER = "pulser"
    """The Pulser backend."""
    HORQRUX = "horqrux"
    """The horqrux backend."""


class _Engine(StrEnum):
    TORCH = "torch"
    JAX = "jax"


# If proprietary qadence_extensions is available, import the
# right function since more backends are supported.
try:
    module = importlib.import_module("qadence_extensions.types")
    BackendName = getattr(module, "BackendName")
    DiffMode = getattr(module, "DiffMode")
    Engine = getattr(module, "Engine")
except ModuleNotFoundError:
    BackendName = _BackendName
    DiffMode = _DiffMode
    Engine = _Engine


class SerializationFormat(StrEnum):
    """Available serialization formats for circuits."""

    PT = "PT"
    """The PT format used by Torch."""
    JSON = "JSON"
    """The Json format."""


class FigFormat(StrEnum):
    """Available output formats for exporting visualized circuits to a file."""

    PNG = "PNG"
    """PNG format."""
    PDF = "PDF"
    """PDF format."""
    SVG = "SVG"
    """SVG format."""


ParamDictType = dict[str, ArrayLike]
DifferentiableExpression = Callable[..., ArrayLike]


class ExperimentTrackingTool(StrEnum):
    TENSORBOARD = "tensorboard"
    """Use the tensorboard experiment tracker."""
    MLFLOW = "mlflow"
    """Use the ml-flow experiment tracker."""


class ExecutionType(StrEnum):
    TORCHRUN = "torchrun"
    """Torchrun based distribution execution."""
    DEFAULT = "default"
    """Default distribution execution."""


LoggablePlotFunction = Callable[[nn.Module, int], tuple[str, Figure]]
LossFunction = Callable[[TBatch, nn.Module], tuple[Tensor, dict[str, Tensor]]]
