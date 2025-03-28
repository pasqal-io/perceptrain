from __future__ import annotations

from functools import singledispatch
from typing import Any, TypeAlias
from torch.nn import Module
from torch import Tensor

Model: Module
QuantumModel: Module
QNN: Module

# Modules to be automatically added to the perceptrain namespace
__all__ = [
    "Model",
    "QuantumModel",
    "QNN",
]


@singledispatch
def rand_featureparameters(x: Model, *args: Any) -> dict[str, Tensor]:
    raise NotImplementedError(f"Unable to generate random featureparameters for object {type(x)}.")


@rand_featureparameters.register
def _(block: Module, batch_size: int = 1) -> dict[str, Tensor]:
    raise NotImplementedError(
        f"Unable to generate random featureparameters for object {type(block)}."
    )


@rand_featureparameters.register
def _(qm: Module, batch_size: int = 1) -> dict[str, Tensor]:
    raise NotImplementedError(f"Unable to generate random featureparameters for object {type(qm)}.")


@rand_featureparameters.register
def _(qnn: Module, batch_size: int = 1) -> dict[str, Tensor]:
    raise NotImplementedError(
        f"Unable to generate random featureparameters for object {type(qnn)}."
    )
