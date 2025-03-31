from __future__ import annotations

import logging
import logging.config
import os
from importlib import import_module
from pathlib import Path

import yaml
from torch import cdouble, set_default_dtype
from torch import float64 as torchfloat64

DEFAULT_FLOAT_DTYPE = torchfloat64
DEFAULT_COMPLEX_DTYPE = cdouble
set_default_dtype(DEFAULT_FLOAT_DTYPE)

logging_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOG_CONFIG_PATH = os.environ.get(
    "PERCEPTRAIN_LOG_CONFIG", f"{Path(__file__).parent}/log_config.yaml"
)
LOG_BASE_LEVEL = os.environ.get("PERCEPTRAIN_LOG_LEVEL", "").upper()

with open(LOG_CONFIG_PATH, "r") as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(log_config)

logger: logging.Logger = logging.getLogger(__name__)
LOG_LEVEL = logging_levels.get(LOG_BASE_LEVEL, logging.INFO)  # type: ignore[arg-type]
logger.setLevel(LOG_LEVEL)
[
    h.setLevel(LOG_LEVEL)  # type: ignore[func-returns-value]
    for h in logger.handlers
    if h.get_name() == "console" or h.get_name() == "richconsole"
]
logger.debug(f"Perceptrain logger successfully setup with log level {LOG_LEVEL}")


"""Fetch the functions defined in the __all__ of each sub-module.

Import to the perceptrain name space. Make sure each added submodule has the respective definition:

    - `__all__ = ["function0", "function1", ...]`

Furthermore, add the submodule to the list below to automatically build
the __all__ of the perceptrain namespace. Make sure to keep alphabetical ordering.
"""

list_of_submodules = [
    ".callbacks",
]

__all__ = []
for submodule in list_of_submodules:
    __all_submodule__ = getattr(import_module(submodule, package="perceptrain"), "__all__")
    __all__ += __all_submodule__

from .trainer import *
from .models import *
from .callbacks.saveload import load_checkpoint, load_model, write_checkpoint
from .config import TrainConfig
from .data import DictDataLoader, InfiniteTensorDataset, OptimizeResult, to_dataloader
from .information import InformationContent
from .optimize_step import optimize_step as default_optimize_step
from .parameters import get_parameters, num_parameters, set_parameters
from .tensors import numpy_to_tensor, promote_to, promote_to_tensor
from .trainer import Trainer

# Modules to be automatically added to the perceptrain namespace
__all__ = ["DictDataLoader", "TrainConfig", "Trainer", "QNN", "QuantumModel", "Model"]
