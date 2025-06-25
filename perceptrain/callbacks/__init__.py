from __future__ import annotations

from .callback import (
    Callback,
    EarlyStopping,
    GradientMonitoring,
    LivePlotMetrics,
    LoadCheckpoint,
    LogHyperparameters,
    LogModelTracker,
    LRSchedulerCosineAnnealing,
    LRSchedulerCyclic,
    LRSchedulerStepDecay,
    LRSchedulerReduceOnPlateau,
    PrintMetrics,
    R3Sampling,
    SaveBestCheckpoint,
    SaveCheckpoint,
    WriteMetrics,
    WritePlots,
)
from .callbackmanager import CallbacksManager
from .writer_registry import get_writer

# Modules to be automatically added to the perceptrain.callbacks namespace
__all__ = [
    "CallbacksManager",
    "Callback",
    "LivePlotMetrics",
    "LoadCheckpoint",
    "LogHyperparameters",
    "LogModelTracker",
    "WritePlots",
    "PrintMetrics",
    "R3Sampling",
    "SaveBestCheckpoint",
    "SaveCheckpoint",
    "WriteMetrics",
    "GradientMonitoring",
    "LRSchedulerStepDecay",
    "LRSchedulerReduceOnPlateau",
    "LRSchedulerCyclic",
    "LRSchedulerCosineAnnealing",
    "EarlyStopping",
    "get_writer",
]
