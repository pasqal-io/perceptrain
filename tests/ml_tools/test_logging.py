from __future__ import annotations

import logging
import os
import shutil
from itertools import count
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import mlflow
import pytest
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mlflow import MlflowClient
from mlflow.entities import Run
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from perceptrain import QuantumModel, TrainConfig, Trainer
from perceptrain.callbacks.writer_registry import BaseWriter
from perceptrain.data import to_dataloader
from perceptrain.types import ExperimentTrackingTool


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.cos(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


def setup_model(model: Module) -> tuple[Callable, Optimizer]:
    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    def loss_fn(data: torch.Tensor, model: torch.nn.Module) -> tuple[torch.Tensor, dict]:
        next(cnt)
        out = model()
        loss = criterion(out, torch.rand(1))
        return loss, {}

    return loss_fn, optimizer


def load_mlflow_model(writer: BaseWriter) -> None:
    run_id = writer.run.info.run_id

    mlflow.pytorch.load_model(model_uri=f"runs:/{run_id}/model")


def find_mlflow_artifacts_path(run: Run) -> Path:
    artifact_uri = run.info.artifact_uri
    parsed_uri = urlparse(artifact_uri)
    return Path(os.path.abspath(os.path.join(parsed_uri.netloc, parsed_uri.path)))


def clean_mlflow_experiment(writer: BaseWriter) -> None:
    experiment_id = writer.run.info.experiment_id
    client = MlflowClient()

    runs = client.search_runs(experiment_id)

    def clean_artifacts(run: Run) -> None:
        local_path = find_mlflow_artifacts_path(run)
        shutil.rmtree(local_path)

    for run in runs:
        clean_artifacts(run)

        run_id = run.info.run_id
        client.delete_run(run_id)

        mlruns_base_dir = "./mlruns"
        if os.path.isdir(mlruns_base_dir):
            shutil.rmtree(os.path.join(mlruns_base_dir, experiment_id))


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("perceptrain")
    # an additional streamhandler is needed in perceptrain as
    # caplog does not record richhandler logs.
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    return logger


def test_hyperparams_logging_mlflow(BasicNoInput: torch.nn.Module, tmp_path: Path) -> None:
    model = BasicNoInput

    loss_fn, optimizer = setup_model(model)

    hyperparams = {"max_iter": int(10), "lr": 0.1}

    config = TrainConfig(
        root_folder=tmp_path,
        max_iter=hyperparams["max_iter"],  # type: ignore
        checkpoint_every=1,
        write_every=1,
        hyperparams=hyperparams,
        tracking_tool=ExperimentTrackingTool.MLFLOW,
    )

    trainer = Trainer(model, optimizer, config, loss_fn)
    with trainer.enable_grad_opt():
        trainer.fit()

    writer = trainer.callback_manager.writer
    experiment_id = writer.run.info.experiment_id
    run_id = writer.run.info.run_id

    experiment_dir = Path(f"mlruns/{experiment_id}")
    hyperparams_files = [experiment_dir / run_id / "params" / key for key in hyperparams.keys()]

    assert all([os.path.isfile(hf) for hf in hyperparams_files])

    clean_mlflow_experiment(trainer.callback_manager.writer)


def test_hyperparams_logging_tensorboard(BasicNoInput: torch.nn.Module, tmp_path: Path) -> None:
    model = BasicNoInput

    loss_fn, optimizer = setup_model(model)

    hyperparams = {"max_iter": int(10), "lr": 0.1}

    config = TrainConfig(
        root_folder=tmp_path,
        max_iter=hyperparams["max_iter"],  # type: ignore
        checkpoint_every=1,
        write_every=1,
        hyperparams=hyperparams,
        tracking_tool=ExperimentTrackingTool.TENSORBOARD,
    )

    trainer = Trainer(model, optimizer, config, loss_fn)
    with trainer.enable_grad_opt():
        trainer.fit()


def test_model_logging_mlflow_BasicNoInputQM(BasicNoInput: torch.nn.Module, tmp_path: Path) -> None:
    model = BasicNoInput
    loss_fn, optimizer = setup_model(model)

    config = TrainConfig(
        root_folder=tmp_path,
        max_iter=10,  # type: ignore
        checkpoint_every=1,
        write_every=1,
        log_model=True,
        tracking_tool=ExperimentTrackingTool.MLFLOW,
    )

    trainer = Trainer(model, optimizer, config, loss_fn)
    with trainer.enable_grad_opt():
        trainer.fit()

    load_mlflow_model(trainer.callback_manager.writer)

    clean_mlflow_experiment(trainer.callback_manager.writer)


def test_model_logging_tensorboard(
    BasicNoInput: torch.nn.Module, tmp_path: Path, capsys: pytest.LogCaptureFixture
) -> None:
    setup_logger()
    model = BasicNoInput

    loss_fn, optimizer = setup_model(model)

    config = TrainConfig(
        root_folder=tmp_path,
        max_iter=10,  # type: ignore
        checkpoint_every=1,
        write_every=1,
        log_model=True,
        tracking_tool=ExperimentTrackingTool.TENSORBOARD,
    )

    trainer = Trainer(model, optimizer, config, loss_fn)
    with trainer.enable_grad_opt():
        trainer.fit()

    captured = capsys.readouterr()
    assert "Model logging is not supported by tensorboard. No model will be logged." in captured.err


def test_plotting_mlflow(BasicNoInput: torch.nn.Module, tmp_path: Path) -> None:
    model = BasicNoInput

    loss_fn, optimizer = setup_model(model)

    def plot_error(model: QuantumModel, iteration: int) -> tuple[str, Figure]:
        descr = f"error_epoch_{iteration}.png"
        fig, ax = plt.subplots()
        x = torch.linspace(0, 1, 100).reshape(-1, 1)
        scalar_out = model()
        out = scalar_out.expand_as(x)
        ground_truth = torch.rand_like(out)
        error = ground_truth - out
        ax.plot(x.detach().numpy(), error.detach().numpy())
        return descr, fig

    max_iter = 10
    plot_every = 2
    config = TrainConfig(
        root_folder=tmp_path,
        max_iter=max_iter,
        checkpoint_every=1,
        write_every=1,
        plot_every=plot_every,
        tracking_tool=ExperimentTrackingTool.MLFLOW,
        plotting_functions=(plot_error,),
    )

    trainer = Trainer(model, optimizer, config, loss_fn)
    with trainer.enable_grad_opt():
        trainer.fit()

    all_plot_names = []
    all_plot_names.extend([f"error_epoch_{i}.png" for i in range(0, max_iter, plot_every)])

    artifact_path = find_mlflow_artifacts_path(trainer.callback_manager.writer.run)

    assert all([os.path.isfile(artifact_path / pn) for pn in all_plot_names])

    clean_mlflow_experiment(trainer.callback_manager.writer)


def test_plotting_tensorboard(BasicNoInput: torch.nn.Module, tmp_path: Path) -> None:
    model = BasicNoInput

    loss_fn, optimizer = setup_model(model)

    def plot_error(model: QuantumModel, iteration: int) -> tuple[str, Figure]:
        descr = f"error_epoch_{iteration}.png"
        fig, ax = plt.subplots()
        x = torch.linspace(0, 1, 100).reshape(-1, 1)
        scalar_out = model()
        out = scalar_out.expand_as(x)
        ground_truth = torch.rand_like(out)
        error = ground_truth - out
        ax.plot(x.detach().numpy(), error.detach().numpy())
        return descr, fig

    config = TrainConfig(
        root_folder=tmp_path,
        max_iter=10,
        checkpoint_every=1,
        write_every=1,
        tracking_tool=ExperimentTrackingTool.TENSORBOARD,
        plotting_functions=(plot_error,),
    )

    trainer = Trainer(model, optimizer, config, loss_fn)
    with trainer.enable_grad_opt():
        trainer.fit()
