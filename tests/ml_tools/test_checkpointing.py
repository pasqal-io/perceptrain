from __future__ import annotations

import os
from itertools import count
from pathlib import Path

import torch
from nevergrad.optimization.base import Optimizer as NGOptimizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from perceptrain import QNN, QuantumModel
from perceptrain import (
    TrainConfig,
    Trainer,
    load_checkpoint,
)
from perceptrain.data import to_dataloader
from perceptrain.parameters import get_parameters, set_parameters


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.cos(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


def write_legacy_checkpoint(
    folder: Path,
    model: torch.Module,
    optimizer: Optimizer | NGOptimizer,
    iteration: int | str,
) -> None:
    iteration_substring = f"{iteration:03n}" if isinstance(iteration, int) else iteration
    model_checkpoint_name: str = f"model_{type(model).__name__}_ckpt_{iteration_substring}.pt"
    opt_checkpoint_name: str = f"opt_{type(optimizer).__name__}_ckpt_{iteration_substring}.pt"
    d = (
        model._to_dict(save_params=True)
        if isinstance(model, (QNN, QuantumModel))
        else model.state_dict()
    )
    torch.save((iteration, d), folder / model_checkpoint_name)
    if isinstance(optimizer, Optimizer):
        torch.save(
            (iteration, type(optimizer), optimizer.state_dict()), folder / opt_checkpoint_name
        )
    elif isinstance(optimizer, NGOptimizer):
        optimizer.dump(folder / opt_checkpoint_name)


def test_basic_save_load_ckpts(Basic: torch.nn.Module, tmp_path: Path) -> None:
    data = dataloader()
    model = Basic
    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        x, y = data[0], data[1]
        out = model(x)
        loss = criterion(out, y)
        return loss, {}

    config = TrainConfig(root_folder=tmp_path, max_iter=1, checkpoint_every=1, write_every=1)
    trainer = Trainer(model, optimizer, config, loss_fn, data)
    with trainer.enable_grad_opt():
        model, _ = trainer.fit()
    ps0 = get_parameters(model)
    set_parameters(model, torch.ones(len(get_parameters(model))))
    # write_checkpoint(tmp_path, model, optimizer, 1)
    # check that saved model has ones
    model, _, _ = load_checkpoint(trainer.config.log_folder, model, optimizer)
    ps1 = get_parameters(model)
    assert torch.allclose(ps0, ps1)
