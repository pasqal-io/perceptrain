from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS

from perceptrain.optimizers.adam_lbfgs import AdamLBFGS


def test_adam_lbfgs_initialization(Basic):
    opt = AdamLBFGS(Basic.parameters())
    assert isinstance(opt._adam, Adam)
    assert isinstance(opt._lbfgs, LBFGS)
    assert opt.current_epoch == 0
    assert not opt._switched


def test_adam_lbfgs_state_dict(adamlbfgs_optimizer):
    state = adamlbfgs_optimizer.state_dict()
    assert 'current_epoch' in state
    assert 'switched' in state
    assert 'adam_state' in state
    assert 'lbfgs_state' in state


def test_adam_lbfgs_load_state_dict(Basic):
    opt1 = AdamLBFGS(Basic.parameters(), switch_epoch=5)
    opt2 = AdamLBFGS(Basic.parameters(), switch_epoch=5)
    
    opt1.current_epoch = 3
    state = opt1.state_dict()
    opt2.load_state_dict(state)
    
    assert opt2.current_epoch == 3
    assert opt2._switched == opt1._switched


def test_adam_phase_step(Basic, adamlbfgs_optimizer):
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    
    def closure():
        adamlbfgs_optimizer.zero_grad()
        output = Basic(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        return loss
    
    adamlbfgs_optimizer.step(closure)
    assert adamlbfgs_optimizer.current_epoch == 1
    assert not adamlbfgs_optimizer._switched


def test_lbfgs_phase_step(Basic):
    opt = AdamLBFGS(Basic.parameters(), switch_epoch=3)
    opt.current_epoch = 2
    
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    
    def closure():
        opt.zero_grad()
        output = Basic(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        return loss
    
    opt.step(closure)
    assert opt.current_epoch == 3
    assert opt._switched


def test_lbfgs_requires_closure(Basic):
    opt = AdamLBFGS(Basic.parameters(), switch_epoch=1)
    opt.current_epoch = 1
    
    with pytest.raises(ValueError, match="LBFGS optimizer requires a closure function"):
        opt.step()


def test_custom_parameters(Basic):
    adam_kwargs = {'lr': 0.001, 'betas': (0.8, 0.999)}
    lbfgs_kwargs = {'lr': 0.1, 'max_iter': 10}
    
    opt = AdamLBFGS(
        Basic.parameters(),
        switch_epoch=3,
        adam_kwargs=adam_kwargs,
        lbfgs_kwargs=lbfgs_kwargs
    )
    
    assert opt._adam.defaults['lr'] == 0.001
    assert opt._adam.defaults['betas'] == (0.8, 0.999)
    assert opt._lbfgs.defaults['lr'] == 0.1
    assert opt._lbfgs.defaults['max_iter'] == 10