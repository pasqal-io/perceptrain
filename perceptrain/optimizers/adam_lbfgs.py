from __future__ import annotations

from logging import getLogger
from typing import Dict, Optional, Callable

from torch.optim import Optimizer, Adam, LBFGS
from torch.optim.optimizer import ParamsT

logger = getLogger("perceptrain")


class AdamLBFGS(Optimizer):
    """A hybrid optimizer that switches from Adam to LBFGS after a specified number of epochs.

    This optimizer combines the benefits of Adam's efficient handling of sparse gradients and
    adaptive learning rates in early training with LBFGS's powerful quasi-Newton method for
    fine convergence in later stages.

    Args:
        params (ParamsT): Iterable of parameters to optimize
        switch_epoch (int): Epoch number to switch from Adam to LBFGS
        adam_param (Optional[Dict]): Parameters for Adam optimizer
        lbfgs_param (Optional[Dict]): Parameters for LBFGS optimizer
    """

    def __init__(
        self,
        params: ParamsT,
        switch_epoch: int = 10_000,
        adam_kwargs: Optional[Dict] = {"lr": 1e-2, "betas": (0.9, 0.999)},
        lbfgs_kwargs: Optional[Dict] = {"lr": 1, "max_iter": 20},
    ) -> None:
        self.params = list(params)
        self.switch_epoch = switch_epoch

        # Initialize Adam and LBFGS optimizers
        self._adam = Adam(self.params, **adam_kwargs)
        self._lbfgs = LBFGS(self.params, **lbfgs_kwargs)

        super().__init__(self.params, defaults={})

        # Track current epoch
        self.current_epoch = 0
        self._switched = False

    def state_dict(self) -> Dict:
        """Returns the state of the optimizer as a dict."""
        state_dict = {
            "current_epoch": self.current_epoch,
            "switched": self._switched,
            "adam_state": self._adam.state_dict(),
            "lbfgs_state": self._lbfgs.state_dict() if self._switched else {},
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """Loads the optimizer state.

        Args:
            state_dict (Dict): optimizer state dictionary. Should be an object returned
            from a call to state_dict().
        """
        self.current_epoch = state_dict["current_epoch"]
        self._switched = state_dict["switched"]

        # Load state for both optimizers
        self._adam.load_state_dict(state_dict["adam_state"])
        if self._switched:
            self._lbfgs.load_state_dict(state_dict["lbfgs_state"])

    def step(self, closure: Callable = None) -> None:
        """Performs a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model and returns the loss
        """
        self.current_epoch += 1

        if self.current_epoch < self.switch_epoch:
            self._adam.step(closure)
        else:
            # LBFGS requires a closure
            if closure is None:
                raise ValueError("LBFGS optimizer requires a closure function")
            self._lbfgs.step(closure)
            if self.current_epoch == self.switch_epoch:
                logger.info(f"Switched to LBFGS at epoch {self.switch_epoch}")
                self._switched = True

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clears the gradients of all optimized parameters.

        Args:
            set_to_none (bool): If True, the grads are set to None instead of zero.
        """
        if not self._switched:
            self._adam.zero_grad(set_to_none=set_to_none)
        else:
            self._lbfgs.zero_grad(set_to_none=set_to_none)
