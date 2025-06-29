from __future__ import annotations

import math
from logging import getLogger
from typing import Any, Callable

import livelossplot.plot_losses as llp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from livelossplot.outputs import MatplotlibPlot
from torch import Tensor

from perceptrain.callbacks.saveload import load_checkpoint, write_checkpoint
from perceptrain.callbacks.writer_registry import BaseWriter
from perceptrain.config import TrainConfig
from perceptrain.data import (
    OptimizeResult,
    R3Dataset,
)
from perceptrain.stages import TrainingStage

# Define callback types
CallbackFunction = Callable[..., Any]
CallbackConditionFunction = Callable[..., bool]

logger = getLogger("perceptrain")


class Callback:
    """Base class for defining various training callbacks.

    Attributes:
        on (str): The event on which to trigger the callback.
            Must be a valid on value from: ["train_start", "train_end",
                "train_epoch_start", "train_epoch_end", "train_batch_start",
                "train_batch_end","val_epoch_start", "val_epoch_end",
                "val_batch_start", "val_batch_end", "test_batch_start",
                "test_batch_end"]
        called_every (int): Frequency of callback calls in terms of iterations.
        callback (CallbackFunction | None): The function to call if the condition is met.
        callback_condition (CallbackConditionFunction | None): Condition to check before calling.
        modify_optimize_result (CallbackFunction | dict[str, Any] | None):
            Function to modify `OptimizeResult`.

    A callback can be defined in two ways:

    1. **By providing a callback function directly in the base class**:
       This is useful for simple callbacks that don't require subclassing.

       Example:
       ```python exec="on" source="material-block" result="json"
       from perceptrain.callbacks import Callback

       def custom_callback_function(trainer, config, writer):
           print("Custom callback executed.")

       custom_callback = Callback(
           on="train_end",
           called_every=5,
           callback=custom_callback_function
       )
       ```

    2. **By inheriting and implementing the `run_callback` method**:
       This is suitable for more complex callbacks that require customization.

       Example:
       ```python exec="on" source="material-block" result="json"
       from perceptrain.callbacks import Callback
       class CustomCallback(Callback):
           def run_callback(self, trainer, config, writer):
               print("Custom behavior in the inherited run_callback method.")

       custom_callback = CustomCallback(on="train_end", called_every=10)
       ```
    """

    VALID_ON_VALUES = [
        "train_start",
        "train_end",
        "train_epoch_start",
        "train_epoch_end",
        "train_batch_start",
        "train_batch_end",
        "val_epoch_start",
        "val_epoch_end",
        "val_batch_start",
        "val_batch_end",
        "test_batch_start",
        "test_batch_end",
    ]

    def __init__(
        self,
        on: str | TrainingStage = "idle",
        called_every: int = 1,
        callback: CallbackFunction | None = None,
        callback_condition: CallbackConditionFunction | None = None,
        modify_optimize_result: CallbackFunction | dict[str, Any] | None = None,
    ):
        if not isinstance(called_every, int):
            raise ValueError("called_every must be a positive integer or 0")

        self.callback: CallbackFunction | None = callback
        self.on: str | TrainingStage = on
        self.called_every: int = called_every
        self.callback_condition = (
            callback_condition if callback_condition else Callback.default_callback
        )

        if isinstance(modify_optimize_result, dict):
            self.modify_optimize_result = lambda opt_res: Callback.modify_opt_res_dict(
                opt_res, modify_optimize_result
            )
        else:
            self.modify_optimize_result = (
                modify_optimize_result
                if modify_optimize_result
                else Callback.modify_opt_res_default
            )

    @staticmethod
    def default_callback(_: Any) -> bool:
        return True

    @staticmethod
    def modify_opt_res_dict(
        opt_res: OptimizeResult,
        modify_optimize_result: dict[str, Any] = {},
    ) -> OptimizeResult:
        opt_res.extra.update(modify_optimize_result)
        return opt_res

    @staticmethod
    def modify_opt_res_default(opt_res: OptimizeResult) -> OptimizeResult:
        return opt_res

    @property
    def on(self) -> TrainingStage | str:
        """
        Returns the TrainingStage.

        Returns:
            TrainingStage: TrainingStage for the callback
        """
        return self._on

    @on.setter
    def on(self, on: str | TrainingStage) -> None:
        """
        Sets the training stage on for the callback.

        Args:
            on (str | TrainingStage): TrainingStage for the callback
        """
        if isinstance(on, str):
            if on not in self.VALID_ON_VALUES:
                raise ValueError(f"Invalid value for 'on'. Must be one of {self.VALID_ON_VALUES}.")
            self._on = TrainingStage(on)
        elif isinstance(on, TrainingStage):
            self._on = on
        else:
            raise ValueError("Invalid value for 'on'. Must be `str` or `TrainingStage`.")

    def _should_call(self, when: str, opt_result: OptimizeResult) -> bool:
        """Checks if the callback should be called.

        Args:
            when (str): The event when the callback is considered for execution.
            opt_result (OptimizeResult): The current optimization results.

        Returns:
            bool: Whether the callback should be called.
        """
        if when in [TrainingStage("train_start"), TrainingStage("train_end")]:
            return True
        if self.called_every == 0 or opt_result.iteration == 0:
            return False
        if opt_result.iteration % self.called_every == 0 and self.callback_condition(opt_result):
            return True
        return False

    def __call__(
        self, when: TrainingStage, trainer: Any, config: TrainConfig, writer: BaseWriter
    ) -> Any:
        """Executes the callback if conditions are met.

        Args:
            when (str): The event when the callback is triggered.
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.

        Returns:
            Any: Result of the callback function if executed.
        """
        opt_result = trainer.opt_result
        if self.on == when:
            if opt_result:
                opt_result = self.modify_optimize_result(opt_result)
            if self._should_call(when, opt_result):
                return self.run_callback(trainer, config, writer)

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Executes the defined callback.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.

        Returns:
            Any: Result of the callback execution.

        Raises:
            NotImplementedError: If not implemented in subclasses.
        """
        if self.callback is not None:
            return self.callback(trainer, config, writer)
        raise NotImplementedError("Subclasses should override the run_callback method.")


class PrintMetrics(Callback):
    """Callback to print metrics using the writer.

    The `PrintMetrics` callback can be added to the `TrainConfig`
    callbacks as a custom user defined callback.

    Example Usage in `TrainConfig`:
    To use `PrintMetrics`, include it in the `callbacks` list when
    setting up your `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    from perceptrain.callbacks import PrintMetrics

    # Create an instance of the PrintMetrics callback
    print_metrics_callback = PrintMetrics(on = "val_batch_end", called_every = 100)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback that runs every 100 val_batch_end
        callbacks=[print_metrics_callback]
    )
    ```
    """

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Prints metrics using the writer.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        opt_result = trainer.opt_result
        writer.print_metrics(opt_result)


class WriteMetrics(Callback):
    """Callback to write metrics using the writer.

    The `WriteMetrics` callback can be added to the `TrainConfig` callbacks as
    a custom user defined callback.

    Example Usage in `TrainConfig`:
    To use `WriteMetrics`, include it in the `callbacks` list when setting up your
    `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    from perceptrain.callbacks import WriteMetrics

    # Create an instance of the WriteMetrics callback
    write_metrics_callback = WriteMetrics(on = "val_batch_end", called_every = 100)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback that runs every 100 val_batch_end
        callbacks=[write_metrics_callback]
    )
    ```
    """

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Writes metrics using the writer.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        if trainer.accelerator.rank == 0:
            opt_result = trainer.opt_result
            writer.write(opt_result.iteration, opt_result.metrics)


class WritePlots(Callback):
    """Callback to plot metrics using the writer.

    The `WritePlots` callback can be added to the `TrainConfig` callbacks as
    a custom user defined callback.

    Example Usage in `TrainConfig`:
    To use `WritePlots`, include it in the `callbacks` list when setting up your
    `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    from perceptrain.callbacks import WritePlots

    # Create an instance of the WritePlots callback
    plot_metrics_callback = WritePlots(on = "val_batch_end", called_every = 100)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback that runs every 100 val_batch_end
        callbacks=[plot_metrics_callback]
    )
    ```
    """

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Plots metrics using the writer.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        if trainer.accelerator.rank == 0:
            opt_result = trainer.opt_result
            plotting_functions = config.plotting_functions
            writer.plot(trainer.model, opt_result.iteration, plotting_functions)


class LivePlotMetrics(Callback):
    """Callback to follow metrics on screen during training.

    It uses [livelossplot](https://github.com/stared/livelossplot) to update losses and metrics
    at every call and plot them via matplotlib.
    """

    def __init__(self, on: str, called_every: int, arrange: dict[str, list[str]] | None = None):
        """Initializes the callback.

        Args:
            on (str): The event to trigger the callback.
            called_every (int): Frequency of callback calls in terms of iterations.
            arrange (dict[str, dict[str, list[str]]): How metrics are arranged for the subplots. Each entry
                is a different group, which will correspond to a different subplot.
                If None, all metrics will be plotted in a single subplot.
                Defaults to None.
        """
        super().__init__(on=on, called_every=called_every)
        self.arrange = arrange

        self.output_mode = llp.get_mode()

        self._first_call = True

    def _after_plots(self, fig: plt.Figure) -> None:
        fig.tight_layout()
        if self.output_mode == "script":
            plt.draw()
            plt.pause(0.1)
            plt.close(fig)
        else:
            plt.show()

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        if self._first_call:
            if self.arrange is None:
                self.arrange = {"Training": [key for key in trainer.opt_result.metrics.keys()]}

            self.liveloss = llp.PlotLosses(
                outputs=[MatplotlibPlot(after_plots=self._after_plots)],
                groups=self.arrange,
            )
            self._first_call = False

        if trainer.accelerator.rank == 0:
            opt_result = trainer.opt_result
            self.liveloss.update(opt_result.metrics, current_step=trainer.current_epoch)
            with torch.no_grad():
                self.liveloss.send()


class LogHyperparameters(Callback):
    """Callback to log hyperparameters using the writer.

    The `LogHyperparameters` callback can be added to the `TrainConfig` callbacks
    as a custom user defined callback.

    Example Usage in `TrainConfig`:
    To use `LogHyperparameters`, include it in the `callbacks` list when setting up your
    `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    from perceptrain.callbacks import LogHyperparameters

    # Create an instance of the LogHyperparameters callback
    log_hyper_callback = LogHyperparameters(on = "val_batch_end", called_every = 100)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback that runs every 100 val_batch_end
        callbacks=[log_hyper_callback]
    )
    ```
    """

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Logs hyperparameters using the writer.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        if trainer.accelerator.rank == 0:
            hyperparams = config.hyperparams
            writer.log_hyperparams(hyperparams)


class SaveCheckpoint(Callback):
    """Callback to save a model checkpoint.

    The `SaveCheckpoint` callback can be added to the `TrainConfig` callbacks
    as a custom user defined callback.

    Example Usage in `TrainConfig`:
    To use `SaveCheckpoint`, include it in the `callbacks` list when setting up your
    `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    from perceptrain.callbacks import SaveCheckpoint

    # Create an instance of the SaveCheckpoint callback
    save_checkpoint_callback = SaveCheckpoint(on = "val_batch_end", called_every = 100)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback that runs every 100 val_batch_end
        callbacks=[save_checkpoint_callback]
    )
    ```
    """

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Saves a model checkpoint.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        if trainer.accelerator.rank == 0:
            folder = config.log_folder
            model = trainer.model
            optimizer = trainer.optimizer
            opt_result = trainer.opt_result
            write_checkpoint(folder, model, optimizer, opt_result.iteration)


class SaveBestCheckpoint(SaveCheckpoint):
    """Callback to save the best model checkpoint based on a validation criterion."""

    def __init__(self, on: str, called_every: int):
        """Initializes the SaveBestCheckpoint callback.

        Args:
            on (str): The event to trigger the callback.
            called_every (int): Frequency of callback calls in terms of iterations.
        """
        super().__init__(on=on, called_every=called_every)
        self.best_loss = float("inf")

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Saves the checkpoint if the current loss is better than the best loss.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        if trainer.accelerator.rank == 0:
            opt_result = trainer.opt_result
            if config.validation_criterion and config.validation_criterion(
                opt_result.loss, self.best_loss, config.val_epsilon
            ):
                self.best_loss = opt_result.loss

                folder = config.log_folder
                model = trainer.model
                optimizer = trainer.optimizer
                opt_result = trainer.opt_result
                write_checkpoint(folder, model, optimizer, "best")


class LoadCheckpoint(Callback):
    """Callback to load a model checkpoint."""

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Loads a model checkpoint.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.

        Returns:
            Any: The result of loading the checkpoint.
        """
        if trainer.accelerator.rank == 0:
            folder = config.log_folder
            model = trainer.model
            optimizer = trainer.optimizer
            device = trainer.accelerator.execution.log_device
            return load_checkpoint(folder, model, optimizer, device=device)


class LogModelTracker(Callback):
    """Callback to log the model using the writer."""

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Logs the model using the writer.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        if trainer.accelerator.rank == 0:
            model = trainer.model
            writer.log_model(
                model, trainer.train_dataloader, trainer.val_dataloader, trainer.test_dataloader
            )


class LRSchedulerStepDecay(Callback):
    """
    Reduces the learning rate by a factor at regular intervals.

    This callback adjusts the learning rate by multiplying it with a decay factor
    after a specified number of iterations. The learning rate is updated as:
        lr = lr * gamma

    Example Usage in `TrainConfig`:
    To use `LRSchedulerStepDecay`, include it in the `callbacks` list when setting
    up your `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    from perceptrain.callbacks import LRSchedulerStepDecay

    # Create an instance of the LRSchedulerStepDecay callback
    lr_step_decay = LRSchedulerStepDecay(on="train_epoch_end",
                                         called_every=100,
                                         gamma=0.5)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback
        callbacks=[lr_step_decay]
    )
    ```
    """

    def __init__(self, on: str, called_every: int, gamma: float = 0.5):
        """Initializes the LRSchedulerStepDecay callback.

        Args:
            on (str): The event to trigger the callback.
            called_every (int): Frequency of callback calls in terms of iterations.
            gamma (float, optional): The decay factor applied to the learning rate.
                A value < 1 reduces the learning rate over time. Default is 0.5.
        """
        super().__init__(on=on, called_every=called_every)
        self.gamma = gamma

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> None:
        """
        Runs the callback to apply step decay to the learning rate.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter): The writer object for logging.
        """
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] *= self.gamma


class LRSchedulerCyclic(Callback):
    """
    Applies a cyclic learning rate schedule during training.

    This callback oscillates the learning rate between a minimum (base_lr)
    and a maximum (max_lr) over a defined cycle length (step_size). The learning
    rate follows a triangular wave pattern.

    Example Usage in `TrainConfig`:
    To use `LRSchedulerCyclic`, include it in the `callbacks` list when setting
    up your `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    from perceptrain.callbacks import LRSchedulerCyclic

    # Create an instance of the LRSchedulerCyclic callback
    lr_cyclic = LRSchedulerCyclic(on="train_batch_end",
                                  called_every=1,
                                  base_lr=0.001,
                                  max_lr=0.01,
                                  step_size=2000)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback
        callbacks=[lr_cyclic]
    )
    ```
    """

    def __init__(self, on: str, called_every: int, base_lr: float, max_lr: float, step_size: int):
        """Initializes the LRSchedulerCyclic callback.

        Args:
            on (str): The event to trigger the callback.
            called_every (int): Frequency of callback calls in terms of iterations.
            base_lr (float): The minimum learning rate.
            max_lr (float): The maximum learning rate.
            step_size (int): Number of iterations for half a cycle.
        """
        super().__init__(on=on, called_every=called_every)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> None:
        """
        Adjusts the learning rate cyclically.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter): The writer object for logging.
        """
        cycle = trainer.opt_result.iteration // (2 * self.step_size)
        x = abs(trainer.opt_result.iteration / self.step_size - 2 * cycle - 1)
        scale = max(0, (1 - x))
        new_lr = self.base_lr + (self.max_lr - self.base_lr) * scale
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = new_lr


class LRSchedulerCosineAnnealing(Callback):
    """
    Applies cosine annealing to the learning rate during training.

    This callback decreases the learning rate following a cosine curve,
    starting from the initial learning rate and annealing to a minimum (min_lr).

    Example Usage in `TrainConfig`:
    To use `LRSchedulerCosineAnnealing`, include it in the `callbacks` list
    when setting up your `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    from perceptrain.callbacks import LRSchedulerCosineAnnealing

    # Create an instance of the LRSchedulerCosineAnnealing callback
    lr_cosine = LRSchedulerCosineAnnealing(on="train_batch_end",
                                           called_every=1,
                                           t_max=5000,
                                           min_lr=1e-6)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback
        callbacks=[lr_cosine]
    )
    ```
    """

    def __init__(self, on: str, called_every: int, t_max: int, min_lr: float = 0.0):
        """Initializes the LRSchedulerCosineAnnealing callback.

        Args:
            on (str): The event to trigger the callback.
            called_every (int): Frequency of callback calls in terms of iterations.
            t_max (int): The total number of iterations for one annealing cycle.
            min_lr (float, optional): The minimum learning rate. Default is 0.0.
        """
        super().__init__(on=on, called_every=called_every)
        self.t_max = t_max
        self.min_lr = min_lr

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> None:
        """
        Adjusts the learning rate using cosine annealing.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter): The writer object for logging.
        """
        for param_group in trainer.optimizer.param_groups:
            max_lr = param_group["lr"]
            new_lr = (
                self.min_lr
                + (max_lr - self.min_lr)
                * (1 + math.cos(math.pi * trainer.opt_result.iteration / self.t_max))
                / 2
            )
            param_group["lr"] = new_lr


class LRSchedulerReduceOnPlateau(Callback):
    """
    Reduces learning rate when a given metric reaches a plateau.

    This callback decreases the learning rate by a factor `gamma` when a given
    metric does not improve after a given number of epochs by more than
    a given threshold, until a minimum learning rate is reached.

    Example Usage in `TrainConfig`:
    To use `LRSchedulerReduceOnPlateau`, include it in the `callbacks` list
    when setting up your `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    from perceptrain.callbacks import LRSchedulerReduceOnPlateau

    # Create an instance of the LRSchedulerReduceOnPlateau callback
    lr_plateau = LRSchedulerReduceOnPlateau(
                    on="train_epoch_end",
                    called_every=1,
                    monitor="train_loss",
                    patience=20,
                    mode="min",
                    gamma=0.5,
                    threshold=1e-4,
                    min_lr=1e-5,
                )

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback
        callbacks=[lr_plateau]
    )
    ```
    """

    def __init__(
        self,
        on: str,
        called_every: int = 1,
        monitor: str = "train_loss",
        patience: int = 20,
        mode: str = "min",
        gamma: float = 0.5,
        threshold: float = 1e-4,
        min_lr: float = 1e-6,
        verbose: bool = True,
    ):
        """Initializes the LRSchedulerReduceOnPlateau callback.

        Args:
            on (str): The event to trigger the callback. Default is `train_epoch_end`.
            called_every (int, optional): Frequency of callback calls in terms of iterations.
                Default is 1.
            monitor (str, optional): The metric to monitor (e.g., "val_loss" or "train_loss").
                All metrics returned by optimize step are available to monitor.
                Please add "val_" and "train_" strings at the start of the metric name.
                Default is "train_loss".
            mode (str, optional): Whether to minimize ("min") or maximize ("max") the metric.
                Default is "min".
            patience (int, optional): Number of allowed iterations with no loss
                improvement before reducing the learning rate. Default is 20.
            gamma (float, optional): The decay factor applied to the learning rate. A value
                < 1 reduces the learning rate over time. Default is 0.5.
            threshold (float, optional): Amount by which the loss must improve to count as an
                improvement. Default is 1e-4.
            min_lr (float, optional): Minimum learning rate past which no further reducing
                is applied.  Default is 1e-5.
            verbose (bool, optional): If True, the logger prints when the learning rate
                decreases or reaches the minimum (INFO level)
        """
        super().__init__(on=on, called_every=called_every)
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.gamma = gamma
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose

        self.best_value = float("inf")
        self.counter = 0
        self.reached_minimum_lr = False

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> None:
        """
        Reduces the learning rate when the loss reaches a plateau.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter): The writer object for logging.
        """
        if self.reached_minimum_lr:
            pass
        else:
            current_value = trainer.opt_result.metrics.get(self.monitor)
            if current_value is None:
                raise ValueError(
                    f"Metric '{self.monitor}' is not available in the trainer's metrics."
                )

            if (self.mode == "min" and current_value + self.threshold < self.best_value) or (
                self.mode == "max" and current_value - self.threshold > self.best_value
            ):
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= self.patience:
                for param_group in trainer.optimizer.param_groups:
                    new_lr = param_group["lr"] * self.gamma
                    if new_lr < self.min_lr:
                        param_group["lr"] = self.min_lr
                        self.reached_minimum_lr = True
                        if self.verbose:
                            logger.info(
                                f"Learning rate has reached the minimum learning rate {self.min_lr}"
                            )
                    else:
                        param_group["lr"] = new_lr
                        if self.verbose:
                            logger.info(
                                f"Loss has reached a plateau, reducing learning rate to {new_lr}"
                            )
                self.counter = 0


class EarlyStopping(Callback):
    """
    Stops training when a monitored metric has not improved for a specified number of epochs.

    This callback monitors a specified metric (e.g., validation loss or accuracy). If the metric
    does not improve for a given patience period, training is stopped.

    Example Usage in `TrainConfig`:
    To use `EarlyStopping`, include it in the `callbacks` list when setting up your `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    from perceptrain.callbacks import EarlyStopping

    # Create an instance of the EarlyStopping callback
    early_stopping = EarlyStopping(on="val_epoch_end",
                                   called_every=1,
                                   monitor="val_loss",
                                   patience=5,
                                   mode="min")

    config = TrainConfig(
        max_iter=10000,
        print_every=1000,
        callbacks=[early_stopping]
    )
    ```
    """

    def __init__(
        self, on: str, called_every: int, monitor: str, patience: int = 5, mode: str = "min"
    ):
        """Initializes the EarlyStopping callback.

        Args:
            on (str): The event to trigger the callback (e.g., "val_epoch_end").
            called_every (int): Frequency of callback calls in terms of iterations.
            monitor (str): The metric to monitor (e.g., "val_loss" or "train_loss").
                All metrics returned by optimize step are available to monitor.
                Please add "val_" and "train_" strings at the start of the metric name.
            patience (int, optional): Number of iterations to wait for improvement. Default is 5.
            mode (str, optional): Whether to minimize ("min") or maximize ("max") the metric.
                Default is "min".
        """
        super().__init__(on=on, called_every=called_every)
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else -float("inf")
        self.counter = 0

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> None:
        """
        Monitors the metric and stops training if no improvement is observed.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter): The writer object for logging.
        """
        current_value = trainer.opt_result.metrics.get(self.monitor)
        if current_value is None:
            raise ValueError(f"Metric '{self.monitor}' is not available in the trainer's metrics.")

        if (self.mode == "min" and current_value < self.best_value) or (
            self.mode == "max" and current_value > self.best_value
        ):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(
                f"EarlyStopping: No improvement in '{self.monitor}' for {self.patience} epochs. "
                "Stopping training."
            )
            trainer._stop_training.fill_(1)


class GradientMonitoring(Callback):
    """
    Logs gradient statistics (e.g., mean, standard deviation, max) during training.

    This callback monitors and logs statistics about the gradients of the model parameters
    to help debug or optimize the training process.

    Example Usage in `TrainConfig`:
    To use `GradientMonitoring`, include it in the `callbacks` list when
    setting up your `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    from perceptrain.callbacks import GradientMonitoring

    # Create an instance of the GradientMonitoring callback
    gradient_monitoring = GradientMonitoring(on="train_batch_end", called_every=10)

    config = TrainConfig(
        max_iter=10000,
        print_every=1000,
        callbacks=[gradient_monitoring]
    )
    ```
    """

    def __init__(self, on: str, called_every: int = 1):
        """Initializes the GradientMonitoring callback.

        Args:
            on (str): The event to trigger the callback (e.g., "train_batch_end").
            called_every (int): Frequency of callback calls in terms of iterations.
        """
        super().__init__(on=on, called_every=called_every)

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> None:
        """
        Logs gradient statistics.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter): The writer object for logging.
        """
        if trainer.accelerator.rank == 0:
            gradient_stats = {}
            for name, param in trainer.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    gradient_stats.update(
                        {
                            name + "_mean": grad.mean().item(),
                            name + "_std": grad.std().item(),
                            name + "_max": grad.max().item(),
                            name + "_min": grad.min().item(),
                        }
                    )

            writer.write(trainer.opt_result.iteration, gradient_stats)


class R3Sampling(Callback):
    def __init__(
        self,
        initial_dataset: R3Dataset,
        fitness_function: Callable[[Tensor, nn.Module], Tensor],
        verbose: bool = False,
        called_every: int = 1,
    ):
        """Callback for R3 sampling (https://arxiv.org/abs/2207.02338#).

        Args:
            initial_dataset (R3Dataset): The dataset updating according to the R3 scheme.
            fitness_function (Callable[[Tensor, nn.Module], Tensor]): The function to
                compute fitness scores for samples. Based on the fitness scores, the
                samples are retained or released.
            verbose (bool, optional): Whether to print the callback's summary.
                Defaults to False.
            called_every (int, optional): Every how many events the callback is called.
                Defaults to 1.

        Notes:
            - R3 sampling was developed as a technique for efficient sampling of physics-informed
            neural networks (PINNs). In this case, the fitness function can be any function of the
            residuals of the equations

        Examples:
            Learning an harmonic oscillator with PINNs and R3 sampling. For a well-posed problem, also
            add the two initial conditions.

            ```python
                import torch

                m = 1.0
                k = 1.0

                def uniform_1d(n: int):
                    return torch.rand(size=(n, 1))

                def harmonic_oscillator(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
                    u = model(x)
                    dudt = torch.autograd.grad(
                        outputs=u,
                        inputs=x,
                        grad_outputs=torch.ones_like(u),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    d2udt2 = torch.autograd.grad(
                        outputs=dudt,
                        inputs=x,
                        grad_outputs=torch.ones_like(dudt),
                    )[0]
                    return m * d2udt2 - kappa * u

                def fitness_function(x: torch.Tensor, model: PINN) -> torch.Tensor:
                    return torch.linalg.vector_norm(harmonic_oscillator(x, model.nn), ord=2)

                dataset = R3Dataset(
                    proba_dist=uniform_1d,
                    n_samples=20,
                    release_threshold=1.0,
                )

                callback_r3 = R3Sampling(
                    initial_dataset=dataset,
                    fitness_function=fitness_function,
                    called_every=10,
                )
            ```
        """
        self.dataset = initial_dataset
        self.fitness_function = fitness_function
        self.verbose = verbose

        super().__init__(on="train_epoch_start", called_every=called_every)

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> None:
        """Runs the callback.

        Computes fitness scores for samples and triggers the dataset update.

        Args:
            trainer (Any): The trainer instance.
            config (TrainConfig): The training configuration.
            writer (BaseWriter): The writer instance.
        """
        # Compute fitness function on all samples
        fitnesses = self.fitness_function(self.dataset.features, trainer.model)

        # R3 update
        self.dataset.update(fitnesses)

        if self.verbose:
            print(
                f"Epoch {trainer.current_epoch};"
                f" num. retained: {self.dataset.n_samples - self.dataset.n_released:5d};"
                f" num. released: {self.dataset.n_released:5d}"
            )
