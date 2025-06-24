from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Callable

from torch import dtype

from perceptrain.types import ExperimentTrackingTool, LoggablePlotFunction

logger = getLogger(__file__)


@dataclass
class TrainConfig:
    """Default configuration for the training process.

    This class provides default settings for various aspects of the training loop,
    such as logging, checkpointing, and validation. The default values for these
    fields can be customized when an instance of `TrainConfig` is created.

    Example:
    ```python exec="on" source="material-block" result="json"
    from perceptrain import TrainConfig
    c = TrainConfig(root_folder="/tmp/train")
    print(str(c)) # markdown-exec: hide
    ```
    """

    max_iter: int = 10000
    """Number of training iterations (epochs) to perform.

    This defines the total number
    of times the model will be updated.

    In case of InfiniteTensorDataset, each epoch will have 1 batch.
    In case of TensorDataset, each epoch will have len(dataloader) batches.
    """

    print_every: int = 0
    """Frequency (in epochs) for printing loss and metrics to the console during training.

    Set to 0 to disable this output, meaning that metrics and loss will not be printed
    during training.
    """

    write_every: int = 0
    """Frequency (in epochs) for writing loss and metrics using the tracking tool during training.

    Set to 0 to disable this logging, which prevents metrics from being logged to the tracking tool.
    Note that the metrics will always be written at the end of training regardless of this setting.
    """

    checkpoint_every: int = 0
    """Frequency (in epochs) for saving model and optimizer checkpoints during training.

    Set to 0 to disable checkpointing. This helps in resuming training or recovering
    models.
    Note that setting checkpoint_best_only = True will disable this and only best checkpoints will
    be saved.
    """

    plot_every: int = 0
    """Frequency (in epochs) for generating and saving figures during training.

    Set to 0 to disable plotting.
    """

    live_plot_every: int = 0
    """Frequency for live plotting all the metrics in a single dynamic subplot.

    Set to 0 to disable.

    __NOTE__: for more personalized behaviour, such as showing only a subset of the
        metrics or arranging over different subplots, leave this parameter to 0,
        define a `LivePlotMetrics` callback and pass it to `callbacks`.
    """

    callbacks: list = field(default_factory=lambda: list())
    """List of callbacks to execute during training.

    Callbacks can be used for
    custom behaviors, such as early stopping, custom logging, or other actions
    triggered at specific events.
    """

    log_model: bool = False
    """Whether to log a serialized version of the model.

    When set to `True`, the
    model's state will be logged, useful for model versioning and reproducibility.
    """

    root_folder: Path = Path("./qml_logs")
    """The root folder for saving checkpoints and tensorboard logs.

    The default path is "./qml_logs"

    This can be set to a specific directory where training artifacts are to be stored.
    Checkpoints will be saved inside a subfolder in this directory. Subfolders will be
    created based on `create_subfolder_per_run` argument.
    """

    create_subfolder_per_run: bool = False
    """Whether to create a subfolder for each run, named `<id>_<timestamp>_<PID>`.

    This ensures logs and checkpoints from different runs do not overwrite each other,
    which is helpful for rapid prototyping. If `False`, training will resume from
    the latest checkpoint if one exists in the specified log folder.
    """

    log_folder: Path = Path("./")
    """The log folder for saving checkpoints and tensorboard logs.

    This stores the path where all logs and checkpoints are being saved
    for this training session. `log_folder` takes precedence over `root_folder`,
    but it is ignored if `create_subfolders_per_run=True` (in which case, subfolders
    will be spawned in the root folder).
    """

    checkpoint_best_only: bool = False
    """If `True`, checkpoints are only saved if there is an improvement in the.

    validation metric. This conserves storage by only keeping the best models.

    validation_criterion is required when this is set to True.
    """

    val_every: int = 0
    """Frequency (in epochs) for performing validation.

    If set to 0, validation is not performed.
    Note that metrics from validation are always written, regardless of the `write_every` setting.
    Note that initial validation happens at the start of training (when val_every > 0)
        For initial validation  - initial metrics are written.
                                - checkpoint is saved (when checkpoint_best_only = False)
    """

    val_epsilon: float = 1e-5
    """A small safety margin used to compare the current validation loss with the.

    best previous validation loss. This is used to determine improvements in metrics.
    """

    validation_criterion: Callable | None = None
    """A function to evaluate whether a given validation metric meets a desired condition.

    The validation_criterion has the following format:
    def validation_criterion(val_loss: float, best_val_loss: float, val_epsilon: float) -> bool:
        # process

    If `None`, no custom validation criterion is applied.
    """

    trainstop_criterion: Callable | None = None
    """A function to determine if the training process should stop based on a.

    specific stopping metric. If `None`, training continues until `max_iter` is reached.
    """

    batch_size: int = 1
    """The batch size to use when processing a list or tuple of torch.Tensors.

    This specifies how many samples are processed in each training iteration.
    """

    verbose: bool = True
    """Whether to print metrics and status messages during training.

    If `True`, detailed metrics and status updates will be displayed in the console.
    """

    tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
    """The tool used for tracking training progress and logging metrics.

    Options include tools like TensorBoard, which help visualize and monitor
    model training.
    """

    hyperparams: dict = field(default_factory=dict)
    """A dictionary of hyperparameters to be tracked.

    This can include learning rates,
    regularization parameters, or any other training-related configurations.
    """

    plotting_functions: tuple[LoggablePlotFunction, ...] = field(default_factory=tuple)  # type: ignore
    """Functions used for in-training plotting.

    These are called to generate
    plots that are logged or saved at specified intervals.
    """

    _subfolders: list[str] = field(default_factory=list)
    """List of subfolders used for logging different runs using the same config inside the.

    root folder.

    Each subfolder is of structure `<id>_<timestamp>_<PID>`.
    """

    nprocs: int = 1
    """
    The number of processes to use for training when spawning subprocesses.

    For effective parallel processing, set this to a value greater than 1.
    - In case of Multi-GPU or Multi-Node-Multi-GPU setups, nprocs should be equal to
    the total number of GPUs across all nodes (world size), or total number of GPU to be used.

    If nprocs > 1, multiple processes will be spawned for training. The training framework will launch
    additional processes (e.g., for distributed or parallel training).
    - For CPU setup, this will launch a true parallel processes
    - For GPU setup, this will launch a distributed training routine.
    This uses the DistributedDataParallel framework from PyTorch.
    """

    compute_setup: str = "cpu"
    """
    Compute device setup; options are "auto", "gpu", or "cpu".

    - "auto": Automatically uses GPU if available; otherwise, falls back to CPU.
    - "gpu": Forces GPU usage, raising an error if no CUDA device is available.
    - "cpu": Forces the use of CPU regardless of GPU availability.
    """

    backend: str = "gloo"
    """
    Backend used for distributed training communication.

    The default is "gloo". Other options may include "nccl" - which is optimized for GPU-based training or "mpi",
    depending on your system and requirements.
    It should be one of the backends supported by `torch.distributed`. For further details, please look at
    [torch backends](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend)
    """

    log_setup: str = "cpu"
    """
    Logging device setup; options are "auto" or "cpu".

    - "auto": Uses the same device for logging as for computation.
    - "cpu": Forces logging to occur on the CPU. This can be useful to avoid potential conflicts with GPU processes.
    """

    dtype: dtype | None = None
    """
    Data type (precision) for computations.

    Both model parameters, and dataset will be of the provided precision.

    If not specified or None, the default torch precision (usually torch.float32) is used.
    If provided dtype is torch.complex128, model parameters will be torch.complex128, and data parameters will be torch.float64
    """

    all_reduce_metrics: bool = False
    """
    Whether to aggregate metrics (e.g., loss, accuracy) across processes.

    When True, metrics from different training processes are averaged to provide a consolidated metrics.
    Note: Since aggregation requires synchronization/all_reduce operation, this can increase the
     computation time significantly.
    """
