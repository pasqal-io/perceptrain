## 1. Adding Callbacks to `TrainConfig`

To use callbacks in `TrainConfig`, add them to the `callbacks` list when configuring the training process.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import SaveCheckpoint, PrintMetrics

config = TrainConfig(
    max_iter=10000,
    callbacks=[
        SaveCheckpoint(on="val_epoch_end", called_every=50),
        PrintMetrics(on="train_epoch_end", called_every=100),
    ]
)
```

## 2. Using Callbacks with `Trainer`

The `Trainer` class in `perceptrain` provides built-in support for executing callbacks at various stages in the training process, managed through a callback manager. By default, several callbacks are added to specific hooks to automate common tasks, such as check-pointing, metric logging, and model tracking.

### Default Callbacks

Below is a list of the default callbacks and their assigned hooks:

- **`train_start`**: `TrackPlots`, `SaveCheckpoint`, `WriteMetrics`
- **`train_epoch_end`**: `SaveCheckpoint`, `PrintMetrics`, `TrackPlots`, `WriteMetrics`
- **`val_epoch_end`**: `SaveBestCheckpoint`, `WriteMetrics`
- **`train_end`**: `LogHyperparameters`, `LogModelTracker`, `WriteMetrics`, `SaveCheckpoint`, `TrackPlots`

These defaults handle common needs, but you can also add custom callbacks to any hook.

### Example: Adding a Custom Callback

To create a custom `Trainer` that includes a `PrintMetrics` callback executed specifically at the end of each epoch, follow the steps below.

```python exec="on" source="material-block" html="1"
from perceptrain.trainer import Trainer
from perceptrain.callbacks import PrintMetrics

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_metrics_callback = PrintMetrics(on="train_epoch_end", called_every = 10)

    def on_train_epoch_end(self, train_epoch_loss_metrics):
        self.print_metrics_callback.run_callback(self)
```
