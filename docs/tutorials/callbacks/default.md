
# Built-in Callbacks

`perceptrain` offers several built-in callbacks for common tasks like saving checkpoints, logging metrics, and tracking models. Below is an overview of each.

### 1. `PrintMetrics`

Prints metrics at specified intervals.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import PrintMetrics

print_metrics_callback = PrintMetrics(on="val_batch_end", called_every=100)

config = TrainConfig(
    max_iter=10000,
    callbacks=[print_metrics_callback]
)
```

### 2. `WriteMetrics`

Writes metrics to a specified logging destination.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import WriteMetrics

write_metrics_callback = WriteMetrics(on="train_epoch_end", called_every=50)

config = TrainConfig(
    max_iter=5000,
    callbacks=[write_metrics_callback]
)
```

### 3. `PlotMetrics`

Plots metrics based on user-defined plotting functions.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import PlotMetrics

plot_metrics_callback = PlotMetrics(on="train_epoch_end", called_every=100)

config = TrainConfig(
    max_iter=5000,
    callbacks=[plot_metrics_callback]
)
```

### 4. `LogHyperparameters`

Logs hyperparameters to keep track of training settings.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import LogHyperparameters

log_hyper_callback = LogHyperparameters(on="train_start", called_every=1)

config = TrainConfig(
    max_iter=1000,
    callbacks=[log_hyper_callback]
)
```

### 5. `SaveCheckpoint`

Saves model checkpoints at specified intervals.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import SaveCheckpoint

save_checkpoint_callback = SaveCheckpoint(on="train_epoch_end", called_every=100)

config = TrainConfig(
    max_iter=10000,
    callbacks=[save_checkpoint_callback]
)
```

### 6. `SaveBestCheckpoint`

Saves the best model checkpoint based on a validation criterion.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import SaveBestCheckpoint

save_best_checkpoint_callback = SaveBestCheckpoint(on="val_epoch_end", called_every=10)

config = TrainConfig(
    max_iter=10000,
    callbacks=[save_best_checkpoint_callback]
)
```

### 7. `LoadCheckpoint`

Loads a saved model checkpoint at the start of training.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import LoadCheckpoint

load_checkpoint_callback = LoadCheckpoint(on="train_start")

config = TrainConfig(
    max_iter=10000,
    callbacks=[load_checkpoint_callback]
)
```

### 8. `LogModelTracker`

Logs the model structure and parameters.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import LogModelTracker

log_model_callback = LogModelTracker(on="train_end")

config = TrainConfig(
    max_iter=1000,
    callbacks=[log_model_callback]
)
```

### 9. `LRSchedulerStepDecay`

Reduces the learning rate by a factor at regular intervals.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import LRSchedulerStepDecay

lr_step_decay = LRSchedulerStepDecay(on="train_epoch_end", called_every=100, gamma=0.5)

config = TrainConfig(
    max_iter=10000,
    callbacks=[lr_step_decay]
)
```

### 10. `LRSchedulerCyclic`

Applies a cyclic learning rate schedule during training.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import LRSchedulerCyclic

lr_cyclic = LRSchedulerCyclic(on="train_batch_end", called_every=1, base_lr=0.001, max_lr=0.01, step_size=2000)

config = TrainConfig(
    max_iter=10000,
    callbacks=[lr_cyclic]
)
```

### 11. `LRSchedulerCosineAnnealing`

Applies cosine annealing to the learning rate during training.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import LRSchedulerCosineAnnealing

lr_cosine = LRSchedulerCosineAnnealing(on="train_batch_end", called_every=1, t_max=5000, min_lr=1e-6)

config = TrainConfig(
    max_iter=10000,
    callbacks=[lr_cosine]
)
```

### 12. `EarlyStopping`

Stops training when a monitored metric has not improved for a specified number of epochs.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import EarlyStopping

early_stopping = EarlyStopping(on="val_epoch_end", called_every=1, monitor="val_loss", patience=5, mode="min")

config = TrainConfig(
    max_iter=10000,
    callbacks=[early_stopping]
)
```

### 13. `GradientMonitoring`

Logs gradient statistics (e.g., mean, standard deviation, max) during training.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import GradientMonitoring

gradient_monitoring = GradientMonitoring(on="train_batch_end", called_every=10)

config = TrainConfig(
    max_iter=10000,
    callbacks=[gradient_monitoring]
)
```
