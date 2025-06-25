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

### 3. `WritePlots`

Plots metrics based on user-defined plotting functions.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import WritePlots

plot_metrics_callback = WritePlots(on="train_epoch_end", called_every=100)

config = TrainConfig(
    max_iter=5000,
    callbacks=[plot_metrics_callback]
)
```

### 3. `LivePlotMetrics`

Plots dynamically on screen the metrics followed during training. The `arrange` parameter allows for custom arrangement of subplots.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import LivePlotMetrics

live_plot_callback = LivePlotMetrics(on="train_epoch_end",
    called_every=100,
    arrange={"training": ["train_loss", "train_metric_first"], "validation": ["val_loss", "val_metric_second"]},
)

config = TrainConfig(
    max_iter=5000,
    callbacks=[live_plot_callback]
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

### 11. `LRSchedulerReduceOnPlateau`

Reduces the learning rate when a given metric does not improve for a number of epochs.

```python exec="on" source="material-block" html="1"
from perceptrain import TrainConfig
from perceptrain.callbacks import LRSchedulerReduceOnPlateau

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
    callbacks=[lr_plateau]
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

### 14. `R3Sampling`

Triggers the update of the dataset using the R3 sampling technique (ref. [here](https://arxiv.org/abs/2207.02338#)).

The following example shows how to set-up R3 Sampling to learn a harmonic oscillator with physics-informed neural networks.

```python exec="on" source="material-block" html="1"
import torch

from perceptrain import TrainConfig
from perceptrain.callbacks import R3Sampling
from perceptrain.data import R3Dataset
from perceptrain.models import PINN

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
    called_every=100,
)

config = TrainConfig(
    max_iter=1000,
    callbacks=[callback_r3]
)
```
