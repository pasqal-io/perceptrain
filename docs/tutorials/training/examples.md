# Examples

### 1. Training with `Trainer` and `TrainConfig`

#### Setup
Let's do the necessary imports and declare a `DataLoader`. We can already define some hyperparameters here, including the seed for random number generators. mlflow can log hyperparameters with arbitrary types, for example the observable that we want to monitor (`Z` in this case, which has a `perceptrain.Operation` type).

```python
import random
from itertools import count

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.nn import Module
from torch.utils.data import DataLoader

from perceptrain import hea, QuantumCircuit, Z
from perceptrain.constructors import feature_map, hamiltonian_factory
from perceptrain import Trainer, TrainConfig
from perceptrain.data import to_dataloader
from perceptrain.utils import rand_featureparameters
from perceptrain import QNN, QuantumModel
from perceptrain.types import ExperimentTrackingTool

hyperparams = {
    "seed": 42,
    "batch_size": 10,
    "n_qubits": 2,
    "ansatz_depth": 1,
    "observable": Z,
}

np.random.seed(hyperparams["seed"])
torch.manual_seed(hyperparams["seed"])
random.seed(hyperparams["seed"])


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.cos(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)
```

We continue with the regular QNN definition, together with the loss function and optimizer.

```python
obs = hamiltonian_factory(register=hyperparams["n_qubits"], detuning=hyperparams["observable"])

data = dataloader(hyperparams["batch_size"])
fm = feature_map(hyperparams["n_qubits"], param="x")

model = QNN(
    QuantumCircuit(
        hyperparams["n_qubits"], fm, hea(hyperparams["n_qubits"], hyperparams["ansatz_depth"])
    ),
    observable=obs,
    inputs=["x"],
)

cnt = count()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

inputs = rand_featureparameters(model, 1)

def loss_fn(model: QuantumModel, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
    next(cnt)
    out = model.expectation(inputs)
    loss = criterion(out, torch.rand(1))
    return loss, {}
```

#### `TrainConfig` specifications
perceptrain offers different tracking options via `TrainConfig`. Here we use the `ExperimentTrackingTool` type to specify that we want to track the experiment with mlflow. Tracking with tensorboard is also possible. We can then indicate *what* and *how often* we want to track or log.

**For Training**
`write_every` controls the number of epochs after which the loss values is logged. Thanks to the `plotting_functions` and `plot_every`arguments, we are also able to plot model-related quantities throughout training. Notice that arbitrary plotting functions can be passed, as long as the signature is the same as `plot_fn` below. Finally, the trained model can be logged by setting `log_model=True`. Here is an example of plotting function and training configuration

```python
def plot_fn(model: Module, iteration: int) -> tuple[str, Figure]:
    descr = f"ufa_prediction_epoch_{iteration}.png"
    fig, ax = plt.subplots()
    x = torch.linspace(0, 1, 100).reshape(-1, 1)
    out = model.expectation(x)
    ax.plot(x.detach().numpy(), out.detach().numpy())
    return descr, fig


config = TrainConfig(
    root_folder="mlflow_demonstration",
    max_iter=10,
    checkpoint_every=1,
    plot_every=2,
    write_every=1,
    log_model=True,
    tracking_tool=ExperimentTrackingTool.MLFLOW,
    hyperparams=hyperparams,
    plotting_functions=(plot_fn,),
)
```

#### Training and inspecting
Model training happens as usual
```python
trainer = Trainer(model, optimizer, config, loss_fn)
trainer.fit(train_dataloader=data)
```

After training , we can inspect our experiment via the mlflow UI
```bash
mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
```
In this case, since we're running on a local server, we can access the mlflow UI by navigating to http://localhost:8080/.


### 2. Fitting a function with a QNN using `Trainer`

In Quantum Machine Learning, the general consensus is to use `complex128` precision for states and operators and `float64` precision for parameters. This is also the convention which is used in `perceptrain`.
However, for specific usecases, lower precision can greatly speed up training and reduce memory consumption. When using the `pyqtorch` backend, `perceptrain` offers the option to move a `QuantumModel` instance to a specific precision using the torch `to` syntax.

Let's look at a complete example of how to use `Trainer` now. Here we perform a validation check during training and use a validation criterion that checks whether the validation loss in the current iteration has decreased compared to the lowest validation loss from all previous iterations. For demonstration, the train and the validation data are kept the same here. However, it is beneficial and encouraged to keep them distinct in practice to understand model's generalization capabilities.

```python exec="on" source="material-block" html="1"
from pathlib import Path
import torch
from functools import reduce
from operator import add
from itertools import count
import matplotlib.pyplot as plt

from perceptrain import Parameter, QuantumCircuit, Z
from perceptrain import hamiltonian_factory, hea, feature_map, chain
from perceptrain import QNN
from perceptrain import  TrainConfig, Trainer, to_dataloader

Trainer.set_use_grad(True)

n_qubits = 4
fm = feature_map(n_qubits)
ansatz = hea(n_qubits=n_qubits, depth=3)
observable = hamiltonian_factory(n_qubits, detuning=Z)
circuit = QuantumCircuit(n_qubits, fm, ansatz)

model = QNN(circuit, observable, backend="pyqtorch", diff_mode="ad")
batch_size = 100
input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
pred = model(input_values)

cnt = count()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
    next(cnt)
    x, y = data[0], data[1]
    out = model(x)
    loss = criterion(out, y)
    return loss, {}

def validation_criterion(
    current_validation_loss: float, current_best_validation_loss: float, val_epsilon: float
) -> bool:
    return current_validation_loss <= current_best_validation_loss - val_epsilon

n_epochs = 300

config = TrainConfig(
    max_iter=n_epochs,
    batch_size=batch_size,
    checkpoint_best_only=True,
    val_every=10,  # The model will be run on the validation data after every `val_every` epochs.
    validation_criterion=validation_criterion
)

fn = lambda x, degree: .05 * reduce(add, (torch.cos(i*x) + torch.sin(i*x) for i in range(degree)), 0.)
x = torch.linspace(0, 10, batch_size).reshape(-1, 1)
y = fn(x, 5)

train_dataloader = to_dataloader(x, y, batch_size=batch_size, infinite=True)
val_dataloader =  to_dataloader(x, y, batch_size=batch_size, infinite=True)

trainer = Trainer(model, optimizer, config, loss_fn=loss_fn,
                    train_dataloader=train_dataloader, val_dataloader=val_dataloader)
trainer.fit()

plt.clf()
plt.plot(x.numpy(), y.numpy(), label='truth')
plt.plot(x.numpy(), model(x).detach().numpy(), "--", label="final", linewidth=3)
plt.legend()
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```


### 3. Fitting a function - Low-level API

For users who want to use the low-level API of `perceptrain`, here an example written without `Trainer`.

```python exec="on" source="material-block"
from pathlib import Path
import torch
from itertools import count
from perceptrain.constructors import hamiltonian_factory, hea, feature_map
from perceptrain import chain, Parameter, QuantumCircuit, Z
from perceptrain import QNN
from perceptrain import TrainConfig

n_qubits = 2
fm = feature_map(n_qubits)
ansatz = hea(n_qubits=n_qubits, depth=3)
observable = hamiltonian_factory(n_qubits, detuning=Z)
circuit = QuantumCircuit(n_qubits, fm, ansatz)

model = QNN(circuit, observable, backend="pyqtorch", diff_mode="ad")
batch_size = 1
input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
pred = model(input_values)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
n_epochs=50
cnt = count()

tmp_path = Path("/tmp")

config = TrainConfig(
    root_folder=tmp_path,
    max_iter=n_epochs,
    checkpoint_every=100,
    write_every=100,
    batch_size=batch_size,
)

x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
y = torch.sin(x)

for i in range(n_epochs):
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
```



### 4. Performing pre-training Exploratory Landscape Analysis (ELA) with Information Content (IC)

Before one embarks on training a model, one may wish to analyze the loss landscape to judge the trainability and catch vanishing gradient issues early.
One way of doing this is made possible via calculating the [Information Content of the loss landscape](https://www.nature.com/articles/s41534-024-00819-8).
This is done by discretizing the gradient in the loss landscapes and then calculating the information content therein.
This serves as a measure of flatness or ruggedness of the loss landscape.
Quantitatively, the information content allows us to get bounds on the average norm of the gradient in the loss landscape.

Using the information content technique, we can get two types of bounds on the average of the norm of the gradient.
1. The bounds as achieved in the maximum Information Content regime: Gives us a lower and upper bound on the average norm of the gradient in case high Information Content is achieved.
2. The bounds as achieved in the sensitivity regime: Gives us an upper bound on the average norm of the gradient corresponding to the sensitivity IC achieved.

Thus, we get 3 bounds. The upper and lower bounds for the maximum IC and the upper bound for the sensitivity IC.

The `Trainer` class provides a method to calculate these gradient norms.

```python exec="on" source="material-block" html="1"
import torch
from torch.optim.adam import Adam

from perceptrain.constructors import ObservableConfig
from perceptrain.config import AnsatzConfig, FeatureMapConfig, TrainConfig
from perceptrain.data import to_dataloader
from perceptrain import QNN
from perceptrain.optimize_step import optimize_step
from perceptrain.trainer import Trainer
from perceptrain.operations.primitive import Z

fm_config = FeatureMapConfig(num_features=1)
ansatz_config = AnsatzConfig(depth=4)
obs_config = ObservableConfig(detuning=Z)

qnn = QNN.from_configs(
    register=4,
    obs_config=obs_config,
    fm_config=fm_config,
    ansatz_config=ansatz_config,
)

optimizer = Adam(qnn.parameters(), lr=0.001)

batch_size = 25
x = torch.linspace(0, 1, 32).reshape(-1, 1)
y = torch.sin(x)
train_loader = to_dataloader(x, y, batch_size=batch_size, infinite=True)

train_config = TrainConfig(max_iter=100)

trainer = Trainer(
    model=qnn,
    optimizer=optimizer,
    config=train_config,
    loss_fn="mse",
    train_dataloader=train_loader,
    optimize_step=optimize_step,
)

# Perform exploratory landscape analysis with Information Content
ic_sensitivity_threshold = 1e-4
epsilons = torch.logspace(-2, 2, 10)

max_ic_lower_bound, max_ic_upper_bound, sensitivity_ic_upper_bound = (
    trainer.get_ic_grad_bounds(
        eta=ic_sensitivity_threshold,
        epsilons=epsilons,
    )
)

print(
    f"Using maximum IC, the gradients are bound between {max_ic_lower_bound:.3f} and {max_ic_upper_bound:.3f}\n"
)
print(
    f"Using sensitivity IC, the gradients are bounded above by {sensitivity_ic_upper_bound:.3f}"
)

# Resume training as usual...

trainer.fit(train_loader)
```

The `get_ic_grad_bounds` function returns a tuple containing a tuple containing the lower bound as achieved in maximum IC case, upper bound as achieved in maximum IC case, and the upper bound for the sensitivity IC case.

The sensitivity IC bound is guaranteed to appear, while the usually much tighter bounds that we get via the maximum IC case is only meaningful in the case of the maximum achieved information content $H(\epsilon)_{max} \geq log_6(2)$.



### 5. Custom `train` loop

If you need custom training functionality that goes beyond what is available in
`perceptrain.Trainer` you can write your own
training loop based on the building blocks that are available in perceptrain.

A simplified version of perceptrain's train loop is defined below. Feel free to copy it and modify at
will.

For logging we can use the `get_writer` from the `Writer Registry`. This will set up the default writer based on the experiment tracking tool.
All writers from the `Writer Registry` offer `open`, `close`, `print_metrics`, `write_metrics`, `plot_metrics`, etc methods.


```python
from typing import Callable, Union

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from perceptrain.config import TrainConfig
from perceptrain.data import DictDataLoader, data_to_device
from perceptrain.optimize_step import optimize_step
from perceptrain.callbacks import get_writer
from perceptrain.callbacks.saveload import load_checkpoint, write_checkpoint


def train(
    model: Module,
    data: DataLoader,
    optimizer: Optimizer,
    config: TrainConfig,
    loss_fn: Callable,
    device: str = "cpu",
    optimize_step: Callable = optimize_step,
    write_tensorboard: Callable = write_tensorboard,
) -> tuple[Module, Optimizer]:

    # Move model to device before optimizer is loaded
    model = model.to(device)

    # load available checkpoint
    init_iter = 0
    if config.log_folder:
        model, optimizer, init_iter = load_checkpoint(config.log_folder, model, optimizer)

    # Initialize writer based on the tracking tool specified in the configuration
    writer = get_writer(config.tracking_tool)  # Uses ExperimentTrackingTool to select writer
    writer.open(config, iteration=init_iter)

    dl_iter = iter(dataloader)

    # outer epoch loop
    for iteration in range(init_iter, init_iter + config.max_iter):
        data = data_to_device(next(dl_iter), device)
        loss, metrics = optimize_step(model, optimizer, loss_fn, data)

        if iteration % config.print_every == 0 and config.verbose:
            writer.print_metrics(OptimizeResult(iteration, model, optimizer, loss, metrics))

        if iteration % config.write_every == 0:
            writer.write(iteration, metrics)

        if config.log_folder:
            if iteration % config.checkpoint_every == 0:
                write_checkpoint(config.log_folder, model, optimizer, iteration)

    # Final writing and checkpointing
    if config.log_folder:
        write_checkpoint(config.log_folder, model, optimizer, iteration)
    writer.write(iteration,metrics)
    writer.close()

    return model, optimizer
```
