# Loss Functions

Perceptrain works with loss functions having the following interface:

```python exec="on" source="material-block"
def loss_fn(batch: TBatch, model: torch.nn.Module) -> tuple[torch.Tensor, dict]:
    ...
```

Therefore, loss functions from [`torch.nn`](https://docs.pytorch.org/docs/stable/nn.html) won't work out of the box.

Users can either choose a built-in loss function or define a custom one.

## Built-in loss functions

- [`mse_loss`][perceptrain.loss.loss.mse_loss]: wrapper around [`torch.nn.MSELoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss).
- [`cross_entropy_loss`][perceptrain.loss.loss.cross_entropy_loss]: wrapper around [`torch.nn.CrossEntropyLoss()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss).
- [`GradWeigthedLoss`][perceptrain.loss.loss.GradWeightedLoss]: dynamic loss function that implements the learning rate annealing algorithm introduced [here](https://arxiv.org/abs/2001.04536). It allows to redistribute weight on those metrics with smaller gradients, by updating the weight by the ratio of maximum derivative of a fixed metric and the mean partial derivative value of the metric being re-weighted. This trick can prevent falling into trivial local minima.

## Custom loss functions

Users can define custom loss functions tailored to their specific tasks.
The `Trainer` accepts a `loss_fn` parameter, which should be a callable that takes the data batch and the model as inputs and returns a tuple containing the loss tensor and a dictionary of metrics.

Example of using a custom loss function:

```python exec="on" source="material-block"
import torch
from itertools import count
cnt = count()
criterion = torch.nn.MSELoss()

def loss_fn_custom(batch: TBatch, model: torch.nn.Module) -> tuple[torch.Tensor, dict]:
    next(cnt)
    x, y = batch
    out = model(x)
    loss = criterion(out, y)
    return loss, {}
```

The custom loss function can be used in the trainer

```python
from perceptrain import Trainer, TrainConfig
from torch.optim import Adam

# Initialize model and optimizer
model = ...  # Define or load a model here
optimizer = Adam(model.parameters(), lr=0.01)
config = TrainConfig(max_iter=100, print_every=10)

trainer = Trainer(model=model, optimizer=optimizer, config=config, loss_fn=loss_fn_custom)
```

**NOTE**: when working with custom loss functions, you must make sure that type of your data batch is compatible with the model. For instance, `batch` is a `Tensor` if `model` is a [`FFNN`][perceptrain.models.FFNN] (feed-forward neural network), but it must be a `dict[str, Tensor]` if `model` is a [`PINN`][perceptrain.models.PINN] (physics-informed neural network).

---
