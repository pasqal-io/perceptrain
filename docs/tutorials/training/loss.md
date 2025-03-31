
# Loss Functions

Users can define custom loss functions tailored to their specific tasks.
The `Trainer` accepts a `loss_fn` parameter, which should be a callable that takes the model and data as inputs and returns a tuple containing the loss tensor and a dictionary of metrics.

Example of using a custom loss function:

```python exec="on" source="material-block"
import torch
from itertools import count
cnt = count()
criterion = torch.nn.MSELoss()

def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
    next(cnt)
    x, y = data
    out = model(x)
    loss = criterion(out, y)
    return loss, {}
```

This custom loss function can be used in the trainer
```python
from perceptrain import Trainer, TrainConfig
from torch.optim import Adam

# Initialize model and optimizer
model = ...  # Define or load a quantum model here
optimizer = Adam(model.parameters(), lr=0.01)
config = TrainConfig(max_iter=100, print_every=10)

trainer = Trainer(model=model, optimizer=optimizer, config=config, loss_fn=loss_fn)
```


---
