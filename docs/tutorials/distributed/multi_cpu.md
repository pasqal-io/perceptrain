# Multi-Processing CPU Training Example

Multi-Processing Training: Best for large datasets, utilizes multiple CPU processes. Use `backend="gloo"` and set `nprocs`.

```python exec="on" source="material-block" result="json" html="1"
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from perceptrain import TrainConfig, Trainer
from perceptrain.optimize_step import optimize_step
Trainer.set_use_grad(True)

# __main__ is recommended.
if __name__ == "__main__":
    x = torch.linspace(0, 1, 100).reshape(-1, 1)
    y = torch.sin(2 * torch.pi * x)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Multi-Process Training Configuration
    train_config = TrainConfig(
        compute_setup="cpu",
        backend="gloo",
        nprocs=4,
        max_iter=5,
        print_every=1)

    trainer = Trainer(model, optimizer, train_config, loss_fn="mse", optimize_step=optimize_step)
    trainer.fit(dataloader)
```

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
