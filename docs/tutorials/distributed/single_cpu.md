# Single-Process CPU Training Example

Single-Process Training: Simple and suitable for small datasets. Use `backend="cpu"`.

```python exec="on" source="material-block" result="json"
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from perceptrain import TrainConfig, Trainer
from perceptrain.optimize_step import optimize_step
Trainer.set_use_grad(True)

# Dataset, Model, and Optimizer
x = torch.linspace(0, 1, 100).reshape(-1, 1)
y = torch.sin(2 * torch.pi * x)
dataloader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Single-Process Training Configuration
train_config = TrainConfig(compute_setup="cpu", max_iter=5, print_every=1)

# Training
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
