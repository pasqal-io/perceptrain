import torch
from torch.optim.adam import Adam
import torch.nn as nn

from perceptrain.config import TrainConfig
from perceptrain.data import to_dataloader
from perceptrain.optimize_step import optimize_step
from perceptrain.trainer import Trainer

class Model(nn.Module):
    def __init__(self, in_features: int = 1, hidden_size: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = Model()
optimizer = Adam(model.parameters(), lr=0.001)

batch_size = 25
x = torch.linspace(0, 1, 32).reshape(-1, 1)
y = torch.sin(x)
train_loader = to_dataloader(x, y, batch_size=batch_size, infinite=True)

train_config = TrainConfig(max_iter=100)

trainer = Trainer(
    model=model,
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
