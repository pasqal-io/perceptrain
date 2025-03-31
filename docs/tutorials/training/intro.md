
# Perceptrain Trainer Guide

The [`Trainer`][perceptrain.Trainer] class in `perceptrain` is a versatile tool designed to streamline the training of quantum machine learning models.
It offers flexibility for both gradient-based and gradient-free optimization methods, supports custom loss functions, and integrates seamlessly with tracking tools like TensorBoard and MLflow.
Additionally, it provides hooks for implementing custom behaviors during the training process.

---

## Overview

The `Trainer` class simplifies the training workflow by managing the training loop, handling data loading, and facilitating model evaluation.
It is compatible with various optimization strategies and allows for extensive customization to meet specific training requirements.

Example of initializing the `Trainer`:

```python
from perceptrain import Trainer, TrainConfig
from torch.optim import Adam

# Initialize model and optimizer
model = ...  # Define or load a quantum model here
optimizer = Adam(model.parameters(), lr=0.01)
config = TrainConfig(max_iter=100, print_every=10)

# Initialize Trainer with model, optimizer, and configuration
trainer = Trainer(model=model, optimizer=optimizer, config=config)
```
<!--
> Notes:
> `perceptrain` versions prior to 1.9.0 provided `train_with_grad` and `train_no_grad` functions, which are being replaced with `Trainer`. The user can transition as following.
> ```python
> from perceptrain import train_with_grad
> train_with_grad(model=model, optimizer=optimizer, config=config, data = data)
> ```
> to
> ```python
> from perceptrain import Trainer
> trainer = Trainer(model=model, optimizer=optimizer, config=config)
> trainer.fit(train_dataloader = data)
> ```
-->
