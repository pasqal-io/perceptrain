# Gradient-Based and Gradient-Free Optimization

The `Trainer` supports both gradient-based and gradient-free optimization methods.
Default is gradient-based optimization.

## Using Context Managers for Mixed Optimization

For cases requiring both optimization methods in a single training session, the `Trainer` class provides context managers to enable or disable gradients.

```python
# Temporarily switch to gradient-based optimization
with trainer.enable_grad_opt(optimizer):
    print("Gradient Based Optimization")
    # trainer.fit(train_loader)

# Switch to gradient-free optimization for specific steps
with trainer.disable_grad_opt(ng_optimizer):
    print("Gradient Free Optimization")
    # trainer.fit(train_loader)
```

## Using set_grad for optimization type

We can achieve gradient free optimization with `Trainer.set_use_grad(False)` or `trainer.disable_grad_opt(ng_optimizer)`. For example, while solving a problem using gradient free optimization based on `Nevergrad` optimizers and `Trainer`.

- **Gradient-Based Optimization**: Utilizes optimizers from PyTorch's `torch.optim` module.
This is the default behaviour of the `Trainer`, thus setting this is not necessary.
However, it can be explicity mentioned as follows.
Example of using gradient-based optimization:

```python exec="on" source="material-block"
from perceptrain import Trainer

# set_use_grad(True) to enable gradient based training. This is the default behaviour of Trainer.
Trainer.set_use_grad(True)
```

- **Gradient-Free Optimization**: Employs optimization algorithms from the [Nevergrad](https://facebookresearch.github.io/nevergrad/) library.


Example of using gradient-free optimization with Nevergrad:

```python exec="on" source="material-block"
from perceptrain import Trainer

# set_use_grad(False) to disable gradient based training.
Trainer.set_use_grad(False)
```



---
