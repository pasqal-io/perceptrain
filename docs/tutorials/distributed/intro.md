# Distributed Training

`Trainer` from `perceptrain` supports distributed training across multiple CPUs and GPUs. This is achieved using the `Accelerator` provided by `perceptrain`, which can also be used to design custom distributed training loops. 

## **Configurations:**
- **`compute_setup`**: Selected training setup. (`gpu` or `auto`)
- **`backend="nccl"`**: Optimized backend for GPU training.
- **`nprocs=1`**: Uses one GPU.
```python
train_config = TrainConfig(
    compute_setup="auto",
    backend="nccl",
    nprocs=1,
)
```

## Using Accelerator with Trainer

`Accelerator` is already integrated into the `Trainer` class from `perceptrain`, and `Trainer` can automatically distribute the training process based on the configurations provided in `TrainConfig`.

```python
from perceptrain.trainer import Trainer
from perceptrain import TrainConfig

config = TrainConfig(nprocs=4)

trainer = Trainer(model, optimizer, config)
model, optimizer = trainer.fit(dataloader)
```

&nbsp;

&nbsp;

&nbsp;

&nbsp;
