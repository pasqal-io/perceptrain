# Training on CPU with `Trainer`

This guide explains how to train models on **CPU** using `Trainer` from `perceptrain`, covering **single-process** and **multi-processing** setups.

### Understanding Arguments
- *nprocs*: Number of processes to run. To enable multi-processing and launch separate processes, set nprocs > 1.
- *compute_setup*: The computational setup used for training. Options include `cpu`, `gpu`, and `auto`.

For more details on the advanced training options, please refer to [TrainConfig Documentation](./data_and_config.md)

## **Configuring `TrainConfig` for CPU Training**

By adjusting `TrainConfig`, you can seamlessly switch between single and multi-core CPU training. To enable CPU-based training, update these fields in `TrainConfig`:

### Single-Process Training Configuration:
- **`backend="cpu"`**: Ensures training runs on the CPU.
- **`nprocs=1`**: Uses one CPU core.

```python
train_config = TrainConfig(
    compute_setup="cpu",
)
```

### Multi-Processing Configuration
- **`backend="gloo"`**: Uses the Gloo backend for CPU multi-processing.
- **`nprocs=4`**: Utilizes 4 CPU cores.

```python
train_config = TrainConfig(
    compute_setup="cpu",
    backend="gloo",
    nprocs=4,
)
```
