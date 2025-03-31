# Training on GPU with `Trainer`

This guide explains how to train models on **GPU** using `Trainer` from `perceptrain`, covering **single-GPU**, **multi-GPU (single node)**, and **multi-node multi-GPU** setups.

### Understanding Arguments
- *nprocs*: Number of processes to run. To enable multi-processing and launch separate processes, set nprocs > 1.
- *compute_setup*: The computational setup used for training. Options include `cpu`, `gpu`, and `auto`.

For more details on the advanced training options, please refer to [TrainConfig Documentation](./data_and_config.md)

## **Configuring `TrainConfig` for GPU Training**
By adjusting `TrainConfig`, you can switch between single and multi-GPU training setups. Below are the key settings for each configuration:

### **Single-GPU Training Configuration:**
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

### **Multi-GPU (Single Node) Training Configuration:**
- **`compute_setup`**: Selected training setup. (`gpu` or `auto`)
- **`backend="nccl"`**: Multi-GPU optimized backend.
- **`nprocs=2`**: Utilizes 2 GPUs on a single node.
```python
train_config = TrainConfig(
    compute_setup="auto",
    backend="nccl",
    nprocs=2,
)
```

### **Multi-Node Multi-GPU Training Configuration:**
- **`compute_setup`**: Selected training setup. (`gpu` or `auto`)
- **`backend="nccl"`**: Required for multi-node setups.
- **`nprocs=4`**: Uses 4 GPUs across nodes.
```python
train_config = TrainConfig(
    compute_setup="auto",
    backend="nccl",
    nprocs=4,
)
```
---

## Examples

The next sections provide Python scripts and training approach scripts for each setup.

> Some organizations use [SLURM](https://slurm.schedmd.com) to manage resources. Slurm is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. If you are using slurm, you can use the `Trainer` by submitting a batch script using sbatch.

> You can also use `torchrun` to run the training process - which provides a superset of the functionality as `torch.distributed.launch `. Here you need to specify the [torchrun arguments](https://pytorch.org/docs/stable/elastic/run.html) arguments to set up the distributed training setup. We also include the `torchrun` sbatch scripts for each setup below.
