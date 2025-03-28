# Multi GPU Training

The following section provide Python scripts and training approach scripts for Multi GPU Training.


> Some organizations use [SLURM](https://slurm.schedmd.com) to manage resources. Slurm is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. If you are using slurm, you can use the `Trainer` by submitting a batch script using sbatch. Further below, we also provide the sbatch scripts for each setup.

> You can also use `torchrun` to run the training process - which provides a superset of the functionality as `torch.distributed.launch `. Here you need to specify the [torchrun arguments](https://pytorch.org/docs/stable/elastic/run.html) arguments to set up the distributed training setup. We also include the `torchrun` sbatch scripts for each setup below.

## Example Training Script (`train.py`):

We are going to use the following training script for the examples below.
**Python Script:**
```python
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from perceptrain import TrainConfig, Trainer
from perceptrain.optimize_step import optimize_step
Trainer.set_use_grad(True)

# __main__ is recommended.
if __name__ == "__main__":
    # simple dataset for y = 2πx
    x = torch.linspace(0, 1, 100).reshape(-1, 1)
    y = torch.sin(2 * torch.pi * x)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    # Simple model with no hidden layer and ReLU activation to fit the data for y = 2πx
    model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    # SGD optimizer with 0.01 learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # TrainConfig
    parser = argparse.ArgumentParser()
    parser.add_argument("--nprocs", type=int,
                        default=1, help="Number of processes (GPUs) to use.")
    parser.add_argument("--compute_setup", type=str,
                        default="auto", choices=["cpu", "gpu", "auto"], help="Computational Setup.")
    parser.add_argument("--backend", type=str,
                        default="nccl", choices=["nccl", "gloo", "mpi"], help="Distributed backend.")
    args = parser.parse_args()
    train_config = TrainConfig(
                                backend=args.backend,
                                nprocs=args.nprocs,
                                compute_setup=args.compute_setup,
                                print_every=5,
                                max_iter=50
                            )

    trainer = Trainer(model, optimizer, train_config, loss_fn="mse", optimize_step=optimize_step)
    trainer.fit(dataloader)
```

---

## Multi-GPU (Single Node):

For high performance using multiple GPUs in one node.
- *Assuming that you have 1 node with 2 GPU. These numbers can be changed depending on user needs.*

You can train by simply calling this on the head node.
```bash
python3 train.py --backend nccl --nprocs 2
```

#### SLURM
Slurm can be used to train the model but also to dispatch the workload on multiple GPUs or CPUs.
- Here, we should have one task per gpu. i.e. `ntasks` is equal to the number of nodes
- `nprocs` should be equal to the total number of gpus (world_size). which is this case is 2.

```bash
#!/bin/bash
#SBATCH --job-name=multi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

srun python3 train.py --backend nccl --nprocs 2
```

#### TORCHRUN
Torchrun takes care of setting the `nprocs` based on the cluster setup. We only need to specify to use the `compute_setup`, which can be either `auto` or `gpu`.
- `nnodes` for torchrun should be the number of nodes
- `nproc_per_node` should be equal to the number of GPUs per node.

> Note: We use the first node of the allocated resources on the cluster as the head node. However, any other node can also be chosen.
```bash
#!/bin/bash
#SBATCH --job-name=multi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
export LOGLEVEL=INFO

srun torchrun \
--nnodes 1 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29501 \
train.py --compute_setup auto
```
