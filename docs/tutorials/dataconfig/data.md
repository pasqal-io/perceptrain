# Dataloaders

When using Perceptrain, you can supply classical data to a quantum machine learning
algorithm by using a standard PyTorch `DataLoader` instance. Perceptrain also provides
the `DictDataLoader` convenience class which allows
to build dictionaries of `DataLoader`s instances and easily iterate over them.

```python exec="on" source="material-block" result="json"
import torch
from torch.utils.data import DataLoader, TensorDataset
from perceptrain import DictDataLoader, to_dataloader


def dataloader(data_size: int = 25, batch_size: int = 5, infinite: bool = False) -> DataLoader:
    x = torch.linspace(0, 1, data_size).reshape(-1, 1)
    y = torch.sin(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=infinite)


def dictdataloader(data_size: int = 25, batch_size: int = 5) -> DictDataLoader:
    dls = {}
    for k in ["y1", "y2"]:
        x = torch.rand(data_size, 1)
        y = torch.sin(x)
        dls[k] = to_dataloader(x, y, batch_size=batch_size, infinite=True)
    return DictDataLoader(dls)


# iterate over standard DataLoader
for (x,y) in dataloader(data_size=6, batch_size=2):
    print(f"Standard {x = }")

# construct an infinite dataset which will keep sampling indefinitely
n_epochs = 5
dl = iter(dataloader(data_size=6, batch_size=2, infinite=True))
for _ in range(n_epochs):
    (x, y) = next(dl)
    print(f"Infinite {x = }")

# iterate over DictDataLoader
ddl = dictdataloader()
data = next(iter(ddl))
print(f"{data = }")
```

Note:
    In case of `infinite`=True, the dataloader iterator will provide a random sample from the dataset.
