
# Custom Callbacks

The base `Callback` class in perceptrain allows defining custom behavior that can be triggered at specified events (e.g., start of training, end of epoch). You can set parameters such as when the callback runs (`on`), frequency of execution (`called_every`), and optionally define a `callback_condition`.

### Defining Callbacks

There are two main ways to define a callback:
1. **Directly providing a function** in the `Callback` instance.
2. **Subclassing** the `Callback` class and implementing custom logic.

#### Example 1: Providing a Callback Function Directly

```python exec="on" source="material-block" html="1"
from perceptrain.callbacks import Callback

# Define a custom callback function
def custom_callback_function(trainer, config, writer):
    print("Executing custom callback.")

# Create the callback instance
custom_callback = Callback(
    on="train_end",
    callback=custom_callback_function
)
```

#### Example 2: Subclassing the Callback

```python exec="on" source="material-block" html="1"
from perceptrain.callbacks import Callback

class CustomCallback(Callback):
    def run_callback(self, trainer, config, writer):
        print("Custom behavior in run_callback method.")

# Create the subclassed callback instance
custom_callback = CustomCallback(on="train_batch_end", called_every=10)
```
