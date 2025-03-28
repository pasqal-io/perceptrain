# Hooks for Custom Behavior

The `Trainer` class provides several hooks that enable users to inject custom behavior at different stages of the training process.
These hooks are methods that can be overridden in a subclass to execute custom code.
The available hooks include:

- `on_train_start`: Called at the beginning of the training process.
- `on_train_end`: Called at the end of the training process.
- `on_train_epoch_start`: Called at the start of each training epoch.
- `on_train_epoch_end`: Called at the end of each training epoch.
- `on_train_batch_start`: Called at the start of each training batch.
- `on_train_batch_end`: Called at the end of each training batch.

Each "start" and "end" hook receives data and loss metrics as arguments. The specific values provided for these arguments depend on the training stage associated with the hook. The context of the training stage (e.g., training, validation, or testing) determines which metrics are relevant and how they are populated. For details of inputs on each hook, please review the documentation of [`BaseTrainer`][perceptrain.train_utils.BaseTrainer].

    - Example of what inputs are provided to training hooks.

        ```
        def on_train_batch_start(self, batch: Tuple[torch.Tensor, ...] | None) -> None:
            """
            Called at the start of each training batch.

            Args:
                batch: A batch of data from the DataLoader. Typically a tuple containing
                    input tensors and corresponding target tensors.
            """
            pass
        ```
        ```
        def on_train_batch_end(self, train_batch_loss_metrics: Tuple[torch.Tensor, Any]) -> None:
            """
            Called at the end of each training batch.

            Args:
                train_batch_loss_metrics: Metrics for the training batch loss.
                    Tuple of (loss, metrics)
            """
            pass
        ```

Example of using a hook to log a message at the end of each epoch:

```python exec="on" source="material-block"
from perceptrain import Trainer

class CustomTrainer(Trainer):
    def on_train_epoch_end(self, train_epoch_loss_metrics):
        print(f"End of epoch - Loss and Metrics: {train_epoch_loss_metrics}")
```

> Notes:
> Trainer offers inbuilt callbacks as well. Callbacks are mainly for logging/tracking purposes, but the above mentioned hooks are generic. The workflow for every train batch looks like:
> 1. perform on_train_batch_start callbacks,
> 2. call the on_train_batch_start hook,
> 3. do the batch training,
> 4. call the on_train_batch_end hook, and
> 5. perform on_train_batch_end callbacks.
>
> The use of `on_`*{phase}*`_start` and `on_`*{phase}*`_end` hooks is not specifically to add extra callbacks, but for any other generic pre/post processing. For example, reshaping input batch in case of RNNs/LSTMs, post processing loss and adding an extra metric. They could also be used to add more callbacks (which is not recommended - as we provide methods to add extra callbacks in the TrainCofig)

---
