# Callbacks for Trainer

Perceptrain provides a powerful callback system for customizing various stages of the training process. With callbacks, you can monitor, log, save, and alter your training workflow efficiently. A `CallbackManager` is used with [`Trainer`][perceptrain.Trainer] to execute the training process with defined callbacks. Following default callbacks are already provided in the [`Trainer`][perceptrain.Trainer].

### Default Callbacks

Below is a list of the default callbacks already implemented in the `CallbackManager` used with [`Trainer`][perceptrain.Trainer]:

- **`train_start`**: `TrackPlots`, `SaveCheckpoint`, `WriteMetrics`
- **`train_epoch_end`**: `SaveCheckpoint`, `PrintMetrics`, `TrackPlots`, `WriteMetrics`
- **`val_epoch_end`**: `SaveBestCheckpoint`, `WriteMetrics`
- **`train_end`**: `LogHyperparameters`, `LogModelTracker`, `WriteMetrics`, `SaveCheckpoint`, `TrackPlots`

This guide covers how to define and use callbacks in `TrainConfig`, integrate them with the `Trainer` class, and create custom callbacks using hooks.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
