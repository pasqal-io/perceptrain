# Experiment tracking with mlflow

perceptrain allows to track runs and log hyperparameters, models and plots with [tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) and [mlflow](https://mlflow.org/). In the following, we demonstrate the integration with mlflow.

### mlflow configuration
We have control over our tracking configuration by setting environment variables. First, let's look at the tracking URI. For the purpose of this demo we will be working with a local database, in a similar fashion as described [here](https://mlflow.org/docs/latest/tracking/tutorials/local-database.html),
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
```

perceptrain can also read the following two environment variables to define the mlflow experiment name and run name
```bash
export MLFLOW_EXPERIMENT=test_experiment
export MLFLOW_RUN_NAME=run_0
```

If no tracking URI is provided, mlflow stores run information and artifacts in the local `./mlflow` directory and if no names are defined, the experiment and run will be named with random UUIDs.
