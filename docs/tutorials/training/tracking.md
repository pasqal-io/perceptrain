# Experiment Tracking with TensorBoard and MLflow

The `Trainer` integrates with TensorBoard and MLflow for experiment tracking:

- **TensorBoard**: Logs metrics and visualizations during training, allowing users to monitor the training process.

- **MLflow**: Tracks experiments, logs parameters, metrics, and artifacts, and provides a user-friendly interface for comparing different runs.

To utilize these tracking tools, the `Trainer` can be configured with appropriate writers that handle the logging of metrics and other relevant information during training.

Example of using TensorBoard tracking:

```python
from perceptrain import TrainConfig
from perceptrain.types import ExperimentTrackingTool

# Set up tracking with TensorBoard
config = TrainConfig(max_iter=100, tracking_tool=ExperimentTrackingTool.TENSORBOARD)
```

Example of using MLflow tracking:

```python
from perceptrain.types import ExperimentTrackingTool

# Set up tracking with MLflow
config = TrainConfig(max_iter=100, tracking_tool=ExperimentTrackingTool.MLFLOW)
```

