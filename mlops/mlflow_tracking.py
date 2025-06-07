import mlflow
from backend.core.config import config  # your custom Config class
import os

# ✅ Access nested values via `config.data`
MLFLOW_URI = config.data.mlops.mlflow.tracking_uri
EXPERIMENT_NAME = config.data.mlops.mlflow.experiment_name

def start_run(run_name=None):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run = mlflow.start_run(run_name=run_name)
    return run

def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def log_artifact(filepath: str):
    if os.path.exists(filepath):
        mlflow.log_artifact(filepath)
    else:
        print(f"Artifact not found: {filepath}")

def end_run():
    mlflow.end_run()

# ✅ Example usage
if __name__ == "__main__":
    run = start_run("Example Training Run")
    log_params({"epochs": 10, "batch_size": 32, "lr": 0.001})
    log_metrics({"accuracy": 0.95, "loss": 0.1})
    # log_artifact("path/to/model.joblib")
    end_run()
