import os
import yaml
import json
from src.preprocessing.process_all_videos_from_json import process_videos
from src.models.train import train_model
from src.models.eval import evaluate_model
from src.mlflow.mlflow_utils import init_mlflow_run
import mlflow
import torch

def main():
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # Create artifact directories if they don't exist
    artifacts_dirs = config.get("artifacts", {})
    for key, dir_path in artifacts_dirs.items():
        os.makedirs(dir_path, exist_ok=True)

    # Initialize MLflow run
    mlflow_run = init_mlflow_run(config)

    try:
        processed_train_dir = os.path.join(config["data"]["processed_data_path"], "train")
        labels_json_path = os.path.join(processed_train_dir, "labels.json")
        npy_files_exist = False
        if os.path.exists(processed_train_dir):
            npy_files_exist = any(f.endswith(".npy") for f in os.listdir(processed_train_dir))

        if not (os.path.exists(labels_json_path) and npy_files_exist):
            print("Processed data not found. Starting preprocessing...")
            process_videos(
                raw_path=config["data"]["raw_data_path"],
                splits_json_dir=config["data"]["splits_json_dir"],
                processed_path=config["data"]["processed_data_path"],
                max_frames=config["data"]["max_frames"],
                synonyms_path=config["data"]["synonyms_path"],
                class_list_path=config["data"]["class_list_path"]
            )
        else:
            print("Processed data found. Skipping preprocessing.")

        print("Starting training...")
        model, history = train_model(config)

        # Save the trained model to artifacts/models directory
        model_save_path = os.path.join(artifacts_dirs.get("model_dir", "artifacts/models"), "best_model.pth")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure directory exists
        print(f"Saving model to {model_save_path} ...")
        torch.save(model.state_dict(), model_save_path)

        print("Starting evaluation...")
        metrics = evaluate_model(model, config)

        # Save evaluation metrics to artifacts/metrics directory
        metrics_path = os.path.join(artifacts_dirs.get("metrics_dir", "artifacts/metrics"), "eval_metrics.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)  # Ensure directory exists
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics to {metrics_path}")

        # Log metrics and artifacts to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(artifacts_dirs.get("model_dir", "artifacts/models"), artifact_path="models")
        mlflow.log_artifacts(artifacts_dirs.get("metrics_dir", "artifacts/metrics"), artifact_path="metrics")

        print("Pipeline finished successfully.")

    except Exception as e:
        print(f"Pipeline failed: {e}")

    finally:
        if mlflow.active_run():
            mlflow.end_run()

if __name__ == "__main__":
    main()
