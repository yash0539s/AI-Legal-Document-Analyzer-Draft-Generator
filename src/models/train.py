import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mlflow
from src.models.model import SignLanguageLSTM
import numpy as np
import os
import json

class SignDataset(Dataset):
    def __init__(self, npy_folder, labels_map):
        self.npy_folder = npy_folder
        self.labels_map = labels_map
        self.files = [f for f in os.listdir(npy_folder) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(os.path.join(self.npy_folder, file))
        label_name = file.split("_")[0]
        label = self.labels_map.get(label_name, 0)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label)

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class list
    with open(config["data"]["class_list_path"]) as f:
        class_list = json.load(f)
    labels_map = {cls: idx for idx, cls in enumerate(class_list)}

    # Validate num_classes
    config_num_classes = config["model"]["num_classes"]
    actual_num_classes = len(class_list)
    if config_num_classes != actual_num_classes:
        raise ValueError(f"[ERROR] num_classes ({config_num_classes}) in config != number of classes in class_list ({actual_num_classes})")

    # Validate input size
    input_dim = config["model"]["input_shape"][1]
    expected_dim = 21 * 3 * 2 + 33 * 3  # 225
    if input_dim != expected_dim:
        raise ValueError(f"[ERROR] input_dim ({input_dim}) does not match expected {expected_dim}. Update config['model']['input_shape'].")

    hidden_dim = config["model"]["hidden_units"]
    output_dim = actual_num_classes
    epochs = config["model"]["epochs"]
    batch_size = config["data"]["batch_size"]
    lr = config["model"]["learning_rate"]

    train_data_path = os.path.join(config["data"]["processed_data_path"], "train")
    train_dataset = SignDataset(train_data_path, labels_map)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SignLanguageLSTM(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Log hyperparameters
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if inputs.shape[-1] != input_dim:
                raise ValueError(f"[ERROR] input.size(-1) must be {input_dim}, got {inputs.shape[-1]}")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path)

    return model, None
