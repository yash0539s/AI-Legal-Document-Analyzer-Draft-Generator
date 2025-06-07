import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from src.models.model import SignLanguageLSTM
import numpy as np
import os
import json

class SignDatasetEval(Dataset):
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

def evaluate_model(model, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    val_data_path = os.path.join(config["data"]["processed_data_path"], "val")

    # Load class list from file and create label map
    with open(config["data"]["class_list_path"]) as f:
        class_list = json.load(f)
    labels_map = {cls: idx for idx, cls in enumerate(class_list)}

    val_dataset = SignDatasetEval(val_data_path, labels_map)
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"])

    preds, targets = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    accuracy = accuracy_score(targets, preds)
    print(f"Validation accuracy: {accuracy:.4f}")

    return {"val_accuracy": accuracy}
