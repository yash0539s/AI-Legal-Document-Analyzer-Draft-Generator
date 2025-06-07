import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json

class KeypointDataset(Dataset):
    def __init__(self, processed_data_path, split="train", config=None):
        self.processed_data_path = processed_data_path
        self.split = split
        self.config = config

        # Load the JSON file describing video samples and labels
        json_path = os.path.join(config["data"]["raw_data_path"], f"{split}.json")
        with open(json_path, "r") as f:
            self.video_list = json.load(f)

        self.samples = []
        for entry in self.video_list:
            video_npy_path = os.path.join(self.processed_data_path, entry["video_path"] + ".npy")
            if os.path.exists(video_npy_path):
                self.samples.append((video_npy_path, entry["label"]))
        
        # Build label mapping from class.json
        class_json_path = os.path.join(config["data"]["raw_data_path"], "class.json")
        with open(class_json_path, "r") as f:
            self.label_map = json.load(f)
        
        self.label2idx = {label: idx for idx, label in enumerate(self.label_map)}
        self.input_size = np.load(self.samples[0][0]).shape[1]
        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        keypoints = np.load(npy_path)
        label_idx = self.label2idx[label]
        # Return (sequence_length, features) tensor and label index
        return torch.tensor(keypoints), torch.tensor(label_idx)
