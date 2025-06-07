import json

def load_label_map(path):
    with open(path, 'r') as f:
        labels = json.load(f)
    return labels  # expecting list of class labels
