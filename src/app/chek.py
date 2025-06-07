import torch
import numpy as np
from collections import deque
from src.models.model import SignLanguageLSTM
import yaml

# Load model configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

input_dim = config["model"]["input_dim"]
hidden_dim = config["model"]["hidden_dim"]
output_dim = config["model"]["output_dim"]

# Initialize model with correct architecture
model = SignLanguageLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Load the model weights
state_dict = torch.load("artifacts/models/best_model.pth", map_location=torch.device("cpu"), weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Parameters
confidence_threshold = 0.85
buffer_size = 10
prediction_buffer = deque(maxlen=buffer_size)

# Label map (must match your training label encoding)
label_map = {
    0: "Hello",
    1: "Thanks",
    2: "Yes",
    3: "No",
    # ...
    21: "No Gesture"
}

def predict_with_smoothing(input_data: torch.Tensor):
    """
    Predicts the gesture label with smoothing over a buffer of predictions.

    Args:
        input_data (torch.Tensor): Preprocessed input of shape [1, seq_len, input_dim]

    Returns:
        Tuple[str, float]: Predicted label and confidence score
    """
    if input_data.ndim != 3:
        raise ValueError("Input tensor must be 3D: [1, sequence_length, input_dim]")

    with torch.no_grad():
        outputs = model(input_data)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        conf = conf.item()
        pred = pred.item()

        # Use 'No Gesture' if confidence is too low
        if conf < confidence_threshold:
            pred = 21  # index for "No Gesture"

        prediction_buffer.append(pred)
        final_pred = max(set(prediction_buffer), key=prediction_buffer.count)

        # Safely map predicted index to label
        return label_map.get(final_pred, "Unknown"), conf


input_data = torch.rand(3, 30, input_dim)  # Assuming 30-frame sequence
label, conf = predict_with_smoothing(input_data)
print(f"Prediction: {label}, Confidence: {conf:.2f}")
