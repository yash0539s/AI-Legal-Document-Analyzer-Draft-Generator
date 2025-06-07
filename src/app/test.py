from src.preprocessing.extract import extract_keypoints_from_video
import torch
from src.models.model import SignLanguageLSTM  # your model class

def predict_video(video_path, model):
    keypoints = extract_keypoints_from_video(video_path)
    print(f"[Inference] Extracted keypoints shape: {keypoints.shape}")

    expected_dim = model.input_size  # make sure your model has this attribute or set it manually
    if keypoints.shape[1] != expected_dim:
        print(f"Warning: Keypoints feature dim {keypoints.shape[1]} != model input_dim {expected_dim}. Adjusting...")
        keypoints = keypoints[:, :expected_dim]

    input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)  # (1, frames, features)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        # Process outputs into labels and confidence here (depends on your model)
    
    return outputs

if __name__ == "__main__":
    test_video_path = r"D:\sign-text\cookiecutter-sign-language-translator\Real-Time-Sign-Language-Recognition-System\new_data\raw\11(1).mp4"
    
    # Model parameters - match your training config
    input_dim = 225
    hidden_dim = 256   # your checkpoint hidden size
    output_dim = 707   # your output classes


    # Instantiate model
    model = SignLanguageLSTM(input_dim, hidden_dim, output_dim)
    model.input_size = input_dim  # add this attribute for your code

    # Load saved weights
    model.load_state_dict(torch.load(r"D:\sign-text\cookiecutter-sign-language-translator\Real-Time-Sign-Language-Recognition-System\models\best_model.pth"))

    # Call prediction
    outputs = predict_video(test_video_path, model)
    print("Model outputs:", outputs)
