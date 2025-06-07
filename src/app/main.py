from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import os
import json
import tempfile
from src.preprocessing.extract_keypoints import extract_keypoints_from_video
from src.models.model import SignLanguageLSTM

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_mapping = None

@app.on_event("startup")
def load_model():
    global model, class_mapping

    input_dim = 225
    hidden_dim = 256
    output_dim = 1000

    model = SignLanguageLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    model_path = "artifacts/models/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    class_json_path = "metadata/classes.json"
    if not os.path.exists(class_json_path):
        raise FileNotFoundError(f"Class mapping file not found at {class_json_path}")

    with open(class_json_path, "r") as f:
        class_mapping = json.load(f)

@app.get("/")
async def root():
    return {"message": "Sign Language Recognition API is running."}

@app.get("/favicon.ico")
async def favicon():
    return {}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_path = tmp.name
            tmp.write(await file.read())

        # Extract keypoints
        keypoints = extract_keypoints_from_video(video_path)
        input_tensor = torch.tensor(keypoints).unsqueeze(0).to(device).float()

        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_index = outputs.argmax(dim=1).item()

        predicted_label = class_mapping.get(str(predicted_index), "Unknown")

        return {
            "predicted_class_index": predicted_index,
            "predicted_class_label": predicted_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
