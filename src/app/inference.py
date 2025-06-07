import cv2
import torch
import numpy as np
from collections import deque, Counter
import json
from src.models.model import SignLanguageLSTM
import mediapipe as mp

# Constants
MODEL_PATH = "artifacts/models/best_model.pth"
LABELS_PATH = "metadata/classes.json"
SEQUENCE_LENGTH = 30
INPUT_SIZE = 225
CONFIDENCE_THRESHOLD = 0.85  # increase threshold to reduce false positives
SMOOTHING_WINDOW = 15

# Load label map (ensure keys are int)
def load_label_map(path):
    with open(path, "r") as f:
        label_map = json.load(f)
    return {int(k): v for k, v in label_map.items()}

label_map = load_label_map(LABELS_PATH)
print(f"Loaded {len(label_map)} labels.")

# Load model
model = SignLanguageLSTM(input_dim=INPUT_SIZE, hidden_dim=256, output_dim=len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Mediapipe init
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
pose = mp_pose.Pose(static_image_mode=False)

sequence = deque(maxlen=SEQUENCE_LENGTH)
recent_predictions = deque(maxlen=SMOOTHING_WINDOW)

def extract_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(frame_rgb)
    results_pose = pose.process(frame_rgb)

    keypoints = []

    # Extract hand keypoints (up to 2 hands)
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks[:2]:  # Max 2 hands
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        # If only one hand is detected, pad for the second hand
        while len(keypoints) < 21 * 3 * 2:
            keypoints.extend([0.0, 0.0, 0.0])
    else:
        keypoints.extend([0.0] * 21 * 3 * 2)

    # Extract pose keypoints (33)
    if results_pose.pose_landmarks:
        for lm in results_pose.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 33 * 3)

    # Total collected: (2 hands * 21 * 3) + (33 * 3) = 63 + 63 = 189 keypoints
    # Pad or truncate to exactly 225
    if len(keypoints) > 225:
        keypoints = keypoints[:225]
    else:
        keypoints.extend([0.0] * (225 - len(keypoints)))

    return keypoints


def predict_sequence(seq):
    x = torch.tensor(np.array(seq)).unsqueeze(0).float()
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        return pred_idx.item(), conf.item()

def run_realtime():
    cap = cv2.VideoCapture(0)
    display_text = ""
    last_prediction = None

    print("Starting real-time recognition. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints(frame)
        sequence.append(keypoints)

        if len(sequence) == SEQUENCE_LENGTH:
            pred_idx, conf = predict_sequence(sequence)
            # Debug print for checking outputs
            print(f"Predicted class idx: {pred_idx}, confidence: {conf:.3f}")

            if conf >= CONFIDENCE_THRESHOLD:
                recent_predictions.append(pred_idx)
                most_common_pred, count = Counter(recent_predictions).most_common(1)[0]

                # If majority agrees, update displayed label
                if count > SMOOTHING_WINDOW // 2:
                    if most_common_pred != last_prediction:
                        display_text = label_map.get(most_common_pred, "Unknown")
                        last_prediction = most_common_pred
                else:
                    display_text = ""
            else:
                recent_predictions.clear()
                display_text = ""
                last_prediction = None

        if display_text:
            cv2.putText(frame, f"Sign: {display_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Sign: (no confident detection)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 2)

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()
