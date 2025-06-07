import cv2
import torch
import numpy as np
from collections import deque
from src.models.model import SignLanguageLSTM
from src.utils.helper import load_label_map
import mediapipe as mp

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

import cv2
import torch
import numpy as np
from collections import deque
from src.models.model import SignLanguageLSTM
from src.utils.helper import load_label_map
import mediapipe as mp

# MediaPipe solutions
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Config
MODEL_PATH = "artifacts/models/best_model.pth"
LABELS_PATH = "metadata/classes.json"
SEQUENCE_LENGTH = 30
INPUT_SIZE = 225
CONFIDENCE_THRESHOLD = 0.75  # Adjust this threshold as needed
SMOOTHING_WINDOW = 5  # Number of frames for prediction smoothing

# Load label map (assumed list of labels)
label_map = load_label_map(LABELS_PATH)

# Add a special label for "No Gesture" (if not already in your label_map)
NO_GESTURE_LABEL = "No Gesture"

# Load model
model = SignLanguageLSTM(input_dim=INPUT_SIZE, hidden_dim=256, output_dim=len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Initialize MediaPipe Hands and Pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
pose = mp_pose.Pose(static_image_mode=False)

# Buffers
sequence = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = deque(maxlen=SMOOTHING_WINDOW)

def extract_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(frame_rgb)
    results_pose = pose.process(frame_rgb)
    keypoints = []

    # Hands keypoints
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0] * 21 * 3)  # zero padding for hands

    # Pose keypoints
    if results_pose.pose_landmarks:
        for lm in results_pose.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0] * 33 * 3)  # zero padding for pose

    # Pad if fewer keypoints than INPUT_SIZE
    while len(keypoints) < INPUT_SIZE:
        keypoints.append(0.0)

    return keypoints

def predict_sequence(sequence, threshold=CONFIDENCE_THRESHOLD):
    x = np.array(sequence)
    x = torch.tensor(x).unsqueeze(0).float()
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        max_prob, pred_idx = torch.max(probs, dim=1)
        max_prob = max_prob.item()
        pred_idx = pred_idx.item()

        if max_prob < threshold:
            return NO_GESTURE_LABEL
        else:
            return label_map[pred_idx]

def run_realtime():
    cap = cv2.VideoCapture(0)
    print("Starting real-time sign language recognition. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints(frame)
        sequence.append(keypoints)

        if len(sequence) == SEQUENCE_LENGTH:
            prediction = predict_sequence(sequence)
            prediction_history.append(prediction)

            # Smoothing: most common prediction in last N frames
            prediction_to_show = max(set(prediction_history), key=prediction_history.count)

            cv2.putText(frame, f"Prediction: {prediction_to_show}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Sign Language Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()
