import numpy as np
import mediapipe as mp

def extract_keypoints_from_video(video_path):
    # Example stub: your actual code will extract keypoints per frame
    keypoints = []

    # Your existing MediaPipe processing and extraction loop here
    # For demonstration:
    # for each frame:
    #    keypoints.append(np.concatenate([pose_keypoints, hand_left, hand_right]))

    # Example dummy data (replace with your actual extraction)
    # Let's say you extract 75 keypoints with 3 coords each (x,y,z), total 225 features
    keypoints_np = np.random.rand(117, 225)  # shape (frames, features)

    print(f"[Training] Extracted keypoints shape: {keypoints_np.shape}")
    return keypoints_np
