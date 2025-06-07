# src/preprocessing/extract_keypoints.py

import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def extract_keypoints_from_video(video_path: str, max_frames: int = 30) -> np.ndarray:
    if not os.path.exists(video_path):
        print(f"[ERROR] File does not exist: {video_path}")
        return np.zeros((max_frames, 21 * 3 * 2 + 33 * 3))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return np.zeros((max_frames, 21 * 3 * 2 + 33 * 3))

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    pose = mp_pose.Pose(static_image_mode=False)

    keypoints_seq = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hand = hands.process(frame_rgb)
        results_pose = pose.process(frame_rgb)

        # Initialize with zeros for 2 hands (21 landmarks * 3 coords * 2 hands)
        hand_keypoints = [0] * (21 * 3 * 2)

        if results_hand.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hand.multi_hand_landmarks):
                if i >= 2:
                    break
                for j, lm in enumerate(hand_landmarks.landmark):
                    idx = i * 21 * 3 + j * 3
                    hand_keypoints[idx] = lm.x
                    hand_keypoints[idx + 1] = lm.y
                    hand_keypoints[idx + 2] = lm.z

        keypoints = hand_keypoints

        if results_pose.pose_landmarks:
            for lm in results_pose.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0] * 33 * 3)

        keypoints_seq.append(keypoints)
        frame_count += 1

    cap.release()
    hands.close()
    pose.close()

    # Pad to max_frames if needed
    while len(keypoints_seq) < max_frames:
        keypoints_seq.append([0] * len(keypoints_seq[0]))

    return np.array(keypoints_seq)
