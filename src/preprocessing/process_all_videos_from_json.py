# src/preprocessing/process_videos.py

import os
import json
from pathlib import Path
import numpy as np
from src.preprocessing.extract_keypoints import extract_keypoints_from_video
import unicodedata


def normalize_text(text):
    """Normalize text by decomposing unicode characters and replacing ligatures."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    # Replace common ligatures manually
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return text.lower().strip()


def load_synonyms_and_classes(synonyms_path, class_list_path):
    with open(synonyms_path, 'r') as f:
        synonyms_list = json.load(f)

    synonyms = {}
    for group in synonyms_list:
        if not group:
            continue
        original = normalize_text(group[0])
        for word in group:
            normalized_word = normalize_text(word)
            synonyms[normalized_word] = original

    with open(class_list_path, 'r') as f:
        class_list_raw = json.load(f)

    if isinstance(class_list_raw, list):
        class_list = {str(i): v for i, v in enumerate(class_list_raw)}
    else:
        class_list = class_list_raw

    class_name_to_index = {normalize_text(v): int(k) for k, v in class_list.items()}
    return synonyms, class_name_to_index


def process_videos(raw_path, processed_path, max_frames=30,
                   synonyms_path=None, class_list_path=None,
                   splits_json_dir=None):
    if not synonyms_path or not class_list_path:
        raise ValueError("Must provide both synonyms.json and class_list.json")

    # Load synonyms and class mappings once
    synonyms, class_name_to_index = load_synonyms_and_classes(synonyms_path, class_list_path)

    Path(processed_path).mkdir(parents=True, exist_ok=True)

    # Use splits_json_dir if provided, else fallback to raw_path
    json_folder = splits_json_dir if splits_json_dir else raw_path

    # Detect splits based on json files in json_folder
    possible_splits = ["train", "val", "test"]
    available_splits = [split for split in possible_splits if os.path.exists(os.path.join(json_folder, f"{split}.json"))]

    if not available_splits:
        print(f"[ERROR] No split files (train/val/test) found in: {json_folder}")
        return

    for split in available_splits:
        json_file = os.path.join(json_folder, f"{split}.json")
        processed_split_path = os.path.join(processed_path, split)
        Path(processed_split_path).mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Processing {split} split from {json_file}")

        with open(json_file, "r") as f:
            data = json.load(f)

        labels_output = []

        for entry in data:
            raw_label = normalize_text(entry.get("clean_text", entry.get("text", entry.get("label", ""))))
            label = synonyms.get(raw_label, raw_label)

            if label not in class_name_to_index:
                print(f"[SKIP] Label '{label}' not in class list.")
                continue
            class_idx = class_name_to_index[label]

            video_filename = entry["file"] + ".mp4"
            video_path = os.path.join(raw_path, video_filename)  # always look in raw_path for videos
            if not os.path.exists(video_path):
                print(f"[SKIP] File not found: {video_path}")
                continue

            keypoints = extract_keypoints_from_video(video_path, max_frames)
            if np.all(keypoints == 0):
                print(f"[SKIP] Empty keypoints: {video_path}")
                continue

            save_name = os.path.splitext(video_filename)[0]
            np.save(os.path.join(processed_split_path, save_name + ".npy"), keypoints)

            labels_output.append({
                "file": save_name + ".npy",
                "label": label,
                "class_index": class_idx
            })

        with open(os.path.join(processed_split_path, "labels.json"), "w") as f:
            json.dump(labels_output, f, indent=2)

        print(f"[INFO] Finished {split} set. Saved {len(labels_output)} examples.")



# To enable running this script directly
if __name__ == "__main__":
    process_videos(
        raw_path=r'D:\sign-text\cookiecutter-sign-language-translator\Real-Time-Sign-Language-Recognition-System\new_data\raw',
        processed_path="data/processed",
        max_frames=30,
        synonyms_path="metadata/synonym.json",
        class_list_path="metadata/classes.json"
    )
