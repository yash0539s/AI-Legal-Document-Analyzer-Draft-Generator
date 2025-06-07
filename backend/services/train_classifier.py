import os
import re
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Your DAC file parser
def parse_dac_file(filepath):
    texts = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split entries by empty lines
    entries = content.strip().split("\n\n")
    for entry in entries:
        id_match = re.search(r"ID:\s*(.+)", entry)
        text_match = re.search(r"Answer Text:\s*(.+)", entry)

        if not id_match or not text_match:
            continue

        id_str = id_match.group(1)
        answer_text = text_match.group(1).strip()

        # Extract label from ID string (between __ and last _number)
        label_match = re.search(r"__(.+?)_\d+$", id_str)
        label = label_match.group(1) if label_match else "Unknown"

        if answer_text != "None" and answer_text != "":
            texts.append(answer_text)
            labels.append(label)

    return texts, labels

def train_classifier():
    print("[INFO] Loading data from dac.txt...")

    # Set path to your dac.txt file (update as needed)
    data_file = "data/processed/train/doctxt.txt"  # change to your actual path

    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"DAC file not found: {data_file}")

    X, y = parse_dac_file(data_file)
    print(f"[INFO] Loaded {len(X)} samples across {len(set(y))} classes.")

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Training classifier...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)

    print("[INFO] Evaluating classifier on test data...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the trained model
    model_path = "artifacts/classifier_model2.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    print(f"[INFO] Model saved to {model_path}")

if __name__ == "__main__":
    train_classifier()
