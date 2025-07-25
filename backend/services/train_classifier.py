import os
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


MODEL_NAME = "nlpaueb/legal-bert-base-uncased"  # or your choice


# Step 1: Load and parse DAC format
def parse_dac_file(filepath):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    entries = content.strip().split("\n\n")

    for entry in entries:
        id_match = re.search(r"ID:\s*(.+)", entry)
        text_match = re.search(r"Answer Text:\s*(.+)", entry)
        if not id_match or not text_match:
            continue

        id_str = id_match.group(1)
        answer_text = text_match.group(1).strip()

        label_match = re.search(r"__(.+?)_\d+$", id_str)
        label = label_match.group(1) if label_match else "Unknown"

        if answer_text and answer_text != "None":
            texts.append(answer_text)
            labels.append(label)

    return texts, labels


# Step 2: Convert to HuggingFace Dataset
def prepare_dataset(texts, labels):
    label_list = sorted(list(set(labels)))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    data = {
        "text": texts,
        "label": [label2id[l] for l in labels]
    }

    dataset = Dataset.from_dict(data)
    return dataset, label2id, id2label


# Step 3: Metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Step 4: Train the model
def train_model():
    filepath = "data/processed/train/doctxt.txt"  # Update path

    texts, labels = parse_dac_file(filepath)
    dataset, label2id, id2label = prepare_dataset(texts, labels)
    train_ds, test_ds = dataset.train_test_split(test_size=0.2).values()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./artifacts/legalbert_classifier",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model("./artifacts/legalbert_classifier")
    tokenizer.save_pretrained("./artifacts/legalbert_classifier")

    print("LegalBERT classifier saved to artifacts/legalbert_classifier")


if __name__ == "__main__":
    train_model()
