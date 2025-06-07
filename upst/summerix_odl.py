from transformers import pipeline
from backend.core.config import config
import os

class Summarizer:
    def __init__(self):
        try:
            model_path = config.paths.artifacts.summarizer_model_dir
        except AttributeError:
            model_path = ""

        if not model_path or not os.path.isdir(model_path) or not os.listdir(model_path):
            print(f"[INFO] Local summarizer model folder '{model_path}' missing or empty. Loading pretrained model from HuggingFace hub...")
            model_path = "facebook/bart-large-cnn"

        self.summarizer = pipeline("summarization", model=model_path)

    def summarize(self, text: str, max_length=150, min_length=40):
        if not text or len(text.strip()) < 20:
            return "Text too short to summarize."

        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
