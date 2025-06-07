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

        max_chunk_tokens = 1000  # BART limit is 1024, keeping buffer
        sentences = text.split('. ')
        current_chunk = ""
        chunks = []

        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) <= max_chunk_tokens:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())

        summaries = []
        for chunk_text in chunks:
            try:
                summary = self.summarizer(
                    chunk_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                summaries.append(f"[Error summarizing chunk: {str(e)}]")

        return " ".join(summaries)
