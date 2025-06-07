import spacy
from backend.core.config import config

class NERService:
    def __init__(self):
        # Load a pre-trained spaCy model or custom NER model path from config
        self.nlp = spacy.load("en_core_web_sm")  

    def extract_entities(self, text: str):
        doc = self.nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        return entities

