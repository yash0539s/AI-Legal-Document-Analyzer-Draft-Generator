import os
from backend.services.preprocces import preprocess_documents
from backend.services.classifier import TextClassifier
from backend.services.ner import NERService
from backend.services.summarizer import Summarizer
from backend.services.draft_generator import DraftGenerator
from backend.services.risk_detector import RiskDetector

# Import your training function
from backend.services.train_classifier import train_classifier  

class LexiDraftPipeline:
    def __init__(self):
        print("[INFO] Initializing Text Classifier...")
        self.classifier = TextClassifier()

        print("[INFO] Initializing NER Service...")
        self.ner = NERService()

        print("[INFO] Initializing Summarizer...")
        self.summarizer = Summarizer()

        print("[INFO] Initializing Draft Generator...")
        try:
            self.draft_generator = DraftGenerator()
        except Exception as e:
            print(f"[ERROR] Failed to initialize Draft Generator: {e}")
            self.draft_generator = None

        print("[INFO] Initializing Risk Detector...")
        try:
            self.risk_detector = RiskDetector()
        except Exception as e:
            print(f"[WARNING] Risk Detector initialization failed: {e}")
            self.risk_detector = None

    def process_text(self, text: str):
        print("[INFO] Running Text Classification...")
        classification = self.classifier.predict(text)

        print("[INFO] Extracting Named Entities...")
        entities = self.ner.extract_entities(text)

        print("[INFO] Generating Summary...")
        summary = self.summarizer.summarize(text)

        draft = None
        if self.draft_generator:
            try:
                print("[INFO] Generating Draft...")
                draft = self.draft_generator.generate(text)
            except Exception as e:
                print(f"[ERROR] Draft generation failed: {e}")
                draft = "Draft generation failed."
        else:
            draft = "Draft Generator not initialized."

        print("[INFO] Detecting Risks...")
        if self.risk_detector:
            try:
                risk = self.risk_detector.detect(text)
            except Exception as e:
                print(f"[ERROR] Risk detection failed: {e}")
                risk = {"risk": "Risk detection failed", "confidence": 0.0}
        else:
            risk = {"risk": "Risk Detector not available", "confidence": 0.0}

        processed = {
            "classification": classification,
            "entities": entities,
            "summary": summary,
            "draft": draft,
            "risk": risk,
        }
        return processed

def main():
    model_path = "artifacts/classifier_model.joblib"

    # If model doesn't exist, train it first
    if not os.path.exists(model_path):
        print("[INFO] Classifier model not found. Training now...")
        train_classifier()
        print("[INFO] Training complete and model saved.\n")

    print("[INFO] Initializing LexiDraftPipeline...")
    pipeline = LexiDraftPipeline()

    sample_text = (
        "LexiDraft Pro is an advanced document processing system that classifies, summarizes, "
        "extracts entities, generates drafts, and detects risks in texts. It is designed to "
        "boost productivity and accuracy for legal and business documents."
    )

    print("[INFO] Running LexiDraft pipeline on sample text...\n")
    results = pipeline.process_text(sample_text)

    print("=== Pipeline Results ===")
    print(f"Classification: {results['classification']}\n")
    print(f"Entities: {results['entities']}\n")
    print(f"Summary: {results['summary']}\n")
    print(f"Draft Generated: {results['draft']}\n")
    print(f"Risk Detection: {results['risk']}\n")

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
