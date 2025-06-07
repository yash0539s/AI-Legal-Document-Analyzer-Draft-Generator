from fastapi import APIRouter
from backend.api.schemas import (
    TextRequest, ClassifyResponse, NERResponse, 
    SummarizeResponse, DraftResponse, RiskResponse
)
from backend.services.classifier import TextClassifier
from backend.services.ner import NERService
from backend.services.summarizer import Summarizer
from backend.services.draft_generator import DraftGenerator
from backend.services.risk_detector import RiskDetector

router = APIRouter()

classifier = TextClassifier()
ner = NERService()
summarizer = Summarizer()
draft_generator = DraftGenerator()
risk_detector = RiskDetector()

@router.post("/classify", response_model=ClassifyResponse)
def classify_text(request: TextRequest):
    label = classifier.predict(request.text)
    confidence = classifier.predict_proba(request.text)
    return {"label": label, "confidence": confidence}

@router.post("/ner", response_model=NERResponse)
def extract_entities(request: TextRequest):
    entities = ner.extract_entities(request.text)
    return {"entities": entities}

@router.post("/summarize", response_model=SummarizeResponse)
def summarize_text(request: TextRequest):
    summary = summarizer.summarize(request.text)
    return {"summary": summary}

@router.post("/generate_draft", response_model=DraftResponse)
def generate_draft(request: TextRequest):
    draft = draft_generator.generate(request.text)
    return {"draft_text": draft}

@router.post("/detect_risk", response_model=RiskResponse)
def detect_risk(request: TextRequest):
    result = risk_detector.detect(request.text)
    # Return all fields matching RiskResponse schema
    return {
        "risk": result.get("risk", "Unknown"),
        "confidence": result.get("confidence", 0.0),
        "severity": result.get("severity", "Unknown")
    }
