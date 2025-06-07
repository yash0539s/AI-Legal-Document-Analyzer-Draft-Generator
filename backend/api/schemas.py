from pydantic import BaseModel
from typing import List, Dict, Any

class TextRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    label: str
    confidence: Dict[str, float]  # or Dict[str, Any] depending on your classifier output

class NERResponse(BaseModel):
    entities: List[Dict[str, Any]]  # list of dicts with entity info

class SummarizeResponse(BaseModel):
    summary: str

class DraftResponse(BaseModel):
    draft_text: str

class RiskResponse(BaseModel):
    risk: str        # "Low Risk", "Medium Risk", "High Risk"
    confidence: float
    severity: str    # "Low", "Medium", "High"
