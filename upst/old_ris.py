import logging
import joblib
import os
from backend.core.config import config
from backend.core.utils import clean_text

logger = logging.getLogger(__name__)

class RiskDetector:
    def __init__(self):
        try:
            model_path = config.data.paths.artifacts.risk_detector_model
        except AttributeError:
            logger.error("Config missing path to risk detector model")
            self.model = None
            return

        if not os.path.exists(model_path):
            logger.warning(f"Risk detector model file not found at: {model_path}. Using dummy detector.")
            self.model = None
        else:
            self.model = joblib.load(model_path)

    def detect(self, text: str):
        if self.model is None:
            return {"risk": "Risk Detector not available", "confidence": 0.0, "severity": "Unknown"}

        processed_text = clean_text(text)
        try:
            pred = self.model.predict([processed_text])[0]
            proba = self.model.predict_proba([processed_text])

            logger.debug(f"predict_proba output: {proba}, shape: {proba.shape}")
            logger.debug(f"predicted label: {pred}")

            confidence = 0.0
            if hasattr(self.model, 'classes_'):
                class_idx = list(self.model.classes_).index(pred)
                confidence = float(proba[0][class_idx])
            else:
            # fallback
                confidence = float(proba[0].max())

            confidence_pct = round(confidence * 100, 2)

            risk_map = {
                0: "Low Risk",
                1: "Medium Risk",
                2: "High Risk"
            }
        # pred might be a label, so try to get int key
            risk_label = risk_map.get(pred, pred if isinstance(pred, str) else "Unknown Risk")

        # Severity based on confidence thresholds
            if confidence_pct > 80:
                severity = "High"
            elif confidence_pct > 50:
                severity = "Medium"
            else:
                severity = "Low"

            return {
                "risk": risk_label,
                "confidence": confidence_pct,
                "severity": severity
            }
        except Exception as e:
            logger.error(f"Risk detection failed: {e}")
            return {"risk": "Risk detection failed", "confidence": 0.0, "severity": "Unknown"}
