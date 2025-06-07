import os
import joblib
from backend.core.config import config
from backend.core.utils import clean_text

class TextClassifier:
    def __init__(self):
        model_path = self._get_model_path_from_config()

        if not model_path:
            raise ValueError("Classifier model path is not defined in the config file.")

        # Resolve path relative to current working directory (cwd)
        abs_path = os.path.abspath(model_path)
        print(f"[DEBUG] Trying to load model from: {abs_path}")

        if not os.path.exists(abs_path):
            # Try relative to project root (assumed to be one folder up from cwd)
            project_root = os.path.dirname(os.getcwd())
            alt_path = os.path.abspath(os.path.join(project_root, model_path))
            print(f"[DEBUG] Model not found at {abs_path}, trying alternative path: {alt_path}")
            
            if os.path.exists(alt_path):
                abs_path = alt_path
            else:
                raise FileNotFoundError(f"Classifier model file not found at either {abs_path} or {alt_path}")

        try:
            self.model = joblib.load(abs_path)
        except EOFError as e:
            raise EOFError(f"Error loading model file {abs_path}: {e}")

    def _get_model_path_from_config(self):
        try:
            return config.paths.artifacts.classifier_model
        except AttributeError:
            pass

        try:
            paths = getattr(config, "paths", None) or config.get("paths", None)
            if paths is None:
                return None
            artifacts = getattr(paths, "artifacts", None) or paths.get("artifacts", None)
            if artifacts is None:
                return None
            classifier_model = getattr(artifacts, "classifier_model", None) or artifacts.get("classifier_model", None)
            return classifier_model
        except Exception:
            return None

    def predict(self, text: str):
        processed_text = clean_text(text)
        return self.model.predict([processed_text])[0]

    def predict_proba(self, text: str):
        processed_text = clean_text(text)
        proba = self.model.predict_proba([processed_text])[0]
        return {self.model.classes_[i]: float(proba[i]) for i in range(len(proba))}
