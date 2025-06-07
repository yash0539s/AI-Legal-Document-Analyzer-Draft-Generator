import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path: str = "mlops/config.yaml"):
        self.config_path = Path(config_path)
        self.data = self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            return self._dict_to_obj(config_dict)

    def _dict_to_obj(self, d):
        if isinstance(d, dict):
            # Recursively convert dict to object with attributes
            return type("ConfigNamespace", (), {k: self._dict_to_obj(v) for k, v in d.items()})()
        elif isinstance(d, list):
            return [self._dict_to_obj(i) for i in d]
        else:
            return d

    def get(self, key, default=None):
        # Get top-level config attribute safely
        return getattr(self.data, key, default)

# Create a single global config object
config = Config()
