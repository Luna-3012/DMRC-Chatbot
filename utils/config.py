import yaml
import os

DEFAULT_CONFIG_PATH = os.path.join("configs", "config.yaml")

def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    