import json
from pathlib import Path


def load_project_config(config_path: Path) -> dict:
    """
    Loads and returns the project configurations from a JSON file.
    """
    with open(config_path, 'r') as f:
        return json.load(f)
