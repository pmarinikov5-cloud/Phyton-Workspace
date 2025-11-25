import json
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON config file into a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Minimal sanity checks
    required_keys = ["name", "excel_market_file", "sheet_da"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")

    return cfg
