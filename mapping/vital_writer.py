# mapping/vital_writer.py

import json
import math


def _sanitize_for_json(obj):
    """
    Recursively replace NaN/Inf in preset so Vital (and JSON) never see invalid numbers.
    Vital can crash on load if any setting is NaN or Inf.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return 0.0
        return max(-1e6, min(1e6, obj))
    if isinstance(obj, int):
        if not math.isfinite(obj):
            return 0
        return max(-1e6, min(1e6, obj))
    return obj


def export_vital_preset(preset_data: dict, output_path: str):
    """Write preset to disk; sanitize all numbers so Vital never sees NaN/Inf."""
    safe = _sanitize_for_json(preset_data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, allow_nan=False)