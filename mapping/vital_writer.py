# mapping/vital_writer.py

import json
import math


# Match Vital's typical export range so the loader doesn't choke.
SANITIZE_MIN = -1000.0
SANITIZE_MAX = 1000.0


def _sanitize_for_json(obj):
    """
    Recursively replace NaN/Inf and clamp numbers so Vital never crashes on load.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return 0.0
        return max(SANITIZE_MIN, min(SANITIZE_MAX, obj))
    if isinstance(obj, int):
        if not math.isfinite(obj):
            return 0
        return max(int(SANITIZE_MIN), min(int(SANITIZE_MAX), obj))
    return obj


def export_vital_preset(preset_data: dict, output_path: str):
    """
    Write preset to disk. Sanitize numbers; use single-line JSON like the template
    so Vital's parser sees the format it expects.
    """
    safe = _sanitize_for_json(preset_data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, separators=(",", ":"), allow_nan=False)