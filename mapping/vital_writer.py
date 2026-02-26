# mapping/vital_writer.py

import json
import math


# Clamp to range Vital/doc expect (e.g. envelope 0-32s) so the plugin never chokes.
SANITIZE_MIN = -1000.0
SANITIZE_MAX = 1000.0


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf and clamp numbers. Preserve dict/list structure."""
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


def _validate_preset_structure(data: dict) -> None:
    """Raise if structure is invalid so we don't write a corrupt file."""
    if not isinstance(data.get("settings"), dict):
        raise ValueError("Preset must have 'settings' dict")
    for k, v in data["settings"].items():
        if isinstance(v, (int, float)) and not math.isfinite(v):
            raise ValueError(f"Non-finite value in settings.{k}")


def export_vital_preset(preset_data: dict, output_path: str):
    """
    Write a valid Vital preset to disk. Uses single-line JSON (no indent) to match
    how Vital exports presets; pretty-printed output can cause Vital to fail loading.
    """
    safe = _sanitize_for_json(preset_data)
    _validate_preset_structure(safe)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(safe, f, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    with open(output_path, "r", encoding="utf-8") as f:
        json.load(f)