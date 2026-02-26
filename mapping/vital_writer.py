# mapping/vital_writer.py

import json
import math
import re

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


def _format_json_number(v):
    """Format a number for JSON so Vital's parser accepts it (no scientific notation for small numbers)."""
    if isinstance(v, int):
        return str(v)
    if not math.isfinite(v):
        return "0.0"
    v = max(SANITIZE_MIN, min(SANITIZE_MAX, v))
    s = repr(v)
    if "e" in s.lower() and abs(v) < 1e-2:
        return format(v, ".10f").rstrip("0").rstrip(".")
    return s


# JSON number: integer or float, optional exponent
_JSON_NUM = r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def export_vital_preset_by_patch(template_path: str, param_updates: dict, output_path: str):
    """
    Export by patching only our parameter values into the template file.
    The rest of the file stays byte-identical so Vital's loader never sees
    re-serialized JSON (which may trigger the crash).
    """
    with open(template_path, "r", encoding="utf-8") as f:
        text = f.read()
    for key, value in param_updates.items():
        safe_val = _format_json_number(value)
        pattern = '"' + re.escape(key) + r'":\s*' + _JSON_NUM
        match = re.search(pattern, text)
        if match:
            start, end = match.span()
            text = text[:start] + '"' + key + '":' + safe_val + text[end:]
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def export_vital_preset(preset_data: dict, output_path: str, template_path: str = None):
    """
    Write a Vital preset to disk. If template_path is given, uses patch-based
    export (only our param values change; rest of file unchanged) so Vital can load it.
    """
    from mapping.vital_params import CONTROLLED_PARAMS

    if template_path:
        updates = {}
        settings = preset_data["settings"]
        for k in CONTROLLED_PARAMS:
            if k not in settings:
                continue
            v = settings[k]
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            v = max(SANITIZE_MIN, min(SANITIZE_MAX, v)) if isinstance(v, float) else v
            updates[k] = int(v) if isinstance(v, float) and v == int(v) and abs(v) < 1e10 else v
        export_vital_preset_by_patch(template_path, updates, output_path)
        return
    safe = _sanitize_for_json(preset_data)
    _validate_preset_structure(safe)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(safe, f, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    with open(output_path, "r", encoding="utf-8") as f:
        json.load(f)