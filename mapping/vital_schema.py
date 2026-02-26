# mapping/vital_schema.py

import json
import os
from copy import deepcopy

from mapping.vital_params import filter_to_controlled

# Full Vital presets have "settings" dict; simplified ones (WideSawLead, etc.) do not.
DEFAULT_TEMPLATE_PATHS = [
    "vital_templates/Presets/blank-template.vital",
    "vital_templates/Presets/TEMPLATE.vital",
    "mapping/template.vital",
]


def is_full_vital_preset(data: dict) -> bool:
    """True if this is a loadable Vital preset (has 'settings' dict)."""
    return isinstance(data.get("settings"), dict)


def load_template(path: str = None):
    """Load a full-format Vital preset. If path is None, try default locations."""
    if path is None:
        for p in DEFAULT_TEMPLATE_PATHS:
            if os.path.isfile(p):
                path = p
                break
        if path is None:
            path = "mapping/template.vital"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not is_full_vital_preset(data):
        raise ValueError(
            f"Not a full Vital preset (no 'settings'): {path}. "
            "Use a preset exported from Vital, not a simplified-format file."
        )
    return data


def apply_parameters(template: dict, param_updates: dict) -> dict:
    """
    Apply only controlled params that exist in the template and are numeric.
    Values are sanitized (no NaN/Inf). We do NOT modify modulations/sample/etc.
    so Vital's loader never hits an unexpected structure and crashes.
    """
    preset = deepcopy(template)
    settings = preset["settings"]
    allowed = filter_to_controlled(param_updates, settings)
    for key, value in allowed.items():
        if key in settings and isinstance(settings[key], (int, float)):
            settings[key] = value
    return preset