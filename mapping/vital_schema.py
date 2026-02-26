# mapping/vital_schema.py

import json
from copy import deepcopy

from mapping.vital_params import filter_to_controlled

def load_template(path="mapping/template.vital"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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