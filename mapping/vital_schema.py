# mapping/vital_schema.py

import json
from copy import deepcopy

from mapping.vital_params import filter_to_controlled


def load_template(path="mapping/template.vital"):
    with open(path, "r") as f:
        return json.load(f)


def apply_parameters(template: dict, param_updates: dict) -> dict:
    """Apply only controlled params that exist in the template. Safe for ML output."""
    preset = deepcopy(template)
    settings = preset["settings"]
    allowed = filter_to_controlled(param_updates, settings)
    for key, value in allowed.items():
        settings[key] = value
    return preset