# mapping/vital_schema.py

import json
from copy import deepcopy

from mapping.vital_params import filter_to_controlled

# Routing required for envelope to affect sound. Blank template has no env_1 -> level.
ENV_1_TO_OSC_1_LEVEL = {"source": "env_1", "destination": "osc_1_level"}


def load_template(path="mapping/template.vital"):
    with open(path, "r") as f:
        return json.load(f)


def _ensure_env1_routes_to_osc_level(settings: dict) -> None:
    """Ensure Env 1 modulates osc 1 level so envelope settings are audible. In-place."""
    mods = settings.get("modulations")
    if not mods:
        return
    # Find first slot with no source, or any slot already routing env_1 -> osc_1_level
    for m in mods:
        if m.get("source") == "env_1" and m.get("destination") == "osc_1_level":
            return
    for m in mods:
        if not m.get("source") and not m.get("destination"):
            m["source"] = "env_1"
            m["destination"] = "osc_1_level"
            return


def apply_parameters(template: dict, param_updates: dict) -> dict:
    """Apply only controlled params that exist in the template. Safe for ML output."""
    preset = deepcopy(template)
    settings = preset["settings"]
    allowed = filter_to_controlled(param_updates, settings)
    for key, value in allowed.items():
        settings[key] = value
    _ensure_env1_routes_to_osc_level(settings)
    return preset