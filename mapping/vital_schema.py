# mapping/vital_schema.py

import json
from copy import deepcopy


def load_template(path="mapping/template.vital"):
    with open(path, "r") as f:
        return json.load(f)


def apply_parameters(template: dict, param_updates: dict) -> dict:

    preset = deepcopy(template)

    for key, value in param_updates.items():
        if key in preset["settings"]:
            preset["settings"][key] = value

    return preset